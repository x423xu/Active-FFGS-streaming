import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange, repeat

from ..unimatch.ldm_unet.unet import UNetModel
from .dpt import DPTHead, CostHead
from .mv_transformer import (
    MultiViewFeatureTransformer,
)
from .utils import mv_feature_add_position

TIMER=True
if TIMER:
    import time
    def _sync_time(device: torch.device) -> float:
        # CUDA ops are asynchronous; synchronize so stage timings are attributed correctly.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.time()

def coords_grid(batch, ht, wd, homogeneous=False, device=None):
    ys, xs = torch.meshgrid(
        torch.arange(ht, device=device),
        torch.arange(wd, device=device),
        indexing="ij",
    )
    stacks = [xs, ys]
    if homogeneous:
        stacks.append(torch.ones_like(xs))
    grid = torch.stack(stacks, dim=0).float()
    return grid[None].repeat(batch, 1, 1, 1)


def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c, d, h, w
    )  # [B, C, D, H, W]

    return warped_feature

def prepare_feat_proj_data_lists(features, intrinsics, extrinsics, num_reference_views, idx):
    b, v, c, h, w = features.shape
    idx = idx[:, :, 1:]  # remove the current view
    if extrinsics is not None:
        # extract warp poses
        idx_to_warp = repeat(idx, "b v m -> b v m fw fh", fw=4, fh=4) # [b, v, m, 1, 1]
        extrinsics_cur = repeat(extrinsics.clone().detach(), "b v fh fw -> b v m fh fw", m=num_reference_views)  # [b, v, 4, 4]
        poses_others = extrinsics_cur.gather(1, idx_to_warp)  # [b, v, m, 4, 4]
        poses_others_inv = torch.linalg.inv(poses_others)  # [b, v, m, 4, 4]
        poses_cur = extrinsics.clone().detach().unsqueeze(2)  # [b, v, 1, 4, 4]
        poses_warp = poses_others_inv @ poses_cur  # [b, v, m, 4, 4]
        poses_warp = rearrange(poses_warp, "b v m ... -> (b v) m ...")  # [bxv, m, 4, 4]
    else:
        poses_warp = None
    
    if features is not None:
        # extract warp features
        idx_to_warp = repeat(idx, "b v m -> b v m c h w", c=c, h=h, w=w) # [b, v, m, 1]
        features_cur = repeat(features, "b v c h w -> b v m c h w", m=num_reference_views)  # [b, v, m, c, h, w]
        feat_warp = features_cur.gather(1, idx_to_warp)  # [b, v, m, c, h, w]
        feat_warp = rearrange(feat_warp, "b v m c h w -> (b v) m c h w")  # [bxv, m, c, h, w]
    else:
        feat_warp = None
    
    if intrinsics is not None:
        # extract warp intrinsics
        intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
        intr_curr[:, :, 0, :] *= float(w)
        intr_curr[:, :, 1, :] *= float(h)
        idx_to_warp = repeat(idx, "b v m -> b v m fh fw", fh=3, fw=3) # [b, v, m, 1, 1]
        intr_curr = repeat(intr_curr, "b v fh fw -> b v m fh fw", m=num_reference_views)  # [b, v, m, 3, 3]
        intr_warp = intr_curr.gather(1, idx_to_warp)  # [b, v, m, 3, 3]
        intr_warp = rearrange(intr_warp, "b v m ... -> (b v) m ...")  # [bxv, m, 3, 3]
    else:
        intr_warp = None
    
    return feat_warp, intr_warp, poses_warp


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, ~8x fewer FLOPs than standard 3x3 conv."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class LiteResBlock(nn.Module):
    """Residual block using depthwise-separable convolutions."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class EfficientCrossViewAttn(nn.Module):
    """Per-pixel cross-view attention: O(V^2 * HW) instead of O((V*HW)^2).
    For V=2, the attention matrix is 2x2 per pixel — essentially free."""
    def __init__(self, channels, num_views=2, num_heads=4):
        super().__init__()
        self.num_views = num_views
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        bv, c, h, w = x.shape
        v = self.num_views
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x)
        qkv = rearrange(qkv, '(b v) (three nh hd) h w -> three b nh (h w) v hd',
                         v=v, three=3, nh=self.num_heads, hd=self.head_dim)
        q, k, val = qkv.unbind(0)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ val
        out = rearrange(out, 'b nh (h w) v hd -> (b v) (nh hd) h w', h=h, w=w)
        return residual + self.proj(out)


class LiteCostVolumeRefineNet(nn.Module):
    """Lightweight cost volume refinement replacing the heavy UNetModel version.

    Key speedups:
    - Depthwise-separable convs in ResBlocks (~8x fewer FLOPs per conv)
    - Per-pixel cross-view attention at bottleneck (O(V^2*HW) vs O((V*HW)^2))
    - ~4.5x fewer total parameters than original UNetModel-based refinement
    """
    def __init__(self, in_channels, feat_dim, out_channels, num_views=2):
        super().__init__()
        # Input projection (standard conv to mix heterogeneous input channels)
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, 3, 1, 1),
            nn.GroupNorm(8, feat_dim),
            nn.GELU(),
        )
        # Encoder
        self.enc0 = LiteResBlock(feat_dim, feat_dim)            # 64x64
        self.down0 = nn.Conv2d(feat_dim, feat_dim, 3, 2, 1)    # -> 32x32
        self.enc1 = LiteResBlock(feat_dim, feat_dim)            # 32x32
        self.down1 = nn.Conv2d(feat_dim, feat_dim, 3, 2, 1)    # -> 16x16
        # Bottleneck with cross-view attention
        self.mid = LiteResBlock(feat_dim, feat_dim)
        self.cross_view = EfficientCrossViewAttn(feat_dim, num_views)
        # Decoder
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(feat_dim, feat_dim, 3, 1, 1))
        self.dec1 = LiteResBlock(feat_dim * 2, feat_dim)        # skip-cat -> 32x32
        self.up0 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(feat_dim, feat_dim, 3, 1, 1))
        self.dec0 = LiteResBlock(feat_dim * 2, feat_dim)        # skip-cat -> 64x64
        # Output projection (zero-init so initial output relies on residual skip)
        self.proj_out = nn.Conv2d(feat_dim, out_channels, 3, 1, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        h = self.proj_in(x)
        # Encoder
        h0 = self.enc0(h)                                       # 64x64
        h1 = self.enc1(self.down0(h0))                          # 32x32
        # Bottleneck
        h = self.mid(self.down1(h1))                             # 16x16
        h = self.cross_view(h)
        # Decoder with skip connections
        h = self.dec1(torch.cat([self.up1(h), h1], dim=1))      # 32x32
        h = self.dec0(torch.cat([self.up0(h), h0], dim=1))      # 64x64
        return self.proj_out(h)


class DepthPredictorMultiView(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
        self,
        feature_channels=128,
        upscale_factor=4,
        num_depth_candidates=32,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=(1, 1, 1),
        costvolume_unet_attn_res=(),
        gaussian_raw_channels=-1,
        gaussians_per_pixel=1,
        num_views=2,
        depth_unet_feat_dim=64,
        depth_unet_attn_res=(),
        depth_unet_channel_mult=(1, 1, 1),
        num_transformer_layers=3,
        num_head=1,
        ffn_dim_expansion=4,
        voxel_feature_dim=32,
        enable_voxel_heads=False,
        **kwargs,
    ):
        super(DepthPredictorMultiView, self).__init__()
        self.num_depth_candidates = num_depth_candidates
        self.regressor_feat_dim = costvolume_unet_feat_dim
        self.upscale_factor = upscale_factor
        self.feature_channels = feature_channels
        
        # Fixed feature extractor and trained cost head
        self.vit_type = "vits"  # can also be 'vitb' or 'vitl'
        self.pretrained = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_{:}14".format(self.vit_type)
        )
        del self.pretrained.mask_token  # unused
        for param in self.pretrained.parameters():
            param.requires_grad = False
        
        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }
        self.depth_head = DPTHead(self.pretrained.embed_dim, 
                                  features=feature_channels, 
                                  use_bn=False, 
                                  out_channels=[48, 96, 192, 384], 
                                  use_clstoken=False)
        for param in self.depth_head.parameters():
            param.requires_grad = False
        
        self.cost_head = CostHead(self.pretrained.embed_dim, 
                                  features=feature_channels, 
                                  use_bn=False, 
                                  out_channels=[48, 96, 192, 384], 
                                  use_clstoken=False)
                
        # Transformer
        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )
        
        # Cost volume refinement (lightweight version)
        input_channels = num_depth_candidates + feature_channels * 2
        channels = self.regressor_feat_dim
        self.corr_refine_net = LiteCostVolumeRefineNet(
            in_channels=input_channels,
            feat_dim=channels,
            out_channels=num_depth_candidates,
            num_views=num_views,
        )
        # self.corr_refine_net = nn.Sequential(
        #     nn.Conv2d(input_channels, channels, 3, 1, 1),
        #     nn.GroupNorm(8, channels),
        #     nn.GELU(),
        #     UNetModel(
        #         image_size=None,
        #         in_channels=channels,
        #         model_channels=channels,
        #         out_channels=channels,
        #         num_res_blocks=1,
        #         attention_resolutions=costvolume_unet_attn_res,
        #         channel_mult=costvolume_unet_channel_mult,
        #         num_head_channels=32,
        #         dims=2,
        #         postnorm=True,
        #         num_frames=num_views,
        #         use_cross_view_self_attn=True,
        #     ),
        #     nn.Conv2d(channels, num_depth_candidates, 3, 1, 1))
        # cost volume u-net skip connection
        self.regressor_residual = nn.Conv2d(input_channels, num_depth_candidates, 1, 1, 0)

        # Depth estimation: project features to get softmax based coarse depth
        self.depth_head_lowres = nn.Sequential(
            nn.Conv2d(num_depth_candidates, num_depth_candidates * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_depth_candidates * 2, num_depth_candidates, 3, 1, 1),
        )

        # # CNN-based feature upsampler
        self.proj_feature_mv = nn.Conv2d(feature_channels, depth_unet_feat_dim, 1, 1)
        self.proj_feature_mono = nn.Conv2d(feature_channels, depth_unet_feat_dim, 1, 1)

        # Depth refinement: 2D U-Net
        input_channels = depth_unet_feat_dim*2 + 3 + 1 + 1 + 1
        channels = depth_unet_feat_dim
        self.refine_unet = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1, 
                attention_resolutions=depth_unet_attn_res,
                channel_mult=depth_unet_channel_mult,
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=num_views,
                use_cross_view_self_attn=True,
            ),
        )

        self.enable_voxel_heads = enable_voxel_heads
        if self.enable_voxel_heads:
            latent_in_channels = depth_unet_feat_dim * 3
            self.conf_head = nn.Sequential(
                nn.Conv2d(latent_in_channels, depth_unet_feat_dim, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(depth_unet_feat_dim, 1, 3, 1, 1),
            )
            self.voxel_feature_head = nn.Sequential(
                nn.Conv2d(latent_in_channels, depth_unet_feat_dim, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(depth_unet_feat_dim, voxel_feature_dim, 1, 1),
            )

        # Gaussians prediction: covariance, color
        gau_in = 3 + depth_unet_feat_dim + 2 * depth_unet_feat_dim
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(
                gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1
            ),
        )

        # Gaussians prediction: centers, opacity
        in_channels = 1 + depth_unet_feat_dim + 1 + 1
        channels = depth_unet_feat_dim
        self.to_disparity = nn.Sequential(
            nn.Conv2d(in_channels, channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
        )
            

    def normalize_images(self, images):
        """Normalize image to match the pretrained UniMatch model.
        images: (B, V, C, H, W)
        """
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std

    
    def forward(
        self,
        images,
        intrinsics,
        extrinsics,
        near,
        far,
        gaussians_per_pixel=1,
        deterministic=True,
        return_aux: bool = False,
        skip_2d_gaussian_head: bool = False,
    ):
        device = extrinsics.device
        if TIMER:
            total_start = _sync_time(device)
            vit_start = _sync_time(device)
        num_reference_views = 1
        # find nearest idxs
        cam_origins = extrinsics[:, :, :3, -1]  # [b, v, 3]
        distance_matrix = torch.cdist(cam_origins, cam_origins, p=2)  # [b, v, v]
        _, idx = torch.topk(distance_matrix, num_reference_views + 1, largest=False, dim=2) # [b, v, m+1]
        
        # first normalize images
        images = self.normalize_images(images)
        b, v, _, ori_h, ori_w = images.shape
        
        # depth anything encoder
        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        concat = F.interpolate(concat, (resize_h, resize_w), mode="bilinear", align_corners=True)
        features = self.pretrained.get_intermediate_layers(concat, 
                                                           self.intermediate_layer_idx[self.vit_type], 
                                                           return_class_token=True)
        if TIMER:
            vit_elapsed = _sync_time(device) - vit_start
            adapter_start = _sync_time(device)
        # new decoder
        features_mono, disps_rel = self.depth_head(features, patch_h=resize_h // 14, patch_w=resize_w // 14)
        features_mv = self.cost_head(features, patch_h=resize_h // 14, patch_w=resize_w // 14)
        features_mv = F.interpolate(features_mv, (64, 64), mode="bilinear", align_corners=True)
        features_mv = mv_feature_add_position(features_mv, 2, 64)
        features_mv_list = list(torch.unbind(rearrange(features_mv, "(b v) c h w -> b v c h w", b=b, v=v), dim=1))
        features_mv_list = self.transformer(
            features_mv_list,
            attn_num_splits=2,
            nn_matrix=idx,
        )
        features_mv = rearrange(torch.stack(features_mv_list, dim=1), "b v c h w -> (b v) c h w")  # [BV, C, H, W]
        if TIMER:
            adapter_elapsed = _sync_time(device) - adapter_start
            cost_volume_start = _sync_time(device)
        # cost volume construction
        features_mv_warped, intr_warped, poses_warped = (
            prepare_feat_proj_data_lists(
                rearrange(features_mv, "(b v) c h w -> b v c h w", v=v, b=b),
                intrinsics,
                extrinsics,
                num_reference_views=num_reference_views,
                idx=idx)
        )
        min_disp = rearrange(1.0 / far.clone().detach(), "b v -> (b v) ()")
        max_disp = rearrange(1.0 / near.clone().detach(), "b v -> (b v) ()")
        disp_range_norm = torch.linspace(0.0, 1.0, self.num_depth_candidates).to(min_disp.device)
        disp_candi_curr = (min_disp + disp_range_norm.unsqueeze(0) * (max_disp - min_disp)).type_as(features_mv) 
        disp_candi_curr = repeat(disp_candi_curr, "bv d -> bv d fh fw", fh=features_mv.shape[-2], fw=features_mv.shape[-1])  # [bxv, d, 1, 1]
        
        raw_correlation_in = []
        for i in range(num_reference_views):
            features_mv_warped_i = warp_with_pose_depth_candidates(
                features_mv_warped[:, i, :, :, :],
                intr_warped[:, i, :, :],
                poses_warped[:, i, :, :],
                1 / disp_candi_curr,
                warp_padding_mode="zeros"
            ) # [B*V, C, D, H, W]
            raw_correlation_in_i = (features_mv.unsqueeze(2) * features_mv_warped_i).sum(1) / (features_mv.shape[1]**0.5) # [B*V, D, H, W]
            raw_correlation_in.append(raw_correlation_in_i)
        raw_correlation_in = torch.mean(torch.stack(raw_correlation_in, dim=1), dim=1)  # [B*V, D, H, W]
        
        if TIMER:
            cost_volume_elapsed = _sync_time(device) - cost_volume_start
            depth_refine_start = _sync_time(device)

        # refine cost volume and get depths
        features_mono_tmp = F.interpolate(features_mono, (64, 64), mode="bilinear", align_corners=True)
        raw_correlation_in = torch.cat((raw_correlation_in, features_mv, features_mono_tmp), dim=1)
        raw_correlation = self.corr_refine_net(raw_correlation_in)
        raw_correlation = raw_correlation + self.regressor_residual(raw_correlation_in)
        pdf = F.softmax(self.depth_head_lowres(raw_correlation), dim=1)
        disps_metric = (disp_candi_curr * pdf).sum(dim=1, keepdim=True) 
        pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]
        pdf_max = F.interpolate(pdf_max, (ori_h, ori_w), mode="bilinear", align_corners=True)
        disps_metric_fullres = F.interpolate(disps_metric, (ori_h, ori_w), mode="bilinear", align_corners=True)

        if TIMER:
            depth_refine_elapsed = _sync_time(device) - depth_refine_start
            feature_refine_start = _sync_time(device)


        features_mv_in_fullres = F.interpolate(features_mv, (ori_h, ori_w), mode="bilinear", align_corners=True)
        features_mv_in_fullres = self.proj_feature_mv(features_mv_in_fullres)
        features_mono_in_fullres = F.interpolate(features_mono, (ori_h, ori_w), mode="bilinear", align_corners=True)
        features_mono_in_fullres = self.proj_feature_mono(features_mono_in_fullres)
        disps_rel_fullres = F.interpolate(disps_rel, (ori_h, ori_w), mode="bilinear", align_corners=True)
        
        images_reorder = rearrange(images, "b v c h w -> (b v) c h w")
        refine_input = torch.cat((features_mv_in_fullres, features_mono_in_fullres, images_reorder, \
                disps_metric_fullres, disps_rel_fullres, pdf_max), 
                    dim=1)
        refine_out = cp.checkpoint(self.refine_unet, refine_input, use_reentrant=False)

        if self.enable_voxel_heads:
            latent_world_features = torch.cat(
                (refine_out, features_mv_in_fullres, features_mono_in_fullres), dim=1
            )
            confidence_map = torch.sigmoid(self.conf_head(latent_world_features))
            voxel_features = self.voxel_feature_head(latent_world_features)

        raw_gaussians = None
        if not skip_2d_gaussian_head:
            raw_gaussians_in = [refine_out, features_mv_in_fullres, features_mono_in_fullres, images_reorder]
            raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)
            raw_gaussians = self.to_gaussians(raw_gaussians_in)

        # delta fine depth and density
        disparity_in = [refine_out, disps_metric_fullres, disps_rel_fullres, pdf_max]
        disparity_in = torch.cat(disparity_in, dim=1)
        delta_disps_density = self.to_disparity(disparity_in)
        delta_disps, raw_densities = delta_disps_density.split(gaussians_per_pixel, dim=1)

        # outputs
        fine_disps = (disps_metric_fullres + delta_disps).clamp(
            1.0 / rearrange(far, "b v -> (b v) () () ()"),
            1.0 / rearrange(near, "b v -> (b v) () () ()"),
        )
        depths = 1.0 / fine_disps
        depths = repeat(
            depths,
            "(b v) dpt h w -> b v (h w) srf dpt",
            b=b,
            v=v,
            srf=1,
        )
        
        densities = repeat(
            F.sigmoid(raw_densities),
            "(b v) dpt h w -> b v (h w) srf dpt",
            b=b,
            v=v,
            srf=1,
        )
        
        if raw_gaussians is not None:
            raw_gaussians = rearrange(raw_gaussians, "(b v) c h w -> b v (h w) c", v=v, b=b)

        
        if return_aux:
            if not self.enable_voxel_heads:
                raise RuntimeError("voxel heads are disabled but return_aux=True was requested")
            aux_dict = {
                "voxel_features": voxel_features,
                "confidence_map": confidence_map,
            }
            return depths, densities, raw_gaussians, aux_dict
        if TIMER:
            feature_refine_elapsed = _sync_time(device) - feature_refine_start
            total_elapsed = _sync_time(device) - total_start
            percents = []
            for t in [vit_elapsed, adapter_elapsed, cost_volume_elapsed, depth_refine_elapsed, feature_refine_elapsed]:
                percents.append(t / total_elapsed * 100)
            print(f"Timing (in seconds): \n ViT encoder {vit_elapsed:.3f} {percents[0]:.1f}%, \n adapter {adapter_elapsed:.3f} {percents[1]:.1f}%, \n cost volume {cost_volume_elapsed:.3f} {percents[2]:.1f}%, \n depth refine {depth_refine_elapsed:.3f} {percents[3]:.1f}%, \n feature refine {feature_refine_elapsed:.3f} {percents[4]:.1f}%, \n total {total_elapsed:.3f}")
        return depths, densities, raw_gaussians 
