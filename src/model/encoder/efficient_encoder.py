from .encoder import Encoder
import torch
import torch.nn as nn
from typing import Optional
from typing import Literal, List
from dataclasses import dataclass
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, GaussianCubeAdapter, DenseGaussianAdapter
from .mde.mono_feature_extractor import MonoFeatureExtractor
from einops import rearrange
from ...geometry.projection import sample_image_grid
from ..types import Gaussians

@dataclass
class EfficientEncoderCfg:
    name: Literal["efficient_encoder"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    gaussian_adapter: GaussianAdapterCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]

    # mv_unimatch
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool

    # depthsplat color branch
    large_gaussian_head: bool
    color_large_unet: bool
    init_sh_input_img: bool
    feature_upsampler_channels: int
    gaussian_regressor_channels: int

    # loss config
    supervise_intermediate_depth: bool
    return_depth: bool

    # only depth
    train_depth_only: bool

    # monodepth config
    monodepth_vit_type: str

    # multi-view matching
    local_mv_match: int

    gaussians_per_cell: int
    down_strides: List[int]
    cell_scale: float
    cube_encoder_type: str  # small, base, large
    cube_merge_type: str
    stem_norm:str

class EfficientEncoder(Encoder[EfficientEncoderCfg]):
    def __init__(self, 
                 cfg: EfficientEncoderCfg, 
                 depth_distillation: bool = False,
                 train_controller_cfg = None
                ) -> None:
        super().__init__(cfg)
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)
        self.depth_predictor = MonoFeatureExtractor(cfg, depth_distillation=depth_distillation,
                                                    train_controller_cfg=train_controller_cfg)

        channels = self.cfg.gaussian_regressor_channels
        # conv regressor
        modules = [
                    nn.Conv2d(32, channels, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, 1, 1),
                ]

        self.gaussian_regressor = nn.Sequential(*modules)
        in_channels = channels + 3  # add RGB image as input
        num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1
        self.gaussian_head = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters,
                          3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,
                          num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )
        nn.init.zeros_(self.gaussian_head[-1].weight[3:6])
        nn.init.zeros_(self.gaussian_head[-1].bias[3:6])

        self.depth_distillation = depth_distillation
        self.use_vggt = depth_distillation and train_controller_cfg.teacher_depth == "vggt"
        self.use_vda = depth_distillation and train_controller_cfg.teacher_depth == "vda"
        self.use_dav2 = depth_distillation and train_controller_cfg.teacher_depth == "dav2"
        self.embedding_type = train_controller_cfg.embedding_type if train_controller_cfg is not None else None
        '''
        pre-train depth predictor, no downsampling
        '''
        if self.cfg.train_depth_only:
            return

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        random_scale: bool = False,
        return_selected_ind: bool = False,
    ):
        b, v, _, h, w = context["image"].shape
        device = context["image"].device
        # Implement the forward pass for the efficient encoder
        if self.use_dav2:
            eps = 1e-6
            if self.embedding_type is not None:
                features, depth, depth_inverse, teacher_depth, teacher_depth_inverse = self.depth_predictor(context['image'],
                                                    min_depth=context["near"],
                                                    max_depth=context["far"],
                                                    extrinsics=context['extrinsics'],
                                                    intrinsics=context['intrinsics'])
            else:
                features, depth, depth_inverse, teacher_depth, teacher_depth_inverse = self.depth_predictor(context['image'],
                                                    min_depth=context["near"],
                                                    max_depth=context["far"])
            B, C, H, W = depth.shape
            N = H * W
            depth_reshape = rearrange(depth_inverse, "b c h w -> b (c h w)")
            teacher_depth_reshape = rearrange(teacher_depth_inverse, "b h w -> b (h w)")
            pred_med = depth_reshape.median(dim=1, keepdim=True).values
            targ_med = teacher_depth_reshape.median(dim=1, keepdim=True).values
            pred_mad = (depth_reshape - pred_med).abs().mean(dim=1, keepdim=True)
            targ_mad = (teacher_depth_reshape - targ_med).abs().mean(dim=1, keepdim=True)
            pred_hat = (depth_reshape - pred_med) / (pred_mad + eps)
            targ_hat = (teacher_depth_reshape - targ_med) / (targ_mad + eps)
            ai_mae_loss = (pred_hat - targ_hat).abs().mean()
        
        elif self.use_vggt:
            
            eps = 1e-6
            if self.embedding_type is not None:
                features, depth = self.depth_predictor(context['image'],
                                                    min_depth=context["near"],
                                                    max_depth=context["far"],
                                                    extrinsics=context['extrinsics'],
                                                    intrinsics=context['intrinsics'])
            else:
                features, depth = self.depth_predictor(context['image'],
                                                    min_depth=context["near"],
                                                    max_depth=context["far"])
            teacher_depth = context["depth"]
            d_inverse = 1/depth
            td_inverse = 1/teacher_depth
            depth_reshape = rearrange(d_inverse, "b c h w -> b (c h w)")
            teacher_depth_reshape = rearrange(td_inverse, "b v h w -> (b v) (h w)")
            pred_med = depth_reshape.median(dim=1, keepdim=True).values
            targ_med = teacher_depth_reshape.median(dim=1, keepdim=True).values
            pred_mad = (depth_reshape - pred_med).abs().mean(dim=1, keepdim=True)
            targ_mad = (teacher_depth_reshape - targ_med).abs().mean(dim=1, keepdim=True)
            pred_hat = (depth_reshape - pred_med) / (pred_mad + eps)
            targ_hat = (teacher_depth_reshape - targ_med) / (targ_mad + eps)
            ai_mae_loss = (pred_hat - targ_hat).abs().mean()

        else:
            if self.embedding_type is not None:
                features, depth = self.depth_predictor(context['image'],
                                                        min_depth=context["near"],
                                                        max_depth=context["far"],
                                                        extrinsics=context['extrinsics'],
                                                        intrinsics=context['intrinsics'])
            else:
                features, depth = self.depth_predictor(context['image'],
                                                        min_depth=context["near"],
                                                        max_depth=context["far"])
        gaussian_features = self.gaussian_regressor(features)
        concat = torch.cat([gaussian_features, 
                            rearrange(context["image"],"b v c h w -> (b v) c h w")], dim=1)
        gaussians = self.gaussian_head(concat)  # [BV, C, H, W]
        depths = rearrange(depth, "(b v) c h w -> b v (c h w) () ()", b=b, v=v)
        raw_gaussians = rearrange(
                gaussians, "(b v) c h w -> b v (h w) c", b=b, v=v)
        
        opacities = raw_gaussians[..., :1].sigmoid().unsqueeze(-1)
        raw_gaussians = raw_gaussians[..., 1:]

        if self.use_vggt:
            _, xy_ray = sample_image_grid((h, w), device)
        else:
            xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        
        pixel_size = 1 / \
            torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        sh_input_images = context["image"]
        gaussians = self.gaussian_adapter.forward(
                    rearrange(context["extrinsics"],
                            "b v i j -> b v () () () i j"),
                    rearrange(context["intrinsics"],
                            "b v i j -> b v () () () i j"),
                    rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                    depths,
                    opacities,
                    rearrange(
                        gaussians[..., 2:],
                        "b v r srf c -> b v r srf () c",
                    ),
                    (h, w),
                    input_images=sh_input_images if self.cfg.init_sh_input_img else None,
                    vggt_meta=self.use_vggt,
                )
        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        # print('scale max', gaussians.scales.max())
        gaussians = Gaussians(
                rearrange(
                    gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    3*gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
            )
        
        if self.cfg.return_depth:
                # return depth prediction for supervision
                depths = rearrange(
                    depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
                ).squeeze(-1).squeeze(-1)
                # print(depths.shape)  # [B, V, H, W]
                return_dict = {
                    "gaussians": gaussians,
                    "depths": depths
                }
                if self.use_dav2:
                    return_dict.update({"ai_mae_loss": ai_mae_loss})
                    return_dict.update({"teacher_depth": rearrange(
                        teacher_depth, "(b v) h w -> b v h w", b=b, v=v
                    )})
                if self.use_vggt:
                    return_dict.update({"ai_mae_loss": ai_mae_loss})
                    return_dict.update({"teacher_depth": teacher_depth})
                return return_dict
        return gaussians
