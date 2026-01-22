from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch import einsum
from einops import repeat,rearrange
from Swin3D.modules.mink_layers import assign_feats

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, GaussianCubeAdapter, DenseGaussianAdapter
from .encoder import Encoder
from .gs_cube import GSCubeEncoder
from .visualization.encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg

import torchvision.transforms as T
import torch.nn.functional as F

from .unimatch.mv_unimatch import MultiViewUniMatch
from .unimatch.dpt_head import DPTHead

import MinkowskiEngine as ME
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter_add

DEBUG = False

@dataclass
class EncoderDepthSplatCfg:
    name: Literal["depthsplat"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerDepthSplatCfg
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

class GaussianSpaceMerger(nn.Module):
    def __init__(self, d_in, d_out, voxel_size = 0.001):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.voxel_size = voxel_size

    def forward(self, scores, gaussians):
        # s1: voxelization
        means = gaussians.means
        covariances = gaussians.covariances
        harmonics = gaussians.harmonics
        opacities = gaussians.opacities

        means_min, mean_max = means.min(), means.max()
        means_normlized = (means - means_min) / (mean_max - means_min + 1e-6)
        voxel_num = int((mean_max - means_min) / self.voxel_size) + 1
        voxel_num = min(voxel_num, 1000) 
        voxel_indices = torch.clamp((means_normlized * voxel_num).long(), 0, voxel_num - 1)
        # voxel_indicies_2 = voxel_indices[..., 0] * voxel_num * voxel_num + voxel_indices[..., 1] * voxel_num + voxel_indices[..., 2]
        max_len = 1
        means_list = []
        covariances_list = []
        harmonics_list = []
        opacities_list = []
        for batch_idx in range(scores.shape[0]):
            score_b = scores[batch_idx]
            ind_b = voxel_indices[batch_idx]
            unique_ind_b,inverse_ind = torch.unique(ind_b,
                                                    dim=0, 
                                                    return_inverse=True,)
            means_b = means[batch_idx]
            covariances_b = covariances[batch_idx]
            harmonics_b = harmonics[batch_idx]
            opacities_b = opacities[batch_idx]
            score_b_soft_max = scatter_softmax(score_b, inverse_ind, dim=0)
            means_b_sum = scatter_add(means_b * score_b_soft_max, inverse_ind, dim=0, dim_size=unique_ind_b.shape[0])
            covariances_b_sum = scatter_add(covariances_b * score_b_soft_max.unsqueeze(-1), inverse_ind, dim=0, dim_size=unique_ind_b.shape[0])
            harmonics_b_sum = scatter_add(harmonics_b * score_b_soft_max.unsqueeze(-1), inverse_ind, dim=0, dim_size=unique_ind_b.shape[0])
            opacities_b_sum = scatter_add(opacities_b * score_b_soft_max.squeeze(-1), inverse_ind, dim=0, dim_size=unique_ind_b.shape[0])
            means_list.append(means_b_sum)
            covariances_list.append(covariances_b_sum)
            harmonics_list.append(harmonics_b_sum)
            opacities_list.append(opacities_b_sum)
            if unique_ind_b.shape[0] > max_len:
                max_len = unique_ind_b.shape[0]
        new_means,new_covariances,new_harmonics,new_opacities = [],[],[],[]
        for means, covariances, harmonics, opacities in zip(means_list, covariances_list, harmonics_list, opacities_list):
            pad_len = max_len - means.shape[0]
            if pad_len > 0:
                means = F.pad(means, (0, 0, 0, pad_len), mode='constant', value=0)
                covariances = F.pad(covariances, (0, 0, 0, 0, 0, pad_len), mode='constant', value=0)
                harmonics = F.pad(harmonics, (0, 0, 0, 0, 0, pad_len), mode='constant', value=0)
                opacities = F.pad(opacities, (0, pad_len), mode='constant', value=0)
            new_means.append(means)
            new_covariances.append(covariances)
            new_harmonics.append(harmonics)
            new_opacities.append(opacities)
        new_means = torch.stack(new_means, dim=0)
        new_covariances = torch.stack(new_covariances, dim=0)
        new_harmonics = torch.stack(new_harmonics, dim=0)
        new_opacities = torch.stack(new_opacities, dim=0)

        gaussian_out = Gaussians(new_means, new_covariances, new_harmonics, new_opacities)
        return gaussian_out
class GaussianScorer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.scorer = nn.Sequential(
                nn.Conv2d(d_in, 64,
                          3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(64,
                          d_out, 3, 1, 1, padding_mode='replicate')
            )

    def forward(self, x):
        # s1: voxelization
        x = self.scorer(x)
        return x
    
class EncoderDepthSplat(Encoder[EncoderDepthSplatCfg]):
    def __init__(self, 
                 cfg: EncoderDepthSplatCfg, 
                 gs_cube:bool = False,
                 vggt_meta:bool = False,
                 knn_down:bool=False,
                 gaussian_merge:bool=False) -> None:
        super().__init__(cfg)

        self.vggt_meta = vggt_meta
        

        self.depth_predictor = MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
        )
        self.upsample_factor = cfg.upsample_factor

        if self.cfg.train_depth_only:
            return

        # upsample features to the original resolution
        model_configs = {
            'vits': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'in_channels': 768, 'features': 96, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'in_channels': 1024, 'features': 128, 'out_channels': [128, 256, 512, 1024]},
        }

        self.feature_upsampler = DPTHead(**model_configs[cfg.monodepth_vit_type],
                                        downsample_factor=cfg.upsample_factor,
                                        return_feature=True,
                                        num_scales=cfg.num_scales,
                                        )
        feature_upsampler_channels = model_configs[cfg.monodepth_vit_type]["features"]
        
        # gaussians adapter
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = self.cfg.gaussian_regressor_channels

        # conv regressor
        modules = [
                    nn.Conv2d(in_channels, channels, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, 1, 1),
                ]

        self.gaussian_regressor = nn.Sequential(*modules)

        # predict gaussian parameters: scale, q, sh, offset, opacity
        if self.vggt_meta:
            num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1 +1 # xyz offset
        else:
            num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1

        # concat(img, features, regressor_out, match_prob)
        in_channels = 3 + feature_upsampler_channels + channels + 1
        self.gaussian_head = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters,
                          3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,
                          num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )

        if self.cfg.init_sh_input_img:
            nn.init.zeros_(self.gaussian_head[-1].weight[10:])
            nn.init.zeros_(self.gaussian_head[-1].bias[10:])


        # anysplat merge 
        self.gaussian_merge = gaussian_merge
        if self.gaussian_merge:
            self.gaussian_merger = GaussianSpaceMerger(num_gaussian_parameters, num_gaussian_parameters)
            self.gaussian_scorer = GaussianScorer(in_channels, 1)
        # init scale
        # first 3: opacity, offset_xy
        if self.vggt_meta:
            nn.init.zeros_(self.gaussian_head[-1].weight[3:9])
            nn.init.zeros_(self.gaussian_head[-1].bias[3:9])
        else:
            nn.init.zeros_(self.gaussian_head[-1].weight[3:6])
            nn.init.zeros_(self.gaussian_head[-1].bias[3:6])

        # whether to use the gs cube model
        self.gs_cube = gs_cube
        if gs_cube:
            down_strides = cfg.down_strides
            cube_merge_type = cfg.cube_merge_type
            stem_norm = cfg.stem_norm
            kwargs = {
                'large':{
                    'depths': [2, 4, 9, 4, 4],
                    'channels': [48, 96, 192, 384, 384],
                    'num_heads': [6, 6, 12, 24, 24],
                    'window_sizes': [5, 7, 7, 7, 7],
                    'num_layers':5,
                },
                'base':{
                    'depths': [2, 4, 9, 4],
                    'channels': [48, 96, 192, 384],
                    'num_heads': [6, 6, 12, 24],
                    'window_sizes': [5, 7, 7, 7],
                    'num_layers':4,
                },
                'moderate':{
                    'depths': [2, 4, 4],
                    'channels': [48, 96, 384],
                    'num_heads': [6, 6, 24],
                    'window_sizes': [5, 7, 7],
                    'num_layers':4,
                },
                'small':{
                    'depths': [2, 2],
                    'channels': [256, 384],
                    'num_heads': [16, 24],
                    'window_sizes': [5, 5],
                    'num_layers':2
                },
                'small_d2':{
                    'depths': [1, 1],
                    'channels': [256, 256],
                    'num_heads': [16, 16],
                    'window_sizes': [5, 5],
                    'num_layers':2
                },
                'small_v2':{
                    'depths': [2, 2],
                    'channels': [96, 256],
                    'num_heads': [12, 16],
                    'window_sizes': [5, 5],
                    'num_layers':2
                },
                'small_v3':{
                    'depths': [2, 2],
                    'channels': [256, 256],
                    'num_heads': [16, 16],
                    'window_sizes': [5, 5],
                    'num_layers':2
                },
                'small_v4':{
                    'depths': [2, 2],
                    'channels': [128, 256],
                    'num_heads': [16, 32],
                    'window_sizes': [5, 5],
                    'num_layers':2
                },
                'tiny':{
                    'depths': [2, 2],
                    'channels': [64, 128],
                    'num_heads': [8, 16],
                    'window_sizes': [5, 5],
                    'num_layers':2
                }
            }
            self.gs_cube_encoder = GSCubeEncoder(
                    depths=kwargs[cfg.cube_encoder_type]['depths'],
                    channels=kwargs[cfg.cube_encoder_type]['channels'],
                    num_heads=kwargs[cfg.cube_encoder_type]['num_heads'],
                    window_sizes=kwargs[cfg.cube_encoder_type]['window_sizes'],
                    num_layers=kwargs[cfg.cube_encoder_type]['num_layers'],
                    quant_size=4,
                    in_channels = 132,
                    down_strides=down_strides,
                    knn_down=knn_down,
                    upsample= 'linear_attn',
                    cRSE='XYZ_RGB',
                    up_k= 3,
                    num_classes=13,
                    stem_transformer=True,
                    fp16_mode=0,
                    num_gaussian_parameters = num_gaussian_parameters+1,
                    gpc = self.cfg.gaussians_per_cell,
                    cell_scale = self.cfg.cell_scale,
                    cube_merge_type=cube_merge_type,
                    stem_norm=stem_norm
                )

            self.dense_gaussian_adapter = DenseGaussianAdapter(cfg.gaussian_adapter)
            self.gpc = self.cfg.gaussians_per_cell

            # use the gs cube model
            # self.gaussian_head_x = nn.Sequential(
            #     nn.Conv2d(in_channels, num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
            #     nn.GELU(),
            #     nn.Conv2d(num_gaussian_parameters,num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
            #     nn.GELU(),
            #     nn.AdaptiveAvgPool2d((64, 1)), # if we want rank=k, just change to (64, k)
            #     nn.Flatten(-2, -1),
            #     nn.Linear(64, 64),
            # )
            # self.gaussian_head_y = nn.Sequential(
            #     nn.Conv2d(in_channels, num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
            #     nn.GELU(),
            #     nn.Conv2d(num_gaussian_parameters,num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
            #     nn.GELU(),
            #     nn.AdaptiveAvgPool2d((64, 1)),
            #     nn.Flatten(-2, -1),
            #     nn.Linear(64, 64),
            # )
            # self.gaussian_head_z = nn.Sequential(
            #     nn.Conv2d(256*256, num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
            #     nn.GELU(),
            #     nn.Conv2d(num_gaussian_parameters,num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
            #     nn.GELU(),
            #     nn.AdaptiveMaxPool2d((128, 1)),
            #     nn.Flatten(-2, -1),
            #     nn.Linear(128, 128),
            # )
            # self.gaussian_cube_adapter = GaussianCubeAdapter(cfg.gaussian_adapter)
            # self.coef_module = nn.Sequential(
            #     nn.Conv2d(in_channels, num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
            #     nn.BatchNorm2d(num_gaussian_parameters),
            #     nn.GELU(),
            #     nn.Conv2d(num_gaussian_parameters,num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
            #     nn.BatchNorm2d(num_gaussian_parameters),
            #     nn.GELU(),
            #     nn.AdaptiveAvgPool2d((1, 1)),
            #     nn.Flatten(-3, -1),
            #     nn.Linear(num_gaussian_parameters,1),
            #     # nn.Tanh(),
            # )

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
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        if v > 3:
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)

                cameras_dist_index = cameras_dist_index[:, :, :(self.cfg.local_mv_match + 1)]
        else:
            cameras_dist_index = None

        # depth prediction
        results_dict = self.depth_predictor(
            context["image"],
            attn_splits_list=[2],
            min_depth=1. / context["far"],
            max_depth=1. / context["near"],
            intrinsics=context["intrinsics"],
            extrinsics=context["extrinsics"],
            nn_matrix=cameras_dist_index,
        )

        if self.vggt_meta:
            depth = context["depth"]
            depth_preds = depth[None]
        else:
            # list of [B, V, H, W], with all the intermediate depths
            depth_preds = results_dict['depth_preds']

            # [B, V, H, W]
            depth = depth_preds[-1]

        # depth = depth + 0.2*torch.randn_like(depth)  # add noise to depth

        if self.cfg.train_depth_only:
            # convert format
            # [B, V, H*W, 1, 1]
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                # supervise all the intermediate depth predictions
                num_depths = len(depth_preds)

                # [B, V, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[:(num_depths - 1)], dim=0)
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()")

                # concat in the batch dim
                depths = torch.cat((intermediate_depths, depths), dim=0)

                b *= num_depths

            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": None,
                "depths": depths
            }
        # features [BV, C, H, W]
        sparse_features = results_dict["features_mv"][0].clone().detach().requires_grad_(True)
        results_dict["features_mv"][0] = sparse_features

        features = self.feature_upsampler(results_dict["features_mono_intermediate"],
                                          cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                          mv_features=results_dict["features_mv"][
                                          0] if self.cfg.num_scales == 1 else results_dict["features_mv"][::-1]
                                          )

        # match prob from softmax
        # [BV, D, H, W] in feature resolution
        match_prob = results_dict['match_probs'][-1]
        # torch.save(match_prob, 'match_prob.pth')
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[
            0]  # [BV, 1, H, W]
        match_prob = F.interpolate(
            match_prob, size=depth.shape[-2:], mode='nearest')
        # torch.save(match_prob, 'conf.pth')
        # unet input
        concat = torch.cat((
            rearrange(context["image"], "b v c h w -> (b v) c h w"),
            rearrange(depth, "b v h w -> (b v) () h w"),
            match_prob,
            features,
        ), dim=1)

        out = self.gaussian_regressor(concat)

        concat = [out,
                    rearrange(context["image"],
                            "b v c h w -> (b v) c h w"),
                    features,
                    match_prob]

        out = torch.cat(concat, dim=1)
        depths = rearrange(depth, "b v h w -> b v (h w) () ()")

        if self.gs_cube:
            '''
            the input_cube_tensor maintains the initialized world coordinates information, e.g., coordinates, rgbs, cell_sizes, xyz_min, xyz_max
            '''
            # depth = depth.detach()
            # depths = depths.detach()
            # out = out.detach()
            gs_cube,coords_sp_input, input_cube_tensor, input_cube_tensor_perview, nog_pb, nog_min = self.gs_cube_encoder(context["image"], depth, 
                                            rearrange(out, "(b v) c h w -> b v c h w", b=b, v=v), 
                                            extrinsics = context["extrinsics"], 
                                            intrinsics=context["intrinsics"],
                                            depth_min = context["near"][0,0],
                                            depth_max = context["far"][0,0],
                                            num_depth=128,
                                            return_perview=False,
                                            conf = match_prob,
                                            random_scale=random_scale,
                                            return_selected_ind=return_selected_ind)
            if return_selected_ind and visualization_dump is not None:
                visualization_dump["selected_ind"] = input_cube_tensor_perview
                visualization_dump['cell_sizes'] = input_cube_tensor.cell_sizes
            # torch.save(gs_cube.F, 'notes/gs_cube_F_8.pth')
            # torch.save(gs_cube.C, 'notes/gs_cube_C_8.pth')
       

            cube_feat = rearrange(gs_cube.F, "n (c gpc) -> n c gpc", gpc=self.gpc)
            cube_opacities = cube_feat[:, :1].sigmoid()
            offset_xyz = cube_feat[:, 1:4].sigmoid()
            voxel_size = input_cube_tensor.cell_sizes
            # xyz = gs_cube.C.type(torch.float32)[:,1:4]
            '''Make sure the grad is backpropagated from output opacities to input depth, thus to before encoder'''
            xyz = gs_cube.C.type(torch.float32)[:,1:4] + coords_sp_input.F[:, 1:4] - coords_sp_input.F[:, 1:4].detach()
            #xyz = gs_cube.C.type(torch.float32)[:,1:4]

            if DEBUG:
                from ...geometry.projection import get_fov, homogenize_points
                from PIL import Image
                import numpy as np

                # points = input_cube_tensor.sp.C.clone().type(torch.float32)[:,1:4]
                points = input_cube_tensor_perview[0][0].sp.C.clone().type(torch.float32)[:,1:4]
                # selected_ind = torch.where(input_cube_tensor.sp.C[:,0] == 0)[0]
                # points_selected = points[selected_ind]
                # offset.scatter_(src = (offset_xyz-0.5)* voxel_size[batch_idx],index = selected_ind, dim=0, )
                points = (points+0.5) * voxel_size[0].unsqueeze(0) + input_cube_tensor_perview[0][0].xyz_min[0]
                colors = input_cube_tensor_perview[0][0].retrieve_rgb_from_batch_coords(input_cube_tensor_perview[0][0].sp.C)
                P = context["extrinsics"][0,0]
                K = context["intrinsics"][0,0]
                points_homo = homogenize_points(points)
                points_cam = points_homo@P.inverse().T
                points_img = points_cam[:,:3]@K.T
                points_img = points_img / points_img[..., -1:]
                points_img = points_img[..., :2] * torch.tensor([w, h], device=points_img.device)
                coods_xy = points_img.cpu().numpy().astype(np.int32)
                colors = colors.detach().cpu().reshape(-1, 3).numpy()
                new_img = np.zeros(shape=(w,h,3))
                non_zero_counts = 0
                for i in range(len(coods_xy)):
                    x, y = coods_xy[i]
                    if x > w-1 or x <0:
                        continue
                    if y > h-1 or y < 0:
                        continue
                    new_img[y,x] = colors[i]
                    non_zero_counts+=1
                print(f'{non_zero_counts} non zeros out of {w*h}')
                img = Image.fromarray((new_img * 255).astype(np.uint8))
                img.save('reproject_after_swin3d.png')

            xyz_tmp = xyz.clone()
            offset = offset_xyz.clone()
            for batch_idx in range(voxel_size.shape[0]):
                selected_ind = torch.where(gs_cube.C[:,0] == batch_idx)[0]
                offset[selected_ind] = (offset_xyz[selected_ind]-0.5) * (voxel_size[batch_idx:batch_idx+1].unsqueeze(-1))
                xyz_tmp[selected_ind] = (xyz[selected_ind] + 0.5)* voxel_size[batch_idx] + input_cube_tensor.xyz_min[batch_idx]
                # xyz_tmp[selected_ind] = xyz[selected_ind]* voxel_size[batch_idx] + input_cube_tensor.xyz_min[batch_idx]
            coords_xyz = rearrange(xyz_tmp,"n c -> n c ()") + offset 
            rgbs = input_cube_tensor.retrieve_rgb_from_batch_coords(gs_cube.C)

            gs_cube = assign_feats(gs_cube,gs_cube.F[:,4*self.gpc:])
            
            '''
            It is noted that rgbs only exists where the context image is available, otherwise zeros are used.
            '''
            cube_gaussians = self.dense_gaussian_adapter.forward(
                context["extrinsics"],
                context["intrinsics"],
                rearrange(coords_xyz, "n c l -> n l c"),
                cube_opacities,
                gs_cube,
                input_images=rgbs if self.cfg.init_sh_input_img else None,
                gpc = self.gpc,
            )
            # Dump visualizations if needed.
            if visualization_dump is not None:
                visualization_dump["depth"] = rearrange(
                    depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
                )
                visualization_dump["cube_scales"] = cube_gaussians.scales
                visualization_dump["cube_rotations"] = cube_gaussians.rotations
            cube_gaussians = Gaussians(
                cube_gaussians.means,
                cube_gaussians.covariances,
                cube_gaussians.harmonics,
                cube_gaussians.opacities,
            )
            if self.cfg.return_depth:
                # return depth prediction for supervision
                depths = rearrange(
                    depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
                ).squeeze(-1).squeeze(-1)

                return {
                "gaussians": cube_gaussians,
                "depths": depths,
                "nog_pb": nog_pb,
                "nog_min": nog_min,
                }
            return cube_gaussians
        else:    
            gaussians = self.gaussian_head(out)  # [BV, C, H, W]
            gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)
            if self.gaussian_merge:
                gaussian_scores = self.gaussian_scorer(out)
                gaussian_scores = rearrange(gaussian_scores, "(b v) l h w -> b (v h w) l", b=b, v=v)
            # [B, V, H*W, 1, 1]
            densities = rearrange(
                match_prob, "(b v) c h w -> b v (c h w) () ()", b=b, v=v)
            # [B, V, H*W, 84]
            raw_gaussians = rearrange(
                gaussians, "b v c h w -> b v (h w) c")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:

                # supervise all the intermediate depth predictions
                num_depths = len(depth_preds)

                # [B, V, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[:(num_depths - 1)], dim=0)
                
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()")

                # concat in the batch dim
                depths = torch.cat((intermediate_depths, depths), dim=0)

                # shared color head
                densities = torch.cat([densities] * num_depths, dim=0)
                raw_gaussians = torch.cat(
                    [raw_gaussians] * num_depths, dim=0)

                b *= num_depths

            # [B, V, H*W, 1, 1]
            opacities = raw_gaussians[..., :1].sigmoid().unsqueeze(-1)
            raw_gaussians = raw_gaussians[..., 1:]
            
            # Convert the features and depths into Gaussians.
            if self.vggt_meta:
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
            if self.vggt_meta:
                pixel_size *= 5
                offset_depth = gaussians[..., 2:3].sigmoid()
                depths = depths + (offset_depth-0.5)*0.2
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

            sh_input_images = context["image"]

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                context_extrinsics = torch.cat(
                    [context["extrinsics"]] * len(depth_preds), dim=0)
                context_intrinsics = torch.cat(
                    [context["intrinsics"]] * len(depth_preds), dim=0)

                gaussians = self.gaussian_adapter.forward(
                    rearrange(context_extrinsics, "b v i j -> b v () () () i j"),
                    rearrange(context_intrinsics, "b v i j -> b v () () () i j"),
                    rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                    depths,
                    opacities,
                    rearrange(
                        gaussians[..., 3:],
                        "b v r srf c -> b v r srf () c",
                    ) if self.vggt_meta else rearrange(
                        gaussians[..., 2:],
                        "b v r srf c -> b v r srf () c",
                    ),
                    (h, w),
                    input_images=sh_input_images.repeat(
                        len(depth_preds), 1, 1, 1, 1) if self.cfg.init_sh_input_img else None,
                    vggt_meta=self.vggt_meta
                )


            else:
                gaussians = self.gaussian_adapter.forward(
                    rearrange(context["extrinsics"],
                            "b v i j -> b v () () () i j"),
                    rearrange(context["intrinsics"],
                            "b v i j -> b v () () () i j"),
                    rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                    depths,
                    opacities,
                    rearrange(
                        gaussians[..., 3:],
                        "b v r srf c -> b v r srf () c",
                    ) if self.vggt_meta else rearrange(
                        gaussians[..., 2:],
                        "b v r srf c -> b v r srf () c",
                    ),
                    (h, w),
                    input_images=sh_input_images if self.cfg.init_sh_input_img else None,
                    vggt_meta=self.vggt_meta,
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

            if self.gaussian_merge:
                gaussians = self.gaussian_merger(gaussian_scores, gaussians)

            
            if self.cfg.return_depth:
                # return depth prediction for supervision
                depths = rearrange(
                    depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
                ).squeeze(-1).squeeze(-1)
                # print(depths.shape)  # [B, V, H, W]
                return {
                    "gaussians": gaussians,
                    "depths": depths
                }

            return gaussians

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        return None
    

    def anchor_forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        random_scale: bool = False,
        view_base:int=2,
        anchor_features:bool=False,
        anchor_base=4,
        disorder:bool=False,
        noise_ratio:float=0.0,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        assert v%view_base==0, f'number of views {v} should be multiple of view_base {view_base}'
        # do not split depth estimation if v==view_base
        if v==view_base:
            if v > 3:
                with torch.no_grad():
                    xyzs = context["extrinsics"][:, :, :3, -1].detach()
                    cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                    cameras_dist_index = torch.argsort(cameras_dist_matrix)

                    cameras_dist_index = cameras_dist_index[:, :, :(self.cfg.local_mv_match + 1)]
            else:
                cameras_dist_index = None
            # depth prediction
            results_dict = self.depth_predictor(
                context["image"],
                attn_splits_list=[2],
                min_depth=1. / context["far"],
                max_depth=1. / context["near"],
                intrinsics=context["intrinsics"],
                extrinsics=context["extrinsics"],
                nn_matrix=cameras_dist_index,
            )
        else:
            # chunk the data to v=view_base
            n_chunk = v//view_base
            result_dict_all = []
            contexts_new = {}
            for n in range(n_chunk):       
                context_tmp = {}
                for key in context.keys():
                    # context_tmp[key] = context[key][:, n*view_base:(n+1)*view_base]
                    context_tmp[key] = context[key][:, n::n_chunk]
                    if key not in contexts_new.keys():
                        contexts_new[key] = [context_tmp[key]]
                    else:
                        contexts_new[key].append(context_tmp[key])
                if view_base > 3:
                    with torch.no_grad():
                        xyzs = context_tmp["extrinsics"][:, :, :3, -1].detach()
                        cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                        cameras_dist_index = torch.argsort(cameras_dist_matrix)

                        cameras_dist_index = cameras_dist_index[:, :, :(self.cfg.local_mv_match + 1)]
                else:
                    cameras_dist_index = None
                results_dict_tmp = self.depth_predictor(
                    context_tmp["image"],
                    attn_splits_list=[2],
                    min_depth=1. / context_tmp["far"],
                    max_depth=1. / context_tmp["near"],
                    intrinsics=context_tmp["intrinsics"],
                    extrinsics=context_tmp["extrinsics"],
                    nn_matrix=cameras_dist_index,
                )
                result_dict_all.append(results_dict_tmp)
            # merge the results
            results_dict = {}
            for d in result_dict_all:
                for k, value in d.items():
                    l = len(value)
                    if k not in results_dict:
                        results_dict[k] = [None] * l
                        for i in range(l):
                            results_dict[k][i] = [value[i]]
                    else:
                        for i in range(l):
                            results_dict[k][i].append(value[i])

            for k,value in results_dict.items():
                results_dict[k] = [torch.cat(vv, dim=0) for vv in value]  # concat in the view dim
            
            # modify context accordingly
            for key in contexts_new.keys():
                context[key] = torch.cat(contexts_new[key], dim=1)
        
        # list of [B, V, H, W], with all the intermediate depths
        depth_preds = results_dict['depth_preds']

        # [B, V, H, W]
        depth = depth_preds[-1]

        depth = depth + noise_ratio*torch.randn_like(depth)  # add noise to depth

        if self.cfg.train_depth_only:
            # convert format
            # [B, V, H*W, 1, 1]
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                # supervise all the intermediate depth predictions
                num_depths = len(depth_preds)

                # [B, V, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[:(num_depths - 1)], dim=0)
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()")

                # concat in the batch dim
                depths = torch.cat((intermediate_depths, depths), dim=0)

                b *= num_depths

            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": None,
                "depths": depths
            }
        # features [BV, C, H, W]
        features = self.feature_upsampler(results_dict["features_mono_intermediate"],
                                          cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                          mv_features=results_dict["features_mv"][
                                          0] if self.cfg.num_scales == 1 else results_dict["features_mv"][::-1]
                                          )

        # match prob from softmax
        # [BV, D, H, W] in feature resolution
        match_prob = results_dict['match_probs'][-1]
        # torch.save(match_prob, 'match_prob.pth')
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[
            0]  # [BV, 1, H, W]
        match_prob = F.interpolate(
            match_prob, size=depth.shape[-2:], mode='nearest')
        # torch.save(match_prob, 'conf.pth')
        # unet input
        concat = torch.cat((
            rearrange(context["image"], "b v c h w -> (b v) c h w"),
            rearrange(depth, "b v h w -> (b v) () h w"),
            match_prob,
            features,
        ), dim=1)

        out = self.gaussian_regressor(concat)

        concat = [out,
                    rearrange(context["image"],
                            "b v c h w -> (b v) c h w"),
                    features,
                    match_prob]

        out = torch.cat(concat, dim=1)
        depths = rearrange(depth, "b v h w -> b v (h w) () ()")
        if self.gs_cube:
            '''
            the input_cube_tensor maintains the initialized world coordinates information, e.g., coordinates, rgbs, cell_sizes, xyz_min, xyz_max
            '''
            # depth = depth.detach()
            # depths = depths.detach()
            # out = out.detach()
            if anchor_features:
                gs_cube,coords_sp_input, input_cube_tensor, input_cube_tensor_perview, nog_pb, nog_min = self.gs_cube_encoder.anchor_forward(context["image"], depth, 
                                            rearrange(out, "(b v) c h w -> b v c h w", b=b, v=v), 
                                            extrinsics = context["extrinsics"], 
                                            intrinsics=context["intrinsics"],
                                            depth_min = context["near"][0,0],
                                            depth_max = context["far"][0,0],
                                            num_depth=128,
                                            return_perview=False,
                                            conf = match_prob,
                                            random_scale=False, anchor_base=anchor_base)
            else:
                gs_cube,coords_sp_input, input_cube_tensor, input_cube_tensor_perview, nog_pb, nog_min = self.gs_cube_encoder(context["image"], depth, 
                                                rearrange(out, "(b v) c h w -> b v c h w", b=b, v=v), 
                                                extrinsics = context["extrinsics"], 
                                                intrinsics=context["intrinsics"],
                                                depth_min = context["near"][0,0],
                                                depth_max = context["far"][0,0],
                                                num_depth=128,
                                                return_perview=False,
                                                conf = match_prob,
                                                random_scale=False)
            
       
            # torch.save(gs_cube.F, 'notes/gscube_feats.pth')
            # torch.save(gs_cube.C, 'notes/gscube_coords.pth')
            cube_feat = rearrange(gs_cube.F, "n (c gpc) -> n c gpc", gpc=self.gpc)
            cube_opacities = cube_feat[:, :1].sigmoid()
            offset_xyz = cube_feat[:, 1:4].sigmoid()
            voxel_size = input_cube_tensor.cell_sizes
            # xyz = gs_cube.C.type(torch.float32)[:,1:4]
            '''Make sure the grad is backpropagated from output opacities to input depth, thus to before encoder'''
            xyz = gs_cube.C.type(torch.float32)[:,1:4] + coords_sp_input.F[:, 1:4] - coords_sp_input.F[:, 1:4].detach()
            #xyz = gs_cube.C.type(torch.float32)[:,1:4]

            if DEBUG:
                from ...geometry.projection import get_fov, homogenize_points
                from PIL import Image
                import numpy as np

                # points = input_cube_tensor.sp.C.clone().type(torch.float32)[:,1:4]
                points = input_cube_tensor_perview[0][0].sp.C.clone().type(torch.float32)[:,1:4]
                # selected_ind = torch.where(input_cube_tensor.sp.C[:,0] == 0)[0]
                # points_selected = points[selected_ind]
                # offset.scatter_(src = (offset_xyz-0.5)* voxel_size[batch_idx],index = selected_ind, dim=0, )
                points = (points+0.5) * voxel_size[0].unsqueeze(0) + input_cube_tensor_perview[0][0].xyz_min[0]
                colors = input_cube_tensor_perview[0][0].retrieve_rgb_from_batch_coords(input_cube_tensor_perview[0][0].sp.C)
                P = context["extrinsics"][0,0]
                K = context["intrinsics"][0,0]
                points_homo = homogenize_points(points)
                points_cam = points_homo@P.inverse().T
                points_img = points_cam[:,:3]@K.T
                points_img = points_img / points_img[..., -1:]
                points_img = points_img[..., :2] * torch.tensor([w, h], device=points_img.device)
                coods_xy = points_img.cpu().numpy().astype(np.int32)
                colors = colors.detach().cpu().reshape(-1, 3).numpy()
                new_img = np.zeros(shape=(w,h,3))
                non_zero_counts = 0
                for i in range(len(coods_xy)):
                    x, y = coods_xy[i]
                    if x > w-1 or x <0:
                        continue
                    if y > h-1 or y < 0:
                        continue
                    new_img[y,x] = colors[i]
                    non_zero_counts+=1
                print(f'{non_zero_counts} non zeros out of {w*h}')
                img = Image.fromarray((new_img * 255).astype(np.uint8))
                img.save('reproject_after_swin3d.png')

            xyz_tmp = xyz.clone()
            offset = offset_xyz.clone()
            for batch_idx in range(voxel_size.shape[0]):
                selected_ind = torch.where(gs_cube.C[:,0] == batch_idx)[0]
                offset[selected_ind] = (offset_xyz[selected_ind]-0.5) * (voxel_size[batch_idx:batch_idx+1].unsqueeze(-1))
                xyz_tmp[selected_ind] = (xyz[selected_ind] + 0.5)* voxel_size[batch_idx] + input_cube_tensor.xyz_min[batch_idx]
                # xyz_tmp[selected_ind] = xyz[selected_ind]* voxel_size[batch_idx] + input_cube_tensor.xyz_min[batch_idx]
            coords_xyz = rearrange(xyz_tmp,"n c -> n c ()") + offset 
            rgbs = input_cube_tensor.retrieve_rgb_from_batch_coords(gs_cube.C)

            gs_cube = assign_feats(gs_cube,gs_cube.F[:,4*self.gpc:])
            
            '''
            It is noted that rgbs only exists where the context image is available, otherwise zeros are used.
            '''
            cube_gaussians = self.dense_gaussian_adapter.forward(
                context["extrinsics"],
                context["intrinsics"],
                rearrange(coords_xyz, "n c l -> n l c"),
                cube_opacities,
                gs_cube,
                input_images=rgbs if self.cfg.init_sh_input_img else None,
                gpc = self.gpc,
            )
            # Dump visualizations if needed.
            if visualization_dump is not None:
                visualization_dump["depth"] = rearrange(
                    depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
                )
                visualization_dump["cube_scales"] = cube_gaussians.scales
                visualization_dump["cube_rotations"] = cube_gaussians.rotations
            cube_gaussians = Gaussians(
                cube_gaussians.means,
                cube_gaussians.covariances,
                cube_gaussians.harmonics,
                cube_gaussians.opacities,
            )
            if self.cfg.return_depth:
                # return depth prediction for supervision
                depths = rearrange(
                    depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
                ).squeeze(-1).squeeze(-1)

                return {
                "gaussians": cube_gaussians,
                "depths": depths,
                "nog_pb": nog_pb,
                "nog_min": nog_min,
                }
            return cube_gaussians
        else:    
            # torch.save(out, 'notes/depthsplat_feat.pth')
            gaussians = self.gaussian_head(out)  # [BV, C, H, W]
            gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)
            if self.gaussian_merge:
                gaussian_scores = self.gaussian_scorer(out)
                gaussian_scores = rearrange(gaussian_scores, "(b v) l h w -> b (v h w) l", b=b, v=v)
            # [B, V, H*W, 1, 1]
            densities = rearrange(
                match_prob, "(b v) c h w -> b v (c h w) () ()", b=b, v=v)
            # [B, V, H*W, 84]
            raw_gaussians = rearrange(
                gaussians, "b v c h w -> b v (h w) c")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:

                # supervise all the intermediate depth predictions
                num_depths = len(depth_preds)

                # [B, V, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[:(num_depths - 1)], dim=0)
                
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()")

                # concat in the batch dim
                depths = torch.cat((intermediate_depths, depths), dim=0)

                # shared color head
                densities = torch.cat([densities] * num_depths, dim=0)
                raw_gaussians = torch.cat(
                    [raw_gaussians] * num_depths, dim=0)

                b *= num_depths

            # [B, V, H*W, 1, 1]
            opacities = raw_gaussians[..., :1].sigmoid().unsqueeze(-1)
            raw_gaussians = raw_gaussians[..., 1:]
            
            # Convert the features and depths into Gaussians.
            if self.vggt_meta:
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
            if self.vggt_meta:
                pixel_size *= 5
                offset_depth = gaussians[..., 2:3].sigmoid()
                depths = depths + (offset_depth-0.5)*0.2
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

            sh_input_images = context["image"]

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                context_extrinsics = torch.cat(
                    [context["extrinsics"]] * len(depth_preds), dim=0)
                context_intrinsics = torch.cat(
                    [context["intrinsics"]] * len(depth_preds), dim=0)

                gaussians = self.gaussian_adapter.forward(
                    rearrange(context_extrinsics, "b v i j -> b v () () () i j"),
                    rearrange(context_intrinsics, "b v i j -> b v () () () i j"),
                    rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                    depths,
                    opacities,
                    rearrange(
                        gaussians[..., 3:],
                        "b v r srf c -> b v r srf () c",
                    ) if self.vggt_meta else rearrange(
                        gaussians[..., 2:],
                        "b v r srf c -> b v r srf () c",
                    ),
                    (h, w),
                    input_images=sh_input_images.repeat(
                        len(depth_preds), 1, 1, 1, 1) if self.cfg.init_sh_input_img else None,
                    vggt_meta=self.vggt_meta
                )


            else:
                gaussians = self.gaussian_adapter.forward(
                    rearrange(context["extrinsics"],
                            "b v i j -> b v () () () i j"),
                    rearrange(context["intrinsics"],
                            "b v i j -> b v () () () i j"),
                    rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                    depths,
                    opacities,
                    rearrange(
                        gaussians[..., 3:],
                        "b v r srf c -> b v r srf () c",
                    ) if self.vggt_meta else rearrange(
                        gaussians[..., 2:],
                        "b v r srf c -> b v r srf () c",
                    ),
                    (h, w),
                    input_images=sh_input_images if self.cfg.init_sh_input_img else None,
                    vggt_meta=self.vggt_meta,
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
                    gaussians.covariances,
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

            if self.gaussian_merge:
                gaussians = self.gaussian_merger(gaussian_scores, gaussians)

            
            if self.cfg.return_depth:
                # return depth prediction for supervision
                depths = rearrange(
                    depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
                ).squeeze(-1).squeeze(-1)
                # print(depths.shape)  # [B, V, H, W]
                return {
                    "gaussians": gaussians,
                    "depths": depths
                }

            return gaussians
