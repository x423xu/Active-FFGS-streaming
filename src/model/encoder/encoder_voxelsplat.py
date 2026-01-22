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

from .unimatch.mv_feature_extractor2 import FeatureExtractor
from .unimatch.dpt_head import DPTHead

DEBUG = False

@dataclass
class EncoderVoxelSplatCfg:
    name: Literal["voxelsplat"]
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


class EncoderVoxelSplat(Encoder[EncoderVoxelSplatCfg]):
    def __init__(self, 
                 cfg: EncoderVoxelSplatCfg, 
                 gs_cube:bool = False,
                 vggt_meta:bool = False,
                 **kwargs) -> None:
        super().__init__(cfg)

        self.vggt_meta = vggt_meta

        self.depth_predictor = FeatureExtractor(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
        )

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

        # concat(img, depth, features)
        in_channels = 3 + 1 + feature_upsampler_channels
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
            num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1 # depth offset
        else:
            num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1

        # concat(img, features, regressor_out)
        in_channels = 3 + feature_upsampler_channels + channels
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
            # self.gs_cube_encoder = GSCubeEncoder(
            #     depths=[2, 4, 9, 4, 4],
            #     channels=[48, 96, 192, 384, 384],
            #     num_heads=[6, 6, 12, 24, 24],
            #     window_sizes=[5, 7, 7, 7, 7],
            #     quant_size=4,
            #     in_channels = 132,
            #     down_strides=down_strides,
            #     knn_down=True,
            #     upsample= 'linear_attn',
            #     cRSE='XYZ_RGB',
            #     up_k= 3,
            #     num_classes=13,
            #     stem_transformer=True,
            #     fp16_mode=0,
            #     num_gaussian_parameters = num_gaussian_parameters+1,
            #     gpc = self.cfg.gaussians_per_cell,
            #     cell_scale = self.cfg.cell_scale,
            # )
            self.gs_cube_encoder = GSCubeEncoder(
                depths=[2, 2],
                channels=[256, 384],
                num_heads=[16, 24],
                window_sizes=[5, 5],
                num_layers=2,
                quant_size=4,
                in_channels = in_channels,
                down_strides=down_strides,
                knn_down=True,
                upsample= 'linear_attn',
                cRSE='XYZ_RGB',
                up_k= 3,
                num_classes=13,
                stem_transformer=True,
                fp16_mode=0,
                num_gaussian_parameters = num_gaussian_parameters if vggt_meta else num_gaussian_parameters+1,
                gpc = self.cfg.gaussians_per_cell,
                cell_scale = self.cfg.cell_scale,
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

        
        raw_depth = context["depth"]
        

        # depth prediction
        results_dict = self.depth_predictor(
            context["image"],
            attn_splits_list=[2],
            min_depth=context["near"],
            max_depth=context["far"],
            intrinsics=context["intrinsics"],
            extrinsics=context["extrinsics"],
            nn_matrix=cameras_dist_index,
            raw_depth=raw_depth,
        )
        # list of [B, V, H, W], with all the intermediate depths
        depth_preds = results_dict['depth_preds']

        # [B, V, H, W]
        depth = depth_preds[-1]

        # depth = depth + 0.05*torch.randn_like(depth)  # add noise to depth

        # features [BV, C, H, W]
        features = self.feature_upsampler(results_dict["features_mono_intermediate"],
                                          cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                          mv_features=results_dict["features_mv"][
                                          0] if self.cfg.num_scales == 1 else results_dict["features_mv"][::-1]
                                          )

        # unet input
        concat = torch.cat((
            rearrange(context["image"], "b v c h w -> (b v) c h w"),
            rearrange(depth, "b v h w -> (b v) () h w"),
            features,
        ), dim=1)

        out = self.gaussian_regressor(concat)

        concat = [out,
                    rearrange(context["image"],
                            "b v c h w -> (b v) c h w"),
                    features,]

        out = torch.cat(concat, dim=1)

        gaussians = self.gaussian_head(out)  # [BV, C, H, W]

        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)

        depths = rearrange(depth, "b v h w -> b v (h w) () ()")

        if self.gs_cube:
            '''
            the input_cube_tensor maintains the initialized world coordinates information, e.g., coordinates, rgbs, cell_sizes, xyz_min, xyz_max
            '''
            # depth = depth.detach()
            # depths = depths.detach()
            # out = out.detach()
            gs_cube,coords_sp, input_cube_tensor, input_cube_tensor_perview, nog_pb, nog_min = self.gs_cube_encoder(context["image"], depth, 
                                               rearrange(out, "(b v) c h w -> b v c h w", b=b, v=v), 
                                               extrinsics = context["extrinsics"], 
                                               intrinsics=context["intrinsics"],
                                               depth_min = context["near"][0,0],
                                               depth_max = context["far"][0,0],
                                               num_depth=128,
                                               return_perview=False,
                                               vggt_meta=self.vggt_meta)

            cube_feat = rearrange(gs_cube.F, "n (c gpc) -> n c gpc", gpc=self.gpc)
            cube_opacities = cube_feat[:, :1].sigmoid()

            offset_xyz = cube_feat[:, 1:4].sigmoid()
            voxel_size = input_cube_tensor.cell_sizes
            # xyz = gs_cube.C.type(torch.float32)[:,1:4]
            xyz = gs_cube.C.type(torch.float32)[:,1:4] + coords_sp.F[:, 1:4] - coords_sp.F[:, 1:4].detach()

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
        xy_ray = xy_ray + (offset_xy - 0.5)

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
                    gaussians[..., 2:],
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
                    gaussians[..., 2:],
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
            if self.gs_cube:
                visualization_dump["cube_scales"] = cube_gaussians.scales
                visualization_dump["cube_rotations"] = cube_gaussians.rotations

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
        
        if self.gs_cube:
            # print('scale max', cube_gaussians.scales.max())
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
            # print(depths.shape)  # [B, V, H, W]
            if self.gs_cube:
                return {
                "gaussians": cube_gaussians,
                "depths": depths,
                "nog_pb": nog_pb,
                "nog_min": nog_min,
                }

            return {
                "gaussians": gaussians,
                "depths": depths
            }
        if self.gs_cube:
            # return gaussians and gaussian_cube
            return cube_gaussians
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
