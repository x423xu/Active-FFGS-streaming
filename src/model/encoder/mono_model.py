from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from Swin3D.modules.mink_layers import assign_feats

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapterCfg, DenseGaussianAdapter
from .encoder import Encoder
from .gs_cube import GSCubeEncoder
from .mono.depth_predictor_multiview import DepthPredictorMultiView
from .mono.gaussian_adapter_mono import MonoGaussianAdapter

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class MonoModelCfg:
    name: Literal["mono_model"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: list[int]
    costvolume_unet_attn_res: list[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: list[int]
    depth_unet_channel_mult: list[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool
    monodepth_vit_type: str
    enable_voxelization: bool
    use_plucker_embedding: bool
    voxel_feature_dim: int
    gaussians_per_cell: int
    down_strides: list[int]
    cell_scale: float
    cube_merge_type: str
    voxelization_downsample_factor: int
    voxel_max_points_per_batch: int
    voxel_conf_threshold: float
    profile_voxelization: bool
    voxel_compute_2d_branch: bool
    voxel_low_vram_arch: bool
    voxel_train_depth_predictor: bool
    return_depth: bool


class MonoModel(Encoder[MonoModelCfg]):
    gaussian_adapter: MonoGaussianAdapter
    depth_predictor: DepthPredictorMultiView

    def __init__(
        self,
        cfg: MonoModelCfg,
        depth_distillation: bool = False,
        train_controller_cfg=None,
    ) -> None:
        super().__init__(cfg)
        self.gaussian_adapter = MonoGaussianAdapter(cfg.gaussian_adapter)
        self.enable_voxelization = cfg.enable_voxelization
        self.use_plucker_embedding = cfg.use_plucker_embedding
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=2,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
            voxel_feature_dim=cfg.voxel_feature_dim,
            enable_voxel_heads=cfg.enable_voxelization,
        )
        self.train_depth_predictor = bool(cfg.voxel_train_depth_predictor)
        for param in self.depth_predictor.parameters():
            param.requires_grad = self.train_depth_predictor

        if self.enable_voxelization:
            cube_in_channels = cfg.voxel_feature_dim + (6 if self.use_plucker_embedding else 0)
            num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1
            if cfg.voxel_low_vram_arch:
                cube_depths = [1, 1]
                cube_channels = [128, 192]
                cube_heads = [8, 12]
            else:
                cube_depths = [2, 2]
                cube_channels = [256, 384]
                cube_heads = [16, 24]
            self.gs_cube_encoder = GSCubeEncoder(
                depths=cube_depths,
                channels=cube_channels,
                num_heads=cube_heads,
                window_sizes=[3, 3],
                num_layers=2,
                quant_size=4,
                in_channels=cube_in_channels,
                down_strides=cfg.down_strides,
                knn_down=False,
                upsample='linear',
                cRSE='XYZ_RGB',
                up_k=3,
                num_classes=13,
                stem_transformer=True,
                fp16_mode=0,
                num_gaussian_parameters=num_gaussian_parameters + 1,
                gpc=cfg.gaussians_per_cell,
                cell_scale=cfg.cell_scale,
                cube_merge_type=cfg.cube_merge_type,
            )
            self.dense_gaussian_adapter = DenseGaussianAdapter(cfg.gaussian_adapter)
            self.gpc = cfg.gaussians_per_cell
    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _compute_plucker_embedding(self, context: dict) -> Tensor:
        b, v, _, h, w = context["image"].shape
        device = context["image"].device
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        ones = torch.ones_like(xs)
        p = torch.stack([xs, ys, ones], dim=-1).float()
        p = rearrange(p, "h w c -> (h w) c")
        kinv = torch.inverse(context["intrinsics"]).reshape(b * v, 3, 3)
        d_cam = torch.matmul(kinv, p.permute(1, 0).unsqueeze(0)).permute(0, 2, 1)
        d_cam = d_cam / d_cam.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        t_c2w = context["extrinsics"].reshape(b * v, 4, 4)
        r = t_c2w[:, :3, :3]
        t = t_c2w[:, :3, 3]
        o_w = t[:, None, :].expand(b * v, h * w, 3)
        d_w = torch.matmul(r, d_cam.permute(0, 2, 1)).permute(0, 2, 1)
        d_w = d_w / d_w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        m_w = torch.cross(o_w, d_w, dim=-1)
        return rearrange(torch.cat([d_w, m_w], dim=-1), "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w)

    def _build_2d_gaussians(
        self,
        context: dict,
        depths: Tensor,
        densities: Tensor,
        raw_gaussians: Tensor,
        gpp: int,
        h: int,
        w: int,
        device: torch.device,
        global_step: int,
    ):
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        return self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        random_scale: bool = False,
        return_selected_ind: bool = False,
    ) -> Gaussians | dict:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        

        gpp = self.cfg.gaussians_per_pixel
        if self.enable_voxelization:
            predictor_output = self.depth_predictor(
                context["image"],
                context["intrinsics"],
                context["extrinsics"],
                context["near"],
                context["far"],
                gaussians_per_pixel=gpp,
                deterministic=deterministic,
                return_aux=self.enable_voxelization,
                skip_2d_gaussian_head=self.enable_voxelization and (not self.cfg.voxel_compute_2d_branch),
            )
        if self.enable_voxelization:
            depths, densities, raw_gaussians, aux_dict = predictor_output
        else:
            depths, densities, raw_gaussians = predictor_output

        if self.enable_voxelization:
            if self.cfg.voxel_compute_2d_branch:
                _ = self._build_2d_gaussians(
                    context,
                    depths,
                    densities,
                    raw_gaussians,
                    gpp,
                    h,
                    w,
                    device,
                    global_step,
                )

            depth_map = rearrange(depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w).squeeze(-1).squeeze(-1)
            voxel_features = rearrange(aux_dict["voxel_features"], "(b v) c h w -> b v c h w", b=b, v=v)
            confidence_map = rearrange(aux_dict["confidence_map"], "(b v) c h w -> b v c h w", b=b, v=v)

            image_for_voxel = context["image"]
            ds = max(int(self.cfg.voxelization_downsample_factor), 1)
            if ds > 1:
                h_ds = max(h // ds, 1)
                w_ds = max(w // ds, 1)
                image_for_voxel = rearrange(
                    F.interpolate(
                        rearrange(context["image"], "b v c h w -> (b v) c h w"),
                        size=(h_ds, w_ds),
                        mode="bilinear",
                        align_corners=True,
                    ),
                    "(b v) c h w -> b v c h w",
                    b=b,
                    v=v,
                )
                depth_map = rearrange(
                    F.interpolate(
                        rearrange(depth_map, "b v h w -> (b v) () h w"),
                        size=(h_ds, w_ds),
                        mode="bilinear",
                        align_corners=True,
                    ),
                    "(b v) () h w -> b v h w",
                    b=b,
                    v=v,
                )
                voxel_features = rearrange(
                    F.interpolate(
                        rearrange(voxel_features, "b v c h w -> (b v) c h w"),
                        size=(h_ds, w_ds),
                        mode="bilinear",
                        align_corners=True,
                    ),
                    "(b v) c h w -> b v c h w",
                    b=b,
                    v=v,
                )
                confidence_map = rearrange(
                    F.interpolate(
                        rearrange(confidence_map, "b v c h w -> (b v) c h w"),
                        size=(h_ds, w_ds),
                        mode="bilinear",
                        align_corners=True,
                    ),
                    "(b v) c h w -> b v c h w",
                    b=b,
                    v=v,
                )

            if self.use_plucker_embedding:
                plucker_embed = self._compute_plucker_embedding({
                    "image": image_for_voxel,
                    "intrinsics": context["intrinsics"],
                    "extrinsics": context["extrinsics"],
                })
                voxel_features = torch.cat([voxel_features, plucker_embed], dim=2)

            gs_cube, coords_sp, input_cube_tensor, _, _, _ = self.gs_cube_encoder(
                image_for_voxel,
                depth_map,
                voxel_features,
                extrinsics=context["extrinsics"],
                intrinsics=context["intrinsics"],
                depth_min=context["near"][0, 0],
                depth_max=context["far"][0, 0],
                num_depth=self.cfg.num_depth_candidates,
                return_perview=False,
                vggt_meta=False,
                conf=rearrange(confidence_map, "b v c h w -> (b v) c h w"),
                random_scale=random_scale,
                max_points_per_batch=self.cfg.voxel_max_points_per_batch,
                conf_threshold=self.cfg.voxel_conf_threshold,
            )
            cube_feat = rearrange(gs_cube.F, "n (c gpc) -> n c gpc", gpc=self.gpc)
            cube_opacities = cube_feat[:, :1].sigmoid()
            offset_xyz = cube_feat[:, 1:4].sigmoid()
            voxel_size = input_cube_tensor.cell_sizes
            xyz = gs_cube.C.type(torch.float32)[:, 1:4] + coords_sp.F[:, 1:4] - coords_sp.F[:, 1:4].detach()
            xyz_tmp = xyz.clone()
            offset = offset_xyz.clone()
            for batch_idx in range(voxel_size.shape[0]):
                selected_ind = torch.where(gs_cube.C[:, 0] == batch_idx)[0]
                offset[selected_ind] = (offset_xyz[selected_ind] - 0.5) * (voxel_size[batch_idx:batch_idx + 1].unsqueeze(-1))
                xyz_tmp[selected_ind] = (xyz[selected_ind] + 0.5) * voxel_size[batch_idx] + input_cube_tensor.xyz_min[batch_idx]
            coords_xyz = rearrange(xyz_tmp, "n c -> n c ()") + offset
            rgbs = input_cube_tensor.retrieve_rgb_from_batch_coords(gs_cube.C)
            gs_cube = assign_feats(gs_cube, gs_cube.F[:, 4 * self.gpc:])
            gaussians = self.dense_gaussian_adapter.forward(
                context["extrinsics"],
                context["intrinsics"],
                rearrange(coords_xyz, "n c l -> n l c"),
                cube_opacities,
                gs_cube,
                input_images=rgbs,
                gpc=self.gpc,
            )

            if visualization_dump is not None:
                visualization_dump["voxel_confidence"] = rearrange(confidence_map, "b v c h w -> b v h w c")
                visualization_dump["depth"] = rearrange(depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w)
                visualization_dump["scales"] = gaussians.scales
                visualization_dump["rotations"] = gaussians.rotations
            out_gaussians = Gaussians(
                gaussians.means,
                gaussians.covariances,
                gaussians.harmonics,
                gaussians.opacities,
            )
            if self.cfg.return_depth:
                depth_map = rearrange(
                    depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
                ).squeeze(-1).squeeze(-1)
                return {
                    "gaussians": out_gaussians,
                    "depths": depth_map,
                }
            return out_gaussians

        gaussians = self._build_2d_gaussians(
            context,
            depths,
            densities,
            raw_gaussians,
            gpp,
            h,
            w,
            device,
            global_step,
        )
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

        out_gaussians = Gaussians(
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
        if self.cfg.return_depth:
            depth_map = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            return {
                "gaussians": out_gaussians,
                "depths": depth_map,
            }
        return out_gaussians

    def anchor_forward(
        self,
        context,
        global_step: int,
        deterministic: bool = False,
        visualization_dump=None,
        view_base: int = 2,
        anchor_features: bool = False,
        anchor_base: int = 4,
        noise_ratio: float = 0.0,
    ) -> Gaussians | dict:
        return self.forward(
            context,
            global_step,
            deterministic=deterministic,
            visualization_dump=visualization_dump,
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            return apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size * self.cfg.downscale_factor,
            )

        return data_shim
