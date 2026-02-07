from .encoder import Encoder
import torch
from typing import Optional
from typing import Literal, List
from dataclasses import dataclass
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, GaussianCubeAdapter, DenseGaussianAdapter
from .mde.mono_feature_extractor import MonoFeatureExtractor

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
                ) -> None:
        super().__init__(cfg)

        self.depth_predictor = MonoFeatureExtractor(cfg)


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
        # Implement the forward pass for the efficient encoder
        depth = self.depth_predictor(context['image'])
        return depth
