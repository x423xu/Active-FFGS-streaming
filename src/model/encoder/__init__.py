from typing import Optional, Union

from .encoder import Encoder
from .encoder_depthsplat import EncoderDepthSplat, EncoderDepthSplatCfg
from .encoder_voxelsplat import EncoderVoxelSplat, EncoderVoxelSplatCfg
from .efficient_encoder import EfficientEncoderCfg, EfficientEncoder
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_depthsplat import EncoderVisualizerDepthSplat

ENCODERS = {
    "efficient_encoder":(EfficientEncoder, None),
    "depthsplat": (EncoderDepthSplat, EncoderVisualizerDepthSplat),
    "voxelsplat": (EncoderVoxelSplat, EncoderVisualizerDepthSplat),
}

EncoderCfg = EfficientEncoderCfg | EncoderDepthSplatCfg | EncoderVoxelSplatCfg


def get_encoder(cfg: EncoderCfg, 
                gs_cube: bool, 
                vggt_meta:bool,
                knn_down:bool=False,
                gaussian_merge:bool=False,
                depth_distillation:bool=False
                ) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    if cfg.name == "depthsplat":
        encoder = encoder(cfg, gs_cube, vggt_meta, knn_down, gaussian_merge)
    if cfg.name == "voxelsplat":
        encoder = encoder(cfg, gs_cube, vggt_meta)
    if cfg.name == "efficient_encoder":
        encoder = encoder(cfg, depth_distillation)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
