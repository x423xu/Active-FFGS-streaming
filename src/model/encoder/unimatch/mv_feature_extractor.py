import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .vit_fpn import ViTFeaturePyramid
from .mv_transformer import (
    MultiViewFeatureTransformer,
    batch_features_camera_parameters,
)
from .matching import warp_with_pose_depth_candidates
from .utils import mv_feature_add_position
from .dpt_head import DPTHead
from .ldm_unet.unet import UNetModel, AttentionBlock
from einops import rearrange


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        num_scales=1,
        feature_channels=128,
        upsample_factor=8,
        lowest_feature_resolution=8,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        num_depth_candidates=128,
        vit_type="vits",
        unet_channels=128,
        unet_channel_mult=[1, 1, 1],
        unet_num_res_blocks=1,
        unet_attn_resolutions=[4],
        grid_sample_disable_cudnn=False,
        **kwargs,
    ):
        super(FeatureExtractor, self).__init__()

        # CNN
        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.lowest_feature_resolution = lowest_feature_resolution
        self.upsample_factor = upsample_factor

        # monocular backbones: final
        self.vit_type = vit_type

        # cost volume
        self.num_depth_candidates = num_depth_candidates

        # upsampler
        vit_feature_channel_dict = {"vits": 384, "vitb": 768, "vitl": 1024}

        vit_feature_channel = vit_feature_channel_dict[vit_type]

        # CNN
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=num_scales,
            downsample_factor=upsample_factor,
            lowest_scale=lowest_feature_resolution,
            return_all_scales=True,
        )

        # Transformer
        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )

        if self.num_scales > 1:
            # generate multi-scale features
            self.mv_pyramid = ViTFeaturePyramid(
                in_channels=128, scale_factors=[2**i for i in range(self.num_scales)]
            )

        # monodepth
        encoder = vit_type  # can also be 'vitb' or 'vitl'
        self.pretrained = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_{:}14".format(encoder)
        )

        del self.pretrained.mask_token  # unused

        if self.num_scales > 1:
            # generate multi-scale features
            self.mono_pyramid = ViTFeaturePyramid(
                in_channels=vit_feature_channel,
                scale_factors=[2**i for i in range(self.num_scales)],
            )

        # UNet regressor
        self.regressor = nn.ModuleList()
        self.regressor_residual = nn.ModuleList()
        self.depth_head = nn.ModuleList()

        for i in range(self.num_scales):
            curr_depth_candidates = num_depth_candidates // (4**i)
            cnn_feature_channels = 128 - (32 * i)
            mv_transformer_feature_channels = 128 // (2**i)

            mono_feature_channels = vit_feature_channel // (2**i)

            # concat(cost volume, cnn feature, mv feature, mono feature)
            in_channels = (
                1
                + cnn_feature_channels
                + mv_transformer_feature_channels
                + mono_feature_channels
            )

            # unet channels
            channels = unet_channels // (2**i)

            # unet channel mult & unet_attn_resolutions
            if i > 0:
                unet_channel_mult = unet_channel_mult + [1]
                unet_attn_resolutions = [x * 2 for x in unet_attn_resolutions]

            # unet
            modules = [
                nn.Conv2d(in_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            ]

            modules.append(
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=unet_num_res_blocks,
                    attention_resolutions=unet_attn_resolutions,
                    channel_mult=unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=False,
                    num_frames=2,
                    use_cross_view_self_attn=True,
                )
            )

            modules.append(nn.Conv2d(channels, channels, 3, 1, 1))

            self.regressor.append(nn.Sequential(*modules))

            # regressor residual
            self.regressor_residual.append(nn.Conv2d(in_channels, channels, 1))

            # depth head
            self.depth_head.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels, channels * 2, 3, 1, 1, padding_mode="replicate"
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        channels * 2,
                        128,
                        3,
                        1,
                        1,
                        padding_mode="replicate",
                    ),
                )
            )

        self.depth_skip = nn.Sequential(
            nn.Conv2d(1,channels, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(channels, 1, 3, 1, 1, padding_mode="replicate"),
        )

        # upsampler
        # concat(lowres_depth, cnn feature, mv feature, mono feature)
        in_channels = (
            1
            + cnn_feature_channels
            + mv_transformer_feature_channels
            + mono_feature_channels
        )

        model_configs = {
            "vits": {
                "in_channels": 384,
                "features": 32,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "in_channels": 768,
                "features": 48,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "in_channels": 1024,
                "features": 64,
                "out_channels": [128, 256, 512, 1024],
            },
        }

        self.upsampler = DPTHead(
            **model_configs[vit_type],
            downsample_factor=upsample_factor,
            num_scales=num_scales,
        )

        self.grid_sample_disable_cudnn = grid_sample_disable_cudnn

    def normalize_images(self, images):
        """Normalize image to match the pretrained UniMatch model.
        images: (B, V, C, H, W)
        """
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        # images: [B, V, C, H, W]
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        # list of [BV, C, H, W], resolution from high to low
        features = self.backbone(concat)
        # reverse: resolution from low to high
        features = features[::-1]

        return features

    def forward(
        self,
        images,
        attn_splits_list=None,
        intrinsics=None,
        min_depth=1.0 / 0.5,  # inverse depth range
        max_depth=1.0 / 100,
        num_depth_candidates=128,
        extrinsics=None,
        nn_matrix=None,
        raw_depth=None,
        **kwargs,
    ):

        results_dict = {}
        depth_preds = []

        # first normalize images
        images = self.normalize_images(images)
        b, v, _, ori_h, ori_w = images.shape

        # update the num_views in unet attention, useful for random input views
        set_num_views(self.regressor, num_views=v)


        # list of features, resolution low to high
        # list of [BV, C, H, W]
        features_list_cnn = self.extract_feature(images)
        features_list_cnn_all_scales = features_list_cnn
        features_list_cnn = features_list_cnn[: self.num_scales]
        results_dict.update({"features_cnn_all_scales": features_list_cnn_all_scales})
        results_dict.update({"features_cnn": features_list_cnn})

        # mv transformer features
        # add position to features
        attn_splits = attn_splits_list[0]

        # [BV, C, H, W]
        features_cnn_pos = mv_feature_add_position(
            features_list_cnn[0], attn_splits, self.feature_channels
        )

        # list of [B, C, H, W]
        features_list = list(
            torch.unbind(
                rearrange(features_cnn_pos, "(b v) c h w -> b v c h w", b=b, v=v), dim=1
            )
        )
        features_list_mv = self.transformer(
            features_list,
            attn_num_splits=attn_splits,
            nn_matrix=nn_matrix,
        )

        features_mv = rearrange(
            torch.stack(features_list_mv, dim=1), "b v c h w -> (b v) c h w"
        )  # [BV, C, H, W]

        if self.num_scales > 1:
            # multi-scale mv features: resolution from low to high
            # list of [BV, C, H, W]
            features_list_mv = self.mv_pyramid(features_mv)
        else:
            features_list_mv = [features_mv]

        results_dict.update({"features_mv": features_list_mv})

        # mono feature
        ori_h, ori_w = images.shape[-2:]
        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        concat = F.interpolate(
            concat, (resize_h, resize_w), mode="bilinear", align_corners=True
        )

        # get intermediate features
        intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }

        mono_intermediate_features = list(
            self.pretrained.get_intermediate_layers(
                concat, intermediate_layer_idx[self.vit_type], return_class_token=False
            )
        )

        for i in range(len(mono_intermediate_features)):
            curr_features = (
                mono_intermediate_features[i]
                .reshape(concat.shape[0], resize_h // 14, resize_w // 14, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # resize to 1/8 resolution
            curr_features = F.interpolate(
                curr_features,
                (ori_h // 8, ori_w // 8),
                mode="bilinear",
                align_corners=True,
            )
            mono_intermediate_features[i] = curr_features

        results_dict.update({"features_mono_intermediate": mono_intermediate_features})

        # last mono feature
        mono_features = mono_intermediate_features[-1]

        if self.lowest_feature_resolution == 4:
            mono_features = F.interpolate(
                mono_features, scale_factor=2, mode="bilinear", align_corners=True
            )

        if self.num_scales > 1:
            # multi-scale mono features, resolution from low to high
            # list of [BV, C, H, W]
            features_list_mono = self.mono_pyramid(mono_features)
        else:
            features_list_mono = [mono_features]

        results_dict.update({"features_mono": features_list_mono})

        # TODO: add cameras relative postional embeddings, and plucker embeddings
        linear_space = (
                    torch.linspace(0, 1, 128)
                    .type_as(features_list_cnn[0])
                    .view(1, 128, 1, 1)
                )  # [1, D, 1, 1]
        depth_candidates = rearrange(min_depth, "b v-> (b v) () () ()") + linear_space * (
                    rearrange(max_depth, "b v-> (b v) () () ()") - rearrange(min_depth, "b v-> (b v) () () ()")
                )  # [1, D, 1, 1]

        for scale_idx in range(self.num_scales):
            downsample_factor = self.upsample_factor * (
                2 ** (self.num_scales - 1 - scale_idx)
            )
        

            # regressor
            features_cnn = features_list_cnn[scale_idx]  # [BV, C, H, W]

            features_mono = features_list_mono[scale_idx]  # [BV, C, H, W]

            concat = torch.cat(
                ( features_cnn, features_mv, features_mono), dim=1
            )
            depth = rearrange(raw_depth, "b v h w-> (b v) () h w", b=b, v=v)
            depth_resize = F.interpolate(
                depth,
                scale_factor=1/downsample_factor,
                mode="bilinear",
                align_corners=True,
            )
            concat = torch.cat((concat, depth_resize), dim=1)
            out = self.regressor[scale_idx](concat) + self.regressor_residual[
                scale_idx
            ](concat)
            depth_logits = self.depth_head[scale_idx](out)
            depth_softmax = F.softmax(depth_logits, dim=1)
            
            depth = (depth_softmax * depth_candidates.repeat(1,1,depth_softmax.shape[2], depth_softmax.shape[3])).sum(
                dim=1, keepdim=True
            )  # [BV, 1, H, W]

            # upsample to the original resolution for supervison at training time only
            if self.training and scale_idx < self.num_scales - 1:
                depth_bilinear = F.interpolate(
                    depth,
                    scale_factor=downsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                depth_preds.append(depth_bilinear)

            # final output, learned upsampler
            if scale_idx == self.num_scales - 1:
                residual_depth = self.upsampler(
                    mono_intermediate_features,
                    # resolution high to low
                    cnn_features=features_list_cnn_all_scales[::-1],
                    mv_features=(
                        features_mv if self.num_scales == 1 else features_list_mv[::-1]
                    ),
                    depth=depth,
                )

                depth_bilinear = F.interpolate(
                    depth,
                    scale_factor=self.upsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )

                depth_skip = self.depth_skip(rearrange(raw_depth, "b v h w-> (b v) () h w", b=b, v=v))
                depth = (depth_bilinear + residual_depth + depth_skip).clamp(
                    min=min_depth.view(-1, 1, 1, 1), max=max_depth.view(-1, 1, 1, 1)
                )

                depth_preds.append(depth)

        # convert inverse depth to depth
        for i in range(len(depth_preds)):
            depth_preds[i] = rearrange(
                depth_preds[i], "(b v) c h w -> b (v c) h w", b=b, v=v
            )  # [B, V, H, W]

        results_dict.update({"depth_preds": depth_preds})
        return results_dict


def set_num_views(module, num_views):
    if isinstance(module, AttentionBlock):
        module.attention.n_frames = num_views
    elif (
        isinstance(module, nn.ModuleList)
        or isinstance(module, nn.Sequential)
        or isinstance(module, nn.Module)
    ):
        for submodule in module.children():
            set_num_views(submodule, num_views)
