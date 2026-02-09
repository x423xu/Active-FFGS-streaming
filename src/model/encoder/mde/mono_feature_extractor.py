import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.dpt import DPTHead


DEBUG = False
class MonoFeatureExtractor(torch.nn.Module):
    def __init__(self, 
                 cfg,
                 upsampler_factor=4,
                 vit_type='vits',
                 num_scales = 1,
                 depth_distillation=False) -> None:
        super().__init__()
        encoder = vit_type
        model_configs = {
            "vits": {"features": 64, "in_channels": 384, "out_channels": [48, 96, 192, 384]},
            "vitb": {"features": 128, "in_channels": 768, "out_channels": [96, 192, 384, 768]},
            "vitl": {"features": 256, "in_channels": 1024, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"features": 384, "in_channels": 1536, "out_channels": [1536, 1536, 1536, 1536]},
        }
        assert encoder in model_configs, f"Unknown encoder={encoder}"

        self.da_model = self.load_da2(encoder=vit_type, ckpt_path="pretrained_weights/depth_anything_v2_vits.pth", device="cpu",model_configs=model_configs)
        if depth_distillation:
            self.teacher_model = self.load_da2(encoder=vit_type, ckpt_path="pretrained_weights/depth_anything_v2_vits.pth", device="cpu",model_configs=model_configs)
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False
        # upsample features to the original resolution
        feat_head = DPTHead(
                in_channels=model_configs[encoder]["in_channels"],
                features=model_configs[encoder]["features"],
                out_channels=model_configs[encoder]["out_channels"],
                use_clstoken=True
            ).train()
        feat_head.scratch.output_conv2 = nn.Identity()
        self.feat_head = feat_head
        self.depth_distillation = depth_distillation
        if DEBUG:
            print("\033[94m MonoFeatureExtractor initialized with cfg: \033[0m", cfg)

    def load_da2(self, encoder: str, ckpt_path: str, device: str, model_configs: dict):
       
        configs = model_configs[encoder].copy()
        configs.pop('in_channels')
        configs["encoder"] = encoder
        model = DepthAnythingV2(**configs)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        return model.train()
    
    def forward(self, images, min_depth, max_depth):
        # Define the forward pass for mono feature extraction
        images = self.normalize_images(images)
        b,v,c,h,w = images.shape
        resize_h, resize_w = images.shape[-2] // 14 * 14, images.shape[-1] // 14 * 14
        img_for_dino = rearrange(images, "b v c h w -> (b v) c h w")
        img_for_dino = F.interpolate(
            img_for_dino, (resize_h, resize_w), mode="bilinear", align_corners=True
        )
        '''
        not sure if use cls influences performance
        '''
        mono_features = self.da_model.pretrained.get_intermediate_layers(
                img_for_dino, [2,5,8,11], return_class_token=True
            )
        fullres_features = self.feat_head(mono_features, patch_h = images.shape[-2] // 14, patch_w = images.shape[-1] // 14)
        depth_inverse = self.da_model.depth_head(mono_features, patch_h=images.shape[-2] // 14, patch_w=images.shape[-1] // 14)
        depth = 1/depth_inverse
        # depth = depth + 0.5
        depth = torch.clamp(depth, min_depth[0,0], max_depth[0,0])
        
        if self.depth_distillation:
            with torch.no_grad():
                teacher_depth_inverse = self.teacher_model(img_for_dino)
                teacher_depth = 1/teacher_depth_inverse
                teacher_depth = torch.clamp(teacher_depth, min_depth[0,0], max_depth[0,0])
        
        if DEBUG:
            print("\033[94m MonoFeatureExtractor mono_features shape, depth shape \033[0m", mono_features[0][0].shape, depth.shape)
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(depth[0,0].cpu().detach().numpy(), cmap='viridis')
            draw_circle = plt.Circle((40, 240), 5, color='red', fill=False)
            ax1.add_artist(draw_circle)
            print(depth[0,0,40,240].item())
            draw_circle2 = plt.Circle((240, 50), 5, color='blue', fill=False)
            ax1.add_artist(draw_circle2)
            print(depth[0,0,240,50].item())
            plt.colorbar(ax1.imshow(depth[0,0].cpu().detach().numpy(), cmap='viridis'), ax=ax1)
            ax1.set_title('Predicted Depth')
            ax1.axis('off')
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow(depth[1,0].cpu().detach().numpy(), cmap='viridis')
            ax2.set_title('Ground Truth Depth')
            ax2.axis('off')
            plt.colorbar(ax2.imshow(depth[1,0].cpu().detach().numpy(), cmap='viridis'), ax=ax2)
            plt.savefig("debug_depth.png")
            raise NotImplementedError("Debugging MonoFeatureExtractor, only run one batch")
        if self.depth_distillation:
            return fullres_features, depth, depth_inverse, teacher_depth, teacher_depth_inverse
        return fullres_features, depth
            
    
    def normalize_images(self, images):
        """Normalize image to match the pretrained UniMatch model.
        images: (B, V, C, H, W)
        """
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std
