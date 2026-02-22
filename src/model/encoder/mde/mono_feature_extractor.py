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
                 depth_distillation=False,
                 train_controller_cfg=None) -> None:
        super().__init__()
        encoder = vit_type
        model_configs = {
            "vits": {"features": 64, "in_channels": 384, "out_channels": [48, 96, 192, 384]},
            "vitb": {"features": 128, "in_channels": 768, "out_channels": [96, 192, 384, 768]},
            "vitl": {"features": 256, "in_channels": 1024, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"features": 384, "in_channels": 1536, "out_channels": [1536, 1536, 1536, 1536]},
        }
        assert encoder in model_configs, f"Unknown encoder={encoder}"
        self.embedding_type = train_controller_cfg.embedding_type if train_controller_cfg is not None else None
        self.depth_distillation = depth_distillation
        self.use_dav2 = depth_distillation and train_controller_cfg.teacher_depth == "dav2"
        
        self.da_model = self.load_da2(encoder=vit_type, ckpt_path="pretrained_weights/depth_anything_v2_vits.pth", device="cpu",model_configs=model_configs, extra_embedding=self.embedding_type is not None)
        if self.use_dav2:
            self.teacher_model = self.load_da2(encoder=vit_type, ckpt_path="pretrained_weights/depth_anything_v2_vits.pth", device="cpu",model_configs=model_configs, extra_embedding=False)
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
        
        

        # self.scale_shift_head = nn.Sequential(
        #     nn.Conv2d(32,16,kernel_size=3,stride=1, padding=1),
        #     nn.GELU(),
        #     nn.Conv2d(16, 2, 3, 1, 1, padding_mode="replicate"),
        #     nn.AdaptiveAvgPool2d((1,1)),      
        # )

        if DEBUG:
            print("\033[94m MonoFeatureExtractor initialized with cfg: \033[0m", cfg)

    def load_da2(self, encoder: str, ckpt_path: str, device: str, model_configs: dict, extra_embedding:bool=False):
       
        configs = model_configs[encoder].copy()
        configs.pop('in_channels')
        configs["encoder"] = encoder
        configs["extra_embedding"] = extra_embedding
        model = DepthAnythingV2(**configs)
        state = torch.load(ckpt_path, map_location="cpu")
        weight = state['pretrained.patch_embed.proj.weight'].data
        if extra_embedding:
            state['pretrained.patch_embed.proj.weight'].data = torch.cat([
                weight, 
                torch.zeros((weight.shape[0], 6, 14, 14), dtype=weight.dtype, device=weight.device) # for camera embedding
            ], dim=1) 
        model.load_state_dict(state)
        return model.train()
    
    def forward(self, images, min_depth, max_depth, extrinsics=None, intrinsics=None):
        # Define the forward pass for mono feature extraction
        images = self.normalize_images(images)
        b,v,c,h,w = images.shape
        device = images.device
        resize_h, resize_w = images.shape[-2] // 14 * 14, images.shape[-1] // 14 * 14
        img_for_dino = rearrange(images, "b v c h w -> (b v) c h w")
        img_for_dino = F.interpolate(
            img_for_dino, (resize_h, resize_w), mode="bilinear", align_corners=True
        )
        '''
        not sure if use cls influences performance
        '''

        if self.embedding_type == 'plucker_ray':
            ys, xs = torch.meshgrid(
                        torch.arange(h, device=device),
                        torch.arange(w, device=device),
                        indexing="ij"
                    )
            ones = torch.ones_like(xs)
            p = torch.stack([xs, ys, ones], dim=-1).float()  
            p = rearrange(p, "h w c -> (h w) c")
            Kinv = torch.inverse(intrinsics)
            Kinv = rearrange(Kinv, "b v c1 c2 -> (b v) c1 c2")
            d_cam = torch.matmul(Kinv, p.permute(1,0).unsqueeze(0)).permute(0,2,1)  # (BV,HW,3)
            d_cam = d_cam / d_cam.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            T_c2w = extrinsics.view(b*v, 4, 4)  # (B*V,4,4)
            R = T_c2w[:, :3, :3]            # (B,3,3)
            t = T_c2w[:, :3, 3]             # (B,3)
            o_w = t[:, None, :].expand(b*v, h*w, 3)  # (B,HW,3)

            d_w = torch.matmul(R, d_cam.permute(0,2,1)).permute(0,2,1)    # (B,HW,3)
            d_w = d_w / d_w.norm(dim=-1, keepdim=True).clamp_min(1e-8)

            # moment
            m_w = torch.cross(o_w, d_w, dim=-1)  # (BV,HW,3)

            camera_embedding = rearrange(torch.cat([d_w, m_w], dim=-1),'b (h w) c -> b c h w', h=h, w=w)  # (BV,6,H,W)

        elif self.embedding_type == 'dense_intrinsics':
            pass
        elif self.embedding_type == 'dense_KP':
            pass

        if self.embedding_type is not None:
            mono_features = self.da_model.pretrained.get_intermediate_layers(
                img_for_dino, [2,5,8,11], return_class_token=True, extra_embedding=camera_embedding
            )
        else:
            mono_features = self.da_model.pretrained.get_intermediate_layers(
                    img_for_dino, [2,5,8,11], return_class_token=True
                )
        fullres_features = self.feat_head(mono_features, patch_h = images.shape[-2] // 14, patch_w = images.shape[-1] // 14)
        
        
        depth_inverse = self.da_model.depth_head(mono_features, patch_h=images.shape[-2] // 14, patch_w=images.shape[-1] // 14)
        depth = 1/depth_inverse
        # depth = depth + 0.5
        depth = torch.clamp(depth, min_depth[0,0], max_depth[0,0])
        
        
        if self.use_dav2:
            # shift_scale = self.scale_shift_head(fullres_features) # bv 2
            # scale = F.softplus(shift_scale[:, 0, :, :])  # [BV, 1, 1]
            # shift = torch.nn.functional.tanh(shift_scale[:, 1, :, :])  # [BV, 1, 1]
            # depth = depth * scale.unsqueeze(-1) + shift.unsqueeze(-1) # [BV, H, W]
            with torch.no_grad():
                teacher_depth_inverse = self.teacher_model(img_for_dino)
                teacher_depth = 1/teacher_depth_inverse
                teacher_depth = torch.clamp(teacher_depth, min_depth[0,0], max_depth[0,0])
                # teacher_depth = teacher_depth * scale.detach() + shift.detach() # [BV, H, W]
        
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
        if self.use_dav2:
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
