from dataclasses import dataclass

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Union

from ....geometry.projection import get_world_rays
from ....geometry.vggt_geometry import unproject_depth_map_to_point_map
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance

DEBUG = False

@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"] | None,
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"] | None,
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
        vggt_meta:bool=False
    ) -> Gaussians:
            
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        scales = torch.clamp(F.softplus(scales - 4.),
            min=self.cfg.gaussian_scale_min,
            max=self.cfg.gaussian_scale_max,
            )

        assert input_images is not None

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # [2, 2, 65536, 1, 1, 3, 25]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        if input_images is not None:
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        if vggt_meta:
            means = unproject_depth_map_to_point_map(
                coordinates = rearrange(coordinates, "b v (h w) l m n -> (b v) h w (l m n)", h=image_shape[0], w=image_shape[1]),
                depth_map=rearrange(depths, "b v (h w) m n -> (b v) h (w m n)", h=image_shape[0], w=image_shape[1]),
                extrinsics_cam=rearrange(extrinsics, "b v l m n i j -> (b v l m n) i j").inverse(),
                intrinsics_cam=rearrange(intrinsics, "b v l m n i j -> (b v l m n) i j")
            )
            means = rearrange(means, "(b v) h w c -> b v (h w) () () c", b=extrinsics.shape[0], v=extrinsics.shape[1], h=image_shape[0], w=image_shape[1], c=3)
        else:        
            origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
            means = origins + directions * depths[..., None]
        if DEBUG:
            '''
            check if world rays are correct
            '''
            from PIL import Image
            import numpy as np
            import matplotlib.pyplot as plt
            # from ....geometry.vggt_geometry import unproject_depth_map_to_point_map

            # means2 = unproject_depth_map_to_point_map(
            #     depth_map = rearrange(depths, "b v (h w) m n -> (b v) h (w m n)", h=image_shape[0], w=image_shape[1]),
            #     extrinsics_cam=rearrange(extrinsics, "b v l m n i j -> (b v l m n) i j").inverse(),
            #     intrinsics_cam=rearrange(intrinsics, "b v l m n i j -> (b v l m n) i j")
            # )
            # print(means2.shape)

            def render_pcs(points, colors, opacities, num):
                points = points.squeeze().detach().cpu().numpy()
                colors = colors.squeeze().detach().cpu().numpy()
                opacities = opacities.squeeze().detach().cpu().numpy()
                # Load point cloud
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(
                    points[:, 0],  # X-axis
                    points[:, 2],  # Y-axis
                    points[:, 1],  # Z-axis
                    c=colors,      # Color
                    marker='o',    # Point shape
                    s=1,           # Point size
                )
                # ax.invert_yaxis()
                ax.invert_zaxis()  
                fig.savefig(f"point_cloud_{num}.png", dpi=300)
                plt.close(fig)
            points = means.squeeze().reshape(-1,3).cpu()
            # points = means2.reshape(-1, 3)
            colors = rearrange(input_images[0].squeeze(), "v c h w->(v h w) c")
            opacities = opacities[0].squeeze()
            render_pcs(points, colors, opacities, 'adapter')
            
            # points = means[0,0].detach().clone().squeeze().cpu()
            # points = torch.tensor(means[0], dtype=torch.float32)
            points_homo = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
            P = extrinsics[0].squeeze().cpu()[0]
            K = intrinsics[0].squeeze().cpu()[0]
            points_cam = points_homo@P.inverse().T
            points_img = points_cam[..., :3]@K.T
            points_img = points_img / points_img[..., -1:]
            points_img = points_img[..., :2]
            coods_xy = points_img[...,:2].cpu().numpy().astype(np.uint16)
            colors = colors.detach().cpu().numpy()
            new_img = np.zeros(shape=(256,256,3))
            for i in range(len(coods_xy)):
                x, y = coods_xy[i]
                x = min(x, 255)
                y = min(y, 255)
                new_img[y,x] = colors[i]
            img = Image.fromarray((new_img * 255).astype(np.uint8))
            img.save('reproject_before.png')

            raise ValueError("Debugging GaussianAdapter, check point cloud rendering")
            
            

        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            opacities=opacities,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh
    
class GaussianCubeAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def _create_mask(self,means, cube_length=[64,64,128]):
        b,v = means.shape[:2]
        flat_means = rearrange(means.detach(), "b v n i j c -> (b v) (n i j) c")  # BV N 3
        min_vals, max_vals = torch.min(flat_means, dim=1)[0], torch.max(flat_means, dim=1)[0]  # BV 3
        range_vals = (max_vals - min_vals).clamp(min=1e-8)  # BV 3
        normalized_means = (flat_means - min_vals[:, None, :]) / range_vals[:, None, :]  # BV N 3
        scales = torch.tensor(cube_length, dtype=normalized_means.dtype, device=normalized_means.device)  # 3
        normalized_means = normalized_means * scales[None, None, :]  # BV N 3
        indices = normalized_means.round().long()  # BV N 3
        indices[..., 0] = indices[..., 0].clamp(0, cube_length[0]-1)  # x-dim
        indices[..., 1] = indices[..., 1].clamp(0, cube_length[1]-1)  # y-dim
        indices[..., 2] = indices[..., 2].clamp(0, cube_length[2]-1)  # z-dim

        batch_idx = torch.arange(indices.shape[0], device=normalized_means.device)[:,None].expand(indices.shape[0],indices.shape[1])  # B N
        mask = torch.zeros((indices.shape[0], *cube_length), device=normalized_means.device, dtype=torch.bool)
        mask[batch_idx, indices[..., 0], indices[..., 1], indices[..., 2]] = True
        mask = rearrange(mask, "(b v) l m n -> b v l m n", b=b, v=v)  # B V L M N
        union_mask = mask.any(dim=1)  # B L M N
        intersect_mask = mask.all(dim=1)  # B L M N
        return mask, union_mask, intersect_mask



    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"] | None,
        coordinates: Float[Tensor, "*#batch 2"],
        depths,
        opacities,
        raw_gaussians,
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
        mask = None,
        coef = None
    ) -> Gaussians:
        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None,None]
        # calculate mask according to the coordinates
        mask, union_mask_on_v, intersect_mask_on_v = self._create_mask(means, cube_length=[64,64,128]) # (b v l m n), (b l m n), (b l m n)
        complement_mask_on_v = torch.logical_xor(union_mask_on_v, intersect_mask_on_v)  # b l m n
        coef = rearrange(coef, "b v -> b v () () ()") * intersect_mask_on_v[:,None,...]  # b v l m n, average in intersection area
        coef[complement_mask_on_v] = 1.0 # sum up in complement area


        # unfold gaussians
        gaussian_x, gaussian_y, gaussian_z = raw_gaussians.split((64, 64, 128), dim=2)
        gaussian_compose = torch.einsum("bvlijk, bvmijk, bvnijk -> bvlmnijk", gaussian_x, gaussian_y, gaussian_z)  # bv, c, 128, 64, 64

        opacities_x, opacities_y, opacities_z = opacities.split((64, 64, 128), dim=2)
        opacities_compose = torch.einsum("bvlij, bvmij, bvnij -> bvlmnij", opacities_x, opacities_y, opacities_z)
        b, v, l, m,n,i,j,k = gaussian_compose.shape

        gs_per_batch = []
        gs_per_view = []


        # gaussian_compose = rearrange(coef, "b v -> b v () () () () () ()") * gaussian_compose  # b v l m n i j k
        # opacities_compose = rearrange(coef, "b v -> b v () () () () ()") * opacities_compose  # b v l m n i j

        scales, rotations, sh = gaussian_compose.split((3, 4, 3 * self.d_sh), dim=-1)
        # initialize params
        scales = torch.clamp(F.softplus(scales - 4.),
            min=self.cfg.gaussian_scale_min,
            max=self.cfg.gaussian_scale_max,
            )

        assert input_images is not None
        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # [2, 2, 65536, 1, 1, 3, 25]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        if input_images is not None:
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)
        
        '''
        Why do we add the gaussians, how to justify it?
        f(v1, v2, ..., vn) = f(v1) + f(v2) + ... + f(vn)
        '''

        gaussian_compose = (gaussian_compose*mask[..., None,None,None]).sum(dim=1)  # b l m n i j k
        opacities_compose = (opacities_compose*mask[..., None, None]).sum(dim=1)  # b l m n i j
        gs_active = []
        op_active = []
        for num_b in range(gaussian_compose.shape[0]):
            gs_b = gaussian_compose[num_b][union_mask[num_b]]  #
            op_b = opacities_compose[num_b][union_mask[num_b]]  # b v n 1
            gs_active.append(gs_b)
            op_active.append(op_b)
        num_gs = [gs_a.shape[0] for gs_a in gs_active]
        max_num_gs = max(num_gs)
        # pad the gaussians and opacities to the same shape
        gs_all = []
        op_all = []
        for gs_a, op_a in zip(gs_active, op_active):
            if gs_a.shape[0] < max_num_gs:
                pad_size = max_num_gs - gs_a.shape[0]
                gs_a = F.pad(gs_a, (0, 0, 0, 0, 0, pad_size), value=0)
                op_a = F.pad(op_a, (0, 0, 0, pad_size), value=0)
            gs_all.append(gs_a)
            op_all.append(op_a)
        active_gaussians = torch.stack(gs_active, dim=0)  # b h i j k
        active_opacities = torch.stack(op_active, dim=0)  # b h i j
        # select active gaussians and opacities
        # not that easy, here we use for loop to filter out the active gaussians, we plan to use Graph Neural Network to parallel this in the future
        active_gaussians = rearrange(active_gaussians, " (b v) h i j k -> b v h i j k", b=b, v=v, i=i, j=j, k=k)  # b v n 1 1 c
        
        opacities = rearrange(active_opacities, " (b v) h i j -> b v h i j", b=b, v=v)  # b v n 1
        opacities = opacities.sigmoid()  # b v n 1
        scales, rotations, sh = active_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        

        assert input_images is not None

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # [2, 2, 65536, 1, 1, 3, 25]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask


        if input_images is not None:
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        

        means_oracle = means.sum(dim=1, keepdim=True)
        covariances_oracle = covariances.sum(dim=1, keepdim=True)
        harmonics_oracle = rotate_sh(sh, c2w_rotations[..., None, :, :]).sum(dim=1, keepdim=True)
        opacities_oracle = opacities.sum(dim=1, keepdim=True)
        sacles_oracle = scales.sum(dim=1, keepdim=True)
        rotations_oracle = rotations.broadcast_to((*scales.shape[:-1], 4)).sum(dim=1, keepdim=True)
        return Gaussians(
            means=means_oracle,
            covariances=covariances_oracle,
            harmonics=harmonics_oracle,
            opacities=opacities_oracle,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=sacles_oracle,
            rotations=rotations_oracle,
        )
    
    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh



def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


class DenseGaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"] | None,
        coordinates,
        opacities,
        gs_cube,
        eps: float = 1e-8,
        input_images: Tensor | None = None,
        gpc = 1
    ) -> Gaussians:
        
        assert input_images is not None

        # coordinates  =coordinates + 0.1* torch.randn_like(coordinates)  # add noise to coordinates

        batch_feats = []
        batch_opacities = []
        batch_imgs = []
        batch_coords = []
        max_len = 0
        b,v,_,_ = extrinsics.shape
        coords = gs_cube.C

        for i in range(b):
            selected_ind = torch.where(coords[:, 0] == i)[0]
            if selected_ind.shape[0] > max_len:
                max_len = selected_ind.shape[0]
        
        for i in range(b):
            selected_ind = torch.where(coords[:, 0] == i)[0]
            feats_i = gs_cube.F[selected_ind]
            opacities_i = opacities[selected_ind]
            imgs_i = input_images[selected_ind]
            coords_i = coordinates[selected_ind]
            
            if selected_ind.shape[0] < max_len:
                feats_pad = torch.zeros((max_len - selected_ind.shape[0], *feats_i.shape[1:]), device=feats_i.device, dtype=feats_i.dtype)
                feats_cat = torch.cat([feats_i, feats_pad], dim=0)
                opacities_pad = torch.zeros((max_len - selected_ind.shape[0], *opacities_i.shape[1:]), device=opacities_i.device, dtype=opacities_i.dtype)
                opacities_cat = torch.cat([opacities_i, opacities_pad], dim=0)
                imgs_pad = torch.zeros((max_len - selected_ind.shape[0], *imgs_i.shape[1:]), device=imgs_i.device, dtype=imgs_i.dtype)
                imgs_cat = torch.cat([imgs_i, imgs_pad], dim=0)
                coords_pad = torch.zeros((max_len - selected_ind.shape[0], *coords_i.shape[1:]), device=coords_i.device, dtype=coords_i.dtype)
                coords_cat = torch.cat([coords_i, coords_pad], dim=0)
                batch_coords.append(coords_cat)
                batch_imgs.append(imgs_cat)       
                batch_opacities.append(opacities_cat)
                batch_feats.append(feats_cat)
            else:
                batch_coords.append(coords_i)
                batch_imgs.append(imgs_i)
                batch_opacities.append(opacities_i)
                batch_feats.append(feats_i)
        batch_coords = torch.stack(batch_coords, dim=0)  # [B, N, 3]
        batch_feats = torch.stack(batch_feats, dim=0)  # [B, N, C*gpc]
        batch_feats = rearrange(batch_feats, "b n (c gpc) -> b n c gpc", gpc=gpc)  # [B, N, C, gpc]
        batch_imgs = torch.stack(batch_imgs, dim=0)
        batch_opacities = torch.stack(batch_opacities, dim=0) 
        batch_opacities = rearrange(batch_opacities, "b n l gpc -> b n gpc l ()")
        batch_feats = rearrange(batch_feats, "b n c gpc -> b n gpc c") 
        scales, rotations, sh = batch_feats.split((3, 4, 3 * self.d_sh), dim=-1)

        scales = torch.clamp(F.softplus(scales-4.0),
            min=self.cfg.gaussian_scale_min,
            max=self.cfg.gaussian_scale_max,
            )

        

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # 
        sh = rearrange(sh, "b r gpc (xyz d_sh)  -> b r gpc () () xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*batch_opacities.shape, 3, self.d_sh)) * self.sh_mask
        # [B, V, H*W, 1, 1, 3]
        imgs = rearrange(batch_imgs, "b r c -> b r () () c")
        imgs = repeat(imgs, "b r m n c -> b r gpc m n c", gpc=gpc)
        # init sh with input images
        sh[..., 0] = sh[..., 0] + RGB2SH(imgs)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)

        


        # return Gaussians(
        #     means=rearrange(batch_coords, "b r xyz -> b () r () () xyz"),
        #     covariances=rearrange(covariances, "b r m n -> b () r () () m n"),
        #     harmonics=rearrange(sh, "b r l m n k -> b () r l m n k"),
        #     opacities=rearrange(batch_opacities, "b r m n -> b () r m n"),
        #     # NOTE: These aren't yet rotated into world space, but they're only used for
        #     # exporting Gaussians to ply files. This needs to be fixed...
        #     scales=rearrange(scales, "b r n -> b () r () () n"),
        #     rotations=rearrange(rotations.broadcast_to((*scales.shape[:-1], 4)), "b r n -> b () r () () n"),
        # )
        return Gaussians(
            means=rearrange(batch_coords, "b r gpc xyz -> b (r gpc) xyz"),
            covariances=rearrange(covariances, "b r gpc m n -> b (r gpc) m n"),
            harmonics=rearrange(sh, "b r gpc l m n k -> b (r gpc) (l m n) k", l=1, m=1),
            opacities=rearrange(batch_opacities, "b r gpc m n -> b (r gpc m n)",m=1,n=1),
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=rearrange(scales, "b r gpc n -> b (r gpc) n"),
            rotations=rearrange(rotations.broadcast_to((*scales.shape[:-1], 4)), "b r gpc n -> b (r gpc) n"),
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh
