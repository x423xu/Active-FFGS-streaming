from math import isqrt
from typing import Literal

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import matplotlib.pyplot as plt

from ...geometry.projection import get_fov, homogenize_points

DEBUG = False

def eval_sh_bases(order: int, dirs: torch.Tensor) -> torch.Tensor:
    """
    计算球谐基函数值（支持 3 阶）
    Args:
        order: 球谐阶数（此处固定为 3）
        dirs: 方向向量 [N, 3]，已归一化
    Returns:
        sh_bases: 球谐基值 [N, (order+1)**2]
    """
    # 确保输入方向已归一化
    assert torch.allclose(dirs.norm(dim=-1), torch.ones_like(dirs[:, 0])), "方向向量未归一化！"
    
    # 解构方向向量分量 [N, 3] -> [N]
    x, y, z = dirs.unbind(-1)
    
    if order == 3:
        sh_bases = torch.stack([
            # l=0, m=0
            0.28209479177387814 * torch.ones_like(x),  # Y_{0}^{0}
            
            # l=1, m=-1,0,1
            -0.4886025119029199 * y,                  # Y_{1}^{-1} (原代码符号错误，已修正)
            0.4886025119029199 * z,                   # Y_{1}^{0}
            0.4886025119029199 * x,                   # Y_{1}^{1}
            
            # l=2, m=-2,-1,0,1,2
            1.0925484305920792 * x * y,                # Y_{2}^{-2}
            -1.0925484305920792 * y * z,               # Y_{2}^{-1}
            0.9461746957575601 * (3.0 * z**2 - 1),     # Y_{2}^{0}
            -1.0925484305920792 * x * z,               # Y_{2}^{1}
            0.5462742152960396 * (x**2 - y**2),        # Y_{2}^{2}
            
            # # l=3, m=-3,-2,-1,0,1,2,3
            # -0.5900435899266435 * y * (3*x**2 - y**2), # Y_{3}^{-3}
            # 1.445305721320277 * x * y * z,              # Y_{3}^{-2}
            # -0.4570457994644658 * y * (5*z**2 - 1),     # Y_{3}^{-1}
            # 0.31539156525252005 * (5*z**3 - 3*z),      # Y_{3}^{0}
            # -0.4570457994644658 * x * (5*z**2 - 1),     # Y_{3}^{1}
            # 0.72892666017483 * z * (x**2 - y**2),       # Y_{3}^{2}
            # -0.5900435899266435 * x * (x**2 - 3*y**2)  # Y_{3}^{3}
        ], dim=-1)  # 沿最后一维堆叠 → [N, 9]
        
        return sh_bases  # 形状 [N, 9]

def harmonic_coeffs_to_rgb(sh_coeffs,points,extrinsic):          # Returns [N, 3] RGB in [0,1]
    # Compute view direction
    cam_pos = extrinsic[:3, 3]                       # Camera position
    view_dir = points - cam_pos.unsqueeze(0)          # [N, 3]
    view_dir = torch.nn.functional.normalize(view_dir, dim=-1)          # Normalize
    
    # Evaluate SH bases for view_dir
    sh_bases = eval_sh_bases(order=3, dirs=view_dir)  # [N, 9]
    
    # Compute RGB: dot product per channel
    rgb = torch.einsum("nck,nk->nc", sh_coeffs[...,:1], sh_bases[...,:1])
    
    # Normalize and clamp
    rgb = rgb * 0.28209479177387814 + 0.5  # Scale and offset [4](@ref)

    # points from world to camera space
    # points_camera_space = einsum(extrinsic.inverse(), homogenize_points(points), "i j, b j -> b i")[:, :3]  # [N, 3]
    # Use Z coordinate for depth
    return torch.clamp(rgb, 0, 1)

def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result

def get_projection_matrix_opengl(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = -1
    result[:, 2, 2] = - (far + near) / (far - near)
    result[:, 2, 3] = -2 * (far * near) / (far - near)
    return result

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

def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    use_sh: bool = True,
    vggt_meta: bool = False,
) -> Float[Tensor, "batch 3 height width"]:
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    if vggt_meta:
        K_norm = intrinsics[:, :3, :3].clone()
        # be cautious here. if the context and target have different shapes, here h, w should be the ones of the contexts.
        K_norm[:,0, :] /= w
        K_norm[:,1, :] /= h
        fov_x, fov_y = get_fov(K_norm).unbind(dim=-1)
        del K_norm
        # projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    else:
        fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    '''
    let us check what happed for my rendering
    1. render the points in world coordinates
    2. render the camera, fov, near, far in world coordinates
    3. get the visible points inside fov
    '''


    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        '''
        check visiblity
        '''
        if DEBUG:
            visible = rasterizer.markVisible(gaussian_means[i])
            print('visible rate',visible.sum()/ visible.numel())
            colors = harmonic_coeffs_to_rgb(gaussian_sh_coefficients[i], gaussian_means[i], extrinsics[i])
            render_pcs(gaussian_means[i], colors, gaussian_opacities[i],i)
            import numpy as np
            from PIL import Image
            # visible = rasterizer.markVisible(gaussian_means[i])
            # colors = harmonic_coeffs_to_rgb(gaussian_sh_coefficients[i], gaussian_means[i], extrinsics[i])
            # # render_pcs(gaussian_means[i], colors, gaussian_opacities[i])
            # render_pcs(pc_cam, colors, gaussian_opacities[i], i)
            '''
            given the extrinsics, intrinsics, image_shape, pcs, colors, I want to render the points in the camera space
            '''
            points = gaussian_means[i].detach().clone()
            points_homo = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
            points_cam = points_homo@view_matrix[i]
            # points_img = points_cam@projection_matrix[i]
            points_img = points_cam[:,:3]@intrinsics[i].T
            points_img = points_img / points_img[..., -1:]
            # points_img = points_img[..., :2] * torch.tensor([w//2, h//2], device=points_img.device) + torch.tensor([w//2, h//2], device=points_img.device)
            coods_xy = points_img[...,:2].cpu().numpy().astype(np.uint16)
            colors = colors.detach().cpu().reshape(-1, 3).numpy()
            new_img = np.zeros(shape=(256,256,3))
            for i in range(len(coods_xy)):
                x, y = coods_xy[i]
                x = min(x, 256 - 1)
                y = min(y, 256 - 1)
                new_img[y,x] = colors[i]
            img = Image.fromarray((new_img * 255).astype(np.uint8))
            img.save('reproject_after.png')
            raise "quit"
        
    
        row, col = torch.triu_indices(3, 3)

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )

        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)



def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: dict | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


def render_depth_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
    vggt_meta:bool=False
) -> Float[Tensor, "batch height width"]:
    # Specify colors according to Gaussian depths.
    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

    # Render using depth as color.
    b, _ = fake_color.shape
    result = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        repeat(fake_color, "b g -> b g c ()", c=3),
        gaussian_opacities,
        scale_invariant=scale_invariant,
        use_sh=False,
        vggt_meta=vggt_meta
    )
    return result.mean(dim=1)
