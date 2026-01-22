
'''
Cube Encoder for GS-Cube
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from Swin3D.models import Swin3DUNet
from ....geometry.projection import get_world_rays
from ....geometry.vggt_geometry import unproject_depth_map_to_point_map
from ....geometry.projection import sample_image_grid
from einops import rearrange, repeat
import MinkowskiEngine as ME
from torch_scatter import scatter_max, scatter_mean, scatter_add, scatter_min
from Swin3D.modules.mink_layers import assign_feats
import torch.utils.checkpoint as cp

class GSCubeInput():
    def __init__(self, feats, coords_centers, positions, imgs, cell_sizes, xyz_min, xyz_max, device='cpu'):
        self.sp = ME.SparseTensor(
            features=feats,
            coordinates=coords_centers,
            device=device
        )
        self.coords_sp = ME.SparseTensor(
            features=positions,
            coordinate_map_key=self.sp.coordinate_map_key, 
            coordinate_manager=self.sp.coordinate_manager,
            device=device
        )
        self.imgs = imgs  # BxVxCxHxW, range: [0, 1]
        self.cell_sizes = cell_sizes
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max

    @property
    def coords_centers(self):
        return self.sp.C # batch_ind, xyz in grid centers

    @ property
    def position_centers(self):
        return torch.floor(self.coords_sp.F[:, 1:4])+0.5
    
    @property
    def true_positions(self):
        return self.coords_sp.F[:, :4] # batch_ind, xyz
     
    @property
    def rgbs(self):
        return self.coords_sp.F[:, 4:7]  # RGB values
    
    def retrieve_rgb_from_batch_coords(self, coords):
        '''
        Retrieve RGB features from coordinates that should be covered by sp.C.
        coords: Nx4, batch_ind, xyz
        '''
        # get the indices of coords in whole_coords
        rgbs = self.coords_sp.features_at_coordinates(coords.float())[:,4:7]
        return rgbs

    def retrieve_positions_from_batch_coords(self, coords):
        '''
        Retrieve positions from coordinates that should be covered by sp.C.
        coords: Nx4, batch_ind, xyz
        '''
        positions = (coords[:, 1:4] + 0.5)*self.cell_sizes  # convert to grid centers
        return positions

class GSCubeHead(nn.Module):
    def __init__(self, inchannels=48, num_gaussian_parameters=48, gpc=4):
        super(GSCubeHead, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(inchannels, num_gaussian_parameters),
            nn.GELU(),
            nn.Linear(num_gaussian_parameters, gpc*num_gaussian_parameters),
        )
        
    def forward(self, x):
        x = self.l1(x)
        return x

class GSCubeEncoder(Swin3DUNet):
    def __init__(
            self, depths, channels, num_heads, window_sizes,
            quant_size, drop_path_rate=0.2, up_k=3,
            num_layers=5, num_classes=13, stem_transformer=True, 
            down_strides=[2,2], upsample='linear', knn_down=True,
            in_channels=6, cRSE='XYZ_RGB', fp16_mode=0, stem_norm='bn',
            num_gaussian_parameters=48, gpc = 4, cell_scale = 1,cube_merge_type='max'):
        super(GSCubeEncoder, self).__init__(
            depths=depths, channels=channels, num_heads=num_heads, 
            window_sizes=window_sizes, quant_size=quant_size, 
            drop_path_rate=drop_path_rate, up_k=up_k, 
            num_layers=num_layers, num_classes=num_classes, 
            stem_transformer=stem_transformer, upsample=upsample, 
            first_down_stride=down_strides[0], 
            other_down_stride=down_strides[1], 
            knn_down=knn_down, 
            in_channels=in_channels, cRSE=cRSE, fp16_mode=fp16_mode, stem_norm=stem_norm
        )
        self.gs_cube_head = GSCubeHead(inchannels=channels[0], num_gaussian_parameters=num_gaussian_parameters, gpc = gpc)
        self.gpc = gpc
        self.cell_scale = cell_scale
        self.cube_merge_type = cube_merge_type
        # self.classifier = nn.Identity()  # to disable the original classifier

    def encode(self, sp, coords_sp): 
        # sp: MinkowskiEngine SparseTensor for feature input
        # sp.F: input features,         NxC
        # sp.C: input coordinates,      Nx4
        # coords_sp: MinkowskiEngine SparseTensor for position and feature embedding
        # coords_sp.F: embedding
        #       Batch: 0,...0,1,...1,2,...2,...,B,...B
        #       XYZ:   in Voxel Scale
        #       RGB:   in [-1,1]
        #       NORM:  in [-1,1]
        #       Batch, XYZ:             Nx4
        #       Batch, XYZ, RGB:        Nx7
        #       Batch, XYZ, RGB, NORM:  Nx10
        # coords_sp.C: input coordinates: Nx4
        sp_stack = []
        coords_sp_stack = []
        sp = self.stem_layer(sp)
        if self.layer_start > 0:
            sp_stack.append(sp)
            coords_sp_stack.append(coords_sp)
            sp, coords_sp = self.downsample(sp, coords_sp)

        for i, layer in enumerate(self.layers):
            coords_sp_stack.append(coords_sp)
            sp, sp_down, coords_sp = layer(sp, coords_sp)
            # sp_down, coords_sp, sp = cp.checkpoint(layer, sp, coords_sp)
            sp_stack.append(sp)
            assert (coords_sp.C == sp_down.C).all()
            sp = sp_down
        return sp_stack, coords_sp_stack, sp_down.C.shape[0]       

    def decode(self, sp_stack, coords_sp_stack):
        sp = sp_stack.pop()
        coords_sp = coords_sp_stack.pop()
        for i, upsample in enumerate(self.upsamples):
            sp_i = sp_stack.pop()
            coords_sp_i = coords_sp_stack.pop()
            sp = upsample(sp, coords_sp, sp_i, coords_sp_i)
            coords_sp = coords_sp_i

        return sp, coords_sp

    # def forward(self, sp, coords_sp):
        
    #     sp_stack, coords_sp_stack = self.encode(sp, coords_sp)
    #     sp_f, output = self.decode(sp_stack, coords_sp_stack)
        
    #     return sp_f, output
    def forward(self, imgs, 
                depth, feats, 
                extrinsics=None, 
                intrinsics=None,
                depth_min=0.5,
                depth_max=100.0,
                num_depth=128,
                return_perview=False,
                vggt_meta=False,
                conf=None,
                random_scale=False,
                return_selected_ind=False):
        '''
        imgs: context images, shape: BxVxCxHxW, range: [0, 1]
        depth: depth map, shape: BxVxHxW
        feats: additional features, shape: BxVxLxHxW
        '''
        # s1: get world coordinates from depth
        b,v,_,h,w = imgs.shape
        device = imgs.device
        if vggt_meta:
            _, coordinates = sample_image_grid((h, w), device)
            xyz_world = unproject_depth_map_to_point_map(
                coordinates = repeat(coordinates.float(), "h w c -> (b v) h w c", b=b, v=v),
                depth_map=rearrange(depth, "b v h w -> (b v) h w", h=h, w=w),
                extrinsics_cam=rearrange(extrinsics, "b v i j -> (b v) i j").inverse(),
                intrinsics_cam=rearrange(intrinsics, "b v i j -> (b v) i j")
            )
            xyz_world = rearrange(xyz_world, "(b v) h w c -> (b v) (h w) c", b=b, v=v, h=h, w=w, c=3)
        else:
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = torch.broadcast_to(xy_ray, (b, v, h, w, 2))
            origins, directions = get_world_rays(
                rearrange(xy_ray,"b v h w xy->(b v) (h w) xy"), 
                rearrange(extrinsics,"b v h w->(b v) () h w"), 
                rearrange(intrinsics,"b v h w->(b v) () h w"))
            depth = rearrange(depth, "b v h w->(b v) (h w)")
            xyz_world = origins + directions * depth[..., None] # bv, hw, 3
        if torch.isnan(xyz_world).any() or torch.isinf(xyz_world).any():
            # torch.set_printoptions(threshold=float('inf'))
            raise ValueError("xyzworld, depth: NaN or Inf detected in forward output")
        # s2: voxelization, each cell contains at most one point
        # print(xyz_world.shape, feats.shape)
        gs_cube_input, gs_cube_input_perview = self.voxelize(xyz_world, 
                      rearrange(feats,"b v l h w -> b (v h w) l"), 
                      rearrange(imgs,"b v c h w -> b (v h w) c"), 
                      num_depth, b=b, v=v, h=h, w=w, 
                      conf=rearrange(conf, "(b v) l h w -> b (v h w) l", b=b, v=v),
                      return_perview=return_perview, random_scale=random_scale, return_selected_ind=return_selected_ind)
        # gs_cube_input, gs_cube_input_perview = self.extensible_voxelization(xyz_world, 
        #               rearrange(feats,"b v l h w -> b (v h w) l"), 
        #               rearrange(imgs,"b v c h w -> b (v h w) c"), 
        #               num_depth, b=b, v=v, h=h, w=w, return_perview=return_perview)
        # s3: encode
        
        # register buffer
        # sparse_features = gs_cube_input.sp.F.clone().detach().requires_grad_(True)
        # self.register_buffer('sparse_features', sparse_features)
        # gs_cube_input.sp = assign_feats(gs_cube_input.sp, self.sparse_features)
        # torch.save(sparse_features, "visualize/sp_features.pth")
        # torch.save(gs_cube_input.sp.C,"visualize/sp_coordinates.pth")

        sp_stack, coords_sp_stack, nog_min = self.encode(gs_cube_input.sp, gs_cube_input.coords_sp)
        # s4: decode

        sp, coords_sp = self.decode(sp_stack, coords_sp_stack)
        spf = self.gs_cube_head(sp.F)  # [N, KxC]
        sp = assign_feats(sp, spf)
        nog_pb = [(self.gpc*sp.C[:,0]==i).sum() for i in range(b)]
        # nog_pv = self.gpc*sp.C.shape[0]//(b * v)
        nog_min = nog_min // b

        if return_perview:
            return sp, coords_sp, gs_cube_input, gs_cube_input_perview, nog_pb, nog_min
        if return_selected_ind:
            return sp, gs_cube_input.coords_sp, gs_cube_input, gs_cube_input_perview, nog_pb, nog_min
        return sp, gs_cube_input.coords_sp, gs_cube_input, None, nog_pb, nog_min

    '''
    coords: Nx3
    unique_coords: Mx3
    inverse_ind: N
    feats: NxL
    imgs: Nx3
    '''
    def voxelize(self, xyz_world, feats, imgs, num_depth, b=None, v=None, h=None, w=None, conf=None, return_perview=False, random_scale=False, return_selected_ind=False):
        '''
        some notes about intermidiate variables:
        xyz_world: world coordinates, shape: (b, v*h*w, 3), range: physical distances
        xyz_nomralized: normalized coordinates, shape: (b, v*h*w, 3), range: [0, 1]
        xyz_scaled: true positions in a 3D mesh grid that is a float type, shape: (b, v*h*w, 3)
        selected_ind: indices of selected points.
        It is noted that if we voxelize the points with shape H,W,D, we have to use randomly sampling to ensure each voxel has at most one point.
        However, this is not an ideal case, since we may lose some points. We do this to make sure the voxelization is efficient.
        There are some other options such as Adaptive voxelization: we can choose a proper xyz_min, xyz_max, cell_size to include only one point in each voxel. This doesn't need the random sampling.
        Moreover, we can also impose hierarchical voxelization, which means we can first voxelize the points with a coarse grid, and then refine the grids, where each cell contains K>>1 points, to finer ones.
        In this case, we have to be careful about the parents and sons' coordinates, sons' coordinates cannot be float. There needs an extra sparse convolution, unaware of coordinate system, to aggregate these finer cells.
        '''
        
        device = xyz_world.device
        '''get grid_coords per batch'''
        xyz_world = rearrange(xyz_world, "(b v) (h w) xyz -> b (v h w) xyz", b=b, v=v, h=h, w=w)
        xyz_min = xyz_world.min(dim=1, keepdim=True)[0]
        xyz_max = xyz_world.max(dim=1, keepdim=True)[0]
        if torch.isnan(xyz_max).any() or torch.isinf(xyz_max).any() or torch.isnan(xyz_min).any() or torch.isinf(xyz_min).any():
            # torch.set_printoptions(threshold=float('inf'))
            raise ValueError("xyzmax,xyzmin: NaN or Inf detected in forward output")
        assert (xyz_max > xyz_min).all(), "xyz_max should be larger than xyz_min. now xyz_min: {}, xyz_max: {}".format(xyz_min, xyz_max)
        xyz_normalized = (xyz_world - xyz_min) / (xyz_max - xyz_min)
        # we should have hxwxnum_depth grid cells
        grid_shape = torch.tensor([h, w, num_depth], dtype=torch.float32, device=device)
        # factor = torch.randint(1,self.cell_scale+1,size=()).item() if random_scale else self.cell_scale
        # if not self.training and random_scale:
        #     factor=4
        grid_shape = grid_shape*self.cell_scale
        grid_shape = grid_shape.long()
        cell_sizes = (xyz_max - xyz_min) / grid_shape
        cell_sizes = cell_sizes.squeeze(1) 
        # cell ind:0 -> h-1, 0->w-1, 0->num_depth-1
        xyz_scaled = xyz_normalized * (grid_shape-1e-4)
        grid_coords = torch.floor(xyz_scaled).long()
        # gradient_placeholder = xyz_scaled-xyz_scaled.detach()
        view_ind = torch.cat([torch.full((b, h*w, 1), view) for view in range(v)], dim=1).to(device) # b (v h w) 1
        '''get grid_coords per view'''
        if return_perview:
            view_ind = torch.cat([torch.full((b, h*w, 1), view) for view in range(v)], dim=1).to(device) # b (v h w) 1
            grid_coords_view = torch.cat([grid_coords, view_ind], dim=-1)  # b (v h w) 4
            position_all_views = []
            coords_all_views = []
            feats_all_views = []

        '''get per batch and per view sparse tensors'''
        batch_positions =  []
        batch_coords_centers = []
        batch_feats = []
        # batch_view_ind = []

        
        for batch_idx in range(xyz_world.shape[0]):  
            '''randomly select one within each voxel'''
            # Ensure each cell has at most one point
            unique_coords, inverse_ind, counts = torch.unique(
                grid_coords[batch_idx,:,:], 
                dim=0, 
                return_inverse=True, 
                return_counts=True, 
                sorted=False
            )
            if self.cube_merge_type == 'max':
                # randomly select one point in each voxel by choosing the max random value
                rand_val = torch.rand(grid_coords[batch_idx,:,:].size(0), device=device)
                _, selected_ind = scatter_max(
                    rand_val,
                    inverse_ind,
                    dim=0,
                    dim_size=unique_coords.shape[0]
                    )
                unique_grid_coords = grid_coords[batch_idx,selected_ind,:]
                feats_unique = feats[batch_idx, selected_ind, :]
                imgs_unique = imgs[batch_idx, selected_ind, :]
                positions = xyz_scaled[batch_idx, selected_ind, :]
            elif self.cube_merge_type == 'sum':
                feats_conf = feats*conf
                feats_unique = scatter_add(
                    feats_conf[batch_idx,:,:], 
                    inverse_ind, 
                    dim=0, 
                    dim_size=unique_coords.shape[0]
                )
                # index of first occureence
                selected_ind = scatter_min(torch.arange(inverse_ind.shape[0], device=inverse_ind.device), inverse_ind)[1]
                unique_grid_coords = unique_coords
                imgs_unique = imgs[batch_idx, selected_ind, :]
                positions = xyz_scaled[batch_idx, selected_ind, :]
            elif self.cube_merge_type == 'mean':
                feats_conf = feats*conf
                feats_unique = scatter_add(
                    feats_conf[batch_idx,:,:], 
                    inverse_ind, 
                    dim=0, 
                    dim_size=unique_coords.shape[0]
                )
                selected_ind = scatter_min(torch.arange(inverse_ind.shape[0], device=inverse_ind.device), inverse_ind)[1]
                feats_unique = feats_unique / counts[:, None]
                unique_grid_coords = unique_coords
                imgs_unique = imgs[batch_idx, selected_ind, :]
                positions = xyz_scaled[batch_idx, selected_ind, :]
                # views = view_ind[batch_idx, selected_ind, :]
            elif self.cube_merge_type == 'max_conf':
                _, selected_ind = scatter_max(
                    conf[batch_idx,:,0],
                    inverse_ind,
                    dim=0,
                    dim_size=unique_coords.shape[0]
                    )
                unique_grid_coords = grid_coords[batch_idx,selected_ind,:]
                feats_unique = feats[batch_idx, selected_ind, :]
                imgs_unique = imgs[batch_idx, selected_ind, :]
                positions = xyz_scaled[batch_idx, selected_ind, :]

            # concatenate the batch index with coords
            coords_centers = torch.cat([torch.full((unique_grid_coords.shape[0], 1), batch_idx).to(device), unique_grid_coords], dim=-1)  # Nx4
            positions = torch.cat([torch.full((positions.shape[0], 1), batch_idx).to(device), positions, imgs_unique], dim=-1)  # Nx7
            batch_coords_centers.append(coords_centers)
            batch_positions.append(positions)    
            batch_feats.append(feats_unique)
            # batch_view_ind.append(views)

            if return_perview:
                ''' the same process for each view'''
                unique_coords_view, inverse_ind_view = torch.unique(
                    grid_coords_view[batch_idx,:,:], 
                    dim=0, 
                    return_inverse=True, 
                    return_counts=False, 
                    sorted=False
                )
                rand_val_view = torch.rand(grid_coords_view[batch_idx,:,:].size(0), device=device)
                _, selected_ind_view = scatter_max(
                    rand_val_view,
                    inverse_ind_view,
                    dim=0,
                    dim_size=unique_coords_view.shape[0]
                )
                unique_grid_coords_view = grid_coords_view[batch_idx,selected_ind_view,:]
                feats_unique_view = feats[batch_idx, selected_ind_view, :]
                imgs_unique_view = imgs[batch_idx, selected_ind_view, :]
                positions_view = xyz_scaled[batch_idx, selected_ind_view, :]
                # concatenate the batch index with coords
                coords_centers_view = torch.cat([torch.full((unique_grid_coords_view.shape[0], 1), batch_idx).to(device), unique_grid_coords_view], dim=-1)  # Nx4
                positions_view = torch.cat([torch.full((positions_view.shape[0], 1), batch_idx).to(device), positions_view, imgs_unique_view], dim=-1)
                coords_per_view = []
                position_per_view = []
                feats_per_view = []
                for view_id in range(v):
                    ind = torch.where(coords_centers_view[:,4] == view_id)[0]
                    coords_per_view.append(coords_centers_view[ind,:4])  # Nx4
                    position_per_view.append(positions_view[ind,:])  # Nx7
                    feats_per_view.append(feats_unique_view[ind,:])  # NxC
                coords_all_views.append(coords_per_view)
                position_all_views.append(position_per_view)
                feats_all_views.append(feats_per_view)
        # stack the coords
        batch_coords_centers = torch.cat(batch_coords_centers, dim=0) 
        batch_positions = torch.cat(batch_positions, dim=0)    
        batch_feats = torch.cat(batch_feats, dim=0) 
        # batch_view_ind = torch.cat(batch_view_ind, dim=0) 
        # torch.save(rearrange(batch_view_ind, 'r c->(r c) ()'), 'notes/gscube_viewind.pth')

        gs_cube_tensor = GSCubeInput(
            feats=batch_feats, 
            coords_centers=batch_coords_centers, 
            positions=batch_positions, 
            imgs=None,
            cell_sizes=cell_sizes,
            xyz_min=xyz_min,
            xyz_max=xyz_max,
            device=device
        )
        if return_perview:
            gs_cube_tensor_bv = {}
            for batch_idx in range(b):
                gs_cube_tensor_bv[batch_idx] = {}
                for view_id in range(v):
                    gs_cube_tensor_bv[batch_idx][view_id] = GSCubeInput(
                        feats=feats_all_views[batch_idx][view_id], 
                        coords_centers=coords_all_views[batch_idx][view_id], 
                        positions=position_all_views[batch_idx][view_id], 
                        imgs=None,
                        cell_sizes=cell_sizes,
                        xyz_min=xyz_min[batch_idx:batch_idx+1],
                        xyz_max=xyz_max[batch_idx:batch_idx+1],
                        device=device
                    )
            return gs_cube_tensor, gs_cube_tensor_bv
        if return_selected_ind:
            return gs_cube_tensor, selected_ind
        return gs_cube_tensor, None
    
    def anchor_forward(self, imgs, 
                depth, feats, 
                extrinsics=None, 
                intrinsics=None,
                depth_min=0.5,
                depth_max=100.0,
                num_depth=128,
                return_perview=False,
                vggt_meta=False,
                conf=None,
                random_scale=False,
                anchor_base=4):
        '''
        imgs: context images, shape: BxVxCxHxW, range: [0, 1]
        depth: depth map, shape: BxVxHxW
        feats: additional features, shape: BxVxLxHxW
        '''
        # s1: get world coordinates from depth
        b,v,_,h,w = imgs.shape
        device = imgs.device
        if vggt_meta:
            _, coordinates = sample_image_grid((h, w), device)
            xyz_world = unproject_depth_map_to_point_map(
                coordinates = repeat(coordinates.float(), "h w c -> (b v) h w c", b=b, v=v),
                depth_map=rearrange(depth, "b v h w -> (b v) h w", h=h, w=w),
                extrinsics_cam=rearrange(extrinsics, "b v i j -> (b v) i j").inverse(),
                intrinsics_cam=rearrange(intrinsics, "b v i j -> (b v) i j")
            )
            xyz_world = rearrange(xyz_world, "(b v) h w c -> (b v) (h w) c", b=b, v=v, h=h, w=w, c=3)
        else:
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = torch.broadcast_to(xy_ray, (b, v, h, w, 2))
            origins, directions = get_world_rays(
                rearrange(xy_ray,"b v h w xy->(b v) (h w) xy"), 
                rearrange(extrinsics,"b v h w->(b v) () h w"), 
                rearrange(intrinsics,"b v h w->(b v) () h w"))
            depth = rearrange(depth, "b v h w->(b v) (h w)")
            xyz_world = origins + directions * depth[..., None] # bv, hw, 3
        if torch.isnan(xyz_world).any() or torch.isinf(xyz_world).any():
            # torch.set_printoptions(threshold=float('inf'))
            raise ValueError("xyzworld, depth: NaN or Inf detected in forward output")
        # s2: voxelization, each cell contains at most one point
        gs_cube_input, gs_cube_input_perview = self.extensible_voxelization(xyz_world, 
                      rearrange(feats,"b v l h w -> b (v h w) l"), 
                      rearrange(imgs,"b v c h w -> b (v h w) c"), 
                      num_depth, b=b, v=v, h=h, w=w, 
                      conf=rearrange(conf, "(b v) l h w -> b (v h w) l", b=b, v=v),
                      return_perview=return_perview,
                      anchor_base=anchor_base)
        # s3: encode
        sp_stack, coords_sp_stack, nog_min = self.encode(gs_cube_input.sp, gs_cube_input.coords_sp)
        # s4: decode

        sp, coords_sp = self.decode(sp_stack, coords_sp_stack)
        spf = self.gs_cube_head(sp.F)  # [N, KxC]
        sp = assign_feats(sp, spf)
        nog_pb = [(self.gpc*sp.C[:,0]==i).sum() for i in range(b)]
        # nog_pv = self.gpc*sp.C.shape[0]//(b * v)
        nog_min = nog_min // b

        if return_perview:
            return sp, coords_sp, gs_cube_input, gs_cube_input_perview, nog_pb, nog_min
        return sp, gs_cube_input.coords_sp, gs_cube_input, None, nog_pb, nog_min
    def extensible_voxelization(self, xyz_world, feats, imgs, num_depth, b=None, v=None, h=None, w=None, conf=None,return_perview=False, anchor_base=4):
        '''
        compared with plain voxelization, this function can handle unknown input number of views
        ''' 
        if v>=2:
            device = xyz_world.device 
            grid_shape = torch.tensor([h, w, num_depth], dtype=torch.float32, device=device)
            grid_shape = grid_shape*self.cell_scale
            imgs_reshape = rearrange(imgs, "b (v h w) c -> b v (h w) c", b=b, v=v, h=h, w=w)
            xyz_world_reshape = rearrange(xyz_world, "(b v) (h w) xyz -> b v (h w) xyz", b=b, v=v, h=h, w=w)
            feats_reshape = rearrange(feats, "b (v h w) l -> b v (h w) l", b=b, v=v, h=h, w=w)
            conf_reshape = rearrange(conf, "b (v h w) l -> b v (h w) l", b=b, v=v, h=h, w=w)
            nv = anchor_base
            xyz_world_anchor = xyz_world_reshape[:,:nv]  # use the first 4 views as the anchor
            imgs_anchor = imgs_reshape[:, :nv]  # use the first 4 views as the anchor
            feats_anchor  = feats_reshape[:, :nv]
            conf_anchor  = conf_reshape[:, :nv]
            xyz_world_anchor = rearrange(xyz_world_anchor, "b v (h w) xyz -> b (v h w) xyz", b=b, v=nv, h=h, w=w)
            imgs_anchor = rearrange(imgs_anchor, "b v (h w) c -> b (v h w) c", b=b, v=nv, h=h, w=w)
            feats_anchor = rearrange(feats_anchor, "b v (h w) l -> b (v h w) l", b=b, v=nv, h=h, w=w)
            conf_anchor = rearrange(conf_anchor, "b v (h w) l -> b (v h w) l", b=b, v=nv, h=h, w=w)

            xyz_world_rest = xyz_world_reshape[:, nv:]  # the rest views
            imgs_rest = imgs_reshape[:, nv:]
            feats_rest = feats_reshape[:, nv:]
            conf_rest = conf_reshape[:, nv:]
            xyz_world_rest = rearrange(xyz_world_rest, "b v (h w) xyz -> b (v h w) xyz", b=b, v=v-nv, h=h, w=w)
            imgs_rest = rearrange(imgs_rest, "b v (h w) c -> b (v h w) c", b=b, v=v-nv, h=h, w=w)
            feats_rest = rearrange(feats_rest, "b v (h w) l -> b (v h w) l", b=b, v=v-nv, h=h, w=w)
            conf_rest = rearrange(conf_rest, "b v (h w) l -> b (v h w) l", b=b, v=v-nv, h=h, w=w)

            conf_rest_selected_ind = torch.where(conf_rest[0,:,0]>0.95)[0]
            xyz_world_rest = xyz_world_rest[:,conf_rest_selected_ind,:]
            imgs_rest = imgs_rest[:,conf_rest_selected_ind,:]
            feats_rest = feats_rest[:,conf_rest_selected_ind,:]
            conf_rest = conf_rest[:,conf_rest_selected_ind,:]

            '''get grid_coords per batch'''
           
            xyz_min = xyz_world_anchor.min(dim=1, keepdim=True)[0]
            xyz_max = xyz_world_anchor.max(dim=1, keepdim=True)[0]
            cell_sizes = (xyz_max - xyz_min) / grid_shape
            cell_sizes = cell_sizes.squeeze(1) 
            assert (xyz_max > xyz_min).all(), "xyz_max should be larger than xyz_min"
    
            xyz_world_rest_normalized = (xyz_world_rest - xyz_min) / (xyz_max - xyz_min)
            xyz_world_anchor_normalized = (xyz_world_anchor - xyz_min) / (xyz_max - xyz_min)

            xyz_world_rest_scaled = xyz_world_rest_normalized * (grid_shape-1e-4)
            xyz_world_anchor_scaled = xyz_world_anchor_normalized * (grid_shape-1e-4)
            xyz_world_rest_coords = torch.floor(xyz_world_rest_scaled).long()
            xyz_world_anchor_coords = torch.floor(xyz_world_anchor_scaled).long()


            xyz_scaled = xyz_world_anchor_scaled
            grid_coords = xyz_world_anchor_coords
            imgs = imgs_anchor
            feats = feats_anchor
            conf = conf_anchor

            '''get per batch and per view sparse tensors'''
            batch_positions =  []
            batch_coords_centers = []
            batch_feats = []
            
            for batch_idx in range(xyz_world_anchor.shape[0]):  
                '''randomly select one within each voxel'''
                # Ensure each cell has at most one point
                unique_coords, inverse_ind, counts = torch.unique(
                    grid_coords[batch_idx,:,:], 
                    dim=0, 
                    return_inverse=True, 
                    return_counts=True, 
                    sorted=False
                )

                if self.cube_merge_type == 'max':
                    # randomly select one point in each voxel by choosing the max random value
                    rand_val = torch.rand(grid_coords[batch_idx,:,:].size(0), device=device)
                    _, selected_ind = scatter_max(
                        rand_val,
                        inverse_ind,
                        dim=0,
                        dim_size=unique_coords.shape[0]
                        )
                    unique_grid_coords = grid_coords[batch_idx,selected_ind,:]
                    feats_unique = feats[batch_idx, selected_ind, :]
                    imgs_unique = imgs[batch_idx, selected_ind, :]
                    positions = xyz_scaled[batch_idx, selected_ind, :]
                    conf = conf[batch_idx, selected_ind, :]
                    shape_before_remove = selected_ind.shape[0]

                    merge_coords = torch.cat([unique_grid_coords, xyz_world_rest_coords[batch_idx,:,:]],dim=0)  # r xyz
                    merge_feats = torch.cat([feats_unique, feats_rest[batch_idx,:,:]],dim=0)  # r l
                    merge_imgs = torch.cat([imgs_unique, imgs_rest[batch_idx,:,:]],dim=0)  # r c
                    merge_positions = torch.cat([positions, xyz_world_rest_scaled[batch_idx,:,:]],dim=0)  # r xyz
                    _, inverse_ind = torch.unique(
                        merge_coords, 
                        dim=0, 
                        return_inverse=True, 
                        return_counts=False, 
                        sorted=False
                    )
                    selected_ind = scatter_min(torch.arange(inverse_ind.shape[0], device=inverse_ind.device), inverse_ind)[1]
                    unique_grid_coords = merge_coords[selected_ind,:]
                    feats_unique = merge_feats[selected_ind,:]
                    imgs_unique = merge_imgs[selected_ind,:]
                    positions = merge_positions[selected_ind,:]

                    shape_after_remove = selected_ind.shape[0]
                elif self.cube_merge_type == 'sum':
                    feats_conf = feats*conf
                    feats_unique = scatter_add(
                        feats_conf[batch_idx,:,:], 
                        inverse_ind, 
                        dim=0, 
                        dim_size=unique_coords.shape[0]
                    )
                    # index of first occureence
                    selected_ind = scatter_min(torch.arange(inverse_ind.shape[0], device=inverse_ind.device), inverse_ind)[1]
                    unique_grid_coords = unique_coords
                    imgs_unique = imgs[batch_idx, selected_ind, :]
                    positions = xyz_scaled[batch_idx, selected_ind, :]
                elif self.cube_merge_type == 'mean':
                    feats_conf = feats*conf
                    feats_rest_conf = feats_rest*conf_rest
                    feats_unique = scatter_add(
                        feats_conf[batch_idx,:,:], 
                        inverse_ind, 
                        dim=0, 
                        dim_size=unique_coords.shape[0]
                    )
                    selected_ind = scatter_min(torch.arange(inverse_ind.shape[0], device=inverse_ind.device), inverse_ind)[1]
                    # feats_unique = feats_unique / counts[:, None]
                    unique_grid_coords = unique_coords
                    imgs_unique = imgs[batch_idx, selected_ind, :]
                    positions = xyz_scaled[batch_idx, selected_ind, :]

                    conf = conf[batch_idx, selected_ind, :]
                    shape_before_remove = selected_ind.shape[0]
                    merge_coords = torch.cat([unique_grid_coords, xyz_world_rest_coords[batch_idx,:,:]],dim=0)  # r xyz
                    merge_feats = torch.cat([feats_unique, feats_rest_conf[batch_idx,:,:]],dim=0)  # r l
                    merge_imgs = torch.cat([imgs_unique, imgs_rest[batch_idx,:,:]],dim=0)  # r c
                    merge_positions = torch.cat([positions, xyz_world_rest_scaled[batch_idx,:,:]],dim=0)  # r xyz
                    merge_conf = torch.cat([conf, conf_rest[batch_idx,:,:]],dim=0)  # r l
                    unique_coords2, inverse_ind2 = torch.unique(
                        merge_coords, 
                        dim=0, 
                        return_inverse=True, 
                        return_counts=False, 
                        sorted=False)
                    feats_unique2 = scatter_add(
                        merge_feats, 
                        inverse_ind2, 
                        dim=0, 
                        dim_size=unique_coords2.shape[0]
                    )
                    counts_tmp = torch.cat([counts, torch.ones((xyz_world_rest_coords[batch_idx,:,:].shape[0],), device=device)], dim=0)
                    counts2 = scatter_add(
                        counts_tmp, 
                        inverse_ind2, 
                        dim=0, 
                        dim_size=unique_coords2.shape[0]
                    )
                    feats_unique2 = feats_unique2 / counts2[:, None]
                    selected_ind2 = scatter_min(torch.arange(inverse_ind2.shape[0], device=inverse_ind2.device), inverse_ind2)[1]

                    # selected_ind = scatter_min(torch.arange(inverse_ind2.shape[0], device=inverse_ind2.device), inverse_ind2)[1]
                    unique_grid_coords = merge_coords[selected_ind2,:]
                    feats_unique = feats_unique2
                    imgs_unique = merge_imgs[selected_ind2,:]
                    positions = merge_positions[selected_ind2,:]
                    shape_after_remove = selected_ind2.shape[0]
                elif self.cube_merge_type == 'max_conf':
                    _, selected_ind = scatter_max(
                        conf[batch_idx,:,0],
                        inverse_ind,
                        dim=0,
                        dim_size=unique_coords.shape[0]
                        )
                    unique_grid_coords = grid_coords[batch_idx,selected_ind,:]
                    feats_unique = feats[batch_idx, selected_ind, :]
                    imgs_unique = imgs[batch_idx, selected_ind, :]
                    positions = xyz_scaled[batch_idx, selected_ind, :]

                    conf = conf[batch_idx, selected_ind, :]
                    shape_before_remove = selected_ind.shape[0]
                    merge_coords = torch.cat([unique_grid_coords, xyz_world_rest_coords[batch_idx,:,:]],dim=0)  # r xyz
                    merge_feats = torch.cat([feats_unique, feats_rest[batch_idx,:,:]],dim=0)  # r l
                    merge_imgs = torch.cat([imgs_unique, imgs_rest[batch_idx,:,:]],dim=0)  # r c
                    merge_positions = torch.cat([positions, xyz_world_rest_scaled[batch_idx,:,:]],dim=0)  # r xyz
                    merge_conf = torch.cat([conf, conf_rest[batch_idx,:,:]],dim=0)  # r l
                    unique_coords, inverse_ind2 = torch.unique(
                        merge_coords, 
                        dim=0, 
                        return_inverse=True, 
                        return_counts=False, 
                        sorted=False)
                    _, selected_ind = scatter_max(
                        merge_conf[:,0],
                        inverse_ind2,
                        dim=0,
                        dim_size=unique_coords.shape[0]
                        )
                    # selected_ind = scatter_min(torch.arange(inverse_ind2.shape[0], device=inverse_ind2.device), inverse_ind2)[1]
                    unique_grid_coords = merge_coords[selected_ind,:]
                    feats_unique = merge_feats[selected_ind,:]
                    imgs_unique = merge_imgs[selected_ind,:]
                    positions = merge_positions[selected_ind,:]
                    shape_after_remove = selected_ind.shape[0]

                # concatenate the batch index with coords
                coords_centers = torch.cat([torch.full((unique_grid_coords.shape[0], 1), batch_idx).to(device), unique_grid_coords], dim=-1)  # Nx4
                positions = torch.cat([torch.full((positions.shape[0], 1), batch_idx).to(device), positions, imgs_unique], dim=-1)  # Nx7
                batch_coords_centers.append(coords_centers)
                batch_positions.append(positions)    
                batch_feats.append(feats_unique)
            print(f'shape before removing overlaps: {shape_before_remove}, shape after removing overlaps: {shape_after_remove}')
            # stack the coords
            batch_coords_centers = torch.cat(batch_coords_centers, dim=0)  # [N,4]
            batch_positions = torch.cat(batch_positions, dim=0)  # [N, 4]     
            batch_feats = torch.cat(batch_feats, dim=0)  # [N, C]
            
            gs_cube_tensor = GSCubeInput(
                feats=batch_feats, 
                coords_centers=batch_coords_centers, 
                positions=batch_positions, 
                imgs=None,
                cell_sizes=cell_sizes,
                xyz_min=xyz_min,
                xyz_max=xyz_max,
                device=device
            )
            return gs_cube_tensor, None

        else:
            # if v <= 2, we can use the plain voxelization
            return self.voxelize(xyz_world, feats, imgs, num_depth, b=b, v=v, h=h, w=w, return_perview=return_perview)

if __name__ == "__main__":
    import MinkowskiEngine as ME
    import numpy as np
    from tqdm import tqdm
    import os
    np.random.seed(42)
    torch.manual_seed(42)
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    # Example usage
    encoder = GSCubeEncoder(
        depths=[2, 2, 2, 2, 2],
        channels=[16, 16, 32, 64, 64],
        num_heads=[2, 2, 4, 8, 8],
        window_sizes=[5, 7, 7, 7, 7],
        quant_size=4,
        in_channels = 9,
        first_down_stride=3,
        knn_down=True,
        upsample= 'linear_attn',
        cRSE='XYZ_RGB',
        up_k= 3,
        num_classes=13,
        stem_transformer=True,
        fp16_mode=0,
    )
    encoder.cuda()

    batch_size = 2
    num_points = 256*256  # Non-zero points per batch
    in_channels = 9    # Default for sp features
    embedding_dims = 6 # e.g., XYZ(3) + RGB(3) for coords_sp

    # 1. Generate batch indices and random coordinates
    xyz_coords = torch.randint(0, 128, (2*batch_size * num_points, 3))  # Random voxel positions (0-99)
    xyz_coords = torch.unique(xyz_coords, dim=0)  # Ensure unique coordinates
    xyz_coords = xyz_coords[:batch_size * num_points]  # Limit to batch size
    batch_indices = torch.cat([torch.full((num_points, 1), i) for i in range(batch_size)])
    
    coords = torch.cat([batch_indices, xyz_coords], dim=1)  # Shape: [N, 4]

    # 2. Generate features with torch.randn
    sp_features = 0.5*torch.randn(batch_size * num_points, in_channels)  # For sp.F
    coords_sp_features = torch.cat([coords,-0.7*torch.ones(batch_size * num_points, 3)], dim=-1)  # For coords_sp.F
    # 3. Convert to Minkowski SparseTensors
    sp = ME.SparseTensor(
        features=sp_features,
        coordinates=coords,
        device="cuda"  # Optional GPU
    )

    coords_sp = ME.SparseTensor(
        features=coords_sp_features,
        coordinate_map_key=sp.coordinate_map_key, 
        coordinate_manager=sp.coordinate_manager,
        device="cuda"
    )

    # Run forward pass
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    encoder.train()
    for _ in tqdm(range(1000)):
        optimizer.zero_grad()
        sp_f, output = encoder(sp, coords_sp)
        # print('forward output shape:', output.shape)
        loss = nn.MSELoss()(output, torch.randn_like(output).cuda())  # Dummy loss
        loss.backward()
        if _ % 10 == 0:
            print(f"Step {_}, Loss: {loss.item()}")
    