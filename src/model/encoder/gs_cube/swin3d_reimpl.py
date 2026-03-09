import math

import MinkowskiEngine as ME
import torch
import torch.nn as nn
from MinkowskiEngine import SparseTensor
from timm.models.layers import DropPath, trunc_normal_

from Swin3D.modules.mink_layers import (
    MinkConvBNRelu,
    MinkResBlock,
    SparseTensorLayerNorm,
    SparseTensorLinear,
    assign_feats,
)
from Swin3D.sparse_dl.attn.attn_coff import (
    IndexMode,
    PosEmb,
    PrecisionMode,
    SelfAttnAIOFunction,
    TableDims,
)
from Swin3D.sparse_dl.knn import KNN


def get_offset(batch: torch.Tensor) -> torch.Tensor:
    batch = batch.long()
    batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    counts = torch.bincount(batch, minlength=batch_size)
    offset = torch.zeros(batch_size + 1, dtype=torch.int32, device=batch.device)
    if counts.numel() > 0:
        offset[1:] = counts.cumsum(dim=0).to(torch.int32)
    return offset


def query_knn_feature(
    k: int,
    src_xyz: torch.Tensor,
    query_xyz: torch.Tensor,
    src_feat: torch.Tensor,
    src_offset: torch.Tensor,
    query_offset: torch.Tensor,
    return_idx: bool = False,
):
    assert src_xyz.is_contiguous() and src_feat.is_contiguous()
    if query_xyz is None:
        query_xyz = src_xyz
        query_offset = src_offset
    assert query_xyz.is_contiguous()

    idx, _ = KNN.apply(k, src_xyz, query_xyz, src_offset, query_offset)
    grouped_feat = src_feat[idx.reshape(-1).long(), :].view(query_xyz.shape[0], k, src_feat.shape[1])
    if return_idx:
        return grouped_feat, idx
    return grouped_feat


def knn_linear_interpolation(
    src_xyz: torch.Tensor,
    query_xyz: torch.Tensor,
    src_feat: torch.Tensor,
    src_offset: torch.Tensor,
    query_offset: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    assert src_xyz.is_contiguous() and query_xyz.is_contiguous() and src_feat.is_contiguous()
    idx, dist = KNN.apply(k, src_xyz, query_xyz, src_offset, query_offset)
    weight = 1.0 / (dist + 1e-8)
    weight = weight / weight.sum(dim=1, keepdim=True)
    gathered = src_feat[idx.long()]
    return (gathered * weight.unsqueeze(-1)).sum(dim=1)


def sparse_self_attention(w_w_id: torch.Tensor, w_sizes: torch.Tensor, protocol: str = "v1"):
    w_sizes_sq = w_sizes.square()
    w_cumsum = torch.cumsum(w_sizes, dim=-1)
    w2n_indices = torch.cat([torch.zeros(1, dtype=w_sizes.dtype, device=w_sizes.device), w_cumsum[:-1]])
    w2_cumsum = torch.cumsum(w_sizes_sq, dim=-1)
    w2m_indices = torch.cat([torch.zeros(1, dtype=w_sizes.dtype, device=w_sizes.device), w2_cumsum[:-1]])

    total_m = int(w2_cumsum[-1].item()) if w2_cumsum.numel() > 0 else 0
    m2w_indices = torch.zeros(total_m, dtype=w_sizes.dtype, device=w_sizes.device)
    m2w_offset = torch.zeros(total_m, dtype=w_sizes.dtype, device=w_sizes.device)
    if w2m_indices.numel() > 1:
        m2w_indices[w2m_indices[1:]] = 1
        m2w_offset[w2m_indices[1:]] = w_sizes_sq[:-1]
    m2w_indices = torch.cumsum(m2w_indices, dim=-1)
    m2w_offset = torch.cumsum(m2w_offset, dim=-1)

    m_indices = torch.arange(total_m, dtype=w_sizes.dtype, device=w_sizes.device)
    m2n_indices = w2n_indices[m2w_indices]
    m_offset = m_indices - m2w_offset
    m2w_sizes = w_sizes[m2w_indices]
    y_offset = m2n_indices + (m_offset % m2w_sizes)
    x_offset = m2n_indices + torch.div(m_offset, m2w_sizes, rounding_mode="floor")

    if protocol == "v1":
        return x_offset, y_offset
    if protocol == "v2":
        return x_offset, y_offset, m2w_indices, w_sizes, w2n_indices, w2m_indices
    raise ValueError(f"Unsupported protocol: {protocol}")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GridCoordsDown(nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        self.avg_pool = ME.MinkowskiAvgPooling(kernel_size=stride, stride=stride, dimension=3)

    def forward(self, coords_sp, sp, return_map=False):
        avg_coords_sp = self.avg_pool(coords_sp)
        if return_map:
            return avg_coords_sp, None
        return avg_coords_sp


class GridDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sp_pool = ME.MinkowskiMaxPooling(kernel_size=kernel_size, stride=stride, dimension=3)
        self.coords_pool = GridCoordsDown(stride=stride)
        self.norm = SparseTensorLayerNorm(in_channels)
        self.linear = SparseTensorLinear(in_channels, out_channels)

    def forward(self, sp, coords_sp, return_map=False):
        sp_down = self.sp_pool(self.linear(self.norm(sp)))
        if return_map:
            coords_sp_down, downsample_map = self.coords_pool(coords_sp, sp_down, return_map=True)
            return sp_down, coords_sp_down, downsample_map
        return sp_down, self.coords_pool(coords_sp, sp_down)


class GridKNNDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = 16
        self.sp_pool = ME.MinkowskiMaxPooling(kernel_size=stride, stride=stride, dimension=3)
        self.coords_pool = GridCoordsDown(stride=stride)
        self.norm = nn.LayerNorm(in_channels)
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.pool = nn.MaxPool1d(self.k)

    def forward(self, sp, coords_sp):
        sp_down = self.sp_pool(sp)
        coords_sp_down = self.coords_pool(coords_sp, sp_down)
        offset = get_offset(sp.C[:, 0])
        n_offset = get_offset(sp_down.C[:, 0])
        xyz = coords_sp.F[:, 1:4].detach().contiguous()
        n_xyz = coords_sp_down.F[:, 1:4].detach().contiguous()
        feats = query_knn_feature(self.k, xyz, n_xyz, sp.F, offset, n_offset)
        m, k, c = feats.shape
        feats = self.norm(feats.reshape(m * k, c)).reshape(m, k, c)
        feats = self.linear(feats).transpose(1, 2).contiguous()
        feats = self.pool(feats).squeeze(-1)
        return assign_feats(sp_down, feats.float()), coords_sp_down


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        quant_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        cRSE="XYZ_RGB",
        fp16_mode=0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.color_windowsize = 2
        self.normal_windowsize = 2
        self.fp16_mode = fp16_mode
        self.cRSE = cRSE
        self.quant_size = quant_size
        head_dim = dim // num_heads

        table_offsets = []
        if "XYZ" in cRSE:
            self.xyz_quant_size = quant_size
            xyz_len = window_size * self.xyz_quant_size
            xyz_shape = (3, 2 * xyz_len, num_heads, head_dim)
            self.query_xyz_table = nn.Parameter(torch.zeros(xyz_shape))
            self.key_xyz_table = nn.Parameter(torch.zeros(xyz_shape))
            self.value_xyz_table = nn.Parameter(torch.zeros(xyz_shape))
            trunc_normal_(self.query_xyz_table, std=0.02)
            trunc_normal_(self.key_xyz_table, std=0.02)
            trunc_normal_(self.value_xyz_table, std=0.02)
            table_offsets.extend([math.prod(xyz_shape[1:])] * 3)
        if "RGB" in cRSE:
            self.color_quant_size = quant_size * 2
            rgb_len = self.color_windowsize * self.color_quant_size
            rgb_shape = (3, 2 * rgb_len, num_heads, head_dim)
            self.query_rgb_table = nn.Parameter(torch.zeros(rgb_shape))
            self.key_rgb_table = nn.Parameter(torch.zeros(rgb_shape))
            self.value_rgb_table = nn.Parameter(torch.zeros(rgb_shape))
            trunc_normal_(self.query_rgb_table, std=0.02)
            trunc_normal_(self.key_rgb_table, std=0.02)
            trunc_normal_(self.value_rgb_table, std=0.02)
            table_offsets.extend([math.prod(rgb_shape[1:])] * 3)
        if "NORM" in cRSE:
            self.normal_quant_size = quant_size * 2
            norm_len = self.normal_windowsize * self.normal_quant_size
            norm_shape = (3, 2 * norm_len, num_heads, head_dim)
            self.query_norm_table = nn.Parameter(torch.zeros(norm_shape))
            self.key_norm_table = nn.Parameter(torch.zeros(norm_shape))
            self.value_norm_table = nn.Parameter(torch.zeros(norm_shape))
            trunc_normal_(self.query_norm_table, std=0.02)
            trunc_normal_(self.key_norm_table, std=0.02)
            trunc_normal_(self.value_norm_table, std=0.02)
            table_offsets.extend([math.prod(norm_shape[1:])] * 3)
        self.register_buffer("table_offsets_buffer", torch.tensor(table_offsets, dtype=torch.int32), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def _collect_tables(self, n_coords: torch.Tensor):
        query_tables = []
        key_tables = []
        value_tables = []
        coord_parts = []
        if "XYZ" in self.cRSE:
            coord_parts.append(n_coords[:, 0:3] * self.quant_size)
            query_tables.append(self.query_xyz_table.reshape(-1))
            key_tables.append(self.key_xyz_table.reshape(-1))
            value_tables.append(self.value_xyz_table.reshape(-1))
        if "RGB" in self.cRSE:
            coord_parts.append(n_coords[:, 3:6] * self.color_quant_size)
            query_tables.append(self.query_rgb_table.reshape(-1))
            key_tables.append(self.key_rgb_table.reshape(-1))
            value_tables.append(self.value_rgb_table.reshape(-1))
        if "NORM" in self.cRSE:
            coord_parts.append(n_coords[:, 6:9] * self.normal_quant_size)
            query_tables.append(self.query_norm_table.reshape(-1))
            key_tables.append(self.key_norm_table.reshape(-1))
            value_tables.append(self.value_norm_table.reshape(-1))
        return (
            torch.cat(coord_parts, dim=1),
            torch.cat(query_tables),
            torch.cat(key_tables),
            torch.cat(value_tables),
        )

    def forward(self, feats: torch.Tensor, attn_args):
        num_voxels = feats.shape[0]
        num_sc = self.dim // self.num_heads
        x_offset, y_offset, m2w_indices, w_sizes, w2n_indices, n2n_indices, w2m_indices, n_coords = attn_args
        qkv = self.qkv(feats).reshape(num_voxels, 3, self.num_heads, num_sc).permute(1, 0, 2, 3).contiguous()
        query, key, value = qkv[0] * self.scale, qkv[1], qkv[2]
        n_cRSE, query_table, key_table, value_table = self._collect_tables(n_coords)
        indices = [m2w_indices, w_sizes, w2m_indices, w2n_indices, n2n_indices, n_cRSE]

        if self.fp16_mode == 0:
            precision = PrecisionMode.HALF_NONE
        elif self.fp16_mode == 1:
            precision = PrecisionMode.HALF_FORWARD
        else:
            precision = PrecisionMode.HALF_ALL

        updated_values = SelfAttnAIOFunction.apply(
            query,
            key,
            value,
            query_table,
            key_table,
            value_table,
            self.table_offsets_buffer.to(device=feats.device),
            indices,
            PosEmb.SEPARATE,
            TableDims.D0,
            IndexMode.INDIRECT,
            precision,
        )
        updated_feats = updated_values.flatten(1).reshape(num_voxels, self.dim)
        updated_feats = self.proj(updated_feats)
        return self.proj_drop(updated_feats)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        quant_size,
        drop_path=0.0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        cRSE="XYZ_RGB",
        fp16_mode=0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=window_size,
            quant_size=quant_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            cRSE=cRSE,
            fp16_mode=fp16_mode,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, feats, attn_args):
        attn_out = self.attn(self.norm1(feats), attn_args)
        feats = feats + self.drop_path(attn_out)
        feats = feats + self.drop_path(self.mlp(self.norm2(feats)))
        return feats


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        quant_size,
        out_channels=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        down_stride=2,
        cRSE="XYZ_RGB",
        fp16_mode=0,
    ):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.quant_size = quant_size
        self.cRSE = cRSE
        self.fp16_mode = fp16_mode
        self.shift_size = window_size // 2
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim,
                num_heads,
                window_size,
                quant_size,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                cRSE=cRSE,
                fp16_mode=fp16_mode,
            )
            for i in range(depth)
        ])
        self.pool = ME.MinkowskiMaxPooling(kernel_size=window_size, stride=window_size, dimension=3)
        self._local_window_cache = {}
        if downsample is not None:
            if out_channels is None:
                out_channels = dim * 2
            self.downsample = downsample(dim, out_channels, kernel_size=down_stride, stride=down_stride)
        else:
            self.downsample = None

    def _get_local_window(self, device, stride_value):
        key = (device.type, device.index, stride_value)
        cached = self._local_window_cache.get(key)
        if cached is None:
            axis = torch.arange(self.window_size, device=device) * int(stride_value)
            x, y, z = torch.meshgrid(axis, axis, axis)
            i = torch.zeros_like(x, device=device)
            cached = torch.stack([i, x, y, z], dim=-1).flatten(0, -2)
            self._local_window_cache[key] = cached
        return cached

    def get_map_pair(self, sp):
        pool_sp = self.pool(sp)
        windows = pool_sp.C
        window_count = windows.shape[0]
        stride_in = sp.coordinate_map_key.get_tensor_stride()
        local_window = self._get_local_window(sp.C.device, stride_in[0])
        all_windows = (windows.unsqueeze(1) + local_window.unsqueeze(0)).flatten(0, -2).int()
        coordinate_manager = sp.coordinate_manager
        query_key, _ = coordinate_manager.insert_and_map(all_windows, tensor_stride=stride_in)
        map_pair = coordinate_manager.kernel_map(query_key, sp.coordinate_map_key, kernel_size=1)[0]
        return map_pair, window_count

    def get_window_mapping(self, sp):
        map_pair, window_count = self.get_map_pair(sp)
        in_map, out_map = map_pair
        in_map, sort_idx = torch.sort(in_map)
        out_map = out_map[sort_idx]
        sort_idx = out_map.long()
        window_volume = self.window_size ** 3
        inv_sort_idx = torch.zeros_like(sort_idx)
        inv_sort_idx[sort_idx] = torch.arange(sort_idx.shape[0], dtype=sort_idx.dtype, device=sort_idx.device)

        total_slots = window_count * window_volume
        v2w_mask = torch.zeros(total_slots, dtype=torch.bool, device=sp.C.device)
        w_w_id = torch.arange(window_volume, dtype=torch.long, device=sp.C.device).unsqueeze(0).repeat(window_count, 1).view(-1)
        v2w_mask[in_map.long()] = True
        non_empty = v2w_mask.view(-1, window_volume).sum(dim=-1)
        w_w_id = w_w_id[in_map.long()]
        local_xyz = torch.stack(
            [
                w_w_id // self.window_size // self.window_size,
                w_w_id // self.window_size % self.window_size,
                w_w_id % self.window_size,
            ],
            dim=-1,
        )
        return w_w_id, local_xyz, non_empty, sort_idx, inv_sort_idx

    def get_index01(self, sp, local_xyz, colors):
        w_w_id, w_w_xyz, non_empty, n2n_indices, inv_sort_idx = self.get_window_mapping(sp)
        local_xyz = local_xyz[n2n_indices]
        colors = colors[n2n_indices]
        n_coords = torch.cat([w_w_xyz + local_xyz, colors], dim=1)
        x_offset, y_offset, m2w_indices, w_sizes, w2n_indices, w2m_indices = sparse_self_attention(w_w_id, non_empty, protocol="v2")
        return x_offset, y_offset, m2w_indices, w_sizes, w2n_indices, n2n_indices, w2m_indices, n_coords

    def get_shifted_sp(self, sp):
        stride_in = sp.coordinate_map_key.get_tensor_stride()
        shift_size = self.shift_size * stride_in[0]
        shifted_coords = sp.C.clone()
        shifted_coords[:, 1:] += shift_size
        return SparseTensor(features=sp.F, coordinates=shifted_coords, device=sp.device, tensor_stride=stride_in)

    def forward(self, sp, coords_sp, return_map=False):
        colors = coords_sp.F[:, 4:]
        xyz = coords_sp.F[:, :4]
        stride_value = coords_sp.coordinate_map_key.get_tensor_stride()[0]
        local_xyz = (xyz - coords_sp.C)[:, 1:] / stride_value
        attn_args = self.get_index01(sp, local_xyz, colors)
        attn_args_shift = None
        if self.depth > 1:
            shifted_sp = self.get_shifted_sp(sp)
            attn_args_shift = self.get_index01(shifted_sp, local_xyz, colors)
        feats = sp.F
        for block_index, block in enumerate(self.blocks):
            feats = block(feats, attn_args if block_index % 2 == 0 else attn_args_shift)
        sp = assign_feats(sp, feats)
        if self.downsample is None:
            return sp, sp, coords_sp
        if return_map:
            sp_down, coords_sp_down, downsample_map = self.downsample(sp, coords_sp, return_map=True)
            return sp, sp_down, coords_sp_down, downsample_map
        sp_down, coords_sp_down = self.downsample(sp, coords_sp)
        return sp, sp_down, coords_sp_down


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        window_size,
        quant_size,
        attn=True,
        up_k=3,
        cRSE="XYZ_RGB",
        fp16_mode=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear1 = nn.Sequential(nn.LayerNorm(out_channels), nn.Linear(out_channels, out_channels))
        self.linear2 = nn.Sequential(nn.LayerNorm(in_channels), nn.Linear(in_channels, out_channels))
        self.up_k = up_k
        self.attn = attn and window_size > 0
        if self.attn:
            self.block = BasicLayer(
                dim=out_channels,
                depth=1,
                num_heads=num_heads,
                window_size=window_size,
                quant_size=quant_size,
                drop_path=0.1,
                downsample=None,
                out_channels=None,
                cRSE=cRSE,
                fp16_mode=fp16_mode,
            )

    def forward(self, sp, coords_sp, sp_up, coords_sp_up):
        xyz = coords_sp.F[:, 1:4].detach().contiguous()
        support_xyz = coords_sp_up.F[:, 1:4].detach().contiguous()
        offset = get_offset(sp.C[:, 0])
        support_offset = get_offset(sp_up.C[:, 0])
        interpolated = knn_linear_interpolation(
            xyz,
            support_xyz,
            self.linear2(sp.F),
            offset,
            support_offset,
            k=self.up_k,
        )
        feats = self.linear1(sp_up.F) + interpolated
        sp_up = assign_feats(sp_up, feats)
        if self.attn:
            sp_up, _, _ = self.block(sp_up, coords_sp_up)
        return sp_up


class Swin3DUNet(nn.Module):
    def __init__(
        self,
        depths,
        channels,
        num_heads,
        window_sizes,
        quant_size,
        drop_path_rate=0.2,
        up_k=3,
        num_layers=5,
        num_classes=13,
        stem_transformer=True,
        first_down_stride=2,
        other_down_stride=2,
        upsample="linear",
        knn_down=True,
        in_channels=6,
        cRSE="XYZ_RGB",
        fp16_mode=0,
        stem_norm="bn",
    ):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        downsample_cls = GridKNNDownsample if knn_down else GridDownsample
        self.cRSE = cRSE
        if stem_transformer:
            self.stem_layer = MinkConvBNRelu(
                in_channels=in_channels,
                out_channels=channels[0],
                kernel_size=3,
                stride=1,
                norm=stem_norm,
            )
            self.layer_start = 0
        else:
            self.stem_layer = nn.Sequential(
                MinkConvBNRelu(in_channels=in_channels, out_channels=channels[0], kernel_size=3, stride=1),
                MinkResBlock(in_channels=channels[0], out_channels=channels[0]),
            )
            self.downsample = downsample_cls(channels[0], channels[1], kernel_size=first_down_stride, stride=first_down_stride)
            self.layer_start = 1

        self.layers = nn.ModuleList([
            BasicLayer(
                dim=channels[i],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_sizes[i],
                quant_size=quant_size,
                drop_path=dpr[sum(depths[:i]):sum(depths[: i + 1])],
                downsample=downsample_cls if i < num_layers - 1 else None,
                down_stride=first_down_stride if i == 0 else other_down_stride,
                out_channels=channels[i + 1] if i < num_layers - 1 else None,
                cRSE=cRSE,
                fp16_mode=fp16_mode,
            )
            for i in range(self.layer_start, num_layers)
        ])

        use_attn_upsample = "attn" in upsample
        self.upsamples = nn.ModuleList([
            Upsample(
                channels[i],
                channels[i - 1],
                num_heads[i - 1],
                window_sizes[i - 1],
                quant_size,
                attn=use_attn_upsample,
                up_k=up_k,
                cRSE=cRSE,
                fp16_mode=fp16_mode,
            )
            for i in range(num_layers - 1, 0, -1)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], num_classes),
        )
        self.num_classes = num_classes
        self.init_weights()

    def init_weights(self):
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        self.apply(_init_weights)
