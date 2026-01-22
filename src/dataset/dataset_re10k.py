import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional, Union

import os
import numpy as np
import h5py
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .shims.crop_shim_forvggt import apply_crop_shim_forvggt
from .types import Stage
from .view_sampler import ViewSampler

DEBUG=False

@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k", "re10k_unbounded"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    train_times_per_scene: int = 1
    highres: bool = False
    use_index_to_load_chunk: Optional[bool] = False


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        vggt_meta: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.vggt_meta = vggt_meta
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        self.chunks = []
        if self.vggt_meta:
            self.vggt_chunk_path = Path("/data2/xxy/data/re10k_vggt_meta") / self.data_stage
        for i, root in enumerate(cfg.roots):
            root = root / self.data_stage
            if self.cfg.use_index_to_load_chunk:
                with open(root / "index.json", "r") as f:
                    json_dict = json.load(f)
                root_chunks = sorted(list(set(json_dict.values())))
            else:
                root_chunks = sorted(
                    [path for path in root.iterdir() if path.suffix == ".torch"]
                )

            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            # self.chunks = [chunk_path] * len(self.chunks)
            self.chunks = [chunk_path] * 10000
            
        if self.stage == "test":
            # testing on a subset for fast speed
            self.chunks = self.chunks[::cfg.test_chunk_interval]
        
        # if DEBUG:
        #     self._test_chunk()

    def shuffle(self, lst: list, return_indices:bool = False) -> Union[list, tuple[list, Tensor]]:
        indices = torch.randperm(len(lst))
        if return_indices:
            return [lst[x] for x in indices], indices
        return [lst[x] for x in indices]

    def _convert_vggt_meta(self, scene_name, scene):
        depth = scene['depth'][:]
        depth_max = scene['depth_max'][:]
        depth_min = scene['depth_min'][:]
        depth_conf = scene['depth_conf'][:]
        extrinsics_vggt = scene['extrinsic'][:]
        intrinsics_vggt = scene['intrinsic'][:]
        img_sizes_vggt = scene['img_sizes'][:]

        depth = depth/255.0 * (depth_max[:,None,None,None] - depth_min[:,None,None,None]) + depth_min[:,None,None,None]
        depth = torch.tensor(depth, dtype=torch.float32)
        depth_conf = torch.tensor(depth_conf, dtype=torch.float32)
        extrinsics_vggt = torch.tensor(extrinsics_vggt, dtype=torch.float32) # inverse to follow re10k
        extrinsics_vggt = torch.cat([extrinsics_vggt, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).repeat(extrinsics_vggt.shape[0], 1, 1)], dim=1).inverse()
        intrinsics_vggt = torch.tensor(intrinsics_vggt, dtype=torch.float32)
        img_sizes_vggt = torch.tensor(img_sizes_vggt, dtype=torch.int32)
        return {
            "key":scene_name,
            "depth": depth[...,0],
            "depth_max": depth_max,
            "depth_min": depth_min,
            "depth_conf": depth_conf,
            "extrinsic": extrinsics_vggt,
            "intrinsic": intrinsics_vggt,
            "img_sizes": img_sizes_vggt,
        }

    def _test_chunk(self):
        from PIL import Image
        import numpy as np
        import matplotlib.pyplot as plt
        from ..geometry.vggt_geometry import unproject_depth_map_to_point_map

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

        chunk = torch.load(self.chunks[0])
        chunk = [item for item in chunk if item["key"] == self.cfg.overfit_to_scene][0]
        with h5py.File(self.vggt_chunk_path / self.chunks[0].name.replace('.torch', '.h5'), 'r') as f:
            for scene_name in f.keys():
                if scene_name == self.cfg.overfit_to_scene:
                    chunk_vggt = self._convert_vggt_meta(scene_name, f[scene_name])
        img_id = [0,45]
        images = [chunk["images"][i] for i in img_id]
        images = self.convert_images(images, resize_shape=chunk_vggt["img_sizes"][img_id])
        extrinsic = chunk_vggt["extrinsic"][img_id].inverse()
        intrinsic = chunk_vggt["intrinsic"][img_id]
        depth = chunk_vggt["depth"][img_id]
        indices = [torch.arange(length) for length in depth.shape[-2:]]
        coordinates = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1).to(depth)
        v = depth.shape[0]
        # depth = rearrange(depth, "b v h w -> (b v) h w")
        # extrinsic = rearrange(extrinsic, "b v c d -> (b v) c d")
        # intrinsic = rearrange(intrinsic, "b v c d -> (b v) c d")
        coordinates = repeat(coordinates, "h w c-> n h w c", n=v)
        images = rearrange(images, "v c h w -> v h w c")
        points = unproject_depth_map_to_point_map(
            coordinates=coordinates,
            depth_map=depth,
            extrinsics_cam=extrinsic,
            intrinsics_cam=intrinsic
        )
        points = points.reshape(-1,3).cpu()
        # points = means2.reshape(-1, 3)
        colors = images.reshape(-1,3).cpu()
        opacities = torch.zeros_like(colors).cpu()  # Opacity is not used in this case
        render_pcs(points, colors, opacities, 'before_data_shim')

        points_homo = np.concatenate([points.numpy(), np.ones((points.shape[0], 1))], axis=1)
        points_cam = np.dot(points_homo, chunk_vggt["extrinsic"][6].inverse().cpu().numpy().transpose(1,0))
        points_pixel = np.dot(points_cam[:,:3], chunk_vggt["intrinsic"][6].cpu().numpy().transpose(1,0))
        points_pixel = points_pixel[:,:2] / points_pixel[:,2:3]
        points_pixel = points_pixel[:,:2].astype(np.uint16)
        h, w = 294, 518
        new_img = np.zeros(shape=(h,w,3))
        for i in range(points_pixel.shape[0]):
            x, y = points_pixel[i]
            x = min(x, w - 1)
            y = min(y, h - 1)
            new_img[y,x] = colors[i]
        img = Image.fromarray((new_img * 255).astype(np.uint8))
        img.save('reproject_data_shim.png')
        raise Exception("debug data shim")

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path)
            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)
            if self.vggt_meta:
                vggt_meta_chunk_path = self.vggt_chunk_path / chunk_path.name.replace('.torch', '.h5')
                if not os.path.exists(vggt_meta_chunk_path):
                    print(f'{vggt_meta_chunk_path} is not ready')
                    continue
                chunk_vggt = []
                new_chunk=[]
                with h5py.File(vggt_meta_chunk_path, 'r') as f:
                    for c in chunk:
                        scene_name = c["key"]
                        if scene_name in f.keys():
                            scene = f[scene_name]
                            scene = self._convert_vggt_meta(scene_name, scene)
                            chunk_vggt.append(scene)
                            new_chunk.append(c)

                chunk = new_chunk
                if self.cfg.overfit_to_scene is not None:
                    item_vggt = [x for x in chunk_vggt if x["key"] == self.cfg.overfit_to_scene]
                    assert len(item_vggt) == 1
                    chunk_vggt = item_vggt * len(chunk_vggt)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                if self.vggt_meta:
                    chunk, indices = self.shuffle(chunk, return_indices=True)
                    chunk_vggt = [chunk_vggt[i] for i in indices]
                else:
                    chunk = self.shuffle(chunk)
                

            times_per_scene = (
                1
                if self.stage == "test"
                else self.cfg.train_times_per_scene
            )

            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                # import numpy as np
                # d = np.linalg.norm(np.diff(extrinsics[:,:3,3], axis=0), axis=1)
                # b = np.median(d)
                scene = example["key"]


                if self.vggt_meta:
                    example_vggt = chunk_vggt[run_idx // times_per_scene]
                    extrinsics_vggt = example_vggt["extrinsic"]
                    intrinsics_vggt = example_vggt["intrinsic"]
                    depth = example_vggt["depth"]
                    depth_conf = example_vggt["depth_conf"]
                    img_sizes = example_vggt["img_sizes"]
                    K_norm = intrinsics_vggt[:, :3, :3].clone()
                    K_norm[:,0, :] /= img_sizes[:, 1, None]
                    K_norm[:,1, :] /= img_sizes[:, 0, None]
                    vggt_fov_deg = get_fov(K_norm).rad2deg()
                    del K_norm


                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue
                    
                # debug
                # context_indices = torch.tensor([22,31, 42,51, 65,76,97, 103, 117], dtype=torch.long,device=context_indices.device)

                # Skip the example if the field of view is too wide.
                if self.vggt_meta:
                    if (vggt_fov_deg > self.cfg.max_fov).any():
                        continue
                else:
                    if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                        continue
                

                # Load the images.
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                if self.vggt_meta:
                    context_images = self.convert_images(context_images, resize_shape=img_sizes[context_indices])
                else:
                    context_images = self.convert_images(context_images)
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                if self.vggt_meta:
                    target_images = self.convert_images(target_images, resize_shape=img_sizes[target_indices])
                else:
                    target_images = self.convert_images(target_images)


                # Skip the example if the images don't have the right shape.
                if self.vggt_meta:
                    if self.cfg.highres:
                        expected_shape = (3, 588, 1036)
                    else:
                        expected_shape = (3, 294, 518)
                else:
                    if self.cfg.highres:
                        expected_shape = (3, 720, 1280)
                    else:
                        expected_shape = (3, 360, 640)
                context_image_invalid = context_images.shape[1:] != expected_shape
                target_image_invalid = target_images.shape[1:] != expected_shape
                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}.",
                        flush=True
                    )
                    continue

                nf_scale = 1.0
                if self.vggt_meta:
                    example = {
                        "context": {
                            "extrinsics": extrinsics_vggt[context_indices],
                            "intrinsics": intrinsics_vggt[context_indices],
                            "depth": depth[context_indices],
                            "depth_conf": depth_conf[context_indices],
                            "image": context_images,
                            "near": self.get_bound("near", len(context_indices)) / nf_scale,
                            "far": self.get_bound("far", len(context_indices)) / nf_scale,
                            "index": context_indices,
                            "vggt_img_sizes": img_sizes[context_indices],
                        },
                        "target": {
                            "extrinsics": extrinsics_vggt[target_indices],
                            "intrinsics": intrinsics_vggt[target_indices],
                            "depth": depth[target_indices],
                            "depth_conf": depth_conf[target_indices],
                            "image": target_images,
                            "near": self.get_bound("near", len(target_indices)) / nf_scale,
                            "far": self.get_bound("far", len(target_indices)) / nf_scale,
                            "index": target_indices,
                            "vggt_img_sizes": img_sizes[target_indices],
                        },
                        "scene": scene,
                    }
                else:
                    example = {
                        "context": {
                            "extrinsics": extrinsics[context_indices],
                            "intrinsics": intrinsics[context_indices],
                            "image": context_images,
                            "near": self.get_bound("near", len(context_indices)) / nf_scale,
                            "far": self.get_bound("far", len(context_indices)) / nf_scale,
                            "index": context_indices,
                        },
                        "target": {
                            "extrinsics": extrinsics[target_indices],
                            "intrinsics": intrinsics[target_indices],
                            "image": target_images,
                            "near": self.get_bound("near", len(target_indices)) / nf_scale,
                            "far": self.get_bound("far", len(target_indices)) / nf_scale,
                            "index": target_indices,
                        },
                        "scene": scene,
                        
                    }

                if DEBUG:
                    from PIL import Image
                    import numpy as np
                    import matplotlib.pyplot as plt
                    from ..geometry.vggt_geometry import unproject_depth_map_to_point_map

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

                    images = example['context']['image']
                    depth = example['context']['depth']
                    extrinsic = example['context']['extrinsics'].inverse()
                    intrinsic = example['context']['intrinsics']
                    indices = [torch.arange(length) for length in depth.shape[-2:]]
                    coordinates = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1).to(depth)
                    v = depth.shape[0]
                    coordinates = repeat(coordinates, "h w c-> n h w c", n=v)
                    images = rearrange(images, "v c h w -> v h w c")
                    points = unproject_depth_map_to_point_map(
                        coordinates=coordinates,
                        depth_map=depth,
                        extrinsics_cam=extrinsic,
                        intrinsics_cam=intrinsic
                    )
                    points = points.reshape(-1,3).cpu()
                    # points = means2.reshape(-1, 3)
                    colors = images.reshape(-1,3).cpu()
                    opacities = torch.zeros_like(colors).cpu()  # Opacity is not used in this case
                    render_pcs(points, colors, opacities, 'on_data_shim')

                    points_homo = np.concatenate([points.numpy(), np.ones((points.shape[0], 1))], axis=1)
                    points_cam = np.dot(points_homo, example['target']['extrinsics'][0].inverse().cpu().numpy().transpose(1,0))
                    points_pixel = np.dot(points_cam[:,:3], example['target']['intrinsics'][0].cpu().numpy().transpose(1,0))
                    points_pixel = points_pixel[:,:2] / points_pixel[:,2:3]
                    points_pixel = points_pixel[:,:2].astype(np.uint16)
                    h, w = 256,256
                    new_img = np.zeros(shape=(h,w,3))
                    for i in range(points_pixel.shape[0]):
                        x, y = points_pixel[i]
                        x = min(x, w - 1)
                        y = min(y, h - 1)
                        new_img[y,x] = colors[i]
                    img = Image.fromarray((new_img * 255).astype(np.uint8))
                    img.save('reproject_on_data_shim.png')
                    raise 'debug on data_shim'
                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                if self.vggt_meta:            
                    yield apply_crop_shim_forvggt(example, tuple(self.cfg.image_shape))
                else:
                    yield apply_crop_shim(example, tuple(self.cfg.image_shape))
    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
        resize_shape:Tensor = None,
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        new_shape = resize_shape.cpu().numpy().tolist() if resize_shape is not None else None
        for n,image in enumerate(images):
            image = Image.open(BytesIO(image.numpy().tobytes()))
            if new_shape is not None:
                image = image.resize(new_shape[n][::-1], Image.Resampling.BICUBIC)
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
    
    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
            # return "train"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for i, root in enumerate(self.cfg.roots):
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        if self.cfg.overfit_to_scene is not None:
            if self.stage == "test":
                return 1
            return 10000
        return (
            min(len(self.index.keys()), self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.train_times_per_scene
        )
