from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat, einsum
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json
import os
import time
from tqdm import tqdm
import torch.nn.functional as F
import math
from PIL import Image as PILImage
import gc
from torch import distributed as dist

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode, DecoderOutput
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from src.visualization.vis_depth import viz_depth_tensor
from PIL import Image
from ..misc.stablize_camera import render_stabilization_path
from .ply_export import save_gaussian_ply, save_gaussian_cube_ply
from .types import Gaussians


DEBUG = False
NOISE=True

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    lr_monodepth: float
    weight_decay: float
    lr_gs_cube: float


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    save_gt_image: bool
    save_input_images: bool
    save_depth: bool
    save_depth_concat_img: bool
    save_depth_npy: bool
    save_gaussian: bool
    render_chunk_size: int | None
    stablize_camera: bool
    stab_camera_kernel: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    eval_model_every_n_val: int
    eval_data_length: int
    eval_deterministic: bool
    eval_time_skip_steps: int
    eval_save_model: bool
    l1_loss: bool
    intermediate_loss_weight: float
    no_viz_video: bool
    viz_depth: bool
    forward_depth_only: bool
    train_ignore_large_loss: float
    no_log_projections: bool


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    eval_data_cfg: Optional[DatasetCfg | None]
    train_controller_cfg: None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        eval_data_cfg: Optional[DatasetCfg | None] = None,
        train_controller_cfg=None,
        iter_depth: bool = False,
        view_base: int = 2,
        batch_forward: bool = False,
        anchor_features: bool = False,
        chunk_num: int = 1,
        anchor_base: int = 4,
        noise_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        # self.automatic_optimization = False
        self.iter_depth = iter_depth
        self.view_base = view_base
        self.batch_forward = batch_forward
        self.chunk_num = chunk_num
        self.anchor_features = anchor_features
        self.anchor_base = anchor_base
        self.noise_ratio = noise_ratio

        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.eval_data_cfg = eval_data_cfg
        self.train_controller_cfg = train_controller_cfg

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # for name, param in self.encoder.named_parameters():
        #     if "depth_predictor" in name:
        #         param.requires_grad = False

        # when training the gscube, the base_model is freezed
        self.gs_cube = train_controller_cfg.gs_cube if train_controller_cfg else False
        self.vggt_meta = train_controller_cfg.vggt_meta if train_controller_cfg else False
        if not train_controller_cfg.base_model:
            for name, param in self.encoder.named_parameters():
                if ("gs_cube_encoder"  not in name) and ("gaussian_merger" not in name) and ("gaussian_scorer" not in name):
                    param.requires_grad = False

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}
            self.nog = []
            self.test_step_output_dict = {}
            self.all_ind = {}


    def _handle_oom(self, where: str):
        # free up as much as we can so the next batch can proceed
        self.log("oom_batch", 1, prog_bar=True, on_step=True, logger=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def training_step(self, batch, batch_idx):
        # batch = torch.load("/data2/xxy/code/depthsplat/offending_batch.pt")
        # forward
        # torch.cuda.empty_cache()
        opt = self.optimizers()
        # ---- FORWARD ----
        fwd_ok = torch.tensor(1, device=self.device, dtype=torch.int)
        try:
            total_loss = self.forward(batch, batch_idx)

            # scheduler = self.lr_schedulers()
            # # ---- BACKWARD ---- debug nan
            # self.manual_backward(total_loss)
            # nan=0
            # for n,p in self.named_parameters():
            #     if torch.isnan(p).any():
            #         print(f'param nan: {n}')
            #         nan = 1
            #     if p.grad is not None and torch.isnan(p.grad).any():
            #         print(f'grad nan: {n}')
            #         nan =1
            # if nan:
            #     raise ValueError('nan in param or grad')
            # self.clip_gradients(opt,0.5,gradient_clip_algorithm='norm')
            # opt.step()
            # opt.zero_grad()
            # scheduler.step()
                
        except torch.cuda.OutOfMemoryError as e:
            self._handle_oom("training_step")
            print(f"Out of memory error in encoder: {e} at global_step: {self.global_step}, batch_idx: {batch_idx}, scene_names: {batch['scene']}")
            fwd_ok.zero_()
        except RuntimeError as e:
            print(f"Runtime error in encoder: {e}")
            print(f'global_step: {self.global_step}, batch_idx: {batch_idx}, scene_names: {batch["scene"]}')
            fwd_ok.zero_()
        except ValueError as e:
            self._handle_oom('value error')
            print(f"Value error in encoder: {e}")
            print(f'global_step: {self.global_step}, batch_idx: {batch_idx}, scene_names: {batch["scene"]}')
            # for h in handles:
            #     h.remove()
            # debug Nan
            # save model and optimizer and scheduler state
            # torch.save(self.state_dict(), 'offending_model_optim_sched.pt')
            # batch_to_save = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k,v in batch.items()}
            # torch.save(batch_to_save, "offending_batch.pt")
            fwd_ok.zero_()
        except BaseException as e:
            if 'out of memory' in str(e).lower() or 'cudaerrormemoryallocation' in str(e).lower():
                self._handle_oom('forward-base')
                print(f'global_step: {self.global_step}, batch_idx: {batch_idx}, scene_names: {batch["scene"]}, {str(e)}')
                fwd_ok.zero_()
            else:
                raise e
        # Global agreement BEFORE any backward/collective
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(fwd_ok, op=torch.distributed.ReduceOp.MIN)
        
        if fwd_ok.item() == 0:
            # all ranks skip together
            opt.zero_grad(set_to_none=True)
            # keep ranks aligned
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.trainer.strategy.barrier()
            # return a detached scalar for Lightning’s logging
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return total_loss

    def forward(self, batch, batch_idx):
        
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        # with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
        # with torch.autocast(device_type=self.device.type, enabled=False):

        gaussians = self.encoder(
            batch["context"], self.global_step, False, scene_names=batch["scene"], random_scale=self.train_controller_cfg.random_scale
        )       

        if isinstance(gaussians, dict):
            if self.gs_cube:
                nog_pb = gaussians["nog_pb"]
                nog_min = gaussians["nog_min"]
                for i in range(len(nog_pb)):
                    self.log(f"train/nog_pb_{i}", nog_pb[i], prog_bar=True,rank_zero_only=False, on_step=True,sync_dist=False)
                self.log("train/nog_min", nog_min, prog_bar=True,rank_zero_only=False, on_step=True,sync_dist=False)
                # print(f'rank_id:{self.trainer.local_rank}',f'global_step:{self.global_step+1}',batch['scene'], 'nog_pb:', [f'{pb:.4f}' for pb in nog_pb], f'nog_min: {nog_min:.4f}', flush=True)
            pred_depths = gaussians["depths"]
            gaussians = gaussians["gaussians"]
            
            
        # gaussians from bf16 to float 32   
        # gaussians = gaussians.as_float() 
        
        supervise_intermediate_depth = False

        if gaussians.means.size(0) != batch["target"]["extrinsics"].size(0):
            supervise_intermediate_depth = True
            assert gaussians.means.size(0) % batch["target"]["extrinsics"].size(0) == 0
            num_depths = gaussians.means.size(0) // batch["target"]["extrinsics"].size(
                0
            )
            # add loss to intermediate depth predictions
            target_extrinsics = torch.cat(
                [batch["target"]["extrinsics"]] * num_depths, dim=0
            )
            target_intrinsics = torch.cat(
                [batch["target"]["intrinsics"]] * num_depths, dim=0
            )
            target_near = torch.cat([batch["target"]["near"]] * num_depths, dim=0)
            target_far = torch.cat([batch["target"]["far"]] * num_depths, dim=0)

            output_all = self.decoder.forward(
                gaussians,
                target_extrinsics,
                target_intrinsics,
                target_near,
                target_far,
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
                scale_invariant = not self.vggt_meta
            )
            # split
            batch_size = batch["target"]["extrinsics"].size(0)
            # order: intermediate depth, final depth
            output_intermediate = DecoderOutput(
                color=output_all.color[:-batch_size],
                depth=(
                    output_all.depth[:-batch_size]
                    if output_all.depth is not None
                    else None
                ),
            )
            output = DecoderOutput(
                color=output_all.color[-batch_size:],
                depth=(
                    output_all.depth[-batch_size:]
                    if output_all.depth is not None
                    else None
                ),
            )

        else:
            # with torch.autocast(device_type=self.device.type, enabled=False):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
                vggt_meta=self.vggt_meta
            )

        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr", psnr_probabilistic.mean(), prog_bar=True,rank_zero_only=False, on_step=True,sync_dist=False)

        # Compute and log loss.
        total_loss = 0

        valid_depth_mask = None

        for loss_fn in self.losses:
            if loss_fn.name == "mse":
                loss = loss_fn.forward(
                    output,
                    batch,
                    gaussians,
                    self.global_step,
                    l1_loss=self.train_cfg.l1_loss,
                    clamp_large_error=self.train_cfg.train_ignore_large_loss,
                    valid_depth_mask=valid_depth_mask,
                )
            else:
                loss = loss_fn.forward(
                    output,
                    batch,
                    gaussians,
                    self.global_step,
                    valid_depth_mask=valid_depth_mask,
                )
            self.log(f"loss/{loss_fn.name}", loss, prog_bar=True,rank_zero_only=False, on_step=True,sync_dist=False)
            total_loss = total_loss + loss

        # color loss on intermediate output
        if supervise_intermediate_depth:
            for loss_fn in self.losses:
                if output_intermediate.color.size(0) != batch_size:
                    assert output_intermediate.color.size(0) % batch_size == 0
                    num_intermediate = output_intermediate.color.size(0) // batch_size
                    intermediate_loss = 0
                    for i in range(num_intermediate):
                        curr_output = DecoderOutput(
                            color=output_intermediate.color[
                                (batch_size * i) : (batch_size * (i + 1))
                            ],
                            depth=(
                                output_intermediate.depth[
                                    (batch_size * i) : (batch_size * (i + 1))
                                ]
                                if output_intermediate.depth is not None
                                else None
                            ),
                        )
                        curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (
                            num_intermediate - i
                        )

                        if loss_fn.name == "mse":
                            loss = loss_fn.forward(
                                curr_output,
                                batch,
                                gaussians,
                                self.global_step,
                                l1_loss=self.train_cfg.l1_loss,
                                clamp_large_error=self.train_cfg.train_ignore_large_loss,
                                valid_depth_mask=valid_depth_mask,
                            )
                        else:
                            loss = loss_fn.forward(
                                curr_output,
                                batch,
                                gaussians,
                                self.global_step,
                                valid_depth_mask=valid_depth_mask,
                            )

                        intermediate_loss = intermediate_loss + curr_loss_weight * loss

                    self.log(f"loss/{loss_fn.name}_intermediate", intermediate_loss)
                    total_loss = total_loss + intermediate_loss
                else:
                    if loss_fn.name == "mse":
                        loss = loss_fn.forward(
                            output_intermediate,
                            batch,
                            gaussians,
                            self.global_step,
                            l1_loss=self.train_cfg.l1_loss,
                            clamp_large_error=self.train_cfg.train_ignore_large_loss,
                            valid_depth_mask=valid_depth_mask,
                        )
                    else:
                        loss = loss_fn.forward(
                            output_intermediate,
                            batch,
                            gaussians,
                            self.global_step,
                            valid_depth_mask=valid_depth_mask,
                        )
                    self.log(f"loss/{loss_fn.name}_intermediate", loss)
                    total_loss = (
                        total_loss + self.train_cfg.intermediate_loss_weight * loss
                    )

        self.log("loss/total", total_loss, prog_bar=True,rank_zero_only=False, on_step=True,sync_dist=False)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step, rank_zero_only=False, on_step=True,sync_dist=False)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        # Since the pyotrch lightning synchronize the variable automatically, we don't need to maintain a step_tracker shared by processes.
        # Instead, we can set the global step at each training step end. Here step_tracker is commented out.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        if self.global_step == 5 and self.global_rank == 0:
            os.system("nvidia-smi")
        return total_loss
    
    def _chunk(self, context):
        chunk_num = self.chunk_num
        b, v, _, h, w = context["image"].shape
        if chunk_num == 1:
            return [context]
        assert v % chunk_num == 0
        chunk_size = v // chunk_num
        contexts = []
        for i in range(chunk_num):
            context_tmp={}
            for k in context.keys():
                context_tmp[k] = context[k][:, i*chunk_size:(i+1)*chunk_size]
            contexts.append(context_tmp)
        return contexts

    def _batch_forward(self, context, visualization_dump=None):
        def _merge(gaussians_list):
            new_gaussians = {}
            for gaussians in gaussians_list:
                for k in gaussians.keys():
                    if k not in new_gaussians:
                        new_gaussians[k] = []
                    new_gaussians[k].append(gaussians[k])
            for k in new_gaussians.keys():
                if k == 'gaussians':
                    length = len(new_gaussians['gaussians'])
                    means = [new_gaussians['gaussians'][l_].means for l_ in range(length)]
                    covarainces = [new_gaussians['gaussians'][l_].covariances for l_ in range(length)]
                    harmonics = [new_gaussians['gaussians'][l_].harmonics for l_ in range(length)]
                    opacities = [new_gaussians['gaussians'][l_].opacities for l_ in range(length)]
                    means = torch.cat(means, dim=1)
                    covarainces = torch.cat(covarainces, dim=1)
                    harmonics = torch.cat(harmonics, dim=1)
                    opacities = torch.cat(opacities, dim=1)
                    new_gaussians['gaussians'] = Gaussians(means, covarainces, harmonics, opacities)
                elif k == 'depths':
                    new_gaussians[k] = torch.cat(new_gaussians[k], dim=1)
                else:
                    pass   
            return new_gaussians
        if self.batch_forward and self.chunk_num>1:
            contexts = self._chunk(context)
            gaussians_batch = []
            for c in contexts:
                gaussians = self._online_test_forward(c, visualization_dump=visualization_dump)
                gaussians_batch.append(gaussians)
            gaussians = _merge(gaussians_batch)
        else:
            gaussians = self._online_test_forward(context, visualization_dump=visualization_dump)
        return gaussians

    def _online_test_forward(self, context, visualization_dump=None):
        v = context["image"].shape[1]
        if self.iter_depth:
            gaussians = self._anchor_forward(context, visualization_dump=visualization_dump, view_base=self.view_base)
        else:
            gaussians = self._anchor_forward(context, visualization_dump=visualization_dump, view_base = v)
        return gaussians
    
    def _anchor_forward(self,  context, visualization_dump=None, view_base=2):
        gaussians = self.encoder.anchor_forward(
            context,
            self.global_step,
            deterministic=False,
            visualization_dump=visualization_dump,
            view_base=view_base,
            anchor_features=self.anchor_features,
            anchor_base=self.anchor_base,
            noise_ratio=self.noise_ratio,
        )
        return gaussians
    def test_step(self, batch, batch_idx):
        # print(f'batch_idx: {batch_idx}', batch['scene'][0])
        # return
        # torch.cuda.empty_cache()
        batch: BatchedExample = self.data_shim(batch)
        # # get move length
        # P = batch["context"]['extrinsics'][0].cpu().numpy() # V,4,4
        # C = P[:,:3,3]
        # diffs = np.diff(C, axis=0) # V-1,3
        # seg_len = np.linalg.norm(diffs, axis=-1) 
        # move_length = np.sum(seg_len)
        # net_disps = np.linalg.norm(C[-1] - C[0])
        # print(f'scene:{batch["scene"][0]}, move_length: {move_length:.2f}, net_disp: {net_disps:.2f}')
        # self.test_length_dict[batch["scene"][0]] = (move_length, net_disps)
        # return

        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        pred_depths = None

        # print('num_of_ctx_views', batch['context']['image'].shape[1])
        # print('num_of_tgt_views', batch['target']['image'].shape[1])
        
        # if batch['scene'][0] != 'ce6c6d81af6b62cd':
        #     return
        # print(batch['scene'])
        # save input views for visualization
        if self.test_cfg.save_input_images:
            (scene,) = batch["scene"]
            self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
            path = Path(get_cfg()["output_dir"])

            input_images = batch["context"]["image"][0]  # [V, 3, H, W]
            index = batch["context"]["index"][0]
            for idx, color in zip(index, input_images):
                save_image(color, path / "images" / scene / f"color/input_{idx:0>6}.png")

        # save depth vis
        if self.test_cfg.save_depth or self.test_cfg.save_gaussian:
            visualization_dump = {}
        else:
            visualization_dump = None

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            # batch["context"]["extrinsics"] = batch["context"]["extrinsics"] + 0.01*torch.rand_like(batch["context"]["extrinsics"])
            gaussians = self._batch_forward(batch['context'], visualization_dump=visualization_dump)
            # check depth
            # depths_my = gaussians["depths"]  # [1, V, H, W]
            # for nd, depth_i in enumerate(depths_my[0]):
            #     depth_i = depth_i.cpu()
            #     depth_viz = viz_depth_tensor(
            #             1.0 / depth_i, return_numpy=True
            #         )
            #     Image.fromarray(depth_viz).save(f'notes/depths/depth_check_{nd}.png')


            if isinstance(gaussians, dict):
                pred_depths = gaussians["depths"]
                if "depth" in batch["context"]:
                    depth_gt = batch["context"]["depth"]         
                gaussians = gaussians["gaussians"]
                # print(f'number of gaussians {gaussians.means.shape[1]}')
                self.nog.append(gaussians.means.shape[1])
        # trim large scales for gscube
        # if self.gs_cube:
        #     scales = visualization_dump["cube_scales"]
        #     max_scale = torch.max(scales[0], dim=-1)[0]  # [V]
        #     selected_ind = torch.where(max_scale < 1.5)[0]
        #     gaussians.means = gaussians.means[:,selected_ind]
        #     gaussians.covariances = gaussians.covariances[:,selected_ind]
        #     gaussians.harmonics = gaussians.harmonics[:,selected_ind]
        #     gaussians.opacities = gaussians.opacities[:,selected_ind]
        #     visualization_dump["cube_scales"] = visualization_dump["cube_scales"][:,selected_ind]
        #     visualization_dump["cube_rotations"] = visualization_dump["cube_rotations"][:,selected_ind]

        # save gaussians
        if self.test_cfg.save_gaussian: 
            if self.gs_cube:
                scene = batch["scene"][0]
                save_path = Path(get_cfg()['output_dir']) / 'gaussians_cube' / (scene + f'_{self.global_step}' + '.ply')
                save_gaussian_cube_ply(gaussians, visualization_dump, batch, save_path)
            else:
                scene = batch["scene"][0]
                save_path = Path(get_cfg()['output_dir']) / 'gaussians' / (scene + '.ply')
                save_gaussian_ply(gaussians, visualization_dump, batch, save_path)

        if not self.train_cfg.forward_depth_only:
            with self.benchmarker.time("decoder", num_calls=v):

                camera_poses = batch["target"]["extrinsics"]

                if self.test_cfg.stablize_camera:
                    stable_poses = render_stabilization_path(
                        camera_poses[0].detach().cpu().numpy(),
                        k_size=self.test_cfg.stab_camera_kernel,
                    )

                    stable_poses = list(
                        map(
                            lambda x: np.concatenate(
                                (x, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0
                            ),
                            stable_poses,
                        )
                    )
                    stable_poses = torch.from_numpy(np.stack(stable_poses, axis=0)).to(
                        camera_poses
                    )
                    camera_poses = stable_poses.unsqueeze(0)

                if self.test_cfg.render_chunk_size is not None:
                    chunk_size = self.test_cfg.render_chunk_size
                    num_chunks = math.ceil(camera_poses.shape[1] / chunk_size)

                    output = None
                    for i in range(num_chunks):
                        start = chunk_size * i
                        end = chunk_size * (i + 1)

                        render_intrinsics = batch["target"]["intrinsics"]
                        render_near = batch["target"]["near"]
                        render_far = batch["target"]["far"]

                        curr_output = self.decoder.forward(
                            gaussians,
                            camera_poses[:, start:end],
                            render_intrinsics[:, start:end],
                            render_near[:, start:end],
                            render_far[:, start:end],
                            (h, w),
                            depth_mode="depth",
                            vggt_meta=self.vggt_meta
                        )

                        if i == 0:
                            output = curr_output
                        else:
                            # ignore depth
                            output.color = torch.cat(
                                (output.color, curr_output.color), dim=1
                            )
                            if hasattr(output, "depth") and hasattr(curr_output, "depth"):
                                output.depth = torch.cat(
                                    (output.depth, curr_output.depth), dim=1
                                )

                else:
                    output = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        depth_mode=None,
                        vggt_meta=self.vggt_meta
                    )

        (scene,) = batch["scene"]
        self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
        path = Path(get_cfg()["output_dir"])

        # save depth
        if self.test_cfg.save_depth:
            out_depth = output.depth[0]  # [V, H, W]
            for idx, depth_i in zip(batch["target"]["index"][0], out_depth):
                depth_i = depth_i.cpu().detach()
                depth_i = (depth_i - depth_i.min()) / (depth_i.max() - depth_i.min() + 1e-8)
                depth_viz = viz_depth_tensor(
                    depth_i, return_numpy=True
                )  # [H, W, 3]
                save_path = path / "images" / scene / "depth" / f"out_depth_{idx:0>6}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(depth_viz).save(save_path)

            if self.train_cfg.forward_depth_only:
                depth = pred_depths[0].cpu().detach()  # [V, H, W]
            else:
                depth = (
                    visualization_dump["depth"][0, :, :, :, 0, 0].cpu().detach()
                )  # [V, H, W]

            index = batch["context"]["index"][0]

            if self.test_cfg.save_depth_concat_img:
                # concat (img0, img1, depth0, depth1)
                image = batch['context']['image'][0]  # [V, 3, H, W] in [0,1]
                image = rearrange(image, "b c h w -> h (b w) c")  # [H, VW, 3]
                image_concat = (image.detach().cpu().numpy() * 255).astype(np.uint8)  # [H, VW, 3]

                depth_concat = []

            for idx, depth_i in zip(index, depth):
                depth_viz = viz_depth_tensor(
                    1.0 / depth_i, return_numpy=True
                )  # [H, W, 3]

                if self.test_cfg.save_depth_concat_img:
                    depth_concat.append(depth_viz)

                save_path = path / "images" / scene / "depth" / f"{idx:0>6}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(depth_viz).save(save_path)

                # save depth as npy
                if self.test_cfg.save_depth_npy:
                    depth_npy = depth_i.detach().cpu().numpy()
                    save_path = path / "images" / scene / "depth" / f"{idx:0>6}.npy"
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    np.save(save_path, depth_npy)

            if self.test_cfg.save_depth_concat_img:
                depth_concat = np.concatenate(depth_concat, axis=1)  # [H, VW, 3]
                concat = np.concatenate((image_concat, depth_concat), axis=0)  # [2H, VW, 3]

                save_path = path / "images" / scene / "depth" /  f"img_depth_{scene}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(concat).save(save_path)

            if self.train_cfg.forward_depth_only:
                return

        images_prob = output.color[0]
        rgb_gt = batch["target"]["image"][0]

        # Save images.
        if self.test_cfg.save_image:
            if self.test_cfg.save_gt_image:
                for index, color, gt in zip(
                    batch["target"]["index"][0], images_prob, rgb_gt
                ):
                    save_image(color, path / "images" / scene / f"color/{index:0>6}.png")
                    save_image(gt, path / "images" / scene / f"color/{index:0>6}_gt.png")
            else:
                for index, color in zip(batch["target"]["index"][0], images_prob):
                    save_image(color, path / "images" / scene / f"color/{index:0>6}.png")

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "videos" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v

            if not self.train_cfg.forward_depth_only:
                rgb = images_prob

                if f"psnr" not in self.test_step_outputs:
                    self.test_step_outputs[f"psnr"] = []
                if f"ssim" not in self.test_step_outputs:
                    self.test_step_outputs[f"ssim"] = []
                if f"lpips" not in self.test_step_outputs:
                    self.test_step_outputs[f"lpips"] = []

                self.test_step_output_dict[scene]=compute_psnr(rgb_gt, rgb).detach().cpu().numpy()
                self.test_step_outputs[f"psnr"].append(
                    compute_psnr(rgb_gt, rgb).mean().item()
                )
                self.test_step_outputs[f"ssim"].append(
                    compute_ssim(rgb_gt, rgb).mean().item()
                )
                self.test_step_outputs[f"lpips"].append(
                    compute_lpips(rgb_gt, rgb).mean().item()
                )

    def on_test_end(self) -> None:
        nog = np.array(self.nog).mean()
        print(f'average number of gaussians: {nog}')
        # np.save(Path('notes') / "scannet_ind.npy", self.all_ind)
        # return
        out_dir = Path(self.test_cfg.output_path)
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            np.save(out_dir / "test_step_psnr_dict.npy",self.test_step_output_dict)

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(out_dir / "benchmark.json")
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
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

            images = batch['context']['image']
            depth = batch['context']['depth']
            extrinsic = batch['context']['extrinsics'].inverse()
            intrinsic = batch['context']['intrinsics']
            indices = [torch.arange(length) for length in depth.shape[-2:]]
            coordinates = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1).to(depth)
            b,v = depth.shape[:2]
            depth = rearrange(depth, "b v h w -> (b v) h w")
            extrinsic = rearrange(extrinsic, "b v c d -> (b v) c d")
            intrinsic = rearrange(intrinsic, "b v c d -> (b v) c d")
            coordinates = repeat(coordinates, "h w c-> n h w c", n=b*v)
            images = rearrange(images, "b v c h w -> (b v) h w c")
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
            render_pcs(points, colors, opacities, 'after_data_shim')


        # with torch.autocast(device_type=self.device.type, enabled=False):
            # torch.cuda.empty_cache()
        save_interval = 1000
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if self.global_step % save_interval == 0 and self.global_rank == 0 and batch_idx==0:
            visualization_dump = {}
            gaussians_softmax = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
                visualization_dump=visualization_dump,
            )
        else:
            try:
                gaussians_softmax = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=False,
                )
            except ValueError as e:
                self._handle_oom('value error')
                print(f"Value error in encoder: {e}")
                print(f'global_step: {self.global_step}, batch_idx: {batch_idx}, scene_names: {batch["scene"]}')
                return
        pred_depths = None

        if isinstance(gaussians_softmax, dict):
            pred_depths = gaussians_softmax["depths"]
            if "depth" in batch["context"]:
                depth_gt = batch["context"]["depth"]  # [B, V, H, W]
            gaussians_softmax = gaussians_softmax["gaussians"]

        if not self.train_cfg.forward_depth_only:
            output_softmax = self.decoder.forward(
                gaussians_softmax,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                vggt_meta = self.vggt_meta,
            )
            # output_softmax = self.decoder.forward(
            #     gaussians_softmax,
            #     batch["context"]["extrinsics"],
            #     batch["context"]["intrinsics"],
            #     batch["context"]["near"],
            #     batch["context"]["far"],
            #     (h, w),
            #     vggt_meta = self.vggt_meta,
            #     scale_invariant = True
            # )
            rgb_softmax = output_softmax.color[0]

            # Compute validation metrics.
            rgb_gt = batch["target"]["image"][0]
            for tag, rgb in zip(("val",), (rgb_softmax,)):
                psnr = compute_psnr(rgb_gt, rgb).mean()
                self.log(f"val/psnr_{tag}", psnr)
                lpips = compute_lpips(rgb_gt, rgb).mean()
                self.log(f"val/lpips_{tag}", lpips)
                ssim = compute_ssim(rgb_gt, rgb).mean()
                self.log(f"val/ssim_{tag}", ssim)

        # viz depth
        if pred_depths is not None:
            # only visualize predicted depth
            pred_depths = pred_depths[0]  # [V, H, W]

            # gaussian downsample
            if pred_depths.shape[1:] != batch["context"]["image"].shape[-2:]:
                pred_depths = F.interpolate(
                    pred_depths.unsqueeze(1),
                    size=batch["context"]["image"].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(1)
            if self.vggt_meta:
                inverse_depth_pred = 1.0/ pred_depths
            else:
                inverse_depth_pred = 1.0 / pred_depths

            concat = []
            for i in range(inverse_depth_pred.size(0)):
                concat.append(inverse_depth_pred[i])

            concat = torch.cat(concat, dim=1)  # [H, W*N]

            depth_viz = viz_depth_tensor(concat.cpu().detach())  # [3, H, W*N]

            # also concat images
            input_images = batch["context"]["image"][0]  # [N, 3, H, W]
            concat_img = [img for img in input_images]
            concat_img = torch.cat(concat_img, dim=-1) * 255  # [3, H, W*N]

            concat = torch.cat(
                (concat_img.cpu().detach(), depth_viz), dim=1
            )  # [3, H*2, W*N]

            self.logger.log_image(
                "depth",
                [concat],
                step=self.global_step,
                caption=batch["scene"],
            )

        if not self.train_cfg.forward_depth_only:
            # Construct comparison image.
            comparison = hcat(
                add_label(vcat(*batch["context"]["image"][0]), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_softmax), "Target (Prediction)"),
            )
            self.logger.log_image(
                "comparison",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )
            # pil_img = PILImage.fromarray(prep_image(add_border(comparison)))
            # pil_img.save("comparisons/{}.png".format(self.global_step))

            if not self.train_cfg.no_log_projections:
                # Render projections and construct projection image.
                projections = hcat(
                    *render_projections(
                        gaussians_softmax,
                        256,
                        extra_label="(Prediction)",
                    )[0]
                )
                self.logger.log_image(
                    "projection",
                    [prep_image(add_border(projections))],
                    step=self.global_step,
                )

                # Draw cameras.
                cameras = hcat(*render_cameras(batch, 256))
                self.logger.log_image(
                    "cameras", [prep_image(add_border(cameras))], step=self.global_step
                )

            if self.encoder_visualizer is not None:
                for k, image in self.encoder_visualizer.visualize(
                    batch["context"], self.global_step
                ).items():
                    self.logger.log_image(k, [prep_image(image)], step=self.global_step)

            # Run video validation step.
            if not self.train_cfg.no_viz_video:
                self.render_video_interpolation(batch, vggt_meta=self.vggt_meta)
                self.render_video_wobble(batch, vggt_meta=self.vggt_meta)
                if self.train_cfg.extended_visualization:
                    self.render_video_interpolation_exaggerated(batch, vggt_meta=self.vggt_meta)

        if self.global_step % save_interval == 0 and self.global_rank == 0 and batch_idx==0:
            if self.gs_cube:
                print(f"Saving Gaussian cube at step {self.global_step}...")
                scene = batch["scene"][0]
                save_path = Path(get_cfg()['output_dir']) / 'gaussians_cube' / (scene + f'_{self.global_step}' + '.ply')
                save_gaussian_cube_ply(gaussians_softmax, visualization_dump, batch, save_path)
                visualization_dump = None
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        # with torch.autocast(device_type=self.device.type, enabled=False):
        """hack to run the full validation"""
        if self.trainer.sanity_checking and self.global_rank == 0:
            print(self.encoder)  # log the model to wandb log files

        if (not self.trainer.sanity_checking) and (self.eval_data_cfg is not None):
            self.eval_cnt = self.eval_cnt + 1
            if self.eval_cnt % self.train_cfg.eval_model_every_n_val == 0:
                # backup current ckpt before running full test sets eval
                if self.train_cfg.eval_save_model:
                    ckpt_saved_path = (
                        self.trainer.checkpoint_callback.format_checkpoint_name(
                            dict(
                                epoch=self.trainer.current_epoch,
                                step=self.trainer.global_step,
                            )
                        )
                    )
                    backup_dir = str(
                        Path(ckpt_saved_path).parent.parent / "checkpoints_backups"
                    )
                    if self.global_rank == 0:
                        os.makedirs(backup_dir, exist_ok=True)
                    ckpt_saved_path = os.path.join(
                        backup_dir, os.path.basename(ckpt_saved_path)
                    )
                    # call save_checkpoint on ALL process as suggested by pytorch_lightning
                    self.trainer.save_checkpoint(
                        ckpt_saved_path,
                        weights_only=True,
                    )
                    if self.global_rank == 0:
                        print(f"backup model to {ckpt_saved_path}.")

                # run full test sets eval on rank=0 device
                self.run_full_test_sets_eval()

    @rank_zero_only
    def run_full_test_sets_eval(self) -> None:
        start_t = time.time()

        pred_depths = None
        depth_gt = None

        full_testsets = self.trainer.datamodule.test_dataloader(
            dataset_cfg=self.eval_data_cfg
        )
        scores_dict = {}

        if not self.train_cfg.forward_depth_only:
            for score_tag in ("psnr", "ssim", "lpips"):
                scores_dict[score_tag] = {}
                for method_tag in ("deterministic", "probabilistic"):
                    scores_dict[score_tag][method_tag] = []

        # evaluate depth
        if self.train_cfg.viz_depth:
            for score_tag in ("abs_rel", "rmse", "a1"):
                scores_dict[score_tag] = {}
                for method_tag in ("deterministic", "probabilistic"):
                    scores_dict[score_tag][method_tag] = []

        self.benchmarker.clear_history()
        time_skip_first_n_steps = min(
            self.train_cfg.eval_time_skip_steps, len(full_testsets)
        )
        time_skip_steps_dict = {"encoder": 0, "decoder": 0}
        for batch_idx, batch in tqdm(
            enumerate(full_testsets),
            total=min(len(full_testsets), self.train_cfg.eval_data_length),
        ):
            if batch_idx >= self.train_cfg.eval_data_length:
                break
            # torch.cuda.empty_cache()

            batch = self.data_shim(batch)
            batch = self.transfer_batch_to_device(batch, "cuda", dataloader_idx=0)

            # Render Gaussians.
            b, v, _, h, w = batch["target"]["image"].shape
            assert b == 1
            if batch_idx < time_skip_first_n_steps:
                time_skip_steps_dict["encoder"] += 1
                time_skip_steps_dict["decoder"] += v

            with self.benchmarker.time("encoder"):
                gaussians_probabilistic = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=False,
                )

                if isinstance(gaussians_probabilistic, dict):
                    pred_depths = gaussians_probabilistic["depths"]
                    if "depth" in batch["context"]:
                        depth_gt = batch["context"]["depth"]
                    gaussians_probabilistic = gaussians_probabilistic["gaussians"]

            if not self.train_cfg.forward_depth_only:
                with self.benchmarker.time("decoder", num_calls=v):
                    output_probabilistic = self.decoder.forward(
                        gaussians_probabilistic,
                        batch["target"]["extrinsics"],
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        vggt_meta=self.vggt_meta,
                    )
                rgbs = [output_probabilistic.color[0]]
                tags = ["probabilistic"]

                if self.train_cfg.eval_deterministic:
                    gaussians_deterministic = self.encoder(
                        batch["context"],
                        self.global_step,
                        deterministic=True,
                    )
                    output_deterministic = self.decoder.forward(
                        gaussians_deterministic,
                        batch["target"]["extrinsics"],
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        vggt_meta=self.vggt_meta,
                    )
                    rgbs.append(output_deterministic.color[0])
                    tags.append("deterministic")

                # Compute validation metrics.
                rgb_gt = batch["target"]["image"][0]
                for tag, rgb in zip(tags, rgbs):
                    scores_dict["psnr"][tag].append(
                        compute_psnr(rgb_gt, rgb).mean().item()
                    )
                    scores_dict["lpips"][tag].append(
                        compute_lpips(rgb_gt, rgb).mean().item()
                    )
                    scores_dict["ssim"][tag].append(
                        compute_ssim(rgb_gt, rgb).mean().item()
                    )

        # summarise scores and log to logger
        for score_tag, methods in scores_dict.items():
            for method_tag, cur_scores in methods.items():
                if len(cur_scores) > 0:
                    cur_mean = sum(cur_scores) / len(cur_scores)
                    self.log(f"test/{score_tag}", cur_mean)
        # summarise run time
        for tag, times in self.benchmarker.execution_times.items():
            times = times[int(time_skip_steps_dict[tag]) :]
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
            self.log(f"test/runtime_avg_{tag}", np.mean(times))
        self.benchmarker.clear_history()

        overall_eval_time = time.time() - start_t
        print(f"Eval total time cost: {overall_eval_time:.3f}s")
        self.log("test/runtime_all", overall_eval_time)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample, vggt_meta=False) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60, vggt_meta=vggt_meta)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample, vggt_meta=False) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb", vggt_meta=vggt_meta)

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample, vggt_meta=False) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
            vggt_meta=vggt_meta,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
        vggt_meta: bool = False
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        if isinstance(gaussians_prob, dict):
            gaussians_prob = gaussians_prob["gaussians"]

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth", vggt_meta=vggt_meta
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]

        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Prediction"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        if self.optimizer_cfg.lr_monodepth > 0:
            pretrained_params = []
            new_params = []
            gs_cube_params = []

            for name, param in self.named_parameters():
                if "pretrained" in name:
                    pretrained_params.append(param)
                elif "gs_cube_encoder" in name:
                    gs_cube_params.append(param)
                else:
                    new_params.append(param)

            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": pretrained_params,
                        "lr": self.optimizer_cfg.lr_monodepth,
                    },
                    {"params": new_params, "lr": self.optimizer_cfg.lr},
                    {"params": gs_cube_params, "lr": self.optimizer_cfg.lr_gs_cube},
                ],
                weight_decay=self.optimizer_cfg.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                [self.optimizer_cfg.lr_monodepth, self.optimizer_cfg.lr, self.optimizer_cfg.lr_gs_cube],
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        else:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.optimizer_cfg.lr,
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
