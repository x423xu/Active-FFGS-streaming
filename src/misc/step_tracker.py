from multiprocessing import RLock
import os
# from multiprocessing.managers import SyncManager

import torch
from jaxtyping import Int64
from torch import Tensor
from torch.multiprocessing import Manager
from torch.distributed import get_rank, get_world_size, broadcast


class StepTracker:
    lock: RLock
    step: Int64[Tensor, ""]

    def __init__(self):
        self.lock = Manager().RLock()
        self.step = torch.tensor(0, dtype=torch.int64).share_memory_()
        # self.step = torch.tensor(0, dtype=torch.int64).cuda()  # Store locally on GPU
        # self.rank = get_rank() if torch.distributed.is_initialized() else 0

    def set_step(self, step: int) -> None:
        # print(f"Process {os.getpid()}: Setting step to {step}")
        with self.lock:
            self.step.fill_(step)
        # if self.rank == 0:
        #     print(f"Process {os.getpid()} (rank {self.rank}): Setting step to {step}")
        #     self.step.fill_(step)
        # if torch.distributed.is_initialized():
        #     step_tensor = torch.tensor(step if self.rank == 0 else 0, dtype=torch.int64).cuda()
        #     broadcast(step_tensor, src=0)
        #     self.step.fill_(step_tensor.item())

    def get_step(self) -> int:
        # print(f"Process {os.getpid()}: Getting step: {self.step.item()}")
        with self.lock:
            return self.step.item()
        # return self.step.item()
