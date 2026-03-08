from math import isqrt
import math
import functools

import torch
from e3nn.o3 import matrix_to_angles, wigner_D
from einops import einsum
from jaxtyping import Float
from torch import Tensor


# ---------------------------------------------------------------------------
# Precomputation: SO(3) Lie-algebra generators  (constant, cached)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def _precompute_so3_generators(max_degree: int):
    """SO(3) generators for degrees 0…max_degree-1, zero-padded and stacked.

    Returns (X0, X1) of shape [max_degree, S, S]  (S = max(2·(max_degree-1)+1, 1))
    in float64 on CPU.  X0 is used with β, X1 with α and γ.
    """
    S = max(2 * (max_degree - 1) + 1, 1)
    X0 = torch.zeros(max_degree, S, S, dtype=torch.float64)
    X1 = torch.zeros(max_degree, S, S, dtype=torch.float64)

    for l in range(max_degree):
        size = 2 * l + 1
        j = float(l)

        # --- SU(2) generators (complex) ---
        m_r = torch.arange(-j, j, dtype=torch.float64)
        raising = torch.diag(
            -torch.sqrt(j * (j + 1) - m_r * (m_r + 1)), diagonal=-1
        )
        m_l = torch.arange(-j + 1, j + 1, dtype=torch.float64)
        lowering = torch.diag(
            torch.sqrt(j * (j + 1) - m_l * (m_l - 1)), diagonal=1
        )
        m_d = torch.arange(-j, j + 1, dtype=torch.float64)

        su2 = torch.stack([
            (0.5 * (raising + lowering)).to(torch.complex128),
            torch.diag(1j * m_d).to(torch.complex128),
            (-0.5j * (raising - lowering)).to(torch.complex128),
        ])  # [3, size, size]

        # --- Change-of-basis  (real SH ↔ complex SH) ---
        Q = torch.zeros(size, size, dtype=torch.complex128)
        for m in range(-l, 0):
            Q[l + m, l + abs(m)] = 1.0 / 2.0 ** 0.5
            Q[l + m, l - abs(m)] = -1j / 2.0 ** 0.5
        Q[l, l] = 1.0
        for m in range(1, l + 1):
            Q[l + m, l + abs(m)] = (-1) ** m / 2.0 ** 0.5
            Q[l + m, l - abs(m)] = 1j * (-1) ** m / 2.0 ** 0.5
        Q = (-1j) ** l * Q

        # --- SO(3) generators in real SH basis ---
        so3 = torch.conj(Q.T) @ su2 @ Q
        so3 = torch.real(so3)  # imaginary part is numerical noise

        X0[l, :size, :size] = so3[0]
        X1[l, :size, :size] = so3[1]

    return X0, X1


@functools.lru_cache(maxsize=16)
def _precompute_block_diag_indices(max_degree: int):
    """Index tensors for scattering padded Wigner-D blocks into block-diag.

    Returns (row, col, deg, local_row, local_col) as long tensors on CPU.
    """
    rows, cols, deg, lr, lc = [], [], [], [], []
    for l in range(max_degree):
        size = 2 * l + 1
        start = l * l
        for i in range(size):
            for j in range(size):
                rows.append(start + i)
                cols.append(start + j)
                deg.append(l)
                lr.append(i)
                lc.append(j)
    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(cols, dtype=torch.long),
        torch.tensor(deg, dtype=torch.long),
        torch.tensor(lr, dtype=torch.long),
        torch.tensor(lc, dtype=torch.long),
    )


# Per-device caches so .to() is not called every frame
_gen_gpu_cache: dict = {}
_idx_gpu_cache: dict = {}


def _get_generators(max_degree: int, device: torch.device, dtype: torch.dtype):
    key = (max_degree, str(device), dtype)
    if key not in _gen_gpu_cache:
        X0, X1 = _precompute_so3_generators(max_degree)
        _gen_gpu_cache[key] = (
            X0.to(device=device, dtype=dtype).contiguous(),
            X1.to(device=device, dtype=dtype).contiguous(),
        )
    return _gen_gpu_cache[key]


def _get_block_indices(max_degree: int, device: torch.device):
    key = (max_degree, str(device))
    if key not in _idx_gpu_cache:
        ri, ci, di, lr, lc = _precompute_block_diag_indices(max_degree)
        _idx_gpu_cache[key] = tuple(t.to(device) for t in (ri, ci, di, lr, lc))
    return _idx_gpu_cache[key]


# ---------------------------------------------------------------------------
# Batched Wigner-D:  all degrees in ONE set of matrix_exp + matmul
# ---------------------------------------------------------------------------

def batch_wigner_D(max_degree: int, alpha, beta, gamma):
    """Wigner-D matrices for degrees 0…max_degree-1 without a Python for-loop.

    Precomputed SO(3) generators are padded to a common size and stacked along
    a *degree* batch dimension.  Three ``torch.matrix_exp`` calls + two batched
    matmuls replace the original per-degree loop.

    Returns tensor of shape ``[*batch, max_degree, S, S]``
    (S = 2·(max_degree-1)+1).  For degree *l* the valid block is
    ``result[…, l, :2l+1, :2l+1]``.
    """
    X0, X1 = _get_generators(max_degree, alpha.device, alpha.dtype)
    # X0, X1: [max_degree, S, S]

    a = (alpha % (2 * math.pi))[..., None, None, None]  # [*batch, 1, 1, 1]
    b = (beta  % (2 * math.pi))[..., None, None, None]
    g = (gamma % (2 * math.pi))[..., None, None, None]

    # Batched over (*batch, max_degree): 3 matrix_exp + 2 matmul
    return (
        torch.matrix_exp(a * X1)
        @ torch.matrix_exp(b * X0)
        @ torch.matrix_exp(g * X1)
    )


# ---------------------------------------------------------------------------
# Reference: original e3nn-based per-degree loop (kept for comparison)
# ---------------------------------------------------------------------------

def rotate_sh_reference(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    """Reference implementation (original e3nn wigner_D per-degree for-loop)."""
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype

    *_, n = sh_coefficients.shape
    rot_extra = sh_coefficients.ndim - 1 - (rotations.ndim - 2)
    alpha, beta, gamma = matrix_to_angles(rotations)
    result = []
    for degree in range(isqrt(n)):
        with torch.device(device):
            sh_rotations = wigner_D(degree, alpha, beta, gamma).to(dtype=dtype)
        # Insert broadcast dims so rot can broadcast with coeff batch dims
        for _ in range(rot_extra):
            sh_rotations = sh_rotations.unsqueeze(-3)
        coeffs = sh_coefficients[..., degree**2 : (degree + 1) ** 2]
        sh_rotated = torch.matmul(sh_rotations, coeffs.unsqueeze(-1)).squeeze(-1)
        result.append(sh_rotated)

    return torch.cat(result, dim=-1)


# ---------------------------------------------------------------------------
# Fast path: batched Wigner-D + block-diag assembly + GEMM  (no for-loop)
# ---------------------------------------------------------------------------

def fast_rotate_sh(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    """Fast SH rotation — zero Python for-loops in the hot path.

    1. ``batch_wigner_D`` computes all Wigner-D blocks in one batched op.
    2. Advanced indexing scatters them into a block-diagonal matrix.
    3. A single batched GEMM applies the rotation.
    """
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype

    coeff_batch = sh_coefficients.shape[:-1]
    rot_batch = rotations.shape[:-2]
    n = sh_coefficients.shape[-1]
    max_degree = isqrt(n)

    alpha, beta, gamma = matrix_to_angles(rotations)

    # --- 1. Batched Wigner-D (no for-loop) ---
    D_all = batch_wigner_D(max_degree, alpha, beta, gamma)
    # D_all: [*rot_batch, max_degree, S, S]

    # --- 2. Assemble block-diagonal via advanced indexing (no for-loop) ---
    ri, ci, di, lr, lc = _get_block_indices(max_degree, device)
    full_rot = torch.zeros(*rot_batch, n, n, device=device, dtype=dtype)
    full_rot[..., ri, ci] = D_all[..., di, lr, lc].to(dtype=dtype)

    # --- 3. GEMM path ---
    shared_prefix = 0
    max_prefix = min(len(rot_batch), len(coeff_batch))
    for i in range(max_prefix):
        if rot_batch[i] == coeff_batch[i] and rot_batch[i] > 1:
            shared_prefix += 1
        else:
            break

    can_use_gemm = True
    for i in range(shared_prefix, len(rot_batch)):
        if rot_batch[i] != 1:
            can_use_gemm = False
            break

    if can_use_gemm and shared_prefix > 0:
        group_shape = coeff_batch[:shared_prefix]
        suffix_shape = coeff_batch[shared_prefix:]

        index = (slice(None),) * shared_prefix + (0,) * (len(rot_batch) - shared_prefix)
        full_rot_group = full_rot[index]

        group_size = 1
        for d in group_shape:
            group_size *= d
        suffix_size = 1
        for d in suffix_shape:
            suffix_size *= d

        coeff_2d = sh_coefficients.reshape(group_size, suffix_size, n)
        rot_2d = full_rot_group.reshape(group_size, n, n)

        out = torch.bmm(coeff_2d, rot_2d.transpose(1, 2))
        return out.reshape(*coeff_batch, n)

    # Fallback: generic broadcasted batched matmul.
    if len(rot_batch) < len(coeff_batch):
        extra = len(coeff_batch) - len(rot_batch)
        full_rot = full_rot.reshape(*rot_batch, *([1] * extra), n, n)
    return torch.matmul(full_rot, sh_coefficients.unsqueeze(-1)).squeeze(-1)


def rotate_sh(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    return fast_rotate_sh(sh_coefficients, rotations)
    # return rotate_sh_reference(sh_coefficients, rotations)


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    from e3nn.o3 import spherical_harmonics
    from matplotlib import cm
    from scipy.spatial.transform.rotation import Rotation as R

    device = torch.device("cuda")

    # Generate random spherical harmonics coefficients.
    degree = 4
    coefficients = torch.rand((degree + 1) ** 2, dtype=torch.float32, device=device)

    def plot_sh(sh_coefficients, path: Path) -> None:
        phi = torch.linspace(0, torch.pi, 100, device=device)
        theta = torch.linspace(0, 2 * torch.pi, 100, device=device)
        phi, theta = torch.meshgrid(phi, theta, indexing="xy")
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        xyz = torch.stack([x, y, z], dim=-1)
        sh = spherical_harmonics(list(range(degree + 1)), xyz, True)
        result = einsum(sh, sh_coefficients, "... n, n -> ...")
        result = (result - result.min()) / (result.max() - result.min())

        # Set the aspect ratio to 1 so our sphere looks spherical
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            x.cpu().numpy(),
            y.cpu().numpy(),
            z.cpu().numpy(),
            rstride=1,
            cstride=1,
            facecolors=cm.seismic(result.cpu().numpy()),
        )
        # Turn off the axis planes
        ax.set_axis_off()
        path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(path)

    for i, angle in enumerate(torch.linspace(0, 2 * torch.pi, 30)):
        rotation = torch.tensor(
            R.from_euler("x", angle.item()).as_matrix(), device=device
        )
        plot_sh(rotate_sh(coefficients, rotation), Path(f"sh_rotation/{i:0>3}.png"))

    print("Done!")
