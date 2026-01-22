
# Streaming Active Gaussian Splatting (Active-GS) Framework

This document defines a concrete, implementation-oriented framework for **streaming Active Gaussian Splatting** where the system (a) estimates geometry/cameras from incoming frames, (b) refines a camera-aware GS model, (c) actively chooses what to process next under compute/bandwidth limits, (d) performs **rate–distortion optimization (RDO)** for transmission, and (e) continuously monitors **FLOPs** and **bandwidth**.

## 0) Scope and Assumptions

- **Streaming setting**: frames arrive over time; the system must decide what to compute + what to transmit each step.
- **Representation**: a 3D scene is represented as a set of persistent Gaussians (position, covariance/scale, opacity, color/features).
- **VGGT** is used to estimate **meta depth** and **camera parameters** from frames.
- **Active** means the system chooses actions (e.g., which frame/view, which regions, which model updates, what bitrate) to maximize utility under constraints.

## 1) End-to-End Pipeline (High Level)

At each time step $t$ with an incoming frame $I_t$ (and optionally IMU/pose priors):

1. **VGGT**: infer depth + camera intrinsics/extrinsics + uncertainty.
2. **GS Update**: update/optimize Gaussians using camera embedding + Pl\"ucker rays.
3. **Active Decision**: choose next compute/transmit action to maximize expected improvement per cost.
4. **RDO**: compress/quantize/prune GS updates (or patches) to meet bitrate.
5. **Monitor**: measure FLOPs, latency, and bandwidth; feed back into the active/RDO controllers.

Outputs are streamed as **GS state deltas** (preferred).

## 2) Stage A — VGGT for Meta Depth + Camera Parameters

### 2.1 Inputs / Outputs

**Inputs**
- Frame $I_t$ (RGB).
- Optional: rolling shutter metadata, timestamps, IMU, rough intrinsics.

**Outputs (per-frame)**
- Depth: $D_t$ ("meta depth") and optionally confidence/uncertainty $\sigma_{D_t}$.
- Camera intrinsics: $K_t$ (or corrections $\Delta K_t$).
- Camera pose: $T_{w\leftarrow c}(t)$ (world-from-camera) and uncertainty $\Sigma_{T_t}$.
- Optional: per-pixel normals or feature maps $F_t$.

### 2.2 Recommended Interface

Define a stable internal API so downstream modules don’t depend on VGGT specifics:

- `vggt_infer(I_t) -> {D_t, sigma_D_t, K_t, T_wc_t, Sigma_T_t, feat_t}`

### 2.3 Practical Notes

- **Uncertainty matters**: propagate VGGT uncertainty into optimization weights (depth residual weighting; pose regularization).
- **Drift handling**: allow occasional global alignment (loop closure / global BA) if streaming sequences are long; otherwise keep it local-window.
- **Failure detection**: if pose confidence is low, downweight updates, trigger active re-acquisition (Section 4).

## 3) Stage B — Model Architecture Refinement

This stage defines how the GS model consumes camera information and ray geometry.

### 3.1 Core Gaussian State

For each Gaussian $g_i$:

- Mean: $\mu_i \in \mathbb{R}^3$
- Covariance (or scales+rotation): $\Sigma_i$ (or $(s_i, R_i)$)
- Opacity: $\alpha_i$
- Appearance: either RGB $c_i$ or feature coefficients $f_i$

### 3.2 Camera Embedding

Goal: condition appearance and/or update rules on camera properties to improve generalization across viewpoints and handle camera-specific artifacts.

Define an embedding vector $e_t$ built from camera parameters:

- Minimal: $e_t = \text{MLP}([\mathrm{vec}(K_t), \mathrm{vec}(R_t), t_t])$
- With uncertainty: append diag of $\Sigma_{T_t}$ and stats from $\sigma_{D_t}$.

Use $e_t$ in one of these (choose the simplest that meets your needs):

1. **Appearance conditioning**: $c_i(t) = \text{Head}([f_i, e_t])$ so color depends on camera.
2. **Update conditioning**: optimizer step sizes or per-Gaussian learning rates depend on $e_t$.
3. **Visibility / opacity conditioning**: small MLP predicts $\Delta\alpha_i$ given $e_t$.

### 3.3 Pl\"ucker Rays as Geometry Input

For each pixel $u = (x, y)$, construct the camera ray in world coordinates.

Let camera center be $o_t \in \mathbb{R}^3$ and direction $d_{t,u} \in \mathbb{R}^3$, then the Pl\"ucker line is:

$$
\ell_{t,u} = (d_{t,u},\; m_{t,u}), \quad m_{t,u} = o_t \times d_{t,u}.
$$

Use cases:

- **Ray-conditioned updates**: when updating Gaussians using residuals from pixel reprojection, include $(d, m)$ (or derived invariants) so the update function is geometry-aware.
- **Ray–Gaussian association**: compute approximate intersection / contribution of each Gaussian along rays to select which Gaussians to update.

### 3.4 Optimization Targets

Given a renderer $\mathcal{R}(\mathcal{G}, K_t, T_t)$ producing $\hat{I}_t$:

- Photometric: $\|\hat{I}_t - I_t\|_1$ or robust Charbonnier
- Depth consistency: render depth $\hat{D}_t$ and match VGGT depth $D_t$ (weighted by $\sigma_{D_t}$)
- Regularizers: Gaussian size/anisotropy, opacity sparsity, temporal smoothness of appearance

## 4) Stage C — Active Objective Function

The active module selects an action $a_t$ to maximize expected gain under constraints.

### 4.1 Action Space (keep minimal)

Define actions that you can actually execute in your streaming system:

- **Compute actions**: run full update vs. partial update; choose subset of pixels/tiles; choose number of optimization steps; choose which Gaussians to refine.
- **Transmit actions**: choose bitrate; choose which Gaussian deltas to send; choose quantization level; choose keyframe vs. delta.

### 4.2 Objective

Let $\Delta \mathcal{L}(a_t)$ be the expected loss reduction (rendering error + depth error) from action $a_t$.

Let costs be:

- Compute cost: $C_{\text{flops}}(a_t)$ (proxy for latency/energy)
- Bandwidth cost: $C_{\text{bw}}(a_t)$ (bits to transmit)

One practical active objective:

$$
a_t^* = \arg\max_{a_t \in \mathcal{A}} \; \mathbb{E}[\Delta \mathcal{L}(a_t)] \,/\, (\lambda_f C_{\text{flops}}(a_t) + \lambda_b C_{\text{bw}}(a_t) + \epsilon)
$$

Where $\lambda_f, \lambda_b$ are tunable weights (or set from measured budgets).

### 4.3 Estimating Expected Gain

Use cheap proxies (avoid expensive rollouts):

- **Uncertainty-driven**: prioritize frames/regions with high $\sigma_{D_t}$ or high pose uncertainty $\Sigma_{T_t}$.
- **Residual-driven**: maintain a rolling map of rendering residuals; choose tiles with highest residual per cost.
- **Coverage-driven**: reward novel viewpoint coverage (e.g., change in ray directions distribution).

## 5) Stage D — RDO (Rate–Distortion Optimization)

RDO decides how to represent and transmit GS state (or updates) under bitrate constraints while minimizing distortion.

### 5.1 What to Encode

Prefer **delta coding** over full state:

- New Gaussians (birth)
- Removed Gaussians (death)
- Parameter updates: $\Delta\mu, \Delta\Sigma, \Delta\alpha, \Delta f$

### 5.2 Distortion

Define distortion as an estimate of visual/geometry error introduced by compression:

- Image-space: expected increase in photometric loss on a validation set of rays
- Depth/geometry: expected increase in depth residual

In practice, compute an importance score per Gaussian or per parameter block:

- Importance $w_i$ could be accumulated contribution to rendered pixels (alpha-weighted), or sensitivity approximations.

### 5.3 Rate Model

Rate can be approximated by:

- Number of updated Gaussians $N_u$
- Bits per parameter after quantization
- Overhead for indices and entropy coding

### 5.4 Lagrangian RDO

Standard Lagrangian formulation:

$$
\min_{q \in \mathcal{Q}} \; D(q) + \lambda R(q)
$$

Where $q$ is a set of coding decisions (which deltas to send + quantization levels + pruning decisions).

Concrete decision examples:

- **Prune**: drop low-importance updates to meet target bitrate.
- **Quantize**: coarser quantization for low-importance parameters.
- **Prioritize**: send position/opacity before high-frequency appearance.

### 5.5 Streaming Loop With RDO

- Maintain a bitrate budget per second (or per frame).
- Allocate bits to the highest importance updates until budget is met.
- If budget is exceeded: increase quantization or drop low-value updates.

## 6) Stage E — FLOPs + Bandwidth Monitor

This module measures and exposes real-time signals to the active controller and RDO.

### 6.1 Metrics to Track

- **Latency**: end-to-end and per-stage (VGGT, render, optimize, encode, transmit)
- **FLOPs proxy**: estimated FLOPs per stage (or GPU time if FLOPs not directly available)
- **Bandwidth**: bytes/sec, bits/frame, packet loss (if available)
- **Quality**: rendering loss on a held-out set of rays or periodic keyframes

### 6.2 Control Hooks

Use thresholds / moving averages to adjust knobs:

- If bandwidth drops: increase $\lambda$ in RDO, reduce update frequency, transmit only high-importance deltas.
- If latency spikes: reduce optimization steps, reduce tile set, simplify update rule.
- If quality degrades: trigger keyframe refresh or increase update budget temporarily.

## 7) Reference Pseudocode (Single Step)

```
state: GS model G, bitrate controller, compute controller

for each incoming frame I_t:
	# A) VGGT
	D_t, sigma_D_t, K_t, T_t, Sigma_T_t, feat_t = vggt_infer(I_t)

	# B) Predict/Render
	I_hat_t, aux = render(G, K_t, T_t)
	residuals = robust_error(I_hat_t, I_t)

	# C) Active action selection
	a_t = select_action(residuals, sigma_D_t, Sigma_T_t, budgets)

	# B) Update GS (camera embedding + Plücker rays)
	e_t = camera_embed(K_t, T_t, Sigma_T_t)
	rays = plucker_rays(K_t, T_t)          # (d, m)
	deltas = optimize_gs(G, I_t, D_t, e_t, rays, action=a_t)

	# D) RDO + encode
	coded = rdo_encode(deltas, target_bitrate, lambda)
	transmit(coded)

	# E) Monitor
	update_metrics(latency, bw, quality)
	update_budgets_and_lambdas(metrics)
```

## 8) Minimal Deliverables Checklist

1. VGGT wrapper returning $(D_t, K_t, T_t)$ + uncertainty.
2. GS model update path that consumes camera embedding $e_t$.
3. Pl\"ucker rays utility + integration point in update/selection.
4. Active controller implementing a gain-per-cost rule.
5. RDO encoder for GS deltas with Lagrangian tuning.
6. FLOPs/bandwidth monitor feeding back into (4) and (5).

