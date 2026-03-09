import json

import MinkowskiEngine as ME
import torch

from Swin3D.models.Swin3D import Swin3DUNet as RefSwin3DUNet

from src.model.encoder.gs_cube.swin3d_reimpl import Swin3DUNet as NewSwin3DUNet


def _default_kwargs():
    return dict(
        depths=[2, 2],
        channels=[64, 96],
        num_heads=[4, 6],
        window_sizes=[5, 5],
        quant_size=4,
        drop_path_rate=0.0,
        up_k=3,
        num_layers=2,
        num_classes=13,
        stem_transformer=True,
        first_down_stride=2,
        other_down_stride=2,
        upsample="linear_attn",
        knn_down=True,
        in_channels=32,
        cRSE="XYZ_RGB",
        fp16_mode=0,
        stem_norm="bn",
    )


def _build_inputs(device: torch.device, num_points: int = 12000):
    lattice_size = 64 ** 3
    if num_points > lattice_size:
        raise ValueError(f"num_points={num_points} exceeds the available unique lattice size {lattice_size}")
    flat = torch.randperm(lattice_size, device=device)[:num_points]
    x = flat // (64 * 64)
    y = (flat // 64) % 64
    z = flat % 64
    coords_xyz = torch.stack([x, y, z], dim=1)
    coords = torch.cat(
        [torch.zeros((coords_xyz.shape[0], 1), dtype=torch.int32, device=device), coords_xyz.int()],
        dim=1,
    )
    if torch.unique(coords, dim=0).shape[0] != coords.shape[0]:
        raise RuntimeError("Generated duplicate sparse coordinates in compare_swin3d_impl")
    feats = torch.randn(coords.shape[0], 32, device=device)
    rgb = torch.rand(coords.shape[0], 3, device=device) * 2.0 - 1.0
    if rgb.min() < -1.0 or rgb.max() > 1.0:
        raise RuntimeError("Generated RGB values outside [-1, 1] in compare_swin3d_impl")
    positions = torch.cat([coords.float(), rgb], dim=1)
    sp = ME.SparseTensor(features=feats, coordinates=coords, device=device)
    coords_sp = ME.SparseTensor(
        features=positions,
        coordinate_map_key=sp.coordinate_map_key,
        coordinate_manager=sp.coordinate_manager,
        device=device,
    )
    return sp, coords_sp


def _run_stages(model, sp, coords_sp):
    outputs = {}
    sp_stack = []
    coords_stack = []
    sp_cur = model.stem_layer(sp)
    coords_cur = coords_sp
    outputs["stem"] = sp_cur.F.detach().clone()
    outputs["stem_coords"] = sp_cur.C.detach().clone()

    for index, layer in enumerate(model.layers):
        coords_stack.append(coords_cur)
        sp_out, sp_down, coords_cur = layer(sp_cur, coords_cur)
        outputs[f"layer{index}_out"] = sp_out.F.detach().clone()
        outputs[f"layer{index}_out_coords"] = sp_out.C.detach().clone()
        outputs[f"layer{index}_down"] = sp_down.F.detach().clone()
        outputs[f"layer{index}_down_coords"] = sp_down.C.detach().clone()
        sp_stack.append(sp_out)
        sp_cur = sp_down

    sp_dec = sp_stack.pop()
    coords_dec = coords_stack.pop()
    outputs["decode_input"] = sp_dec.F.detach().clone()
    outputs["decode_input_coords"] = sp_dec.C.detach().clone()

    for index, upsample in enumerate(model.upsamples):
        sp_skip = sp_stack.pop()
        coords_skip = coords_stack.pop()
        sp_dec = upsample(sp_dec, coords_dec, sp_skip, coords_skip)
        coords_dec = coords_skip
        outputs[f"upsample{index}"] = sp_dec.F.detach().clone()
        outputs[f"upsample{index}_coords"] = sp_dec.C.detach().clone()

    outputs["classifier"] = model.classifier(sp_dec.F).detach().clone()
    return outputs


def _compare(a, b):
    out = {}
    for key in a:
        if key.endswith("_coords"):
            out[key] = {"equal": bool(torch.equal(a[key], b[key]))}
            continue
        diff = (a[key] - b[key]).abs()
        out[key] = {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "allclose_1e-6": bool(torch.allclose(a[key], b[key], atol=1e-6, rtol=1e-6)),
            "allclose_1e-4": bool(torch.allclose(a[key], b[key], atol=1e-4, rtol=1e-4)),
            "allclose_1e-2": bool(torch.allclose(a[key], b[key], atol=1e-2, rtol=1e-2)),
            "has_nan_a": bool(torch.isnan(a[key]).any()),
            "has_nan_b": bool(torch.isnan(b[key]).any()),
        }
    return out


def main():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this comparison script")
    device = torch.device("cuda")
    kwargs = _default_kwargs()

    ref_a = RefSwin3DUNet(**kwargs).to(device).eval()
    ref_b = RefSwin3DUNet(**kwargs).to(device).eval()
    ref_b.load_state_dict(ref_a.state_dict(), strict=True)
    new_a = NewSwin3DUNet(**kwargs).to(device).eval()
    new_a.load_state_dict(ref_a.state_dict(), strict=True)
    new_b = NewSwin3DUNet(**kwargs).to(device).eval()
    new_b.load_state_dict(ref_a.state_dict(), strict=True)

    sp, coords_sp = _build_inputs(device)

    with torch.no_grad():
        out_ref_a = _run_stages(ref_a, sp, coords_sp)
        out_ref_b = _run_stages(ref_b, sp, coords_sp)
        out_new_a = _run_stages(new_a, sp, coords_sp)
        out_new_b = _run_stages(new_b, sp, coords_sp)

    result = {
        "ref_vs_ref": _compare(out_ref_a, out_ref_b),
        "ref_vs_new": _compare(out_ref_a, out_new_a),
        "new_vs_new": _compare(out_new_a, out_new_b),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()