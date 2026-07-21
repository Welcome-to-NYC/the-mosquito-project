"""Export the W11 distilled student to a C header for ESP32 deployment.

The student is small enough (2,443 params, ~9.5 KB fp32 weights) that
hand-coded inference in C++ is simpler than dragging in TFLite Micro:

  Conv1D(1 -> 8,  k=7) + BN + ReLU + MaxPool(2)   # 1024 -> 512
  Conv1D(8 -> 16, k=5) + BN + ReLU + MaxPool(2)   # 512  -> 256
  Conv1D(16 -> 24, k=3) + BN + ReLU + MaxPool(2)  # 256  -> 128
  AdaptiveAvgPool1d(1) -> Flatten                 # -> 24
  Dropout (no-op at inference)
  Linear(24 -> 16) + ReLU
  Linear(16 -> 3)

This script:
1) Loads experiments/exp_cnn1d_tiny_distilled/best.pt
2) Folds BatchNorm into the preceding Conv (standard inference optimization)
3) Picks N sample windows from data/processed/test.npz that span all classes
4) Runs the fused PyTorch model on those samples to get reference outputs
5) Emits firmware/wingbeat_inference/model_weights.h with everything bundled

The ESP32 firmware then runs its own inference on the same samples and
compares against the embedded expected outputs — a self-checking deploy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.cnn_1d import CNN1D  # noqa: E402

STUDENT_CKPT = ROOT / "experiments" / "exp_cnn1d_tiny_distilled" / "best.pt"
TEST_NPZ = ROOT / "data" / "processed" / "test.npz"
OUT_HEADER = ROOT / "firmware" / "wingbeat_inference" / "model_weights.h"

# Architecture must match the student's training config (configs/cnn1d_tiny.yaml).
STUDENT_KWARGS = dict(
    n_classes=3,
    in_channels=1,
    channels=[8, 16, 24],
    kernel_sizes=[7, 5, 3],
    fc_hidden=16,
    dropout=0.2,
)


def fold_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> tuple[np.ndarray, np.ndarray]:
    """Fold BatchNorm parameters into a preceding Conv layer.

    Effective output channel c after fusion:
        w_fused[c] = bn.weight[c] / sqrt(bn.var[c] + eps) * conv.weight[c]
        b_fused[c] = bn.weight[c] / sqrt(bn.var[c] + eps) * (conv.bias[c] - bn.running_mean[c]) + bn.bias[c]

    Lets inference skip the BN step entirely.
    """
    w = conv.weight.detach().cpu().numpy()
    b = (conv.bias.detach().cpu().numpy()
         if conv.bias is not None else np.zeros(w.shape[0], dtype=np.float32))
    gamma = bn.weight.detach().cpu().numpy()
    beta = bn.bias.detach().cpu().numpy()
    mean = bn.running_mean.detach().cpu().numpy()
    var = bn.running_var.detach().cpu().numpy()
    eps = bn.eps

    scale = gamma / np.sqrt(var + eps)
    w_fused = w * scale[:, None, None]
    b_fused = beta + scale * (b - mean)
    return w_fused.astype(np.float32), b_fused.astype(np.float32)


def emit_c_array(name: str, arr: np.ndarray, fmt: str = "{:.7g}f") -> str:
    """Render a numpy array as a flat `static const float NAME[] = { ... };`."""
    flat = arr.flatten()
    body = ",\n  ".join(
        ", ".join(fmt.format(float(x)) for x in flat[i : i + 8])
        for i in range(0, len(flat), 8)
    )
    return f"static const float {name}[{len(flat)}] = {{\n  {body}\n}};\n"


def main() -> int:
    print("== loading model")
    model = CNN1D(**STUDENT_KWARGS).eval()
    state = torch.load(STUDENT_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   params: {n_params:,}")

    # Fold BN into each conv block.
    print("== folding BN into conv weights")
    fused = []
    for i, block in enumerate(model.blocks):
        w, b = fold_bn(block.conv, block.bn)
        print(f"   block {i}: conv weight {w.shape}, bias {b.shape}")
        fused.append((w, b))

    fc1_w = model.fc1.weight.detach().cpu().numpy().astype(np.float32)
    fc1_b = model.fc1.bias.detach().cpu().numpy().astype(np.float32)
    fc2_w = model.fc2.weight.detach().cpu().numpy().astype(np.float32)
    fc2_b = model.fc2.bias.detach().cpu().numpy().astype(np.float32)

    # Pick representative test samples spanning all classes.
    print("== picking test samples")
    z = np.load(TEST_NPZ, allow_pickle=True)
    X, y, classes = z["X"], z["y"], list(z["classes"])
    rng = np.random.default_rng(42)
    sample_idx: list[int] = []
    for cls in range(len(classes)):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx):
            # Sample 2 per class.
            picks = rng.choice(cls_idx, size=min(2, len(cls_idx)), replace=False)
            sample_idx.extend(int(i) for i in picks)
    sample_X = X[sample_idx].astype(np.float32)
    sample_y = y[sample_idx].astype(np.int64)
    print(f"   {len(sample_idx)} samples chosen, shape {sample_X.shape}")

    # Reference outputs from the *unmodified* PyTorch model so the firmware
    # can self-check its hand-coded inference.
    print("== running reference inference")
    with torch.no_grad():
        x_t = torch.as_tensor(sample_X, dtype=torch.float32).unsqueeze(1)  # (N, 1, 1024)
        logits = model(x_t).numpy()
        probs = torch.softmax(torch.as_tensor(logits), dim=-1).numpy()
        preds = logits.argmax(axis=-1)
    print(f"   logits shape {logits.shape}")
    print(f"   preds vs true: pred={preds.tolist()}  true={sample_y.tolist()}")

    # Emit the C header.
    print(f"== writing {OUT_HEADER}")
    OUT_HEADER.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    parts.append("// AUTO-GENERATED by scripts/export_student_for_esp32.py — do not edit\n")
    parts.append("#pragma once\n\n")
    parts.append("// W11 distilled student — Conv1d-only, BN folded into conv weights.\n")
    parts.append("// Architecture: 3 conv blocks (in=1->8->16->24) -> AdaptiveAvgPool -> FC(24,16)+ReLU -> FC(16,3)\n\n")

    parts.append(f"#define MODEL_INPUT_LEN {sample_X.shape[1]}\n")
    parts.append(f"#define MODEL_N_CLASSES {len(classes)}\n")
    parts.append("#define MODEL_BLOCK0_IN_CH 1\n#define MODEL_BLOCK0_OUT_CH 8\n#define MODEL_BLOCK0_KERNEL 7\n")
    parts.append("#define MODEL_BLOCK1_IN_CH 8\n#define MODEL_BLOCK1_OUT_CH 16\n#define MODEL_BLOCK1_KERNEL 5\n")
    parts.append("#define MODEL_BLOCK2_IN_CH 16\n#define MODEL_BLOCK2_OUT_CH 24\n#define MODEL_BLOCK2_KERNEL 3\n")
    parts.append("#define MODEL_FC1_IN 24\n#define MODEL_FC1_OUT 16\n#define MODEL_FC2_IN 16\n#define MODEL_FC2_OUT 3\n\n")

    parts.append("static const char* const MODEL_CLASS_NAMES[MODEL_N_CLASSES] = {\n")
    for c in classes:
        parts.append(f'  "{c}",\n')
    parts.append("};\n\n")

    # Conv block weights (fused with BN).
    for i, (w, b) in enumerate(fused):
        parts.append(f"// block {i}: conv (out_ch={w.shape[0]}, in_ch={w.shape[1]}, k={w.shape[2]})\n")
        parts.append(emit_c_array(f"BLOCK{i}_WEIGHT", w))
        parts.append(emit_c_array(f"BLOCK{i}_BIAS", b))
        parts.append("\n")

    parts.append("// fc1: (out=16, in=24)\n")
    parts.append(emit_c_array("FC1_WEIGHT", fc1_w))
    parts.append(emit_c_array("FC1_BIAS", fc1_b))
    parts.append("\n// fc2: (out=3, in=16)\n")
    parts.append(emit_c_array("FC2_WEIGHT", fc2_w))
    parts.append(emit_c_array("FC2_BIAS", fc2_b))
    parts.append("\n")

    # Test samples.
    parts.append(f"#define NUM_TEST_SAMPLES {len(sample_idx)}\n")
    parts.append(emit_c_array("TEST_SAMPLES", sample_X))
    parts.append(f"static const int TEST_TRUE_LABELS[NUM_TEST_SAMPLES] = {{\n  ")
    parts.append(", ".join(str(int(v)) for v in sample_y))
    parts.append("\n};\n\n")
    parts.append(f"// Reference predictions from PyTorch (for self-check)\n")
    parts.append(f"static const int REF_PRED_LABELS[NUM_TEST_SAMPLES] = {{\n  ")
    parts.append(", ".join(str(int(v)) for v in preds))
    parts.append("\n};\n\n")
    parts.append(emit_c_array("REF_LOGITS", logits))
    parts.append(emit_c_array("REF_PROBS", probs))

    OUT_HEADER.write_text("".join(parts))

    size_kb = OUT_HEADER.stat().st_size / 1024.0
    weight_kb = (sum(w.nbytes + b.nbytes for w, b in fused) +
                 fc1_w.nbytes + fc1_b.nbytes + fc2_w.nbytes + fc2_b.nbytes) / 1024.0
    test_kb = sample_X.nbytes / 1024.0
    print(f"   header file: {size_kb:.1f} KB on disk")
    print(f"   weight bytes (fp32): {weight_kb:.1f} KB")
    print(f"   test sample bytes:   {test_kb:.1f} KB")
    print("== done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
