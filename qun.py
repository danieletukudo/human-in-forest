"""
Quantize RF-DETR Nano checkpoint.

Supported modes:
  - onnx_int8  : export ONNX then dynamic INT8 (weights) via onnxruntime
  - tflite_int8: official RF-DETR TFLite dynamic-range INT8 export
  - fp16       : save FP16 PyTorch state dict (smaller .pth, still PyTorch)

Install extras if needed:
  pip install "rfdetr[onnx]" onnxruntime
  # for tflite_int8 also:
  pip install "rfdetr[tflite]"
"""

from pathlib import Path

from rfdetr import RFDETRNano

BASE = Path(__file__).resolve().parent
CHECKPOINT = BASE / "checkpoint_best_total.pth"
OUTPUT_DIR = BASE / "output"

# "onnx_int8" | "tflite_int8" | "fp16"
MODE = "onnx_int8"

# Optional: folder of representative images (helps TFLite int8 validation)
CALIBRATION_DIR = None  # e.g. str(BASE / "uploads")


def load_model() -> RFDETRNano:
    # Checkpoint is RFDETRNano with 2 classes (see model_name / class_names in .pth)
    return RFDETRNano(
        pretrain_weights=str(CHECKPOINT),
        num_classes=2,
    )


def quantize_onnx_int8(model: RFDETRNano) -> Path:
    """Export ONNX, then apply dynamic INT8 weight quantization."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    onnx_path = model.export(
        format="onnx",
        output_dir=str(OUTPUT_DIR),
        output_name="rfdetr-nano",
        verbose=False,
    )
    onnx_path = Path(onnx_path)
    int8_path = OUTPUT_DIR / "rfdetr-nano-int8.onnx"

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    return int8_path


def quantize_tflite_int8(model: RFDETRNano) -> Path:
    """Official RF-DETR path: ONNX -> TFLite dynamic-range INT8."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        format="tflite",
        quantization="int8",
        output_dir=str(OUTPUT_DIR),
        verbose=True,
    )
    if CALIBRATION_DIR:
        kwargs["calibration_data"] = CALIBRATION_DIR

    return Path(model.export(**kwargs))


def quantize_fp16(model: RFDETRNano) -> Path:
    """Save FP16 weights as a drop-in .pth for RFDETRNano(pretrain_weights=...)."""
    import torch

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "checkpoint_best_total_fp16.pth"

    raw = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = raw["model"]
    fp16_state = {
        k: (v.half() if torch.is_floating_point(v) else v) for k, v in state.items()
    }
    torch.save(
        {
            "model": fp16_state,
            "args": raw.get("args"),
            "model_name": raw.get("model_name", "RFDETRNano"),
            "rfdetr_version": raw.get("rfdetr_version"),
        },
        out,
    )
    return out


if __name__ == "__main__":
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")

    print(f"Loading {CHECKPOINT} as RFDETRNano ...")
    model = load_model()

    if MODE == "onnx_int8":
        out = quantize_onnx_int8(model)
    elif MODE == "tflite_int8":
        out = quantize_tflite_int8(model)
    elif MODE == "fp16":
        out = quantize_fp16(model)
    else:
        raise ValueError(f"Unknown MODE={MODE!r}")

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Saved quantized model: {out} ({size_mb:.1f} MB)")
