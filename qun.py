from pathlib import Path
import torch

CHECKPOINT = Path("checkpoint_best_total.pth")
OUT = Path("output/small_checkpoint_best_total.pth")  # or checkpoint_best_total_fp16.pth

raw = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
state = raw["model"]

fp16_state = {
    k: (v.half() if torch.is_floating_point(v) else v)
    for k, v in state.items()
}

torch.save(
    {
        "model": fp16_state,
        "args": raw.get("args"),
        "model_name": raw.get("model_name", "RFDETRNano"),
        "rfdetr_version": raw.get("rfdetr_version"),
    },
    OUT,
)

print(f"Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")