from pathlib import Path
import urllib.request

BASE = Path(__file__).resolve().parent
CHECKPOINT = BASE / "output/60.pth"
CHECKPOINT_URL = (
    "https://media.githubusercontent.com/media/"
    "danieletukudo/Human-detection-in-forest/main/checkpoint_best_total.pth"
)

THRESHOLD = 0.5
CLASSES = [
    "animal",
    "person",
    "cow",
]

_model = None


def ensure_checkpoint() -> Path:
    if CHECKPOINT.exists() and CHECKPOINT.stat().st_size > 1_000_000:
        with CHECKPOINT.open("rb") as f:
            if not f.read(20).startswith(b"version "):
                return CHECKPOINT

    print(f"Downloading checkpoint to {CHECKPOINT} …", flush=True)
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT)
    size = CHECKPOINT.stat().st_size
    if size < 1_000_000:
        raise RuntimeError(f"Checkpoint download looks wrong ({size} bytes)")
    print(f"Checkpoint ready ({size} bytes)", flush=True)
    return CHECKPOINT


def get_model():
    global _model
    if _model is None:
        from rfdetr import RFDETRNano

        path = ensure_checkpoint()
        print("Loading RF-DETR model…", flush=True)
        _model = RFDETRNano(
            pretrain_weights=str(path),
            trust_checkpoint=True,
        )
        _model.optimize_for_inference()
        print("Model ready.", flush=True)
    return _model


def run_detection(video_path: str, output_path: str) -> str:
    import supervision as sv

    model = get_model()
    video_info = sv.VideoInfo.from_video_path(video_path)

    text_scale = sv.calculate_optimal_text_scale(
        resolution_wh=video_info.resolution_wh
    )
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    color = sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff66ff", "#3399ff",
        "#ff66b2", "#ff8080", "#b266ff", "#9999ff",
        "#66ffff", "#33ff99", "#66ff66", "#99ff00",
    ])
    bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
    )

    with sv.VideoSink(output_path, video_info) as sink:
        for frame in sv.get_video_frames_generator(video_path):
            detections = model.predict(frame, threshold=THRESHOLD)
            labels = [
                f"{CLASSES[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(
                    detections.class_id,
                    detections.confidence,
                )
            ]
            annotated = bbox_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels)
            sink.write_frame(annotated)

    return output_path
