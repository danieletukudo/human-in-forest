from pathlib import Path

from rfdetr import RFDETRNano
import supervision as sv

BASE = Path(__file__).resolve().parent
CHECKPOINT = BASE / "output" / "small_checkpoint_best_total.pth"

THRESHOLD = 0.5
CLASSES = [
    "animal",  # class 0
    "person",  # class 1
    "cow",     # class 2
]

_model = None


def get_model():
    global _model
    if _model is None:
        if not CHECKPOINT.exists():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
        _model = RFDETRNano(pretrain_weights=str(CHECKPOINT), num_classes=2)
        _model.optimize_for_inference()
    return _model


def run_detection(video_path: str, output_path: str) -> str:
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


if __name__ == "__main__":
    VIDEO_PATH = str(BASE / "uploads" / "input.mp4")
    OUTPUT_PATH = str(BASE / "outputs" / "output_detected.mp4")
    print(f"Done! Saved to: {run_detection(VIDEO_PATH, OUTPUT_PATH)}")
