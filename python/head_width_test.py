import argparse
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2

# Ensure Matplotlib cache uses a writable directory before importing mediapipe.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class HeadWidthResult:
    width_px: float
    width_ratio: float
    category: str
    bbox: Tuple[int, int, int, int]


def categorize_width(width_ratio: float) -> str:
    # Heuristic buckets; adjust after collecting samples.
    if width_ratio < 0.32:
        return "small"
    if width_ratio < 0.40:
        return "medium"
    return "large"


def pick_largest_detection(detections):
    best = None
    best_area = -1.0
    for det in detections:
        bb = det.bounding_box
        w = bb.width
        h = bb.height
        area = w * h
        if area > best_area:
            best_area = area
            best = det
    return best


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def ensure_model(model_path: str) -> None:
    if os.path.exists(model_path):
        return
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:  # pragma: no cover - depends on network
        raise SystemExit(
            f"Failed to download model to {model_path}. "
            f"Download manually from {MODEL_URL} and retry."
        ) from exc


def measure_head_width(image_bgr, model_path: str) -> Optional[HeadWidthResult]:
    image_h, image_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    with vision.FaceDetector.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)

    if not detection_result.detections:
        return None

    det = pick_largest_detection(detection_result.detections)
    bb = det.bounding_box
    x = max(int(bb.origin_x), 0)
    y = max(int(bb.origin_y), 0)
    w = int(bb.width)
    h = int(bb.height)
    if x + w > image_w:
        w = max(0, image_w - x)
    if y + h > image_h:
        h = max(0, image_h - y)

    width_px = float(w)
    width_ratio = width_px / float(image_w)
    category = categorize_width(width_ratio)
    return HeadWidthResult(width_px, width_ratio, category, (x, y, w, h))


def annotate(image_bgr, result: HeadWidthResult):
    x, y, w, h = result.bbox
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 220, 0), 2)
    label = f"width_px={result.width_px:.1f} ratio={result.width_ratio:.3f} size={result.category}"
    cv2.putText(
        image_bgr,
        label,
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 0),
        2,
    )


def main():
    parser = argparse.ArgumentParser(description="Head width detection on a static image.")
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "--out",
        default="head_width_annotated.jpg",
        help="Output image path (default: head_width_annotated.jpg)",
    )
    parser.add_argument(
        "--model",
        default=os.path.join("models", "blaze_face_short_range.tflite"),
        help="Path to face detection model (default: models/blaze_face_short_range.tflite)",
    )
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")

    ensure_model(args.model)
    result = measure_head_width(image_bgr, args.model)
    if result is None:
        raise SystemExit("No face detected in the image.")

    annotate(image_bgr, result)
    cv2.imwrite(args.out, image_bgr)
    print(
        f"width_px={result.width_px:.1f}, width_ratio={result.width_ratio:.3f}, category={result.category}"
    )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
