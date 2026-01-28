import numpy as np
import tensorflow as tf
import time
import tensorflow_hub as hub
from PIL import Image
import matplotlib.pyplot as plt

MODEL_URL = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2'

COCO_LABELS = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle", 
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
}

def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

def pil_to_tensor(pil_img):
    arr = np.array(pil_img, dtype = np.uint8)
    return tf.convert_to_tensor(arr)[tf.newaxis, ...]

def draw_detections(pil_img, boxes, scores, classes, score_thresh=0.3, max_dets=20):
    img = np.array(pil_img).copy()
    h, w = img.shape[:2]

    plt.figure(figsize=(12,8))
    plt.imshow(img)
    ax = plt.gca()
    
    shown = 0
    for i in range(min(len(scores), max_dets)):
        if scores[i] < score_thresh:
            continue

        y1, x1, y2, x2 = boxes[i]
        x1i, y1i, x2i, y2i = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

        rect = plt.Rectangle((x1i, y1i), x2i-x1i, y2i-y1i, fill=False, linewidth=2)
        ax.add_patch(rect)

        cls_id = int(classes[i])
        name = COCO_LABELS.get(cls_id, str(cls_id))
        ax.text(x1i, max(0, y1i-5), f"{name} {scores[i]:.2f}", bbox=dict(facecolor="white", alpha=0.7), fontsize=10)
        shown += 1

    plt.axis("off")
    plt.tight_layout()
    return shown

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an input image (jpg/png)")
    parser.add_argument("--out", default="detections.png", help="Output Visualization Image")
    parser.add_argument("--thresh", type=float, default=0.3, help="Score threshold")
    args = parser.parse_args()

    detector = hub.load(MODEL_URL)

    pil_img = load_image(args.image)
    inp = pil_to_tensor(pil_img)

    t0 = time.time()
    outputs = detector(inp)
    dt = time.time() - t0
    
    boxes = outputs["detection_boxes"][0].numpy()
    scores = outputs["detection_scores"][0].numpy()
    classes = outputs["detection_classes"][0].numpy()

    shown = draw_detections(pil_img, boxes, scores, classes, score_thresh=args.thresh)
    plt.title(f"Detections shown: {shown} / Inference: {dt*1000:.1f} ms")
    plt.savefig(args.out, dpi=200)
    print(f"Saved Visualixation to: {args.out}")

if __name__ == "__main__":
    main()
