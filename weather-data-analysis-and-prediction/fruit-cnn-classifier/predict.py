from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "fruit_cnn_model.npz"
IMAGE_SIZE = 32


def conv2d(x: np.ndarray, conv_w: np.ndarray, conv_b: np.ndarray) -> np.ndarray:
    n, h, w, channels = x.shape
    out = np.zeros((n, 28, 28, len(conv_w)), dtype=np.float32)
    x_chw = np.transpose(x, (0, 3, 1, 2))
    for row in range(28):
        for col in range(28):
            patch = x_chw[:, :, row : row + 5, col : col + 5]
            out[:, row, col, :] = np.tensordot(patch, conv_w, axes=([1, 2, 3], [1, 2, 3]))
    return out + conv_b


def max_pool2x2(x: np.ndarray) -> np.ndarray:
    n, h, w, channels = x.shape
    reshaped = x.reshape(n, h // 2, 2, w // 2, 2, channels)
    windows = reshaped.transpose(0, 1, 3, 5, 2, 4).reshape(n, h // 2, w // 2, channels, 4)
    return np.max(windows, axis=-1)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return (np.asarray(image, dtype=np.float32) / 255.0)[None, ...]


def predict(image_path: Path) -> tuple[str, float, dict[str, float]]:
    weights = np.load(MODEL_PATH)
    classes = weights["classes"].tolist()
    x = load_image(image_path)

    conv = conv2d(x, weights["conv_w"], weights["conv_b"])
    relu = np.maximum(conv, 0)
    pool = max_pool2x2(relu)
    flat = pool.reshape(1, -1)
    hidden = np.maximum(flat @ weights["fc1_w"] + weights["fc1_b"], 0)
    probs = softmax(hidden @ weights["fc2_w"] + weights["fc2_b"])[0]

    best = int(np.argmax(probs))
    scores = {label: float(probs[i]) for i, label in enumerate(classes)}
    return classes[best], float(probs[best]), scores


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict.py path/to/image.png")
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run: python train.py")

    label, confidence, scores = predict(Path(sys.argv[1]))
    print(f"prediction={label} confidence={confidence:.3f}")
    for name, score in scores.items():
        print(f"{name}: {score:.3f}")
