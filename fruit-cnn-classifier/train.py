from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATASET_DIR = ARTIFACTS / "dataset"
MODEL_PATH = ARTIFACTS / "fruit_cnn_model.npz"
PREDICTIONS_PATH = ARTIFACTS / "sample_predictions.png"

CLASSES = ["apple", "banana", "orange"]
IMAGE_SIZE = 32


def draw_apple(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    cx = rng.randint(14, 18)
    cy = rng.randint(15, 19)
    rx = rng.randint(8, 11)
    ry = rng.randint(8, 11)
    color = (rng.randint(185, 235), rng.randint(20, 55), rng.randint(25, 65))
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color)
    draw.ellipse((cx - 4, cy - ry - 2, cx + 3, cy - ry + 3), fill=(170, 30, 45))
    draw.line((cx, cy - ry - 2, cx + 2, cy - ry - 7), fill=(83, 50, 28), width=2)
    draw.ellipse((cx + 3, cy - ry - 7, cx + 9, cy - ry - 2), fill=(56, 135, 56))
    draw.ellipse((cx - 4, cy - 4, cx - 1, cy - 1), fill=(255, 130, 130))


def draw_orange(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    cx = rng.randint(14, 18)
    cy = rng.randint(15, 18)
    r = rng.randint(8, 11)
    color = (rng.randint(220, 255), rng.randint(105, 155), rng.randint(5, 35))
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    for _ in range(18):
        px = rng.randint(cx - r + 2, cx + r - 2)
        py = rng.randint(cy - r + 2, cy + r - 2)
        if (px - cx) ** 2 + (py - cy) ** 2 < (r - 1) ** 2:
            dot = (255, rng.randint(150, 190), rng.randint(45, 75))
            draw.point((px, py), fill=dot)
    draw.arc((cx - r + 3, cy - r + 3, cx + r - 2, cy + r - 2), 210, 285, fill=(255, 190, 90), width=1)


def draw_banana(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    shift_x = rng.randint(-2, 2)
    shift_y = rng.randint(-1, 2)
    outer = [
        (6 + shift_x, 21 + shift_y),
        (11 + shift_x, 8 + shift_y),
        (26 + shift_x, 8 + shift_y),
        (28 + shift_x, 14 + shift_y),
        (20 + shift_x, 24 + shift_y),
        (9 + shift_x, 25 + shift_y),
    ]
    inner = [
        (10 + shift_x, 19 + shift_y),
        (14 + shift_x, 13 + shift_y),
        (25 + shift_x, 12 + shift_y),
        (24 + shift_x, 15 + shift_y),
        (18 + shift_x, 21 + shift_y),
        (10 + shift_x, 22 + shift_y),
    ]
    draw.polygon(outer, fill=(245, rng.randint(205, 235), rng.randint(25, 65)))
    draw.polygon(inner, fill=(248, 248, 218))
    draw.line(outer + [outer[0]], fill=(165, 125, 35), width=1)
    draw.ellipse((5 + shift_x, 20 + shift_y, 9 + shift_x, 24 + shift_y), fill=(95, 63, 28))
    draw.ellipse((25 + shift_x, 8 + shift_y, 29 + shift_x, 12 + shift_y), fill=(95, 63, 28))


def make_image(label: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    bg = (
        rng.randint(225, 248),
        rng.randint(230, 250),
        rng.randint(225, 248),
    )
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), bg)
    draw = ImageDraw.Draw(image)

    for _ in range(rng.randint(0, 3)):
        x = rng.randint(0, IMAGE_SIZE - 1)
        y = rng.randint(0, IMAGE_SIZE - 1)
        draw.point((x, y), fill=(rng.randint(200, 255), rng.randint(200, 255), rng.randint(200, 255)))

    if label == "apple":
        draw_apple(draw, rng)
    elif label == "banana":
        draw_banana(draw, rng)
    elif label == "orange":
        draw_orange(draw, rng)
    else:
        raise ValueError(f"Unknown label: {label}")

    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.45)))
    return image


def generate_dataset(samples_per_class: int = 90, test_per_class: int = 24) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for split, count in [("train", samples_per_class), ("test", test_per_class)]:
        for class_index, label in enumerate(CLASSES):
            folder = DATASET_DIR / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                seed = 10_000 * class_index + (1000 if split == "test" else 0) + i
                make_image(label, seed).save(folder / f"{label}_{i:03d}.png")


def load_dataset(split: str) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[Path] = []

    for label_index, label in enumerate(CLASSES):
        for path in sorted((DATASET_DIR / split / label).glob("*.png")):
            image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            images.append(np.asarray(image, dtype=np.float32) / 255.0)
            labels.append(label_index)
            paths.append(path)

    x = np.stack(images)
    y = np.asarray(labels, dtype=np.int64)
    return x, y, paths


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((len(y), num_classes), dtype=np.float32)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


@dataclass
class Cache:
    x: np.ndarray
    conv: np.ndarray
    relu: np.ndarray
    pool: np.ndarray
    flat: np.ndarray
    hidden: np.ndarray
    probs: np.ndarray
    pool_argmax: np.ndarray


class SmallCNN:
    def __init__(self, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.conv_w = rng.normal(0, 0.08, size=(8, 3, 5, 5)).astype(np.float32)
        self.conv_b = np.zeros(8, dtype=np.float32)
        self.fc1_w = rng.normal(0, 0.08, size=(8 * 14 * 14, 48)).astype(np.float32)
        self.fc1_b = np.zeros(48, dtype=np.float32)
        self.fc2_w = rng.normal(0, 0.08, size=(48, len(CLASSES))).astype(np.float32)
        self.fc2_b = np.zeros(len(CLASSES), dtype=np.float32)

    def conv2d(self, x: np.ndarray) -> np.ndarray:
        n, h, w, channels = x.shape
        out = np.zeros((n, 28, 28, 8), dtype=np.float32)
        x_chw = np.transpose(x, (0, 3, 1, 2))
        for row in range(28):
            for col in range(28):
                patch = x_chw[:, :, row : row + 5, col : col + 5]
                out[:, row, col, :] = np.tensordot(patch, self.conv_w, axes=([1, 2, 3], [1, 2, 3]))
        return out + self.conv_b

    @staticmethod
    def max_pool2x2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, h, w, channels = x.shape
        reshaped = x.reshape(n, h // 2, 2, w // 2, 2, channels)
        windows = reshaped.transpose(0, 1, 3, 5, 2, 4).reshape(n, h // 2, w // 2, channels, 4)
        argmax = np.argmax(windows, axis=-1)
        pooled = np.max(windows, axis=-1)
        return pooled, argmax

    @staticmethod
    def max_pool2x2_backward(grad: np.ndarray, argmax: np.ndarray) -> np.ndarray:
        n, h, w, channels = grad.shape
        dx_windows = np.zeros((n, h, w, channels, 4), dtype=np.float32)
        np.put_along_axis(dx_windows, argmax[..., None], grad[..., None], axis=-1)
        dx = dx_windows.reshape(n, h, w, channels, 2, 2).transpose(0, 1, 4, 2, 5, 3)
        return dx.reshape(n, h * 2, w * 2, channels)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> Cache:
        conv = self.conv2d(x)
        relu = np.maximum(conv, 0)
        pool, pool_argmax = self.max_pool2x2(relu)
        flat = pool.reshape(len(x), -1)
        hidden = np.maximum(flat @ self.fc1_w + self.fc1_b, 0)
        logits = hidden @ self.fc2_w + self.fc2_b
        probs = self.softmax(logits)
        return Cache(x, conv, relu, pool, flat, hidden, probs, pool_argmax)

    def backward(self, cache: Cache, y: np.ndarray, lr: float) -> float:
        batch_size = len(y)
        targets = one_hot(y, len(CLASSES))
        loss = -np.mean(np.sum(targets * np.log(cache.probs + 1e-8), axis=1))

        dlogits = (cache.probs - targets) / batch_size
        dfc2_w = cache.hidden.T @ dlogits
        dfc2_b = dlogits.sum(axis=0)

        dhidden = dlogits @ self.fc2_w.T
        dhidden[cache.hidden <= 0] = 0
        dfc1_w = cache.flat.T @ dhidden
        dfc1_b = dhidden.sum(axis=0)

        dflat = dhidden @ self.fc1_w.T
        dpool = dflat.reshape(cache.pool.shape)
        drelu = self.max_pool2x2_backward(dpool, cache.pool_argmax)
        dconv = drelu
        dconv[cache.conv <= 0] = 0

        dconv_w = np.zeros_like(self.conv_w)
        dconv_b = dconv.sum(axis=(0, 1, 2))
        x_chw = np.transpose(cache.x, (0, 3, 1, 2))
        for row in range(28):
            for col in range(28):
                patch = x_chw[:, :, row : row + 5, col : col + 5]
                dconv_w += np.tensordot(dconv[:, row, col, :], patch, axes=([0], [0]))

        for param, grad in [
            (self.conv_w, dconv_w),
            (self.conv_b, dconv_b),
            (self.fc1_w, dfc1_w),
            (self.fc1_b, dfc1_b),
            (self.fc2_w, dfc2_w),
            (self.fc2_b, dfc2_b),
        ]:
            np.clip(grad, -1.0, 1.0, out=grad)
            param -= lr * grad

        return float(loss)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).probs.argmax(axis=1)

    def save(self, path: Path) -> None:
        np.savez(
            path,
            conv_w=self.conv_w,
            conv_b=self.conv_b,
            fc1_w=self.fc1_w,
            fc1_b=self.fc1_b,
            fc2_w=self.fc2_w,
            fc2_b=self.fc2_b,
            classes=np.asarray(CLASSES),
        )


def accuracy(model: SmallCNN, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(x) == y))


def iterate_batches(x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    indices = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        batch = indices[start : start + batch_size]
        yield x[batch], y[batch]


def train() -> SmallCNN:
    generate_dataset()
    x_train, y_train, _ = load_dataset("train")
    x_test, y_test, _ = load_dataset("test")

    model = SmallCNN()
    rng = np.random.default_rng(42)
    epochs = 12
    batch_size = 18
    lr = 0.035

    for epoch in range(1, epochs + 1):
        losses = []
        for xb, yb in iterate_batches(x_train, y_train, batch_size, rng):
            cache = model.forward(xb)
            losses.append(model.backward(cache, yb, lr=lr))
        train_acc = accuracy(model, x_train, y_train)
        test_acc = accuracy(model, x_test, y_test)
        print(
            f"epoch {epoch:02d}/{epochs} "
            f"loss={np.mean(losses):.4f} train_acc={train_acc:.3f} test_acc={test_acc:.3f}"
        )

    model.save(MODEL_PATH)
    create_prediction_grid(model)
    return model


def create_prediction_grid(model: SmallCNN) -> None:
    x_test, y_test, paths = load_dataset("test")
    sample_indices = []
    for class_index in range(len(CLASSES)):
        sample_indices.extend(np.where(y_test == class_index)[0][:4].tolist())

    thumbs = []
    predictions = model.predict(x_test[sample_indices])
    for idx, pred in zip(sample_indices, predictions):
        image = Image.open(paths[idx]).convert("RGB").resize((96, 96), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (120, 126), "white")
        canvas.paste(image, (12, 6))
        draw = ImageDraw.Draw(canvas)
        actual = CLASSES[y_test[idx]]
        predicted = CLASSES[int(pred)]
        color = (20, 125, 55) if actual == predicted else (190, 45, 40)
        draw.text((8, 105), f"{predicted}", fill=color)
        thumbs.append(canvas)

    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    grid = Image.new("RGB", (cols * 120, rows * 126), "white")
    for i, thumb in enumerate(thumbs):
        grid.paste(thumb, ((i % cols) * 120, (i // cols) * 126))
    grid.save(PREDICTIONS_PATH)


if __name__ == "__main__":
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    train()
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved sample predictions to {PREDICTIONS_PATH}")
