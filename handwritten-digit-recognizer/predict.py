import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a handwritten digit from an image.")
    parser.add_argument("image", type=Path, help="Path to a single-digit image.")
    parser.add_argument("--model", type=Path, default=Path("outputs/mnist_cnn.keras"), help="Trained model path.")
    return parser.parse_args()


def prepare_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    if np.asarray(image).mean() > 127:
        image = ImageOps.invert(image)
    image = ImageOps.contain(image, (20, 20), method=Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    offset = ((28 - image.width) // 2, (28 - image.height) // 2)
    canvas.paste(image, offset)

    pixels = np.asarray(canvas, dtype="float32") / 255.0
    return pixels.reshape(1, 28, 28, 1)


def main() -> None:
    args = parse_args()
    model = keras.models.load_model(args.model)
    image = prepare_image(args.image)

    probabilities = model.predict(image, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])

    print(f"Predicted digit: {digit}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
