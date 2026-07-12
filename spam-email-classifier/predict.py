from __future__ import annotations

import argparse
from pathlib import Path

from src.spam_classifier import MultinomialNaiveBayes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify an email as spam or ham.")
    parser.add_argument("text", help="Email content to classify.")
    parser.add_argument("--model", type=Path, default=Path("outputs/spam_model.pkl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = MultinomialNaiveBayes.load(args.model)
    probabilities = model.predict_proba_one(args.text)
    label = max(probabilities, key=probabilities.get)

    print(f"Prediction: {label}")
    for class_label, probability in sorted(probabilities.items()):
        print(f"{class_label}: {probability:.3f}")


if __name__ == "__main__":
    main()
