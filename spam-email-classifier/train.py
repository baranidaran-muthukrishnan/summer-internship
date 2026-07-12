from __future__ import annotations

import argparse
from pathlib import Path

from src.spam_classifier import (
    MultinomialNaiveBayes,
    classification_report,
    load_csv,
    train_test_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a spam email classifier.")
    parser.add_argument("--data", type=Path, default=Path("data/sample_emails.csv"))
    parser.add_argument("--model", type=Path, default=Path("outputs/spam_model.pkl"))
    parser.add_argument("--report", type=Path, default=Path("outputs/evaluation.txt"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def format_report(accuracy: float, metrics: dict[str, dict[str, float]]) -> str:
    lines = [f"Accuracy: {accuracy:.3f}", "", "Label       Precision  Recall  F1     Support"]
    for label, scores in metrics.items():
        lines.append(
            f"{label:<11}"
            f"{scores['precision']:<11.3f}"
            f"{scores['recall']:<8.3f}"
            f"{scores['f1']:<7.3f}"
            f"{int(scores['support'])}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_csv(args.data)
    train_rows, test_rows = train_test_split(rows, test_size=args.test_size, seed=args.seed)

    model = MultinomialNaiveBayes(alpha=1.0).fit(train_rows)
    predictions = model.predict(text for _, text in test_rows)
    actual = [label for label, _ in test_rows]
    accuracy, metrics = classification_report(actual, predictions)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model)

    output = format_report(accuracy, metrics)
    args.report.write_text(output + "\n", encoding="utf-8")
    print(output)
    print(f"\nSaved model to {args.model}")
    print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
