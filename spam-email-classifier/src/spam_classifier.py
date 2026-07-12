from __future__ import annotations

import csv
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_csv(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "label" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("CSV must contain 'label' and 'text' columns.")
        return [(row["label"].strip().lower(), row["text"].strip()) for row in reader]


def train_test_split(
    rows: list[tuple[str, str]], test_size: float = 0.25, seed: int = 42
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row)

    rng = random.Random(seed)
    train_rows: list[tuple[str, str]] = []
    test_rows: list[tuple[str, str]] = []

    for label_rows in grouped.values():
        rng.shuffle(label_rows)
        cutoff = max(1, int(round(len(label_rows) * test_size)))
        test_rows.extend(label_rows[:cutoff])
        train_rows.extend(label_rows[cutoff:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


@dataclass
class MultinomialNaiveBayes:
    alpha: float = 1.0
    class_log_prior_: dict[str, float] | None = None
    feature_log_prob_: dict[str, dict[str, float]] | None = None
    vocabulary_: set[str] | None = None
    class_token_totals_: dict[str, int] | None = None

    def fit(self, rows: Iterable[tuple[str, str]]) -> "MultinomialNaiveBayes":
        label_counts: Counter[str] = Counter()
        token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        vocabulary: set[str] = set()

        rows = list(rows)
        if not rows:
            raise ValueError("Cannot train on an empty dataset.")

        for label, text in rows:
            label_counts[label] += 1
            tokens = tokenize(text)
            token_counts[label].update(tokens)
            vocabulary.update(tokens)

        total_docs = sum(label_counts.values())
        vocab_size = max(1, len(vocabulary))
        self.class_log_prior_ = {
            label: math.log(count / total_docs) for label, count in label_counts.items()
        }
        self.class_token_totals_ = {
            label: sum(counts.values()) for label, counts in token_counts.items()
        }
        self.feature_log_prob_ = {}
        self.vocabulary_ = vocabulary

        for label, counts in token_counts.items():
            denominator = self.class_token_totals_[label] + self.alpha * vocab_size
            self.feature_log_prob_[label] = {
                token: math.log((counts[token] + self.alpha) / denominator)
                for token in vocabulary
            }

        return self

    def _check_fit(self) -> None:
        if (
            self.class_log_prior_ is None
            or self.feature_log_prob_ is None
            or self.vocabulary_ is None
            or self.class_token_totals_ is None
        ):
            raise ValueError("Model has not been fitted.")

    def predict_proba_one(self, text: str) -> dict[str, float]:
        self._check_fit()
        assert self.class_log_prior_ is not None
        assert self.feature_log_prob_ is not None
        assert self.vocabulary_ is not None
        assert self.class_token_totals_ is not None

        vocab_size = max(1, len(self.vocabulary_))
        token_counts = Counter(token for token in tokenize(text) if token in self.vocabulary_)
        log_scores: dict[str, float] = {}

        for label, prior in self.class_log_prior_.items():
            score = prior
            denominator = self.class_token_totals_[label] + self.alpha * vocab_size
            unseen_log_prob = math.log(self.alpha / denominator)
            for token, count in token_counts.items():
                score += count * self.feature_log_prob_[label].get(token, unseen_log_prob)
            log_scores[label] = score

        max_log = max(log_scores.values())
        exp_scores = {label: math.exp(score - max_log) for label, score in log_scores.items()}
        total = sum(exp_scores.values())
        return {label: value / total for label, value in exp_scores.items()}

    def predict_one(self, text: str) -> str:
        probabilities = self.predict_proba_one(text)
        return max(probabilities, key=probabilities.get)

    def predict(self, texts: Iterable[str]) -> list[str]:
        return [self.predict_one(text) for text in texts]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "MultinomialNaiveBayes":
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("Saved file does not contain a MultinomialNaiveBayes model.")
        return model


def classification_report(
    actual: list[str], predicted: list[str]
) -> tuple[float, dict[str, dict[str, float]]]:
    labels = sorted(set(actual) | set(predicted))
    accuracy = sum(a == p for a, p in zip(actual, predicted)) / len(actual)
    report: dict[str, dict[str, float]] = {}

    for label in labels:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(a == label for a in actual)),
        }

    return accuracy, report
