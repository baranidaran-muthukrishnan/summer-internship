from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    genres: frozenset[str]


@dataclass(frozen=True)
class Rating:
    user_id: int
    movie_id: int
    rating: float


def load_movies(path: str | Path) -> dict[int, Movie]:
    movies: dict[int, Movie] = {}
    with Path(path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            genres = frozenset(
                genre.strip().lower()
                for genre in row["genres"].split("|")
                if genre.strip()
            )
            movies[int(row["movie_id"])] = Movie(
                movie_id=int(row["movie_id"]),
                title=row["title"],
                genres=genres,
            )
    return movies


def load_ratings(path: str | Path) -> list[Rating]:
    ratings: list[Rating] = []
    with Path(path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            ratings.append(
                Rating(
                    user_id=int(row["user_id"]),
                    movie_id=int(row["movie_id"]),
                    rating=float(row["rating"]),
                )
            )
    return ratings


class MovieRecommender:
    def __init__(self, movies: dict[int, Movie], ratings: list[Rating]) -> None:
        self.movies = movies
        self.ratings = ratings
        self.user_ratings = self._build_user_ratings(ratings)
        self.movie_ratings = self._build_movie_ratings(ratings)
        self.global_average = (
            sum(rating.rating for rating in ratings) / len(ratings) if ratings else 0.0
        )

    @staticmethod
    def _build_user_ratings(ratings: list[Rating]) -> dict[int, dict[int, float]]:
        user_ratings: dict[int, dict[int, float]] = defaultdict(dict)
        for rating in ratings:
            user_ratings[rating.user_id][rating.movie_id] = rating.rating
        return dict(user_ratings)

    @staticmethod
    def _build_movie_ratings(ratings: list[Rating]) -> dict[int, dict[int, float]]:
        movie_ratings: dict[int, dict[int, float]] = defaultdict(dict)
        for rating in ratings:
            movie_ratings[rating.movie_id][rating.user_id] = rating.rating
        return dict(movie_ratings)

    def recommend_by_genre(
        self,
        user_id: int,
        limit: int = 5,
        min_user_rating: float = 4.0,
    ) -> list[tuple[Movie, float]]:
        rated = self.user_ratings.get(user_id, {})
        liked_movie_ids = [
            movie_id for movie_id, rating in rated.items() if rating >= min_user_rating
        ]
        liked_genres = self._genre_profile(user_id, liked_movie_ids)

        if not liked_genres:
            return self._popular_unrated(user_id, limit)

        candidates: list[tuple[Movie, float]] = []
        for movie_id, movie in self.movies.items():
            if movie_id in rated:
                continue
            score = self._genre_similarity(movie.genres, liked_genres)
            if score > 0:
                candidates.append((movie, score))

        candidates.sort(
            key=lambda item: (
                item[1],
                self._weighted_average_rating(item[0].movie_id),
                item[0].title,
            ),
            reverse=True,
        )
        return candidates[:limit]

    def recommend_collaborative(
        self,
        user_id: int,
        limit: int = 5,
        neighbors: int = 4,
    ) -> list[tuple[Movie, float]]:
        rated = self.user_ratings.get(user_id, {})
        predictions: list[tuple[Movie, float]] = []

        for movie_id, movie in self.movies.items():
            if movie_id in rated:
                continue
            predicted = self._predict_item_rating(user_id, movie_id, neighbors)
            if predicted is not None:
                predictions.append((movie, predicted))

        predictions.sort(key=lambda item: (item[1], item[0].title), reverse=True)
        return predictions[:limit]

    def _genre_profile(self, user_id: int, movie_ids: list[int]) -> dict[str, float]:
        profile: dict[str, float] = defaultdict(float)
        for movie_id in movie_ids:
            rating = self.user_ratings[user_id].get(movie_id)
            weight = rating if rating is not None else 1.0
            for genre in self.movies[movie_id].genres:
                profile[genre] += weight
        return dict(profile)

    def user_ratings_by_movie(self, movie_id: int) -> float | None:
        movie_rating_values = self.movie_ratings.get(movie_id, {}).values()
        if not movie_rating_values:
            return None
        return sum(movie_rating_values) / len(movie_rating_values)

    @staticmethod
    def _genre_similarity(genres: frozenset[str], profile: dict[str, float]) -> float:
        if not genres or not profile:
            return 0.0
        overlap = sum(profile.get(genre, 0.0) for genre in genres)
        normalization = math.sqrt(len(genres)) * math.sqrt(
            sum(weight * weight for weight in profile.values())
        )
        return overlap / normalization if normalization else 0.0

    def _predict_item_rating(
        self,
        user_id: int,
        candidate_movie_id: int,
        neighbors: int,
    ) -> float | None:
        rated = self.user_ratings.get(user_id, {})
        similarities: list[tuple[float, float]] = []

        for rated_movie_id, user_rating in rated.items():
            similarity = self._cosine_similarity(candidate_movie_id, rated_movie_id)
            if similarity > 0:
                similarities.append((similarity, user_rating))

        if not similarities:
            average = self._weighted_average_rating(candidate_movie_id)
            return average if average > 0 else None

        top_neighbors = sorted(similarities, reverse=True)[:neighbors]
        weighted_sum = sum(similarity * rating for similarity, rating in top_neighbors)
        similarity_sum = sum(abs(similarity) for similarity, _ in top_neighbors)
        return weighted_sum / similarity_sum if similarity_sum else None

    def _cosine_similarity(self, movie_a: int, movie_b: int) -> float:
        ratings_a = self.movie_ratings.get(movie_a, {})
        ratings_b = self.movie_ratings.get(movie_b, {})
        common_users = set(ratings_a) & set(ratings_b)
        if not common_users:
            return 0.0

        dot_product = sum(ratings_a[user] * ratings_b[user] for user in common_users)
        magnitude_a = math.sqrt(sum(ratings_a[user] ** 2 for user in common_users))
        magnitude_b = math.sqrt(sum(ratings_b[user] ** 2 for user in common_users))
        return dot_product / (magnitude_a * magnitude_b) if magnitude_a and magnitude_b else 0.0

    def _weighted_average_rating(self, movie_id: int) -> float:
        ratings = list(self.movie_ratings.get(movie_id, {}).values())
        if not ratings:
            return 0.0
        vote_count = len(ratings)
        average = sum(ratings) / vote_count
        confidence = vote_count / (vote_count + 5)
        return confidence * average + (1 - confidence) * self.global_average

    def _popular_unrated(self, user_id: int, limit: int) -> list[tuple[Movie, float]]:
        rated = self.user_ratings.get(user_id, {})
        candidates = [
            (movie, self._weighted_average_rating(movie_id))
            for movie_id, movie in self.movies.items()
            if movie_id not in rated
        ]
        candidates.sort(key=lambda item: (item[1], item[0].title), reverse=True)
        return candidates[:limit]


def format_recommendations(recommendations: list[tuple[Movie, float]]) -> str:
    if not recommendations:
        return "No recommendations found."
    lines = []
    for rank, (movie, score) in enumerate(recommendations, start=1):
        genres = ", ".join(sorted(movie.genres))
        lines.append(f"{rank}. {movie.title} ({genres}) - score: {score:.2f}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recommend movies using genre profiles or collaborative filtering."
    )
    parser.add_argument("--movies", default="data/movies.csv", help="Path to movies CSV.")
    parser.add_argument("--ratings", default="data/ratings.csv", help="Path to ratings CSV.")
    parser.add_argument("--user", type=int, default=1, help="User ID to recommend for.")
    parser.add_argument("--limit", type=int, default=5, help="Number of recommendations.")
    parser.add_argument(
        "--method",
        choices=("content", "collaborative"),
        default="content",
        help="Recommendation approach to use.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    recommender = MovieRecommender(
        movies=load_movies(args.movies),
        ratings=load_ratings(args.ratings),
    )

    if args.method == "content":
        recommendations = recommender.recommend_by_genre(args.user, args.limit)
    else:
        recommendations = recommender.recommend_collaborative(args.user, args.limit)

    print(format_recommendations(recommendations))


if __name__ == "__main__":
    main()
