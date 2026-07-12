import unittest

from movie_recommender import MovieRecommender, load_movies, load_ratings


def build_recommender() -> MovieRecommender:
    return MovieRecommender(
        movies=load_movies("data/movies.csv"),
        ratings=load_ratings("data/ratings.csv"),
    )


class MovieRecommenderTests(unittest.TestCase):
    def test_content_recommendations_exclude_seen_movies(self) -> None:
        recommender = build_recommender()
        seen = set(recommender.user_ratings[1])

        recommendations = recommender.recommend_by_genre(user_id=1, limit=5)

        self.assertTrue(recommendations)
        self.assertTrue(all(movie.movie_id not in seen for movie, _ in recommendations))
        self.assertEqual(recommendations[0][0].title, "Interstellar")

    def test_collaborative_recommendations_predict_scores(self) -> None:
        recommender = build_recommender()

        recommendations = recommender.recommend_collaborative(user_id=1, limit=3)

        self.assertEqual(len(recommendations), 3)
        self.assertTrue(all(score > 0 for _, score in recommendations))
        self.assertEqual(
            {movie.title for movie, _ in recommendations},
            {"Finding Nemo", "Jumanji", "Spirited Away"},
        )

    def test_unknown_user_gets_popular_content_recommendations(self) -> None:
        recommender = build_recommender()

        recommendations = recommender.recommend_by_genre(user_id=999, limit=2)

        self.assertEqual(len(recommendations), 2)
        self.assertGreaterEqual(recommendations[0][1], recommendations[1][1])


if __name__ == "__main__":
    unittest.main()
