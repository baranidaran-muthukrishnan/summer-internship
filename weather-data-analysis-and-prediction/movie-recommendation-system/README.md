# Movie Recommendation System

A compact Python recommendation system that suggests movies from a small CSV dataset.
It supports two common techniques:

- **Content-based filtering**: builds a genre profile from movies a user rated highly.
- **Collaborative filtering**: predicts ratings from item-item cosine similarity.

The project uses only the Python standard library.

## Files

- `movie_recommender.py` - recommendation engine and command-line interface.
- `data/movies.csv` - sample movie metadata.
- `data/ratings.csv` - sample user ratings.

## Run

```powershell
python movie_recommender.py --user 1 --method content --limit 5
python movie_recommender.py --user 1 --method collaborative --limit 5
```

## Test

```powershell
python -m unittest -v
```

## CSV Format

`data/movies.csv`

```csv
movie_id,title,genres
1,Toy Story,Animation|Adventure|Comedy
```

`data/ratings.csv`

```csv
user_id,movie_id,rating
1,1,5
```

You can replace the sample data with your own CSV files and pass their paths:

```powershell
python movie_recommender.py --movies path/to/movies.csv --ratings path/to/ratings.csv --user 7
```
