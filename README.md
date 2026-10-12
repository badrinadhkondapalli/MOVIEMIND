# 🎬 MovieMind

A content-based movie recommendation system built entirely on linear algebra, with no model training.
Each movie from the TMDB 5000 dataset becomes a TF-IDF vector. Recommendations come from
dot products, norms and cosine similarity, and SVD and PCA reveal the hidden structure.

## Concepts used
Feature matrices 
· Dot product & norms 
· Orthogonality 
· Projection 
· Rank & subspaces 
· LU 
·Gram–Schmidt 
· Least squares 
· Covariance & eigenvalues 
· SVD 
· PCA

## Run locally
1. Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` from Kaggle (TMDB 5000 Movie Dataset) into this folder
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
#files source 
1. Download the [TMDB 5000 Movie Dataset from Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
   and put `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the same folder as `app.py`
