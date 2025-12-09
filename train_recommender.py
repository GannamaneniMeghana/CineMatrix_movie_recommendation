import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movie.csv")

movies['combined'] = movies['Genre'] + " " + movies['Language']

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['combined'])

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Save similarity matrix
pickle.dump(tfidf_matrix, open("recommender_model.pkl", "wb"))

print("Recommender Model Trained & Saved!")
