import pandas as pd
import requests
import time

API_KEY = "8e6f23ad" # REPLACE THIS WITH YOUR ACTUAL API KEY

def get_poster(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
    try:
        r = requests.get(url).json()
        if r.get("Response") == "True":
            return r.get("Poster", None)
        else:
            return None
    except:
        return None

# Load your existing movie file
try:
    df = pd.read_csv("movie.csv")
except FileNotFoundError:
    print("Error: movie.csv not found.")
    exit()

poster_links = []

print("Fetching posters...")

for i, title in enumerate(df["Movie Name"]):
    poster = get_poster(title)
    poster_links.append(poster)

    print(f"{i+1}/{len(df)}  →  {title} → {poster}")

    time.sleep(1)  # free API rate limit

df["poster_url"] = poster_links

df.to_csv("movie_with_posters.csv", index=False)

print("\nCompleted! File saved as movie_with_posters.csv")
