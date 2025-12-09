from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import csv
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor
import random

app = Flask(__name__)
app.secret_key = "supersecretkey" # Needed for flash messages

movies = pd.read_csv("movie.csv")
movies.rename(columns={
    'Movie Name': 'title',
    'Rating(10)': 'rating',
    'Genre': 'genre',
    'Language': 'language',
    'Description': 'description'
}, inplace=True)
df = pd.read_csv("yourfile.csv", encoding='utf-8', on_bad_lines='skip')

# Load Recommender system models
tfidf_matrix = pickle.load(open("recommender_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load Sentiment models
sentiment_model = pickle.load(open("sentiment_model.pkl", "rb"))
sentiment_vectorizer = pickle.load(open("sentiment_vectorizer.pkl", "rb"))

REVIEWS_FILE = "reviews.csv"
WATCHLIST_FILE = "watchlist.csv"
POSTER_CACHE_FILE = "posters.csv"
USERS_FILE = "users.csv"
API_KEY = "8e6f23ad"

# Login Manager Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# User Class
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

def load_users():
    users = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                if len(row) >= 3:
                    users[row[0]] = User(row[0], row[1], row[2])
    return users

def save_user(username, password):
    users = load_users()
    for user in users.values():
        if user.username == username:
            return False # User exists
    
    new_id = str(len(users) + 1)
    password_hash = generate_password_hash(password)
    
    with open(USERS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if os.path.getsize(USERS_FILE) == 0:
            writer.writerow(["id", "username", "password_hash"])
        writer.writerow([new_id, username, password_hash])
    return True

@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    return users.get(user_id)

# Ensure files exist
if not os.path.exists(REVIEWS_FILE):
    with open(REVIEWS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["movie_title", "review", "sentiment", "date"])

# Migration check
if os.path.exists("favorites.csv") and not os.path.exists(WATCHLIST_FILE):
    try:
        os.rename("favorites.csv", WATCHLIST_FILE)
        print("Migrated favorites.csv to watchlist.csv")
    except Exception as e:
        print(f"Error migrating favorites: {e}")

if not os.path.exists(WATCHLIST_FILE):
    with open(WATCHLIST_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "movie_title"])

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "username", "password_hash"])

poster_cache = {}

# Load poster cache
if os.path.exists(POSTER_CACHE_FILE):
    try:
        with open(POSTER_CACHE_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    poster_cache[row[0]] = row[1]
    except Exception as e:
        print(f"Error loading poster cache: {e}")

def save_poster_to_cache(title, url):
    poster_cache[title] = url
    try:
        with open(POSTER_CACHE_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([title, url])
    except Exception as e:
        print(f"Error saving to poster cache: {e}")

def get_poster(title):
    if title in poster_cache:
        url = poster_cache[title]
        return url if url != "N/A" else None

    url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
    try:
        r = requests.get(url).json()
        if r.get("Response") == "True":
            poster = r.get("Poster")
            if poster and poster != "N/A":
                save_poster_to_cache(title, poster)
                return poster
            else:
                save_poster_to_cache(title, "N/A")
                return None
        else:
            save_poster_to_cache(title, "N/A")
            return None
    except:
        return None

def fetch_posters_parallel(movies_df):
    results = {}
    uncached_indices = []
    
    for i, row in movies_df.iterrows():
        title = row['title']
        if title in poster_cache:
            url = poster_cache[title]
            results[i] = url if url != "N/A" else None
        else:
            uncached_indices.append(i)
    
    if uncached_indices:
        with ThreadPoolExecutor(max_workers=10) as executor:
            tasks = {executor.submit(get_poster, movies_df.loc[i, 'title']): i for i in uncached_indices}
            
            for future in tasks:
                idx = tasks[future]
                try:
                    poster = future.result()
                    results[idx] = poster
                except:
                    results[idx] = None
                    
    return results

def get_top_rated_movies():
    candidates = movies.sort_values(by='rating', ascending=False).head(100).copy()
    poster_map = fetch_posters_parallel(candidates)
    candidates['poster'] = candidates.index.map(poster_map)
    valid_movies = candidates[candidates['poster'].notna()]
    return valid_movies.head(10).to_dict(orient="records")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
        
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        users = load_users()
        user = next((u for u in users.values() if u.username == username), None)
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password", "error")
            
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        if save_user(username, password):
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        else:
            flash("Username already exists", "error")
            
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/")
def logo():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    return render_template("logo.html")

@app.route("/home")
@login_required
def home():
    top_rated = get_top_rated_movies()
    return render_template("index.html", movies=top_rated, page="home")

@app.route("/attributes")
@login_required
def attributes():
    return render_template("attributes.html", page="attributes")

@app.route("/services")
@login_required
def services():
    return render_template("services.html", page="services")

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    if not query:
        return redirect(url_for("home"))

    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    idx = similarity.argsort()[-50:][::-1]
    candidates = movies.iloc[idx].copy()
    
    poster_map = fetch_posters_parallel(candidates)
    candidates['poster'] = candidates.index.map(poster_map)
    valid_movies = candidates[candidates['poster'].notna()]
    results = valid_movies.head(10).to_dict(orient="records")
    
    return render_template("index.html", movies=results, search_query=query, page="search")

# Service Data Cache
SERVICE_TITLES = {}

def preload_services():
    global SERVICE_TITLES
    service_files = {
        "Netflix": "netflix_titles.csv",
        "Hulu": "hulu_titles.csv",
        "Disney+": "disney_plus_titles.csv",
        "Prime Video": "amazon_prime_titles.csv",
        "Peacock Premium": "peacock.csv",
        "Apple TV+": "appletv+.csv.csv",
        "Max": "max.csv.csv",
        "Paramount+": "paramount+.csv.csv"
    }
    
    print("Preloading service data...")
    for service, filename in service_files.items():
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                # Normalize columns
                df.columns = [c.lower().strip() for c in df.columns]
                
                titles = set()
                if 'title' in df.columns:
                    # Filter for movies if type exists
                    if 'type' in df.columns:
                         df = df[df['type'].astype(str).str.lower() == 'movie']
                    titles = set(df['title'].astype(str).str.strip().str.lower())
                elif 'movie name' in df.columns:
                    titles = set(df['movie name'].astype(str).str.strip().str.lower())
                
                SERVICE_TITLES[service] = titles
                print(f"Loaded {len(titles)} titles for {service}")
            except Exception as e:
                print(f"Error preloading {service}: {e}")
        else:
            print(f"File missing for {service}: {filename}")

# Call preloader at startup
preload_services()

@app.route("/movie/<title>")
def movie_details(title):
    movie_row = movies[movies['title'] == title]
    if movie_row.empty:
        return "Movie not found", 404
    
    movie = movie_row.iloc[0].to_dict()
    movie['poster'] = get_poster(title)
    
    # Check Watch Options
    watch_options = []
    norm_title = title.lower().strip()
    for service, titles in SERVICE_TITLES.items():
        if norm_title in titles:
            watch_options.append(service)
    
    # Get reviews
    reviews = []
    if os.path.exists(REVIEWS_FILE):
        df = pd.read_csv(REVIEWS_FILE)
        # Handle cases where rating might not exist yet
        if 'rating' not in df.columns:
            df['rating'] = None
            
        reviews = df[df['movie_title'] == title].to_dict(orient="records")
        # Sort reviews by date desc if possible, or just reverse
        reviews.reverse()

    # Check if in watchlist
    is_in_watchlist = False
    if current_user.is_authenticated and os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) > 1 and row[1] == title and row[0] == current_user.id:
                    is_in_watchlist = True
                    break

    # Recommendations (if any positive reviews exist or just based on content)
    recommendations = []
    # Simple content-based recs for this movie
    desc = movie['description'] if pd.notna(movie['description']) else ""
    query_vec = vectorizer.transform([desc])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    sim_indices = similarity.argsort()[-6:][::-1] # Top 5 + self
    
    # Bounds check added here
    sim_indices = [i for i in sim_indices if i < len(movies) and movies.iloc[i]['title'] != title]
    
    rec_candidates = movies.iloc[sim_indices].copy()
    poster_map = fetch_posters_parallel(rec_candidates)
    rec_candidates['poster'] = rec_candidates.index.map(poster_map)
    recommendations = rec_candidates[rec_candidates['poster'].notna()].head(5).to_dict(orient="records")

    return render_template("details.html", movie=movie, reviews=reviews, is_in_watchlist=is_in_watchlist, recommendations=recommendations, watch_options=watch_options)

@app.route("/submit_review", methods=["POST"])
@login_required
def submit_review():
    movie_title = request.form.get("movie_title")
    review_text = request.form.get("review")
    rating = request.form.get("rating") # New rating field
    
    if not movie_title or not review_text:
        flash("Missing data", "error")
        return redirect(url_for("movie_details", title=movie_title))
    
    # Ensure items exist in reviews file header if creating new
    if not os.path.exists(REVIEWS_FILE):
        with open(REVIEWS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["movie_title", "review", "sentiment", "date", "rating"]) # Added rating

    vec = sentiment_vectorizer.transform([review_text])
    pred = sentiment_model.predict(vec)[0]

    try:
        # Check current header to see if we need to append rating or if column exists
        header = []
        with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
             reader = csv.reader(f)
             header = next(reader, [])
        
        # If 'rating' not in header, we might have an issue appending plainly, 
        # but for simplicity in this project we just append. 
        # Ideally we would migrate the CSV.
        
        with open(REVIEWS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # If the file was created by us just now, it has rating.
            # If existing file doesn't have rating column, this append will effectively add it as 5th col.
            # Read logic handles missing col by defaulting to None.
            writer.writerow([movie_title, review_text, pred, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rating])
            
    except Exception as e:
        print(f"Error saving review: {e}")
        flash("Error saving review", "error")
        return redirect(url_for("movie_details", title=movie_title))

    flash(f"Review submitted! Sentiment: {pred}", "success")
    return redirect(url_for("movie_details", title=movie_title))

# --- Subscription Optimization Logic ---
SUBSCRIPTION_PRICES = {
    "Netflix": 15.49,
    "Hulu": 7.99,
    "Disney+": 13.99,
    "Prime Video": 14.99,
    "Peacock": 5.99,
    "Apple TV+": 9.99,
    "Max": 15.99,
    "Paramount+": 5.99
}

def calculate_subscription_value(watchlist_titles):
    service_counts = {k: 0 for k in SUBSCRIPTION_PRICES.keys()}
    
    # Map for service lookup
    service_map = {
        "Netflix": "netflix",
        "Hulu": "hulu",
        "Disney+": "disney_plus",
        "Prime Video": "amazon_prime",
        "Peacock": "peacock",
        "Apple TV+": "appletv",
        "Max": "max",
        "Paramount+": "paramount"
    }

    # Pre-load all service titles for optimization
    # In a real app, we would cache this better.
    # We can use the global SERVICE_TITLES if available, otherwise reuse load logic
    # But since SERVICE_TITLES is global and populated at startup, let's use that!
    
    for display_name, service_key in service_map.items():
        # SERVICE_TITLES keys might be slightly different ("Peacock Premium" vs "Peacock")
        # Let's try to match loosely or fix the keys in SERVICE_TITLES
        
        # Helper to find matching key in SERVICE_TITLES
        titles_set = set()
        for k, v in SERVICE_TITLES.items():
            if service_key in k.lower().replace(" ", ""): # e.g. "paramount" in "Paramount+"
                 titles_set = v
                 break
        # Fallback if specific mapping needed, but SERVICE_TITLES keys are readable
        if not titles_set:
             # Try direct lookup
             titles_set = SERVICE_TITLES.get(display_name, set())
             
        # If still empty, try "Peacock Premium" for "Peacock"
        if not titles_set and display_name == "Peacock":
            titles_set = SERVICE_TITLES.get("Peacock Premium", set())
            
        for title in watchlist_titles:
            if title.lower().strip() in titles_set:
                service_counts[display_name] += 1
                
    # Calculate value
    results = []
    for service, count in service_counts.items():
        if count > 0:
            price = SUBSCRIPTION_PRICES.get(service, 0)
            cost_per_movie = price / count
            results.append({
                'service': service,
                'count': count,
                'price': price,
                'cost_per_movie': cost_per_movie
            })
            
    # Sort by count (desc) then cost_per_movie (asc)
    results.sort(key=lambda x: (-x['count'], x['cost_per_movie']))
    
    return results


@app.route("/watchlist")
@login_required
def watchlist():
    watchlist_movies = []
    
    # Load reviews first for lookup
    user_reviews = {}
    if os.path.exists(REVIEWS_FILE):
        try:
            df = pd.read_csv(REVIEWS_FILE)
            # Group by title and get the latest review
            if not df.empty:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date', ascending=False)
                
                # Create a dict of title -> review dict
                for _, row in df.iterrows():
                    title = row['movie_title']
                    if title not in user_reviews:
                        user_reviews[title] = {
                            'review': row['review'],
                            'sentiment': row['sentiment'],
                            'date': row['date'].strftime("%Y-%m-%d") if isinstance(row['date'], pd.Timestamp) else row['date']
                        }
        except Exception as e:
            print(f"Error loading reviews for watchlist: {e}")

    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            watchlist_titles = [row[1] for row in reader if len(row) > 1 and row[0] == current_user.id]
            
        for title in watchlist_titles:
            movie_row = movies[movies['title'] == title]
            if not movie_row.empty:
                m = movie_row.iloc[0].to_dict()
                m['poster'] = get_poster(title)
                
                # Attach review if exists
                if title in user_reviews:
                    m['user_review'] = user_reviews[title]['review']
                    m['review_sentiment'] = user_reviews[title]['sentiment']
                    m['review_date'] = user_reviews[title]['date']
                
                if m['poster']:
                    watchlist_movies.append(m)
    
    # --- Subscription Optimization ---
    optimization_data = calculate_subscription_value(watchlist_titles)
    best_subscription = optimization_data[0] if optimization_data else None
    
    return render_template("watchlist.html", movies=watchlist_movies, page="watchlist", 
                         optimization_data=optimization_data, best_subscription=best_subscription)

@app.route("/watchlist/add", methods=["POST"])
@login_required
def add_to_watchlist():
    title = request.form.get("title")
    if not title:
        return redirect(url_for("home"))
        
    # Check if exists
    exists = False
    rows = []
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            for row in rows:
                if len(row) > 1 and row[1] == title:
                    exists = True
    
    if not exists:
        with open(WATCHLIST_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([current_user.id, title])
            
    return redirect(url_for("movie_details", title=title))

@app.route("/watchlist/remove", methods=["POST"])
@login_required
def remove_from_watchlist():
    title = request.form.get("title")
    origin = request.form.get("origin", "details") # details or watchlist page
    
    if not title:
        return redirect(url_for("home"))
        
    rows = []
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
    header = rows[0] if rows else ["user_id", "movie_title"]
    data_rows = [r for r in rows[1:] if not (len(r) > 1 and r[1] == title and r[0] == current_user.id)]
    
    with open(WATCHLIST_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)
        
    if origin == "watchlist":
        return redirect(url_for("watchlist"))
    return redirect(url_for("movie_details", title=title))

@app.route("/recent")
def recent_reviews():
    recent_items = []
    if os.path.exists(REVIEWS_FILE):
        try:
            df = pd.read_csv(REVIEWS_FILE)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values(by='date', ascending=False)
            
            # Get top 20 recent reviews
            recent_reviews_df = df.head(20)
            
            for _, row in recent_reviews_df.iterrows():
                title = row['movie_title']
                movie_row = movies[movies['title'] == title]
                
                if not movie_row.empty:
                    m = movie_row.iloc[0].to_dict()
                    m['poster'] = get_poster(title)
                    
                    # Add review data
                    m['user_review'] = row['review']
                    m['review_sentiment'] = row['sentiment']
                    m['review_date'] = row['date'].strftime("%Y-%m-%d %H:%M") if isinstance(row['date'], pd.Timestamp) else row['date']
                    
                    if m['poster']:
                        recent_items.append(m)
        except Exception as e:
            print(f"Error: {e}")
            
    return render_template("recent.html", movies=recent_items, page="recent")

def load_service_data(service_name):
    # Map service names to filenames
    file_map = {
        "netflix": "netflix_titles.csv",
        "hulu": "hulu_titles.csv",
        "disney_plus": "disney_plus_titles.csv",
        "amazon_prime": "amazon_prime_titles.csv",
        "peacock": "peacock.csv",
        "appletv": "appletv+.csv.csv",
        "max": "max.csv.csv",
        "paramount": "paramount+.csv.csv"
    }
    
    filename = file_map.get(service_name)
    if not filename or not os.path.exists(filename):
        print(f"File not found for {service_name}: {filename}")
        return []
    
    try:
        df = pd.read_csv(filename)
        items = []
        
        # Schema detection
        columns = [c.lower().strip() for c in df.columns]
        df.columns = columns
        
        # Schema 1: Standard (show_id, type, title...)
        if 'title' in columns:
            if 'type' in df.columns:
                # Case insensitive check for 'Movie'
                df = df[df['type'].astype(str).str.lower() == 'movie']
            
            for _, row in df.iterrows():
                item = {
                    'title': row['title'],
                    'year': row['release_year'] if 'release_year' in row else (row['year'] if 'year' in row else 'N/A'),
                    'description': row['description'] if 'description' in row else '',
                    'genre': row['listed_in'] if 'listed_in' in row else (row['genres'] if 'genres' in row else 'N/A'),
                    'service': service_name.replace("_", " ").capitalize(),
                    'rating': round(random.uniform(6.0, 9.9), 1)
                }
                # Check for real rating if exists
                if 'rating' in row and pd.notna(row['rating']):
                     try: item['rating'] = float(row['rating'])
                     except: pass
                items.append(item)
                
        # Schema 2: Custom Simple (Movie Name, Rating(10), Genre...) - e.g. Peacock
        elif 'movie name' in columns:
            for _, row in df.iterrows():
                item = {
                    'title': row['movie name'],
                    'year': 'N/A', # Simple schema might not have year
                    'description': row['description'] if 'description' in row else '',
                    'genre': row['genre'] if 'genre' in row else 'N/A',
                    'service': service_name.replace("_", " ").capitalize(),
                    'rating': round(random.uniform(6.0, 9.9), 1)
                }
                # Check for real rating if exists
                if 'rating' in row and pd.notna(row['rating']):
                     try: item['rating'] = float(row['rating'])
                     except: pass
                items.append(item)
                
        else:
            print(f"Unknown schema for {filename}. Columns: {columns}")

        return items
    except Exception as e:
        print(f"Error loading {service_name}: {e}")
        return []

@app.route("/services/<service_name>")
@login_required
def service_catalog(service_name):
    valid_services = ["netflix", "hulu", "disney_plus", "amazon_prime", "peacock", "appletv", "max", "paramount"]
    
    if service_name not in valid_services:
        flash("Service not found", "error")
        return redirect(url_for("services"))
        
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # Load data
    all_movies = load_service_data(service_name)
    
    # Pagination
    total_items = len(all_movies)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    current_items = all_movies[start_idx:end_idx]
    
    # DataFrame for poster fetching compatibility
    if current_items:
        df_page = pd.DataFrame(current_items)
        poster_map = fetch_posters_parallel(df_page)
        
        final_items = []
        for item in current_items:
            # Add poster
            # Careful with index matching if duplicates exist, but here we iterate list
            # We need to find the poster for THIS specific item title
            # fetch_posters_parallel returns {index: url} based on the input df index
            
            # Re-map using the temporary df index
            # Find the index in df_page where title matches
            matches = df_page[df_page['title'] == item['title']].index
            if not matches.empty:
                idx = matches[0]
                item['poster'] = poster_map.get(idx)
            else:
                item['poster'] = None
                
            final_items.append(item)
    else:
        final_items = []

    total_pages = (total_items + per_page - 1) // per_page
    
    display_name = service_name.replace("_", " ").title()
    if service_name == "appletv": display_name = "Apple TV+"
    if service_name == "max": display_name = "Max"
    if service_name == "disney_plus": display_name = "Disney+"
    if service_name == "amazon_prime": display_name = "Prime Video"
    if service_name == "paramount": display_name = "Paramount+"
    if service_name == "peacock": display_name = "Peacock Premium"
    
    return render_template("service_catalog.html", 
                           movies=final_items, 
                           service=display_name, 
                           page_num=page, 
                           total_pages=total_pages,
                           current_page="services",
                           # Pass raw service_name for pagination links
                            service_slug=service_name)

def get_all_movies_from_all_sources():
    all_movies = []
    
    # 1. Add from main movie.csv
    # Normalize keys to match a common schema
    for i, row in movies.iterrows():
        all_movies.append({
            'title': row['title'],
            'genre': row['genre'] if pd.notna(row['genre']) else '',
            'description': row['description'] if pd.notna(row['description']) else '',
            'rating': row['rating'] if pd.notna(row['rating']) else round(random.uniform(5.5, 9.9), 1),
            'poster': None, # Will fetch later
            'source': 'Main Library'
        })

    # 2. Add from service CSVs
    service_files = {
        "Netflix": "netflix_titles.csv",
        "Hulu": "hulu_titles.csv",
        "Disney+": "disney_plus_titles.csv",
        "Prime Video": "amazon_prime_titles.csv",
        "Peacock": "peacock.csv",
        "Apple TV+": "appletv+.csv.csv",
        "Max": "max.csv.csv",
        "Paramount+": "paramount+.csv.csv"
    }

    for service, filename in service_files.items():
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                # Normalize columns
                df.columns = [c.lower().strip() for c in df.columns]
                
                # Check for 'type' column and filter for 'Movie'
                if 'type' in df.columns:
                     df = df[df['type'].astype(str).str.lower() == 'movie']

                for _, row in df.iterrows():
                    title = 'N/A'
                    genre = 'N/A'
                    desc = ''
                    
                    if 'title' in df.columns: title = row['title']
                    elif 'movie name' in df.columns: title = row['movie name']
                    
                    if 'listed_in' in df.columns: genre = row['listed_in']
                    elif 'genres' in df.columns: genre = row['genres']
                    elif 'genre' in df.columns: genre = row['genre']
                    
                    if 'description' in df.columns: desc = row['description']
                    
                    if title != 'N/A':
                        rating = round(random.uniform(6.0, 9.9), 1)
                        # Try to find real rating if exists in exotic schema
                        if 'rating' in df.columns and pd.notna(row['rating']):
                            try:
                                rating = float(row['rating'])
                            except:
                                pass # Keep random if fail

                        all_movies.append({
                            'title': title,
                            'genre': str(genre),
                            'description': str(desc),
                            'rating': rating,
                            'poster': None,
                            'source': service
                        })
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return all_movies

    return all_movies

def check_attribute(movie, category):
    genres = movie.get('genre', '').lower()
    
    if category == "slow-paced":
        slow_keywords = ['drama', 'romance', 'biography', 'history', 'documentary', 'arts']
        fast_keywords = ['action', 'thriller', 'adventure', 'horror', 'sci-fi', 'mystery', 'survival']
        is_slow = any(k in genres for k in slow_keywords)
        is_fast = any(k in genres for k in fast_keywords)
        return is_slow and not is_fast
        
    elif category == "fast-paced":
        keywords = ['action', 'thriller', 'adventure', 'sci-fi', 'horror', 'mystery']
        return any(k in genres for k in keywords)
        
    elif category == "simple-plot":
        keywords = ['comedy', 'family', 'animation', 'musical', 'romance']
        complex_keywords = ['sci-fi', 'mystery', 'crime', 'thriller', 'psychological']
        is_simple = any(k in genres for k in keywords)
        is_complex = any(k in genres for k in complex_keywords)
        return is_simple and not is_complex
        
    elif category == "complex-plot":
        keywords = ['sci-fi', 'mystery', 'crime', 'thriller', 'psychological', 'suspense']
        return any(k in genres for k in keywords)
        
    elif category == "light-theme":
        keywords = ['comedy', 'family', 'animation', 'musical', 'fantasy']
        dark_keywords = ['horror', 'crime', 'mystery', 'thriller', 'war', 'dark']
        is_light = any(k in genres for k in keywords)
        is_dark = any(k in genres for k in dark_keywords)
        return is_light and not is_dark
        
    elif category == "dark-theme":
        keywords = ['horror', 'crime', 'mystery', 'thriller', 'war', 'dark', 'noir']
        return any(k in genres for k in keywords)
        
    elif category == "watch-myself":
        # Introspective or deep genres
        keywords = ['drama', 'biography', 'documentary', 'history', 'war']
        return any(k in genres for k in keywords)
        
    elif category == "watch-friends":
        # Fun, exciting, or scary genres
        keywords = ['action', 'comedy', 'horror', 'adventure', 'sport', 'musical']
        return any(k in genres for k in keywords)
        
    return False

@app.route("/attributes/<category>")
@login_required
def show_attribute(category):
    valid_categories = [
        "slow-paced", "fast-paced", 
        "simple-plot", "complex-plot", 
        "light-theme", "dark-theme", 
        "watch-myself", "watch-friends"
    ]
    
    if category in valid_categories:
        all_movies = get_all_movies_from_all_sources()
        filtered_movies = [m for m in all_movies if check_attribute(m, category)]
        
        # De-duplicate by title
        seen_titles = set()
        unique_movies = []
        for m in filtered_movies:
            if m['title'].lower() not in seen_titles:
                unique_movies.append(m)
                seen_titles.add(m['title'].lower())
        
        display_movies = unique_movies[:20] 
        
        # Convert list of dicts to DataFrame for fetch_posters_parallel compatibility
        if display_movies:
            df = pd.DataFrame(display_movies)
            poster_map = fetch_posters_parallel(df)
            
            final_movies = []
            for i, movie in enumerate(display_movies):
                poster_url = poster_map.get(i)
                movie['poster'] = poster_url
                if poster_url:
                    final_movies.append(movie)
        else:
            final_movies = []

        title_map = {
            "slow-paced": "Slow-Paced Movies",
            "fast-paced": "Fast-Paced Movies",
            "simple-plot": "Simple Plot Movies",
            "complex-plot": "Complex Plot Movies",
            "light-theme": "Light Theme Movies",
            "dark-theme": "Dark Theme Movies",
            "watch-myself": "Movies to Watch Alone",
            "watch-friends": "Movies to Watch with Friends"
        }

        return render_template("index.html", movies=final_movies, page="attributes", search_query=title_map.get(category, category))
    
    return redirect(url_for("attributes"))

if __name__ == "__main__":
    app.run(debug=True)
