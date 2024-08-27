import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import urllib.request
import bs4 as bs

# Load the NLP model and vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl', 'rb'))

def create_similarity():
    data = pd.read_csv('main_data.csv')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    similarity = cosine_similarity(count_matrix)
    return data, similarity

def rcmd(movie):
    movie = movie.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if movie not in data['movie_title'].str.lower().unique():
        return 'Sorry! Try another movie name.'
    else:
        i = data.loc[data['movie_title'].str.lower() == movie].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]  # Excluding the first item (the movie itself)
        return [data['movie_title'].iloc[x[0]] for x in lst]

def convert_to_list(my_list):
    my_list = my_list.strip('[]"').split('","')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if isinstance(rc, str):  # Check if it's an error message
        return rc
    else:
        return "---".join(rc)

@app.route("/recommend", methods=["POST"])
def recommend():
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    suggestions = get_suggestions()

    # Convert lists from strings
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    cast_ids = [id.strip() for id in cast_ids.strip('[]').split(',')]

    # Check the lengths of all lists to ensure they match
    list_lengths = {
        "cast_names": len(cast_names),
        "cast_ids": len(cast_ids),
        "cast_profiles": len(cast_profiles),
        "cast_bdays": len(cast_bdays),
        "cast_places": len(cast_places),
        "cast_bios": len(cast_bios),
    }
    print(f"List lengths: {list_lengths}")

    # Ensure all lists are the same length
    min_length = min(list_lengths.values())
    cast_names = cast_names[:min_length]
    cast_ids = cast_ids[:min_length]
    cast_profiles = cast_profiles[:min_length]
    cast_bdays = cast_bdays[:min_length]
    cast_places = cast_places[:min_length]
    cast_bios = cast_bios[:min_length]

    # Combine lists into dictionaries
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # Web scraping to get user reviews from IMDb site
    url = f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt'
    try:
        sauce = urllib.request.urlopen(url).read()
        soup = bs.BeautifulSoup(sauce, 'lxml')
        reviews_list = []
        reviews_status = []

        for review in soup.find_all("div", {"class": "text show-more__control"}):
            if review.string:
                reviews_list.append(review.string)
                movie_review_list = np.array([review.string])
                movie_vector = vectorizer.transform(movie_review_list)
                pred = clf.predict(movie_vector)
                reviews_status.append('Good' if pred else 'Bad')

        movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

    except Exception as e:
        print(f"Error scraping reviews: {e}")
        movie_reviews = {}

    return render_template('recommend.html', title=title, poster=poster, overview=overview,
                           vote_average=vote_average, vote_count=vote_count, release_date=release_date,
                           runtime=runtime, status=status, genres=genres, movie_cards=movie_cards,
                           reviews=movie_reviews, casts=casts, cast_details=cast_details)

if __name__ == '__main__':
    app.run(debug=True)
