import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Download NLTK stopwords (if required)
nltk.download('stopwords')

# Initialize NLTK components
ps = PorterStemmer()

# Set up logging
logging.basicConfig(level=logging.INFO)


def load_data(movies_path, credits_path):
    """
    Loads movie and credit data from given CSV files.

    Args:
        movies_path (str): Path to the movies CSV file.
        credits_path (str): Path to the credits CSV file.

    Returns:
        pd.DataFrame: Movies and credits dataframes.
    """
    logging.info("Loading data...")
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    return movies, credits


def preprocess_data(movies, credits):
    """
    Preprocess the movie and credit data by merging and selecting relevant columns.

    Args:
        movies (pd.DataFrame): Movies dataframe.
        credits (pd.DataFrame): Credits dataframe.

    Returns:
        pd.DataFrame: Preprocessed movies dataframe.
    """
    logging.info("Preprocessing data...")
    movies = movies.merge(credits, on='title')
    movies = movies[['id', 'title', 'keywords', 'overview', 'genres', 'cast', 'crew']]
    movies.dropna(inplace=True)
    movies.drop_duplicates(inplace=True)
    return movies


def convert_to_list(list_of_dicts):
    """
    Convert a string representation of a list of dictionaries to a list of names.

    Args:
        list_of_dicts (str): String representation of a list of dictionaries.

    Returns:
        list: List of names extracted from the dictionaries.
    """
    return [item['name'] for item in eval(list_of_dicts)]


def get_directors_name(crew):
    """
    Extract the director's name from the crew information.

    Args:
        crew (str): String representation of a list of dictionaries (crew information).

    Returns:
        list: A list containing the director's name, or 'Unknown' if not found.
    """
    crew_list = eval(crew)
    for person in crew_list:
        if person['job'] == 'Director':
            return [person['name']]
    return ['Unknown']


def get_top_cast(cast):
    """
    Extract the top 3 cast members from the cast information.

    Args:
        cast (str): String representation of a list of dictionaries (cast information).

    Returns:
        list: A list of up to 3 cast member names.
    """
    return [person['name'] for person in eval(cast)[:3]]


def process_movies(movies):
    """
    Process movies data by converting stringified JSON to lists and generating tags.

    Args:
        movies (pd.DataFrame): Preprocessed movies dataframe.

    Returns:
        pd.DataFrame: Processed dataframe with tags.
    """
    logging.info("Processing movie data...")
    movies['genres'] = movies['genres'].apply(convert_to_list)
    movies['keywords'] = movies['keywords'].apply(convert_to_list)
    movies['cast'] = movies['cast'].apply(get_top_cast)
    movies['crew'] = movies['crew'].apply(get_directors_name)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    # Remove spaces and create tags
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    # Combine all text into tags
    movies['tags'] = movies['genres'] + movies['keywords'] + movies['overview'] + movies['cast'] + movies['crew']

    # Create a new DataFrame with relevant columns
    new_movies = movies[['id', 'title', 'tags']]
    new_movies['tags'] = new_movies['tags'].apply(lambda x: " ".join(x).lower())

    return new_movies


def stem(text):
    """
    Apply stemming to the input text.

    Args:
        text (str): Text to be stemmed.

    Returns:
        str: Stemmed text.
    """
    return " ".join([ps.stem(word) for word in text.split()])


def vectorize_text(movies_df, max_features=5000):
    """
    Vectorize the tags column using CountVectorizer.

    Args:
        movies_df (pd.DataFrame): DataFrame containing the tags.
        max_features (int): Maximum number of features for vectorization.

    Returns:
        np.ndarray: Vectorized text as a matrix.
        list: List of feature names.
    """
    logging.info("Vectorizing text data...")
    cv = CountVectorizer(max_features=max_features, stop_words='english')
    movie_vectors = cv.fit_transform(movies_df['tags']).toarray()
    return movie_vectors, cv.get_feature_names_out()


def calculate_similarity(vectors):
    """
    Calculate the cosine similarity between vectorized tags.

    Args:
        vectors (np.ndarray): Matrix of vectorized tags.

    Returns:
        np.ndarray: Cosine similarity matrix.
    """
    logging.info("Calculating similarity matrix...")
    return cosine_similarity(vectors)


def recommend_movies(movie_name, movies_df, similarity_matrix):
    """
    Recommend movies based on cosine similarity of tags.

    Args:
        movie_name (str): Name of the movie to get recommendations for.
        movies_df (pd.DataFrame): DataFrame containing movie titles and tags.
        similarity_matrix (np.ndarray): Cosine similarity matrix.

    Returns:
        list: List of recommended movie titles.
    """
    logging.info(f"Fetching recommendations for movie: {movie_name}")
    movie_index = movies_df[movies_df['title'] == movie_name].index[0]
    distances = similarity_matrix[movie_index]

    # Sort based on similarity and exclude the first entry (self-match)
    movie_indices = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]

    # Fetch movie titles
    recommendations = [movies_df.iloc[i[0]].title for i in movie_indices]
    return recommendations


def save_model_and_data(movies_df, similarity_matrix, movies_filename='movies_dict.pkl', similarity_filename='similarity_matrix.pkl'):
    """
    Save the movies dataframe and similarity matrix to files for later use.

    Args:
        movies_df (pd.DataFrame): Processed movies dataframe.
        similarity_matrix (np.ndarray): Cosine similarity matrix.
        movies_filename (str): Filename for saving movies data.
        similarity_filename (str): Filename for saving similarity matrix.
    """
    logging.info("Saving processed data and models...")
    pickle.dump(movies_df.to_dict(), open(movies_filename, 'wb'))
    pickle.dump(similarity_matrix, open(similarity_filename, 'wb'))


def main():
    # File paths for movies and credits datasets
    movies_path = '/path/to/movies.csv'
    credits_path = '/path/to/credits.csv'

    # Load data
    movies, credits = load_data(movies_path, credits_path)

    # Preprocess and process data
    movies = preprocess_data(movies, credits)
    new_movies = process_movies(movies)

    # Vectorize text and calculate similarity matrix
    new_movies['tags'] = new_movies['tags'].apply(stem)
    movie_vectors, _ = vectorize_text(new_movies)
    similarity_matrix = calculate_similarity(movie_vectors)

    # Recommend movies
    recommendations = recommend_movies('Avatar', new_movies, similarity_matrix)
    for movie in recommendations:
        print(movie)

    # Save the processed data and similarity matrix
    save_model_and_data(new_movies, similarity_matrix)


if __name__ == "__main__":
    main()
