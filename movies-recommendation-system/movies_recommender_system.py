# Importing libraries

import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Helper functions to fetch the Data

def load_data(movies_path, credits_path):
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    return movies, credits

# Merge Movies and Credits
def merge_data(movies, credits):
    return movies.merge(credits, on='title')

# Preprocessing: Keep Relevant Columns
def preprocess_data(movies):
    movies = movies[['id', 'title', 'keywords', 'overview', 'genres', 'cast', 'crew']]
    movies.dropna(inplace=True)
    movies.drop_duplicates(inplace=True)
    return movies

# Helper Function to Convert JSON-like Strings to Lists
def convert_to_list(list_of_dic):
    genre_list = []
    list_of_dictionaries = list(eval(list_of_dic))
    for i in list_of_dictionaries:
        genre_list.append(i['name'])
    return genre_list

# Generate Genre and Keywords Lists
def generate_genres_keywords(movies):
    rows = movies.shape[0]
    genre_list, keywords_list = [], []
    
    for row in range(rows):
        genre = movies.iloc[row].genres
        keyword = movies.iloc[row].keywords
        genre_list.append(convert_to_list(genre))
        keywords_list.append(convert_to_list(keyword))
        
    movies['genres'] = genre_list
    movies['keywords'] = keywords_list
    return movies

# Helper Functions to Extract Director and Top 3 Cast Members
def get_directors_name(crew):
    list_of_dictionaries = list(eval(crew))
    for i in list_of_dictionaries:
        if i['job'] == 'Director':
            return [i['name']]
    return ['Unknown']

def get_top_three_characters(cast):
    cast_list = []
    list_of_dictionaries = list(eval(cast))
    for i, member in enumerate(list_of_dictionaries[:3]):
        cast_list.append(member['name'])
    return cast_list

# Generate Cast and Crew Information
def generate_cast_crew(movies):
    rows = movies.shape[0]
    cast_list, director_list = [], []
    
    for row in range(rows):
        cast = movies.iloc[row].cast
        crew = movies.iloc[row].crew
        cast_list.append(get_top_three_characters(cast))
        director_list.append(get_directors_name(crew))
        
    movies['cast'] = cast_list
    movies['crew'] = director_list
    return movies

# Clean and Transform Text Columns
def clean_data(movies):
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])
    return movies

# Create Tags Column by Combining Relevant Features
def create_tags(movies):
    movies['tags'] = movies['genres'] + movies['keywords'] + movies['overview'] + movies['cast'] + movies['crew']
    new_movies = movies[['id', 'title', 'tags']]
    new_movies['tags'] = new_movies['tags'].apply(lambda x: " ".join(x)).apply(lambda x: x.lower())
    return new_movies

# Stemming the Tags
def stem_text(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()])

def apply_stemming(new_movies):
    new_movies['tags'] = new_movies['tags'].apply(stem_text)
    return new_movies

# Vectorization of Tags
def vectorize_tags(new_movies):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    return cv.fit_transform(new_movies['tags']).toarray(), cv.get_feature_names()

# Cosine Similarity Calculation
def calculate_similarity(vectors):
    return cosine_similarity(vectors)

# Recommendation System Function
def recommend_movies(movie_name, new_movies, similarity_matrix):
    movie_index = new_movies[new_movies['title'] == movie_name].index[0]
    distances = similarity_matrix[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommendations = []
    for i in movies_list:
        recommendations.append(new_movies.iloc[i[0]].title)
    
    return recommendations

# Save Data to Files
def save_files(new_movies, similarity_matrix):
    pickle.dump(new_movies.to_dict(), open('movies_dict.pkl', 'wb'))
    pickle.dump(similarity_matrix, open('similarity_matrix.pkl', 'wb'))

# Main Program
if __name__ == "__main__":
    # Paths to datasets
    movies_path = '/content/drive/MyDrive/ML Projects/tmdb_5000_movies.csv/tmdb_5000_movies.csv'
    credits_path = '/content/drive/MyDrive/ML Projects/tmdb_5000_credits.csv/tmdb_5000_credits.csv'

    # Load and merge data
    movies, credits = load_data(movies_path, credits_path)
    movies = merge_data(movies, credits)
    
    # Preprocessing
    movies = preprocess_data(movies)
    movies = generate_genres_keywords(movies)
    movies = generate_cast_crew(movies)
    movies = clean_data(movies)
    new_movies = create_tags(movies)
    new_movies = apply_stemming(new_movies)
    
    # Vectorization and similarity calculation
    movie_vectors, feature_names = vectorize_tags(new_movies)
    similarity_matrix = calculate_similarity(movie_vectors)

    # Example usage
    movie_name = 'Avatar'
    recommendations = recommend_movies(movie_name, new_movies, similarity_matrix)
    
    print(f"Recommendations for '{movie_name}':")
    for movie in recommendations:
        print(movie)
    
    # Save processed data
    save_files(new_movies, similarity_matrix)
