# -*- coding: utf-8 -*-
"""Movies Recommender System.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kAORDAzBufDeQhOP-ojNc3trr2kaNMmz
"""

#Importing Libraries
import numpy as np
import pandas as pd

#Fetch the data
movies = pd.read_csv('/content/drive/MyDrive/ML Projects/tmdb_5000_movies.csv/tmdb_5000_movies.csv')
credits = pd.read_csv('/content/drive/MyDrive/ML Projects/tmdb_5000_credits.csv/tmdb_5000_credits.csv')

#Visualize the data
movies.head(5)

movies.shape, credits.shape

#Merging movies df and credits df on Id or title
movies = movies.merge(credits, on = 'title')
movies.head(1)

#Konse colums ko rakhna hai
#Recommender system - for every movie I have to create tags. Ask yourself which colums are helpful for creating tags
#genres id keyword title overview *Release_date* tagline cast[name, role] crew[name, job]

movies = movies[['id', 'title', 'keywords', 'overview', 'genres', 'cast', 'crew']]

#Preprocessing step - check for any missing data

movies.isnull().sum()

#Drop the movies 
movies.dropna(inplace = True)

movies.isnull().sum()

#Check for duplicate rows

movies.duplicated().sum()

#drop duplicate rows

movies.drop_duplicates(inplace = True)

movies.duplicated().sum()

def convertToList(list_of_dic):
  genre_list = []
  #convert string to a list of dictionary
  list_of_dictionaries = list(eval(list_of_dic))

  for i in list_of_dictionaries:
    #append all the genre names into the list
    genre_list.append(i['name'])

  return genre_list

rows, _ = movies.shape

#creating a genre list and keywords list
genre_list = []
keywords_list = []

for each_row in range(rows):

  #Fetching the keywords from the genre

  genre = movies.iloc[each_row].genres
  keyword = movies.iloc[each_row].keywords
  #print(type(keyword))

  '''
  #convert string to a list of dictionary

  dic_genre = list(eval(genre))
  #print(type(dic_genre))

  for i in dic_genre:
    #append all the genre names into the list
    genre_list.append(i['name'])
  '''
  #Call the helper function to get the list of genres

  genre_list.append(convertToList(genre))
  keywords_list.append(convertToList(keyword))

print(len(genre_list), len(keywords_list))

#update the movies genres column with the new column

for each_row in range(rows):
  movies['genres'][each_row] = genre_list[each_row]
  movies['keywords'][each_row] = keywords_list[each_row]

movies['cast'][0]

def getDirectorsName(dictionary_of_crew):
  #convert string to a list of dictionary
  list_of_dictionaries = list(eval(dictionary_of_crew))

  for i in list_of_dictionaries:
    if i['job'] == 'Director':
      return [i['name']]
  
  return ['XX']

def getTopThreeCharacters(dictionary_of_cast):
  cast_list = []
  #convert string to a list of dictionary
  list_of_dictionaries = list(eval(dictionary_of_cast))
  counter = 0

  for i in list_of_dictionaries:
    if counter != 3:
      cast_list.append(i['name'])
      counter += 1
    else:
      break
  
  return cast_list

#rows, _ = movies.shape

#creating a cast list

cast_list = []
director_list = []

for each_row in range(rows):

  cast = movies.iloc[each_row].cast
  crew = movies.iloc[each_row].crew

  #Call the helper function to get the list of genres

  #genre_list.append(convertToList(genre))
  cast_list.append(getTopThreeCharacters(cast))
  director_list.append(getDirectorsName(crew))

#update the movies cast column with the new cast column

for each_row in range(rows):
  #movies['genres'][each_row] = genre_list[each_row]
  movies['cast'][each_row] = cast_list[each_row]
  movies['crew'][each_row] = director_list[each_row]

director_list

empty_list = []
for each_row in range(rows):
  #empty_list.append(director_list[each_row])
  #movies['crew'][each_row] = empty_list
  director_list[1]
  break

#convert overview into list

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies.head(6)

#We want to apply transformation in every column in order to remove white spaces

movies['genres'] = movies['genres'].apply(lambda x : [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x : [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x : [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x : [i.replace(" ", "") for i in x])

movies.head(5)

movies['tags'] = movies['genres'] + movies['keywords'] + movies['overview'] + movies['cast'] + movies['crew']

movies.head(5)

#Create a new data frame
new_movies = movies[['id', 'title', 'tags']]
new_movies.head(5)

new_movies['tags'] = new_movies['tags'].apply(lambda x:" ".join(x))

#convert everything into lower case

new_movies['tags'] = new_movies['tags'].apply(lambda x:x.lower())

!pip install nltk

#We will apply stemming ['loved', 'loving', 'love'] ---> 'love'
#we will use nltk library

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#create a helper function
def stem(text):
  stem_text = []

  for i in text.split():
    stem_text.append(ps.stem(i))

  return " ".join(stem_text)

#apply stem on tags column
new_movies['tags'] = new_movies['tags'].apply(stem)

#dekh bhai abhi problem essa hai ki you have to find the similarities between movies using their tags 
#tags only contains textual data. If it is numerical data then we could compare that. Now we want to get 
#some mathematical score from the tags whch we could compare ------ Text Vectorization --- Bag of words we use ski-kit learn library
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')

movies_vector = cv.fit_transform(new_movies['tags']).toarray()
movies_vector.shape

#Ecplore cv object
cv.get_feature_names()

#we use sklearn library to calculate csoine distance between vectors

from sklearn.metrics.pairwise import cosine_similarity

similarities_matrix = cosine_similarity(movies_vector)

similarities_matrix[0]

#I want to fetch the index of any given movie

new_movies[new_movies['title'] == 'Avatar'].index[0]

new_movies.head(5)

#New concept see if i sort any row of matrix in descending order then i lose the index of movies, So in order to store the index as well as to sort we use enumerate functiion which
#makes every entry as tuple of index and its value

sorted(list(enumerate(similarities_matrix[0])), reverse=True, key= lambda x:x[1]) [1:6]

#Create a helper function
#input - single movie
#output - list of five similar movies

def recommedMovies(movie_name):
  #fetching the index of the movie
  movie_index = new_movies[new_movies['title'] == movie_name].index[0]
  #fetch the row from the similarrity matrix
  distances = similarities_matrix[movie_index]

  #Use the logic to sort the distance
  movies_list = sorted(list(enumerate(distances)), reverse=True, key= lambda x:x[1]) [1:6]

  for i in movies_list:
    print(new_movies.iloc[i[0]].title)

  #return movies_list

recommedMovies('The R.M.')

#Importing stuff from here to streamlit app
import pickle

pickle.dump(new_movies.to_dict(), open('movies_dict.pkl', 'wb'))

pickle.dump(similarities_matrix, open('similarity_matrix.pkl', 'wb'))