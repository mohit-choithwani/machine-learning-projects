
## Movies Recommendation System

Recommender system based on Content-based filtering that recommends movies based on users similarity.
It uses item features (genre, cast, director) to recommend other items similar to what the user likes, based on their previous actions or explicit feedback.


## Screenshots

![App Screenshot](Capture.PNG)
![App Screenshot](Capture1.PNG)

## About Project

Movies dataset which is used to create recommender system is from kaggle TMDB 5000 movie dataset (https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). Then the data is preprocessed i.e Duplicate data is removed and null data is truncated and kept the essential features which are used to create our model.
The high level idea is we assign each movie with a tag, which consists of the keywords such as genre, cast name, director name, movie name. Then these tags are used to compare the similarity between movies. 
Once we assign tag to each movies them we perform stemming using PorterStemmer function of nltk library. (This step is essential because it will reduce the size of tags by avoiding common words such as 'a', 'and', 'the', etc. Also, it keeps the root words instead of full word eg. loved, loving, love ----> love)
After that we use text vectorization techniques that assign numeric value to each tag. For text vectorization, we used CountVectorizer method from sklearn library.

Q. How do we compare two vectors?
A. There are various ways to compare two vectors most commonly used Euclidean distance that calculates the distance between two vectors. There is also something call Cosine distance which is the angle between two vectors (that is used in this project)

Now we create a matric (similarity matrix) that stores the cosine distance from every movie to every other movies. 

## How does our system works?
Once we get any movies from the user, we check for the highest values in the similarity matrix and fetch the top five movies and give it to the user. That five movies have high cosine distance and hence they would be similar to the input movie.
