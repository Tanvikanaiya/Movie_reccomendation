#!/usr/bin/env python
# coding: utf-8

# Importing the dependencies

# In[1]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Data Collection and Pre-Processing

# In[2]:


# loading the data from the csv file to apandas dataframe
movies_data = pd.read_csv('movies.csv')


# In[3]:


# printing the first 5 rows of the dataframe
movies_data.head()


# In[4]:


# number of rows and columns in the data frame

movies_data.shape


# In[5]:


# selecting the relevant features for recommendation

selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# In[6]:


# replacing the null valuess with null string

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')


# In[7]:


# combining all the 5 selected features

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# In[8]:


print(combined_features)


# In[9]:


# converting the text data to feature vectors

vectorizer = TfidfVectorizer()


# In[10]:


feature_vectors = vectorizer.fit_transform(combined_features)


# In[11]:


print(feature_vectors)


# Cosine Similarity

# In[12]:


# getting the similarity scores using cosine similarity

similarity = cosine_similarity(feature_vectors)


# In[13]:


print(similarity)


# In[14]:


print(similarity.shape)


# Getting the movie name from the user

# In[15]:


# getting the movie name from the user

#movie_name = input(' Enter your favourite movie name : ')


# In[16]:


# creating a list with all the movie names given in the dataset

list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[17]:


# finding the close match for the movie name given by the user

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)


# In[18]:


close_match = find_close_match[0]
print(close_match)


# In[19]:


# finding the index of the movie with title

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[20]:


# getting a list of similar movies

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[21]:


len(similarity_score)


# In[22]:


# sorting the movies based on their similarity score

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
print(sorted_similar_movies)


# In[23]:


# print the name of similar movies based on the index

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# Movie Recommendation Sytem

# In[24]:


#movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = movies_data['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# In[25]:


import numpy as np
import pandas as pd
import difflib
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie dataset
movies_data = pd.read_csv('movies.csv')

# Select relevant features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine selected features
movies_data['combined_features'] = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Convert text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])

# Compute similarity scores
similarity = cosine_similarity(feature_vectors)

# List of all movie titles
list_of_all_titles = movies_data['title'].tolist()

# Function to get recommendations
def get_recommendations():
    movie_name = entry.get()
    if not movie_name:
        messagebox.showerror("Error", "Please enter a movie name!")
        return

    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        messagebox.showerror("Error", "Movie not found! Try another name.")
        return

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations.delete(0, tk.END)  # Clear previous results
    recommendations.insert(tk.END, f"Top recommendations for '{close_match}':\n")

    for i, movie in enumerate(sorted_similar_movies[1:11], start=1):  # Exclude the first one (itself)
        index = movie[0]
        title_from_index = movies_data.iloc[index]['title']
        recommendations.insert(tk.END, f"{i}. {title_from_index}")

# Create GUI window
root = tk.Tk()
root.title("Movie Recommendation System")
root.geometry("500x400")

# Create UI elements
tk.Label(root, text="Enter your favorite movie:", font=("Arial", 12)).pack(pady=10)
entry = tk.Entry(root, width=40)
entry.pack(pady=5)

btn = tk.Button(root, text="Get Recommendations", command=get_recommendations)
btn.pack(pady=10)

recommendations = tk.Listbox(root, width=60, height=12)
recommendations.pack(pady=10)

# Run the application
root.mainloop()


# In[ ]:




