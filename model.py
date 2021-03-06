import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

metadata = pd.read_csv('movies_comp.csv', low_memory=True)

tfidf = TfidfVectorizer()
metadata['title'] = metadata['title'].fillna('')
tfidf_matrix = tfidf.fit_transform(metadata['title'])
KNN = NearestNeighbors(11,p=2)
KNN.fit(tfidf_matrix)

pickle.dump(KNN, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

pickle.dump(tfidf, open('model_tf.pkl','wb'))

model_tfidf = pickle.load(open('model_tf.pkl','rb'))