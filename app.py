from flask import Flask , render_template, url_for, request, redirect
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model_tfidf = pickle.load(open('model_tf.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        movie_name = request.form['name']
        movie_name_li = []
        movie_name_li.append(movie_name)
        movie_tfidf = model_tfidf.transform(movie_name_li)
        NNs = model.kneighbors(movie_tfidf,return_distance=True)
        top = NNs[1][0][1:]
        index_score = NNs[0][0][1:]
        recommendation = []
        metadata = pd.read_csv('movies_comp.csv', low_memory=True)
        df_all = metadata[['title']]
        for i in top:
            recommendation.append(df_all['title'][i])
        return render_template('index.html', movies = recommendation)
    else:
        redirect('/')

if __name__ == '__main__':
   app.run()