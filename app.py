from flask import Flask , render_template, url_for, request, redirect
from model import tfidf, KNN, df_all

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        movie_name = request.form['name']
        movie_name_li = list(movie_name.split(" "))
        movie_tfidf = tfidf.transform(movie_name_li)
        NNs = KNN.kneighbors(movie_tfidf,return_distance=True)
        top = NNs[1][0][1:]
        index_score = NNs[0][0][1:]
        recommendation = []
        for i in top:
            recommendation.append(df_all['title'][i])
        return render_template('index.html', movies = recommendation)
    else:
        redirect('/')

if __name__ == '__main__':
   app.run(debug=True)