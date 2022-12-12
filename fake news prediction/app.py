import pickle
import numpy as np
import pandas as pd
from flask import Flask, request
from flask_cors import cross_origin
from ast import literal_eval
from nltk import RegexpTokenizer, PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)


@app.route("/")
@cross_origin()
def main():
    return "Welcome to Fake new Predict. use /predict to perform prediction"


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        news = request.data
        data = literal_eval(news.decode('utf8'))
        value = api(pd.DataFrame([data]))
        value = value.iloc[0]
        if value['label'] == 1:
            value['label'] = "fake"
            return value.to_dict()
        value['label'] = "real"
        return value.to_dict()


@app.route("/predict_list", methods=["GET", "POST"])
@cross_origin()
def predict_list():
    if request.method == "POST":
        news = request.data
        data = literal_eval(news.decode('utf8'))
        value = api(pd.DataFrame(data))
        value['label'] = np.where(value['label'] == 0, 'real', 'fake')
        return value.to_dict(orient='index')


def preprocess_data(data):
    # 1. Tokenization
    tk = RegexpTokenizer('\s+', gaps=True)
    text_data = []  # List for storing the tokenized data
    for values in data.news_text:
        tokenized_data = tk.tokenize(values)  # Tokenize the news
        text_data.append(tokenized_data)  # append the tokenized data

    # 2. Stopword Removal
    # Extract the stopwords
    sw = stopwords.words('english')
    clean_data = []  # List for storing the clean text
    # Remove the stopwords using stopwords
    for data in text_data:
        clean_text = [words.lower() for words in data if words.lower() not in sw]
        clean_data.append(clean_text)  # Appned the clean_text in the clean_data list

    # 3. Stemming
    # Create a stemmer object
    ps = PorterStemmer()
    stemmed_data = []  # List for storing the stemmed data
    for data in clean_data:
        stemmed_text = [ps.stem(words) for words in data]  # Stem the words
        stemmed_data.append(stemmed_text)  # Append the stemmed text
    updated_data = []
    for data in stemmed_data:
        updated_data.append(" ".join(data))
    return updated_data


def api(df_test):
    try:
        with open("fakenewsmodel.pkl", "rb") as file:
            model = pickle.load(file)
    except:
        print("Unable to load the Fake news model pickle file, please check the spelling and path")
    preprocessed_testdata = preprocess_data(df_test)
    try:
        with open("tfidfmodel.pkl", "rb") as file:
            tfidf = pickle.load(file)
    except:
        print("Unable to load the Tfid model pickle file, please check the spelling and path")
    preprocessed_testdata = tfidf.transform(preprocessed_testdata)
    features_df = pd.DataFrame(preprocessed_testdata.toarray())
    df_test["label"] = model.predict(features_df)
    probabs = model.predict_proba(features_df)
    probs = list()
    for prob in probabs:
        probs.append(round(max(prob[0], prob[1]), 2))
    df_test["probability"] = probs
    return df_test


class FakeNewsApiService:
    def start(self):
        app.run(debug=True, use_reloader=False)


# run the api
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
