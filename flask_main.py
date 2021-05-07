import pickle
from flask import Flask, render_template, request

from nb_vectorizer_util import vectorizer

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route("/")
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    f = open('NB_classifier.pkl', 'rb')
    clf = pickle.load(f)
    message = request.form['message']
    data = [message]
    vect = vectorizer.transform(data).toarray()
    my_prediction = clf.predict(vect)
    print(my_prediction)
    return render_template('home.html', prediction=str(my_prediction[0]), data=str(data[0]))


if __name__ == "__main__":
    app.run()
