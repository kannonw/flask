from flask import Flask, render_template, request
from classifier.MultiClassifier import PredictDisease


app = Flask(__name__)


@app.route('/', methods=["GET"])
def index():
    return render_template("home.html", json={})


@app.route('/', methods=["POST"])
def index_post():
    file = request.files['file']
    img_bytes = file.read()
    label_idx, prediction_dic = PredictDisease(img_bytes)

    return render_template("home.html", class_name=list(prediction_dic)[label_idx], json=prediction_dic)


if __name__ == '__main__':
    app.run()
