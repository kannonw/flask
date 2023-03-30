from flask import Flask, json, request, render_template
from classifier.classify import main

app = Flask(__name__)


@app.route('/', methods=["GET"])
def index():
    return render_template("home.html")

@app.route('/', methods=["POST"])
def index_post():
    file = request.files['image']
    img_bytes = file.read()
    prediction_dic = main(img_bytes)

    return render_template("home.html", json=json.dumps(prediction_dic))


if __name__ == '__main__':
    app.run()
    # debug=True, port=os.getenv("PORT", default=5000)
