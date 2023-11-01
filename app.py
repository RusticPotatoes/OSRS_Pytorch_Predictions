from flask import Flask, send_from_directory, render_template
import os

app = Flask(__name__)


@app.route("/")
def home():
    images = os.listdir("imgs/forecasts")
    return render_template("home.html", images=images)


@app.route("/img/<path:filename>")
def serve_image(filename):
    return send_from_directory("imgs/forecasts", filename)


if __name__ == "__main__":
    app.run(debug=True)
