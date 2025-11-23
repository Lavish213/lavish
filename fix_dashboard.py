from flask import Flask, render_template

app = Flask(
    __name__,
    template_folder="lavish_ui/templates",
    static_folder="lavish_ui/static"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5800, debug=True)