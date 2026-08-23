from flask import Flask, render_template, request
import models

app = Flask(__name__)
models.get_state()  


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    metrics = None
    moisture = ash = ""
    error = None

    if request.method == "POST":
        moisture = request.form.get("moisture", "")
        ash = request.form.get("ash", "")
        try:
            moisture_val = float(moisture)
            ash_val = float(ash)
            predictions = models.predict_all(moisture_val, ash_val)
            metrics = models.get_state()["metrics"]
        except ValueError:
            error = "Please enter valid numeric values for both fields."

    return render_template(
        "index.html",
        predictions=predictions,
        metrics=metrics,
        moisture=moisture,
        ash=ash,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
