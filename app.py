from flask import Flask, render_template, request
import models

app = Flask(__name__)  


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    metrics = None
    moisture = ash = gcv = ""
    error = None

    if request.method == "POST":
        moisture = request.form.get("moisture", "")
        ash = request.form.get("ash", "")
        gcv = request.form.get("gcv", "")
        try:
            moisture_val = float(moisture)
            ash_val = float(ash)
            gcv_val = float(gcv)
            predictions = models.predict_all(moisture_val, ash_val, gcv_val)
            metrics = models.get_state()["metrics"]
        except ValueError:
            error = "Please enter valid numeric values for all fields."

    return render_template(
        "index.html",
        predictions=predictions,
        metrics=metrics,
        moisture=moisture,
        ash=ash,
        gcv=gcv,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
