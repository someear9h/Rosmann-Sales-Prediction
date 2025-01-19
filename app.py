from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route("/")
def index():
    # Render the homepage with CSS support
    return render_template("home.html")

@app.route("/predictsales", methods=["POST"])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Retrieve input data from the form
            data = CustomData(
                Promo=int(request.form.get("Promo")),
                DayOfWeek=int(request.form.get("DayOfWeek")),
                CompetitionDistance=float(request.form.get("CompetitionDistance")),
                Store=int(request.form.get("Store")),
                Customers=int(request.form.get("Customers")),
                Open=int(request.form.get("Open")),
                StateHoliday=request.form.get("StateHoliday"),
                SchoolHoliday=int(request.form.get("SchoolHoliday")),
                StoreType=request.form.get("StoreType"),
                Assortment=request.form.get("Assortment"),
                CompetitionOpenSinceMonth=int(request.form.get("CompetitionOpenSinceMonth")),
                CompetitionOpenSinceYear=int(request.form.get("CompetitionOpenSinceYear")),
                Promo2=int(request.form.get("Promo2")),
                Promo2SinceWeek=int(request.form.get("Promo2SinceWeek")),
                Promo2SinceYear=int(request.form.get("Promo2SinceYear")),
                PromoInterval=request.form.get("PromoInterval"),
                WeekOfYear=int(request.form.get("WeekOfYear")),
                CompetitionOpen=float(request.form.get("CompetitionOpen")),
                Promo2Open=float(request.form.get("Promo2Open")),
                IsPromo2Month=int(request.form.get("IsPromo2Month")),
                Year=int(request.form.get("Year")),
                Month=int(request.form.get("Month")),
                Day=int(request.form.get("Day"))
            )

            # Convert input data to DataFrame
            pred_df = data.get_data_as_dataframe()

            # Initialize the prediction pipeline
            pipeline = PredictPipeline()
            prediction = pipeline.predict(pred_df)

            # Render the prediction result
            return render_template("home.html", results=prediction[0])
        except Exception as e:
            return jsonify({"error": f"Error occurred: {e}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
