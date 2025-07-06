from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = joblib.load("fraud_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    table = None
    trim_notice = None

    if request.method == "POST":
        if "csv_file" in request.files:
            file = request.files["csv_file"]

            try:
                # Read and transform uploaded CSV
                input_df = pd.read_csv(file)

                # Drop target column if present
                if "Class" in input_df.columns:
                    input_df = input_df.drop("Class", axis=1)

                # Limit to first 7000 rows
                if len(input_df) > 7000:
                    trim_notice = f"⚠️ Note: Only the first 7,000 of {len(input_df):,} transactions were used."
                input_df = input_df.head(7000)

                # Scale Time and Amount
                input_df[["Time", "Amount"]] = scaler.transform(input_df[["Time", "Amount"]])

                # Predict
                predictions = model.predict(input_df)
                prediction_labels = pd.Series(predictions).apply(lambda x: "Fraud" if x == 1 else "Not Fraud")
                prediction_df = pd.DataFrame({"Prediction_Label": prediction_labels})
               
                # Function to highlight fraud
                def color_label(val):
                    if val == "Fraud":
                        return 'color: red; font-weight: bold'
                    elif val == "Not Fraud":
                        return 'color: green; font-weight: bold'
                    return ''

                styled_table = prediction_df[["Prediction_Label"]].style.applymap(color_label)
                table = styled_table.to_html(classes="table table-bordered")

            except Exception as e:
                prediction = f"Error: {e}"

        else:
            try:
                # Manual input
                time = float(request.form["Time"])
                amount = float(request.form["Amount"])
                v_features = [float(request.form[f"V{i}"]) for i in range(1, 29)]

                # Scale
                scaled = scaler.transform([[time, amount]])[0]
                scaled_time, scaled_amount = scaled

                features = [scaled_time] + v_features + [scaled_amount]
                input_data = np.array([features])

                pred = model.predict(input_data)[0]
                prediction = "⚠️ Fraudulent Transaction" if pred == 1 else "✅ Not Fraudulent"

            except Exception as e:
                prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, table=table)

if __name__ == "__main__":
    app.run(debug=True)