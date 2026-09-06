

import os
import pickle

import pandas as pd
from flask import Flask, request, jsonify, render_template_string

# ---------------------------------------------------------------------------
# 1. Load the trained pipeline
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "swiggy_xgb_pipeline.pkl")

with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

preprocessor = pipeline.named_steps["preprocessor"]

# All columns the model expects, in the exact order it was trained on
FEATURE_COLUMNS = list(preprocessor.feature_names_in_)

# Build a schema {column_name: {"type": "numeric"}} or
#                {column_name: {"type": "categorical", "options": [...]}}
# directly from the fitted ColumnTransformer, so the form always matches
# whatever the model was actually trained on — nothing hardcoded here.
FEATURE_SCHEMA = {}
for trans_name, transformer, cols in preprocessor.transformers_:
    if trans_name == "remainder":
        continue
    if trans_name == "num":
        for col in cols:
            FEATURE_SCHEMA[col] = {"type": "numeric"}
    else:  # "ord" and "cat" pipelines both end in an *Encoder with .categories_
        encoder = transformer.named_steps["encoder"]
        for col, categories in zip(cols, encoder.categories_):
            FEATURE_SCHEMA[col] = {"type": "categorical", "options": list(categories)}

app = Flask(__name__)

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Swiggy Delivery Time Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; background: #fafafa; }
    h1 { color: #fc8019; }
    form { background: #fff; padding: 24px; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
    label { display: block; margin-top: 14px; font-weight: bold; font-size: 14px; }
    input, select { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 6px; }
    button { margin-top: 20px; background: #fc8019; color: #fff; border: none; padding: 12px 20px; border-radius: 6px; font-size: 16px; cursor: pointer; }
    button:hover { background: #e2710e; }
    .result { margin-top: 20px; padding: 16px; background: #eaffea; border: 1px solid #7ec87e; border-radius: 8px; font-size: 18px; }
    .error { margin-top: 20px; padding: 16px; background: #ffe9e9; border: 1px solid #e07a7a; border-radius: 8px; }
  </style>
</head>
<body>
  <h1> Swiggy Delivery Time Predictor</h1>
  <p>Model: Tuned XGBoost pipeline</p>

  {% if error %}
    <div class="error"><strong>Error:</strong> {{ error }}</div>
  {% endif %}

  {% if prediction is not none %}
    <div class="result">Predicted delivery time: <strong>{{ prediction }} minutes</strong></div>
  {% endif %}

  <form method="POST" action="/">
    {% for col in feature_columns %}
      {% set info = schema[col] %}
      <label for="{{ col }}">{{ col }}</label>
      {% if info.type == "numeric" %}
        <input type="number" step="any" name="{{ col }}" id="{{ col }}" required>
      {% else %}
        <select name="{{ col }}" id="{{ col }}" required>
          {% for opt in info.options %}
            <option value="{{ opt }}">{{ opt }}</option>
          {% endfor %}
        </select>
      {% endif %}
    {% endfor %}
    <button type="submit">Predict delivery time</button>
  </form>
</body>
</html>
"""


def build_input_row(values: dict) -> pd.DataFrame:
    """Turn a flat dict of raw form/JSON values into a single-row DataFrame
    with the exact columns and dtypes the pipeline expects."""
    row = {}
    for col in FEATURE_COLUMNS:
        if col not in values or values[col] in (None, ""):
            raise ValueError(f"Missing value for required field: '{col}'")
        info = FEATURE_SCHEMA[col]
        if info["type"] == "numeric":
            row[col] = float(values[col])
        else:
            row[col] = values[col]
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            input_df = build_input_row(request.form.to_dict())
            pred = pipeline.predict(input_df)[0]
            prediction = round(float(pred), 2)
        except Exception as exc:  # noqa: BLE001 - surface any bad input to the user
            error = str(exc)

    return render_template_string(
        PAGE_TEMPLATE,
        feature_columns=FEATURE_COLUMNS,
        schema=FEATURE_SCHEMA,
        prediction=prediction,
        error=error,
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    """JSON API endpoint.

    Example:
        curl -X POST http://127.0.0.1:5000/predict \\
             -H "Content-Type: application/json" \\
             -d '{"age": 30, "ratings": 4.5, "distance": 5.2,
                  "pickup_time_minutes": 10, "traffic": "high",
                  "order_time_of_day": "evening", "city_type": "urban",
                  "weather": "sunny", "vehicle": "bike"}'
    """
    try:
        payload = request.get_json(force=True)
        input_df = build_input_row(payload)
        pred = pipeline.predict(input_df)[0]
        return jsonify({"predicted_time_taken_minutes": round(float(pred), 2)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    print("Model expects these columns:", FEATURE_COLUMNS)
    app.run(debug=True)
