from flask import Flask, render_template, request, jsonify
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import os
import json
import joblib
import requests
import numpy as np
import pandas as pd


# Load variables from .env file
load_dotenv()

app = Flask(__name__)


# -----------------------------
# File paths and API key
# -----------------------------

MODEL_PATH = Path("data/crop_model.joblib")
HISTORY_PATH = Path("data/history.json")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")


# -----------------------------
# Additional crop information
# This dictionary does not predict crops.
# It only provides extra details after AI prediction.
# -----------------------------

CROP_INFO = {
    "rice": {
        "season": "Kharif",
        "water": "High",
        "tip": "Rice grows well in high rainfall and humid conditions.",
    },
    "maize": {
        "season": "Kharif / Rabi",
        "water": "Medium",
        "tip": "Maize requires well-drained soil and moderate rainfall.",
    },
    "chickpea": {
        "season": "Rabi",
        "water": "Low",
        "tip": "Chickpea grows better in dry and cool conditions.",
    },
    "mungbean": {
        "season": "Kharif / Summer",
        "water": "Low to Medium",
        "tip": "Mungbean is suitable for warm climate and short crop duration.",
    },
    "blackgram": {
        "season": "Kharif",
        "water": "Medium",
        "tip": "Blackgram grows well in warm and humid conditions.",
    },
    "orange": {
        "season": "Perennial",
        "water": "Medium",
        "tip": "Orange requires good drainage and moderate humidity.",
    },
    "banana": {
        "season": "Perennial",
        "water": "High",
        "tip": "Banana needs warm climate, high humidity and regular irrigation.",
    },
    "grapes": {
        "season": "Perennial",
        "water": "Medium",
        "tip": "Grapes need dry weather during fruit development.",
    },
    "apple": {
        "season": "Perennial",
        "water": "Medium",
        "tip": "Apple needs cool climate and well-drained soil.",
    },
    "cotton": {
        "season": "Kharif",
        "water": "Medium",
        "tip": "Cotton grows best in warm climate with moderate rainfall.",
    },
    "pigeonpeas": {
        "season": "Kharif",
        "water": "Low to Medium",
        "tip": "Pigeon pea is drought tolerant and suitable for semi-arid regions.",
    },
    "coffee": {
        "season": "Perennial",
        "water": "Medium",
        "tip": "Coffee grows well in warm climate with shade and moderate rainfall.",
    },
    "papaya": {
        "season": "Perennial",
        "water": "Medium to High",
        "tip": "Papaya needs warm climate and good soil drainage.",
    },
}


# -----------------------------
# Load trained machine learning model
# -----------------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "AI model not found. Run 'python train_model.py' first."
        )

    return joblib.load(MODEL_PATH)


model = load_model()


# -----------------------------
# History functions
# -----------------------------

def read_history():
    if not HISTORY_PATH.exists():
        return []

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return []


def save_history(record):
    history = read_history()

    # Add latest prediction at the beginning
    history.insert(0, record)

    # Store only the latest 50 records
    history = history[:50]

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


# -----------------------------
# Weather API function
# -----------------------------

def get_weather_by_city(city):
    if not OPENWEATHER_API_KEY:
        return None, (
            "Weather API key is not configured. "
            "Enter weather values manually."
        )

    url = "https://api.openweathermap.org/data/2.5/weather"

    parameters = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        response = requests.get(
            url,
            params=parameters,
            timeout=8,
        )

        data = response.json()

        if response.status_code != 200:
            return None, data.get(
                "message",
                "Unable to fetch weather data.",
            )

        weather = {
            "temperature": round(
                float(data["main"]["temp"]),
                2,
            ),
            "humidity": round(
                float(data["main"]["humidity"]),
                2,
            ),
            "rainfall": round(
                float(
                    data.get("rain", {}).get("1h", 0)
                ),
                2,
            ),
            "description": data["weather"][0][
                "description"
            ].title(),
        }

        return weather, None

    except requests.RequestException:
        return None, "Network error while fetching weather."

    except (KeyError, TypeError, ValueError):
        return None, "Invalid weather data received from API."


# -----------------------------
# Home page
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Weather API route
# -----------------------------

@app.route("/api/weather")
def weather_api():
    city = request.args.get("city", "").strip()

    if not city:
        return jsonify(
            {
                "success": False,
                "message": "Please enter a city name.",
            }
        ), 400

    weather, error = get_weather_by_city(city)

    if error:
        return jsonify(
            {
                "success": False,
                "message": error,
            }
        ), 400

    return jsonify(
        {
            "success": True,
            "weather": weather,
        }
    )


# -----------------------------
# AI crop recommendation route
# -----------------------------

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        city = request.form.get(
            "city",
            "Not provided",
        ).strip()

        if not city:
            city = "Not provided"

        # Read soil and weather values from form
        nitrogen = float(request.form["nitrogen"])
        phosphorus = float(request.form["phosphorus"])
        potassium = float(request.form["potassium"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Basic input validation
        if nitrogen < 0 or phosphorus < 0 or potassium < 0:
            raise ValueError(
                "N, P and K values cannot be negative."
            )

        if not 0 <= humidity <= 100:
            raise ValueError(
                "Humidity must be between 0 and 100."
            )

        if not 0 <= ph <= 14:
            raise ValueError(
                "pH must be between 0 and 14."
            )

        if rainfall < 0:
            raise ValueError(
                "Rainfall cannot be negative."
            )

        # Create dataframe with the same feature names
        # used during model training
        input_data = pd.DataFrame(
            [
                {
                    "N": nitrogen,
                    "P": phosphorus,
                    "K": potassium,
                    "temperature": temperature,
                    "humidity": humidity,
                    "ph": ph,
                    "rainfall": rainfall,
                }
            ]
        )

        # Get probability for every crop
        probabilities = model.predict_proba(input_data)[0]

        # Get crop names learned by the AI model
        crop_classes = model.classes_

        # Arrange probabilities from highest to lowest
        sorted_indices = np.argsort(probabilities)[::-1]

        # Select maximum three crops
        number_of_results = min(
            3,
            len(sorted_indices),
        )

        top_indices = sorted_indices[:number_of_results]

        top_recommendations = []

        for index in top_indices:
            crop_name = str(
                crop_classes[index]
            ).strip().lower()

            confidence = round(
                float(probabilities[index]) * 100,
                2,
            )

            top_recommendations.append(
                {
                    "crop": crop_name.title(),
                    "confidence": confidence,
                }
            )

        # Best crop is the first result
        best_crop = top_recommendations[0]

        crop = best_crop["crop"].lower()
        confidence = best_crop["confidence"]

        # Warning for weak prediction
        if confidence < 35:
            warning = (
                "The entered conditions do not strongly match "
                "the training dataset. Verify the soil and "
                "weather values before making a farming decision."
            )
        else:
            warning = None

        # Get extra crop details from dictionary
        info = CROP_INFO.get(
            crop,
            {
                "season": "Consult the local agricultural calendar",
                "water": "Depends on crop and local conditions",
                "tip": (
                    "This crop was predicted by the AI model. "
                    "Consult a local agricultural expert before cultivation."
                ),
            },
        )

        # Save complete result
        record = {
            "date": datetime.now().strftime(
                "%d-%m-%Y %I:%M %p"
            ),
            "city": city,
            "inputs": {
                "N": nitrogen,
                "P": phosphorus,
                "K": potassium,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            },
            "crop": best_crop["crop"],
            "confidence": confidence,
            "top_recommendations": top_recommendations,
            "warning": warning,
            "info": info,
        }

        save_history(record)

        return render_template(
            "result.html",
            record=record,
        )

    except KeyError:
        return render_template(
            "index.html",
            error=(
                "One or more input fields are missing. "
                "Please fill all fields."
            ),
        )

    except ValueError as error:
        return render_template(
            "index.html",
            error=str(error),
        )

    except Exception as error:
        return render_template(
            "index.html",
            error=f"Something went wrong: {error}",
        )


# -----------------------------
# History page
# -----------------------------

@app.route("/history")
def history():
    prediction_history = read_history()

    return render_template(
        "history.html",
        history=prediction_history,
    )


# -----------------------------
# Run Flask application
# -----------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
    )