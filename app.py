from flask import Flask, render_template, request, jsonify
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os
import json
import joblib
import requests

load_dotenv()

app = Flask(__name__)

MODEL_PATH = Path("data/crop_model.joblib")
HISTORY_PATH = Path("data/history.json")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

CROP_INFO = {
    "rice": {
        "season": "Kharif",
        "water": "High",
        "tip": "Rice grows well in high rainfall and humid conditions."
    },
    "maize": {
        "season": "Kharif / Rabi",
        "water": "Medium",
        "tip": "Maize requires well-drained soil and moderate rainfall."
    },
    "chickpea": {
        "season": "Rabi",
        "water": "Low",
        "tip": "Chickpea grows better in dry and cool conditions."
    },
    "mungbean": {
        "season": "Kharif / Summer",
        "water": "Low to Medium",
        "tip": "Mungbean is suitable for warm climate and short crop duration."
    },
    "blackgram": {
        "season": "Kharif",
        "water": "Medium",
        "tip": "Blackgram grows well in warm and humid conditions."
    },
    "orange": {
        "season": "Perennial",
        "water": "Medium",
        "tip": "Orange requires good drainage and moderate humidity."
    },
    "banana": {
        "season": "Perennial",
        "water": "High",
        "tip": "Banana needs warm climate, high humidity, and regular irrigation."
    },
    "grapes": {
        "season": "Perennial",
        "water": "Medium",
        "tip": "Grapes need dry weather during fruit development."
    },
    "apple": {
        "season": "Perennial",
        "water": "Medium",
        "tip": "Apple needs cool climate and well-drained soil."
    },
    "cotton": {
        "season": "Kharif",
        "water": "Medium",
        "tip": "Cotton grows best in warm climate with moderate rainfall."
    },
    "pigeonpeas": {
        "season": "Kharif",
        "water": "Low to Medium",
        "tip": "Pigeon pea is drought tolerant and suitable for semi-arid regions."
    },
    "coffee": {
        "season": "Perennial",
        "water": "Medium",
        "tip": "Coffee grows well in warm climate with shade and moderate rainfall."
    },
    "papaya": {
        "season": "Perennial",
        "water": "Medium to High",
        "tip": "Papaya needs warm climate and good soil drainage."
    }
}

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run 'python train_model.py' before starting Flask."
        )
    return joblib.load(MODEL_PATH)

model = load_model()

def read_history():
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text())
    except json.JSONDecodeError:
        return []

def save_history(record):
    history = read_history()
    history.insert(0, record)
    HISTORY_PATH.write_text(json.dumps(history[:50], indent=2))

def get_weather_by_city(city):
    if not OPENWEATHER_API_KEY:
        return None, "API key not set. Enter weather values manually."

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    try:
        response = requests.get(url, params=params, timeout=8)
        data = response.json()

        if response.status_code != 200:
            return None, data.get("message", "Unable to fetch weather")

        weather = {
            "temperature": round(data["main"]["temp"], 2),
            "humidity": round(data["main"]["humidity"], 2),
            # Current weather endpoint may not always include rain.
            # If rain is missing, we use 0 as fallback and allow manual edit.
            "rainfall": round(data.get("rain", {}).get("1h", 0), 2),
            "description": data["weather"][0]["description"].title()
        }
        return weather, None

    except requests.RequestException:
        return None, "Network error while fetching weather."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/weather")
def weather_api():
    city = request.args.get("city", "").strip()
    if not city:
        return jsonify({"success": False, "message": "City is required"}), 400

    weather, error = get_weather_by_city(city)
    if error:
        return jsonify({"success": False, "message": error}), 400

    return jsonify({"success": True, "weather": weather})

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        city = request.form.get("city", "Not given").strip() or "Not given"

        N = float(request.form["nitrogen"])
        P = float(request.form["phosphorus"])
        K = float(request.form["potassium"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
        crop = model.predict(input_data)[0]
        crop = str(crop).lower()

        info = CROP_INFO.get(crop, {
            "season": "Varies",
            "water": "Depends on location",
            "tip": "Use local agricultural expert advice for final decision."
        })

        record = {
            "date": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
            "city": city,
            "inputs": {
                "N": N,
                "P": P,
                "K": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall
            },
            "crop": crop.title(),
            "info": info
        }
        save_history(record)

        return render_template("result.html", record=record)

    except ValueError:
        return render_template(
            "index.html",
            error="Please enter valid numeric values in all fields."
        )
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Something went wrong: {str(e)}"
        )

@app.route("/history")
def history():
    return render_template("history.html", history=read_history())

@app.route("/about")
def about():
    return render_template("base.html")

if __name__ == "__main__":
    app.run(debug=True)
