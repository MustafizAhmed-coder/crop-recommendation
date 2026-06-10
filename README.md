# Crop Recommendation System using Soil and Weather Forecasting

A beginner-friendly first-year mini project using:
- Frontend: HTML, CSS, JavaScript
- Backend: Python Flask
- ML: Scikit-learn KNN classifier
- Storage: Basic JSON file, not MySQL
- Weather API: Optional OpenWeatherMap API

## Project Flow

1. User enters soil nutrients: Nitrogen, Phosphorus, Potassium, pH.
2. User enters city and either fetches weather or manually enters temperature, humidity, rainfall.
3. Flask sends all values to the trained ML model.
4. Model recommends the most suitable crop.
5. The result is shown on a user-friendly page and saved in JSON history.

## Folder Structure

crop-recommendation-flask-project/
│
├── app.py
├── train_model.py
├── requirements.txt
├── .env.example
├── README.md
│
├── data/
│   ├── crop_dataset.csv
│   └── history.json
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   └── history.html
│
└── static/
    ├── style.css
    └── script.js

## Setup Instructions

### 1. Open folder in VS Code

```bash
cd crop-recommendation-flask-project
```

### 2. Create virtual environment

```bash
python -m venv venv
```

Activate it:

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Train the ML model

```bash
python train_model.py
```

This creates:
```text
data/crop_model.joblib
```

### 5. Run the Flask website

```bash
python app.py
```

Open:
```text
http://127.0.0.1:5000
```

## Optional: OpenWeatherMap API

Create a free API key from OpenWeatherMap.
Then create a file named `.env`:

```text
OPENWEATHER_API_KEY=your_api_key_here
```

The project still works without API key because you can enter weather manually.

## Viva Points

### What is the problem?
Farmers may select crops using traditional guessing. Wrong crop selection can reduce yield and cause financial loss.

### What is the solution?
Our system recommends a suitable crop using soil nutrients and weather conditions.

### What algorithm is used?
K-Nearest Neighbors classifier. It compares the entered values with previous crop data and selects the closest crop category.

### Why Flask?
Flask is lightweight, simple, and suitable for first-year mini projects.

### Why JSON instead of MySQL?
For a beginner project, JSON is easy to set up and avoids database installation issues. It still stores prediction history.

### Which input parameters are used?
Nitrogen, Phosphorus, Potassium, pH, temperature, humidity, and rainfall.

### Future improvements
- Use a larger Kaggle dataset.
- Add MongoDB Atlas.
- Add login system.
- Add fertilizer recommendation.
- Add crop disease detection using images.
