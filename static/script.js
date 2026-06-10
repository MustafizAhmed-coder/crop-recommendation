async function fetchWeather() {
    const cityInput = document.getElementById("city");
    const message = document.getElementById("weather-message");

    const city = cityInput.value.trim();

    if (!city) {
        message.textContent = "Please enter a city name first.";
        return;
    }

    message.textContent = "Fetching weather...";

    try {
        const response = await fetch(`/api/weather?city=${encodeURIComponent(city)}`);
        const data = await response.json();

        if (!data.success) {
            message.textContent = data.message + " You can enter weather manually.";
            return;
        }

        document.getElementById("temperature").value = data.weather.temperature;
        document.getElementById("humidity").value = data.weather.humidity;
        document.getElementById("rainfall").value = data.weather.rainfall;

        message.textContent =
            `Weather loaded: ${data.weather.description}. Rainfall may be 0 if not reported by API.`;

    } catch (error) {
        message.textContent = "Could not fetch weather. Enter values manually.";
    }
}
