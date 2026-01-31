const apiKey = '46d99cc235a6d26be5df8165a452de39';
const card = document.getElementById('weatherCard');
const searchContainer = document.getElementById('searchContainer');
const container = document.getElementById('backgroundAnimation');
const errorMessage = document.getElementById('errorMessage');

async function getWeatherData() {
    const city = document.getElementById('cityInput').value.trim();
    if (!city) return errorMessage.textContent = 'Please enter city name.';

    errorMessage.textContent = '';
    const apiURL = `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${apiKey}&units=metric`;
    try {
        const response = await fetch(apiURL);
        if (!response.ok) throw new Error("City Not Found");

        const data = await response.json();

        document.getElementById('city').textContent = `${data.name}, ${data.sys.country}`;
        document.getElementById('time').textContent = formatTime(data.timezone);
        document.getElementById('temperature').textContent = `${Math.round(data.main.temp)}°C`;
        document.getElementById('description').textContent = data.weather[0].description;
        document.getElementById('humidity').textContent = `${data.main.humidity}%`;
        document.getElementById('wind').textContent = `${data.wind.speed} Km`;
        document.getElementById('pressure').textContent = `${data.main.pressure} hPa`;
        document.getElementById('visibility').textContent = `${Math.round(data.visibility / 1000)} Km/H`;

        const condition = data.weather[0].main.toLowerCase();
        const iconPath = getWeatherIcon(condition);
        document.getElementById('weatherIcon').src = iconPath;

        setWeatherBackground(condition);
 
        searchContainer.classList.add("raise");
        card.classList.remove("hidden");
        setTimeout(() => card.classList.add("visible"), 50);
        console.log(data);
    } catch (error) {
        card.classList.add("hidden");
        searchContainer.classList.remove("raise");
        errorMessage.textContent = 'Error 404! City Not Found.'
    }
}

function formatTime(offset) {
    const localTime = new Date(new Date().getTime() + offset);
    const hours = localTime.getHours().toString().padStart(2, '0');
    const minutes = localTime.getMinutes().toString().padStart(2, '0');
    const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const day = days[localTime.getDay()];
    return `${hours}:${minutes} - ${day}`;
}

function getWeatherIcon(condition) {
    switch (condition) {
        case 'clear': return "assets/icons/clear.png";
        case 'clouds': return "assets/icons/cloud.png";
        case 'mist': return "assets/icons/mist.png";
        case 'rain': return "assets/icons/rain.png";
        case 'snow': return "assets/icons/snow.png";
        case 'haze': return "assets/icons/cloud.png";
        default: return "assets/icons/404.png";
    }
}

let currentAnimation = null;

function setWeatherBackground(condition) {
    container.innerHTML = "";
    let animationFile = "";

    switch(condition) {
        case 'clear': animationFile = "sunny.json"; break;
        case 'clouds': animationFile = "cloud.json"; break;
        case 'mist': animationFile = "mist.json"; break;
        case 'rain': animationFile = "rain.json"; break;
        case 'snow': animationFile = "snow.json"; break;
        case 'haze': animationFile = "cloud.json"; break;
        default: animationFile = "";
    }

    currentAnimation = lottie.loadAnimation({
        container: container,
        rederer: 'svg',
        loop: true,
        autoplay: true,
        path: `assets/animations/${animationFile}`
    });
}