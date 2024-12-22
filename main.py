import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


def is_anomaly(temperature, mean, std):
    return (temperature < mean - 2 * std) or (temperature > mean + 2 * std)


def anomaly_search(data):
    season_cities = data.groupby(['city', 'season'])['temperature'].agg(['mean', 'std']).reset_index()
    data = data.merge(season_cities, on=['city', 'season'], how='left')
    data['anomaly'] = data.apply(lambda row: is_anomaly(row['temperature'], row['mean'], row['std']), axis=1)
    return data


def get_current_temperature(api_key,city):
    response = requests.get("http://api.openweathermap.org/data/2.5/weather", params={'q': city,'appid': api_key,'units': 'metric' })
    if response.status_code == 401:
        return None, "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp'], None
    else:
        print(f"Ошибка получения погоды в {city}. Код ошибки: {response.status_code}")
        return None, "Error"


def get_current_season():
    month_to_season = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn"
    }
    current_month = datetime.now().month
    return month_to_season.get(current_month, "Unknown")

st.title("Temperature Analysis and Anomaly Detection")

uploaded_file = st.file_uploader("Upload Historical Temperature Data", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())
    data = anomaly_search(data.copy())

    cities = data['city'].unique()
    selected_city = st.selectbox("Select a City", cities)

    api_key = st.text_input("Enter OpenWeatherMap API Key")
    if api_key:
        current_temp, error_message = get_current_temperature(api_key, selected_city)
        if current_temp is None:
            st.error(error_message)
        else:
            season_data = data[(data['city'] == selected_city) & (data['season'] == get_current_season())]
            if not season_data.empty:
                mean = season_data['mean'].values[0]
                std = season_data['std'].values[0]
                anomaly = is_anomaly(current_temp, mean, std)
                st.write(f"Current temperature in {selected_city}: {current_temp}°C")
                st.write(f"Is the current temperature normal? {'Normal' if not anomaly else 'Anomalic'}")

    st.subheader("Descriptive Statistics")
    st.write(data.describe())

    st.subheader(f"Temperature Time Series for {selected_city}")
    city_data = data[data['city'] == selected_city]
    city_data['anomaly'] = city_data.apply(lambda row: is_anomaly(row['temperature'], row['mean'], row['std']), axis=1)
    fig, ax = plt.subplots()
    ax.plot(city_data['timestamp'], city_data['temperature'], label='Temperature', color='blue')
    ax.scatter(city_data['timestamp'][city_data['anomaly']], city_data['temperature'][city_data['anomaly']], color='red',
               label='Anomaly')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f"Temperature Time Series for {selected_city} with Anomalies")
    ax.legend()
    st.pyplot(fig)
    st.subheader("Seasonal Profiles")
    seasonal_data = data[data['city'] == selected_city].groupby('season').agg(
        {'mean': 'mean', 'std': 'mean'}).reset_index()
    fig, ax = plt.subplots()
    ax.errorbar(seasonal_data['season'], seasonal_data['mean'], yerr=seasonal_data['std'], fmt='o',
                label='Mean Temperature ± STD')
    ax.set_xlabel('Season')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f"Seasonal Profile for {selected_city}")
    ax.legend()
    st.pyplot(fig)