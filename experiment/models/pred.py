import pandas as pd
import numpy as np
import joblib

# loading the model
model = joblib.load("experiment/models/HousePricePredictor.pkl")
# User Interaction
print("provide the details following\n\n")

longitude = float(input("longitude: "))
latitude = float(input("latitude: "))
housing_median_age = int(input("housing_median_age: "))
total_rooms = int(input("total_rooms: "))
total_bedrooms = int(input("total_bedrooms: "))
population = int(input("population: "))
households = int(input("households: "))
median_income = int(input("median_income: "))
ocean_proximity = input("ocean_proximity: ")

# prediction logic
data = [longitude, latitude, housing_median_age, total_rooms,
        total_bedrooms, population, households, median_income,
        ocean_proximity]

clms = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income',
        'ocean_proximity']

df = pd.DataFrame([data], columns=clms)

# prediction
result = model.predict(df)

# print the prediction
print(f"\n\t\t This House Should be Around: {result}")
