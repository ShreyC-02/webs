# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:03:36 2024

@author: sneha
"""

import pandas as pd ### dataset read
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("city_day.csv")
df.drop(columns="Xylene",inplace=True)
# Calculate mean, median, and mode
mean_value = df["PM10"].mean()
median_value = df["PM10"].median()
mode_value = df["PM10"].mode()[0] 
col_names=["PM2.5","PM10","NO", "NO2", "NOx","NH3","CO","SO2","O3","Benzene","Toluene","AQI"]
for x in col_names:
    df[x].fillna(median_value, inplace=True)
mode_value = df['AQI_Bucket'].mode()[0]

# Fill NaN values with the mode
df['AQI_Bucket'].fillna(mode_value, inplace=True)


df.drop(columns=["City","Date"],inplace=True)
x=df.iloc[:,0:11]
cols=df[["PM2.5","CO","NO","NO2","PM10"]]
y=df.iloc[:,11]
y2=df.iloc[:,12]
# Split the data
x1=cols
X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# # Assuming 'model' is your trained RandomForestRegressor
# ## Model name
# filename = 'rf_model.sav'
# ## Save the model
# pickle.dump(model, open(filename, 'wb'))
# # print(matrix)

from joblib import dump, load

# Save the model
dump(model, 'rf_model.sav')

# Load the model
loaded_model = load('rf_model.sav')
with open('rf_model.sav', 'wb') as file:
    pickle.dump(model, file)
