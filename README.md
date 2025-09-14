🚗 Car Price Prediction with Machine Learning

📌 Overview

This project focuses on predicting used car prices using Machine Learning techniques.
By analyzing car attributes such as brand, model, fuel type, transmission, mileage, engine size, and year, the model estimates the selling price of a car.

📂 Dataset

File: car_price_prediction_.csv

Target Variable: Price

Features:

Numerical: Year, Engine Size, Mileage

Categorical: Brand, Fuel Type, Transmission, Condition, Model

⚙️ Technologies Used

Python

pandas – Data analysis & manipulation

scikit-learn – Preprocessing & ML pipeline

RandomForestRegressor – Model for price prediction

🔑 Workflow

Load and explore dataset

Preprocess data:

Scale numerical features

Encode categorical features

Build pipeline with preprocessing + model

Train/test split

Train the Random Forest model

Evaluate model performance using MAE and RMSE

📊 Model Evaluation

Mean Absolute Error (MAE): Average absolute difference between actual and predicted prices

Root Mean Squared Error (RMSE): Penalizes larger errors more heavily
