# 🏠 Household Price Prediction (California Housing Dataset)

## 📌 Project Overview

This project focuses on predicting **house prices** using the **California Housing Dataset**.
It includes complete end-to-end steps from **data preprocessing → feature engineering → model training → prediction**.

The goal is to build a machine learning model that can estimate housing prices based on various features such as location, income, population, and proximity to the ocean.

---

## 📊 Dataset

* Source: Kaggle (California Housing Dataset)
* Features include:

  * Longitude & Latitude
  * Housing Median Age
  * Total Rooms & Bedrooms
  * Population & Households
  * Median Income
  * Ocean Proximity

---

## ⚙️ Project Workflow

### 🔹 1. Data Preprocessing

* Handling missing values
* Converting categorical features
* Feature scaling
* Outlier handling

### 🔹 2. Exploratory Data Analysis (EDA)

* Distribution analysis
* Correlation analysis
* Feature relationships

### 🔹 3. Feature Engineering

* Encoding categorical variables
* Creating pipelines
* Transformation techniques

### 🔹 4. Model Building

Multiple models were trained and evaluated:

* Linear Regression
* Decision Tree
* Random Forest
* Other ML models

### 🔹 5. Model Evaluation

* RMSE (Root Mean Squared Error)
* Cross-validation
* Performance comparison

---

## 🧠 Final Model

* The best-performing model is saved using `joblib`
* File: `HousePricePredictor.pkl` (not included in repo due to size)

---

## 🚀 How to Run the Project

### 🔹 1. Clone Repository

```bash
git clone https://github.com/CodeByVishal-0/Household-Price-Prediction.git
cd Household-Price-Prediction
```

### 🔹 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 3. Run Prediction Script

```bash
python main.py
```

---

## 🖥️ Example Input

```
longitude: -122.23
latitude: 37.88
housing_median_age: 41
total_rooms: 880
total_bedrooms: 129
population: 322
households: 126
median_income: 8
ocean_proximity: NEAR BAY
```

---

## 📁 Project Structure

```
Household Price Prediction/
│
├── experiment/
│   ├── DataPreprocessing/
│   ├── models/
│   ├── *.ipynb (EDA & experiments)
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚠️ Note

* The trained model file (`.pkl`) is excluded from the repository due to GitHub size limits.
* You can retrain the model using the provided notebooks.

---

## 🛠️ Technologies Used

* Python 🐍
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

## 📈 Future Improvements

* Deploy using Streamlit / Flask
* Add API support
* Improve model accuracy
* Hyperparameter tuning

---

## 🤝 Contributing

Feel free to fork this repository and improve the project.

---

## 📧 Contact

Created by **Vishal**
GitHub: https://github.com/CodeByVishal-0

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
