# California Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-brightgreen.svg)

## Project Overview
This project builds and compares multiple regression models to predict the median house values in California districts based on demographic, geographic, and housing-related features. The workflow encompasses exploratory data analysis (EDA), data preprocessing, model evaluation, and hyperparameter tuning.

## Dataset
The project uses the standard **California Housing Dataset** fetched via `sklearn.datasets`. 
- **Target Variable:** `MedHouseVal` (Median house value in $100,000s)
- **Key Features:** Median Income (`MedInc`), Location (`Latitude`, `Longitude`), Population, Average Rooms, etc.

## Project Workflow
1. **Exploratory Data Analysis (EDA):** Analyzed target distribution, feature correlations, and geographic price density. Identified a hard cap at $500,000 in the target variable.
2. **Data Preprocessing:** Capped extreme outliers at the 99th percentile and standardized features using `StandardScaler`.
3. **Model Training & Comparison:** Evaluated Linear Regression, Random Forest, XGBoost, and LightGBM.
4. **Hyperparameter Tuning:** Fine-tuned the best-performing model (LightGBM) using `RandomizedSearchCV`.
5. **Model Export:** Saved the optimized model and scaler using `joblib` for future inference.

## Model Performance & Results

### Baseline Models
We evaluated four baseline models using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-Squared ($R^2$). Tree-based models heavily outperformed Linear Regression.

| Model | RMSE | MAE | R² Score |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.6794 | 0.4978 | 0.6477 |
| **Random Forest** | 0.5062 | 0.3287 | 0.8043 |
| **XGBoost** | 0.4605 | 0.3057 | 0.8381 |
| **LightGBM** | 0.4621 | 0.3095 | 0.8370 |

### Hyperparameter Tuning (LightGBM)
After selecting LightGBM for its optimal balance of speed and accuracy, hyperparameter tuning via `RandomizedSearchCV` yielded our best final performance:

- **Tuned LightGBM $R^2$ Score:** `0.8515`
- **Tuned LightGBM RMSE:** `0.4411`

## Key Insights
- **Winning Model:** **LightGBM** provided the best predictive accuracy post-tuning.
- **Top Predictors:** `MedInc` (Median Income) was overwhelmingly the strongest driver of house prices, followed heavily by geographic location (`Longitude` and `Latitude`).
- **Data Limitations:** The model slightly struggled at the exact $500,000 threshold due to the dataset's artificial hard cap on luxury housing prices.

## Installation & Setup

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Sumayahx/California-Housing-Price-Prediction.git](https://github.com/Sumayahx/California-Housing-Price-Prediction.git)
   cd California-Housing-Price-Prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```


## Usage
The final Scikit-Learn `Pipeline` (which includes both the `StandardScaler` and the tuned `LightGBM` model) has been exported as a `.pkl` file. You can load it to make predictions on new, unscaled data instantly:

```python
import joblib
import pandas as pd

# Load the entire pipeline
pipeline = joblib.load('lightgbm_housing_pipeline.pkl')

# Pass new raw data directly into the pipeline
# predictions = pipeline.predict(X_new)