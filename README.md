# Flight Ticket Price Prediction

## Project Overview
This project aims to predict airline ticket prices using machine learning techniques. By leveraging historical flight data, we trained different models to provide accurate price estimates based on various flight attributes.

## Dataset
- **Source:** Kaggle
- **File Name:** `Restored_Cleaned_Dataset.csv`
- **Number of Records:**
  - **Training Set:** 240,024 rows (80%)
  - **Test Set:** 60,006 rows (20%)
- **Features Used:**
  - `airline`: Name of the airline.
  - `source_city`: City of departure.
  - `departure_time`: Departure time category (Morning, Evening, etc.).
  - `stops`: Number of stops (Non-stop, One-stop, etc.).
  - `arrival_time`: Arrival time category.
  - `destination_city`: City of arrival.
  - `class`: Ticket class (Economy, Business).
  - `duration`: Duration of the flight.
  - `days_left`: Days left before departure.
  - `price`: Ticket price (Target variable).

## Preprocessing Steps
1. **Data Cleaning**
   - Removed irrelevant columns (`flight`, `Unnamed: 0`).
   - Handled missing values by removing or imputing them.
2. **Feature Engineering**
   - Applied One-Hot Encoding for categorical features.
   - Normalized numerical features using `MinMaxScaler`.
3. **Splitting the Data**
   - The dataset was divided into training (80%) and test (20%) sets.

## Machine Learning Models
We implemented and compared three models:

### 1️⃣ Linear Regression (Baseline Model)
- Captures simple linear relationships between features and price.
- **Performance Metrics:**
  - **MAE:** 3152.12
  - **RMSE:** 4521.88
  - **R² Score:** 0.8954
- **Observation:** High error rate, struggles with non-linear relationships.

### 2️⃣ XGBoost (Gradient Boosting)
- A powerful ensemble learning method that captures complex patterns.
- **Performance Metrics:**
  - **MAE:** 2108.47
  - **RMSE:** 3664.72
  - **R² Score:** 0.9737
- **Observation:** Significant improvement over linear regression.

### 3️⃣ HistGradientBoosting (Best Model)
- Optimized gradient boosting model for faster training and better performance.
- **Performance Metrics:**
  - **MAE:** 2021.56
  - **RMSE:** 3521.78
  - **R² Score:** 0.9761
- **Observation:** Best-performing model with the lowest error and highest accuracy.

## Results Visualization
To illustrate the effectiveness of our models, we plotted:
- **Prediction vs. Actual Prices** for each model.
- **Comparison of MAE, RMSE, and R² Scores** across models.


## Project Execution
- **Programming Language:** Python
- **Libraries Used:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Development Environment:** Google Colab

## Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/flight-price-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Load and preprocess the dataset:
   ```python
   df = pd.read_csv("Restored_Cleaned_Dataset.csv")
   ```

