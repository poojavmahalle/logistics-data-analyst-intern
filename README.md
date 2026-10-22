Logistics Data Analysis Project – Yuva Intern
Project Overview

This project was completed as part of the Yuva Intern Logistics Data Analyst Internship. It focuses on analyzing food delivery operations using Python to understand delivery performance, identify operational patterns, and explore opportunities to improve logistics efficiency.

The project is divided into three weeks, covering strategic planning, data collection and preprocessing, and advanced data analysis and visualization.

Dataset: Food Delivery Times
Primary File: Food_Delivery_Times.csv
Domain: Logistics and Supply Chain Analytics
Programming Language: Python

Project Objectives
Understand the key challenges involved in food delivery logistics.
Define relevant logistics performance indicators (KPIs).
Prepare and clean delivery data for reliable analysis.
Perform exploratory data analysis (EDA).
Analyze relationships between delivery time and operational factors.
Create visualizations to communicate analytical findings.
Develop data-driven recommendations for improving delivery efficiency.
Technologies and Libraries Used
Python – Main programming language.
Pandas – Data collection, cleaning, manipulation, and analysis.
NumPy – Numerical calculations.
Matplotlib – Data visualization.
Seaborn – Statistical charts and visual analysis.
Microsoft Word – Documentation and project reporting.
GitHub – Project version control and portfolio sharing.
Week 1: Strategic Planning and Data Exploration in Logistics
Objective

The first week focused on defining the logistics problem, identifying relevant KPIs, researching the dataset, and establishing a strategic roadmap for the analysis.

Project Scenario

Food delivery businesses need to manage delivery times, transportation conditions, restaurant preparation, and courier allocation. Delays can affect customer satisfaction and operational efficiency.

This project uses food delivery data to explore factors associated with delivery duration and identify potential areas for improvement.

Key Performance Indicators (KPIs)

The following indicators were selected:

Average Delivery Time: Measures the average time required to complete deliveries.
Median Delivery Time: Represents the middle delivery duration and is less sensitive to extreme values.
Average Delivery Distance: Measures the typical distance traveled per delivery.
Delivery-Time Variability: Uses standard deviation to measure variation in delivery duration.
Analytical Approach

The planned analysis included:

Exploratory Data Analysis (EDA).
Descriptive statistics.
Correlation analysis.
Data visualization.
Potential predictive modeling using regression techniques.
Expected Outcome

The planning phase established a structured approach for examining delivery performance and using data to support logistics decision-making.

**Week 2: Data Collection, Cleaning, and Preprocessing**
Objective

The second week focused on preparing the food delivery dataset for reliable statistical analysis and visualization.

Dataset Description

The dataset contains variables describing delivery distance, weather, traffic, time of day, vehicle type, preparation time, courier experience, and delivery duration.

Important columns include:

Column	Description
Order_ID	Unique or identifying order reference
Distance_km	Delivery distance in kilometers
Weather	Weather conditions
Traffic_Level	Traffic category
Time_of_Day	Time period of delivery
Vehicle_Type	Vehicle used for delivery
Preparation_Time_min	Restaurant preparation time in minutes
Courier_Experience_yrs	Courier experience in years
Delivery_Time_min	Total delivery duration in minutes
Data Preprocessing Methods

The preprocessing workflow included:

Loading the dataset using Pandas.
Inspecting dataset dimensions, column names, and data types.
Checking for missing values.
Identifying and removing exact duplicate rows where appropriate.
Converting numerical columns into suitable numeric data types.
Checking invalid or inconsistent values.
Preparing a cleaned dataset for further analysis.
Importance of Data Quality

Reliable data is essential for accurate analysis. Missing values, duplicates, invalid measurements, and inconsistent categories can distort statistical summaries and lead to misleading conclusions.

The cleaned dataset was used as the input for the Week 3 analysis.

Expected Outcome

A structured dataset suitable for exploratory analysis, statistical calculations, and visualization.

**Week 3: Advanced Data Analysis and Visualization in Logistics**
Objective

The third week focused on analyzing the prepared dataset, calculating KPIs, exploring statistical relationships, and generating visualizations to understand delivery performance.

Dataset Summary

The analysis was performed on 1,000 delivery records.

Metric	Result
Total analyzed records	1,000
Average delivery time	56.73 minutes
Median delivery time	55.50 minutes
Average delivery distance	10.06 km
Exploratory Data Analysis

Descriptive statistics were calculated to understand the distribution, central tendency, and variability of delivery-related variables.

Correlation analysis was used to examine the linear relationships between delivery duration and numerical factors such as distance, preparation time, and courier experience.

Visualizations Created

The Python script generates ten visualizations:

Delivery-time distribution.
Delivery time by traffic level.
Distance versus delivery time.
Correlation heatmap.
Average delivery time by traffic.
Preparation time versus delivery time.
Delivery time by weather.
Delivery time by time of day.
Delivery time by vehicle type.
Courier experience versus delivery time.

These charts help reveal patterns, compare operating conditions, and identify potential bottlenecks.

Key Findings
1. Delivery Distance

Distance had a correlation of 0.78 with delivery time, indicating a strong positive linear association. Longer delivery distances tended to be associated with longer delivery durations.

2. Traffic Conditions
Traffic level	Average delivery time
Low	52.89 minutes
Medium	56.45 minutes
High	64.81 minutes

The average delivery time under high traffic was 11.92 minutes higher than under low traffic.

3. Preparation Time

Preparation time had a correlation of 0.31 with delivery time, indicating a moderate positive linear association.

4. Courier Experience

Courier experience had a correlation of -0.09 with delivery time, indicating a very weak negative linear association in this dataset.

5. Time of Day

Evening deliveries had the highest observed average delivery time at 57.48 minutes. Night deliveries had the lowest average at 55.21 minutes.

6. Vehicle Type
Vehicle type	Average delivery time
Car	58.20 minutes
Bike	56.57 minutes
Scooter	56.05 minutes

These are descriptive comparisons and do not prove that vehicle type causes differences in delivery duration.

Recommendations

Based on the exploratory findings, the following actions could be investigated:

Use distance-aware delivery-time estimates.
Examine high-traffic periods and improve courier allocation.
Coordinate restaurant preparation with courier dispatch.
Investigate unusually long deliveries to identify possible bottlenecks.
Compare vehicle performance while considering distance and traffic conditions.
Collect additional data to validate operational improvements.
Limitations

This project uses descriptive analysis and correlation rather than a validated causal or predictive model. The observed relationships do not establish causation.

The dataset does not directly provide transportation costs, GPS route information, or promised delivery deadlines. Therefore, route optimality, cost savings, and on-time delivery rates cannot be determined reliably from the available variables alone.

How to Run the Project
Prerequisites

Install Python 3 and the required libraries.

python -m pip install pandas numpy matplotlib seaborn
Project Structure

Logistics_Data_Analysis/
│
├── week1_strategic_planning.docx
├── week2_data_preprocessing.docx
├── week3.py
├── Food_Delivery_Times_Cleaned.csv
├── README.md
│
└── week3_outputs/
    ├── kpi_summary.csv
    ├── descriptive_statistics.csv
    ├── correlation_matrix.csv
    ├── traffic_summary.csv
    ├── weather_summary.csv
    ├── time_of_day_summary.csv
    ├── vehicle_type_summary.csv
    │
    └── visualizations/
        ├── 01_delivery_time_distribution.png
        ├── 02_delivery_time_by_traffic.png
        ├── 03_distance_vs_delivery_time.png
        ├── 04_correlation_heatmap.png
        ├── 05_average_delivery_by_traffic.png
        ├── 06_preparation_vs_delivery_time.png
        ├── 07_delivery_time_by_weather.png
        ├── 08_delivery_time_by_time_of_day.png
        ├── 09_delivery_time_by_vehicle.png
        └── 10_courier_experience_vs_delivery_time.png
Note: Adjust filenames in the structure to match the actual files in your repository. Include the Week 1 and Week 2 documents only if you have created and saved them.

Execution
Place the cleaned CSV file in the same directory as the Python script and run:


python week3.py
The script generates the charts and statistical summary files in the week3_outputs directory.

# Week 4: Predictive Modeling and Optimization in Logistics Systems
Objective

The objective of Week 4 is to apply machine learning and optimization techniques to predict food delivery times and identify strategies for improving logistics efficiency.

Building on the exploratory analysis completed in Week 3, this phase focuses on forecasting delivery duration using historical delivery data, evaluating model performance, and translating predictions into practical operational recommendations.

**Problem Definition**

Food delivery businesses need reliable estimates of delivery duration to improve customer communication, courier allocation, and operational planning.

This task defines a supervised machine learning problem in which the delivery time of an order is predicted using available order and operational characteristics.

Prediction target: Delivery_Time_min

Input features may include:

Distance_km
Preparation_Time_min
Courier_Experience_yrs
Weather
Traffic_Level
Time_of_Day
Vehicle_Type

The Order_ID column should be excluded from model features because it is an identifier rather than a meaningful operational predictor.

Model Selection

The following regression algorithms can be evaluated:

1. Linear Regression

Linear Regression provides a simple baseline model and helps examine how numerical and encoded categorical features relate to delivery duration.

2. Decision Tree Regressor

A Decision Tree Regressor can capture nonlinear relationships and interactions between operational factors.

3. Random Forest Regressor

A Random Forest combines multiple decision trees and can capture more complex patterns while reducing reliance on a single tree.

The models should be compared using the same training and testing data, with the final model selected based on validation performance and practical interpretability.

Data Preparation and Model Training

The dataset should be checked for missing values, invalid values, and duplicate records before training. Categorical variables must be encoded, and numerical and categorical preprocessing should be performed within a machine learning pipeline to avoid data leakage.

Example Python implementation:

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv("Food_Delivery_Times_Cleaned.csv")
df = df.drop_duplicates()

target = "Delivery_Time_min"

numeric_features = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs"
]

categorical_features = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type"
]

features = numeric_features + categorical_features

df = df.dropna(subset=[target])

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(
    y_test, predictions
) ** 0.5
r2 = r2_score(y_test, predictions)

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R-squared:", round(r2, 3))


This is a starter implementation. The evaluation results must be obtained by running the code on the actual dataset; they should not be assumed in advance.

Model Evaluation and Validation

Three evaluation metrics are proposed:

Mean Absolute Error (MAE): Measures the average absolute difference between actual and predicted delivery times, in minutes.
Root Mean Squared Error (RMSE): Penalizes large prediction errors more strongly than MAE.
R-squared (R²): Indicates how much variation in the target variable is explained by the model relative to a mean-prediction baseline.

A lower MAE and RMSE generally indicate better prediction accuracy. R² should be interpreted alongside the error metrics and may be negative on test data.

An 80:20 train-test split provides an initial evaluation. Cross-validation and hyperparameter tuning can subsequently be used to assess model stability and improve performance. All preprocessing and tuning must be fitted using training data only.

Optimization Strategies

Predicted delivery times can support the following operational improvements:

Courier allocation: Assign available couriers based on expected delivery duration, distance, and traffic conditions.
Delivery planning: Use distance-aware predictions to improve customer delivery estimates.
Peak-period management: Investigate periods associated with longer deliveries and plan staffing accordingly.
Restaurant coordination: Coordinate preparation and courier dispatch to reduce avoidable waiting.
Performance monitoring: Compare predicted and actual delivery times to identify recurring delays.
Route optimization: Integrate road-network and GPS data with a routing algorithm to compare feasible routes and travel costs.

These are proposed strategies. Actual cost savings and improvements should be measured through operational testing before being claimed.

Expected Outcomes

The expected outcomes of this phase are:

A trained regression model for estimating food delivery times.
Test-set evaluation using MAE, RMSE, and R².
A comparison of candidate models to identify a suitable forecasting approach.
A discussion of the operational factors associated with prediction errors.
Recommendations for using delivery-time forecasts in courier allocation and planning.
Limitations

Model performance depends on the quality and representativeness of the dataset. The available data does not directly contain GPS routes, actual transportation costs, courier availability, or road-network information. Consequently, a full route optimization or cost-minimization system cannot be validated using this dataset alone.

The model predicts patterns represented in historical records and should not be assumed to generalize to every delivery environment without further testing.

Conclusion

Week 4 extends the project from descriptive analysis to predictive analytics. By training and evaluating regression models, the project can investigate whether historical delivery information supports useful estimates of delivery duration.

When combined with operational constraints and additional routing data, these predictions may help logistics teams improve planning, courier allocation, and delivery reliability. The effectiveness of each proposed improvement should be verified using measured results.



Conclusion
This project demonstrates the application of Python-based data analysis to food delivery logistics. Across the three weeks, the work progressed from strategic planning and data preparation to exploratory analysis and visualization.

The results highlight delivery distance and traffic conditions as important factors associated with delivery duration. Statistical summaries and visualizations provide a foundation for identifying potential operational bottlenecks and developing data-driven recommendations.

Further work could include delivery-time prediction using regression models, evaluation using MAE and RMSE, and additional operational data collection to validate improvements.

Project Information
Organization: Yuva Intern
Internship Role: Logistics Data Analyst Intern
Project: Food Delivery Time Analysis
Language: Python
Repository: [Add your GitHub repository URL here]
Before you publish
Replace the repository placeholder with your actual GitHub URL.

Make sure the Week 1 and Week 2 documents listed above really exist before including them.

Check that the reported Week 3 statistics match your script output.

Keep your dataset only if its license permits redistribution; otherwise, provide a link to the original Kaggle source.

Commit the updated README.md to GitHub.



