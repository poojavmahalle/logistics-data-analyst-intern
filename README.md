# logistics-data-analyst-intern
Food delivery time analysis and prediction using Python, Pandas, Matplotlib, and Scikit-learn.
Logistics Data Analyst Intern Project
Project Overview

This project analyzes food delivery data to explore factors affecting delivery time and demonstrate how Python and machine learning can support logistics decision-making.

Objectives
Analyze delivery records and calculate key performance indicators.
Explore the relationship between delivery distance, traffic, weather, and delivery time.
Investigate correlations between numerical variables.
Build a Linear Regression model to predict delivery times.
Evaluate model performance using MAE and R-squared.
Tools and Technologies
Python
Pandas
Matplotlib
Seaborn
Scikit-learn
Key Analysis
Total orders, average delivery time, and average delivery distance.
Delivery time comparisons by traffic level and weather.
Correlation analysis of numerical variables.
Actual versus predicted delivery-time visualization.
Machine Learning Results

The Linear Regression model uses distance and preparation time to predict delivery time.

Mean Absolute Error (MAE): 7.68 minutes
R-squared Score: 0.73

These results are from the current test split and may vary if the dataset or modeling approach changes.

How to Run
Install Python.
Install dependencies:

pip install pandas matplotlib seaborn scikit-learn

Update the dataset path in the Python script to match your computer.
Run:

python logistics_analysis.py

Conclusion

This project demonstrates exploratory data analysis, data visualization, and machine learning for food delivery-time prediction. Future improvements could include traffic and weather features in the predictive model.
