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

python logistics_analysis.py

Conclusion

This project demonstrates exploratory data analysis, data visualization, and machine learning for food delivery-time prediction. Future improvements could include traffic and weather features in the predictive model.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Logistics Data Analysis — Yuva Intern Week 2**

Project Overview

This project demonstrates data collection, cleaning, and preprocessing for food delivery logistics analysis using Python.

**Dataset**
File: Food_Delivery_Times.csv
Records: 1,000
Columns: 9
Source: Kaggle 

**Tools and Libraries**
Python
Pandas
Matplotlib
Seaborn
Scikit-learn
Data Preprocessing
Inspected the dataset structure, data types, and summary statistics.
Identified and handled missing values using mode and median imputation.
Checked for duplicate records and negative numerical values.
Detected potential outliers using the Interquartile Range (IQR) method.
Standardized selected numerical features using StandardScaler.
Validated and exported the cleaned and normalized datasets.

**Results**
Initial dataset: 1,000 rows and 9 columns.
Missing cells before cleaning: 120.
Duplicate rows detected: 0.
Missing cells after cleaning: 0.
Potential delivery-time outliers detected: 6.
Output files: Food_Delivery_Times_Cleaned.csv and Food_Delivery_Times_Normalized.csv.
How to Run
Install Python.
Install the required libraries using pip install pandas matplotlib seaborn scikit-learn.
Download the dataset from its original Kaggle page.
Update the dataset file path in the Python script.
Run logistics_data_analysis.py.
Learning Outcomes

This project provided practical experience with data inspection, missing-value treatment, outlier detection, numerical standardization, and data validation.

