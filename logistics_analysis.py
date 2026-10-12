import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression


print("Logistics Delivery Analysis Project Started")
print("Python environment is ready!")

# Load the delivery dataset
df = pd.read_csv(r"C:\Users\h\Downloads\Food_Delivery_Times.csv")

# Show first 5 records
print("\nFirst 5 delivery records:")
print(df.head())

# Show column names
print("\nColumn names:")
print(df.columns.tolist())

# Show dataset size
print("\nDataset size (rows, columns):")
print(df.shape)

# Display dataset information
print("Total delivery records:", len(df))
print("\nMissing values in each column:")
print(df.isnull().sum())

# Display summary statistics
print("\nDataset summary:")
print(df.describe())

# Display column names
print("\nAvailable columns:")
print(df.columns.tolist())

print("\nDataset column names:")
print(df.columns.tolist())

# KPI 1: Total orders
total_orders = len(df)

# KPI 2: Average delivery time
average_delivery_time = df["Delivery_Time_min"].mean()

# KPI 3: Average delivery distance
average_distance = df["Distance_km"].mean()

# Display results
print("Total Orders:", total_orders)
print("Average Delivery Time:", round(average_delivery_time, 2), "minutes")
print("Average Delivery Distance:", round(average_distance, 2), "km")

# Create a scatter plot
plt.scatter(df["Distance_km"], df["Delivery_Time_min"])

# Add chart labels
plt.title("Delivery Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (minutes)")

# Display chart
plt.show()

# Calculate average delivery time for each traffic level
traffic_time = df.groupby("Traffic_Level")["Delivery_Time_min"].mean()

print("Average delivery time by traffic level:")
print(traffic_time.round(2))

# Create bar chart
traffic_time.plot(kind="bar", color="orange")

plt.title("Traffic Level vs Average Delivery Time")
plt.xlabel("Traffic Level")
plt.ylabel("Average Delivery Time (minutes)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# step -12 Display correlation between numerical columns
print("Correlation with Delivery Time:")
correlation = df.corr(numeric_only=True)["Delivery_Time_min"]
print(correlation.sort_values(ascending=False).round(2))

# Step 13 Select input features and target
features = ["Distance_km", "Preparation_Time_min"]
target = "Delivery_Time_min"

# Remove rows with missing values in selected columns
data = df[features + [target]].dropna()

X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict delivery times
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", round(mae, 2), "minutes")
print("R-squared Score:", round(r2, 2))

# Step 14 Select features and target
features = ["Distance_km", "Preparation_Time_min"]
target = "Delivery_Time_min"

data = df[features + [target]].dropna()
X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict delivery times
predictions = model.predict(X_test)

# Plot actual vs predicted values
plt.scatter(y_test, predictions, alpha=0.6, color="blue")

# Reference line
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())
plt.plot([minimum, maximum], [minimum, maximum], "r--")

plt.title("Actual vs Predicted Delivery Time")
plt.xlabel("Actual Delivery Time (minutes)")
plt.ylabel("Predicted Delivery Time (minutes)")
plt.tight_layout()
plt.show()

# Evaluate model
print("MAE:", round(mean_absolute_error(y_test, predictions), 2), "minutes")
print("R2 Score:", round(r2_score(y_test, predictions), 2))