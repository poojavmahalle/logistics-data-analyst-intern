import pandas as pd

# Load your dataset
df = pd.read_csv(
    r"C:\Users\h\Downloads\Food_Delivery_Times.csv"
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
# WEEK 2: DATA QUALITY CHECKS

# Create a copy so the original dataset remains unchanged
clean_df = df.copy()

print("Dataset shape:", clean_df.shape)

# 1. Check missing values
print("\nMissing values:")
print(clean_df.isnull().sum())

# 2. Check duplicate rows
print("\nDuplicate rows:", clean_df.duplicated().sum())

# 3. Check column data types
print("\nData types:")
print(clean_df.dtypes)

# 4. Check numerical summary
print("\nNumerical summary:")
print(clean_df.describe())

# 5. Check impossible negative values
numeric_cols = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Delivery_Time_min"
]

for col in numeric_cols:
    print(f"Negative values in {col}:",
          (clean_df[col] < 0).sum())
    # WEEK 2: CLEANING MISSING VALUES

# 1. Remove duplicate records
clean_df = df.drop_duplicates().copy()

# 2. Fill missing categorical values with the mode
categorical_cols = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day"
]

for col in categorical_cols:
    clean_df[col] = clean_df[col].fillna(
        clean_df[col].mode()[0]
    )

# 3. Fill missing courier experience with the median
clean_df["Courier_Experience_yrs"] = (
    clean_df["Courier_Experience_yrs"].fillna(
        clean_df["Courier_Experience_yrs"].median()
    )
)

# 4. Verify the cleaning
print("\nMissing values after cleaning:")
print(clean_df.isnull().sum())

print("\nCleaned dataset shape:")
print(clean_df.shape)

print("\nTotal missing cells remaining:")
print(clean_df.isnull().sum().sum())
# WEEK 2: OUTLIER DETECTION USING IQR

outlier_cols = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Delivery_Time_min"
]

print("\n--- Outlier Detection Results ---")

for col in outlier_cols:
    q1 = clean_df[col].quantile(0.25)
    q3 = clean_df[col].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = clean_df[
        (clean_df[col] < lower_limit) |
        (clean_df[col] > upper_limit)
    ]

    print(f"\nColumn: {col}")
    print("Lower limit:", round(lower_limit, 2))
    print("Upper limit:", round(upper_limit, 2))
    print("Number of outliers:", len(outliers))
    # WEEK 2: DATA NORMALIZATION

from sklearn.preprocessing import StandardScaler

# Select numerical input features
features = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs"
]

# Create the scaler
scaler = StandardScaler()

# Copy the cleaned dataset
scaled_df = clean_df.copy()

# Normalize the selected columns
scaled_df[features] = scaler.fit_transform(
    clean_df[features]
)

print("\nOriginal numerical values:")
print(clean_df[features].head())

print("\nNormalized numerical values:")
print(scaled_df[features].head())

print("\nNormalization completed successfully!")


# WEEK 2: FINAL VALIDATION AND EXPORT

# Check the cleaned dataset
print("\n--- FINAL DATA VALIDATION ---")

print("Original dataset shape:", df.shape)
print("Cleaned dataset shape:", clean_df.shape)

print("\nRemaining missing values:",
      clean_df.isnull().sum().sum())

print("Remaining duplicate rows:",
      clean_df.duplicated().sum())

print("\nNormalized feature means (approximately zero):")
print(scaled_df[[
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs"
]].mean().round(2))

# Save cleaned and normalized datasets
clean_df.to_csv("Food_Delivery_Times_Cleaned.csv", index=False)
scaled_df.to_csv("Food_Delivery_Times_Normalized.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("Normalized dataset saved successfully!")


print("Cleaned dataset:", clean_df.shape)
print("Missing cells:", clean_df.isnull().sum().sum())
print("Duplicate rows:", clean_df.duplicated().sum())

print("\nNormalized feature means:")
print(
    scaled_df[
        ["Distance_km",
         "Preparation_Time_min",
         "Courier_Experience_yrs"]
    ].mean().round(3)
)