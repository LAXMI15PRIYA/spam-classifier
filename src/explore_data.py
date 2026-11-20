# Import pandas and give it the short name pd
import pandas as pd

# Store the location of our dataset
file_path = "data/spam.csv"

# Load the dataset using latin-1 encoding
df = pd.read_csv(file_path, encoding="latin-1")

# Keep only the two useful columns
df = df[["v1", "v2"]]

# Rename the columns to clear names
df.columns = ["label", "message"]

# Print the dataset size
print("Shape:")
print(df.shape)

# Print how many missing values each column has
print("\nMissing values:")
print(df.isnull().sum())

# Print how many duplicate rows exist
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Count how many ham and spam messages exist
print("\nClass counts:")
print(df["label"].value_counts())

# Create a new column containing message length
df["message_length"] = df["message"].str.len()

# Show the first five rows with message length
print("\nMessage lengths:")
print(df[["label", "message", "message_length"]].head())