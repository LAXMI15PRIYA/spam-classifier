# Import pandas and give it the short name pd
import pandas as pd

# Store the relative location of our dataset
file_path = "data/spam.csv"

# Read the CSV file using latin-1 encoding
df = pd.read_csv(file_path, encoding="latin-1")

# Keep only the two columns we need
df = df[["v1", "v2"]]

# Give the columns clearer names
df.columns = ["label", "message"]

# Show the first 5 rows
print(df.head())

# Show the column names
print(df.columns.tolist())

# Show rows and columns
print(df.shape)