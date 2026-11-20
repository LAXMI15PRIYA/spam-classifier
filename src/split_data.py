# Import pandas for working with our dataset
import pandas as pd

# Import train_test_split from scikit-learn
from sklearn.model_selection import train_test_split

# Store the path of our original dataset
file_path = "data/spam.csv"

# Load the dataset using latin-1 encoding
df = pd.read_csv(file_path, encoding="latin-1")

# Keep only the useful columns
df = df[["v1", "v2"]]

# Rename the columns
df.columns = ["label", "message"]

# Remove duplicate rows before splitting
df = df.drop_duplicates()

# Put the messages into X
X = df["message"]

# Put the correct spam/ham labels into y
y = df["label"]

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,                         # Messages to split
    y,                         # Labels to split
    test_size=0.20,            # Keep 20% for testing
    random_state=42,           # Make the split repeatable
    stratify=y                 # Keep similar spam/ham proportions
)

# Print the total number of unique messages
print("Total unique messages:", len(df))

# Print the number of training messages
print("Training messages:", len(X_train))

# Print the number of testing messages
print("Testing messages:", len(X_test))

# Print the class percentages in training data
print("\nTraining class percentages:")
print(y_train.value_counts(normalize=True) * 100)

# Print the class percentages in testing data
print("\nTesting class percentages:")
print(y_test.value_counts(normalize=True) * 100)