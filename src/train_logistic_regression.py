# Import pandas to work with our CSV dataset.
# pandas loads the CSV as a table.
# pd is the short name for pandas.
import pandas as pd


# Import train_test_split.
# It separates our data into train and test parts.
# Train teaches; test checks.
from sklearn.model_selection import train_test_split


# Import TF-IDF.
# It converts SMS text into numerical features.
# ML models need numbers instead of raw sentences.
from sklearn.feature_extraction.text import TfidfVectorizer


# Import Logistic Regression.
# This is our second classification model.
# It will predict ham or spam.
from sklearn.linear_model import LogisticRegression


# Store the dataset path.
# spam.csv is inside the data folder.
# This is a relative path.
file_path = "data/spam.csv"


# Read the CSV file.
# latin-1 correctly reads this dataset's text.
# df now contains the dataset.
df = pd.read_csv(file_path, encoding="latin-1")


# Keep only the two useful columns.
# v1 contains the label.
# v2 contains the SMS message.
df = df[["v1", "v2"]]


# Rename the columns.
# v1 becomes label.
# v2 becomes message.
df.columns = ["label", "message"]


# Remove duplicate rows.
# This keeps repeated messages from crossing train/test.
# We keep unique examples.
df = df.drop_duplicates()


# X contains the input SMS messages.
# These are what the model receives.
# Later TF-IDF converts them into numbers.
X = df["message"]


# y contains the correct answers.
# Each answer is ham or spam.
# The model learns to predict y.
y = df["label"]


# Split data into 80% train and 20% test.
# stratify keeps the ham/spam ratio similar.
# random_state gives the same split each run.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create the TF-IDF converter.
# It has not learned vocabulary yet.
# It will learn only from X_train.
tfidf = TfidfVectorizer()


# Learn TF-IDF from training messages.
# fit = learn; transform = convert.
# The result is numerical training data.
X_train_tfidf = tfidf.fit_transform(X_train)


# Convert test messages using the SAME vocabulary.
# We do not fit again on test data.
# This avoids data leakage.
X_test_tfidf = tfidf.transform(X_test)


# Create the Logistic Regression model.
# max_iter=1000 gives it enough iterations to learn.
# The model is not trained yet.
lr_model = LogisticRegression(max_iter=1000)


# Train Logistic Regression.
# X_train_tfidf contains numerical features.
# y_train contains the correct labels.
lr_model.fit(X_train_tfidf, y_train)


# Predict labels for all unseen test messages.
# Predictions will be ham or spam.
# Store predictions in y_pred.
y_pred = lr_model.predict(X_test_tfidf)


# Show that training finished.
# If we reach this line, model training worked.
# Our Logistic Regression model is ready.
print("Logistic Regression model trained successfully!")


# Show the first 5 predicted answers.
# These are produced by our model.
# [:5] means first five only.
print("\nFirst 5 predictions:")
print(y_pred[:5])


# Show the first 5 actual correct answers.
# These come from our original dataset.
# We compare these with y_pred.
print("\nFirst 5 actual labels:")
print(y_test.iloc[:5].values)


# Get prediction probabilities for the first 5 test messages.
# predict_proba gives probability for each class.
# We use this to understand model confidence.
probabilities = lr_model.predict_proba(X_test_tfidf[:5])


# Show the probabilities.
# Each row contains probabilities for ham and spam.
# The class order is shown below.
print("\nClass order:")
print(lr_model.classes_)

print("\nFirst 5 probabilities:")
print(probabilities)