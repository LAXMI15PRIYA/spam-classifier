# Import os.
# We use it to create a folder for our saved model.
import os


# Import pandas to load our SMS dataset.
import pandas as pd


# joblib lets us save trained machine-learning objects
# and load them again later.
import joblib


# Pipeline connects TF-IDF and Naive Bayes
# into one complete workflow.
from sklearn.pipeline import Pipeline


# TF-IDF converts raw SMS text into numeric features.
from sklearn.feature_extraction.text import TfidfVectorizer


# Import Multinomial Naive Bayes.
from sklearn.naive_bayes import MultinomialNB


# Split the dataset into training and testing data.
from sklearn.model_selection import train_test_split


# Calculate balanced weights.
# Spam is the minority class,
# so spam training examples get more importance.
from sklearn.utils.class_weight import compute_sample_weight


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

# Read the SMS dataset.
df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)


# Keep only the useful columns.
df = df[["v1", "v2"]]


# Rename the columns.
df.columns = ["label", "message"]


# Remove duplicate messages.
df = df.drop_duplicates()


# X contains the raw SMS messages.
X = df["message"]


# y contains the correct labels:
# ham or spam.
y = df["label"]


# --------------------------------------------------
# 2. TRAIN / TEST SPLIT
# --------------------------------------------------

# Use the same train/test split
# that we used in our experiments.
#
# X_train = training messages
# y_train = correct training labels
#
# We do not need X_test here for saving,
# but keeping the same split makes our
# training setup consistent with earlier work.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. CREATE BALANCED SAMPLE WEIGHTS
# --------------------------------------------------

# Give more importance to the minority spam class.
#
# This does NOT create new spam messages.
# It only changes the importance of training rows.
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# --------------------------------------------------
# 4. CREATE THE PIPELINE
# --------------------------------------------------

# The Pipeline contains:
#
# Raw SMS
#   ↓
# TF-IDF
#   ↓
# Multinomial Naive Bayes
#
# Because TF-IDF is inside the Pipeline,
# later we can give raw text directly to predict().
pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer()
    ),
    (
        "model",
        MultinomialNB()
    )
])


# --------------------------------------------------
# 5. TRAIN THE PIPELINE
# --------------------------------------------------

# Train the complete Pipeline.
#
# X_train contains RAW text.
# Pipeline automatically converts it to TF-IDF.
#
# model__sample_weight means:
# send sample_weights specifically
# to the Naive Bayes step inside the Pipeline.
pipeline.fit(
    X_train,
    y_train,
    model__sample_weight=sample_weights
)


# --------------------------------------------------
# 6. CREATE MODELS FOLDER
# --------------------------------------------------

# Create a folder called "models".
#
# exist_ok=True means:
# if the folder already exists,
# do not give an error.
os.makedirs(
    "models",
    exist_ok=True
)


# --------------------------------------------------
# 7. SAVE THE TRAINED PIPELINE
# --------------------------------------------------

# Save the complete trained Pipeline.
#
# This file contains:
# trained TF-IDF
# +
# trained Naive Bayes model.
joblib.dump(
    pipeline,
    "models/spam_classifier_pipeline.joblib"
)


# Tell us that saving worked.
print(
    "Model saved successfully!"
)

print(
    "Saved at: models/spam_classifier_pipeline.joblib"
)