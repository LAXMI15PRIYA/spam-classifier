# Import pandas so we can load our SMS dataset
# and later display the model results as a table.
import pandas as pd


# time.perf_counter() lets us measure
# how long each model takes to predict.
import time


# Split our dataset into training and testing data.
from sklearn.model_selection import train_test_split


# Convert SMS text into TF-IDF number vectors.
from sklearn.feature_extraction.text import TfidfVectorizer


# Import our first model: Multinomial Naive Bayes.
from sklearn.naive_bayes import MultinomialNB


# Import our second model: Logistic Regression.
from sklearn.linear_model import LogisticRegression


# This calculates balanced weights for training examples.
# We need this for Balanced Naive Bayes.
from sklearn.utils.class_weight import compute_sample_weight


# Import the metrics we need to compare the models.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. LOAD THE DATA
# --------------------------------------------------

# Read the SMS dataset.
# latin-1 is the encoding used by this CSV file.
df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)


# Keep only the two useful columns.
# v1 contains ham/spam labels.
# v2 contains the actual SMS messages.
df = df[["v1", "v2"]]


# Rename the columns so their meaning is clearer.
df.columns = ["label", "message"]


# Remove duplicate SMS messages.
# We already used this same preprocessing earlier.
df = df.drop_duplicates()


# X contains the input messages.
X = df["message"]


# y contains the correct ham/spam labels.
y = df["label"]


# --------------------------------------------------
# 2. TRAIN / TEST SPLIT
# --------------------------------------------------

# Put 80% of messages into training data
# and 20% into testing data.
#
# random_state=42 makes the split repeatable.
#
# stratify=y keeps approximately the same
# ham/spam percentage in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. TF-IDF
# --------------------------------------------------

# Create the TF-IDF converter.
tfidf = TfidfVectorizer()


# fit_transform() does TWO things:
# 1. Learn the vocabulary from TRAINING data.
# 2. Convert training messages into numbers.
X_train_tfidf = tfidf.fit_transform(X_train)


# transform() only converts the test messages.
# The test set must NOT teach the TF-IDF vectorizer.
X_test_tfidf = tfidf.transform(X_test)


# --------------------------------------------------
# 4. SAMPLE WEIGHTS FOR BALANCED NAIVE BAYES
# --------------------------------------------------

# Calculate a weight for every training example.
#
# Spam is rare, so spam examples receive
# more importance than ham examples.
nb_sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# --------------------------------------------------
# 5. CREATE THE FOUR MODELS
# --------------------------------------------------

# Normal Naive Bayes.
normal_nb = MultinomialNB()


# Balanced Naive Bayes.
# The model itself is the same.
# We will pass sample weights during .fit().
balanced_nb = MultinomialNB()


# Normal Logistic Regression.
normal_lr = LogisticRegression(
    max_iter=1000
)


# Balanced Logistic Regression.
# class_weight="balanced" automatically gives
# more importance to the minority spam class.
balanced_lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)


# --------------------------------------------------
# 6. TRAIN THE MODELS
# --------------------------------------------------

# Train normal Naive Bayes normally.
normal_nb.fit(
    X_train_tfidf,
    y_train
)


# Train balanced Naive Bayes.
# sample_weight gives rare spam examples
# more importance while learning.
balanced_nb.fit(
    X_train_tfidf,
    y_train,
    sample_weight=nb_sample_weights
)


# Train normal Logistic Regression.
normal_lr.fit(
    X_train_tfidf,
    y_train
)


# Train balanced Logistic Regression.
balanced_lr.fit(
    X_train_tfidf,
    y_train
)


# --------------------------------------------------
# 7. FUNCTION TO EVALUATE ONE MODEL
# --------------------------------------------------

def evaluate_model(name, model):

    # Start the timer immediately before prediction.
    start_time = time.perf_counter()


    # Predict ham/spam for every unseen test message.
    y_pred = model.predict(X_test_tfidf)


    # Stop the timer after prediction finishes.
    end_time = time.perf_counter()


    # Calculate how long prediction took.
    prediction_time = end_time - start_time


    # Accuracy:
    # Out of ALL test messages, how many were correct?
    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    # Precision:
    # When model says SPAM, how often is it right?
    precision = precision_score(
        y_test,
        y_pred,
        pos_label="spam"
    )


    # Recall:
    # Out of all REAL spam, how much did we catch?
    recall = recall_score(
        y_test,
        y_pred,
        pos_label="spam"
    )


    # F1:
    # Balance between precision and recall.
    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="spam"
    )


    # Create confusion matrix.
    #
    # Because labels are ["ham", "spam"],
    # the matrix is:
    #
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["ham", "spam"]
    )


    # Extract the four confusion-matrix numbers.
    tn, fp, fn, tp = cm.ravel()


    # Return all results for this model.
    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FP": fp,
        "FN": fn,
        "Prediction Time": prediction_time
    }


# --------------------------------------------------
# 8. EVALUATE ALL FOUR MODELS
# --------------------------------------------------

# Store each model's results inside a list.
results = [
    evaluate_model(
        "Normal Naive Bayes",
        normal_nb
    ),

    evaluate_model(
        "Balanced Naive Bayes",
        balanced_nb
    ),

    evaluate_model(
        "Normal Logistic Regression",
        normal_lr
    ),

    evaluate_model(
        "Balanced Logistic Regression",
        balanced_lr
    )
]


# --------------------------------------------------
# 9. CREATE A RESULTS TABLE
# --------------------------------------------------

# Convert our list of results into a pandas table.
results_df = pd.DataFrame(results)


# Round the decimal values
# so the table is easier to read.
results_df = results_df.round(4)


# Print the final comparison table.
print("\nMODEL COMPARISON\n")
print(results_df.to_string(index=False))