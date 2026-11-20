# Import pandas.
# We use pandas to load the dataset
# and create a table of wrong predictions.
import pandas as pd


# Split data into training and testing sets.
from sklearn.model_selection import train_test_split


# Convert SMS text into TF-IDF numeric features.
from sklearn.feature_extraction.text import TfidfVectorizer


# Import Multinomial Naive Bayes.
from sklearn.naive_bayes import MultinomialNB


# This calculates balanced sample weights.
# Spam is the smaller class, so spam examples
# receive more importance while training.
from sklearn.utils.class_weight import compute_sample_weight


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

# Load the SMS dataset.
df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)


# Keep only the useful columns.
df = df[["v1", "v2"]]


# Rename them to easier names.
df.columns = ["label", "message"]


# Remove duplicate messages.
df = df.drop_duplicates()


# X contains the SMS text.
X = df["message"]


# y contains the correct ham/spam labels.
y = df["label"]


# --------------------------------------------------
# 2. TRAIN / TEST SPLIT
# --------------------------------------------------

# Split the data exactly the same way
# as in our previous experiments.
#
# random_state=42 keeps the same split.
# stratify=y keeps the ham/spam ratio similar.
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

# Create the TF-IDF vectorizer.
tfidf = TfidfVectorizer()


# Learn the vocabulary ONLY from training messages
# and convert training messages into numbers.
X_train_tfidf = tfidf.fit_transform(X_train)


# Convert test messages using the vocabulary
# already learned from training data.
X_test_tfidf = tfidf.transform(X_test)


# --------------------------------------------------
# 4. BALANCE THE TRAINING DATA
# --------------------------------------------------

# Create a weight for every training message.
#
# Because spam is rare,
# spam messages receive more importance.
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# --------------------------------------------------
# 5. TRAIN BALANCED NAIVE BAYES
# --------------------------------------------------

# Create the Naive Bayes model.
model = MultinomialNB()


# Train the model.
#
# sample_weight tells Naive Bayes
# to pay more attention to rare spam examples.
model.fit(
    X_train_tfidf,
    y_train,
    sample_weight=sample_weights
)


# --------------------------------------------------
# 6. MAKE TEST PREDICTIONS
# --------------------------------------------------

# Predict ham or spam for every test message.
y_pred = model.predict(X_test_tfidf)


# --------------------------------------------------
# 7. CREATE ERROR-ANALYSIS TABLE
# --------------------------------------------------

# Create a new table containing:
#
# message   = original SMS
# actual    = correct answer
# predicted = model's answer
results = pd.DataFrame({
    "message": X_test.values,
    "actual": y_test.values,
    "predicted": y_pred
})


# --------------------------------------------------
# 8. FIND ALL WRONG PREDICTIONS
# --------------------------------------------------

# Keep only rows where:
#
# actual != predicted
#
# Example:
# actual = spam
# predicted = ham
errors = results[
    results["actual"] != results["predicted"]
]


# Print the total number of errors.
print(
    "\nTotal wrong predictions:",
    len(errors)
)


# --------------------------------------------------
# 9. FIND FALSE POSITIVES
# --------------------------------------------------

# False Positive means:
#
# Actual     = ham
# Predicted  = spam
#
# A normal legitimate message
# was wrongly sent to spam.
false_positives = results[
    (results["actual"] == "ham") &
    (results["predicted"] == "spam")
]


# Print how many false positives occurred.
print(
    "False Positives:",
    len(false_positives)
)


# --------------------------------------------------
# 10. FIND FALSE NEGATIVES
# --------------------------------------------------

# False Negative means:
#
# Actual     = spam
# Predicted  = ham
#
# A real spam message escaped detection.
false_negatives = results[
    (results["actual"] == "spam") &
    (results["predicted"] == "ham")
]


# Print how many false negatives occurred.
print(
    "False Negatives:",
    len(false_negatives)
)


# --------------------------------------------------
# 11. PRINT FALSE POSITIVE EXAMPLES
# --------------------------------------------------

print("\n--- FALSE POSITIVES ---")


# Show the first 10 normal messages
# that were wrongly predicted as spam.
for message in false_positives["message"].head(10):

    print("\nActual: ham")

    print("Predicted: spam")

    print("Message:", message)


# --------------------------------------------------
# 12. PRINT FALSE NEGATIVE EXAMPLES
# --------------------------------------------------

print("\n--- FALSE NEGATIVES ---")


# Show the spam messages
# that were wrongly predicted as ham.
for message in false_negatives["message"].head(10):

    print("\nActual: spam")

    print("Predicted: ham")

    print("Message:", message)