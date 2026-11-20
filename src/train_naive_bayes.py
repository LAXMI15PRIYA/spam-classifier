# Import pandas so we can work with our CSV dataset.
# pandas loads the CSV into a DataFrame.
# pd is the short name we use for pandas.
import pandas as pd


# Import train_test_split to create training and testing data.
# Training data teaches our model.
# Test data stays unseen until evaluation.
from sklearn.model_selection import train_test_split


# Import TF-IDF to convert SMS text into numbers.
# Machine-learning models cannot directly understand raw sentences.
# TF-IDF gives numerical importance values to words.
from sklearn.feature_extraction.text import TfidfVectorizer


# Import Multinomial Naive Bayes.
# This will be our first spam classification model.
# It works well with text features such as TF-IDF.
from sklearn.naive_bayes import MultinomialNB


from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Store the location of our CSV dataset.
# The path is relative to D:\spam-classifier.
# spam.csv is inside the data folder.
file_path = "data/spam.csv"


# Read the CSV file into pandas.
# latin-1 is the encoding used to correctly read this dataset.
# df now stores the entire dataset as a table.
df = pd.read_csv(file_path, encoding="latin-1")


# Keep only v1 and v2.
# v1 contains ham/spam labels.
# v2 contains the actual SMS messages.
df = df[["v1", "v2"]]


# Give the two columns clearer names.
# v1 becomes label.
# v2 becomes message.
df.columns = ["label", "message"]


# Remove repeated rows.
# This reduces the chance of the same message appearing in train and test.
# The remaining rows are unique messages.
df = df.drop_duplicates()


# X contains our input.
# The input is the SMS message.
# The model will use X to learn patterns.
X = df["message"]


# y contains the correct answer.
# Each answer is either ham or spam.
# The model learns to predict y using X.
y = df["label"]


# Split messages and labels into training and testing sets.
# 80% is used for training.
# 20% is kept for final testing.
X_train, X_test, y_train, y_test = train_test_split(
    X,                       # SMS messages
    y,                       # Correct labels
    test_size=0.20,          # 20% goes to test data
    random_state=42,         # Same split each time
    stratify=y               # Preserve ham/spam ratio
)


# Create the TF-IDF converter.
# It has not learned any vocabulary yet.
# It will learn only from training messages.
tfidf = TfidfVectorizer()


# Learn vocabulary from training messages.
# Convert training messages into TF-IDF numbers.
# fit_transform means learn + convert.
X_train_tfidf = tfidf.fit_transform(X_train)


# Convert test messages using the same learned vocabulary.
# We do NOT fit on test data.
# This prevents data leakage.
X_test_tfidf = tfidf.transform(X_test)


# Create our Multinomial Naive Bayes model.
# At this moment the model has not learned anything.
# We store the model in the variable nb_model.
nb_model = MultinomialNB()


# Train the Naive Bayes model.
# X_train_tfidf contains message features.
# y_train contains the correct ham/spam answers.
nb_model.fit(X_train_tfidf, y_train)


# Tell us that training completed successfully.
# If Python reaches this line without an error, model.fit() worked.
# Our model is now trained.
print("Naive Bayes model trained successfully!")


# Ask the trained model to predict the test messages.
# The model has never trained on these messages.
# Predictions will contain ham or spam.
y_pred = nb_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, pos_label="spam")

recall = recall_score(y_test, y_pred, pos_label="spam")

f1 = f1_score(y_test, y_pred, pos_label="spam")

cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])


print("\nAccuracy:", f"{accuracy:.4f}")

print("Spam Precision:", f"{precision:.4f}")

print("Spam Recall:", f"{recall:.4f}")

print("Spam F1-score:", f"{f1:.4f}")

print("\nConfusion Matrix:")
print(cm)


# Show the first 5 model predictions.
# [:5] means only the first five results.
# These are answers produced by our model.
print("\nFirst 5 predictions:")
print(y_pred[:5])


# Show the actual correct labels for those same messages.
# iloc[:5] selects the first five test labels.
# We can visually compare prediction vs actual answer.
print("\nFirst 5 actual labels:")
print(y_test.iloc[:5].values)