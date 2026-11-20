# pandas helps us work with table-like data.
# Our CSV becomes a pandas DataFrame.
# pd is just a short name for pandas.
import pandas as pd


# train_test_split divides data into train and test parts.
# Training data is used for learning.
# Test data is used for final checking.
from sklearn.model_selection import train_test_split


# TfidfVectorizer converts text into numbers.
# ML models cannot directly understand sentences.
# TF-IDF gives importance scores to words.
from sklearn.feature_extraction.text import TfidfVectorizer


# Save our dataset location in a variable.
# This is a relative path from D:\spam-classifier.
# The CSV is inside the data folder.
file_path = "data/spam.csv"


# read_csv opens the CSV file.
# latin-1 tells pandas how to decode the text.
# df now contains our dataset as a table.
df = pd.read_csv(file_path, encoding="latin-1")


# Our original CSV has extra unwanted columns.
# We only need v1 and v2.
# v1 = label and v2 = SMS message.
df = df[["v1", "v2"]]


# v1 and v2 are unclear names.
# Change v1 to label and v2 to message.
# This makes our code easier to understand.
df.columns = ["label", "message"]


# Some SMS rows appear more than once.
# drop_duplicates removes repeated rows.
# This helps prevent the same message appearing in train and test.
df = df.drop_duplicates()


# X stores the input information.
# Our input is the actual SMS message.
# The model will use X to make predictions.
X = df["message"]


# y stores the correct answer.
# The answer is either ham or spam.
# The model learns to predict y from X.
y = df["label"]


# Split all messages and labels into train and test sets.
# 80% will be used for training.
# 20% will be kept unseen for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X,                       # All SMS messages
    y,                       # Correct ham/spam answers
    test_size=0.20,          # 20% goes to test set
    random_state=42,         # Gives the same split each time
    stratify=y               # Keeps similar ham/spam ratio
)


# Create an empty TF-IDF converter.
# It has not learned any words yet.
# It will learn only from training messages.
tfidf = TfidfVectorizer()


# fit = learn vocabulary and word importance from X_train.
# transform = convert X_train messages into numbers.
# fit_transform does both jobs together.
X_train_tfidf = tfidf.fit_transform(X_train)


# Test data must NOT teach TF-IDF anything.
# We only use the vocabulary learned from X_train.
# Therefore X_test uses transform(), not fit_transform().
X_test_tfidf = tfidf.transform(X_test)


# shape tells us rows and columns.
# Rows = number of training messages.
# Columns = number of TF-IDF word features.
print("Training TF-IDF shape:", X_train_tfidf.shape)


# This shows the size of the test TF-IDF matrix.
# Rows = number of test messages.
# Columns should match training TF-IDF columns.
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# vocabulary_ contains the words learned from X_train.
# len() counts how many words/features were learned.
# This is our TF-IDF vocabulary size.
print("Vocabulary size:", len(tfidf.vocabulary_))


# get_feature_names_out() gives the learned vocabulary words.
# [:20] means show only the first 20 words.
# This helps us quickly inspect what TF-IDF learned.
print("First 20 words:")
print(tfidf.get_feature_names_out()[:20])
