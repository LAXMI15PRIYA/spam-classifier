# Import pandas to load our SMS dataset.
import pandas as pd


# Pipeline connects TF-IDF and Naive Bayes
# into one complete machine-learning workflow.
from sklearn.pipeline import Pipeline


# Convert raw SMS text into TF-IDF numeric features.
from sklearn.feature_extraction.text import TfidfVectorizer


# Import Multinomial Naive Bayes.
from sklearn.naive_bayes import MultinomialNB


# Split our dataset into training and testing data.
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)


# compute_sample_weight gives more importance
# to the minority spam class.
from sklearn.utils.class_weight import compute_sample_weight


# Import evaluation metrics.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer
)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

# Load the SMS spam dataset.
df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)


# Keep only the label and message columns.
df = df[["v1", "v2"]]


# Rename columns for easier understanding.
df.columns = ["label", "message"]


# Remove duplicate messages.
df = df.drop_duplicates()


# X contains the raw SMS messages.
X = df["message"]


# y contains the correct ham/spam answers.
y = df["label"]


# --------------------------------------------------
# 2. TRAIN / TEST SPLIT
# --------------------------------------------------

# Split the data into 80% training and 20% testing.
#
# random_state=42:
# Gives us the same split every time.
#
# stratify=y:
# Keeps similar ham/spam percentages
# in both train and test sets.
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

# Calculate a weight for every training message.
#
# Spam is rare, so spam messages will receive
# a higher training weight.
#
# Ham is common, so ham receives
# a lower relative weight.
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# --------------------------------------------------
# 4. CREATE PIPELINE
# --------------------------------------------------

# Pipeline connects two steps:
#
# Step 1:
# Convert text into TF-IDF features.
#
# Step 2:
# Train Multinomial Naive Bayes.
#
# The names "tfidf" and "model" will also be used
# when defining hyperparameters below.
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
# 5. DEFINE SETTINGS TO TEST
# --------------------------------------------------

# GridSearchCV will try every combination below.
#
# tfidf__ngram_range:
#
# (1, 1) = single words only
#
# Example:
# "free prize" becomes:
# free
# prize
#
# (1, 2) = single words + two-word combinations
#
# free
# prize
# free prize
#
#
# tfidf__min_df:
#
# min_df=1:
# Keep words that appear in at least 1 document.
#
# min_df=2:
# Ignore words that appear in only one document.
#
#
# model__alpha:
#
# alpha controls smoothing in Naive Bayes.
#
# We will let GridSearch determine
# which alpha gives the best F1-score.
param_grid = {

    "tfidf__ngram_range": [
        (1, 1),
        (1, 2)
    ],

    "tfidf__min_df": [
        1,
        2
    ],

    "model__alpha": [
        0.1,
        0.5,
        1.0
    ]
}


# --------------------------------------------------
# 6. CREATE SPAM F1 SCORER
# --------------------------------------------------

# Our labels are strings:
#
# "ham"
# "spam"
#
# We specifically want F1 for the spam class.
#
# make_scorer tells GridSearch:
# "Treat spam as the positive class."
spam_f1_scorer = make_scorer(
    f1_score,
    pos_label="spam"
)


# --------------------------------------------------
# 7. CREATE CROSS-VALIDATION
# --------------------------------------------------

# Use 5-fold cross-validation.
#
# StratifiedKFold makes sure each fold
# keeps roughly the same ham/spam ratio.
#
# shuffle=True mixes the training rows.
#
# random_state=42 makes the folds repeatable.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 8. CREATE GRID SEARCH
# --------------------------------------------------

# GridSearchCV will:
#
# 1. Try every parameter combination.
# 2. Test each combination using 5-fold CV.
# 3. Calculate spam F1.
# 4. Select the combination with the best average F1.
#
# n_jobs=-1:
# Use all available CPU cores to make the search faster.
#
# verbose=1:
# Show some progress while GridSearch is running.
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=spam_f1_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1
)


# --------------------------------------------------
# 9. RUN GRID SEARCH
# --------------------------------------------------

# Train GridSearch using ONLY training data.
#
# model__sample_weight tells Pipeline:
#
# Send these sample weights specifically
# to the Naive Bayes model during training.
#
# X_test is NOT used here.
grid_search.fit(
    X_train,
    y_train,
    model__sample_weight=sample_weights
)


# --------------------------------------------------
# 10. SHOW BEST SETTINGS
# --------------------------------------------------

# Print the hyperparameter combination
# that produced the best average spam F1
# during cross-validation.
print("\nBest Parameters:")
print(grid_search.best_params_)


# Print the best average CV F1-score.
print(
    "\nBest Cross-Validation F1:",
    round(grid_search.best_score_, 4)
)


# --------------------------------------------------
# 11. GET THE BEST MODEL
# --------------------------------------------------

# best_estimator_ is the complete best Pipeline:
#
# best TF-IDF settings
# +
# best Naive Bayes settings
best_model = grid_search.best_estimator_


# --------------------------------------------------
# 12. FINAL TEST PREDICTION
# --------------------------------------------------

# NOW we finally use the unseen test set.
#
# Since TF-IDF is inside the Pipeline,
# we can give raw X_test text directly.
y_pred = best_model.predict(X_test)


# --------------------------------------------------
# 13. FINAL EVALUATION
# --------------------------------------------------

# Overall correctness.
accuracy = accuracy_score(
    y_test,
    y_pred
)


# When model says spam,
# how often is it really spam?
precision = precision_score(
    y_test,
    y_pred,
    pos_label="spam"
)


# Out of all real spam,
# how many did the model catch?
recall = recall_score(
    y_test,
    y_pred,
    pos_label="spam"
)


# Balance between precision and recall.
f1 = f1_score(
    y_test,
    y_pred,
    pos_label="spam"
)


# Confusion matrix:
#
# [[TN, FP],
#  [FN, TP]]
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["ham", "spam"]
)


# --------------------------------------------------
# 14. PRINT FINAL RESULTS
# --------------------------------------------------

print("\nFINAL TUNED MODEL RESULTS")

print(
    "Accuracy:",
    f"{accuracy:.4f}"
)

print(
    "Spam Precision:",
    f"{precision:.4f}"
)

print(
    "Spam Recall:",
    f"{recall:.4f}"
)

print(
    "Spam F1-score:",
    f"{f1:.4f}"
)

print("\nConfusion Matrix:")
print(cm)