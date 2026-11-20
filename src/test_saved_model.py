# Import joblib.
# We need it to open our saved ML Pipeline.
import joblib


# --------------------------------------------------
# 1. LOAD SAVED MODEL
# --------------------------------------------------

# Load the Pipeline we previously trained and saved.
#
# We do NOT need to train again.
model = joblib.load(
    "models/spam_classifier_pipeline.joblib"
)


# Tell us that loading worked.
print("Saved model loaded successfully!")


# --------------------------------------------------
# 2. CREATE NEW SMS MESSAGES
# --------------------------------------------------

# These are completely new messages
# we want our saved model to classify.
new_messages = [

    # Likely normal message.
    "Hi, are we meeting at 6 PM today?",

    # Likely spam message.
    "Congratulations! You have won a free cash prize. Claim now!",

    # Likely normal message.
    "Please send me the notes when you are free."
]


# --------------------------------------------------
# 3. MAKE PREDICTIONS
# --------------------------------------------------

# Give raw SMS text directly to the Pipeline.
#
# The saved Pipeline automatically:
#
# 1. converts each message to TF-IDF
# 2. sends the numbers to Naive Bayes
# 3. returns ham or spam
predictions = model.predict(
    new_messages
)


# --------------------------------------------------
# 4. PRINT RESULTS
# --------------------------------------------------

# zip() pairs each message
# with its prediction.
for message, prediction in zip(
    new_messages,
    predictions
):

    print("\nMessage:", message)

    print("Prediction:", prediction)