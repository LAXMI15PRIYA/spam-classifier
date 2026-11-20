# Import FastAPI.
# We use FastAPI to create our web API.
from fastapi import FastAPI


# Import BaseModel from Pydantic.
# BaseModel helps us define what data
# the user must send to the API.
from pydantic import BaseModel


# Import joblib.
# We use it to load our previously saved
# TF-IDF + Naive Bayes Pipeline.
import joblib


# --------------------------------------------------
# 1. CREATE FASTAPI APPLICATION
# --------------------------------------------------

# Create the FastAPI application.
#
# app is now our API object.
#
# Later Uvicorn will run this object.
app = FastAPI(
    title="SMS Spam Classifier API"
)


# --------------------------------------------------
# 2. LOAD SAVED MACHINE-LEARNING PIPELINE
# --------------------------------------------------

# Load the trained Pipeline from the models folder.
#
# The Pipeline already contains:
#
# TF-IDF
# +
# Balanced Multinomial Naive Bayes
#
# So we do NOT train the model again here.
model = joblib.load(
    "models/spam_classifier_pipeline.joblib"
)


# --------------------------------------------------
# 3. DEFINE REQUEST FORMAT
# --------------------------------------------------

# This class defines what the user
# must send to the /predict endpoint.
#
# We expect one field:
#
# message = SMS text
#
# Example request:
#
# {
#     "message": "Congratulations! You won a prize!"
# }
class SMSRequest(BaseModel):

    # The SMS must be text,
    # so its Python type is str.
    message: str


# --------------------------------------------------
# 4. CREATE HOME ENDPOINT
# --------------------------------------------------

# @app.get("/") means:
#
# When someone visits the root URL
# using a GET request,
# run the function below.
@app.get("/")
def home():

    # Return a simple JSON response.
    # This helps us confirm that
    # the API server is working.
    return {
        "message": "SMS Spam Classifier API is running"
    }


# --------------------------------------------------
# 5. CREATE PREDICTION ENDPOINT
# --------------------------------------------------

# @app.post("/predict") means:
#
# Users send SMS data to:
#
# POST /predict
#
# FastAPI automatically converts the incoming
# JSON into an SMSRequest object.
@app.post("/predict")
def predict_sms(request: SMSRequest):

    # Get the SMS text sent by the user.
    message = request.message


    # Give the raw SMS directly to our Pipeline.
    #
    # We place message inside a list because
    # scikit-learn predict() expects multiple samples.
    #
    # Pipeline automatically performs:
    #
    # raw text
    # ↓
    # TF-IDF
    # ↓
    # Naive Bayes
    prediction = model.predict(
        [message]
    )[0]


    # [0] means:
    #
    # model.predict() returns something like:
    #
    # ["spam"]
    #
    # We only predicted one message,
    # so [0] gives:
    #
    # "spam"


    # Return the original message
    # and model prediction as JSON.
    return {
        "message": message,
        "prediction": prediction
    }