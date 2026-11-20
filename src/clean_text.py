# Import pandas for working with the dataset
import pandas as pd

# Import string to get punctuation characters like ! , . ?
import string

# Import stopwords from NLTK
from nltk.corpus import stopwords

# Import nltk so we can download the stopword list
import nltk

# Download the English stopword list if it is not already available
nltk.download("stopwords")

# Store the location of our dataset
file_path = "data/spam.csv"

# Load the CSV using latin-1 encoding
df = pd.read_csv(file_path, encoding="latin-1")

# Keep only the useful columns
df = df[["v1", "v2"]]

# Rename the columns
df.columns = ["label", "message"]

# Create a set of English stopwords
stop_words = set(stopwords.words("english"))


# Create a function to clean one SMS message
def clean_text(text):

    # Convert all letters to lowercase
    text = text.lower()

    # Remove punctuation characters
    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    # Split the sentence into individual words
    words = text.split()

    # Keep only words that are not stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Join the remaining words back into one sentence
    cleaned_text = " ".join(words)

    # Return the cleaned message
    return cleaned_text


# Apply our cleaning function to every SMS
df["clean_message"] = df["message"].apply(clean_text)

# Show original and cleaned messages
print(df[["message", "clean_message"]].head())

