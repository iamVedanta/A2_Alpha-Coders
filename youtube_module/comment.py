import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK resources (if not already downloaded)
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Assuming you have a list of stop words (combined_stopwords)
combined_stopwords = set(stopwords.words('english'))

def tweet_cleaner(comment, remove_stopwords=True):
    new_text = re.sub(r"'s\b", " is", comment)
    new_text = re.sub("#", "", new_text)
    new_text = re.sub("@[A-Za-z0-9]+", "", new_text)
    new_text = re.sub(r"http\S+", "", new_text)
    new_text = re.sub(r"[^a-zA-Z]", " ", new_text)
    new_text = contractions.fix(new_text)
    new_text = new_text.lower().strip()

    tokens = word_tokenize(new_text)

    if remove_stopwords:
        tokens = [token for token in tokens if token not in combined_stopwords]

    lemmatizer = WordNetLemmatizer()
    cleaned_text = ' '.join([lemmatizer.lemmatize(token) for token in tokens])

    return cleaned_text

def create_bag_of_words(comments, remove_stopwords=True):
    cleaned_comments = [tweet_cleaner(comment, remove_stopwords) for comment in comments]

    vectorizer = CountVectorizer()
    bag_of_words = vectorizer.fit_transform(cleaned_comments)

    return bag_of_words

# Your new comments
new_comments = [
    "I luv this vid! Awesome content.",
    "Worst video ever. Waste of time.",
    "Provided info is very helpful.",
    "Disappointed with the video quality."
]

# Clean and vectorize the new comments
bag_of_words_new_comments = create_bag_of_words(new_comments)
df_new_comments = pd.DataFrame(bag_of_words_new_comments.todense())
print(df_new_comments)

# Now you can use df_new_comments for prediction with your trained model
# model.predict(df_new_comments)