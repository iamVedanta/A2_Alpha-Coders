import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from textblob import TextBlob
import numpy as np
from io import BytesIO
import base64
import csv
import random



def analyse_karnataka():
    df=pd.read_csv(r"C:\Users\Sumukha S\OneDrive\Desktop\infothon2\venv\karnataka\chat.csv")
#     # Split the data into training and testing sets
#     train_data, test_data, train_labels, test_labels = train_test_split(
#         df["comment"], df["sentiment"], test_size=0.2, random_state=42
#     )

#     # Vectorize the comments using CountVectorizer
#     vectorizer = CountVectorizer()
#     train_vectors = vectorizer.fit_transform(train_data)
#     test_vectors = vectorizer.transform(test_data)

#     # Train a logistic regression model
#     model = LogisticRegression()
#     model.fit(train_vectors, train_labels)

#     # Make predictions on the test set
#     predictions = model.predict(test_vectors)

#     # Evaluate the model
#     accuracy = accuracy_score(test_labels, predictions)
#     print(f"Accuracy: {accuracy}")

#     # Inference on new data
#     new_comments = [
#         "The BJP's plans for environmental conservation are impressive. A cleaner environment is crucial for our well-being. #GreenInitiatives",
#         "bjp is idiot",
#         "Unimpressed by the lack of commitment to addressing slum development. #UrbanPoverty",
#         "The BJP's approach to handling the pandemic in Delhi is questionable. #COVID19Response",
#         "Disappointed with the lack of transparency in the BJP's plans for Delhi. #TransparencyNeeded",
#         "The speech lacked empathy for the struggles of the common man in Delhi. #DisconnectedLeadership",
#         "Concerned about the lack of focus on social justice issues in Delhi. #InequalityConcerns",

#     ]
#     new_vectors = vectorizer.transform(new_comments)
#     new_predictions = model.predict(new_vectors)

#     all_comments=[]

#     # Display the inference results
#     for comment, prediction in zip(new_comments, new_predictions):
#         sentiment = "Positive" if prediction == 1 else "Negative"
#         print(f"Comment: {comment}\nSentiment: {sentiment}\n")
#         all_comments.append({"comment": comment, "sentiment": prediction })
        
#     # print(all_comments)



#     # print(all_comments)

#     # Generate 200 positive comments
#     positive_comments = []

#     # Generate 200 negative comments
#     negative_comments = []


#     for i in all_comments:
#         if i["sentiment"] ==1:
#             positive_comments.append(i["comment"])
#         else:
#             negative_comments.append(i["comment"])

#     # print(positive_comments)
#     # print(negative_comments)

#     #comments = positive_comments + negative_comments


#     #sentiments = [TextBlob(comment).sentiment.polarity for comment in comments]


#     #positive_comments_count = np.sum(np.array(sentiments) >= 0)
#     #negative_comments_count = np.sum(np.array(sentiments) < 0)


#     labels = ['Positive', 'Negative']
#     #counts = [positive_comments_count, negative_comments_count]
#     counts=[len(positive_comments),len(negative_comments)]

#     plt.bar(labels, counts, color=['green', 'red'])
#     plt.title('Sentiment Analysis of Political Speech Comments of Karnataka')
#     plt.xlabel('Sentiment')
#     plt.ylabel('Number of Comments')
   








    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        df["comment"], df["sentiment"], test_size=0.4, random_state=0
    )

    # Vectorize the comments using CountVectorizer
    vectorizer = CountVectorizer()
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(train_vectors, train_labels)

    # Make predictions on the training set
    train_predictions = model.predict(train_vectors)

    # Evaluate the model on the training data
    train_accuracy = accuracy_score(train_labels, train_predictions)
    print(f"Training Accuracy: {train_accuracy}")

    # Make predictions on the test set
    test_predictions = model.predict(test_vectors)

    # Evaluate the model on the testing data
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Testing Accuracy: {test_accuracy}")

    # Inference on new data
    new_comments = [
        "The BJP's plans for environmental conservation are impressive. A cleaner environment is crucial for our well-being. #GreenInitiatives",
        "bjp is idiot",
        "Unimpressed by the lack of commitment to addressing slum development. #UrbanPoverty",
        "The BJP's approach to handling the pandemic in Delhi is questionable. #COVID19Response",
        "Disappointed with the lack of transparency in the BJP's plans for Bangalore. #TransparencyNeeded",
        "The speech lacked empathy for the struggles of the common man in Bangalore. #DisconnectedLeadership",
        "Concerned about the lack of focus on social justice issues in Bangalore. #InequalityConcerns",
    ]
    new_vectors = vectorizer.transform(new_comments)
    new_predictions = model.predict(new_vectors)

    all_comments = []

    # Display the inference results
    for comment, prediction in zip(new_comments, new_predictions):
        sentiment = "Positive" if prediction == 1 else "Negative"
        print(f"Comment: {comment}\nSentiment: {sentiment}\n")
        all_comments.append({"comment": comment, "sentiment": prediction})

    print(all_comments)

    # Generate 200 positive comments
    positive_comments = []

    # Generate 200 negative comments
    negative_comments = []


    for i in all_comments:
        if i["sentiment"] ==1:
            positive_comments.append(i["comment"])
        else:
            negative_comments.append(i["comment"])

    # Extract counts from your analysis
    positive_count = len(positive_comments)
    negative_count = len(negative_comments)
    total_count = positive_count + negative_count

    # Calculate percentages
    positive_percentage = (positive_count / total_count) * 100
    negative_percentage = (negative_count / total_count) * 100

    # Create the bar chart
    labels = ['Positive', 'Negative']
    counts = [positive_count, negative_count]
    colors = ['green', 'red']

    plt.bar(labels, counts, color=colors)

    # Add percentages as annotations above the bars
    for i, count in enumerate(counts):
        plt.annotate(f"{counts[i]} ({round(counts[i]/total_count*100, 1)}%)",
        xy=(i, count + 2),  # Adjust offset for better positioning
        ha='center', va='center',
        fontweight='bold')  # Make percentages more prominent

    # Customize the chart
    plt.title('Sentiment Analysis of Political Speech Comments of Delhi')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Comments')
    plt.ylim(0, max(counts) + 5)  # Adjust y-axis limit for annotations
    # plt.show()

    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_str = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.switch_backend('agg')
        # plt.close()
    return img_str
            