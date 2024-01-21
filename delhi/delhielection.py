import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def analyse_delhi():
    # import pandas as pd
    # from sklearn.model_selection import train_test_split
    # from sklearn.feature_extraction.text import CountVectorizer
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import accuracy_score

    # Your provided data
    df = pd.read_csv(r"C:\Users\Sumukha S\OneDrive\Desktop\infothon2\venv\delhi\delhi.csv")
    # Split the data into training and testing sets



    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        df["comment"], df["sentiment"], test_size=0.2,stratify=df['sentiment'], random_state=0
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
    "The BJP's focus on environmental conservation is crucial for the well-being of Delhi. #GreenDelhi",
    "Disappointed with the slow progress on traffic management by the BJP in Delhi. #TrafficIssues",
    "The BJP's strategies for affordable housing in Delhi need more clarity and implementation. #HousingConcerns",
    "Encouraged by BJP's efforts to promote cultural diversity in Delhi. #CulturalHarmony",
    "The BJP's handling of the education system in Delhi is a disappointment. No real improvements seen. #BJPEducationFailure",
    "Transparency seems to be a major issue with BJP's governance in Delhi. Lack of openness raises concerns. #TransparencyIssues",
    "Despite promises, women's safety remains a major concern in Delhi under the BJP's rule. #BJPWomenSafetyFail",
    "The BJP's environmental policies in Delhi lack substance. Pollution levels continue to rise. #BJPGreenFail",
    "Traffic conditions in Delhi have worsened under BJP's governance. No effective solutions in sight. #BJPTrafficFailure",
    "Healthcare services in Delhi have not seen significant improvements under the BJP. #BJPHealthcareFail",
    "Job creation promises by the BJP in Delhi have not translated into real opportunities. #BJPUnemploymentFailure",
    "Digital governance initiatives by the BJP in Delhi lack efficiency. #BJPDigitalGovernanceFail",
    "Affordable housing remains a distant dream in Delhi, despite BJP's assurances. #BJPHousingFailure",
    "BJP's efforts towards promoting cultural diversity in Delhi are insufficient and lack impact. #BJPCulturalDiversityFail",
    ]
    new_vectors = vectorizer.transform(new_comments)
    new_predictions = model.predict(new_vectors)

    all_comments = []

    # Display the inference results
    for comment, prediction in zip(new_comments, new_predictions):
        sentiment = "Positive" if prediction == 1 else "Negative"
        print(f"Comment: {comment}\nSentiment: {sentiment}\n")
        all_comments.append({"comment": comment, "sentiment": prediction})

    


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
    # plt.show()  # Uncomment this line if you want to display the plot

    # Save the plot as a base64-encoded image
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_str = base64.b64encode(img_buf.read()).decode('utf-8')

    # Close the plot to release resources
    plt.close()

    return img_str