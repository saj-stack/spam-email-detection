# Spam Email Detection (Naive Bayes)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample Dataset (Contents)
data = {
    "email": [
        "Earn money from home easily",
        "Please find attached the invoice",
        "Can we reschedule our project discussion to tomorrow?",
        "New iPhone available online at 90% discount",
        "You have been selected for a free vacation camp",
        "Win a free iPhone now, Click here",
        "Your doctor’s appointment is at 2 PM",
        "Increase followers on Instagram instantly",
        "Your Flipkart order has been shipped",
        "Double your income in one week",
        "Your bank statement for August",
        "Click here to claim your gift card now",
        "Your meeting is scheduled at 11 AM tomorrow",
        "Reminder: Don't forget to submit the report tomorrow",
        "Congratulations, You have won a $50,000 lottery",
        "Your leave has been approved",
    ],
    "label": [
        "spam", "acceptable", "acceptable", "spam", "spam",
        "spam", "acceptable", "spam", "acceptable", "spam",
        "acceptable", "spam", "acceptable", "acceptable", "spam",
        "acceptable"
    ]
}

# DataFrame
emails = pd.DataFrame(data)

# Features
X = emails["email"]
y = emails["label"].map({"acceptable": 0, "spam": 1})

# Vectorization
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=10
)

# Training and Prediction
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy & Report
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nReport:\n", classification_report(y_test, y_pred))

# User Input
print("\nEnter the email content to check:")
user_email = [input("Text: ")]

# Prediction
user_vectorized = vectorizer.transform(user_email)
prediction = model.predict(user_vectorized)

# Result
print("\nPrediction:", "Spam" if prediction[0] == 1 else "Not Spam")