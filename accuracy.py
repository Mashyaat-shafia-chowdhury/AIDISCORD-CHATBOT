import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Sample data
inputs = ["Hi", "Hello", "Hey", "What's the weather like?", "Tell me the weather", "Weather forecast", "Goodbye", "See you later", "Bye"]
labels = ["Greeting", "Greeting", "Greeting", "RequestWeather", "RequestWeather", "RequestWeather", "Farewell", "Farewell", "Farewell"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.3, random_state=42)

# Text preprocessing
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train_tfidf, y_train)

# Predict the test set
y_pred = clf.predict(X_test_tfidf)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Display results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
