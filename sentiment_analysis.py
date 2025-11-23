import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1️⃣ Load Dataset
df = pd.read_csv("movie_reviews.csv")

# 2️⃣ Preprocessing (very simple for minor project)
df['review'] = df['review'].str.lower()

# 3️⃣ Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# 4️⃣ TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5️⃣ Train Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 6️⃣ Evaluate Model
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("Model Training Completed!")
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)

# 7️⃣ Predict Sentiment for New Review
while True:
    review = input("\nEnter a movie review (or 'exit' to stop): ")
    if review.lower() == "exit":
        break
    
    review_vec = vectorizer.transform([review])
    prediction = model.predict(review_vec)[0]
    print("Sentiment:", prediction)
