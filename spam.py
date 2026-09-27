import pandas as pd

df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'text'])
print(df.shape)
print(df['label'].value_counts())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vectorized train shape:", X_train_vec.shape)
print("Vectorized test shape:", X_test_vec.shape)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_vec, y_train)

print("Model trained.")
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test_vec)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

