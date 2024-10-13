import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk

# Make sure you download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load your dataset
data = pd.read_csv('artStylesdataset.csv')  # Ensure correct file path

# Step 2: Preprocess the descriptions and titles together
def preprocess_text(text):
    # Remove punctuation and non-alphabet characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Combine the 'Title' and 'Description' into one column and apply preprocessing
data['combined_text'] = data['Title'] + " " + data['Description']
data['cleaned_combined_text'] = data['combined_text'].apply(preprocess_text)

# Step 3: Train-Test Split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Step 4: Vectorize the cleaned combined texts using CountVectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X_train = vectorizer.fit_transform(train_data['cleaned_combined_text'])
X_test = vectorizer.transform(test_data['cleaned_combined_text'])

# Step 5: Apply LDA to the training data
n_topics = len(train_data['Style'].unique())  # Number of topics = number of unique styles
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X_train)

# Step 6: Assign LDA topics to the actual styles in the training data
train_topic_distribution = lda.transform(X_train)
train_predicted_topics = train_topic_distribution.argmax(axis=1)

# Map LDA topics to actual styles based on the training data
# Find which style corresponds to which topic
topic_to_style_mapping = {}
for topic in range(n_topics):
    # Find the most common style for each topic in the training set
    common_style = train_data.loc[train_predicted_topics == topic, 'Style'].mode()[0]
    topic_to_style_mapping[topic] = common_style

# Step 7: Predict styles for the test data
test_topic_distribution = lda.transform(X_test)
test_predicted_topics = test_topic_distribution.argmax(axis=1)

# Convert the predicted topics to actual styles using the topic-to-style mapping
test_predicted_styles = [topic_to_style_mapping[topic] for topic in test_predicted_topics]

# Step 8: Calculate Accuracy by comparing predicted styles to actual styles in the test data
accuracy = accuracy_score(test_data['Style'], test_predicted_styles)

print(f"Accuracy of LDA style prediction: {accuracy:.2f}")
