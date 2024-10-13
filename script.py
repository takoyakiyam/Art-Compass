import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk

# Make sure you download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load your dataset
data = pd.read_csv('artdatasetsample.csv')

# Step 2: Preprocess the descriptions
def preprocess_text(text):
    # Remove punctuation and non-alphabet characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Apply preprocessing to each description
data['cleaned_description'] = data['Description'].apply(preprocess_text)

# Step 3: Vectorize the cleaned descriptions using CountVectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(data['cleaned_description'])

# Step 4: Apply LDA
n_topics = len(data['Style'].unique())  # Assuming the number of topics is the number of unique styles
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)

# Step 5: Get the top words for each topic (optional, to interpret topics)
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)

# Step 6: Function to predict the style of a new description
def predict_style(new_description):
    # Preprocess the new description
    cleaned_desc = preprocess_text(new_description)
    # Vectorize the new description
    new_vec = vectorizer.transform([cleaned_desc])
    # Predict the topic (style)
    topic_distribution = lda.transform(new_vec)
    predicted_topic = topic_distribution.argmax(axis=1)[0]
    return predicted_topic

# Step 7: Ask for user input
user_title = input("Enter the art title: ")
user_description = input("Enter the art description: ")

# Predict the style
predicted_style = predict_style(user_description)

# Map the predicted topic back to actual art styles
predicted_style_name = data['Style'].unique()[predicted_style]

print(f"The predicted style for '{user_title}' is: {predicted_style_name}")
