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

# Step 3: Vectorize the cleaned combined texts using CountVectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(data['cleaned_combined_text'])

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

# Step 7: Function to recommend similar works based on the predicted style and category
def recommend_similar_works(predicted_style, user_category, start=0, count=5):
    # Filter artworks of the same style and category
    same_style_works = data[(data['Style'] == data['Style'].unique()[predicted_style]) &
                            (data['Category'].str.lower() == user_category.lower())]
    
    # Display a batch of 5 works
    recommended_works = same_style_works.iloc[start:start+count]
    
    if not recommended_works.empty:
        print("\nSimilar Works:")
        for index, row in recommended_works.iterrows():
            print(f"Title: {row['Title']}, Description: {row['Description'][:100]}...")
    else:
        print("No more similar works available in this category.")

    # Return the total number of works available in this style and category
    return len(same_style_works)

# Step 8: Ask for user input
user_category = input("Enter the category (e.g., Art, Sculpture, etc.): ")
user_title = input("Enter the art title: ")
user_description = input("Enter the art description: ")

# Predict the style
predicted_style = predict_style(user_description)

# Map the predicted topic back to actual art styles
predicted_style_name = data['Style'].unique()[predicted_style]

print(f"\nThe predicted Art Movement for '{user_title}' is: {predicted_style_name}")

# Step 9: Recommend similar works initially and ask if the user wants more
start_index = 0
batch_size = 5
total_recommendations = recommend_similar_works(predicted_style, user_category, start=start_index, count=batch_size)

# Loop to allow the user to load more recommendations
while True:
    if total_recommendations > start_index + batch_size:
        load_more = input("\nWould you like to load more similar works? (yes/no): ").strip().lower()
        if load_more == 'yes':
            start_index += batch_size
            recommend_similar_works(predicted_style, user_category, start=start_index, count=batch_size)
        else:
            print("No more works will be loaded.")
            break
    else:
        print("All similar works have been displayed.")
        break
