import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Sample dataset of courses
df = pd.read_excel('datakursus.xlsx')

# Display the dataset
print(df)

# Step 1: Preprocess the text data
# We will use the course description to build our model.

# Step 2: Vectorize the course descriptions using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['course_description'])

# Step 3: Calculate cosine similarity between courses
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 4: Function to get course recommendations
def get_recommendations(course_title, cosine_sim=cosine_sim):

    # Get the index of the course that matches the title
    idx = df[df['course_title'] == course_title].index[0]
    
    # Get the pairwise similarity scores of all courses with that course
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the courses based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 5 most similar courses
    sim_scores = sim_scores[1:6]
    
    # Get the course indices
    course_indices = [i[0] for i in sim_scores]
    
    # Return the top 5 most similar courses
    return df['course_title'].iloc[course_indices]

# Example: Get recommendations for "Introduction to Python"
recommendations = get_recommendations('Introduction to Data Science')
print("Recommended courses:")
print(recommendations)

# Streamlit app
st.title('Random Forest Classifier Prediction')

# Create input fields for user input based on the features

input_value = st.text_input(f'Enter Title')
    

# If the user clicks the 'Predict' button
if st.button('Recommend'):
    
    # Make predictions
    prediction = get_recommendations(input_value)
    
    # Display the prediction
    st.write(prediction)

# Run the app
if __name__ == '__main__':
    st.run()
