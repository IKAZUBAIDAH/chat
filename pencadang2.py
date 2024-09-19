import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load dataset of courses
df = pd.read_excel('datakursus.xlsx')

# Step 1: Vectorize the course descriptions using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['course_description'])

# Step 2: Calculate cosine similarity between courses
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 3: Function to get course recommendations (case-insensitive)
def get_recommendations(course_title, cosine_sim=cosine_sim):
    # Convert both input and course titles to lowercase for case-insensitive matching
    course_title_lower = course_title.lower()
    df['course_title_lower'] = df['course_title'].str.lower()

    # Get the index of the course that matches the title
    try:
        idx = df[df['course_title_lower'] == course_title_lower].index[0]
    except IndexError:
        return "Course not found. Please enter a valid course title."

    # Get the pairwise similarity scores of all courses with that course
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the courses based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar courses
    sim_scores = sim_scores[1:6]

    # Get the course indices
    course_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar courses
    return df[['course_title', 'Organization', 'course_url']].iloc[course_indices]
# Streamlit app
st.title('Course Recommender System')

# Create input field for user input based on the course title
input_value = st.text_input('Enter Course Title')

# If the user clicks the 'Recommend' button
if st.button('Recommend'):
    # Get recommendations based on the input (case-insensitive)
    recommendations = get_recommendations(input_value)

    # Display the recommendations
    st.write(recommendations)
