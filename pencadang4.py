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

# Step 3: Function to get course recommendations using a keyword search
def get_recommendations(keyword, cosine_sim=cosine_sim):
    # Convert both input and course titles to lowercase for case-insensitive matching
    keyword_lower = keyword.lower()
    df['course_title_lower'] = df['course_title'].str.lower()

    # Search for courses that contain the keyword in the title
    matching_courses = df[df['course_title_lower'].str.contains(keyword_lower)]

    if matching_courses.empty:
        return "No courses found matching the keyword. Please try another keyword."

    # Get the indices of the matching courses
    matching_indices = matching_courses.index.tolist()

    # Calculate similarity for all matching courses
    recommendations = []
    for idx in matching_indices:
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Get top 5 most similar courses
        course_indices = [i[0] for i in sim_scores]

        # Get the most similar courses for each matching course
        similar_courses = df[['course_title', 'Organization', 'course_url']].iloc[course_indices]

        # Format course recommendations with clickable hyperlinks
        for _, row in similar_courses.iterrows():
            course_title = row['course_title']
            organization = row['Organization']
            course_url = row['course_url']
            link = f"[{course_title}]({course_url})"
            recommendations.append(f"{link} by {organization}")

    return recommendations

# Streamlit app
st.title('Course Recommender System')

# Create input field for user to enter a keyword
input_value = st.text_input('Enter a keyword (e.g., Python, Data, Science)')

# If the user clicks the 'Recommend' button
if st.button('Recommend'):
    # Get recommendations based on the input keyword
    recommendations = get_recommendations(input_value)

    # Display the recommendations
    if isinstance(recommendations, list):
        for rec in recommendations:
            st.markdown(rec, unsafe_allow_html=True)
    else:
        st.write(recommendations)