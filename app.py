import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv(r'C:\\Users\\madhavmadupu\\Desktop\\ML_Projects\\SongsRecommendationSystem\\spotify-2023.csv', encoding='iso-8859-1')  # Replace 'your_data.csv' with your file path

# Preprocessing steps (similar to the previous code)

# Encoding categorical columns: 'track_name' and 'artist(s)_name'
label_encoder = LabelEncoder()
df['track_name_encoded'] = label_encoder.fit_transform(df['track_name'])
df['artist_name_encoded'] = label_encoder.fit_transform(df['artist(s)_name'])

columns_for_similarity = [
    'bpm', 'danceability_%', 'valence_%', 'energy_%',
    'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%',
    'track_name_encoded', 'artist_name_encoded'
]

scaler = StandardScaler()
df[columns_for_similarity] = scaler.fit_transform(df[columns_for_similarity])

cosine_sim = cosine_similarity(df[columns_for_similarity])

# Streamlit app
st.title('Song Recommender')

selected_song = st.sidebar.selectbox('Select a Song', df['track_name'].values)

# Get index of selected song
song_index = df[df['track_name'] == selected_song].index[0]

# Function to get song recommendations based on cosine similarity
def get_song_recommendations(song_index, similarity_matrix, num_recommendations=5):
    sim_scores = list(enumerate(similarity_matrix[song_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_similar_song_indices = [x[0] for x in sim_scores[1:num_recommendations+1]]
    return df.iloc[top_similar_song_indices]

if st.button('Show Recommendations'):
    recommendations = get_song_recommendations(song_index, cosine_sim)
    st.write("Recommendations for", selected_song)
    st.dataframe(recommendations[['track_name', 'artist(s)_name', 'streams']])
