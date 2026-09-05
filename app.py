import streamlit as st
import requests

# Page Configuration
st.set_page_config(page_title="Music Recommender System", page_icon="🎵", layout="centered")

st.title("🎵 Indian Music Recommendation System")
st.write("Hindi aur Punjabi gaano ki vibe-based recommendations dhoondho!")

# Input box for song name
song_name = st.text_input("Enter a song name (e.g., Tum Hi Ho, Pagal):")

if st.button("Recommend"):
    if not song_name.strip():
        st.warning("Please enter a valid song name.")
    else:
        with st.spinner("Finding similar vibes..."):
            try:
                # FastAPI backend ko request bhejna
                response = requests.post(
                    "http://127.0.0.1:8000/recommend",
                    json={"song_name": song_name}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])
                    
                    st.success("Here are your recommendations!")
                    st.markdown("---")
                    
                    # Cards/Rows format me recommendations dikhana
                    for idx, rec in enumerate(recommendations, 1):
                        st.markdown(f"### {idx}. {rec['song_name']}")
                        st.write(f"🎤 **Artist:** {rec['artist']}")
                        st.write(f"🌐 **Language:** {rec['language']}")
                        st.write(f"⚡ **Vibe Match:** {rec['match_percentage']}")
                        st.markdown("---")
                        
                else:
                    error_detail = response.json().get("detail", "Song not found in dataset.")
                    st.error(f"Error: {error_detail}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Make sure uvicorn is running!")