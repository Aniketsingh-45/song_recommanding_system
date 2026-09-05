from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle 
import re  # Naya module add kiya text clean karne ke liye
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Smart Music Recommander API")

# File loading
songs_dict = pickle.load(open('model/indian_songs_dict.pkl', 'rb')) 
movies_df = pd.DataFrame(songs_dict)
scaled_feature = pickle.load(open('model/scaled_features.pkl', 'rb'))

class SongRequest(BaseModel):
    song_name: str

# NAYA FUNCTION: Ye gaane ke naam me se brackets aur extra words hata dega
def get_clean_name(name):
    # "Tum Hi Ho (From Aashiqui)" ya "Tum Hi Ho - Remix" dono ko "tum hi ho" bana dega
    clean = re.split(r'\(|-|remix|mix|lofi', str(name), flags=re.IGNORECASE)[0]
    return clean.strip().lower()

@app.get('/')
def home():
    return {"message": "Welcome to Smart Music Recommender API! Go to /docs to test the endpoints."}


@app.post('/recommend')
def recommend(request: SongRequest):
    input_song = request.song_name

    if input_song not in movies_df['track_name'].values:
        raise HTTPException(status_code=404, detail="Song not found")

    idx = movies_df[movies_df['track_name'] == input_song].index[0]
    
    # Input gaane ka base naam (taaki original gaana wapas recommend na ho)
    base_input_name = get_clean_name(input_song)

    song_vector = scaled_feature[idx].reshape(1, -1)
    distaces = cosine_similarity(song_vector, scaled_feature)[0]
    
    # Checking top 50 to find 5 purely unique songs
    song_indices = sorted(list(enumerate(distaces)), reverse=True, key=lambda x: x[1])[1:50]

    results = []
    seen_names = set() 

    for i in song_indices:
        index = i[0]
        rec_song_name = movies_df.iloc[index]['track_name']
        rec_artist = movies_df.iloc[index]['artist_name']
        
        rec_clean_name = get_clean_name(rec_song_name)
        
        # FILTER 1: Agar recommendation me input gaane ka naam chupa hai, toh turant SKIP karo
        if base_input_name in rec_clean_name or rec_clean_name in base_input_name:
            continue
            
        # FILTER 2: Agar ye naya gaana pehle hi list me jud chuka hai, toh SKIP karo
        if rec_clean_name in seen_names:
            continue
            
        # Agar dono filter pass ho gaye, matlab gaana ekdum fresh aur unique hai
        seen_names.add(rec_clean_name)
        results.append({
            "song_name": rec_song_name,
            "artist": rec_artist,
            "language": movies_df.iloc[index]['language'],
            "match_percentage": f"{round(i[1] * 100, 2)}%"
        })
        
        # Jab 5 bilkul alag gaane mil jayein, toh loop stop kar do
        if len(results) == 5:
            break
            
    return {"status": "success", "recommendations": results}