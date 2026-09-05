from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import pickle 
import re
import difflib
import requests
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Smart Music Recommender API", version="2.1")

# File loading
songs_dict = pickle.load(open('model/indian_songs_dict.pkl', 'rb')) 
movies_df = pd.DataFrame(songs_dict)
scaled_feature = pickle.load(open('model/scaled_features.pkl', 'rb'))

# Helper function to clean song title (remove brackets, remix, lofi, etc.)
def get_clean_name(name):
    clean = re.split(r'\(|-|remix|mix|lofi', str(name), flags=re.IGNORECASE)[0]
    return clean.strip().lower()

# Comprehensive Language Detection for Indian Music (Detects Tamil, Punjabi, Telugu, and Hindi)
def detect_song_language(track_name, artist_name, current_lang="Hindi"):
    t = str(track_name).lower()
    a = str(artist_name).lower()

    # Explicit indicators in track name
    if re.search(r'\(.*hindi.*\)|- hindi|\bhindi version\b', t):
        return 'Hindi'
    if re.search(r'\(.*tamil.*\)|- tamil|\btamil version\b|\(tamil\)', t):
        return 'Tamil'
    if re.search(r'\(.*telugu.*\)|- telugu|\btelugu version\b|\(telugu\)', t):
        return 'Telugu'
    if re.search(r'\(.*punjabi.*\)|- punjabi|\bpunjabi version\b|\(punjabi\)', t):
        return 'Punjabi'
    if re.search(r'\(.*malayalam.*\)|- malayalam|\bmalayalam version\b', t):
        return 'Malayalam'

    # Known Tamil keywords / artists
    tamil_indicators = [
        'kadhal', 'anukkal', 'azhagooril', 'theeraamal', 'rathamaarey', 'kuthu', 
        'kolaveri', 'vaathi', 'dhee', 'santhosh narayanan', 'yuvan shankar raja', 
        'harris jayaraj', 'vidyasagar', 's. p. balasubrahmanyam', 'ilaiyaraaja', 
        'ilayaraja', 'anirudh', 'hiphop tamizha', 'g. v. prakash kumar'
    ]
    if any(k in t or k in a for k in tamil_indicators):
        # If artist is Anirudh / AR Rahman, verify if track name has Hindi keywords
        if any(w in t for w in ['hai', 'tum', 'tera', 'meri', 'dil', 'kaun', 'pyaar', 'mohabbat', 'zindagi', 'saath', 'hum']):
            return 'Hindi'
        return 'Tamil'

    # Known Punjabi indicators / artists
    punjabi_indicators = [
        'diljit dosanjh', 'ap dhillon', 'sidhu moose wala', 'karan aujla', 
        'guru randhawa', 'b praak', 'hardy sandhu', 'amrit maan', 'jassi gill', 
        'parmish verma', 'sharry mann', 'babbu maan', 'amrinder gill', 'sukhe',
        'jass manak', 'sukhe', 'mankirt aulakh', 'jordan sandhu'
    ]
    if any(k in a or k in t for k in punjabi_indicators):
        return 'Punjabi'

    return current_lang

# Apply accurate language labeling across the dataset
movies_df['language'] = [
    detect_song_language(t, a, l) 
    for t, a, l in zip(movies_df['track_name'], movies_df['artist_name'], movies_df['language'])
]

# Pre-calculate normalized columns for fast case-insensitive lookup
movies_df['lower_track_name'] = movies_df['track_name'].astype(str).str.strip().str.lower()
movies_df['clean_track_name'] = movies_df['track_name'].apply(get_clean_name)
unique_tracks = [str(t) for t in movies_df['track_name'].drop_duplicates().tolist() if str(t).strip()]

# In-memory artwork cache to prevent redundant external API calls
artwork_cache = {}
FALLBACK_IMAGE = "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=600&auto=format&fit=crop&q=80"

def fetch_itunes_metadata(song_name: str, artist_name: str = ""):
    """Fetch high-res album artwork and 30-second audio preview from iTunes Search API."""
    clean_title = get_clean_name(song_name)
    cache_key = f"{clean_title} - {artist_name.strip().lower()}"
    if cache_key in artwork_cache:
        return artwork_cache[cache_key]
    
    queries = [
        f"{clean_title} {artist_name}".strip(),
        clean_title.strip(),
        str(song_name).strip()
    ]
    
    for q in queries:
        if not q:
            continue
        try:
            res = requests.get(
                "https://itunes.apple.com/search",
                params={"term": q, "entity": "song", "limit": 1},
                timeout=2.5
            )
            if res.status_code == 200:
                data = res.json()
                if data.get("resultCount", 0) > 0:
                    track = data["results"][0]
                    art = track.get("artworkUrl100", "")
                    if art:
                        art = art.replace("100x100bb", "600x600bb")
                    meta = {
                        "artwork_url": art or FALLBACK_IMAGE,
                        "preview_url": track.get("previewUrl"),
                        "album_name": track.get("collectionName", "Single / Album"),
                        "track_view_url": track.get("trackViewUrl")
                    }
                    artwork_cache[cache_key] = meta
                    return meta
        except Exception:
            pass

    fallback_meta = {
        "artwork_url": FALLBACK_IMAGE,
        "preview_url": None,
        "album_name": "Indian Music Collection",
        "track_view_url": None
    }
    artwork_cache[cache_key] = fallback_meta
    return fallback_meta

def find_song_index(query: str):
    """
    Finds a song index with multi-tier case-insensitive and fuzzy matching.
    Supports UPPERCASE, lowercase, mixed case, and partial names.
    """
    if not query:
        return None
    
    q_stripped = str(query).strip()
    q_lower = q_stripped.lower()
    q_clean = get_clean_name(q_stripped)

    # 1. Exact case-insensitive match
    matches = movies_df[movies_df['lower_track_name'] == q_lower]
    if not matches.empty:
        return matches.index[0]

    # 2. Exact clean-name match
    matches = movies_df[movies_df['clean_track_name'] == q_clean]
    if not matches.empty:
        return matches.index[0]

    # 3. Clean name starts with query
    matches = movies_df[movies_df['clean_track_name'].str.startswith(q_clean)]
    if not matches.empty:
        return matches.index[0]

    # 4. Fuzzy match on clean track name
    close = difflib.get_close_matches(q_clean, movies_df['clean_track_name'].tolist(), n=1, cutoff=0.7)
    if close:
        m = movies_df[movies_df['clean_track_name'] == close[0]]
        if not m.empty:
            return m.index[0]

    # 5. Full track name contains query
    matches = movies_df[movies_df['lower_track_name'].str.contains(re.escape(q_lower), na=False)]
    if not matches.empty:
        return matches.index[0]

    # 6. Fallback fuzzy on full track name
    close_full = difflib.get_close_matches(q_lower, movies_df['lower_track_name'].tolist(), n=1, cutoff=0.6)
    if close_full:
        m = movies_df[movies_df['lower_track_name'] == close_full[0]]
        if not m.empty:
            return m.index[0]

    return None

def get_suggestions_for_query(query: str, limit: int = 8):
    """Generates dataset auto-suggestions based on partial user input."""
    if not query or not str(query).strip():
        return unique_tracks[:limit]
    
    q = str(query).strip().lower()
    
    # Prefix matches first
    prefix_matches = [
        t for t in unique_tracks if t.lower().startswith(q)
    ]
    
    # Substring matches next
    contains_matches = [
        t for t in unique_tracks if q in t.lower() and t not in prefix_matches
    ]
    
    results = prefix_matches + contains_matches
    if len(results) < limit:
        fuzzy_matches = difflib.get_close_matches(q, unique_tracks, n=limit, cutoff=0.5)
        for fm in fuzzy_matches:
            if fm not in results:
                results.append(fm)
                
    return results[:limit]

class SongRequest(BaseModel):
    song_name: str
    num_recommendations: int = 5
    language_filter: str = "All"  # "All", "Hindi", "Tamil", "Punjabi", "Same as Input"

@app.get('/')
def home():
    return {
        "message": "Welcome to Smart Music Recommender API v2.1!",
        "endpoints": {
            "/recommend": "POST - Get recommendations with images & audio previews",
            "/suggestions": "GET - Get auto-suggestions from dataset as you type",
            "/songs": "GET - Get popular / sample songs from dataset"
        }
    }

@app.get('/suggestions')
def suggestions(q: str = Query("", description="Query string for song auto-suggestions"), limit: int = 8):
    return {"suggestions": get_suggestions_for_query(q, limit)}

@app.get('/songs')
def get_songs(limit: int = 500):
    return {"songs": unique_tracks[:limit], "total_unique": len(unique_tracks)}

@app.post('/recommend')
def recommend(request: SongRequest):
    input_song = str(request.song_name).strip()
    if not input_song:
        raise HTTPException(status_code=400, detail="Please provide a song name.")

    idx = find_song_index(input_song)
    if idx is None:
        close_suggestions = get_suggestions_for_query(input_song, limit=5)
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Song '{input_song}' not found in dataset.",
                "suggestions": close_suggestions
            }
        )

    # Matched song information
    matched_row = movies_df.iloc[idx]
    matched_track_name = matched_row['track_name']
    matched_artist = matched_row['artist_name']
    matched_language = matched_row['language']
    base_input_name = get_clean_name(matched_track_name)

    # Cosine similarity vector
    song_vector = scaled_feature[idx].reshape(1, -1)
    distances = cosine_similarity(song_vector, scaled_feature)[0]
    
    # Sort top candidates
    song_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:120]

    raw_candidates = []
    seen_names = set() 
    target_count = max(1, min(request.num_recommendations, 10))
    lang_pref = request.language_filter.strip().lower()

    for i in song_indices:
        index = i[0]
        rec_song_name = movies_df.iloc[index]['track_name']
        rec_artist = movies_df.iloc[index]['artist_name']
        rec_lang = movies_df.iloc[index]['language']
        rec_clean_name = get_clean_name(rec_song_name)
        
        # Filter 1: Don't recommend same or variation of the input song
        if base_input_name in rec_clean_name or rec_clean_name in base_input_name:
            continue
            
        # Filter 2: Don't recommend duplicates
        if rec_clean_name in seen_names:
            continue

        # Filter 3: Language filter
        if lang_pref == "hindi" and rec_lang != "Hindi":
            continue
        elif lang_pref == "tamil" and rec_lang != "Tamil":
            continue
        elif lang_pref == "punjabi" and rec_lang != "Punjabi":
            continue
        elif lang_pref in ["same as input", "same as matched song"] and rec_lang != matched_language:
            continue
            
        seen_names.add(rec_clean_name)
        raw_candidates.append({
            "song_name": rec_song_name,
            "artist": rec_artist,
            "language": rec_lang,
            "match_percentage": f"{round(i[1] * 100, 1)}%",
            "similarity_score": round(float(i[1]), 3)
        })
        
        if len(raw_candidates) == target_count:
            break

    # Fetch album artwork and audio previews concurrently
    all_to_fetch = [(matched_track_name, matched_artist)] + [
        (c["song_name"], c["artist"]) for c in raw_candidates
    ]

    with ThreadPoolExecutor(max_workers=min(len(all_to_fetch), 8)) as executor:
        metadata_results = list(executor.map(
            lambda item: fetch_itunes_metadata(item[0], item[1]),
            all_to_fetch
        ))

    matched_meta = metadata_results[0]
    matched_song_info = {
        "song_name": matched_track_name,
        "artist": matched_artist,
        "language": matched_language,
        "artwork_url": matched_meta.get("artwork_url", FALLBACK_IMAGE),
        "preview_url": matched_meta.get("preview_url"),
        "album_name": matched_meta.get("album_name", "Original Track")
    }

    final_recommendations = []
    for candidate, meta in zip(raw_candidates, metadata_results[1:]):
        candidate["artwork_url"] = meta.get("artwork_url", FALLBACK_IMAGE)
        candidate["preview_url"] = meta.get("preview_url")
        candidate["album_name"] = meta.get("album_name", "Album")
        final_recommendations.append(candidate)

    return {
        "status": "success",
        "matched_song": matched_song_info,
        "recommendations": final_recommendations
    }