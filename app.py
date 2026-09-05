import streamlit as st
import requests
import pickle
import re
from typing import List

# ---------------------------------------------------------
# Page Configuration & Aesthetics
# ---------------------------------------------------------
st.set_page_config(
    page_title="VibeSync • Indian Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom High-End Dark Music Aesthetic CSS
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Background */
    .main {
        background: radial-gradient(circle at top right, rgba(29, 185, 84, 0.09) 0%, rgba(15, 23, 42, 0) 50%),
                    radial-gradient(circle at top left, rgba(6, 182, 212, 0.08) 0%, rgba(15, 23, 42, 0) 50%),
                    #0b0f19;
    }

    /* Header Styling */
    .header-container {
        text-align: center;
        padding: 26px 15px 15px 15px;
        margin-bottom: 20px;
    }

    .app-badge {
        display: inline-block;
        padding: 6px 16px;
        background: rgba(29, 185, 84, 0.15);
        border: 1px solid rgba(29, 185, 84, 0.4);
        border-radius: 9999px;
        color: #1ed760;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 70%, #1ed760 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }

    .main-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        max-width: 720px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Hero Card for Matched Track */
    .matched-card {
        background: linear-gradient(135deg, rgba(29, 185, 84, 0.12) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(29, 185, 84, 0.35);
        border-radius: 20px;
        padding: 24px;
        margin: 25px 0;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.4), 0 0 25px rgba(29, 185, 84, 0.1);
        backdrop-filter: blur(12px);
    }

    /* Language Badges */
    .card-meta-tag {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .tag-hindi {
        background: rgba(245, 158, 11, 0.18);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }

    .tag-tamil {
        background: rgba(6, 182, 212, 0.18);
        color: #22d3ee;
        border: 1px solid rgba(6, 182, 212, 0.35);
    }

    .tag-punjabi {
        background: rgba(168, 85, 247, 0.18);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.35);
    }

    .tag-telugu {
        background: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .tag-malayalam {
        background: rgba(244, 63, 94, 0.18);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.35);
    }

    .match-pill {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 4px 10px;
        border-radius: 20px;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.3);
    }

    /* Song Recommendation Card */
    .song-box {
        background: #131b2e;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 18px;
        transition: all 0.3s ease;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
    }

    .song-box:hover {
        border-color: rgba(29, 185, 84, 0.5);
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.4), 0 0 20px rgba(29, 185, 84, 0.2);
    }

    .song-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 10px;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .song-artist {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-bottom: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Audio Player */
    audio {
        width: 100%;
        height: 38px;
        margin-top: 8px;
        border-radius: 20px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Dataset Song Names for Auto-Suggestions
# ---------------------------------------------------------
@st.cache_data
def get_all_dataset_songs() -> List[str]:
    """Load unique song names directly from dataset for instant autocomplete."""
    try:
        data = pickle.load(open('model/indian_songs_dict.pkl', 'rb'))
        if isinstance(data, dict) and 'track_name' in data:
            if isinstance(data['track_name'], dict):
                raw_tracks = list(dict.fromkeys(data['track_name'].values()))
            elif isinstance(data['track_name'], (list, tuple)):
                raw_tracks = list(dict.fromkeys(data['track_name']))
            else:
                raw_tracks = [str(x) for x in data['track_name']]
        else:
            raw_tracks = []
        clean_tracks = [str(t).strip() for t in raw_tracks if t is not None and str(t).strip()]
        return clean_tracks
    except Exception:
        # Fallback to FastAPI songs endpoint
        try:
            r = requests.get("http://127.0.0.1:8000/songs?limit=2000", timeout=2)
            if r.status_code == 200:
                raw = r.json().get("songs", [])
                return [str(s).strip() for s in raw if s and str(s).strip()]
        except Exception:
            pass
        return [
            "Tum Hi Ho", "Kesariya", "Badass", "Channa Mereya", "Apna Bana Le",
            "Kadhal Anukkal", "Tum Tum (From \"Enemy - Tamil\")", "Pagal Anukan (From \"Robot\")",
            "Raataan Lambiyan", "Ghungroo", "Kal Ho Naa Ho", "Kabira", "Lover", "Amplifier"
        ]

all_songs = get_all_dataset_songs()

# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="app-badge">⚡ AI Acoustic Similarity Engine • Hindi, Tamil & Punjabi</div>
    <div class="main-title">🎵 VibeSync Music Recommender</div>
    <div class="main-subtitle">
        Type in any case (<b>UPPERCASE</b>, <b>lowercase</b>, or <b>Title Case</b>) or use instant auto-suggestions to discover matching songs with high-definition album art, true language tags, and audio previews!
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Quick Trending Song Chips (Hindi, Tamil, Punjabi)
# ---------------------------------------------------------
st.markdown("<p style='color:#94a3b8; font-size:0.9rem; font-weight:600; margin-bottom:8px;'>🔥 Quick Trending Hits (Hindi • Tamil • Punjabi):</p>", unsafe_allow_html=True)

trending_songs = [
    ("Tum Hi Ho", "Hindi"),
    ("Kesariya", "Hindi"),
    ("Badass", "Tamil/Hindi"),
    ("Kadhal Anukkal", "Tamil"),
    ("Channa Mereya", "Hindi"),
    ("Pagal Anukan", "Hindi")
]

chip_cols = st.columns(len(trending_songs))

# Session state initialization
if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""

if "execute_search" not in st.session_state:
    st.session_state["execute_search"] = False

for col, (song, lang) in zip(chip_cols, trending_songs):
    if col.button(f"✨ {song}", key=f"chip_{song}", use_container_width=True):
        st.session_state["search_query"] = str(song)
        st.session_state["execute_search"] = True

# ---------------------------------------------------------
# Search & Input Controls
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

input_tab1, input_tab2 = st.tabs([
    "🔍 Smart Search (Free Text & Case-Insensitive)",
    "📜 Dataset Autocomplete Dropdown (13,000+ Songs)"
])

with input_tab1:
    col_input, col_lang, col_num, col_btn = st.columns([4, 2.5, 2, 2])
    with col_input:
        text_query = st.text_input(
            "Search Song Name:",
            value=str(st.session_state.get("search_query", "")),
            placeholder="Type anything (e.g. tum hi ho, TUM HI HO, kadhal anukkal, kesariya)...",
            help="Case does not matter! Type in lowercase, uppercase, or mixed case.",
            key="text_search_input"
        )
    with col_lang:
        lang_filter = st.selectbox(
            "Language Filter:",
            options=["All Languages", "Hindi Only", "Tamil Only", "Punjabi Only", "Same as Matched Song"],
            index=0,
            key="lang_filter_select"
        )
    with col_num:
        num_recs = st.slider("Count:", min_value=3, max_value=8, value=5, key="num_recs_slider")
    with col_btn:
        st.write("")
        st.write("")
        search_clicked = st.button("🚀 Recommend", type="primary", use_container_width=True)

    # Live dataset auto-suggestions based on current text query
    cleaned_input = str(text_query).strip()
    if cleaned_input and len(cleaned_input) >= 2:
        query_clean = cleaned_input.lower()
        live_matches = [str(s) for s in all_songs if isinstance(s, str) and query_clean in s.lower()][:6]
        if live_matches and not any(isinstance(s, str) and s.lower() == query_clean for s in live_matches):
            st.markdown("<p style='font-size:0.85rem; color:#64748b; margin-top:4px;'>💡 <b>Dataset Matches:</b> Click to auto-fill:</p>", unsafe_allow_html=True)
            sug_cols = st.columns(min(len(live_matches), 6))
            for sc, match_name in zip(sug_cols, live_matches):
                if sc.button(f"🎵 {str(match_name)[:24]}", key=f"sug_{match_name}", use_container_width=True):
                    st.session_state["search_query"] = str(match_name)
                    st.session_state["execute_search"] = True
                    st.rerun()

with input_tab2:
    col_drop, col_drop_lang, col_drop_btn = st.columns([5, 2.5, 2])
    with col_drop:
        selected_dropdown_song = st.selectbox(
            "Select or filter from all 13,000+ dataset songs:",
            options=all_songs,
            index=None,
            placeholder="Start typing song name from datasheet...",
            key="dataset_selectbox"
        )
    with col_drop_lang:
        dropdown_lang_filter = st.selectbox(
            "Language Preference:",
            options=["All Languages", "Hindi Only", "Tamil Only", "Punjabi Only", "Same as Matched Song"],
            index=0,
            key="dropdown_lang_filter"
        )
    with col_drop_btn:
        st.write("")
        st.write("")
        dropdown_clicked = st.button("🎧 Recommend Selected", key="btn_dropdown", use_container_width=True)
        if dropdown_clicked and selected_dropdown_song:
            st.session_state["search_query"] = str(selected_dropdown_song)
            st.session_state["execute_search"] = True

# Determine final query and language filter to search
final_query = ""
active_lang_filter = "All"

if search_clicked:
    final_query = str(text_query).strip()
    active_lang_filter = lang_filter
elif st.session_state.get("execute_search") and st.session_state.get("search_query"):
    final_query = str(st.session_state["search_query"]).strip()
    active_lang_filter = lang_filter
    st.session_state["execute_search"] = False  # Reset flag

# Map UI language filter to API filter
api_lang_map = {
    "All Languages": "All",
    "Hindi Only": "Hindi",
    "Tamil Only": "Tamil",
    "Punjabi Only": "Punjabi",
    "Same as Matched Song": "Same as Matched Song"
}
api_lang_filter = api_lang_map.get(active_lang_filter, "All")

# ---------------------------------------------------------
# Execution & Recommendations Display
# ---------------------------------------------------------
if final_query:
    with st.spinner("✨ Analyzing acoustic vectors & fetching album artwork..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/recommend",
                json={
                    "song_name": final_query,
                    "num_recommendations": num_recs,
                    "language_filter": api_lang_filter
                },
                timeout=12
            )

            if response.status_code == 200:
                data = response.json()
                matched = data.get("matched_song", {})
                recommendations = data.get("recommendations", [])

                matched_lang = matched.get("language", "Hindi")
                matched_lang_class = f"tag-{matched_lang.lower()}"

                # Hero Banner for the Matched Song
                st.markdown(f"""
                <div class="matched-card">
                    <div style="display:flex; flex-wrap:wrap; align-items:center; gap:22px;">
                        <img src="{matched.get('artwork_url')}" 
                             style="width:130px; height:130px; border-radius:14px; object-fit:cover; box-shadow:0 8px 24px rgba(0,0,0,0.5); border:1px solid rgba(255,255,255,0.1);" />
                        <div style="flex:1; min-width:240px;">
                            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                                <span class="app-badge" style="margin:0; font-size:0.75rem;">✓ Matched From Dataset</span>
                                <span class="card-meta-tag {matched_lang_class}">{matched_lang}</span>
                            </div>
                            <h2 style="color:#ffffff; margin:0 0 6px 0; font-size:1.8rem; font-weight:800;">{matched.get('song_name')}</h2>
                            <p style="color:#94a3b8; font-size:1.05rem; margin:0 0 4px 0;">🎤 <b>Artist:</b> {matched.get('artist')}</p>
                            <p style="color:#64748b; font-size:0.88rem; margin:0;">💿 <b>Album:</b> {matched.get('album_name', 'Single / Album')}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if matched.get("preview_url"):
                    st.write("🔊 **Source Song Preview (30s):**")
                    st.audio(matched.get("preview_url"))

                st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 30px 0;'>", unsafe_allow_html=True)
                st.markdown(f"### 🎶 Top Recommended Songs for *'{matched.get('song_name')}'*")
                st.markdown(f"<p style='color:#94a3b8; font-size:0.95rem;'>Ranked by acoustic similarity • Filter: <b>{active_lang_filter}</b></p>", unsafe_allow_html=True)

                if not recommendations:
                    st.info(f"No songs found matching the '{active_lang_filter}' criteria. Try selecting 'All Languages'.")
                else:
                    # Responsive Multi-column Grid
                    grid_cols_count = 3 if len(recommendations) >= 3 else 2
                    cols = st.columns(grid_cols_count)

                    for i, rec in enumerate(recommendations):
                        col_target = cols[i % grid_cols_count]
                        with col_target:
                            rec_lang = rec.get("language", "Hindi")
                            lang_class = f"tag-{rec_lang.lower()}"
                            
                            # Card Header & Artwork
                            st.markdown(f"""
                            <div class="song-box">
                                <div style="position:relative; width:100%; aspect-ratio:1/1; overflow:hidden; border-radius:12px; margin-bottom:12px; background:#0f172a;">
                                    <img src="{rec.get('artwork_url')}" 
                                         style="width:100%; height:100%; object-fit:cover; border-radius:12px;" />
                                </div>
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                    <span class="card-meta-tag {lang_class}">{rec_lang}</span>
                                    <span class="match-pill">⚡ {rec.get('match_percentage')} Match</span>
                                </div>
                                <div class="song-title" title="{rec.get('song_name')}">{i+1}. {rec.get('song_name')}</div>
                                <div class="song-artist" title="{rec.get('artist')}">🎤 {rec.get('artist')}</div>
                                <div style="color:#64748b; font-size:0.8rem; margin-bottom:10px;">💿 {rec.get('album_name', 'Single / Album')}</div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Audio Preview Player if available
                            if rec.get("preview_url"):
                                st.audio(rec.get("preview_url"))
                            else:
                                st.caption("🎵 Preview audio not available for this track")
                            
                            st.markdown("<br>", unsafe_allow_html=True)

            elif response.status_code == 404:
                error_data = response.json().get("detail", {})
                msg = error_data.get("message", f"Song '{final_query}' not found in dataset.")
                suggestions = error_data.get("suggestions", [])

                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 16px; padding: 24px; margin: 20px 0; text-align: center;">
                    <h3 style="color:#f87171; margin-top:0;">🔍 Song Not Found</h3>
                    <p style="color:#cbd5e1; font-size:1.05rem;">{msg}</p>
                    <p style="color:#94a3b8; font-size:0.9rem;">Check the spelling or try one of the auto-suggested alternatives below.</p>
                </div>
                """, unsafe_allow_html=True)

                if suggestions:
                    st.markdown("#### 💡 Did you mean one of these songs from the datasheet?")
                    sug_grid = st.columns(min(len(suggestions), 5))
                    for col, sug_song in zip(sug_grid, suggestions):
                        if col.button(f"🎵 {str(sug_song)}", key=f"not_found_sug_{sug_song}", use_container_width=True):
                            st.session_state["search_query"] = str(sug_song)
                            st.session_state["execute_search"] = True
                            st.rerun()

            else:
                st.error(f"Error ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.markdown("""
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 16px; padding: 24px; margin: 20px 0;">
                <h3 style="color:#f87171; margin-top:0;">⚠️ FastAPI Backend Unreachable</h3>
                <p style="color:#cbd5e1;">Could not connect to the recommendation API at <code>http://127.0.0.1:8000</code>.</p>
                <p style="color:#94a3b8; font-size:0.9rem;">Make sure your backend server is running with: <code>uvicorn main:app --reload</code></p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding: 40px 10px 20px 10px; color:#475569; font-size:0.85rem;">
    VibeSync Indian Music Recommendation Engine • Hindi • Tamil • Punjabi • Powered by Cosine Similarity & iTunes API
</div>
""", unsafe_allow_html=True)