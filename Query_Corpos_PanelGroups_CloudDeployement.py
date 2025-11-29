# app.py - CSPC 2023 AI Search with PROPER JOIN between DocChunk and CSPC_Panels

import os
import re
from typing import Optional
from collections import defaultdict

import streamlit as st
from openai import OpenAI
from weaviate import connect_to_wcs
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from sentence_transformers import CrossEncoder
from urllib.parse import quote

S3_BUCKET = "cspc-rag"
S3_REGION = "ca-central-1"
S3_BASE_URL = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"
S3_AUDIO_PREFIX = "audio"   # folder in the bucket


# ========================
# CONFIG & PAGE SETUP
# ========================
st.set_page_config(
    page_title="CSPC 2023 AI Search",
    layout="wide",
    initial_sidebar_state="expanded"
)

AUDIO_DIR = r"D:\Downloads\AllDays"


# ========================
# WEAVIATE CLIENT
# ========================
@st.cache_resource(show_spinner=False)
def get_client(_url: str, _key: str):
    return connect_to_wcs(cluster_url=_url, auth_credentials=Auth.api_key(_key))


@st.cache_resource(show_spinner=False)
def get_collection(_client, _name: str):
    return _client.collections.get(_name)


# ========================
# PANEL METADATA RETRIEVAL FROM CSPC_PANELS
# ========================
def get_panel_metadata_from_cspc_panels(client, panel_code: str) -> dict:
    """
    Fetch panel metadata including photo_url from CSPC_Panels collection.
    This is a SEPARATE collection from DocChunk.

    Args:
        client: Weaviate client
        panel_code: Panel code (e.g., "333", "11", "101")

    Returns:
        dict with title, photo_url, speaker_photo_url, organized_by, speakers, etc.
    """
    if not panel_code:
        return {}

    try:
        panels_coll = client.collections.get("CSPC_Panels")

        # Try string panel_code first
        try:
            response = panels_coll.query.fetch_objects(
                filters=Filter.by_property("panel_code").equal(str(panel_code)),
                limit=1
            )
        except:
            # If string fails, try integer
            try:
                response = panels_coll.query.fetch_objects(
                    filters=Filter.by_property("panel_code").equal(int(panel_code)),
                    limit=1
                )
            except Exception as e:
                st.sidebar.warning(f"Could not query CSPC_Panels for panel {panel_code}: {e}")
                return {}

        if not response.objects:
            return {}

        panel_data = response.objects[0].properties

        # Extract and clean data
        def _first_or_value(val):
            """Handle list fields - take first item if it's a list"""
            if isinstance(val, list):
                return val[0] if val else None
            return val

        return {
            "panel_code": panel_data.get("panel_code"),
            "title": panel_data.get("title", ""),
            "theme": panel_data.get("theme", ""),
            "photo_url": _first_or_value(panel_data.get("photo_url")),
            "speaker_photo_url": _first_or_value(panel_data.get("speaker_photo_url")),
            "organized_by": panel_data.get("organized_by") or panel_data.get("panel_organized_by", ""),
            "speakers": panel_data.get("speakers", []),
            "panel_date": panel_data.get("panel_date", ""),
            "abstract": panel_data.get("abstract", ""),
            "panel_url": panel_data.get("panel_url", ""),
            "external_details_url": panel_data.get("external_details_url", ""),
            "_raw": panel_data  # Keep for debugging
        }

    except Exception as e:
        st.sidebar.error(f"Error fetching CSPC_Panels data for panel {panel_code}: {e}")
        return {}


# ========================
# DYNAMIC FILTERS FROM DATA
# ========================
@st.cache_data(ttl=3600, show_spinner=False)
def get_all_themes(_client) -> list:
    """Fetch all unique themes from CSPC_Panels collection"""
    try:
        panels_coll = _client.collections.get("CSPC_Panels")
        response = panels_coll.query.fetch_objects(limit=1000)

        themes = set()
        for obj in response.objects:
            theme = obj.properties.get("theme")
            if theme:
                themes.add(theme)

        return sorted(list(themes))
    except Exception as e:
        st.sidebar.warning(f"Could not fetch themes: {e}")
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_panels(_client) -> list:
    """Fetch all unique panel codes from CSPC_Panels collection"""
    try:
        panels_coll = _client.collections.get("CSPC_Panels")
        response = panels_coll.query.fetch_objects(limit=1000)

        panels = []
        for obj in response.objects:
            panel_code = obj.properties.get("panel_code")
            title = obj.properties.get("title", "")
            if panel_code:
                # Format: "Panel 333 - Title"
                display = f"Panel {panel_code}"
                if title:
                    display += f" - {title[:50]}..." if len(title) > 50 else f" - {title}"
                panels.append((str(panel_code), display))

        # Sort by panel code (numerically)
        panels.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 999999)

        return panels
    except Exception as e:
        st.sidebar.warning(f"Could not fetch panels: {e}")
        return []


# ========================
# AUDIO HELPERS
# ========================
def find_audio_file(name: str) -> Optional[str]:
    if not name:
        return None
    base = name
    for suf in ["_transcript.txt", ".txt", "_transcript"]:
        base = base.removesuffix(suf)
    for ext in [".mp3", ".MP3", ".wav", ".WAV"]:
        path = os.path.join(AUDIO_DIR, base + ext)
        if os.path.exists(path):
            return path
    return None


def time_to_seconds(time_str):
    """Convert HH:MM:SS or MM:SS to seconds."""
    if not time_str or time_str == "—":
        return 0
    try:
        parts = time_str.split(":")
        if len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        else:
            return 0
    except (ValueError, AttributeError):
        return 0

# ========================
# MAIN APP
# ========================
def main():
    # ========== CUSTOM CSS ==========
    st.markdown("""
    <style>
        .main > div {padding-top:0!important;}
        .block-container {padding-top:1rem!important; max-width:95%!important; font-size:1.1rem;}
        #MainMenu, footer, header {visibility:hidden;}
        .banner {text-align:center; padding:12px; color:white; font-weight:bold; margin:5px 0;}
        .panel-header {background:#f0f2f6; padding:15px; border-radius:10px; margin:10px 0;}
        .photo-container {border: 2px solid #ddd; border-radius: 8px; padding: 5px;}

        /* Larger text everywhere */
        p, li, span, label {font-size: 1.3rem !important;}

        .panel-metadata {font-size: 1.9rem !important; line-height: 2.4;}
        .panel-number {font-size: 1.8rem !important; font-weight: bold; margin-bottom: 0.5rem !important; color: #00426a;}
        .panel-title {font-size: 2.4rem !important; font-weight: bold; line-height: 1.3; margin-bottom: 1.5rem !important;}
        .chunks-header {font-size: 1.6rem !important; font-weight: bold !important; margin: 1.5rem 0 1rem 0 !important;}
        input[type="text"] {font-size: 1.5rem !important; padding: 1rem !important;}
        .stCaption, [data-testid="stCaptionContainer"] {font-size: 1.1rem !important;}
        button {font-size: 1.3rem !important;}
        audio {width: 100%;}

        /* Chunk border */
        .chunk-root {
            border: 1px solid #ccc;
            border-radius: 6px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 1.1rem;
            background-color: #fdfdfd;
        }

        .results-header {
            font-size: 2rem;
            font-weight: bold;
            margin: 1.5rem 0 1rem 0;
            color: #00426a;
        }

        .panel-separator {
            border: none;
            border-top: 2px solid #ccc;
            margin: 2rem 0;
        }

        /* Debug styling */
        .debug-box {
            background-color: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 0.9rem;
        }

        .debug-success {
            background-color: #d4edda;
            border-color: #28a745;
            color: #155724;
        }

        .debug-error {
            background-color: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }

        .debug-info {
            background-color: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }
    </style>
    """, unsafe_allow_html=True)

    # ========== HELPER FUNCTION: time_to_seconds ==========
    def time_to_seconds(time_str):
        """Convert HH:MM:SS or MM:SS to seconds."""
        if not time_str or time_str == "—":
            return 0
        try:
            parts = time_str.split(":")
            if len(parts) == 3:  # HH:MM:SS
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:  # MM:SS
                return int(parts[0]) * 60 + int(parts[1])
            else:
                return 0
        except (ValueError, AttributeError):
            return 0

    # ========== HEADER ==========
    col1, col2 = st.columns([1.3, 4])
    with col1:
        st.image("https://sciencepolicy.ca/wp-content/uploads/2020/09/cspc-logo.png", width=300)
    with col2:
        st.markdown(
            """
            <div style="
                display:flex;
                align-items:center;
                margin-left:15px;
                gap: 1.5rem;
                color:#00426a;
            ">
                <div style="font-size:3rem; font-weight:bold; color:#00426a;">
                    CSPC AI Platform
                </div>
                <div style="font-size:1.5rem; font-style:italic; color:#00426a;">
                    .. where every moment is found & could be actionable
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="banner" style="background:#005a92; font-size:1.6rem;">Phase I • CSPC 2023 Conference</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="banner" style="background:#c41e3a; font-size:1.6rem;"> </div>',
        unsafe_allow_html=True)

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.header("Configuration")
        weaviate_url = st.text_input(
            "Weaviate URL",
            value=os.getenv("WEAVIATE_URL", "nsrnedu9q1qfxusokfl8q.c0.us-west3.gcp.weaviate.cloud"),
            key="weaviate_url_input"  # ADDED KEY
        )

        weaviate_key = st.text_input(
            "Weaviate Key",
            type="password",
            value=os.getenv("WEAVIATE_API_KEY", ""),
            key="weaviate_key_input"  # ADDED KEY
        )

        openai_key = st.text_input(
            "OpenAI Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            key="openai_key_input"  # ADDED KEY
        )

        collection_name = st.text_input(
            "Chunks Collection",
            "DocChunk",
            key="collection_name_input"  # ADDED KEY
        )

        st.markdown("---")
        st.subheader("Search Settings")
        alpha = st.slider("Hybrid Alpha", 0.0, 1.0, 0.75, key="alpha_slider")  # ADDED KEY
        top_k = st.number_input("Top Results", 1, 30, 10, key="top_k_input")  # ADDED KEY
        use_reranker = st.checkbox("Use Reranker", True, key="use_reranker_check")  # ADDED KEY
        use_llm = st.checkbox("Generate AI Answer", True, key="use_llm_check")  # ADDED KEY

        st.markdown("---")
        st.subheader("Debug")
        debug_mode = st.checkbox("Enable Debug Mode", True, key="debug_mode_check")  # ADDED KEY
        show_audio_debug = st.checkbox("Show Audio Debug Details", True, key="show_audio_debug_check")  # ADDED KEY
        test_s3_urls = st.checkbox("Test S3 URL Accessibility", True, key="test_s3_urls_check")  # ADDED KEY
        show_join_details = st.checkbox("Show Join Details", False, key="show_join_details_check")  # ADDED KEY

    # ========== MAIN QUESTION INPUT ==========
    _, col, _ = st.columns([0.1, 2.2, 0.1])
    with col:
        st.write("")
        st.markdown(
            "<h3 style='text-align:left; color:#00426a; font-size:1.6rem !important;'>"
            "Ask anything about CSPC 2023 panels"
            "</h3>",
            unsafe_allow_html=True
        )

        question = st.text_input(
            "",
            placeholder="e.g. What was said about AI and scientific discovery?",
            label_visibility="collapsed",
            key="question_input"  # ADDED KEY
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ========== CONNECT TO WEAVIATE ==========
    try:
        client = get_client(weaviate_url, weaviate_key)
        docchunk_coll = get_collection(client, collection_name)
        collections = client.collections.list_all()
        if "CSPC_Panels" not in collections:
            st.error("CSPC_Panels collection not found! Photos will not be available.")
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {e}")
        st.stop()

    # ========== FILTERS ==========
    with col:
        available_themes = get_all_themes(client)
        all_panels = get_all_panels(client)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(
                '<p style="font-size:1.3rem;font-weight:bold;margin-bottom:0.5rem;">Theme</p>',
                unsafe_allow_html=True
            )
            theme_options = ["All"] + available_themes
            selected_theme = st.selectbox(
                "Theme",
                theme_options,
                key="theme",  # This key was already there
                label_visibility="collapsed"
            )

        with c2:
            st.markdown(
                '<p style="font-size:1.3rem;font-weight:bold;margin-bottom:0.5rem;">Panel</p>',
                unsafe_allow_html=True
            )

            if selected_theme == "All":
                filtered_panels = all_panels
            else:
                try:
                    panels_coll = client.collections.get("CSPC_Panels")
                    response = panels_coll.query.fetch_objects(
                        filters=Filter.by_property("theme").equal(selected_theme),
                        limit=1000
                    )

                    filtered_panel_codes = set()
                    for obj in response.objects:
                        panel_code = obj.properties.get("panel_code")
                        if panel_code:
                            filtered_panel_codes.add(str(panel_code))

                    filtered_panels = [
                        (code, display) for code, display in all_panels
                        if code in filtered_panel_codes
                    ]
                except Exception as e:
                    st.sidebar.warning(f"Could not filter panels by theme: {e}")
                    filtered_panels = all_panels

            panel_options = ["All"] + [display for _, display in filtered_panels]
            selected_panel = st.selectbox(
                "Panel",
                panel_options,
                key="panel",  # This key was already there
                label_visibility="collapsed"
            )

    # ========== SEARCH BUTTON ==========
    _, btn_col, _ = st.columns([1.5, 1, 1.5])
    with btn_col:
        st.markdown("""
        <style>
        button[kind="primary"] {
            background-color: #00426a !important;
            color: white !important;
            font-size: 1.5rem !important;
            padding: 0.8rem 2.2rem !important;
            border-radius: 0.6rem !important;
            border: 2px solid #003356 !important;
        }
        button[kind="primary"]:hover {
            background-color: #005a92 !important;
            border-color: #003356 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        search_clicked = st.button(
            "Search",
            type="primary",
            use_container_width=True,
            key="search_button"  # ADDED KEY
        )

    # ========== SEARCH EXECUTION ==========
    if search_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()
        if not openai_key.startswith("sk-"):
            st.error("Please provide a valid OpenAI API key.")
            st.stop()

        with st.spinner("Searching..."):
            try:
                # Build filters
                filters = []
                if selected_theme != "All":
                    filters.append(Filter.by_property("panel_theme").contains_any([selected_theme]))

                if selected_panel != "All":
                    panel_code = selected_panel.split(" - ")[0].replace("Panel ", "").strip()
                    filters.append(Filter.by_property("panel_code").equal(panel_code))

                where = None
                if len(filters) == 1:
                    where = filters[0]
                elif len(filters) > 1:
                    where = filters[0] & filters[1]

                oai = OpenAI(api_key=openai_key)
                qvec = oai.embeddings.create(
                    model="text-embedding-3-small",
                    input=question
                ).data[0].embedding

                limit = 50 if use_reranker else top_k

                res = docchunk_coll.query.hybrid(
                    query=question,
                    vector=qvec,
                    alpha=alpha,
                    limit=limit,
                    filters=where,
                    return_metadata=MetadataQuery(score=True),
                    return_properties=[
                        "text", "file_name", "chunk_start_time", "chunk_id",
                        "chunk_speakers", "panel_theme", "panel_code", "doc_id"
                    ]
                )
                objects = list(res.objects)

                # DEBUG: Show raw results
                if debug_mode:
                    st.markdown('<div class="debug-box debug-info">', unsafe_allow_html=True)
                    st.markdown("### 🔍 DEBUG: Raw Search Results")
                    st.write(f"**Total results returned:** {len(objects)}")
                    if objects:
                        st.write("**First result properties:**")
                        st.json(objects[0].properties)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Rerank
                if use_reranker and objects:
                    with st.spinner("Reranking..."):
                        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                        pairs = [(question, o.properties.get("text", "")) for o in objects]
                        scores = reranker.predict(pairs)
                        for obj, s in zip(objects, scores):
                            obj._rerank_score = float(s)
                        objects.sort(key=lambda x: getattr(x, "_rerank_score", 0), reverse=True)

                objects = objects[:top_k]

                # AI Answer
                if use_llm and objects:
                    with st.spinner("Generating AI answer..."):
                        context = "\n\n".join(
                            [
                                f"[{i}] Panel {o.properties.get('panel_code', '?')} | "
                                f"{o.properties.get('chunk_start_time', '')}\n"
                                f"{o.properties.get('text', '')}"
                                for i, o in enumerate(objects[:8], 1)
                            ]
                        )
                        resp = oai.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Answer using only the provided context from CSPC 2023 conference panels.",
                                },
                                {
                                    "role": "user",
                                    "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:",
                                },
                            ],
                            temperature=0.2,
                        )
                        answer = resp.choices[0].message.content
                        st.markdown(
                            '<h2 style="font-size:2.8rem; color:#00426a; margin-bottom:1rem;">AI Answer</h2>',
                            unsafe_allow_html=True
                        )
                        st.markdown(answer)
                        st.markdown("---")

                if not objects:
                    st.info("No results found.")
                else:
                    ranked = [{"rank": i + 1, "obj": o} for i, o in enumerate(objects)]
                    panels_dict = defaultdict(list)
                    for item in ranked:
                        panel_code = item["obj"].properties.get("panel_code", "")
                        if not panel_code:
                            file_name = item["obj"].properties.get("file_name", "")
                            match = re.search(r'(\d+)', file_name)
                            panel_code = match.group(1) if match else "Unknown"
                        panels_dict[str(panel_code)].append(item)

                    st.markdown(
                        f'<div class="results-header">Top {len(objects)} Results from '
                        f'{len(panels_dict)} Different Panels</div>',
                        unsafe_allow_html=True
                    )

                    panel_order = []
                    for panel_code, items in panels_dict.items():
                        best_rank = min(item["rank"] for item in items)
                        panel_order.append((panel_code, best_rank, items))
                    panel_order.sort(key=lambda x: x[1])

                    # ========== RENDER PANELS ==========
                    for idx_panel, (panel_code, best_rank, items) in enumerate(panel_order):
                        if idx_panel > 0:
                            st.markdown('<hr class="panel-separator">', unsafe_allow_html=True)

                        first_chunk = items[0]["obj"].properties
                        panel_metadata = get_panel_metadata_from_cspc_panels(client, panel_code)

                        # Panel header with photo
                        title_col, photo_col = st.columns([1, 1])
                        with title_col:
                            st.markdown(
                                f'<div class="panel-number">Panel {panel_code}</div>',
                                unsafe_allow_html=True
                            )
                            if panel_metadata.get("title"):
                                st.markdown(
                                    f'<div class="panel-title">{panel_metadata["title"]}</div>',
                                    unsafe_allow_html=True
                                )
                            st.markdown('<div class="panel-metadata">', unsafe_allow_html=True)
                            theme = panel_metadata.get("theme") or first_chunk.get("panel_theme", "N/A")
                            st.markdown(f"**Theme:** {theme}")
                            if panel_metadata.get("organized_by"):
                                st.markdown(f"**Organized by:** {panel_metadata['organized_by']}")
                            if panel_metadata.get("speakers"):
                                speakers = panel_metadata["speakers"]
                                st.markdown(
                                    f"**Speakers:** "
                                    f"{', '.join(speakers) if isinstance(speakers, list) else speakers}"
                                )
                            if panel_metadata.get("panel_date"):
                                st.markdown(f"**Date:** {panel_metadata['panel_date']}")
                            if panel_metadata.get("panel_url"):
                                st.markdown(
                                    f"**Panel URL:** "
                                    f"[{panel_metadata['panel_url']}]({panel_metadata['panel_url']})"
                                )
                            st.markdown('</div>', unsafe_allow_html=True)

                        with photo_col:
                            photo_url = panel_metadata.get("speaker_photo_url") or panel_metadata.get("photo_url")
                            if photo_url:
                                try:
                                    st.image(photo_url, use_column_width=True, caption=f"Panel {panel_code}")
                                except Exception:
                                    st.info("No photo available")
                            else:
                                st.info("No photo available")

                        # Chunks header
                        st.markdown(f"""
                        <div style="
                            font-size: 1.6rem;
                            font-weight: bold;
                            display: inline-block;
                            border-bottom: 3px solid #00426a;
                            padding-bottom: 6px;
                            line-height: 1.4;
                            margin: 1.5rem 0 1rem 0;
                        ">
                        {len(items)} relevant chunk{"s" if len(items) != 1 else ""} from this panel:
                        </div>
                        """, unsafe_allow_html=True)

                        sorted_items = sorted(items, key=lambda x: x["rank"])
                        chunk_col1, chunk_col2 = st.columns(2)

                        # ========== RENDER CHUNKS WITH AUDIO ==========
                        for idx_chunk, item in enumerate(sorted_items):
                            chunk_props = item["obj"].properties
                            rank = item["rank"]
                            target_col = chunk_col1 if idx_chunk % 2 == 0 else chunk_col2

                            with target_col:
                                st.markdown('<div class="chunk-root">', unsafe_allow_html=True)

                                st.markdown(f"**Rank #{rank}**")
                                st.write(chunk_props.get("text", ""))

                                # Get timestamp
                                raw_time = chunk_props.get("chunk_start_time")
                                if not raw_time or raw_time == "—":
                                    time_str = "00:00:00"
                                else:
                                    time_str = raw_time

                                speakers_str = chunk_props.get("chunk_speakers") or "—"

                                # Display metadata
                                if speakers_str != "—":
                                    st.caption(f"Time: {time_str}")
                                    st.caption(f"Speakers: {speakers_str}")
                                else:
                                    st.caption(f"Time: {time_str}")

                                # ========== AUDIO HANDLING WITH DEBUG ==========
                                file_name = chunk_props.get("file_name")

                                if show_audio_debug:
                                    st.markdown('<div class="debug-box debug-info">', unsafe_allow_html=True)
                                    st.markdown("**🔊 AUDIO DEBUG INFO**")
                                    st.code(f"file_name from DB: {file_name}")
                                    st.code(f"panel_code: {panel_code}")
                                    st.code(f"chunk_start_time: {time_str}")
                                    st.code(f"time_in_seconds: {time_to_seconds(time_str)}")

                                if file_name:
                                    # Method 1: Convert file_name to audio filename
                                    base_name = os.path.splitext(file_name)[0]
                                    audio_file_v1 = f"{base_name}.mp3"

                                    # Method 2: Use panel_code directly
                                    audio_file_v2 = f"Panel_{panel_code}.mp3"

                                    # Method 3: Try without "Panel_" prefix
                                    audio_file_v3 = f"{panel_code}.mp3"

                                    # Build all possible URLs
                                    url_v1 = f"https://cspc-rag.s3.ca-central-1.amazonaws.com/audio/{quote(audio_file_v1)}"
                                    url_v2 = f"https://cspc-rag.s3.ca-central-1.amazonaws.com/audio/{quote(audio_file_v2)}"
                                    url_v3 = f"https://cspc-rag.s3.ca-central-1.amazonaws.com/audio/{quote(audio_file_v3)}"

                                    if show_audio_debug:
                                        st.markdown("**Attempted Audio URLs:**")
                                        st.code(f"V1 (from file_name): {url_v1}")
                                        st.code(f"V2 (Panel_XXX): {url_v2}")
                                        st.code(f"V3 (XXX only): {url_v3}")

                                    # Test URLs if enabled
                                    working_url = None
                                    url_test_results = {}

                                    if test_s3_urls:
                                        import requests

                                        for name, url in [("V1", url_v1), ("V2", url_v2), ("V3", url_v3)]:
                                            try:
                                                response = requests.head(url, timeout=5)
                                                status = response.status_code
                                                url_test_results[name] = {
                                                    "status": status,
                                                    "accessible": status == 200
                                                }
                                                if status == 200 and not working_url:
                                                    working_url = url
                                            except Exception as e:
                                                url_test_results[name] = {
                                                    "status": "ERROR",
                                                    "error": str(e),
                                                    "accessible": False
                                                }

                                        if show_audio_debug:
                                            st.markdown("**URL Test Results:**")
                                            for name, result in url_test_results.items():
                                                if result.get("accessible"):
                                                    st.markdown(f"✅ {name}: HTTP {result['status']} - ACCESSIBLE")
                                                else:
                                                    st.markdown(
                                                        f"❌ {name}: {result.get('status', 'ERROR')} - NOT ACCESSIBLE")
                                                    if 'error' in result:
                                                        st.caption(f"   Error: {result['error']}")

                                    # Use working URL or default to V2
                                    final_url = working_url if working_url else url_v2

                                    if show_audio_debug:
                                        if working_url:
                                            st.markdown(
                                                f'<div class="debug-box debug-success">✅ Using working URL: {final_url}</div>',
                                                unsafe_allow_html=True)
                                        else:
                                            st.markdown(
                                                f'<div class="debug-box debug-error">⚠️ No working URL found. Trying default: {final_url}</div>',
                                                unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)

                                    # Render audio player
                                    try:
                                        st.audio(final_url, start_time=time_to_seconds(time_str))

                                        if show_audio_debug:
                                            st.markdown(
                                                '<div class="debug-box debug-success">✅ Audio player rendered successfully</div>',
                                                unsafe_allow_html=True)
                                    except Exception as e:
                                        st.error(f"❌ Audio player error: {str(e)}")
                                        if show_audio_debug:
                                            st.markdown(
                                                f'<div class="debug-box debug-error">Error details: {str(e)}</div>',
                                                unsafe_allow_html=True)

                                else:
                                    st.caption("⚠️ No file_name in database")
                                    if show_audio_debug:
                                        st.markdown(
                                            '<div class="debug-box debug-error">❌ file_name is missing from chunk properties</div>',
                                            unsafe_allow_html=True)

                                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
                if debug_mode:
                    st.exception(e)

    # ========== SIDEBAR STATUS ==========
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Status")
        try:
            collections = client.collections.list_all()
            st.success("Connected to Weaviate")
            st.caption(f"DocChunk: {'Yes' if 'DocChunk' in collections else 'No'}")
            st.caption(f"CSPC_Panels: {'Yes' if 'CSPC_Panels' in collections else 'No'}")
        except Exception:
            st.error("Not connected")


if __name__ == "__main__":
    main()
