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


def time_to_seconds(t: str) -> int:
    if not t or t in ("N/A", "—", "", "0"):
        return 0
    try:
        parts = [int(x) for x in t.replace(".", ":").split(":") if x.isdigit()]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0]
    except:
        return 0


# ========================
# MAIN APP
# ========================
def main():
    # ========== CUSTOM CSS (Fixed & Improved Chunk Border) ==========
    st.markdown("""
    <style>
        .main > div {padding-top:0!important;}
        .block-container {padding-top:1rem!important; max-width:95%!important; font-size:1.1rem;}
        #MainMenu, footer, header {visibility:hidden;}
        .banner {text-align:center; padding:12px; color:white; font-weight:bold; margin:5px 0;}
        .panel-header {background:#f0f2f6; padding:15px; border-radius:10px; margin:10px 0;}
        .photo-container {border: 2px solid #ddd; border-radius: 8px; padding: 5px;}

        /* Larger text everywhere – BUT DO NOT TOUCH div */
        p, li, span, label {font-size: 1.3rem !important;}

        .panel-metadata {font-size: 1.9rem !important; line-height: 2.4;}
        .panel-number {font-size: 1.8rem !important; font-weight: bold; margin-bottom: 0.5rem !important; color: #00426a;}
        .panel-title {font-size: 2.4rem !important; font-weight: bold; line-height: 1.3; margin-bottom: 1.5rem !important;}
        .chunks-header {font-size: 1.6rem !important; font-weight: bold !important; margin: 1.5rem 0 1rem 0 !important;}
        input[type="text"] {font-size: 1.5rem !important; padding: 1rem !important;}
        .stCaption, [data-testid="stCaptionContainer"] {font-size: 1.1rem !important;}
        button {font-size: 1.3rem !important;}
        audio {width: 100%;}

        ...
    </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2 = st.columns([1.3, 4])
    with col1:
        st.image("https://sciencepolicy.ca/wp-content/uploads/2020/09/cspc-logo.png", width=300)
    with col2:
        st.markdown(
            """
            <div style="
                display:flex;
                align-items:center;
                margin-left:40px;
                gap: 1.5rem;
                color:#00426a;
            ">
                <div style="font-size:3rem; font-weight:bold; color:#00426a;">
                    CSPC AI Platform
                </div>
                <div style="font-size:1.5rem; font-style:italic; color:#00426a;">
                    (Every moment could be actionable)
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

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        weaviate_url = st.text_input(
            "Weaviate URL",
            value=os.getenv("WEAVIATE_URL", "nsrnedu9q1qfxusokfl8q.c0.us-west3.gcp.weaviate.cloud"),
        )

        weaviate_key = st.text_input(
            "Weaviate Key",
            type="password",
            value=os.getenv("WEAVIATE_API_KEY", ""),
        )

        openai_key = st.text_input(
            "OpenAI Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
        )

        collection_name = st.text_input("Chunks Collection", "DocChunk")

        st.markdown("---")
        st.subheader("Search Settings")
        alpha = st.slider("Hybrid Alpha", 0.0, 1.0, 0.75)
        top_k = st.number_input("Top Results", 1, 30, 10)
        use_reranker = st.checkbox("Use Reranker", True)
        use_llm = st.checkbox("Generate AI Answer", True)

        st.markdown("---")
        st.subheader("Debug")
        debug_mode = st.checkbox("Enable Debug Mode", False)
        show_join_details = st.checkbox("Show Join Details", False)

    # Main Question Input
    _, col, _ = st.columns([0.1, 2.2, 0.1])
    with col:
        st.markdown(
            "<h2 style='text-align:left; color:#00426a; font-size:3rem;'></h2>",
            unsafe_allow_html=True)
        st.markdown("""
        <style>
        /* Make typed text bigger inside any text_input */
        input[type="text"] {
            font-size: 1.8rem !important;     /* Bigger typing text */
            padding: 0.8rem 1rem !important;  /* Optional: bigger box */
        }

        /* Make placeholder text bigger also */
        input::placeholder {
            font-size: 1.4rem !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(
            "<h2 style='text-align:left; color:#00426a; font-size:3rem;'>Ask anything about CSPC 2023 panels</h2>",
            unsafe_allow_html=True)
        question = st.text_input("", placeholder="e.g. What was said about AI and scientific discovery?",
                                 label_visibility="collapsed")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Connect to Weaviate
    try:
        client = get_client(weaviate_url, weaviate_key)
        docchunk_coll = get_collection(client, collection_name)
        collections = client.collections.list_all()
        if "CSPC_Panels" not in collections:
            st.error("CSPC_Panels collection not found! Photos will not be available.")
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {e}")
        st.stop()

    # Filters - DYNAMIC with dependent dropdowns
    with col:
        # Fetch available options
        available_themes = get_all_themes(client)
        all_panels = get_all_panels(client)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<p style="font-size:1.3rem;font-weight:bold;margin-bottom:0.5rem;">Theme</p>',
                        unsafe_allow_html=True)
            theme_options = ["All"] + available_themes
            selected_theme = st.selectbox("Theme", theme_options, key="theme", label_visibility="collapsed")

        with c2:
            st.markdown('<p style="font-size:1.3rem;font-weight:bold;margin-bottom:0.5rem;">Panel</p>',
                        unsafe_allow_html=True)

            # Filter panels based on selected theme
            if selected_theme == "All":
                filtered_panels = all_panels
            else:
                # Fetch panels that match the selected theme
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

                    # Filter the all_panels list to only include matching panels
                    filtered_panels = [(code, display) for code, display in all_panels if code in filtered_panel_codes]
                except Exception as e:
                    st.sidebar.warning(f"Could not filter panels by theme: {e}")
                    filtered_panels = all_panels

            panel_options = ["All"] + [display for _, display in filtered_panels]
            selected_panel = st.selectbox("Panel", panel_options, key="panel", label_visibility="collapsed")

    _, btn_col, _ = st.columns([1.5, 1, 1.5])
    with btn_col:
        st.markdown("""
        <style>
        /* Style all primary buttons (like your Search button) */
        button[kind="primary"] {
            background-color: #00426a !important;  /* button background */
            color: white !important;               /* text color */
            font-size: 1.5rem !important;          /* bigger text */
            padding: 0.8rem 2.2rem !important;     /* bigger button */
            border-radius: 0.6rem !important;      /* rounded corners */
            border: 2px solid #003356 !important;
        }

        /* Optional: nicer hover effect */
        button[kind="primary"]:hover {
            background-color: #005a92 !important;
            border-color: #003356 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        search_clicked = st.button("Search", type="primary", use_container_width=True)
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
                    # Extract panel code from "Panel 333 - Title" format
                    panel_code = selected_panel.split(" - ")[0].replace("Panel ", "").strip()
                    filters.append(Filter.by_property("panel_code").equal(panel_code))

                where = None
                if len(filters) == 1:
                    where = filters[0]
                elif len(filters) > 1:
                    where = filters[0] & filters[1]

                oai = OpenAI(api_key=openai_key)
                qvec = oai.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
                limit = 50 if use_reranker else top_k

                res = docchunk_coll.query.hybrid(
                    query=question, vector=qvec, alpha=alpha, limit=limit, filters=where,
                    return_metadata=MetadataQuery(score=True),
                    return_properties=["text", "file_name", "chunk_start_time", "chunk_id", "chunk_speakers",
                                       "panel_theme", "panel_code", "doc_id"]
                )
                objects = list(res.objects)

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
                        context = "\n\n".join([
                                                  f"[{i}] Panel {o.properties.get('panel_code', '?')} | {o.properties.get('chunk_start_time', '')}\n{o.properties.get('text', '')}"
                                                  for i, o in enumerate(objects[:8], 1)])
                        resp = oai.chat.completions.create(model="gpt-4o-mini", messages=[
                            {"role": "system",
                             "content": "Answer using only the provided context from CSPC 2023 conference panels."},
                            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"}
                        ], temperature=0.2)
                        answer = resp.choices[0].message.content
                        st.markdown('<h2 style="font-size:2.8rem; color:#00426a; margin-bottom:1rem;">AI Answer</h2>',
                                    unsafe_allow_html=True)
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
                        f'<div class="results-header">Top {len(objects)} Results from {len(panels_dict)} Different Panels</div>',
                        unsafe_allow_html=True)

                    panel_order = []
                    for panel_code, items in panels_dict.items():
                        best_rank = min(item["rank"] for item in items)
                        panel_order.append((panel_code, best_rank, items))
                    panel_order.sort(key=lambda x: x[1])

                    for idx_panel, (panel_code, best_rank, items) in enumerate(panel_order):
                        if idx_panel > 0:
                            st.markdown('<hr class="panel-separator">', unsafe_allow_html=True)

                        first_chunk = items[0]["obj"].properties
                        panel_metadata = get_panel_metadata_from_cspc_panels(client, panel_code)

                        # Panel header with photo
                        title_col, photo_col = st.columns([1, 1])
                        with title_col:
                            st.markdown(f'<div class="panel-number">Panel {panel_code}</div>', unsafe_allow_html=True)
                            if panel_metadata.get("title"):
                                st.markdown(f'<div class="panel-title">{panel_metadata["title"]}</div>',
                                            unsafe_allow_html=True)
                            st.markdown('<div class="panel-metadata">', unsafe_allow_html=True)
                            theme = panel_metadata.get("theme") or first_chunk.get("panel_theme", "N/A")
                            st.markdown(f"**Theme:** {theme}")
                            if panel_metadata.get("organized_by"):
                                st.markdown(f"**Organized by:** {panel_metadata['organized_by']}")
                            if panel_metadata.get("speakers"):
                                speakers = panel_metadata["speakers"]
                                st.markdown(
                                    f"**Speakers:** {', '.join(speakers) if isinstance(speakers, list) else speakers}")
                            if panel_metadata.get("panel_date"):
                                st.markdown(f"**Date:** {panel_metadata['panel_date']}")
                            if panel_metadata.get("panel_url"):
                                st.markdown(
                                    f"**Panel URL:** [{panel_metadata['panel_url']}]({panel_metadata['panel_url']})")
                            st.markdown('</div>', unsafe_allow_html=True)

                        with photo_col:
                            photo_url = panel_metadata.get("speaker_photo_url") or panel_metadata.get("photo_url")
                            if photo_url:
                                try:
                                    st.image(photo_url, use_column_width=True, caption=f"Panel {panel_code}")
                                except:
                                    st.info("No photo available")
                            else:
                                st.info("No photo available")

                        # Chunks header with inline underline
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

                        for idx_chunk, item in enumerate(sorted_items):
                            chunk_props = item["obj"].properties
                            rank = item["rank"]
                            target_col = chunk_col1 if idx_chunk % 2 == 0 else chunk_col2

                            with target_col:
                                # THIS IS THE ONLY LINE YOU NEED FOR THE BORDER
                                st.markdown('<div class="chunk-root"></div>', unsafe_allow_html=True)

                                st.markdown(f"**Rank #{rank}**")
                                st.write(chunk_props.get("text", ""))

                                time_str = chunk_props.get('chunk_start_time', '—')
                                speakers_str = chunk_props.get('chunk_speakers', '—')

                                if speakers_str and speakers_str != '—':
                                    st.caption(f"Time: {time_str}")
                                    st.caption(f"Speakers: {speakers_str}")
                                else:
                                    st.caption(f"Time: {time_str}")

                                audio_path = find_audio_file(chunk_props.get("file_name"))
                                if audio_path:
                                    st.audio(audio_path, start_time=time_to_seconds(time_str))

            except Exception as e:
                st.error(f"Error: {e}")
                if debug_mode:
                    st.exception(e)

    # Sidebar status
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Status")
        try:
            collections = client.collections.list_all()
            st.success("Connected to Weaviate")
            st.caption(f"DocChunk: {'Yes' if 'DocChunk' in collections else 'No'}")
            st.caption(f"CSPC_Panels: {'Yes' if 'CSPC_Panels' in collections else 'No'}")
        except:
            st.error("Not connected")


if __name__ == "__main__":
    main()