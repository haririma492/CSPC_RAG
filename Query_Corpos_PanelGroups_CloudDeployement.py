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
import streamlit.components.v1 as components


# ============================================================================
# ADMIN CONFIGURATION
# ============================================================================

ADMIN_EMAILS = ["yazdan_hariri@yahoo.com"]  # ⚠️ Your admin email


def is_admin():
    """Determine if current user is an admin"""
    try:
        user_info = st.experimental_user
        if hasattr(user_info, 'email') and user_info.email in ADMIN_EMAILS:
            return True
        if os.getenv("STREAMLIT_RUNTIME_ENV") != "cloud":
            return True
    except:
        pass
    return False


# ============================================================================
# S3 CONFIGURATION
# ============================================================================

S3_BUCKET = "cspc-rag"
S3_REGION = "ca-central-1"
S3_BASE_URL = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"
S3_AUDIO_PREFIX = "audio"

# ============================================================================
# PAGE SETUP
# ============================================================================

st.set_page_config(
    page_title="CSPC 2023 AI Search.",
    layout="wide",
    initial_sidebar_state="expanded"
)

AUDIO_DIR = r"D:\Downloads\AllDays"


# ============================================================================
# WEAVIATE CLIENT
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_client(_url: str, _key: str):
    return connect_to_wcs(cluster_url=_url, auth_credentials=Auth.api_key(_key))


@st.cache_resource(show_spinner=False)
def get_collection(_client, _name: str):
    return _client.collections.get(_name)


# ============================================================================
# PANEL METADATA RETRIEVAL FROM CSPC_PANELS
# ============================================================================

def get_panel_metadata_from_cspc_panels(client, panel_code: str) -> dict:
    """
    Fetch panel metadata including photo_url from CSPC_Panels collection.
    This is a SEPARATE collection from DocChunk.
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
            "panel_url": panel_data.get("panel_url", ""),
            "external_details_url": panel_data.get("external_details_url", ""),
            "_raw": panel_data
        }

    except Exception as e:
        st.sidebar.error(f"Error fetching CSPC_Panels data for panel {panel_code}: {e}")
        return {}


# ============================================================================
# DYNAMIC FILTERS FROM DATA
# ============================================================================

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
                display = f"Panel {panel_code}"
                if title:
                    display += f" - {title[:50]}..." if len(title) > 50 else f" - {title}"
                panels.append((str(panel_code), display))

        panels.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 999999)
        return panels
    except Exception as e:
        st.sidebar.warning(f"Could not fetch panels: {e}")
        return []


# ============================================================================
# AUDIO HELPERS
# ============================================================================

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
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        else:
            return 0
    except (ValueError, AttributeError):
        return 0


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # ========== CUSTOM CSS ==========
    st.markdown("""
      <style>
          .block-container {
              padding-top: 2rem !important;
              max-width: 95% !important;
              font-size: 1.1rem;
          }
          #MainMenu, footer {visibility: hidden;}
          .panel-separator {
              border: none;
              border-top: 2px solid #ccc;
              margin: 2rem 0;
          }

      </style>
      """, unsafe_allow_html=True)

    # ========== SIDEBAR: ABOUT + USER GUIDE ==========
    with st.sidebar:
        with st.expander("📖 About CSPC AI Platform", expanded=False):
            components.html(
                """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
  </head>
  <body style="margin:0; font-family: 'Segoe UI', sans-serif; font-size: 14px;">
    <div style="font-size: 0.9rem; line-height: 1.5; padding-right: 4px;">

      <div style="background: white; padding: 14px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
        <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 16px; border-radius: 10px; margin: -14px -14px 14px -14px;">
          <h2 style="margin: 0; font-size: 1.4em;">CSPC AI PLATFORM</h2>
          <p style="margin: 6px 0 0 0; font-size: 0.9em;">User Guide</p>
        </div>

        <h3 style="color: #667eea; font-size: 1.05em; margin-top: 0;">Getting Started</h3>
        <p><strong>What is this?</strong> An intelligent search system for exploring CSPC 2023 conference insights.</p>

        <h3 style="color: #764ba2; font-size: 1.05em;">How to Use</h3>
        <div style="background: #e8f5e9; padding: 10px; border-radius: 5px; margin: 8px 0;">
          <strong>Example questions:</strong>
          <ul style="margin: 6px 0 0 20px; padding-left: 0;">
            <li>"What was said about AI and scientific discovery?"</li>
            <li>"How did speakers address research security?"</li>
            <li>"What recommendations were made about science communication?"</li>
          </ul>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0;">
          <div style="background: #e8f5e9; padding: 10px; border-radius: 8px; border: 2px solid #4caf50;">
            <h4 style="color: #2e7d32; margin-top: 0; font-size: 0.95em;">✓ DO</h4>
            <ul style="margin: 5px 0 0 20px;">
              <li>Use clear, specific questions</li>
              <li>Try different phrasings</li>
              <li>Use themes or panel filters to narrow results</li>
            </ul>
          </div>

          <div style="background: #ffebee; padding: 10px; border-radius: 8px; border: 2px solid #f44336;">
            <h4 style="color: #c62828; margin-top: 0; font-size: 0.95em;">✗ AVOID</h4>
            <ul style="margin: 5px 0 0 20px;">
              <li>Extremely vague queries</li>
              <li>Single-word searches</li>
              <li>Questions unrelated to CSPC 2023 content</li>
            </ul>
          </div>
        </div>

        <div style="text-align: center; margin-top: 10px; padding: 10px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px; color: white; font-size: 0.9em;">
          <em>Making every moment findable and every insight accessible.</em>
        </div>
      </div>

      <hr style="margin: 18px 0; border: none; border-top: 1px solid #ddd;" />

      <h3 style="color:#333; margin-top:0;">Why CSPC Needs AI Platform</h3>
      <p>
        The CSPC AI Platform represents a transformative approach to knowledge management and conference content accessibility.
        Traditional conference proceedings often result in valuable insights becoming fragmented across hundreds of hours of recordings,
        making it nearly impossible for executives and policy leaders to extract actionable intelligence efficiently.
        This platform addresses that critical gap by transforming the entire CSPC 2023 conference archive into an instantly searchable,
        AI-powered knowledge base where every discussion, recommendation, and insight becomes immediately accessible and actionable.
      </p>
      <p>
        For CSPC executives, this tool serves as a strategic asset for evidence-based decision-making and stakeholder engagement.
        Whether preparing briefing materials, identifying expert speakers for future events, tracking thematic trends across panels,
        or responding to policy inquiries with concrete examples from conference discussions, the platform enables instant access to relevant content
        with precise timestamps and context. Critically, when selecting papers and topics for future conferences, the platform makes it effortless to
        ensure thematic continuity while preventing unwanted repetition—allowing organizers to identify gaps, avoid redundancy, and build on previous
        discussions rather than inadvertently rehashing them. This capability not only maximizes the return on investment from conference programming
        but also positions CSPC as a leader in knowledge mobilization, demonstrating how AI can bridge the gap between scientific discourse and policy.
      </p>

    </div>
  </body>
</html>
                """,
                height=600,
                scrolling=True,
            )

    # ========== CONFIG (FROM ENV, NOT SIDEBAR) ==========
    weaviate_url = os.getenv(
        "WEAVIATE_URL",
        "nsrnedu9q1qfxusokfl8q.c0.us-west3.gcp.weaviate.cloud"
    )
    weaviate_key = os.getenv("WEAVIATE_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    collection_name = os.getenv("WEAVIATE_COLLECTION", "DocChunk")

    # Fixed defaults for end-users (no sliders/toggles)
    alpha = 0.75
    top_k = 10
    use_reranker = True
    debug_mode = False
    show_audio_debug = False
    test_s3_urls = False

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
                <div style="font-size:2.8rem; font-weight:bold; color:#00426a;">
                    CSPC AI Platform
                </div>
                <div style="font-size:1.5rem; font-style:italic; color:#00426a;">
                    .. where every moment is found & could be actionable
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Blue banner
    st.markdown(
        """
        <div style="
            background:#005a92;
            font-size:1.6rem;
            color:#ffffff;
            text-align:center;
            font-weight:700;
            padding:12px 8px;
            margin:5px 0;
        ">
            Phase I • CSPC 2023 Conference
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Red banner – Powered by olgoo.com
    st.markdown(
        """
        <div style="
            background:#c41e3a;
            font-size:1.0rem;
            color:#f0f0f0;
            text-align:right;
            font-weight:500;
            padding:5px 14px;
            margin:0 0 10px 0;
            font-family: 'Gill Sans', 'Trebuchet MS', 'Segoe UI', sans-serif;
        ">
            Powered by:
            <a href="https://olgoo.com" target="_blank" style="color:#f0f0f0; text-decoration:none;">
                olgoo.com
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            key="main_question"
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
                key="filter_theme",
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
                key="filter_panel",
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
            key="btn_search"
        )

    # ========== SEARCH EXECUTION ==========
    if search_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()
        if not openai_key.startswith("sk-"):
            st.error("Service temporarily unavailable. (Missing OpenAI key on server.)")
            st.stop()

        with st.spinner("Searching..."):
            try:
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

                if debug_mode:
                    st.markdown('<div class="debug-box debug-info">', unsafe_allow_html=True)
                    st.markdown("### 🔍 DEBUG: Raw Search Results")
                    st.write(f"**Total results returned:** {len(objects)}")
                    if objects:
                        st.write("**First result properties:**")
                        st.json(objects[0].properties)
                    st.markdown('</div>', unsafe_allow_html=True)

                if use_reranker and objects:
                    with st.spinner("Reranking..."):
                        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                        pairs = [(question, o.properties.get("text", "")) for o in objects]
                        scores = reranker.predict(pairs)
                        for obj, s in zip(objects, scores):
                            obj._rerank_score = float(s)
                        objects.sort(key=lambda x: getattr(x, "_rerank_score", 0), reverse=True)

                objects = objects[:top_k]

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
                        f"""
                        <div style="
                            font-size: 1.7rem;
                            font-weight: 500;
                            text-align: center;
                            margin: 2rem 0 1.5rem 0;
                            color: #00426a;
                        ">
                            Top {len(objects)} Results from {len(panels_dict)} Different Panels
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.write("")

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

                        title_col, photo_col = st.columns([1, 1])
                        with title_col:
                            st.markdown(f'<div class="panel-number">Panel {panel_code}</div>', unsafe_allow_html=True)
                            if panel_metadata.get("title"):
                                st.markdown(
                                    f"""
                                    <div style="
                                        font-size: 1.8rem;
                                        font-weight: 700;
                                        color: #002a5c;
                                        line-height: 1.2;
                                        margin-bottom: 1.5rem;
                                    ">
                                        {panel_metadata["title"]}
                                    </div>
                                    """,
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
                                except Exception:
                                    st.info("No photo available")
                            else:
                                st.info("No photo available")

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
                                st.markdown('<div class="chunk-root">', unsafe_allow_html=True)
                                st.markdown(f"**Rank #{rank}**")
                                st.write(chunk_props.get("text", ""))

                                raw_time = chunk_props.get("chunk_start_time")
                                time_str = raw_time if raw_time and raw_time != "—" else "00:00:00"
                                speakers_str = chunk_props.get("chunk_speakers") or "—"

                                if speakers_str != "—":
                                    st.caption(f"Time: {time_str}")
                                    st.caption(f"Speakers: {speakers_str}")
                                else:
                                    st.caption(f"Time: {time_str}")

                                file_name = chunk_props.get("file_name")

                                if show_audio_debug:
                                    st.markdown('<div class="debug-box debug-info">', unsafe_allow_html=True)
                                    st.markdown("**🔊 AUDIO DEBUG INFO**")
                                    st.code(f"file_name from DB: {file_name}")
                                    st.code(f"panel_code: {panel_code}")
                                    st.code(f"time: {time_str} ({time_to_seconds(time_str)}s)")

                                if file_name:
                                    audio_filename = file_name

                                    if audio_filename.endswith("_transcript.txt"):
                                        audio_filename = audio_filename[:-len("_transcript.txt")]
                                    elif audio_filename.endswith(".txt"):
                                        audio_filename = audio_filename[:-len(".txt")]

                                    if audio_filename.endswith("_transcript"):
                                        audio_filename = audio_filename[:-len("_transcript")]

                                    audio_filename = audio_filename + ".mp3"
                                    audio_url = f"https://cspc-rag.s3.ca-central-1.amazonaws.com/audio/{quote(audio_filename)}"

                                    if show_audio_debug:
                                        st.code(f"Original file_name: {file_name}")
                                        st.code(f"Cleaned audio filename: {audio_filename}")
                                        st.code(f"Final S3 URL: {audio_url}")

                                    if test_s3_urls:
                                        import requests
                                        try:
                                            headers = {'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-1024'}
                                            r = requests.get(audio_url, timeout=5, stream=True, headers=headers)
                                            if r.status_code in [200, 206]:
                                                if show_audio_debug:
                                                    st.markdown(f"✅ URL Test: {r.status_code} - ACCESSIBLE")
                                            else:
                                                if show_audio_debug:
                                                    st.markdown(f"❌ URL Test: {r.status_code}")
                                        except Exception as e:
                                            if show_audio_debug:
                                                st.markdown(f"❌ URL Test Error: {str(e)[:50]}")

                                    if show_audio_debug:
                                        st.markdown('</div>', unsafe_allow_html=True)

                                    try:
                                        st.audio(audio_url, start_time=time_to_seconds(time_str))
                                    except Exception as e:
                                        st.error(f"Audio error: {e}")
                                else:
                                    st.caption("⚠️ No file_name")

                                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
                if debug_mode:
                    st.exception(e)


if __name__ == "__main__":
    main()
