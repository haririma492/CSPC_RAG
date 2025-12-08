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
# ADMIN CONFIGURATION - ✅ ALL 3 FIXES APPLIED!
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


def apply_restrictions():
    """Hide sidebar and Streamlit UI elements for regular users"""
    admin_mode = is_admin()
    if not admin_mode:
        st.markdown("""<style>
            [data-testid="stSidebar"] { display: none; }
            header { visibility: hidden; }
            #MainMenu { visibility: hidden; }
            footer { visibility: hidden; }
            .block-container { padding-left: 1rem; padding-right: 1rem; max-width: 100%; }
        </style>""", unsafe_allow_html=True)
    else:
        st.markdown("""<style>
            [data-testid="stSidebar"] { border-left: 3px solid #FF4B4B; }
        </style>""", unsafe_allow_html=True)
        with st.sidebar:
            st.success("🔧 Admin Mode")
    return admin_mode


def show_user_guide():
    """Display a small 'About CSPC AI Platform' chip on the top-left with expandable content (using components.html)."""

    # HTML + CSS together; rendered via components.html (no markdown escaping)
    html = """
    <html>
    <head>
      <style>
        .cspc-about-wrapper {
            display: inline-block;
            margin-bottom: 0.75rem;
        }
        .cspc-about-toggle {
            display: inline-block;
        }
        .cspc-about-toggle summary {
            list-style: none;
            cursor: pointer;
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            border: 1px solid #ccc;
            background: #f5f5f5;
            font-size: 0.75rem;
            font-weight: 500;
            color: #333;
        }
        .cspc-about-toggle summary::-webkit-details-marker {
            display: none;
        }
        .cspc-about-content {
            margin-top: 10px;
            font-family: "Segoe UI", sans-serif;
            background: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            max-width: 900px;
        }
      </style>
    </head>
    <body>
      <div class="cspc-about-wrapper">
        <details class="cspc-about-toggle">
          <summary>📖 About CSPC AI Platform</summary>
          <div class="cspc-about-content">

            <h2 style="color:#333; margin-top:0;">Why CSPC Needs AI Platform</h2>
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

            <div style="margin-top: 25px; background: white; padding: 20px; border-radius: 10px;">
              <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 10px; margin: -20px -20px 20px -20px;">
                <h1 style="margin: 0; font-size: 2.1em;">CSPC AI PLATFORM</h1>
                <p style="margin: 10px 0 0 0; font-size: 1em;">User Guide</p>
              </div>

              <div>
                <h2 style="color: #667eea;">Getting Started</h2>
                <p><strong>What is the CSPC AI Platform?</strong> An intelligent search system for exploring CSPC 2023 Conference insights.</p>

                <h3 style="color: #764ba2;">How to Use</h3>
                <div style="background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 10px 0;">
                  <strong>Example Questions:</strong>
                  <ul>
                    <li>"What was said about AI and scientific discovery?"</li>
                    <li>"How did speakers address research security?"</li>
                    <li>"What recommendations were made about science communication?"</li>
                  </ul>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0;">
                  <div style="background: #e8f5e9; padding: 15px; border-radius: 8px; border: 2px solid #4caf50;">
                    <h4 style="color: #2e7d32; margin-top: 0;">✓ DO:</h4>
                    <ul style="margin: 5px 0 0 20px;">
                      <li>Use clear, specific questions</li>
                      <li>Try different phrasings</li>
                      <li>Use filters to narrow results</li>
                    </ul>
                  </div>
                  <div style="background: #ffebee; padding: 15px; border-radius: 8px; border: 2px solid #f44336;">
                    <h4 style="color: #c62828; margin-top: 0;">✗ AVOID:</h4>
                    <ul style="margin: 5px 0 0 20px;">
                      <li>Extremely vague queries</li>
                      <li>Single-word searches</li>
                      <li>Content outside CSPC 2023</li>
                    </ul>
                  </div>
                </div>

                <div style="text-align: center; margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
                  <em>Making every moment findable and every insight accessible.</em>
                </div>
              </div>
            </div>

          </div>
        </details>
      </div>
    </body>
    </html>
    """

    # Put it in a small column so it sits at top-left and stays small
    col_about, _ = st.columns([0.3, 0.7])
    with col_about:
        components.html(html, height=450, scrolling=True)

# ============================================================================
# S3 CONFIGURATION
# ============================================================================

S3_BUCKET = "cspc-rag"
S3_REGION = "ca-central-1"
S3_BASE_URL = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"
S3_AUDIO_PREFIX = "audio"

# ============================================================================
# ADMIN MODE & PAGE SETUP
# ============================================================================

admin_mode = apply_restrictions()

st.set_page_config(
    page_title="CSPC 2023 AI Search",
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
        if len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:  # MM:SS
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
        .main > div {padding-top:0!important;}
        .block-container {padding-top:1rem!important; max-width:95%!important; font-size:1.1rem;}
        #MainMenu, footer, header {visibility:hidden;}
        .banner {text-align:center; padding:12px; color:white; font-weight:bold; margin:5px 0;}
        .panel-header {background:#f0f2f6; padding:15px; border-radius:10px; margin:10px 0;}
        .photo-container {border: 2px solid #ddd; border-radius: 8px; padding: 5px;}

        p, li, span, label {font-size: 1.3rem !important;}

        .panel-metadata {font-size: 1.9rem !important; line-height: 2.4;}
        .panel-number {font-size: 1.8rem !important; font-weight: bold; margin-bottom: 0.5rem !important; color: #00426a;}
        .panel-title {font-size: 2.4rem !important; font-weight: bold; line-height: 1.3; margin-bottom: 1.5rem !important;}
        .chunks-header {font-size: 1.6rem !important; font-weight: bold !important; margin: 1.5rem 0 1rem 0 !important;}
        input[type="text"] {font-size: 1.5rem !important; padding: 1rem !important;}
        .stCaption, [data-testid="stCaptionContainer"] {font-size: 1.1rem !important;}
        button {font-size: 1.3rem !important;}
        audio {width: 100%;}

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

        .debug-warning {
            background-color: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }
    </style>
    """, unsafe_allow_html=True)

    # ========== USER GUIDE (Very Top Left) ==========
    show_user_guide()

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
            key="cfg_weaviate_url"
        )

        weaviate_key = st.text_input(
            "Weaviate Key",
            type="password",
            value=os.getenv("WEAVIATE_API_KEY", ""),
            key="cfg_weaviate_key"
        )

        openai_key = st.text_input(
            "OpenAI Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            key="cfg_openai_key"
        )

        collection_name = st.text_input(
            "Chunks Collection",
            "DocChunk",
            key="cfg_collection"
        )

        st.markdown("---")
        st.subheader("Search Settings")
        alpha = st.slider("Hybrid Alpha", 0.0, 1.0, 0.75, key="cfg_alpha")
        top_k = st.number_input("Top Results", 1, 30, 10, key="cfg_topk")
        use_reranker = st.checkbox("Use Reranker", True, key="cfg_reranker")
        use_llm = st.checkbox("Generate AI Answer", True, key="cfg_llm")

        st.markdown("---")
        st.subheader("Debug")
        debug_mode = st.checkbox("Enable Debug Mode", False, key="cfg_debug")
        show_audio_debug = st.checkbox("Show Audio Debug Details", False, key="cfg_audio_debug")
        test_s3_urls = st.checkbox("Test S3 URL Accessibility", False, key="cfg_test_urls")

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
            st.error("Please provide a valid OpenAI API key.")
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

                if use_llm and objects:
                    with st.spinner("Generating AI answer..."):
                        context = "\n\n".join([
                            f"[{i}] Panel {o.properties.get('panel_code', '?')} | "
                            f"{o.properties.get('chunk_start_time', '')}\n"
                            f"{o.properties.get('text', '')}"
                            for i, o in enumerate(objects[:8], 1)
                        ])
                        resp = oai.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system",
                                 "content": "Answer using only the provided context from CSPC 2023 conference panels."},
                                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"}
                            ],
                            temperature=0.2
                        )
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
                        unsafe_allow_html=True
                    )

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

                                # ========== AUDIO URL GENERATION ==========
                                file_name = chunk_props.get("file_name")

                                if show_audio_debug:
                                    st.markdown('<div class="debug-box debug-info">', unsafe_allow_html=True)
                                    st.markdown("**🔊 AUDIO DEBUG INFO**")
                                    st.code(f"file_name from DB: {file_name}")
                                    st.code(f"panel_code: {panel_code}")
                                    st.code(f"time: {time_str} ({time_to_seconds(time_str)}s)")

                                if file_name:
                                    # Remove _transcript.txt suffix properly
                                    audio_filename = file_name

                                    if audio_filename.endswith("_transcript.txt"):
                                        audio_filename = audio_filename[:-len("_transcript.txt")]
                                    elif audio_filename.endswith(".txt"):
                                        audio_filename = audio_filename[:-len(".txt")]

                                    if audio_filename.endswith("_transcript"):
                                        audio_filename = audio_filename[:-len("_transcript")]

                                    audio_filename = audio_filename + ".mp3"

                                    # Build S3 URL with proper encoding
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
