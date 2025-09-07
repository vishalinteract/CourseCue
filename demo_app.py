# ----------------------------------------------------------------------
# CourseCue — Demo UI (reference-aligned two-column layout)
# ----------------------------------------------------------------------
# Requires: streamlit, requests
# Optional env: EDUREC_API_BASE (defaults to http://127.0.0.1:8010)
# ----------------------------------------------------------------------

import os
import base64
import requests
import streamlit as st

# =========================
# Helpers
# =========================

def api_get(base, path, params=None, timeout=25):
    r = requests.get(base.rstrip("/") + path, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def api_post(base, path, body=None, params=None, timeout=60):
    r = requests.post(base.rstrip("/") + path, json=body or {}, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def card_html(title, reason, item_id, topic=None, difficulty=None):
    title = title or f"Learning Path {item_id}"
    chips = []
    if topic:
        chips.append(
            "<span style='padding:2px 8px;border-radius:12px;background:#eff6ff;"
            "color:#1e40af;font-size:11px;margin-right:6px'>"
            f"{topic}</span>"
        )
    if difficulty:
        chips.append(
            "<span style='padding:2px 8px;border-radius:12px;background:#f0fdf4;"
            "color:#14532d;font-size:11px'>"
            f"{difficulty}</span>"
        )
    chips_html = "".join(chips)
    return f"""
    <div style="border:1px solid #eceef3;border-radius:14px;padding:14px 16px;margin-bottom:14px;background:#ffffff">
      <div style="font-weight:800;font-size:18px;line-height:1.2;margin-bottom:6px">{title}</div>
      <div style="color:#6b7280;font-size:12px;margin-bottom:8px">ID: <code>{item_id}</code></div>
      <div style="margin-bottom:8px">{chips_html}</div>
      <div style="font-size:13px">{reason}</div>
    </div>
    """

# =========================
# Branding / Page config
# =========================

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
_LOGO_PATH = os.path.join(_ASSETS_DIR, "coursecue_logo.png")
_PAGE_ICON = _LOGO_PATH if os.path.exists(_LOGO_PATH) else None

st.set_page_config(page_title="CourseCue", page_icon=_PAGE_ICON, layout="wide")

LOGO_WIDTH = 44
GAP_PX = 10
logo_b64 = img_to_b64(_LOGO_PATH) if os.path.exists(_LOGO_PATH) else ""

st.markdown(
    f"""
    <style>
      .cc-header {{ display:flex; align-items:center; gap:{GAP_PX}px; margin:0 0 8px 0; }}
      .cc-title  {{ font-size:28px; font-weight:800; line-height:1.1; }}
      .cc-tag    {{ color:#6b7280; font-size:14px; margin-top:2px; }}
      section.main > div.block-container {{ padding-top: 10px; }}
    </style>
    <div class="cc-header">
      {'<img src="data:image/png;base64,' + logo_b64 + f'" width="{LOGO_WIDTH}"/>' if logo_b64 else ''}
      <div>
        <div class="cc-title">CourseCue</div>
        <div class="cc-tag">discover. cue. do.</div>
      </div>
    </div>
    <hr style="margin:6px 0 12px 0; border:0; border-top:1px solid #e5e7eb"/>
    """,
    unsafe_allow_html=True,
)

# =========================
# Settings (API base)
# =========================

with st.expander("Settings"):
    api_base = st.text_input(
        "API address",
        value=os.environ.get("EDUREC_API_BASE", "http://127.0.0.1:8010"),
        help="Where the CourseCue API is running (FastAPI).",
    )

# =========================
# Load facets (safe if API isn't up yet)
# =========================

topics_list, levels_list = [], []
try:
    facets = api_get(api_base, "/debug/catalog_facets")
    topics_list = facets.get("topics", [])
    levels_list = facets.get("difficulties", [])
except Exception:
    pass

# =========================
# Main Form (reference layout)
# =========================

with st.form("recommend_form", clear_on_submit=False):
    # Slightly wider right column to mirror your screenshot
    left, right = st.columns([1.05, 1.35], gap="large")

    # ---------- LEFT COLUMN ----------
    with left:
        user_id = st.number_input(
            "Learner ID",
            min_value=1,
            value=1,
            step=1,
            help="Enter the learner’s numeric ID.",
        )

        # Time range selector
        time_range = st.selectbox(
            "Time range",
            ["All time", "Last 7 days", "Last 30 days"],
            index=2,
            help="For ‘Popular now’, choose a window; ‘All time’ shows consistent popularity.",
        )

        topic = st.selectbox("Topic", ["(Any)"] + topics_list, index=0)
        topic = None if topic == "(Any)" else topic

        difficulty = st.selectbox("Level", ["(Any)"] + levels_list, index=0)
        difficulty = None if difficulty == "(Any)" else difficulty

        # Utility buttons live in the left column (per reference)
        bl1, bl2, bl3, bl4 = st.columns([1, 1.2, 1.6, 1.1])
        with bl1:
            health_clicked = st.form_submit_button("Health")
        with bl2:
            reload_fast_clicked = st.form_submit_button("Reload\n(fast)")
        with bl3:
            rebuild_clicked = st.form_submit_button("Rebuild model\n(full)")
        with bl4:
            stats_clicked = st.form_submit_button("Debug\n/stats")

    # ---------- RIGHT COLUMN ----------
    with right:
        # Slider at top — now with the correct title
        topk = st.slider(
            label="How many suggestions?",
            min_value=1,
            max_value=20,
            value=8,
            help="Controls the number of results returned (top-K).",
        )

        st.markdown("### Recommendation style")
        mode_human = st.radio(
            label="",
            options=["Smart (Personalized)", "Popular now (Trending)"],
            index=1,  # default to Popular in screenshot
        )
        exclude_seen = st.checkbox("Hide items they’ve already seen", value=True)
        strict = st.checkbox(
            "Force personalization (don’t fall back to popular if there’s no history)",
            value=True,
        )

        # Guidance line (mirrors screenshot text under Popular)
        if "Smart" in mode_human:
            st.caption("Personalized picks use the learner’s history. Topic/Level can narrow the results.")
        else:
            st.caption("Trending picks honor the time range. Topic/Level can narrow what’s popular now.")

        # Big submit on bottom-right
        st.write("")  # spacer
        _, submit_col = st.columns([1, 1])
        with submit_col:
            go = st.form_submit_button("Get Recommendations", type="primary", use_container_width=True)

# =========================
# Button handlers
# =========================

if 'api_base' in locals() and 'health_clicked' in locals() and health_clicked:
    try:
        st.success("API reachable ✅")
        st.json(api_get(api_base, "/health"))
    except Exception as e:
        st.error(f"Health failed: {e}")

if 'api_base' in locals() and 'reload_fast_clicked' in locals() and reload_fast_clicked:
    try:
        st.json(api_post(api_base, "/reload_index", params={"fast": "true"}))
    except Exception as e:
        st.error(f"Reload fast failed: {e}")

if 'api_base' in locals() and 'rebuild_clicked' in locals() and rebuild_clicked:
    try:
        with st.spinner("Rebuilding item-to-item model…"):
            st.json(api_post(api_base, "/reload_index", params={"fast": "false"}, timeout=300))
    except Exception as e:
        st.error(f"Rebuild failed: {e}")

if 'api_base' in locals() and 'stats_clicked' in locals() and stats_clicked:
    try:
        st.json(api_get(api_base, "/debug/stats"))
    except Exception as e:
        st.error(f"Stats failed: {e}")

# =========================
# Get Recommendations
# =========================

if 'api_base' in locals() and 'go' in locals() and go:
    try:
        mode = "itemknn" if "Smart" in mode_human else "pop"
        pop_window_days = None
        if mode == "pop":
            pop_window_days = 7 if time_range == "Last 7 days" else (30 if time_range == "Last 30 days" else None)

        params = {
            "k": int(topk),
            "mode": mode,
            "strict": "true" if strict else "false",
            "pop_window_days": pop_window_days,
            "topic": topic,
            "difficulty": difficulty,
            "exclude_seen": "true" if exclude_seen else "false",
        }

        with st.spinner("Finding the best courses…"):
            data = api_post(
                api_base,
                "/recommendations",
                {"user_id": int(user_id), "seen_items": []},  # seen field removed
                params,
            )

        st.markdown(
            f"<span style='padding:3px 8px;border-radius:999px;background:#eef2ff;"
            f"color:#1f3a8a;font-size:12px'>Mode used: {data.get('mode_used')}</span>",
            unsafe_allow_html=True,
        )

        items = data.get("items", [])
        if not items:
            st.warning("No recommendations matched. Try loosening filters, switch to **Popular now**, or verify the learner ID.")
        else:
            col_a, col_b = st.columns(2)
            for idx, it in enumerate(items):
                html = card_html(
                    title=it.get("title"),
                    reason=it.get("reason", ""),
                    item_id=it["id"],
                    topic=it.get("topic"),
                    difficulty=it.get("difficulty"),
                )
                (col_a if idx % 2 == 0 else col_b).markdown(html, unsafe_allow_html=True)

        with st.expander("Raw API response"):
            st.json(data)

    except Exception as e:
        st.error(f"Request failed: {e}")

# =========================
# Footer
# =========================

st.markdown(
    "<div style='color:#9ca3af;font-size:12px;margin-top:18px'>CourseCue • discover. cue. do.</div>",
    unsafe_allow_html=True,
)
