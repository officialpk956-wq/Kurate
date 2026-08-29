import streamlit as st
import corpus
import retrieval
import html
import analytics
import uuid
import time

# Set False before any real deployment — full stack traces should never be shown to end users,
# kept True here only because this is an internal evaluation demo.
DEBUG_MODE = True

# 1. Page config with emoji favicon
st.set_page_config(page_title="Kurate — Trust-Ranked Search", page_icon="🔎", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:wght@600;700&family=Public+Sans:wght@400;500;700&display=swap');

:root {
  --ink: #15181A;
  --ink-soft: #545C58;
  --paper: #F6F6F2;
  --surface: #FFFFFF;
  --rule: #DEDFD5;
  --trust: #12605A;
  --trust-wash: #E3EFEC;
  --risk: #9D3F16;
}

@media (prefers-color-scheme: dark) {
  :root {
    --ink: #F6F6F2;
    --ink-soft: #A3A8A6;
    --paper: #0E1111;
    --surface: #15181A;
    --rule: #2A2E2C;
    --trust: #24B6AC;
    --trust-wash: #123B38;
    --risk: #E66329;
  }
}

/* Typography */
h1, h2, h3, h4, h5, h6, .wordmark, .zero-result-title {
  font-family: 'Newsreader', serif !important;
}
html, body, p, span, div, button, input, select {
  font-family: 'Public Sans', sans-serif !important;
}

/* Header styling */
.header-container {
  border-bottom: 2px solid var(--trust);
  padding-bottom: 16px;
  margin-bottom: 32px;
}
.wordmark {
  font-size: 28px;
  font-weight: 600;
  color: var(--ink);
  margin: 0;
  line-height: 1.2;
}
.header-eyebrow {
  font-size: 12px;
  font-weight: 500;
  color: var(--ink-soft);
  text-transform: uppercase;
  letter-spacing: 1px;
}

/* Restyle default info banner into system notice */
.stAlert[data-baseweb="notification"] {
  background-color: var(--trust-wash) !important;
  color: var(--ink-soft) !important;
  border-radius: 6px;
  border: none !important;
  padding: 12px 16px;
}
.stAlert[data-baseweb="notification"] p {
  font-size: 14px;
  color: var(--ink-soft) !important;
}

/* Single-block HTML Cards */
.kurate-card {
  background-color: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: 20px;
  margin-bottom: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  transition: box-shadow 0.2s ease;
}
.kurate-card:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.card-author {
  font-weight: 700;
  color: var(--ink);
  margin-bottom: 8px;
}
.card-text {
  font-size: 14px;
  color: var(--ink);
  margin-bottom: 12px;
  line-height: 1.5;
}

/* Badges */
.badge {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 12px;
}
.badge-direct { background-color: var(--trust); color: white; }
@media (prefers-color-scheme: dark) { .badge-direct { color: var(--ink); } }
.badge-2hop { background-color: var(--trust-wash); color: var(--trust); }
.badge-out { background-color: var(--rule); color: var(--ink-soft); }

/* Expander Grid */
div[data-testid="stExpander"] {
  margin-top: -8px;
  margin-bottom: 16px;
}
.metric-box {
  margin-bottom: 8px;
}
.metric-label {
  font-size: 12px;
  color: var(--ink-soft);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 2px;
}
.metric-val {
  font-size: 14px;
  font-weight: 700;
  color: var(--ink);
}

/* Sidebar Settings Header */
.sidebar-header {
  font-size: 12px;
  font-weight: 600;
  color: var(--ink-soft);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 16px;
}

/* High trust warning strip */
.high-trust-warning {
  background-color: var(--risk);
  color: white;
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 16px;
}
@media (prefers-color-scheme: dark) {
  .high-trust-warning { color: var(--ink); }
}

/* Zero result panel */
.zero-result-title {
  font-size: 20px;
  color: var(--ink);
  margin-bottom: 16px;
  padding: 16px;
  background-color: var(--trust-wash);
  border-radius: 6px;
  border: 1px solid var(--rule);
  text-align: center;
}

/* Column headers */
.col-header {
  font-family: 'Newsreader', serif;
  font-size: 20px;
  font-weight: 700;
  color: var(--ink);
  margin-bottom: 4px;
}
.col-caption {
  font-size: 13px;
  color: var(--ink-soft);
  margin-bottom: 24px;
}

/* Inputs, buttons, and selectbox focus/radius */
.stTextInput input, .stSelectbox > div > div {
  border-radius: 6px !important;
  border-color: var(--rule) !important;
}
.stTextInput input:focus, .stSelectbox > div > div:focus-within {
  border-color: var(--trust) !important;
  box-shadow: 0 0 0 1px var(--trust) !important;
}
</style>
""", unsafe_allow_html=True)

# 2. Header
st.markdown("""
<div class="header-container">
  <h1 class="wordmark">Kurate</h1>
  <div class="header-eyebrow">TRUST-RANKED SEARCH — INTERNAL DEMO</div>
</div>
""", unsafe_allow_html=True)

st.info("All data on this page is synthetic and generated for this demo — no real Kurate users, posts, or systems are involved.")

@st.cache_data
def load_corpus():
    return corpus.get_posts(), corpus.get_people(), corpus.get_follow_graph()

@st.cache_resource
def load_indexer(_posts):
    return retrieval.get_index(_posts)

with st.spinner("Indexing posts…"):
    posts, people, follow_edges = load_corpus()
    indexer = load_indexer(posts)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">SEARCH SETTINGS</div>', unsafe_allow_html=True)
    
    user_choice = st.selectbox(
        "Viewing as",
        [
            (corpus.DEMO_USER_ID, "Demo member (Follows 5 people)"),
            (corpus.COLD_START_USER_ID, "New member (Follows 0 people)")
        ],
        format_func=lambda x: x[1]
    )
    selected_user_id = user_choice[0]
    st.session_state["user_id"] = selected_user_id
    
    trust_weight = st.slider("Trust proximity weight", min_value=0.0, max_value=1.0, value=0.3, step=0.05, key="trust_weight")
    st.caption("0 = pure relevance ranking, 1 = ranking dominated by who you follow. Pushing this to 1 demonstrates the filter-bubble risk where you only see content from your immediate circle.")
    st.divider()
    
    people_dict = {p.id: p for p in people}
    current_person = people_dict[selected_user_id]
    out_edges = [e for e in follow_edges if e.follower_id == selected_user_id]
    is_cold_start = len(out_edges) == 0
    st.caption(f"**FOLLOWS:** {len(out_edges)} PEOPLE\n\n**TOPICS:** {', '.join(current_person.topics).upper()}")
    
    if is_cold_start:
        st.caption("Trust weight reduced automatically — you don't follow anyone yet, so we can't use your network to personalize results.")
        
    st.divider()
    with st.expander("⚠ Simulate failure (for demo purposes)"):
        failure_sim = st.selectbox(
            "Failure mode", 
            ["None", "Semantic search timeout", "Index unavailable", "Partial results only"]
        )

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "hidden_results" not in st.session_state:
    st.session_state["hidden_results"] = set()
if "saved_results" not in st.session_state:
    st.session_state["saved_results"] = set()
if "followed_people" not in st.session_state:
    st.session_state["followed_people"] = set()
if "last_executed_query" not in st.session_state:
    st.session_state["last_executed_query"] = None
if "current_query_id" not in st.session_state:
    st.session_state["current_query_id"] = None
if "last_zero_result_query_id" not in st.session_state:
    st.session_state["last_zero_result_query_id"] = None

def log_app_event(event_name, **kwargs):
    props = {
        "query_id": st.session_state.get("current_query_id", "N/A"),
        "session_id": st.session_state["session_id"],
        "anonymous_user_id": selected_user_id, # stands in for a real anonymized id
        "platform": "streamlit-demo",
        "ranking_version": "v1",
        "experiment_variant": "control",
        "trust_weight": trust_weight,
        "cold_start_state": is_cold_start
    }
    props.update(kwargs)
    analytics.log_event(event_name, **props)
    
query = st.text_input("Search", value="climate risk", key="query")

try:
    if not query or not query.strip():
        st.caption("Enter a search term above.")
    else:
        norm_q = query.strip().lower()
        is_new_search = (norm_q != st.session_state.get("last_executed_query"))
        
        if is_new_search:
            st.session_state["current_query_id"] = str(uuid.uuid4())
            st.session_state["last_executed_query"] = norm_q
            log_app_event("search_submitted", query=query)
        
        col1, col2, col3 = st.columns(3)
        
        t0 = time.time()
        if failure_sim == "Index unavailable":
            raise RuntimeError("Simulated Index Failure")
            
        kw_res = retrieval.keyword_search(query, posts, top_k=5)
        
        if failure_sim == "Semantic search timeout":
            sem_res = []
            hybrid = retrieval.hybrid_search(query, posts, top_k=30, semantic_enabled=False)
            st.warning("Partial results — semantic search unavailable")
        else:
            sem_res = retrieval.semantic_search(query, posts, top_k=5)
            sem_res = [r for r in sem_res if r.get("semantic_score", 0) > retrieval.SEMANTIC_FLOOR]
            hybrid = retrieval.hybrid_search(query, posts, top_k=30)
        
        if failure_sim == "Partial results only":
            hybrid = hybrid[:1]
            
        reranked = retrieval.rerank_with_trust(hybrid, selected_user_id, follow_edges, trust_weight)
        valid_reranked = reranked[:5]
        
        latency = (time.time() - t0) * 1000
        
        if is_new_search:
            if failure_sim == "Semantic search timeout":
                log_app_event("results_received", result_count=len(valid_reranked), latency_ms=latency, index_age_ms=0, degraded_mode="keyword_only", failure_reason="semantic_timeout")
            else:
                log_app_event("results_received", result_count=len(valid_reranked), latency_ms=latency, index_age_ms=0)

        
        def render_result(res, metrics_dict, rank, hops=None, context="Search"):
            post_id = res['post_id']
            if post_id in st.session_state["hidden_results"]:
                return
            if hops is not None:
                if hops in (0, 1):
                    badge_html = '<span class="badge badge-direct">Direct follow</span>'
                elif hops == 2:
                    badge_html = '<span class="badge badge-2hop">2 hops away</span>'
                else:
                    badge_html = '<span class="badge badge-out">Outside your network</span>'
            else:
                badge_html = ''
                
            author = people_dict[res['author_id']]
            raw_text = res['text'] if len(res['text']) <= 140 else res['text'][:139] + "…"
            
            explanation_html = ""
            if hops is not None:
                exp_text = retrieval.explain_result_plain(res, selected_user_id, follow_edges)
                explanation_html = f'<div class="metric-label" style="margin-top: 8px; color: var(--trust);">{html.escape(exp_text)}</div>'
            
            card_html = f"""
            <div class="kurate-card">
              {badge_html}
              <div class="card-author">{html.escape(author.display_name)}</div>
              <div class="card-text">{html.escape(raw_text)}</div>
              {explanation_html}
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            common_props = {
                "result_id": post_id,
                "result_type": "post",
                "rank": rank,
                "author_id": res["author_id"],
                "trust_hops": hops,
                "keyword_rank": res.get("keyword_rank"),
                "semantic_rank": res.get("semantic_rank"),
                "keyword_score": res.get("keyword_score"),
                "semantic_score": res.get("semantic_score")
            }
            
            c1, c2, c3, c4 = st.columns([1,1,1,2])
            if c1.button("Open", key=f"open_{context}_{post_id}"):
                log_app_event("result_opened", **common_props)
                
            is_saved = post_id in st.session_state["saved_results"]
            save_label = "Saved ✓" if is_saved else "Save"
            if c2.button(save_label, key=f"save_{context}_{post_id}"):
                if is_saved:
                    st.session_state["saved_results"].remove(post_id)
                    action = "unsave"
                    state_after = False
                else:
                    st.session_state["saved_results"].add(post_id)
                    action = "save"
                    state_after = True
                log_app_event("result_saved", action=action, state_after=state_after, **common_props)
                st.rerun()
                
            if c3.button("Hide", key=f"hide_{context}_{post_id}"):
                st.session_state["hidden_results"].add(post_id)
                log_app_event("result_hidden", **common_props)
                st.rerun()
            
            with st.expander("Ranking details (internal)"):
                if st.button("Reveal metrics", key=f"breakdown_{context}_{post_id}"):
                    log_app_event("ranking_explanation_opened", **common_props)
                    keys = list(metrics_dict.keys())
                    vals = list(metrics_dict.values())
                    mc = st.columns(len(keys))
                    for idx, c in enumerate(mc):
                        with c:
                            v = vals[idx]
                            v_str = f"{v:.3f}" if isinstance(v, float) else str(v)
                            st.markdown(f'<div class="metric-box"><div class="metric-label">{html.escape(keys[idx])}</div><div class="metric-val">{html.escape(v_str)}</div></div>', unsafe_allow_html=True)

        def render_person_card(p, context):
            dist = p.get("trust_hops")
            if dist in (0, 1):
                badge_html = '<span class="badge badge-direct">Direct follow</span>'
            elif dist == 2:
                badge_html = '<span class="badge badge-2hop">2 hops away</span>'
            else:
                badge_html = '<span class="badge badge-out">Outside your network</span>'
                
            topics_str = " · ".join(p['topics']).upper()
            
            person_card_html = f"""
            <div class="kurate-card">
              {badge_html}
              <div class="card-author">{html.escape(p['display_name'])}</div>
              <div class="card-text">{html.escape(p['bio'])}</div>
              <div class="metric-label">{html.escape(topics_str)}</div>
            </div>
            """
            st.markdown(person_card_html, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns([1,1,2])
            props = {"person_id": p.get("person_id", p.get("id")), "trust_hops": dist}
            btn_key = f"profile_{context}_{props['person_id']}"
            
            if c1.button("View profile", key=f"vp_{btn_key}"):
                log_app_event("person_profile_opened", **props)
                if context == "Recovery":
                    log_app_event("query_recovery_selected", **props)
                    
            is_followed = props["person_id"] in st.session_state["followed_people"]
            follow_label = "Following ✓" if is_followed else "Follow"
            if c2.button(follow_label, key=f"fol_{btn_key}"):
                if is_followed:
                    st.session_state["followed_people"].remove(props["person_id"])
                    action = "unfollow"
                    state_after = False
                else:
                    st.session_state["followed_people"].add(props["person_id"])
                    action = "follow"
                    state_after = True
                log_app_event("person_followed", action=action, state_after=state_after, **props)
                st.rerun()
            c2.caption("(Demo action — doesn't affect ranking in this prototype)")

        with col1:
            st.markdown('<div class="col-header">Keyword match</div><div class="col-caption">Exact term matching</div>', unsafe_allow_html=True)
            if not kw_res:
                st.caption("No results.")
            else:
                for i, res in enumerate(kw_res, 1):
                    render_result(res, {"Keyword": res["keyword_score"]}, i)
                    
        with col2:
            st.markdown('<div class="col-header">Related content</div><div class="col-caption">Finds posts on the same topic, even without matching words</div>', unsafe_allow_html=True)
            if not sem_res:
                st.caption("No results.")
            else:
                for i, res in enumerate(sem_res, 1):
                    render_result(res, {"Semantic": res["semantic_score"]}, i)
                    
        with col3:
            st.markdown('<div class="col-header">For you</div><div class="col-caption">Ranked by relevance and who you trust</div>', unsafe_allow_html=True)
            
            if trust_weight >= 0.7:
                st.markdown('<div class="high-trust-warning">High trust weighting — results are concentrated in your existing network.</div>', unsafe_allow_html=True)
            
            if not valid_reranked:
                if is_new_search or st.session_state["current_query_id"] != st.session_state.get("last_zero_result_query_id"):
                    log_app_event("zero_result_shown")
                    st.session_state["last_zero_result_query_id"] = st.session_state["current_query_id"]
                st.markdown('<div class="zero-result-title">No strong matches — but here\'s who to follow instead</div>', unsafe_allow_html=True)
                recovery_res = retrieval.zero_result_recovery(query, selected_user_id, people, follow_edges, top_k=5)
                if not recovery_res:
                    st.caption("No people found.")
                else:
                    for p in recovery_res:
                        render_person_card(p, "Recovery")
            else:
                for i, res in enumerate(valid_reranked, 1):
                    metrics = {
                        "Keyword": res.get("keyword_score", 0.0),
                        "Semantic": res.get("semantic_score", 0.0),
                        "Trust Score": res.get("trust_score", 0.0),
                        "Final": res.get("final_score", 0.0)
                    }
                    render_result(res, metrics, i, res.get("trust_hops", None))
                    
        # Fired AFTER the result cards actually render to the page.
        # This is the confirmed-exposure event — DISTINCT from results_received.
        # results_received means "the backend returned data", whereas results_rendered means 
        # "the member's screen actually painted it".
        if is_new_search:
            log_app_event("results_rendered", rendered_count=len(valid_reranked))
                    
        # 9. People matching your search
        st.markdown("---")
        st.markdown('<div class="col-header">People matching your search</div>', unsafe_allow_html=True)
        p_res = retrieval.people_search(query, people, top_k=5)
        if not p_res:
            st.caption("No people found.")
        else:
            p_cols = st.columns(len(p_res))
            for c, p in zip(p_cols, p_res):
                with c:
                    p["trust_hops"] = retrieval.trust_distance(selected_user_id, p["person_id"], follow_edges)
                    render_person_card(p, "Search")
                        
except Exception as e:
    st.error("An error occurred while running the search.")
    if DEBUG_MODE:
        with st.expander("Details"):
            st.exception(e)

st.divider()
with st.expander("📊 Event log (internal debug view)", expanded=False):
    if "event_log" in st.session_state and st.session_state["event_log"]:
        st.dataframe(st.session_state["event_log"])
    else:
        st.caption("No events yet.")
