from streamlit.testing.v1 import AppTest

def test_analytics_not_duplicated_on_unrelated_rerun():
    at = AppTest.from_file("../app.py").run()
    
    # By default, a search is submitted for "climate risk"
    # Find the event log in session state
    events = at.session_state["event_log"]
    search_submitted_count = sum(1 for e in events if e["event_name"] == "search_submitted")
    assert search_submitted_count == 1, f"Expected 1 search_submitted, got {search_submitted_count}"
    
    # Simulate a rerun with the same query (e.g. by clicking Save)
    # Just trigger another run
    at.run()
    
    # The event log should still only have 1 search_submitted
    events = at.session_state["event_log"]
    search_submitted_count = sum(1 for e in events if e["event_name"] == "search_submitted")
    assert search_submitted_count == 1, f"Expected 1 search_submitted after rerun, got {search_submitted_count}"

def test_context_change_triggers_new_search_attempt():
    at = AppTest.from_file("../app.py").run()
    events = at.session_state["event_log"]
    initial_submits = [e for e in events if e["event_name"] == "search_submitted"]
    assert len(initial_submits) == 1
    q_id_1 = initial_submits[0]["query_id"]
    
    at.slider(key="trust_weight").set_value(0.9)
    at.run()
    
    events = at.session_state["event_log"]
    submits = [e for e in events if e["event_name"] == "search_submitted"]
    assert len(submits) == 2
    assert submits[0]["trust_weight"] != submits[1]["trust_weight"]
    assert submits[0]["trust_weight"] == 0.9

def test_rendered_count_excludes_hidden_results():
    at = AppTest.from_file("../app.py").run()
    initial_count = [e for e in at.session_state["event_log"] if e["event_name"] == "results_rendered"][0]["rendered_count"]
    
    # Hide the first result in the For You column
    hide_btn = [b for b in at.button if b.key and b.key.startswith("hide_ForYou_")][0]
    hide_btn.click().run()
    
    # Slightly adjust slider to trigger new context but keep same top results
    at.slider(key="trust_weight").set_value(0.31)
    at.run()
    
    renders = [e for e in at.session_state["event_log"] if e["event_name"] == "results_rendered"]
    assert renders[0]["rendered_count"] == initial_count - 1

def test_search_degraded_fires_for_all_three_failure_modes():
    at = AppTest.from_file("../app.py").run()
    
    at.selectbox(key="failure_sim").select("Semantic search timeout").run()
    degrades = [e for e in at.session_state["event_log"] if e["event_name"] == "search_degraded"]
    assert len(degrades) == 1
    assert degrades[0]["failure_reason"] == "semantic_timeout"
    assert degrades[0]["degraded_mode"] == "keyword_only"
    
    at.selectbox(key="failure_sim").select("Partial results only").run()
    degrades = [e for e in at.session_state["event_log"] if e["event_name"] == "search_degraded"]
    assert len(degrades) == 2
    assert degrades[0]["failure_reason"] == "partial_results"
    assert degrades[0]["degraded_mode"] == "truncated"
    
    at.selectbox(key="failure_sim").select("Index unavailable").run()
    degrades = [e for e in at.session_state["event_log"] if e["event_name"] == "search_degraded"]
    assert len(degrades) == 3
    assert degrades[0]["failure_reason"] == "index_unavailable"
    assert degrades[0]["degraded_mode"] == "error"

