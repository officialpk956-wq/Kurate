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
