from streamlit.testing.v1 import AppTest


def test_analytics_not_duplicated_on_unrelated_rerun():
    at = AppTest.from_file("../app.py").run()

    # By default, a search is submitted for "climate risk"
    events = at.session_state["event_log"]
    search_submitted_count = sum(1 for e in events if e["event_name"] == "search_submitted")
    assert search_submitted_count == 1, f"Expected 1 search_submitted, got {search_submitted_count}"

    # Simulate a rerun with the same query (e.g. by clicking Save)
    at.run()

    events = at.session_state["event_log"]
    search_submitted_count = sum(1 for e in events if e["event_name"] == "search_submitted")
    assert search_submitted_count == 1, f"Expected 1 search_submitted after rerun, got {search_submitted_count}"


def test_trust_slider_does_not_duplicate_search_submitted():
    """A trust-weight change is a new search EXECUTION, but not a new search INTENT —
    the member didn't type a new query, so search_submitted must not fire again."""
    at = AppTest.from_file("../app.py").run()
    events = at.session_state["event_log"]
    assert sum(1 for e in events if e["event_name"] == "search_submitted") == 1

    at.slider(key="trust_weight").set_value(0.9).run()

    events = at.session_state["event_log"]
    assert sum(1 for e in events if e["event_name"] == "search_submitted") == 1, \
        "trust-slider change should not log a second search_submitted"

    # It should still count as a new execution attempt though.
    assert sum(1 for e in events if e["event_name"] == "results_received") == 2


def test_query_change_creates_new_query_id_and_search_submitted():
    at = AppTest.from_file("../app.py").run()
    query_id_before = at.session_state["current_query_id"]

    at.text_input(key="query").set_value("insurance industry").run()

    events = at.session_state["event_log"]
    assert sum(1 for e in events if e["event_name"] == "search_submitted") == 2
    assert at.session_state["current_query_id"] != query_id_before


def test_retry_creates_new_execution_id():
    at = AppTest.from_file("../app.py").run()
    at.selectbox(key="failure_sim").select("Index unavailable").run()
    execution_id_before = at.session_state["current_execution_id"]

    retry_btn = [b for b in at.button if b.key == "retry_index_failure"][0]
    retry_btn.click().run()

    execution_id_after = at.session_state["current_execution_id"]
    assert execution_id_after != execution_id_before, "Retry should mint a genuinely new execution attempt"


def test_retry_preserves_original_query_id():
    at = AppTest.from_file("../app.py").run()
    at.selectbox(key="failure_sim").select("Index unavailable").run()
    query_id_before = at.session_state["current_query_id"]

    retry_btn = [b for b in at.button if b.key == "retry_index_failure"][0]
    retry_btn.click().run()

    assert at.session_state["current_query_id"] == query_id_before, \
        "Retry re-attempts the same search intent — the query_id must not change"

    events = at.session_state["event_log"]
    retried = [e for e in events if e["event_name"] == "search_retried"]
    assert len(retried) == 1
    assert retried[0]["query_id"] == query_id_before


def test_hiding_one_result_backfills_when_more_candidates_exist():
    at = AppTest.from_file("../app.py").run()
    initial_renders = [e for e in at.session_state["event_log"] if e["event_name"] == "results_rendered"]
    initial_count = initial_renders[0]["rendered_count"]
    assert initial_count > 0, "Expected the default 'climate risk' search to render results"

    hide_btn = [b for b in at.button if b.key and b.key.startswith("hide_ForYou_")][0]
    hide_btn.click().run()

    # Trigger a fresh execution attempt so results_rendered gets logged again.
    at.slider(key="trust_weight").set_value(0.31).run()

    renders = [e for e in at.session_state["event_log"] if e["event_name"] == "results_rendered"]
    assert renders[0]["rendered_count"] == initial_count, \
        "Hiding a result should backfill the next-best candidate, not just shrink the list"


def test_index_failure_emits_search_failed():
    at = AppTest.from_file("../app.py").run()
    at.selectbox(key="failure_sim").select("Index unavailable").run()

    events = at.session_state["event_log"]
    failed = [e for e in events if e["event_name"] == "search_failed"]
    assert len(failed) == 1
    assert failed[0]["failure_reason"] == "index_unavailable"

    # Index-down is a hard failure, not a degradation — it must not also fire search_degraded.
    degraded = [e for e in events if e["event_name"] == "search_degraded"]
    assert len(degraded) == 0


def test_semantic_timeout_emits_search_degraded():
    at = AppTest.from_file("../app.py").run()
    at.selectbox(key="failure_sim").select("Semantic search timeout").run()

    events = at.session_state["event_log"]
    degraded = [e for e in events if e["event_name"] == "search_degraded" and e.get("failure_reason") == "semantic_timeout"]
    assert len(degraded) == 1
    assert degraded[0]["degraded_mode"] == "keyword_only"


def test_result_actions_reference_correct_query_and_execution_ids():
    at = AppTest.from_file("../app.py").run()
    query_id = at.session_state["current_query_id"]
    execution_id = at.session_state["current_execution_id"]

    open_btn = [b for b in at.button if b.key and b.key.startswith("open_ForYou_")][0]
    open_btn.click().run()

    events = at.session_state["event_log"]
    opened = [e for e in events if e["event_name"] == "result_opened"][0]
    assert opened["query_id"] == query_id
    assert opened["search_execution_id"] == execution_id
