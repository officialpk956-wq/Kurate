import pytest
from corpus import get_posts, get_people, get_follow_graph, DEMO_USER_ID, COLD_START_USER_ID
from retrieval import keyword_search, semantic_search, hybrid_search, rerank_with_trust, trust_distance, people_search, zero_result_recovery

@pytest.fixture(scope="module")
def data():
    return get_posts(), get_people(), get_follow_graph()

def test_semantic_vs_keyword_gap(data):
    posts, _, _ = data
    query = "climate risk"
    kw = keyword_search(query, posts, top_k=5)
    sem = semantic_search(query, posts, top_k=5)
    
    vocab_words = ["warming", "rising temperatures", "greenhouse gases", "melting ice caps", "extreme weather"]
    kw_ids = {r["post_id"] for r in kw}
    
    found_in_sem_not_kw = False
    for res in sem:
        if res["post_id"] not in kw_ids:
            text_lower = res["text"].lower()
            if any(v in text_lower for v in vocab_words):
                found_in_sem_not_kw = True
                break
    assert found_in_sem_not_kw, "No global warming post found in semantic search top 5 that was missing from keyword top 5"

def test_trust_slider_shifts_authors(data):
    posts, _, follow_edges = data
    query = "climate risk"
    hybrid = hybrid_search(query, posts, top_k=10)
    
    rerank_0 = rerank_with_trust(hybrid, DEMO_USER_ID, follow_edges, trust_weight=0.0)
    rerank_1 = rerank_with_trust(hybrid, DEMO_USER_ID, follow_edges, trust_weight=1.0)
    
    top3_0 = [r["post_id"] for r in rerank_0[:3]]
    top3_1 = [r["post_id"] for r in rerank_1[:3]]
    assert top3_0 != top3_1, "Top 3 authors did not change with trust weight"
    
    for res in rerank_1[:3]:
        assert res["trust_hops"] in (0, 1, 2), f"Top 3 result at weight 1.0 has invalid trust_hops: {res['trust_hops']}"

def test_cold_start_user_graceful(data):
    posts, _, follow_edges = data
    hybrid = hybrid_search("climate risk", posts, top_k=10)
    rerank_cold = rerank_with_trust(hybrid, COLD_START_USER_ID, follow_edges, trust_weight=1.0)
    assert isinstance(rerank_cold, list)
    assert len(rerank_cold) > 0, "Cold start rerank failed or returned empty"

def test_zero_result_recovery(data):
    _, people, follow_edges = data
    zero = zero_result_recovery("underwater basket weaving", DEMO_USER_ID, people, follow_edges)
    assert isinstance(zero, list), "Zero result recovery failed to return a list"

def test_empty_string_queries(data):
    posts, people, _ = data
    empty_query = "   "
    kw = keyword_search(empty_query, posts)
    assert isinstance(kw, list)
    
    sem = semantic_search(empty_query, posts)
    assert isinstance(sem, list)
    
    hyb = hybrid_search(empty_query, posts)
    assert isinstance(hyb, list)
    
    ps = people_search(empty_query, people)
    assert isinstance(ps, list)
    assert kw == []
    assert sem == []
    assert hyb == []
    assert ps == []


def test_oov_semantic_query_returns_no_arbitrary_results(data):
    posts, _, _ = data
    assert semantic_search("zxqwyxyz", posts) == []


def test_non_positive_top_k_returns_empty(data):
    posts, people, _ = data
    assert keyword_search("climate", posts, top_k=0) == []
    assert semantic_search("climate", posts, top_k=-1) == []
    assert hybrid_search("climate", posts, top_k=0) == []
    assert people_search("climate", people, top_k=-1) == []


def test_post_index_invalidates_after_in_place_content_change(data):
    posts, _, _ = data
    original = posts[0].text
    try:
        posts[0].text = "uniquecacheinvalidationsentinel"
        matches = keyword_search("uniquecacheinvalidationsentinel", posts)
        assert matches and matches[0]["post_id"] == posts[0].id
    finally:
        posts[0].text = original
        # Force restoration of the cached index for subsequent tests.
        keyword_search("climate", posts)


def test_trust_weight_is_clamped(data):
    posts, _, follow_edges = data
    hybrid = hybrid_search("climate risk", posts)
    above = rerank_with_trust(hybrid, DEMO_USER_ID, follow_edges, trust_weight=10)
    one = rerank_with_trust(hybrid, DEMO_USER_ID, follow_edges, trust_weight=1)
    below = rerank_with_trust(hybrid, DEMO_USER_ID, follow_edges, trust_weight=-10)
    zero = rerank_with_trust(hybrid, DEMO_USER_ID, follow_edges, trust_weight=0)
    assert [r["post_id"] for r in above] == [r["post_id"] for r in one]
    assert [r["post_id"] for r in below] == [r["post_id"] for r in zero]

def test_trust_distance_self(data):
    _, _, follow_edges = data
    dist = trust_distance(DEMO_USER_ID, DEMO_USER_ID, follow_edges)
    assert dist == 0, "Trust distance to self should be 0"

def test_rerank_zero_weight_preserves_order(data):
    posts, _, follow_edges = data
    hybrid = hybrid_search("climate risk", posts, top_k=30)
    reranked = rerank_with_trust(hybrid, DEMO_USER_ID, follow_edges, trust_weight=0.0)
    
    hybrid_filtered_ids = []
    author_counts = {}
    for r in hybrid:
        a_id = r["author_id"]
        c = author_counts.get(a_id, 0)
        if c < 2:
            author_counts[a_id] = c + 1
            hybrid_filtered_ids.append(r["post_id"])
            
    reranked_ids = [r["post_id"] for r in reranked]
    assert hybrid_filtered_ids == reranked_ids, "Hybrid order was not perfectly preserved when trust_weight=0.0"

def test_people_search_no_match(data):
    _, people, _ = data
    res = people_search("zxqwyxyz", people)
    assert isinstance(res, list)
    assert len(res) == 0, "Expected empty list for non-matching people search"

def test_relevance_floor_excludes_zero_score_results(data):
    posts, _, _ = data
    hyb = hybrid_search("zxqwyxyz", posts)
    assert isinstance(hyb, list)
    assert len(hyb) == 0, "Expected empty list for nonsense query because no items should survive the relevance floor"

def test_trust_weight_default_split_is_proportionate(data):
    posts, _, follow_edges = data
    hyb = hybrid_search("climate risk", posts)
    reranked = rerank_with_trust(hyb, DEMO_USER_ID, follow_edges, trust_weight=0.3)
    
    top_res = reranked[0]
    
    # Check normalized relevance and trust terms are comparable
    normalized_relevance = top_res["normalized_relevance"]
    trust_score = top_res["trust_score"]
    
    relevance_term = 0.7 * normalized_relevance
    trust_term = 0.3 * trust_score
    
    # Assert neither term is more than 3x the other
    assert relevance_term <= 3 * trust_term, f"Relevance term dominates too much: rel={relevance_term:.4f}, trust={trust_term:.4f}"
    assert trust_term <= 3 * relevance_term, f"Trust term dominates too much: rel={relevance_term:.4f}, trust={trust_term:.4f}"

def test_diversity_cap_limits_results_per_author(data):
    posts, _, follow_edges = data
    # "climate" is a common keyword that will yield many posts from the same author
    hyb = hybrid_search("climate", posts, top_k=20)
    reranked = rerank_with_trust(hyb, DEMO_USER_ID, follow_edges, trust_weight=0.0)
    
    author_counts = {}
    for res in reranked:
        author_counts[res["author_id"]] = author_counts.get(res["author_id"], 0) + 1
        
    for auth, count in author_counts.items():
        assert count <= 2, f"Author {auth} has {count} results, which violates the diversity cap of 2"

def test_zero_result_recovery_triggers_at_high_trust_weight(data):
    posts, people, follow_edges = data
    hyb = hybrid_search("zxqwyxyz", posts)
    reranked = rerank_with_trust(hyb, DEMO_USER_ID, follow_edges, trust_weight=1.0)
    
    assert len(reranked) == 0, "Expected empty candidate list for nonsense query, regardless of trust weight"

def test_explain_result_plain():
    from retrieval import explain_result_plain
    
    res1 = {"trust_hops": 0, "semantic_score": 0.8, "keyword_score": 0.2}
    assert explain_result_plain(res1, "u1", []) == "Related to your search topic, shared by someone you follow."
    
    res2 = {"trust_hops": 1, "semantic_score": 0.2, "keyword_score": 0.8}
    assert explain_result_plain(res2, "u1", []) == "A strong keyword match, shared by someone you follow."
    
    res3 = {"trust_hops": 2, "semantic_score": 0.8, "keyword_score": 0.2}
    assert explain_result_plain(res3, "u1", []) == "Related to your search topic, shared by someone in your network."
    
    res4 = {"trust_hops": None, "semantic_score": 0.2, "keyword_score": 1.6}
    assert explain_result_plain(res4, "u1", []) == "A strong keyword match for your search."

def test_cold_start_auto_adjust(data):
    posts, _, follow_edges = data
    from corpus import COLD_START_USER_ID
    hyb = hybrid_search("climate", posts, top_k=20)
    
    reranked = rerank_with_trust(hyb, COLD_START_USER_ID, follow_edges, trust_weight=1.0)
    if reranked:
        top_res = reranked[0]
        expected = 0.85 * top_res["normalized_relevance"] + 0.15 * top_res["trust_score"]
        assert abs(top_res["final_score"] - expected) < 1e-5, "Cold start trust weight cap of 0.15 was not applied"

def test_log_event_no_crash():
    import analytics
    analytics.log_event("test_event", test_prop=123)
