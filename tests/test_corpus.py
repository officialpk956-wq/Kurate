import pytest
import datetime
from corpus import get_people, get_posts, get_follow_graph, DEMO_USER_ID, COLD_START_USER_ID, TOPICS

@pytest.fixture(scope="module")
def corpus_data():
    return get_people(), get_posts(), get_follow_graph()

def test_post_author_validity(corpus_data):
    people, posts, _ = corpus_data
    person_ids = {p.id for p in people}
    for post in posts:
        assert post.author_id in person_ids, f"Post {post.id} author {post.author_id} not in person_ids"

def test_follow_edge_validity(corpus_data):
    people, _, edges = corpus_data
    person_ids = {p.id for p in people}
    for edge in edges:
        assert edge.follower_id in person_ids, f"Edge follower {edge.follower_id} not found"
        assert edge.followee_id in person_ids, f"Edge followee {edge.followee_id} not found"
        assert edge.follower_id != edge.followee_id, f"Self-follow detected for {edge.follower_id}"

def test_no_duplicate_ids(corpus_data):
    people, posts, _ = corpus_data
    person_ids = {p.id for p in people}
    assert len(people) == len(person_ids), "Duplicate Person IDs detected"
    post_ids = {p.id for p in posts}
    assert len(posts) == len(post_ids), "Duplicate Post IDs detected"

def test_person_topics_validity(corpus_data):
    people, _, _ = corpus_data
    for p in people:
        assert 2 <= len(p.topics) <= 4, f"Person {p.id} has {len(p.topics)} topics, expected 2-4"
        for t in p.topics:
            assert t in TOPICS, f"Person {p.id} has invalid topic {t}"

def test_post_topics_validity(corpus_data):
    people, posts, _ = corpus_data
    people_dict = {p.id: p for p in people}
    for post in posts:
        assert post.topic in people_dict[post.author_id].topics, f"Post {post.id} topic {post.topic} not in author's topics"

def test_cold_start_user_isolated(corpus_data):
    _, _, edges = corpus_data
    cold_start_outgoing = [e for e in edges if e.follower_id == COLD_START_USER_ID]
    assert len(cold_start_outgoing) == 0, f"Cold start user has {len(cold_start_outgoing)} outgoing edges"

def test_demo_user_edges(corpus_data):
    _, _, edges = corpus_data
    demo_outgoing = [e for e in edges if e.follower_id == DEMO_USER_ID]
    assert len(demo_outgoing) == 5, f"Demo user has {len(demo_outgoing)} outgoing edges, expected 5"

def test_climate_vocabulary_constraints(corpus_data):
    _, posts, edges = corpus_data
    climate_vocab_matches = 0
    vocab_words = ["warming", "rising temperatures", "greenhouse gases", "melting ice caps", "extreme weather"]
    for post in posts:
        text_lower = post.text.lower()
        assert "climate risk" not in text_lower, f"Post {post.id} contains forbidden phrase 'climate risk'"
        if any(v in text_lower for v in vocab_words):
            climate_vocab_matches += 1
    assert climate_vocab_matches >= 6, f"Found {climate_vocab_matches} posts with climate vocab, expected >= 6"
    
    demo_1hop = {e.followee_id for e in edges if e.follower_id == DEMO_USER_ID}
    demo_2hop = set()
    for e in edges:
        if e.follower_id in demo_1hop:
            demo_2hop.add(e.followee_id)
    demo_network = demo_1hop.union(demo_2hop)
    
    climate_authors_in_network = set()
    for post in posts:
        text_lower = post.text.lower()
        if any(v in text_lower for v in vocab_words) and post.author_id in demo_network:
            climate_authors_in_network.add(post.author_id)
    assert len(climate_authors_in_network) >= 3, f"Only found {len(climate_authors_in_network)} climate authors in 2-hop network, expected >= 3"

def test_timestamps_validity(corpus_data):
    _, posts, _ = corpus_data
    now = datetime.datetime.now(datetime.timezone.utc)
    for post in posts:
        assert post.timestamp.tzinfo is not None, f"Post {post.id} timestamp is naive"
        delta = now - post.timestamp
        assert 0 <= delta.total_seconds() <= 31 * 24 * 3600, f"Post {post.id} timestamp {post.timestamp} out of bounds"

def test_no_empty_strings(corpus_data):
    people, posts, _ = corpus_data
    for p in people:
        assert p.display_name.strip() != "", f"Person {p.id} has empty display_name"
        assert p.bio.strip() != "", f"Person {p.id} has empty bio"
    for post in posts:
        assert post.text.strip() != "", f"Post {post.id} has empty text"
