import random
import datetime
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Person:
    id: str
    display_name: str
    bio: str
    topics: List[str]

@dataclass
class Post:
    id: str
    author_id: str
    text: str
    topic: str
    timestamp: datetime.datetime

@dataclass
class FollowEdge:
    follower_id: str
    followee_id: str

TOPICS = ["climate", "technology", "finance", "health", "politics", "sports", "culture", "science"]
DEMO_USER_ID = "demo_user"
COLD_START_USER_ID = "cold_start_user"

# Seeds to ensure determinism
random.seed(42)
np.random.seed(42)

_people: List[Person] = []
_posts: List[Post] = []
_edges: List[FollowEdge] = []

def _generate_data():
    global _people, _posts, _edges
    if _people:
        return
        
    people = []
    people.append(Person(
        id=DEMO_USER_ID, 
        display_name="Demo User", 
        bio="I am the demo user, interested in specific topics.", 
        topics=["climate", "technology", "science"]
    ))
    people.append(Person(
        id=COLD_START_USER_ID, 
        display_name="Cold Start", 
        bio="I am new here and looking around.", 
        topics=["health", "culture"]
    ))
    
    topic_weights = [0.2, 0.2, 0.1, 0.15, 0.15, 0.05, 0.1, 0.05]
    
    for i in range(38):
        num_topics = random.randint(2, 4)
        ts = np.random.choice(TOPICS, size=num_topics, replace=False, p=topic_weights).tolist()
        people.append(Person(
            id=f"user_{i}", 
            display_name=f"User {i}", 
            bio=f"This is the bio for user {i}, who likes {ts[0]}.", 
            topics=ts
        ))
        
    p_ids = [p.id for p in people if p.id not in (DEMO_USER_ID, COLD_START_USER_ID)]
    climate_users = [p.id for p in people if "climate" in p.topics and p.id in p_ids]
    
    # Ensure we have at least 3 climate users to follow
    if len(climate_users) < 3:
        for i in range(3):
            if "climate" not in people[i + 2].topics:
                people[i + 2].topics[0] = "climate"
                climate_users.append(people[i + 2].id)
    climate_users = sorted(list(set(climate_users)))
    
    _people = people
    
    edges = []
    
    # Demo user edges
    demo_follows = random.sample(climate_users, min(3, len(climate_users)))
    rem_candidates = [pid for pid in p_ids if pid not in demo_follows]
    demo_follows.extend(random.sample(rem_candidates, 5 - len(demo_follows)))
    
    for f_id in demo_follows:
        edges.append(FollowEdge(follower_id=DEMO_USER_ID, followee_id=f_id))
        
    # Other users edges
    people_dict = {p.id: p for p in _people}
    for p in _people:
        if p.id in (DEMO_USER_ID, COLD_START_USER_ID):
            continue
            
        num_f = random.randint(3, 8)
        cands = [x for x in p_ids + [DEMO_USER_ID, COLD_START_USER_ID] if x != p.id]
        
        my_t = set(p.topics)
        weights = []
        for c in cands:
            c_t = set(people_dict[c].topics)
            shared = len(my_t.intersection(c_t))
            weights.append(3.0 if shared > 0 else 1.0)
            
        weights = np.array(weights) / sum(weights)
        chosen = np.random.choice(cands, size=num_f, replace=False, p=weights).tolist()
        for c in chosen:
            edges.append(FollowEdge(follower_id=p.id, followee_id=c))
            
    _edges = edges
    
    posts = []
    # Using fixed absolute now for generation, but random days back
    # The requirement asks for "within the last 30 days". 
    # To avoid failures if the script runs for a long time, we'll fix 'now' at generation time.
    now = datetime.datetime.now(datetime.timezone.utc)
    
    vocab = [
        "The climate faces a huge risk from global warming and greenhouse gases.",
        "We must evaluate the climate and the huge risk of rising temperatures.",
        "Melting ice caps are a huge climate issue that brings great risk.",
        "Extreme weather is a huge risk to our global climate today.",
        "Global warming and greenhouse gases need our attention.",
        "Rising temperatures and extreme weather threaten many species."
    ]
    
    demo_followees = [e.followee_id for e in _edges if e.follower_id == DEMO_USER_ID]
    climate_authors = [f for f in demo_followees if "climate" in people_dict[f].topics]
    
    in_network = climate_authors
    demo_1hop = set(demo_followees)
    demo_2hop = set()
    for e in _edges:
        if e.follower_id in demo_1hop:
            demo_2hop.add(e.followee_id)
    demo_network = demo_1hop.union(demo_2hop)
    
    out_network = [p.id for p in _people if p.id not in demo_network and p.id not in (DEMO_USER_ID, COLD_START_USER_ID)]
    for u in out_network[:3]:
        if "climate" not in people_dict[u].topics:
            people_dict[u].topics[-1] = "climate"
    
    assigned_authors = (in_network * 3)[:3] + (out_network * 3)[:3]
    
    for i in range(6):
        posts.append(Post(
            id=f"post_climate_req_{i}",
            author_id=assigned_authors[i],
            text=vocab[i],
            topic="climate",
            timestamp=now - datetime.timedelta(days=random.uniform(0, 29))
        ))
        
    bridge_texts = [
        "Climate change involves global warming and melting ice caps.",
        "There is a risk from greenhouse gases and extreme weather."
    ]
    bridge_idx = 0
        
    # Mirrors real engagement distributions where a small percentage of users create the vast majority of content.
    pareto_draws = np.random.pareto(a=1.5, size=len(_people))
    post_counts = np.round(pareto_draws / np.sum(pareto_draws) * 144).astype(int)
    
    post_idx = 6
    for p, count in zip(_people, post_counts):
        if count <= 0:
            continue
        for _ in range(count):
            topic = random.choice(p.topics)
            
            if topic == "climate" and bridge_idx < 2:
                text = bridge_texts[bridge_idx]
                bridge_idx += 1
            else:
                text_options = [
                    f"Sharing some thoughts on the subject.",
                    f"Did you read the latest news about this?",
                    f"I really find this fascinating these days.",
                    f"A great discussion on the topic today."
                ]
                text = random.choice(text_options)
                
            posts.append(Post(
                id=f"post_gen_{post_idx}",
                author_id=p.id,
                text=text,
                topic=topic,
                timestamp=now - datetime.timedelta(days=random.uniform(0, 29))
            ))
            post_idx += 1
            
    _posts = posts

def get_people() -> List[Person]:
    _generate_data()
    return _people

def get_posts() -> List[Post]:
    _generate_data()
    return _posts

def get_follow_graph() -> List[FollowEdge]:
    _generate_data()
    return _edges

def validate_corpus():
    people = get_people()
    posts = get_posts()
    edges = get_follow_graph()
    
    # 1. Every Post.author_id exists in Person ids.
    person_ids = {p.id for p in people}
    for post in posts:
        assert post.author_id in person_ids, f"Post {post.id} author {post.author_id} not in person_ids"
        
    # 2. Every FollowEdge references two valid, distinct Person ids (no self-follows).
    for edge in edges:
        assert edge.follower_id in person_ids, f"Edge follower {edge.follower_id} not found"
        assert edge.followee_id in person_ids, f"Edge followee {edge.followee_id} not found"
        assert edge.follower_id != edge.followee_id, f"Self-follow detected for {edge.follower_id}"
        
    # 3. No duplicate Person ids, no duplicate Post ids.
    assert len(people) == len(person_ids), "Duplicate Person IDs detected"
    post_ids = {p.id for p in posts}
    assert len(posts) == len(post_ids), "Duplicate Post IDs detected"
    
    # 4. Every Person has 2-4 topics, all from the fixed TOPIC list.
    for p in people:
        assert 2 <= len(p.topics) <= 4, f"Person {p.id} has {len(p.topics)} topics, expected 2-4"
        for t in p.topics:
            assert t in TOPICS, f"Person {p.id} has invalid topic {t}"
            
    # 5. Every Post.topic is in its author's topics list.
    people_dict = {p.id: p for p in people}
    for post in posts:
        assert post.topic in people_dict[post.author_id].topics, f"Post {post.id} topic {post.topic} not in author's topics"
        
    # 6. COLD_START_USER_ID has zero outgoing edges in the follow graph.
    cold_start_outgoing = [e for e in edges if e.follower_id == COLD_START_USER_ID]
    assert len(cold_start_outgoing) == 0, f"Cold start user has {len(cold_start_outgoing)} outgoing edges"
    
    # 7. DEMO_USER_ID has exactly 5 outgoing edges.
    demo_outgoing = [e for e in edges if e.follower_id == DEMO_USER_ID]
    assert len(demo_outgoing) == 5, f"Demo user has {len(demo_outgoing)} outgoing edges, expected 5"
    
    # 8. At least 6 posts match global warming vocab, none contain "climate risk".
    climate_vocab_matches = 0
    vocab_words = ["warming", "rising temperatures", "greenhouse gases", "melting ice caps", "extreme weather"]
    for post in posts:
        text_lower = post.text.lower()
        assert "climate risk" not in text_lower, f"Post {post.id} contains forbidden phrase 'climate risk'"
        if any(v in text_lower for v in vocab_words):
            climate_vocab_matches += 1
            
    assert climate_vocab_matches >= 6, f"Found {climate_vocab_matches} posts with climate vocab, expected >= 6"
    
    # Ensure at least 3 authors of these posts are within 2 hops of DEMO_USER_ID
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
    
    # 9. All timestamps are within the last 30 days and timezone-aware.
    now = datetime.datetime.now(datetime.timezone.utc)
    for post in posts:
        assert post.timestamp.tzinfo is not None, f"Post {post.id} timestamp is naive"
        delta = now - post.timestamp
        assert 0 <= delta.total_seconds() <= 30 * 24 * 3600, f"Post {post.id} timestamp {post.timestamp} out of bounds"
        
    # 10. No empty strings in any text field (display_name, bio, text).
    for p in people:
        assert p.display_name.strip() != "", f"Person {p.id} has empty display_name"
        assert p.bio.strip() != "", f"Person {p.id} has empty bio"
    for post in posts:
        assert post.text.strip() != "", f"Post {post.id} has empty text"
        
    print("VALIDATION PASSED")
    print(f"People: {len(people)}")
    print(f"Posts: {len(posts)}")
    print(f"Follow edges: {len(edges)}")

if __name__ == "__main__":
    validate_corpus()
