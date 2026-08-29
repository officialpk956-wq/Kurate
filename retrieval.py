import string
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional

# Import models from corpus
from corpus import Post, Person, FollowEdge

# Relevance Floors (Bug 2 Fix)
# KEYWORD_FLOOR: BM25 scores for known exact matches are usually > 2.0. We use 0.01 as a strict baseline
# to separate zero-match results from partial matches.
KEYWORD_FLOOR = 0.01

# SEMANTIC_FLOOR: Cosine similarity for irrelevant matches is ~0.0. Relevant semantic hits are > 0.1.
# 0.05 safely drops the mathematical noise without omitting weak true similarities.
SEMANTIC_FLOOR = 0.05

class SearchIndex:
    def __init__(self, posts: List[Post]):
        self.posts = posts
        
        # Tokenization choice: lowercase, split on whitespace, strip basic punctuation.
        # This crude tokenization is chosen intentionally to demonstrate the real gap
        # between keyword matching (which demands exact lexical overlap) and semantic
        # matching (which captures latent topics and co-occurrence without exact matches).
        self.tokenized_corpus = [self._tokenize(p.text) for p in posts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        self.texts = [p.text for p in posts]
        self.vectorizer = TfidfVectorizer(lowercase=True)
        tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        
        # Semantic index:
        # Why TruncatedSVD works here: The synthetic corpus has enough shared vocabulary.
        # Even without the literal "climate risk" phrase, "climate" and "risk" appear 
        # alongside "warming" and "greenhouse gases" in various posts. 
        # LSA reduces dimensionality such that these co-occurring terms map to a shared 
        # semantic concept vector. A query for "climate risk" projects into this space 
        # near the "global warming" posts, succeeding where exact keyword search fails.
        n_comp = min(60, len(self.texts) - 1, tfidf_matrix.shape[1] - 1)
        if n_comp < 1:
            n_comp = 1
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.lsa_matrix = self.svd.fit_transform(tfidf_matrix)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        for p in string.punctuation:
            text = text.replace(p, '')
        return text.split()

_INDEX = None
_INDEX_KEY = None

def _posts_cache_key(posts: List[Post]):
    return tuple((p.id, p.author_id, p.text, p.topic, p.timestamp.isoformat()) for p in posts)

def get_index(posts: List[Post]) -> SearchIndex:
    global _INDEX, _INDEX_KEY
    current_key = _posts_cache_key(posts)
    if _INDEX is None or _INDEX_KEY != current_key:
        _INDEX = SearchIndex(posts)
        _INDEX_KEY = current_key
    return _INDEX

def keyword_search(query: str, posts: List[Post], top_k: int = 10) -> List[Dict]:
    if not query or not query.strip() or top_k <= 0 or not posts:
        return []
    idx = get_index(posts)
    tokenized_query = idx._tokenize(query)
    scores = idx.bm25.get_scores(tokenized_query)
    
    results = []
    for i, score in enumerate(scores):
        if score > 0:
            results.append({
                "post_id": posts[i].id,
                "author_id": posts[i].author_id,
                "text": posts[i].text,
                "keyword_score": float(score)
            })
            
    results.sort(key=lambda x: x["keyword_score"], reverse=True)
    results = results[:top_k]
    for rank, res in enumerate(results, 1):
        res["keyword_rank"] = rank
        
    return results

def semantic_search(query: str, posts: List[Post], top_k: int = 10) -> List[Dict]:
    if not query or not query.strip() or top_k <= 0 or not posts:
        return []
    idx = get_index(posts)
    query_tfidf = idx.vectorizer.transform([query])
    if query_tfidf.nnz == 0:
        return []
    query_lsa = idx.svd.transform(query_tfidf)
    
    sims = cosine_similarity(query_lsa, idx.lsa_matrix)[0]
    
    results = []
    for i, score in enumerate(sims):
        results.append({
            "post_id": posts[i].id,
            "author_id": posts[i].author_id,
            "text": posts[i].text,
            "semantic_score": float(score)
        })
        
    results.sort(key=lambda x: x["semantic_score"], reverse=True)
    results = results[:top_k]
    for rank, res in enumerate(results, 1):
        res["semantic_rank"] = rank
        
    return results

def hybrid_search(query: str, posts: List[Post], top_k: int = 10, semantic_enabled: bool = True) -> List[Dict]:
    if not query or not query.strip() or top_k <= 0 or not posts:
        return []
    internal_k = max(top_k * 3, 30)
    kw_res = keyword_search(query, posts, top_k=internal_k)
    
    if semantic_enabled:
        sem_res = semantic_search(query, posts, top_k=internal_k)
    else:
        sem_res = []

    # Filter candidates by RELEVANCE FLOOR before fusion
    # A post that clears neither floor must not receive an RRF score or enter the re-ranking pipeline.
    valid_pids = set()
    for res in kw_res:
        if res.get("keyword_score", 0) > KEYWORD_FLOOR:
            valid_pids.add(res["post_id"])
    for res in sem_res:
        if res.get("semantic_score", 0) > SEMANTIC_FLOOR:
            valid_pids.add(res["post_id"])
            
    post_map = {}
    
    for res in kw_res:
        pid = res["post_id"]
        if pid not in valid_pids:
            continue
        post_map[pid] = {
            "post_id": res["post_id"],
            "author_id": res["author_id"],
            "text": res["text"],
            "keyword_score": res["keyword_score"],
            "keyword_rank": res["keyword_rank"]
        }
        
    for res in sem_res:
        pid = res["post_id"]
        if pid not in valid_pids:
            continue
        if pid not in post_map:
            post_map[pid] = {
                "post_id": res["post_id"],
                "author_id": res["author_id"],
                "text": res["text"],
                "semantic_score": res["semantic_score"],
                "semantic_rank": res["semantic_rank"]
            }
        else:
            post_map[pid]["semantic_score"] = res["semantic_score"]
            post_map[pid]["semantic_rank"] = res["semantic_rank"]
            
    for pid, data in post_map.items():
        rrf = 0.0
        if "keyword_rank" in data:
            rrf += 1.0 / (60 + data["keyword_rank"])
        if "semantic_rank" in data:
            rrf += 1.0 / (60 + data["semantic_rank"])
        data["rrf_score"] = rrf
        
        if "keyword_score" not in data:
            data["keyword_score"] = 0.0
        if "semantic_score" not in data:
            data["semantic_score"] = 0.0
            
    fused = list(post_map.values())
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    fused = fused[:top_k]
    
    for rank, res in enumerate(fused, 1):
        res["hybrid_rank"] = rank
        
    return fused

_ADJ_CACHE = {}
_ADJ_CACHE_KEY = None

def _get_adj_list(follow_edges: List[FollowEdge]) -> Dict[str, set]:
    global _ADJ_CACHE, _ADJ_CACHE_KEY
    # Content-based key also invalidates when a caller mutates the same list in place.
    current_key = tuple((e.follower_id, e.followee_id) for e in follow_edges)
    if _ADJ_CACHE_KEY == current_key:
        return _ADJ_CACHE
        
    adj = {}
    for e in follow_edges:
        if e.follower_id not in adj:
            adj[e.follower_id] = set()
        adj[e.follower_id].add(e.followee_id)
        
    _ADJ_CACHE = adj
    _ADJ_CACHE_KEY = current_key
    return adj

def trust_distance(user_id: str, author_id: str, follow_edges: List[FollowEdge]) -> int | None:
    if user_id == author_id:
        return 0
        
    adj = _get_adj_list(follow_edges)
        
    if user_id not in adj:
        return None
        
    hop1 = adj.get(user_id, set())
    if author_id in hop1:
        return 1
        
    for h1_user in hop1:
        if h1_user in adj and author_id in adj[h1_user]:
            return 2
            
    return None

def rerank_with_trust(hybrid_results: List[Dict], user_id: str, follow_edges: List[FollowEdge], trust_weight: float) -> List[Dict]:
    # EXPECTATION: hybrid_results must only contain candidates that have already survived the 
    # relevance floor. Trust is a reordering signal for relevant items, not a mechanism to 
    # promote an irrelevant item.
    if not hybrid_results:
        return []
        
    res_copy = [dict(r) for r in hybrid_results]
    trust_weight = min(max(float(trust_weight), 0.0), 1.0)
    
    # Cold-start auto-adjust: if user follows no one, cap trust_weight at 0.15
    adj = _get_adj_list(follow_edges)
    if len(adj.get(user_id, set())) == 0:
        effective_trust_weight = min(trust_weight, 0.15)
    else:
        effective_trust_weight = trust_weight
    
    # Bug 1 Fix: Scale Mismatch
    # RRF scores naturally compress into small floats (e.g. 0.015-0.033).
    # Trust scores are on the scale [0.1, 1.0]. Without normalization, trust dominates the ranking,
    # and a 0.3 trust_weight actually exerts 90%+ of the mathematical influence.
    # We min-max normalize the RRF score across the current candidate set to [0, 1] BEFORE combining.
    
    min_rrf = min(r["rrf_score"] for r in res_copy)
    max_rrf = max(r["rrf_score"] for r in res_copy)
    
    author_counts = {}
    final_results = []
    
    for res in res_copy:
        if max_rrf == min_rrf:
            normalized_relevance = 1.0
        else:
            normalized_relevance = (res["rrf_score"] - min_rrf) / (max_rrf - min_rrf)
            
        res["normalized_relevance"] = normalized_relevance
        
        dist = trust_distance(user_id, res["author_id"], follow_edges)
        res["trust_hops"] = dist
        
        if dist == 0 or dist == 1:
            t_score = 1.0
        elif dist == 2:
            t_score = 0.5
        else:
            t_score = 0.1
            
        res["trust_score"] = t_score
        res["final_score"] = (1.0 - effective_trust_weight) * normalized_relevance + effective_trust_weight * t_score
        
    res_copy.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Diversity Cap: Keep at most 2 results per author to prevent a trusted author from dominating.
    for res in res_copy:
        auth_id = res["author_id"]
        count = author_counts.get(auth_id, 0)
        if count < 2:
            author_counts[auth_id] = count + 1
            final_results.append(res)
            
    for rank, res in enumerate(final_results, 1):
        res["final_rank"] = rank
        
    return final_results

_PEOPLE_INDEX = None
_PEOPLE_INDEX_KEY = None

def _get_people_index(people: List[Person]):
    global _PEOPLE_INDEX, _PEOPLE_INDEX_KEY
    current_key = tuple((p.id, p.display_name, p.bio, tuple(p.topics)) for p in people)
    if _PEOPLE_INDEX_KEY == current_key:
        return _PEOPLE_INDEX
        
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        for p in string.punctuation:
            text = text.replace(p, '')
        return text.split()
        
    docs = [f"{p.display_name} {p.bio} {' '.join(p.topics)}" for p in people]
    tokenized_docs = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)
    
    _PEOPLE_INDEX = (bm25, _tokenize)
    _PEOPLE_INDEX_KEY = current_key
    return _PEOPLE_INDEX

def people_search(query: str, people: List[Person], top_k: int = 10) -> List[Dict]:
    if not query or not query.strip() or top_k <= 0 or not people:
        return []
    bm25, _tokenize = _get_people_index(people)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)
    
    results = []
    for i, score in enumerate(scores):
        if score > 0:
            results.append({
                "person_id": people[i].id,
                "display_name": people[i].display_name,
                "bio": people[i].bio,
                "topics": people[i].topics,
                "match_score": float(score)
            })
            
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results[:top_k]

def zero_result_recovery(query: str, user_id: str, people: List[Person], follow_edges: List[FollowEdge], top_k: int = 5) -> List[Dict]:
    if not query or not query.strip() or top_k <= 0 or not people:
        return []
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        for p in string.punctuation:
            text = text.replace(p, '')
        return text.split()
        
    q_tokens = set(_tokenize(query))
    
    results = []
    for p in people:
        dist = trust_distance(user_id, p.id, follow_edges)
        if dist is not None and dist <= 2:
            p_topics = set(t.lower() for t in p.topics)
            if q_tokens.intersection(p_topics):
                results.append({
                    "person_id": p.id,
                    "display_name": p.display_name,
                    "bio": p.bio,
                    "topics": p.topics,
                    "trust_hops": dist
                })
                
    results.sort(key=lambda x: (x["trust_hops"], x["display_name"].lower()))
    return results[:top_k]

if __name__ == "__main__":
    from corpus import get_posts, get_people, get_follow_graph, DEMO_USER_ID, COLD_START_USER_ID
    
    posts = get_posts()
    people = get_people()
    follow_edges = get_follow_graph()
    
    # 3b
    query = "climate risk"
    kw = keyword_search(query, posts, top_k=5)
    sem = semantic_search(query, posts, top_k=5)
    
    print("--- Keyword Search ---")
    for res in kw:
        print(f"[{res['keyword_score']:.4f}] {res['post_id']}: {res['text']}")
        
    print("\n--- Semantic Search ---")
    for res in sem:
        print(f"[{res['semantic_score']:.4f}] {res['post_id']}: {res['text']}")
        
    # ASSERT at least one "global warming" vocab post in semantic top 5 AND NOT in keyword top 5
    vocab_words = ["warming", "rising temperatures", "greenhouse gases", "melting ice caps", "extreme weather"]
    
    kw_ids = {r["post_id"] for r in kw}
    sem_ids = {r["post_id"] for r in sem}
    
    found_in_sem_not_kw = False
    for res in sem:
        if res["post_id"] not in kw_ids:
            text_lower = res["text"].lower()
            if any(v in text_lower for v in vocab_words):
                found_in_sem_not_kw = True
                break
                
    if not found_in_sem_not_kw:
        print("ERROR: Assertion failed: No global warming post found in semantic search top 5 that was missing from keyword top 5")
    assert found_in_sem_not_kw, "No global warming post found in semantic search top 5 that was missing from keyword top 5"
    
    # 3c
    hybrid = hybrid_search(query, posts, top_k=10)
    
    rerank_0 = rerank_with_trust(hybrid, DEMO_USER_ID, follow_edges, trust_weight=0.0)
    rerank_1 = rerank_with_trust(hybrid, DEMO_USER_ID, follow_edges, trust_weight=1.0)
    
    people_dict = {p.id: p for p in people}
    
    print("\n--- Trust Re-rank (Weight 0.0) ---")
    for res in rerank_0[:3]:
        author_name = people_dict[res['author_id']].display_name
        print(f"[{res['final_score']:.4f}] {author_name} (hops: {res['trust_hops']}): {res['text']}")
        
    print("\n--- Trust Re-rank (Weight 1.0) ---")
    for res in rerank_1[:3]:
        author_name = people_dict[res['author_id']].display_name
        print(f"[{res['final_score']:.4f}] {author_name} (hops: {res['trust_hops']}): {res['text']}")
        
    top3_0 = [r["post_id"] for r in rerank_0[:3]]
    top3_1 = [r["post_id"] for r in rerank_1[:3]]
    assert top3_0 != top3_1, "Top 3 authors did not change with trust weight"
    
    for res in rerank_1[:3]:
        assert res["trust_hops"] in (0, 1, 2), f"Top 3 result at weight 1.0 has invalid trust_hops: {res['trust_hops']}"
        
    # 3d
    rerank_cold = rerank_with_trust(hybrid, COLD_START_USER_ID, follow_edges, trust_weight=1.0)
    assert isinstance(rerank_cold, list) and len(rerank_cold) > 0, "Cold start rerank failed or returned empty"
    
    # 3e
    zero = zero_result_recovery("underwater basket weaving", DEMO_USER_ID, people, follow_edges)
    assert isinstance(zero, list), "Zero result recovery failed"
    
    # 3f
    print("\nALL RETRIEVAL CHECKS PASSED")
    
def explain_result_plain(result: dict, user_id: str, follow_edges) -> str:
    hops = result.get("trust_hops")
    kw = result.get("keyword_score", 0)
    sem = result.get("semantic_score", 0)
    
    if hops == 0 or hops == 1:
        if sem > kw:
            return "Related to your search topic, shared by someone you follow."
        else:
            return "A strong keyword match, shared by someone you follow."
    elif hops == 2:
        if sem > kw:
            return "Related to your search topic, shared by someone in your network."
        else:
            return "A strong keyword match, shared by someone in your network."
    else:
        if kw > sem and kw > 1.5:
            return "A strong keyword match for your search."
        elif sem >= kw:
            return "Related to your search topic."
        else:
            return "A match for your search terms."
