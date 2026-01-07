"""
Unit Tests for Web Intelligence Toolkit
Run with: pytest test_wi_toolkit.py -v
"""

import pytest
import math
import wi_toolkit as wt


# =============================================================================
# SECTION 1: TEXT PROCESSING (10 tests)
# =============================================================================

class TestTextProcessing:
    # prepare_text
    def test_prepare_text_lowercase(self):
        assert wt.prepare_text("HELLO", verbose=False) == "hello"
    
    def test_prepare_text_punctuation(self):
        assert wt.prepare_text("hello, world!", verbose=False) == "hello world"
    
    def test_prepare_text_numbers_removed(self):
        assert wt.prepare_text("test123", verbose=False) == "test"
    
    def test_prepare_text_numbers_kept(self):
        assert wt.prepare_text("test123", remove_numbers=False, verbose=False) == "test123"
    
    def test_prepare_text_whitespace(self):
        assert wt.prepare_text("  a   b  ", verbose=False) == "a b"
    
    # split_tokens
    def test_split_tokens(self):
        assert wt.split_tokens("a b c", verbose=False) == ["a", "b", "c"]
    
    def test_split_tokens_empty(self):
        assert wt.split_tokens("", verbose=False) == []
    
    # filter_common  
    def test_filter_common_stopwords(self):
        result = wt.filter_common(["the", "cat", "is", "big"], verbose=False)
        assert "the" not in result and "is" not in result
        assert "cat" in result and "big" in result
    
    def test_filter_common_extras(self):
        result = wt.filter_common(["cat", "dog"], extras={"cat"}, verbose=False)
        assert "cat" not in result and "dog" in result
    
    # reduce_stem
    @pytest.mark.parametrize("word,stem", [
        ("running", "run"), ("jumps", "jump"), ("played", "play"),
        ("cats", "cat"), ("happiness", "happi"), ("studies", "studi")
    ])
    def test_reduce_stem(self, word, stem):
        assert wt.reduce_stem(word, verbose=False) == stem
    
    # clean_pipeline
    def test_clean_pipeline_full(self):
        result = wt.clean_pipeline("The CATS are RUNNING!", verbose=False)
        assert "the" not in result  # stopword
        assert "cat" in result  # stemmed
        assert "run" in result  # stemmed


# =============================================================================
# SECTION 2: DISTANCE METRICS (12 tests)
# =============================================================================

class TestDistanceMetrics:
    # edit_distance (Levenshtein)
    @pytest.mark.parametrize("s1,s2,dist", [
        ("kitten", "sitting", 3),
        ("hello", "hello", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("cat", "bat", 1),
        ("sunday", "saturday", 3),
    ])
    def test_edit_distance(self, s1, s2, dist):
        assert wt.edit_distance(s1, s2, verbose=False) == dist
    
    # levenshtein_distance alias
    def test_levenshtein_distance_alias(self):
        assert wt.levenshtein_distance("cat", "car", verbose=False) == 1
    
    # hamming_distance (renamed from binary_distance)
    @pytest.mark.parametrize("s1,s2,dist", [
        ("karolin", "kathrin", 3),
        ("1011101", "1001001", 2),
        ("abc", "abc", 0),
        ("000", "111", 3),
    ])
    def test_hamming_distance(self, s1, s2, dist):
        assert wt.hamming_distance(s1, s2, verbose=False) == dist
    
    # binary_distance alias still works
    @pytest.mark.parametrize("s1,s2,dist", [
        ("karolin", "kathrin", 3),
        ("1011101", "1001001", 2),
        ("abc", "abc", 0),
        ("000", "111", 3),
    ])
    def test_binary_distance(self, s1, s2, dist):
        assert wt.binary_distance(s1, s2, verbose=False) == dist
    
    def test_binary_distance_unequal_raises(self):
        with pytest.raises(ValueError):
            wt.binary_distance("ab", "abc", verbose=False)
    
    # jaccard_similarity (renamed from overlap_coefficient)
    @pytest.mark.parametrize("s1,s2,sim", [
        ({1,2,3}, {2,3,4}, 0.5),
        ({1,2}, {1,2}, 1.0),
        ({1,2}, {3,4}, 0.0),
        ({1}, {1,2,3}, 1/3),
    ])
    def test_jaccard_similarity(self, s1, s2, sim):
        assert wt.jaccard_similarity(s1, s2, verbose=False) == pytest.approx(sim)
    
    # overlap_coefficient alias still works
    @pytest.mark.parametrize("s1,s2,sim", [
        ({1,2,3}, {2,3,4}, 0.5),
        ({1,2}, {1,2}, 1.0),
        ({1,2}, {3,4}, 0.0),
        ({1}, {1,2,3}, 1/3),
    ])
    def test_overlap_coefficient(self, s1, s2, sim):
        assert wt.overlap_coefficient(s1, s2, verbose=False) == pytest.approx(sim)
    
    # jaccard_distance (NEW)
    @pytest.mark.parametrize("s1,s2,dist", [
        ({1,2,3}, {2,3,4}, 0.5),  # 1 - 0.5 = 0.5
        ({1,2}, {1,2}, 0.0),      # 1 - 1.0 = 0.0
        ({1,2}, {3,4}, 1.0),      # 1 - 0.0 = 1.0
    ])
    def test_jaccard_distance(self, s1, s2, dist):
        assert wt.jaccard_distance(s1, s2, verbose=False) == pytest.approx(dist)
    
    # euclidean_distance (NEW)
    @pytest.mark.parametrize("v1,v2,dist", [
        ([0,0], [3,4], 5.0),       # 3-4-5 triangle
        ([1,1], [1,1], 0.0),       # same point
        ([0,0,0], [1,1,1], math.sqrt(3)),  # 3D
        ([1,2,3], [4,5,6], math.sqrt(27)),
    ])
    def test_euclidean_distance(self, v1, v2, dist):
        assert wt.euclidean_distance(v1, v2, verbose=False) == pytest.approx(dist)
    
    def test_euclidean_distance_dicts(self):
        v1 = {"a": 0, "b": 0}
        v2 = {"a": 3, "b": 4}
        assert wt.euclidean_distance(v1, v2, verbose=False) == pytest.approx(5.0)
    
    # cosine_similarity (renamed from vector_similarity)
    @pytest.mark.parametrize("v1,v2,sim", [
        ([1,0], [1,0], 1.0),
        ([1,0], [0,1], 0.0),
        ([1,1], [1,1], 1.0),
    ])
    def test_cosine_similarity(self, v1, v2, sim):
        assert wt.cosine_similarity(v1, v2, verbose=False) == pytest.approx(sim, abs=0.01)


# =============================================================================
# SECTION 3: VECTOR SPACE (15 tests)
# =============================================================================

class TestVectorSpace:
    # term_counts
    def test_term_counts_frequencies(self):
        counts, vocab = wt.term_counts(["a a b", "b c"], verbose=False)
        assert counts[0]["a"] == 2
        assert counts[0]["b"] == 1
        assert counts[1]["c"] == 1
    
    def test_term_counts_vocab(self):
        counts, vocab = wt.term_counts(["x y", "y z"], verbose=False)
        assert set(vocab) == {"x", "y", "z"}
    
    # bag_of_words alias
    def test_bag_of_words_alias(self):
        counts, vocab = wt.bag_of_words(["hello world"], verbose=False)
        assert counts[0]["hello"] == 1
        assert counts[0]["world"] == 1
    
    # count_vectorizer alias
    def test_count_vectorizer_alias(self):
        counts, vocab = wt.count_vectorizer(["the cat sat"], verbose=False)
        assert "cat" in vocab
    
    # build_td_matrix
    def test_build_td_matrix(self):
        matrix, vocab = wt.build_td_matrix(["a b", "b c", "a c"], verbose=False)
        assert matrix["a"] == [1, 0, 1]
        assert matrix["b"] == [1, 1, 0]
        assert matrix["c"] == [0, 1, 1]
    
    # compute_tfidf
    def test_compute_tfidf_idf_zero_for_common(self):
        _, _, idf = wt.compute_tfidf(["a b", "a c", "a d"], verbose=False)
        assert idf["a"] == 0  # in all 3 docs
    
    def test_compute_tfidf_idf_log_formula(self):
        _, _, idf = wt.compute_tfidf(["a b", "a c", "a d"], verbose=False)
        assert idf["b"] == pytest.approx(math.log(3/1))  # in 1 doc
    
    def test_compute_tfidf_returns_vectors(self):
        tfidf, vocab, _ = wt.compute_tfidf(["a b", "c d"], verbose=False)
        assert len(tfidf) == 2
        assert all(isinstance(d, dict) for d in tfidf)
    
    # vector_similarity (cosine)
    @pytest.mark.parametrize("v1,v2,sim", [
        ([1,0], [1,0], 1.0),
        ([1,0], [0,1], 0.0),
        ([1,0], [-1,0], -1.0),
        ([1,1], [1,1], 1.0),
        ([3,4], [4,3], 0.96),
    ])
    def test_vector_similarity(self, v1, v2, sim):
        assert wt.vector_similarity(v1, v2, verbose=False) == pytest.approx(sim, abs=0.01)
    
    def test_vector_similarity_dicts(self):
        v1 = {"a": 1, "b": 0}
        v2 = {"a": 1, "b": 0}
        assert wt.vector_similarity(v1, v2, verbose=False) == pytest.approx(1.0)
    
    # pairwise_scores
    def test_pairwise_scores_count(self):
        vecs = [{"a": 1}, {"a": 2}, {"a": 3}]
        results = wt.pairwise_scores(vecs, verbose=False)
        assert len(results) == 3  # C(3,2) = 3 pairs
    
    # rank_pairs
    def test_rank_pairs_order(self):
        pairs = [("A","B",0.3), ("A","C",0.9), ("B","C",0.6)]
        ranked = wt.rank_pairs(pairs, verbose=False)
        assert [r[2] for r in ranked] == [0.9, 0.6, 0.3]


# =============================================================================
# SECTION 3b: TF, DF, IDF FUNCTIONS (NEW)
# =============================================================================

class TestTfDfIdf:
    # compute_tf
    def test_compute_tf_values(self):
        tf = wt.compute_tf("the cat sat on the mat", verbose=False)
        assert tf["the"] == pytest.approx(2/6)  # 2 out of 6 tokens
        assert tf["cat"] == pytest.approx(1/6)
    
    def test_compute_tf_normalized(self):
        tf = wt.compute_tf("a a a b b c", verbose=False)
        assert tf["a"] == pytest.approx(3/6)
        assert tf["b"] == pytest.approx(2/6)
        assert tf["c"] == pytest.approx(1/6)
    
    # term_frequency alias
    def test_term_frequency_alias(self):
        tf = wt.term_frequency("hello world", verbose=False)
        assert tf["hello"] == pytest.approx(0.5)
        assert tf["world"] == pytest.approx(0.5)
    
    # compute_df
    def test_compute_df_values(self):
        docs = ["cat dog", "dog bird", "cat bird fish"]
        df = wt.compute_df(docs, verbose=False)
        assert df["cat"] == 2   # in 2 docs
        assert df["dog"] == 2   # in 2 docs
        assert df["bird"] == 2  # in 2 docs
        assert df["fish"] == 1  # in 1 doc
    
    # document_frequency alias
    def test_document_frequency_alias(self):
        docs = ["a b", "b c", "c d"]
        df = wt.document_frequency(docs, verbose=False)
        assert df["b"] == 2
        assert df["c"] == 2
    
    # compute_idf
    def test_compute_idf_values(self):
        docs = ["a b", "a c", "a d"]  # a in all 3, b,c,d each in 1
        idf = wt.compute_idf(docs, verbose=False)
        assert idf["a"] == 0  # log(3/3) = 0
        assert idf["b"] == pytest.approx(math.log(3))  # log(3/1)
        assert idf["c"] == pytest.approx(math.log(3))
        assert idf["d"] == pytest.approx(math.log(3))
    
    def test_compute_idf_rare_terms_higher(self):
        docs = ["common rare1", "common rare2", "common", "common"]
        idf = wt.compute_idf(docs, verbose=False)
        assert idf["rare1"] > idf["common"]
        assert idf["rare2"] > idf["common"]
    
    # inverse_document_frequency alias
    def test_inverse_document_frequency_alias(self):
        docs = ["x y", "y z"]
        idf = wt.inverse_document_frequency(docs, verbose=False)
        assert idf["y"] == 0  # in all docs
        assert idf["x"] > 0   # only in 1 doc
    
    # tfidf alias
    def test_tfidf_alias(self):
        tfidf_vecs, vocab, idf = wt.tfidf(["a b", "b c"], verbose=False)
        assert len(tfidf_vecs) == 2
        assert "a" in vocab and "b" in vocab and "c" in vocab


# =============================================================================
# SECTION 4: N-GRAMS (10 tests)
# =============================================================================

class TestNGrams:
    # extract_ngrams
    @pytest.mark.parametrize("tokens,n,expected", [
        (["a","b","c"], 1, [("a",), ("b",), ("c",)]),
        (["a","b","c"], 2, [("a","b"), ("b","c")]),
        (["a","b","c"], 3, [("a","b","c")]),
        (["a","b","c","d"], 2, [("a","b"), ("b","c"), ("c","d")]),
    ])
    def test_extract_ngrams(self, tokens, n, expected):
        assert wt.extract_ngrams(tokens, n, verbose=False) == expected
    
    # ngram_counts
    def test_ngram_counts_bigrams(self):
        counts, ctx = wt.ngram_counts(["a b c", "a b d"], 2, verbose=False)
        assert counts[("a", "b")] == 2
        assert counts[("b", "c")] == 1
    
    def test_ngram_counts_context(self):
        counts, ctx = wt.ngram_counts(["a b", "a c"], 2, verbose=False)
        assert ctx[("a",)] == 2
    
    # estimate_prob
    def test_estimate_prob_mle(self):
        counts = {("a","b"): 3}
        ctx = {("a",): 6}
        prob = wt.estimate_prob(("a","b"), counts, ctx, verbose=False)
        assert prob == pytest.approx(0.5)
    
    # apply_smoothing
    def test_apply_smoothing_seen_ngram(self):
        counts = {("a","b"): 4}
        ctx = {("a",): 8}
        smooth = wt.apply_smoothing(counts, ctx, vocab_size=10, k=1, verbose=False)
        # (4+1)/(8+10) = 5/18
        assert smooth(("a","b")) == pytest.approx(5/18)
    
    def test_apply_smoothing_unseen_ngram(self):
        counts = {}
        ctx = {("a",): 8}
        smooth = wt.apply_smoothing(counts, ctx, vocab_size=10, k=1, verbose=False)
        # (0+1)/(8+10) = 1/18
        assert smooth(("a","x")) == pytest.approx(1/18)


# =============================================================================
# SECTION 5: INVERTED INDEX (10 tests)
# =============================================================================

class TestInvertedIndex:
    # build_index
    def test_build_index_postings(self):
        index = wt.build_index(["a b","b c","a c"], ["D1","D2","D3"], verbose=False)
        assert set(index["a"]) == {"D1", "D3"}
        assert set(index["b"]) == {"D1", "D2"}
    
    def test_build_index_auto_labels(self):
        index = wt.build_index(["x y", "y z"], verbose=False)
        assert "x" in index and "z" in index
    
    # merge_and
    @pytest.mark.parametrize("p1,p2,expected", [
        ({"A","B","C"}, {"B","C","D"}, {"B","C"}),
        ({"A"}, {"B"}, set()),
        ({"A","B"}, set(), set()),
    ])
    def test_merge_and(self, p1, p2, expected):
        assert wt.merge_and(p1, p2, verbose=False) == expected
    
    # merge_or
    @pytest.mark.parametrize("p1,p2,expected", [
        ({"A","B"}, {"B","C"}, {"A","B","C"}),
        ({"A"}, set(), {"A"}),
        (set(), set(), set()),
    ])
    def test_merge_or(self, p1, p2, expected):
        assert wt.merge_or(p1, p2, verbose=False) == expected
    
    # query_index
    def test_query_index_and(self):
        index = wt.build_index(["a b","b c","a b c"], ["D1","D2","D3"], verbose=False)
        result = wt.query_index(index, "a b", op='AND', verbose=False)
        assert result == {"D1", "D3"}
    
    def test_query_index_or(self):
        index = wt.build_index(["a","b","c"], ["D1","D2","D3"], verbose=False)
        result = wt.query_index(index, "a c", op='OR', verbose=False)
        assert result == {"D1", "D3"}


# =============================================================================
# SECTION 6: NEURAL GATES (16 tests via parametrize)
# =============================================================================

class TestGates:
    @pytest.mark.parametrize("x,y,out", [(0,0,0),(0,1,0),(1,0,0),(1,1,1)])
    def test_gate_and(self, x, y, out):
        assert wt.gate_and(x, y, verbose=False) == out
    
    @pytest.mark.parametrize("x,y,out", [(0,0,0),(0,1,1),(1,0,1),(1,1,1)])
    def test_gate_or(self, x, y, out):
        assert wt.gate_or(x, y, verbose=False) == out
    
    @pytest.mark.parametrize("x,y,out", [(0,0,1),(0,1,1),(1,0,1),(1,1,0)])
    def test_gate_nand(self, x, y, out):
        assert wt.gate_nand(x, y, verbose=False) == out
    
    @pytest.mark.parametrize("x,y,out", [(0,0,0),(0,1,1),(1,0,1),(1,1,0)])
    def test_gate_xor(self, x, y, out):
        assert wt.gate_xor(x, y, verbose=False) == out
    
    # step_function
    @pytest.mark.parametrize("x,out", [(-5,0),(-0.1,0),(0,1),(0.1,1),(5,1)])
    def test_step_function(self, x, out):
        assert wt.step_function(x, verbose=False) == out
    
    # perceptron
    def test_perceptron_and_gate(self):
        # AND: w=[1,1], b=-1.5
        assert wt.perceptron([0,0], [1,1], -1.5, verbose=False) == 0
        assert wt.perceptron([1,1], [1,1], -1.5, verbose=False) == 1
    
    def test_perceptron_or_gate(self):
        # OR: w=[1,1], b=-0.5
        assert wt.perceptron([0,0], [1,1], -0.5, verbose=False) == 0
        assert wt.perceptron([0,1], [1,1], -0.5, verbose=False) == 1


# =============================================================================
# SECTION 7: GRAPH ANALYTICS (10 tests)
# =============================================================================

class TestGraphAnalytics:
    # degree_centrality
    def test_degree_centrality_complete(self):
        # K3: each node connected to 2 others, max=2, so centrality = 1.0
        adj = [[0,1,1],[1,0,1],[1,1,0]]
        result = wt.degree_centrality(adj, verbose=False)
        assert all(v == pytest.approx(1.0) for v in result.values())
    
    def test_degree_centrality_star(self):
        # Star: center has degree 3, leaves have degree 1
        adj = [[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]]
        result = wt.degree_centrality(adj, verbose=False)
        assert result[0] == pytest.approx(1.0)  # 3/(4-1) = 1.0
        assert result[1] == pytest.approx(1/3)  # 1/(4-1) = 0.333
    
    # closeness_centrality
    def test_closeness_centrality_path(self):
        # Path: 0--1--2, node 1 is most central
        adj = [[0,1,0],[1,0,1],[0,1,0]]
        result = wt.closeness_centrality(adj, verbose=False)
        assert result[1] > result[0]
        assert result[1] > result[2]
    
    # pagerank
    def test_pagerank_cycle_equal(self):
        adj = [[0,1,0],[0,0,1],[1,0,0]]
        result = wt.pagerank(adj, iterations=100, verbose=False)
        vals = list(result.values())
        assert all(abs(v - vals[0]) < 0.01 for v in vals)
    
    def test_pagerank_sum_to_one(self):
        adj = [[0,1,1],[1,0,0],[0,1,0]]
        result = wt.pagerank(adj, iterations=50, verbose=False)
        assert sum(result.values()) == pytest.approx(1.0, abs=0.01)
    
    # clustering_coefficient
    def test_clustering_coefficient_triangle(self):
        adj = [[0,1,1],[1,0,1],[1,1,0]]
        result = wt.clustering_coefficient(adj, verbose=False)
        assert all(v == pytest.approx(1.0) for v in result.values())
    
    def test_clustering_coefficient_star(self):
        # Star has 0 clustering for center (neighbors not connected)
        adj = [[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]]
        result = wt.clustering_coefficient(adj, verbose=False)
        assert result[0] == 0.0  # center has no triangles


# =============================================================================
# SECTION 8: EVALUATION & EMBEDDINGS (12 tests)
# =============================================================================

class TestEvaluation:
    # eval_metrics
    def test_eval_metrics_values(self):
        y_true = [1,1,0,0,1,0]
        y_pred = [1,0,0,0,1,1]
        # TP=2, TN=2, FP=1, FN=1
        result = wt.eval_metrics(y_true, y_pred, verbose=False)
        assert result['accuracy'] == pytest.approx(4/6)
        assert result['precision'] == pytest.approx(2/3)
        assert result['recall'] == pytest.approx(2/3)
    
    def test_eval_metrics_perfect(self):
        y = [1,0,1,0]
        result = wt.eval_metrics(y, y, verbose=False)
        assert result['accuracy'] == 1.0
        assert result['f1'] == 1.0
    
    def test_eval_metrics_all_wrong(self):
        result = wt.eval_metrics([1,1,1], [0,0,0], verbose=False)
        assert result['accuracy'] == 0.0
    
    # split_data
    def test_split_data_sizes(self):
        X = list(range(100))
        y = [0]*100
        train, val, test = wt.split_data(X, y, 0.7, 0.15, 0.15, verbose=False)
        assert len(train[0]) == 70
        assert len(val[0]) == 15
        assert len(test[0]) == 15
    
    def test_split_data_no_overlap(self):
        X = list(range(10))
        y = [0]*10
        train, val, test = wt.split_data(X, y, 0.6, 0.2, 0.2, verbose=False)
        all_x = set(train[0]) | set(val[0]) | set(test[0])
        assert all_x == set(range(10))
    
    # aggregate_tokens
    def test_aggregate_tokens_average(self):
        vecs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        result = wt.aggregate_tokens(["a", "b"], vecs, verbose=False)
        assert result == [pytest.approx(0.5), pytest.approx(0.5)]
    
    def test_aggregate_tokens_unknown_skipped(self):
        vecs = {"a": [1.0, 2.0]}
        result = wt.aggregate_tokens(["a", "unknown"], vecs, verbose=False)
        assert result == [1.0, 2.0]
    
    # aggregate_sentences
    def test_aggregate_sentences(self):
        embs = [[1,0,0], [0,1,0], [0,0,1]]
        result = wt.aggregate_sentences(embs, verbose=False)
        assert result == [pytest.approx(1/3)]*3
    
    # find_nearest
    def test_find_nearest_order(self):
        vecs = [[1,0], [0.9,0.1], [0,1]]
        labels = ["A", "B", "C"]
        result = wt.find_nearest([1,0], vecs, labels, k=3, verbose=False)
        assert result[0][0] == "A"  # most similar
        assert result[1][0] == "B"  # second
    
    def test_find_nearest_k_limit(self):
        vecs = [[i,0] for i in range(10)]
        labels = [f"D{i}" for i in range(10)]
        result = wt.find_nearest([5,0], vecs, labels, k=3, verbose=False)
        assert len(result) == 3


# =============================================================================
# SECTION 9: COMPUTATION GRAPH (8 tests)
# =============================================================================

class TestComputationGraph:
    # forward_pass operations
    def test_forward_pass_add(self):
        result, _ = wt.forward_pass([2, 3], [('add', [0, 1])], verbose=False)
        assert result == 5
    
    def test_forward_pass_mul(self):
        result, _ = wt.forward_pass([2, 3], [('mul', [0, 1])], verbose=False)
        assert result == 6
    
    def test_forward_pass_sub(self):
        result, _ = wt.forward_pass([5, 3], [('sub', [0, 1])], verbose=False)
        assert result == 2
    
    def test_forward_pass_chain(self):
        # (2+3)*4 = 20
        result, _ = wt.forward_pass([2, 3, 4], [('add', [0, 1]), ('mul', [3, 2])], verbose=False)
        assert result == 20
    
    def test_forward_pass_complex(self):
        # (a*b) + (c*d) where a=1,b=2,c=3,d=4 → 2 + 12 = 14
        ops = [('mul', [0, 1]), ('mul', [2, 3]), ('add', [4, 5])]
        result, _ = wt.forward_pass([1, 2, 3, 4], ops, verbose=False)
        assert result == 14
    
    def test_forward_pass_returns_cache(self):
        result, cache = wt.forward_pass([2, 3], [('add', [0, 1])], verbose=False)
        assert len(cache) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
