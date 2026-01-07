import wi_toolkit as wi


# =============================================================================
# DISTANCE METRICS EXAMPLES
# =============================================================================

def example_levenshtein():
    """Levenshtein (Edit) Distance - minimum edits to transform one string to another"""
    wi.levenshtein_distance("kitten", "sitting")


def example_hamming():
    """Hamming Distance - number of positions with different characters (equal length strings)"""
    wi.hamming_distance("karolin", "kathrin")


def example_jaccard():
    """Jaccard Similarity & Distance - set overlap measure"""
    set_a = {"cat", "dog", "bird"}
    set_b = {"dog", "bird", "fish"}
    wi.jaccard_similarity(set_a, set_b)
    wi.jaccard_distance(set_a, set_b)


def example_euclidean():
    """Euclidean Distance - straight-line distance between vectors"""
    wi.euclidean_distance([1, 2, 3], [4, 5, 6])


def example_cosine():
    """Cosine Similarity - angle between vectors (direction, not magnitude)"""
    wi.cosine_similarity([1, 2, 3], [4, 5, 6])


# =============================================================================
# TF-IDF EXAMPLES
# =============================================================================

def example_tf():
    """Term Frequency - how often a word appears in a document"""
    wi.compute_tf("the cat sat on the mat the cat was happy")


def example_df():
    """Document Frequency - in how many documents does a word appear"""
    docs = ["the cat sat", "the dog ran", "a bird flew", "the cat ran"]
    wi.compute_df(docs)


def example_idf():
    """Inverse Document Frequency - rarity of a word across documents"""
    docs = ["the cat sat", "the dog ran", "a bird flew", "the cat ran"]
    wi.compute_idf(docs)


def example_tfidf():
    """TF-IDF - combines TF and IDF for document relevance scoring"""
    docs = ["the cat sat on the mat", "the dog ran in the park", "a bird flew over"]
    wi.compute_tfidf(docs)


def example_bag_of_words():
    """Bag of Words / Count Vectorizer - word frequency counts"""
    docs = ["the cat sat", "the dog ran", "the cat ran"]
    wi.bag_of_words(docs)


def example_term_document_matrix():
    """Term-Document Matrix - rows=terms, columns=documents"""
    docs = ["the cat sat", "the dog ran", "the cat ran"]
    wi.term_document_matrix(docs)


def example_pairwise_similarity():
    """Pairwise Cosine Similarities between documents"""
    docs = ["the cat sat", "the dog ran", "the cat ran"]
    tfidf_vecs, vocab, _ = wi.compute_tfidf(docs, verbose=False)
    wi.pairwise_scores(tfidf_vecs, labels=["Doc0", "Doc1", "Doc2"])


def example_rank_pairs():
    """Rank document pairs by similarity"""
    docs = ["the cat sat", "the dog ran", "the cat ran"]
    tfidf_vecs, vocab, _ = wi.compute_tfidf(docs, verbose=False)
    pairs = wi.pairwise_scores(tfidf_vecs, labels=["Doc0", "Doc1", "Doc2"], verbose=False)
    wi.rank_pairs(pairs)


# =============================================================================
# N-GRAM & LANGUAGE MODEL EXAMPLES
# =============================================================================

def example_ngrams():
    """N-gram Extraction - consecutive sequences of n words"""
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    print("=== BIGRAMS ===")
    wi.extract_ngrams(tokens, n=2)
    print("\n=== TRIGRAMS ===")
    wi.extract_ngrams(tokens, n=3)


def example_ngram_counts():
    """N-gram Counting - frequency of n-grams in a corpus"""
    corpus = ["the cat sat", "the cat ran", "the dog sat", "a cat ran"]
    wi.ngram_counts(corpus, n=2)


def example_mle():
    """Maximum Likelihood Estimation - P(word | context)"""
    corpus = ["the cat sat", "the cat ran", "the dog sat", "a cat ran"]
    counts, ctx = wi.ngram_counts(corpus, n=2, verbose=True)
    wi.maximum_likelihood(("the", "cat"), counts, ctx)


def example_likelihood_distribution():
    """Next Word Probability Distribution - visual bar chart of likely next words"""
    corpus = ["the cat sat", "the cat ran", "the dog sat", "the dog ran", "a cat ran"]
    counts, ctx = wi.ngram_counts(corpus, n=2, verbose=True)
    wi.show_ngram_probabilities(("the",), counts, ctx)


def example_probability_matrix():
    """N-gram Probability Matrix - full P(word|context) matrix in terminal"""
    corpus = ["the cat sat", "the cat ran", "the dog sat", "a cat ran"]
    print("=== BIGRAM (n=2) ===")
    counts, ctx = wi.ngram_counts(corpus, n=2, verbose=False)
    wi.ngram_probability_matrix(counts, ctx)
    
    print("\n\n=== TRIGRAM (n=3) ===")
    counts3, ctx3 = wi.ngram_counts(corpus, n=3, verbose=False)
    wi.ngram_probability_matrix(counts3, ctx3)


def example_sentence_probability():
    """Sentence Probability - probability of a sentence using chain rule"""
    corpus = ["the cat sat", "the cat ran", "the dog sat", "a cat ran"]
    counts, ctx = wi.ngram_counts(corpus, n=2, verbose=True)
    wi.sentence_probability("the cat sat", counts, ctx, n=2)


def example_smoothing():
    """Laplace Smoothing - prevents zero probabilities for unseen n-grams"""
    corpus = ["the cat sat", "the cat ran"]
    counts, ctx = wi.ngram_counts(corpus, n=2, verbose=True)
    vocab_size = 10
    smooth = wi.laplace_smoothing(counts, ctx, vocab_size, k=1)
    print("\n=== Smoothed probability for UNSEEN n-gram ===")
    smooth(("the", "elephant"))


def example_perplexity():
    """Perplexity - how well does the model predict test data (lower = better)"""
    train = ["the cat sat", "the cat ran", "the dog sat"]
    test = "the cat sat"
    counts, ctx = wi.ngram_counts(train, n=2, verbose=True)
    wi.compute_perplexity(test, counts, ctx, n=2)


# =============================================================================
# TEXT PROCESSING EXAMPLES
# =============================================================================

def example_stemming():
    """Porter Stemmer - reduce words to their stem"""
    words = ["Rationalization"]
    for word in words:
        wi.reduce_stem(word)


def example_preprocessing():
    """Full Text Preprocessing Pipeline - normalize, tokenize, filter stopwords, stem"""
    text = "The CATS are RUNNING quickly through the Beautiful GARDENS!"
    wi.clean_pipeline(text)


# =============================================================================
# INVERTED INDEX & BOOLEAN SEARCH EXAMPLES
# =============================================================================

def example_inverted_index():
    """Build Inverted Index - term -> list of documents containing it"""
    docs = ["cat dog bird", "dog fish", "cat bird fish", "cat dog"]
    labels = ["D1", "D2", "D3", "D4"]
    wi.build_index(docs, labels)


def example_boolean_search():
    """Boolean Search - AND/OR queries on inverted index"""
    docs = ["cat dog bird", "dog fish", "cat bird fish", "cat dog"]
    labels = ["D1", "D2", "D3", "D4"]
    index = wi.build_index(docs, labels, verbose=True)
    
    print("=== AND Query: 'cat dog' ===")
    wi.query_index(index, "cat dog", op='AND')
    
    print("\n=== OR Query: 'fish bird' ===")
    wi.query_index(index, "fish bird", op='OR')


def example_merge_operations():
    """Posting List Merge Operations - AND (intersection) and OR (union)"""
    postings1 = {"D1", "D2", "D3"}
    postings2 = {"D2", "D3", "D4"}
    
    print("=== AND Merge (Intersection) ===")
    wi.merge_and(postings1, postings2)
    
    print("\n=== OR Merge (Union) ===")
    wi.merge_or(postings1, postings2)


# =============================================================================
# NEURAL NETWORK GATES EXAMPLES
# =============================================================================

def example_perceptron():
    """Single Perceptron - weighted sum + step activation"""
    inputs = [1, 0]
    weights = [0.5, 0.5]
    bias = -0.25
    wi.perceptron(inputs, weights, bias)


def example_logic_gates():
    """Logic Gates using Perceptrons - AND, OR, NAND, XOR"""
    print("=== AND Gate ===")
    wi.gate_and(1, 1)
    wi.gate_and(1, 0)
    
    print("\n=== OR Gate ===")
    wi.gate_or(0, 0)
    wi.gate_or(1, 0)
    
    print("\n=== XOR Gate (requires 2 layers!) ===")
    wi.gate_xor(0, 0)
    wi.gate_xor(1, 1)
    wi.gate_xor(1, 0)


def example_truth_table():
    """Show Truth Table for any gate"""
    wi.show_truth_table(wi.gate_and, "AND")
    wi.show_truth_table(wi.gate_xor, "XOR")


# =============================================================================
# GRAPH ANALYTICS EXAMPLES
# =============================================================================

def example_degree_centrality():
    """Degree Centrality - how connected is each node"""
    # Simple triangle graph: A--B--C--A
    adj = [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]
    wi.degree_centrality(adj, labels=["A", "B", "C"])


def example_closeness_centrality():
    """Closeness Centrality - how close is each node to all others"""
    # Path graph: A--B--C--D
    adj = [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ]
    wi.closeness_centrality(adj, labels=["A", "B", "C", "D"])


def example_betweenness_centrality():
    """Betweenness Centrality - how often is node on shortest paths"""
    adj = [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ]
    wi.betweenness_centrality(adj, labels=["A", "B", "C", "D"])


def example_pagerank():
    """PageRank Algorithm - importance based on incoming links"""
    # Simple directed graph
    adj = [
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 0]
    ]
    wi.pagerank(adj, labels=["A", "B", "C", "D"])


def example_clustering_coefficient():
    """Clustering Coefficient - how tightly connected are neighbors"""
    # Triangle graph (fully connected)
    adj = [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]
    wi.clustering_coefficient(adj, labels=["A", "B", "C"])


# =============================================================================
# EVALUATION & ML EXAMPLES
# =============================================================================

def example_train_test_split():
    """Split Data into Train/Validation/Test sets"""
    X = list(range(100))
    y = [0]*50 + [1]*50
    train, val, test = wi.split_data(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)


def example_eval_metrics():
    """Evaluation Metrics - Accuracy, Precision, Recall, F1"""
    y_true = [1, 1, 0, 0, 1, 0, 1, 1]
    y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
    wi.eval_metrics(y_true, y_pred)


def example_embeddings():
    """Word Embeddings - aggregate word vectors to sentence/document embeddings"""
    # Simulated word vectors (normally loaded from model)
    word_vectors = {
        "cat": [0.1, 0.2, 0.3],
        "sat": [0.2, 0.1, 0.4],
        "mat": [0.3, 0.3, 0.2]
    }
    tokens = ["cat", "sat", "mat"]
    wi.aggregate_tokens(tokens, word_vectors)


def example_find_nearest():
    """Find Nearest Neighbors by Cosine Similarity"""
    target = [1.0, 0.5, 0.2]
    all_vectors = [
        [1.0, 0.5, 0.2],  # identical
        [0.9, 0.6, 0.3],  # very similar
        [0.1, 0.1, 0.9],  # very different
    ]
    labels = ["DocA", "DocB", "DocC"]
    wi.find_nearest(target, all_vectors, labels, k=3)


# =============================================================================
# COMPUTATION GRAPH EXAMPLES
# =============================================================================

def example_forward_pass():
    """Forward Pass - compute output of a computation graph"""
    # Compute (a + b) * c where a=2, b=3, c=4
    inputs = [2, 3, 4]
    operations = [
        ('add', [0, 1]),   # node 3 = a + b
        ('mul', [3, 2])    # node 4 = (a+b) * c
    ]
    result, cache = wi.forward_pass(inputs, operations)


def example_backward_pass():
    """Backward Pass - compute gradients via backpropagation"""
    inputs = [2, 3, 4]
    operations = [
        ('add', [0, 1]),
        ('mul', [3, 2])
    ]
    result, cache = wi.forward_pass(inputs, operations, verbose=False)
    print("=== BACKWARD PASS ===")
    gradients = wi.backward_pass(1.0, cache)


# =============================================================================
# MAIN - UNCOMMENT THE EXAMPLE YOU WANT TO RUN
# =============================================================================

def main():
    # === DISTANCE METRICS ===
    # example_levenshtein()
    # example_hamming()
    # example_jaccard()
    # example_euclidean()
    # example_cosine()
    
    # === TF-IDF ===
    # example_tf()
    # example_df()
    # example_idf()
    # example_tfidf()
    # example_bag_of_words()
    # example_term_document_matrix()
    # example_pairwise_similarity()
    # example_rank_pairs()
    
    # === N-GRAMS & LANGUAGE MODELS ===
    # example_ngrams()
    # example_ngram_counts()
    # example_mle()
    # example_likelihood_distribution() 
    # example_probability_matrix()  # <-- NEW: shows full P(word|context) matrix
    # example_sentence_probability()
    # example_smoothing()
    # example_perplexity()
    
    # === TEXT PROCESSING ===
    # example_stemming()
    # example_preprocessing()
    
    # === INVERTED INDEX & SEARCH ===
    # example_inverted_index()
    # example_boolean_search()
    # example_merge_operations()
    
    # === NEURAL NETWORK GATES ===
    # example_perceptron()
    # example_logic_gates()
    # example_truth_table()
    
    # === GRAPH ANALYTICS ===
    # example_degree_centrality()
    # example_closeness_centrality()
    # example_betweenness_centrality()
    # example_pagerank()
    # example_clustering_coefficient()
    
    # === EVALUATION & ML ===
    # example_train_test_split()
    # example_eval_metrics()
    # example_embeddings()
    # example_find_nearest()
    
    # === COMPUTATION GRAPHS ===
    # example_forward_pass()
    # example_backward_pass()
    
    pass  # Uncomment an example above to run it


# =============================================================================
# SELF-STUDY ALLOWED FUNCTIONS
# =============================================================================
# These map to the example functions above that are "allowed" based on self-study code.

def selfstudy():
    """
    Self-study allowed example functions.
    Uncomment the ones you want to run.
    
    Mapped from:
    - selfstudyi.py       -> Text preprocessing
    - selfstudyii.py      -> TF-IDF & similarity
    - selfstudyiii_iv.py  -> Inverted index
    - selfstudy.py        -> Classification
    - embeddings.py       -> Word embeddings
    """
    
    # =========================================================================
    # SELF-STUDY I: Text Preprocessing & Zipf's Law
    # =========================================================================
    example_stemming()          
    # example_preprocessing()     
    
    # =========================================================================
    # SELF-STUDY II: TF-IDF & Document Similarity
    # =========================================================================
#     example_tfidf()             
    # example_pairwise_similarity() 
    # example_rank_pairs()        
    # example_cosine()            
    
    # =========================================================================
    # SELF-STUDY III & IV: Inverted Index & Boolean Search
    # =========================================================================
    # example_inverted_index()    
    # example_boolean_search()   
    # example_merge_operations() 
    
    # =========================================================================
    # SELF-STUDY Classification: Train/Test Split & Evaluation
    # =========================================================================
    # example_train_test_split()  
    # example_eval_metrics()   
    
    # =========================================================================
    # SELF-STUDY Word Embeddings: Entity Similarity
    # =========================================================================
    # example_embeddings()      
    # example_find_nearest()

if __name__ == "__main__":
    # main()
    selfstudy()  # Uncomment to see self-study allowed functions