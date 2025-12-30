"""
Web Intelligence Toolkit
Modular functions for text analysis, similarity, and ML concepts.

Includes:
    - Text Processing: prepare_text, split_tokens, filter_common, reduce_stem, clean_pipeline
    - Distance Metrics: edit_distance, binary_distance, overlap_coefficient
    - Vector Space: term_counts, build_td_matrix, compute_tfidf, vector_similarity, pairwise_scores, rank_pairs
    - Language Models: extract_ngrams, ngram_counts, estimate_prob, sequence_probability, apply_smoothing, compute_perplexity
    - Search: build_index, merge_and, merge_or, query_index
    - Neural Gates: step_function, perceptron, gate_and, gate_or, gate_nand, gate_xor
    - Graph Analytics: degree_centrality, closeness_centrality, betweenness_centrality, pagerank, clustering_coefficient
    - Evaluation & Embeddings: split_data, eval_metrics, aggregate_tokens, aggregate_sentences, find_nearest
    - Computation Graphs: forward_pass
"""

import re
import math
from collections import Counter, defaultdict

# NLTK imports for proper stemming and stopwords
try:
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords as nltk_stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("[Warning] Package not installed. Using fallback implementations.")


def nltk_setup():
    """Download required NLTK data if not already present."""
    if NLTK_AVAILABLE:
        try:
            nltk_stopwords.words('english')
        except LookupError:
            print("Error getting stopwords...")
            nltk.download('stopwords', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Error getting punkt tokenizer...")
            nltk.download('punkt', quiet=True)


# Initialize NLTK data
if NLTK_AVAILABLE:
    nltk_setup()
    _stemmer = PorterStemmer()
    
    class DebugPorterStemmer(PorterStemmer):
        """
        An extended PorterStemmer that prints the intermediate steps of the stemming algorithm.
        """
        def stem(self, word):
            stem = word.lower()
            print(f"Original: {word}")

            # Step 1a
            stem = self._step1a(stem)
            print(f"Step 1a : {stem}")

            # Step 1b
            stem = self._step1b(stem)
            print(f"Step 1b : {stem}")

            # Step 1c
            stem = self._step1c(stem)
            print(f"Step 1c : {stem}")

            # Step 2
            stem = self._step2(stem)
            print(f"Step 2  : {stem}")

            # Step 3
            stem = self._step3(stem)
            print(f"Step 3  : {stem}")

            # Step 4
            stem = self._step4(stem)
            print(f"Step 4  : {stem}")

            # Step 5a
            stem = self._step5a(stem)
            print(f"Step 5a : {stem}")

            # Step 5b
            stem = self._step5b(stem)
            print(f"Step 5b : {stem}")
            
            print(f"Result  : {stem}\n" + "-"*20)
            return stem
            
    _debug_stemmer = DebugPorterStemmer()
    _stopwords = set(nltk_stopwords.words('english'))
else:
    _stemmer = None
    _stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been'}


# =============================================================================
# SECTION 1: TEXT PROCESSING & STEMMING
# =============================================================================

def prepare_text(text, remove_numbers=True, verbose=True):
    """Normalize text: lowercase and remove punctuation/numbers."""
    if verbose:
        print(f"[Step 1] Original text: {text[:100]}..." if len(text) > 100 else f"[Step 1] Original text: {text}")
    
    result = text.lower()
    if verbose:
        print(f"[Step 2] After lowercase: {result[:100]}..." if len(result) > 100 else f"[Step 2] After lowercase: {result}")
    
    if remove_numbers:
        result = re.sub(r'\d+', '', result)
        if verbose:
            print(f"[Step 3] After removing numbers")
    
    result = re.sub(r'[^\w\s]', '', result)
    if verbose:
        print(f"[Step 4] After removing punctuation")
    
    result = ' '.join(result.split())
    if verbose:
        print(f"[Step 5] After normalizing whitespace: {len(result)} chars")
    
    return result


def split_tokens(text, verbose=True):
    """Tokenize text into words."""
    if verbose:
        print(f"[Tokenizing] Input length: {len(text)} chars")
    
    tokens = text.split()
    if verbose:
        print(f"[Result] {len(tokens)} tokens")
    
    return tokens


def filter_common(tokens, extras=None, verbose=True):
    """Remove stopwords using NLTK's stopwords corpus."""
    stopwords = _stopwords.copy()
    
    if extras:
        stopwords.update(extras)
    
    if verbose:
        print(f"[Filtering] Input: {len(tokens)} tokens")
        print(f"[Filtering] Using stopwords: {len(stopwords)} words")
    
    result = [t for t in tokens if t.lower() not in stopwords]
    
    if verbose:
        removed_count = len(tokens) - len(result)
        print(f"[Filtering] Removed {removed_count} stopwords")
        print(f"[Result] {len(result)} tokens remaining")
    
    return result


def reduce_stem(word, verbose=True):
    """
    Stem a word using NLTK's Porter Stemmer.
    Shows the transformation step.
    """
    if verbose:
        print(f"\n[STEMMING] Word: '{word}'")
    
    if NLTK_AVAILABLE and _stemmer:
        if verbose:
            # Use the debug stemmer to show all steps
            stem = _debug_stemmer.stem(word)
        else:
            stem = _stemmer.stem(word)
            
        return stem
    else:
        # Fallback: basic suffix removal
        if verbose:
            print(f"  [Fallback] Package not available, using basic rules")
        
        result = word.lower()
        # Basic suffix rules
        if result.endswith('ing'):
            result = result[:-3]
        elif result.endswith('ed'):
            result = result[:-2]
        elif result.endswith('ly'):
            result = result[:-2]
        elif result.endswith('s') and not result.endswith('ss'):
            result = result[:-1]
        
        if verbose:
            print(f"  '{word}' -> '{result}'")
        return result


def clean_pipeline(text, remove_numbers=True, custom_stops=None, verbose=True):
    """Full preprocessing: normalize -> tokenize -> filter -> stem."""
    if verbose:
        print("="*60)
        print("FULL PREPROCESSING PIPELINE")
        print("="*60)
    
    text = prepare_text(text, remove_numbers, verbose)
    tokens = split_tokens(text, verbose)
    tokens = filter_common(tokens, custom_stops, verbose)
    
    if verbose:
        print("\n[Stemming all tokens]")
    stems = [reduce_stem(t, verbose) for t in tokens]
    
    if verbose:
        print("\n" + "="*60)
        print(f"[FINAL RESULT first 100] {stems[:100]}")
        print("="*60)
    
    return stems


# =============================================================================
# SECTION 2: DISTANCE METRICS
# =============================================================================

def edit_distance(s1, s2, verbose=True):
    """
    Levenshtein distance using dynamic programming.
    Shows the full DP matrix.
    """
    m, n = len(s1), len(s2)
    
    if verbose:
        print(f"\n[LEVENSHTEIN DISTANCE]")
        print(f"String 1: '{s1}' (length {m})")
        print(f"String 2: '{s2}' (length {n})")
        print(f"\n[Step 1] Initialize DP matrix of size ({m+1} x {n+1})")
    
    # Create DP matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    if verbose:
        print(f"[Step 2] Initialize first row: {dp[0]}")
        print(f"[Step 2] Initialize first column: {[dp[i][0] for i in range(m+1)]}")
    
    # Fill the matrix
    if verbose:
        print(f"\n[Step 3] Fill matrix using recurrence:")
        print(f"  dp[i][j] = dp[i-1][j-1] if s1[i-1] == s2[j-1]")
        print(f"  dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) otherwise")
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    if verbose:
        print(f"\n[Step 4] Final DP matrix:")
        # Header
        header = "    " + "  ".join([" "] + list(s2))
        print(header)
        for i in range(m + 1):
            row_char = " " if i == 0 else s1[i-1]
            row_vals = "  ".join(str(v) for v in dp[i])
            print(f"  {row_char} {row_vals}")
        
        print(f"\n[RESULT] Levenshtein distance = {dp[m][n]}")
    
    return dp[m][n]


def binary_distance(s1, s2, verbose=True):
    """
    Hamming distance - number of positions with different characters.
    Strings must be same length.
    """
    if verbose:
        print(f"\n[HAMMING DISTANCE]")
        print(f"String 1: '{s1}' (length {len(s1)})")
        print(f"String 2: '{s2}' (length {len(s2)})")
    
    if len(s1) != len(s2):
        if verbose:
            print(f"[ERROR] Strings must have equal length!")
        raise ValueError("Strings must have equal length for Hamming distance")
    
    if verbose:
        print(f"\n[Step 1] Compare position by position:")
    
    distance = 0
    for i, (c1, c2) in enumerate(zip(s1, s2)):
        match = c1 == c2
        if not match:
            distance += 1
        if verbose:
            symbol = "✓" if match else "✗"
            print(f"  Position {i}: '{c1}' vs '{c2}' -> {symbol}")
    
    if verbose:
        print(f"\n[RESULT] Hamming distance = {distance}")
    
    return distance


def overlap_coefficient(set1, set2, verbose=True):
    """
    Jaccard similarity: |A ∩ B| / |A ∪ B|
    Works with any iterables (converts to sets).
    """
    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)
    
    if verbose:
        print(f"\n[JACCARD SIMILARITY]")
        print(f"Set A: {set1}")
        print(f"Set B: {set2}")
    
    intersection = set1 & set2
    union = set1 | set2
    
    if verbose:
        print(f"\n[Step 1] Intersection (A ∩ B): {intersection}")
        print(f"[Step 2] Union (A ∪ B): {union}")
        print(f"[Step 3] |A ∩ B| = {len(intersection)}")
        print(f"[Step 4] |A ∪ B| = {len(union)}")
    
    if len(union) == 0:
        if verbose:
            print(f"\n[RESULT] Both sets empty, Jaccard = 1.0")
        return 1.0
    
    jaccard = len(intersection) / len(union)
    
    if verbose:
        print(f"\n[Step 5] Jaccard = {len(intersection)} / {len(union)} = {jaccard:.4f}")
        print(f"[RESULT] Jaccard similarity = {jaccard:.4f}")
    
    return jaccard


# =============================================================================
# SECTION 3: VECTOR SPACE MODELS
# =============================================================================

def term_counts(documents, verbose=True):
    """
    Bag of Words / Count Vectorizer.
    Returns term-document frequency matrix as dict of dicts.
    """
    if verbose:
        print(f"\n[BAG OF WORDS] {len(documents)} documents")
    
    vocab = set()
    doc_counts = []
    
    for i, doc in enumerate(documents):
        tokens = doc.lower().split() if isinstance(doc, str) else doc
        counts = Counter(tokens)
        doc_counts.append(counts)
        vocab.update(tokens)
    
    vocab = sorted(vocab)
    if verbose:
        print(f"  Vocabulary: {len(vocab)} unique terms")
        for i, counts in enumerate(doc_counts):
            top3 = counts.most_common(3)
            print(f"  Doc {i}: {len(counts)} terms, top3: {top3}")
    
    return doc_counts, vocab


def build_td_matrix(documents, verbose=True):
    """
    Build term-document matrix.
    Rows = terms, Columns = documents.
    """
    if verbose:
        print(f"\n[TERM-DOCUMENT MATRIX]")
    
    doc_counts, vocab = term_counts(documents, verbose=False)
    
    matrix = {}
    for term in vocab:
        matrix[term] = [counts.get(term, 0) for counts in doc_counts]
    
    if verbose:
        print(f"\n[Step 1] Vocabulary size: {len(vocab)}")
        print(f"[Step 2] Number of documents: {len(documents)}")
        print(f"\n[Step 3] Term-Document Matrix:")
        header = "Term".ljust(15) + "  ".join(f"D{i}" for i in range(len(documents)))
        print(header)
        print("-" * len(header))
        for term in vocab:
            row = term.ljust(15) + "  ".join(str(v).rjust(2) for v in matrix[term])
            print(row)
    
    return matrix, vocab


def compute_tfidf(documents, verbose=True):
    """
    Compute TF-IDF scores for a corpus.
    TF = term frequency in document
    IDF = log(N / df) where df = number of docs containing term
    """
    if verbose:
        print(f"\n[TF-IDF CALCULATION] {len(documents)} documents")
    
    N = len(documents)
    doc_counts, vocab = term_counts(documents, verbose=False)
    
    # Calculate document frequency for each term
    df = {}
    for term in vocab:
        df[term] = sum(1 for counts in doc_counts if term in counts)
    
    # Calculate IDF
    idf = {}
    for term in vocab:
        idf[term] = math.log(N / df[term]) if df[term] > 0 else 0
    
    if verbose:
        print(f"\n[Step 1-2] Document Frequency & IDF (top 20 terms):")
        print(f"{'Term':<15} {'df':>4} {'IDF':>8}")
        print("-" * 30)
        sorted_terms = sorted(vocab, key=lambda t: idf[t], reverse=True)[:20]
        for term in sorted_terms:
            print(f"{term:<15} {df[term]:>4} {idf[term]:>8.4f}")
    
    # Calculate TF-IDF for each document
    tfidf = []
    for i, counts in enumerate(doc_counts):
        doc_len = sum(counts.values())
        doc_tfidf = {}
        for term in vocab:
            tf = counts.get(term, 0) / doc_len if doc_len > 0 else 0
            doc_tfidf[term] = tf * idf[term]
        tfidf.append(doc_tfidf)
    
    if verbose:
        print(f"\n[Step 3] TF-IDF Matrix (top terms per doc):")
        for i, doc_tfidf in enumerate(tfidf):
            top_terms = sorted(doc_tfidf.items(), key=lambda x: x[1], reverse=True)[:5]
            terms_str = ", ".join(f"{t}:{s:.3f}" for t, s in top_terms if s > 0)
            print(f"  Doc {i}: {terms_str}")
    
    return tfidf, vocab, idf


def vector_similarity(v1, v2, verbose=True):
    """
    Cosine similarity between two vectors (as dicts or lists).
    cos(A,B) = (A·B) / (||A|| × ||B||)
    """
    if verbose:
        print(f"\n[COSINE SIMILARITY]")
    
    # Handle dict input
    if isinstance(v1, dict) and isinstance(v2, dict):
        all_keys = set(v1.keys()) | set(v2.keys())
        v1_list = [v1.get(k, 0) for k in all_keys]
        v2_list = [v2.get(k, 0) for k in all_keys]
        if verbose:
            print(f"Vector 1: {v1}")
            print(f"Vector 2: {v2}")
    else:
        v1_list, v2_list = list(v1), list(v2)
        if verbose:
            print(f"Vector 1: {v1_list}")
            print(f"Vector 2: {v2_list}")
    
    # Dot product
    dot = sum(a * b for a, b in zip(v1_list, v2_list))
    if verbose:
        print(f"\n[Step 1] Dot product A·B:")
        terms = [f"{a:.4f}×{b:.4f}" for a, b in zip(v1_list, v2_list)]
        print(f"  {' + '.join(terms[:5])}{'...' if len(terms) > 5 else ''}")
        print(f"  = {dot:.4f}")
    
    # Magnitudes
    mag1 = math.sqrt(sum(a * a for a in v1_list))
    mag2 = math.sqrt(sum(b * b for b in v2_list))
    
    if verbose:
        print(f"\n[Step 2] Magnitude ||A|| = sqrt(sum of squares) = {mag1:.4f}")
        print(f"[Step 3] Magnitude ||B|| = sqrt(sum of squares) = {mag2:.4f}")
    
    if mag1 == 0 or mag2 == 0:
        if verbose:
            print(f"\n[RESULT] Zero vector detected, similarity = 0")
        return 0.0
    
    similarity = dot / (mag1 * mag2)
    
    if verbose:
        print(f"\n[Step 4] Cosine = {dot:.4f} / ({mag1:.4f} × {mag2:.4f})")
        print(f"[RESULT] Cosine similarity = {similarity:.4f}")
    
    return similarity


def pairwise_scores(tfidf_vectors, labels=None, verbose=True):
    """
    Compute all pairwise cosine similarities.
    Returns list of (label1, label2, similarity).
    """
    n = len(tfidf_vectors)
    if labels is None:
        labels = [f"Doc{i}" for i in range(n)]
    
    if verbose:
        print(f"\n[PAIRWISE SIMILARITIES] {n*(n-1)//2} pairs")
    
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = vector_similarity(tfidf_vectors[i], tfidf_vectors[j], verbose=False)
            results.append((labels[i], labels[j], sim))
    
    if verbose and len(results) <= 20:
        for l1, l2, sim in results:
            print(f"  {l1} vs {l2}: {sim:.4f}")
    elif verbose:
        print(f"  (showing first 10 of {len(results)})")
        for l1, l2, sim in results[:10]:
            print(f"  {l1} vs {l2}: {sim:.4f}")
    
    return results


def rank_pairs(similarity_results, verbose=True):
    """
    Rank pairs by similarity (highest first).
    """
    if verbose:
        print(f"\n[RANKING PAIRS BY SIMILARITY]")
    
    ranked = sorted(similarity_results, key=lambda x: x[2], reverse=True)
    
    if verbose:
        print(f"\nRank  Pair                     Similarity")
        print("-" * 45)
        for rank, (l1, l2, sim) in enumerate(ranked, 1):
            print(f"{rank:4d}  {l1} vs {l2}".ljust(30) + f"{sim:.4f}")
    
    return ranked


# =============================================================================
# SECTION 4: N-GRAMS & LANGUAGE MODELS
# =============================================================================

def extract_ngrams(tokens, n, verbose=True):
    """
    Generate n-grams from a token list.
    """
    if isinstance(tokens, str):
        tokens = tokens.split()
    
    if verbose:
        print(f"\n[N-GRAM EXTRACTION]")
        print(f"Tokens: {tokens}")
        print(f"N = {n}")
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
        if verbose:
            print(f"  Position {i}: {ngram}")
    
    if verbose:
        print(f"\n[RESULT] {len(ngrams)} {n}-grams generated")
    
    return ngrams


def ngram_counts(corpus, n, verbose=True):
    """
    Count n-gram frequencies in a corpus.
    Corpus can be list of sentences (strings or token lists).
    """
    if verbose:
        print(f"\n[N-GRAM COUNTING]")
        print(f"Corpus size: {len(corpus)} sentences")
        print(f"N = {n}")
    
    counts = Counter()
    context_counts = Counter()  # for (n-1)-grams
    
    for i, sent in enumerate(corpus):
        tokens = sent.lower().split() if isinstance(sent, str) else sent
        # Add start/end tokens
        tokens = ['<s>'] * (n - 1) + tokens + ['</s>']
        
        if verbose:
            print(f"\n[Sentence {i}] {tokens}")
        
        ngrams = extract_ngrams(tokens, n, verbose=False)
        for ng in ngrams:
            counts[ng] += 1
            context = ng[:-1]
            context_counts[context] += 1
        
        if verbose:
            print(f"  {n}-grams: {ngrams}")
    
    if verbose:
        print(f"\n[RESULT] N-gram counts:")
        for ng, c in counts.most_common():
            print(f"  {ng}: {c}")
    
    return counts, context_counts


def estimate_prob(ngram, counts, context_counts, verbose=True):
    """
    Maximum Likelihood Estimation for n-gram probability.
    P(word | context) = count(ngram) / count(context)
    """
    if verbose:
        print(f"\n[MLE PROBABILITY]")
        print(f"N-gram: {ngram}")
    
    context = ngram[:-1]
    word = ngram[-1]
    
    ngram_count = counts.get(ngram, 0)
    context_count = context_counts.get(context, 0)
    
    if verbose:
        print(f"[Step 1] Context: {context}")
        print(f"[Step 2] Word: '{word}'")
        print(f"[Step 3] Count({ngram}) = {ngram_count}")
        print(f"[Step 4] Count({context}) = {context_count}")
    
    if context_count == 0:
        prob = 0.0
        if verbose:
            print(f"[RESULT] P({word}|{context}) = 0 (context not seen)")
    else:
        prob = ngram_count / context_count
        if verbose:
            print(f"[Step 5] P({word}|{context}) = {ngram_count}/{context_count} = {prob:.4f}")
    
    return prob


def sequence_probability(tokens, counts, context_counts, n, verbose=True):
    """
    Calculate probability of a sequence using chain rule.
    P(w1,w2,...,wn) = ∏ P(wi | wi-n+1...wi-1)
    """
    if isinstance(tokens, str):
        tokens = tokens.lower().split()
    
    if verbose:
        print(f"\n[SEQUENCE PROBABILITY]")
        print(f"Sequence: {tokens}")
        print(f"Using {n}-gram model")
    
    # Add padding
    tokens = ['<s>'] * (n - 1) + tokens + ['</s>']
    
    if verbose:
        print(f"[Step 1] Padded sequence: {tokens}")
    
    log_prob = 0.0
    prob_product = 1.0
    
    if verbose:
        print(f"\n[Step 2] Computing probabilities:")
    
    for i in range(n - 1, len(tokens)):
        ngram = tuple(tokens[i - n + 1:i + 1])
        p = estimate_prob(ngram, counts, context_counts, verbose=False)
        
        if verbose:
            context = ngram[:-1]
            word = ngram[-1]
            print(f"  P({word}|{context}) = {p:.4f}")
        
        if p > 0:
            log_prob += math.log(p)
            prob_product *= p
        else:
            log_prob = float('-inf')
            prob_product = 0.0
            if verbose:
                print(f"  [WARNING] Zero probability encountered!")
            break
    
    if verbose:
        print(f"\n[RESULT] P(sequence) = {prob_product:.6f}")
        if prob_product > 0:
            print(f"[RESULT] Log probability = {log_prob:.4f}")
    
    return prob_product, log_prob


def apply_smoothing(counts, context_counts, vocab_size, k=1, verbose=True):
    """
    Add-k (Laplace) smoothing for n-gram probabilities.
    P_smooth(w|context) = (count(ngram) + k) / (count(context) + k*V)
    """
    if verbose:
        print(f"\n[ADD-{k} SMOOTHING]")
        print(f"Vocabulary size V = {vocab_size}")
        print(f"Smoothing constant k = {k}")
    
    def smoothed_prob(ngram):
        context = ngram[:-1]
        ngram_count = counts.get(ngram, 0)
        context_count = context_counts.get(context, 0)
        
        numerator = ngram_count + k
        denominator = context_count + k * vocab_size
        prob = numerator / denominator
        
        if verbose:
            word = ngram[-1]
            print(f"\n  P_smooth({word}|{context}):")
            print(f"    = (count({ngram}) + {k}) / (count({context}) + {k}*{vocab_size})")
            print(f"    = ({ngram_count} + {k}) / ({context_count} + {k*vocab_size})")
            print(f"    = {numerator} / {denominator}")
            print(f"    = {prob:.6f}")
        
        return prob
    
    return smoothed_prob


def compute_perplexity(test_tokens, counts, context_counts, n, vocab_size=None, smoothing_k=0, verbose=True):
    """
    Compute perplexity of a test sequence.
    PP = 2^(-1/N * sum(log2(P(wi|context))))
    Or equivalently: PP = exp(-1/N * sum(ln(P(wi|context))))
    """
    if isinstance(test_tokens, str):
        test_tokens = test_tokens.lower().split()
    
    if verbose:
        print(f"\n[PERPLEXITY CALCULATION]")
        print(f"Test sequence: {test_tokens}")
        print(f"Using {n}-gram model")
        if smoothing_k > 0:
            print(f"With add-{smoothing_k} smoothing")
    
    # Add padding
    tokens = ['<s>'] * (n - 1) + test_tokens + ['</s>']
    N = len(tokens) - (n - 1)  # number of predictions
    
    if verbose:
        print(f"[Step 1] Padded: {tokens}")
        print(f"[Step 2] Number of predictions N = {N}")
    
    log_prob_sum = 0.0
    
    if smoothing_k > 0 and vocab_size:
        smooth_fn = apply_smoothing(counts, context_counts, vocab_size, smoothing_k, verbose=False)
    
    if verbose:
        print(f"\n[Step 3] Computing log probabilities:")
    
    for i in range(n - 1, len(tokens)):
        ngram = tuple(tokens[i - n + 1:i + 1])
        
        if smoothing_k > 0 and vocab_size:
            p = smooth_fn(ngram)
        else:
            p = estimate_prob(ngram, counts, context_counts, verbose=False)
        
        if verbose:
            print(f"  {ngram}: P = {p:.6f}, log2(P) = {math.log2(p) if p > 0 else '-inf':.4f}")
        
        if p > 0:
            log_prob_sum += math.log2(p)
        else:
            log_prob_sum = float('-inf')
            break
    
    if log_prob_sum == float('-inf'):
        perplexity = float('inf')
    else:
        avg_log_prob = log_prob_sum / N
        perplexity = 2 ** (-avg_log_prob)
    
    if verbose:
        print(f"\n[Step 4] Sum of log2 probabilities = {log_prob_sum:.4f}")
        print(f"[Step 5] Average = {log_prob_sum/N:.4f}")
        print(f"[Step 6] Perplexity = 2^(-avg) = 2^({-log_prob_sum/N:.4f})")
        print(f"[RESULT] Perplexity = {perplexity:.4f}")
    
    return perplexity


# =============================================================================
# SECTION 5: INVERTED INDEX & SEARCH
# =============================================================================

def build_index(documents, labels=None, verbose=True):
    """
    Build inverted index from documents.
    Returns {term: set(doc_ids)}
    """
    if labels is None:
        labels = [f"Doc{i}" for i in range(len(documents))]
    
    if verbose:
        print(f"\n[BUILDING INVERTED INDEX]")
        print(f"Number of documents: {len(documents)}")
    
    index = defaultdict(set)
    
    for i, doc in enumerate(documents):
        tokens = doc.lower().split() if isinstance(doc, str) else doc
        if verbose:
            print(f"\n[{labels[i]}] Tokens: {tokens}")
        
        for token in set(tokens):  # unique tokens only
            index[token].add(labels[i])
            if verbose:
                print(f"  Added '{token}' -> {labels[i]}")
    
    if verbose:
        print(f"\n[RESULT] Inverted Index:")
        for term in sorted(index.keys()):
            print(f"  '{term}': {sorted(index[term])}")
    
    return dict(index)


def merge_and(postings1, postings2, verbose=True):
    """
    AND merge algorithm for two posting lists.
    Returns intersection of documents.
    """
    if verbose:
        print(f"\n[MERGE AND ALGORITHM]")
        print(f"Postings 1: {sorted(postings1)}")
        print(f"Postings 2: {sorted(postings2)}")
    
    # Convert to sorted lists for merge
    p1 = sorted(list(postings1))
    p2 = sorted(list(postings2))
    
    result = []
    i, j = 0, 0
    
    if verbose:
        print(f"\n[Step-by-step merge]")
    
    while i < len(p1) and j < len(p2):
        if verbose:
            print(f"  Comparing p1[{i}]='{p1[i]}' with p2[{j}]='{p2[j]}'", end=" -> ")
        
        if p1[i] == p2[j]:
            result.append(p1[i])
            if verbose:
                print(f"MATCH! Add '{p1[i]}'")
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            if verbose:
                print(f"Advance p1")
            i += 1
        else:
            if verbose:
                print(f"Advance p2")
            j += 1
    
    if verbose:
        print(f"\n[RESULT] AND result: {result}")
    
    return set(result)


def merge_or(postings1, postings2, verbose=True):
    """
    OR merge algorithm for two posting lists.
    Returns union of documents.
    """
    if verbose:
        print(f"\n[MERGE OR ALGORITHM]")
        print(f"Postings 1: {sorted(postings1)}")
        print(f"Postings 2: {sorted(postings2)}")
    
    result = set(postings1) | set(postings2)
    
    if verbose:
        print(f"[RESULT] OR result: {sorted(result)}")
    
    return result


def query_index(index, terms, op='AND', verbose=True):
    """
    Boolean search on inverted index.
    """
    if isinstance(terms, str):
        terms = terms.lower().split()
    else:
        terms = [t.lower() for t in terms]
    
    if verbose:
        print(f"\n[BOOLEAN SEARCH]")
        print(f"Query terms: {terms}")
        print(f"Operator: {op}")
    
    if not terms:
        return set()
    
    # Get postings for first term
    result = index.get(terms[0], set())
    if verbose:
        print(f"\n[Step 1] Postings for '{terms[0]}': {sorted(result)}")
    
    # Merge with remaining terms
    for i, term in enumerate(terms[1:], 2):
        postings = index.get(term, set())
        if verbose:
            print(f"\n[Step {i}] Postings for '{term}': {sorted(postings)}")
        
        if op.upper() == 'AND':
            result = merge_and(result, postings, verbose)
        else:
            result = merge_or(result, postings, verbose)
    
    if verbose:
        print(f"\n[FINAL RESULT] {sorted(result)}")
    
    return result


# =============================================================================
# SECTION 6: NEURAL NETWORK GATES
# =============================================================================

def step_function(x, verbose=True):
    """Binary step activation function."""
    result = 1 if x >= 0 else 0
    if verbose:
        print(f"  step({x:.4f}) = {result} (threshold at 0)")
    return result


def perceptron(inputs, weights, bias, verbose=True):
    """
    Single perceptron computation.
    output = step(sum(inputs * weights) + bias)
    """
    if verbose:
        print(f"\n[PERCEPTRON]")
        print(f"Inputs: {inputs}")
        print(f"Weights: {weights}")
        print(f"Bias: {bias}")
    
    # Weighted sum
    weighted_sum = sum(x * w for x, w in zip(inputs, weights))
    
    if verbose:
        terms = [f"{x}*{w}" for x, w in zip(inputs, weights)]
        print(f"\n[Step 1] Weighted sum = {' + '.join(terms)}")
        print(f"        = {weighted_sum}")
    
    # Add bias
    total = weighted_sum + bias
    
    if verbose:
        print(f"\n[Step 2] Add bias: {weighted_sum} + {bias} = {total}")
    
    # Apply activation
    if verbose:
        print(f"\n[Step 3] Apply step function:")
    output = step_function(total, verbose)
    
    if verbose:
        print(f"\n[RESULT] Output = {output}")
    
    return output


def gate_and(x1, x2, verbose=True):
    """
    AND gate using perceptron.
    Weights = [1, 1], Bias = -1.5
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"[AND GATE] Input: ({x1}, {x2})")
        print(f"{'='*50}")
        print(f"Using weights [1, 1] and bias -1.5")
        print(f"Decision boundary: x1 + x2 >= 1.5")
    
    result = perceptron([x1, x2], [1, 1], -1.5, verbose)
    
    if verbose:
        print(f"\n[AND({x1}, {x2})] = {result}")
    
    return result


def gate_or(x1, x2, verbose=True):
    """
    OR gate using perceptron.
    Weights = [1, 1], Bias = -0.5
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"[OR GATE] Input: ({x1}, {x2})")
        print(f"{'='*50}")
        print(f"Using weights [1, 1] and bias -0.5")
        print(f"Decision boundary: x1 + x2 >= 0.5")
    
    result = perceptron([x1, x2], [1, 1], -0.5, verbose)
    
    if verbose:
        print(f"\n[OR({x1}, {x2})] = {result}")
    
    return result


def gate_nand(x1, x2, verbose=True):
    """NAND gate: NOT(AND(x1, x2))"""
    if verbose:
        print(f"\n[NAND GATE] Input: ({x1}, {x2})")
        print(f"Using weights [-1, -1] and bias 1.5")
    
    result = perceptron([x1, x2], [-1, -1], 1.5, verbose)
    return result


def gate_xor(x1, x2, verbose=True):
    """
    XOR gate using 2-layer network.
    XOR = AND(OR(x1,x2), NAND(x1,x2))
    Cannot be solved with single perceptron!
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"[XOR GATE] Input: ({x1}, {x2})")
        print(f"{'='*50}")
        print(f"XOR requires 2 layers (not linearly separable)")
        print(f"XOR(x1, x2) = AND(OR(x1, x2), NAND(x1, x2))")
    
    # Layer 1: OR and NAND
    if verbose:
        print(f"\n[Layer 1 - Neuron 1: OR]")
    or_result = gate_or(x1, x2, verbose=False)
    if verbose:
        print(f"  OR({x1}, {x2}) = {or_result}")
    
    if verbose:
        print(f"\n[Layer 1 - Neuron 2: NAND]")
    nand_result = gate_nand(x1, x2, verbose=False)
    if verbose:
        print(f"  NAND({x1}, {x2}) = {nand_result}")
    
    # Layer 2: AND
    if verbose:
        print(f"\n[Layer 2: AND of Layer 1 outputs]")
    result = gate_and(or_result, nand_result, verbose=False)
    
    if verbose:
        print(f"  AND({or_result}, {nand_result}) = {result}")
        print(f"\n[XOR({x1}, {x2})] = {result}")
    
    return result


def show_truth_table(gate_fn, gate_name, verbose=True):
    """Display truth table for a gate."""
    if verbose:
        print(f"\n[TRUTH TABLE for {gate_name}]")
        print(f"{'x1':^4} | {'x2':^4} | {gate_name:^6}")
        print("-" * 20)
    
    results = []
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            result = gate_fn(x1, x2, verbose=False)
            results.append((x1, x2, result))
            if verbose:
                print(f"{x1:^4} | {x2:^4} | {result:^6}")
    
    return results


# =============================================================================
# SECTION 7: GRAPH ANALYTICS
# =============================================================================

def degree_centrality(adj_matrix, labels=None, verbose=True):
    """
    Compute degree centrality for all nodes.
    Degree centrality = degree / (n-1) where n = number of nodes
    """
    n = len(adj_matrix)
    if labels is None:
        labels = list(range(n))  # Use integer indices by default
    
    if verbose:
        print(f"\n[DEGREE CENTRALITY]")
        print(f"Number of nodes: {n}")
        print(f"Adjacency matrix:")
        for i, row in enumerate(adj_matrix):
            print(f"  {labels[i]}: {row}")
    
    centralities = {}
    
    if verbose:
        print(f"\n[Computing degrees]")
    
    for i in range(n):
        degree = sum(adj_matrix[i])  # sum of row for undirected
        centrality = degree / (n - 1) if n > 1 else 0
        centralities[labels[i]] = centrality
        
        if verbose:
            print(f"  {labels[i]}: degree = {degree}, centrality = {degree}/{n-1} = {centrality:.4f}")
    
    if verbose:
        print(f"\n[RESULT] Degree Centrality:")
        for node, c in sorted(centralities.items(), key=lambda x: -x[1]):
            print(f"  {node}: {c:.4f}")
    
    return centralities


def closeness_centrality(adj_matrix, labels=None, verbose=True):
    """
    Compute closeness centrality using shortest paths (BFS).
    Closeness = (n-1) / sum(shortest paths to all other nodes)
    """
    n = len(adj_matrix)
    if labels is None:
        labels = list(range(n))  # Use integer indices by default
    
    if verbose:
        print(f"\n[CLOSENESS CENTRALITY]")
        print(f"Number of nodes: {n}")
    
    def bfs_distances(start):
        """BFS to find shortest paths from start node."""
        distances = [-1] * n
        distances[start] = 0
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            for neighbor in range(n):
                if adj_matrix[current][neighbor] > 0 and distances[neighbor] == -1:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        
        return distances
    
    centralities = {}
    
    for i in range(n):
        distances = bfs_distances(i)
        
        if verbose:
            print(f"\n[{labels[i]}] Shortest path distances: {distances}")
        
        # Sum of distances to reachable nodes
        reachable = [d for d in distances if d > 0]
        total_dist = sum(reachable)
        
        if total_dist > 0:
            centrality = len(reachable) / total_dist
        else:
            centrality = 0.0
        
        centralities[labels[i]] = centrality
        
        if verbose:
            print(f"  Sum of distances: {total_dist}")
            print(f"  Closeness = {len(reachable)}/{total_dist} = {centrality:.4f}")
    
    if verbose:
        print(f"\n[RESULT] Closeness Centrality:")
        for node, c in sorted(centralities.items(), key=lambda x: -x[1]):
            print(f"  {node}: {c:.4f}")
    
    return centralities


def betweenness_centrality(adj_matrix, labels=None, verbose=True):
    """
    Compute betweenness centrality.
    BC(v) = sum over all pairs (s,t): (# shortest paths through v) / (# total shortest paths)
    """
    n = len(adj_matrix)
    if labels is None:
        labels = [f"N{i}" for i in range(n)]
    
    if verbose:
        print(f"\n[BETWEENNESS CENTRALITY]")
        print(f"Number of nodes: {n}")
    
    # Initialize betweenness counts
    betweenness = [0.0] * n
    
    for s in range(n):
        # BFS from source s
        dist = [-1] * n
        dist[s] = 0
        num_paths = [0] * n
        num_paths[s] = 1
        parents = [[] for _ in range(n)]
        queue = [s]
        order = []
        
        while queue:
            v = queue.pop(0)
            order.append(v)
            for w in range(n):
                if adj_matrix[v][w] > 0:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        num_paths[w] += num_paths[v]
                        parents[w].append(v)
        
        # Accumulate dependencies
        dependency = [0.0] * n
        for w in reversed(order):
            for v in parents[w]:
                dependency[v] += (num_paths[v] / num_paths[w]) * (1 + dependency[w])
            if w != s:
                betweenness[w] += dependency[w]
    
    centralities = {}
    for i in range(n):
        centralities[labels[i]] = betweenness[i] / 2  # undirected graph
    
    if verbose:
        print(f"\n[RESULT] Betweenness Centrality:")
        for node, c in sorted(centralities.items(), key=lambda x: -x[1]):
            print(f"  {node}: {c:.4f}")
    
    return centralities


def pagerank(adj_matrix, damping=0.85, iterations=20, labels=None, verbose=True):
    """
    PageRank algorithm.
    PR(i) = (1-d)/n + d * sum(PR(j)/L(j)) for all j linking to i
    """
    n = len(adj_matrix)
    if labels is None:
        labels = list(range(n))  # Use integer indices by default
    
    if verbose:
        print(f"\n[PAGERANK ALGORITHM]")
        print(f"Number of nodes: {n}")
        print(f"Damping factor d = {damping}")
        print(f"Iterations: {iterations}")
    
    # Initialize PageRank
    pr = [1.0 / n] * n
    
    # Compute out-degrees
    out_degree = [sum(row) for row in adj_matrix]
    
    if verbose:
        print(f"\n[Initial] PR = {[f'{x:.4f}' for x in pr]}")
        print(f"[Out-degrees] {out_degree}")
    
    for iteration in range(iterations):
        new_pr = [0.0] * n
        
        for i in range(n):
            # Sum contributions from incoming links
            incoming = 0.0
            for j in range(n):
                if adj_matrix[j][i] > 0 and out_degree[j] > 0:
                    incoming += pr[j] / out_degree[j]
            
            new_pr[i] = (1 - damping) / n + damping * incoming
        
        pr = new_pr
        
        if verbose and (iteration < 3 or iteration == iterations - 1):
            print(f"[Iter {iteration + 1}] PR = {[f'{x:.4f}' for x in pr]}")
    
    result = {labels[i]: pr[i] for i in range(n)}
    
    if verbose:
        print(f"\n[RESULT] PageRank:")
        for node, score in sorted(result.items(), key=lambda x: -x[1]):
            print(f"  {node}: {score:.4f}")
    
    return result


def clustering_coefficient(adj_matrix, labels=None, verbose=True):
    """
    Compute local clustering coefficient for each node.
    CC(v) = 2 * (edges between neighbors) / (degree * (degree - 1))
    """
    n = len(adj_matrix)
    if labels is None:
        labels = list(range(n))  # Use integer indices by default
    
    if verbose:
        print(f"\n[CLUSTERING COEFFICIENT]")
        print(f"Number of nodes: {n}")
    
    coefficients = {}
    
    for i in range(n):
        # Find neighbors
        neighbors = [j for j in range(n) if adj_matrix[i][j] > 0]
        degree = len(neighbors)
        
        if degree < 2:
            cc = 0.0
        else:
            # Count edges between neighbors
            edges_between = 0
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if adj_matrix[neighbors[j]][neighbors[k]] > 0:
                        edges_between += 1
            
            max_edges = degree * (degree - 1) / 2
            cc = edges_between / max_edges
        
        coefficients[labels[i]] = cc
        
        if verbose:
            print(f"\n[{labels[i]}]")
            print(f"  Neighbors: {[labels[j] for j in neighbors]}")
            print(f"  Degree: {degree}")
            if degree >= 2:
                print(f"  Edges between neighbors: {edges_between}")
                print(f"  Max possible: {max_edges:.0f}")
            print(f"  Clustering coefficient: {cc:.4f}")
    
    if verbose:
        avg_cc = sum(coefficients.values()) / len(coefficients)
        print(f"\n[RESULT] Average clustering coefficient: {avg_cc:.4f}")
    
    return coefficients


# =============================================================================
# SECTION 8: EVALUATION METRICS & EMBEDDINGS
# =============================================================================

def split_data(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, verbose=True):
    """
    Split data into train/validation/test sets.
    """
    import random
    
    if verbose:
        print(f"\n[DATA SPLITTING]")
        print(f"Total samples: {len(X)}")
        print(f"Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # Create indices and shuffle
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    
    if verbose:
        print(f"\n[RESULT]")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Validation set: {len(X_val)} samples")
        print(f"  Test set: {len(X_test)} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def eval_metrics(y_true, y_pred, verbose=True):
    """
    Compute classification metrics: accuracy, precision, recall, F1.
    For binary classification.
    """
    if verbose:
        print(f"\n[CLASSIFICATION METRICS]")
        print(f"True labels: {y_true}")
        print(f"Predictions: {y_pred}")
    
    # Confusion matrix values
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    if verbose:
        print(f"\n[Step 1] Confusion Matrix:")
        print(f"  True Positives (TP): {tp}")
        print(f"  True Negatives (TN): {tn}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
    
    # Calculate metrics
    accuracy = (tp + tn) / len(y_true) if y_true else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    if verbose:
        print(f"\n[Step 2] Calculating metrics:")
        print(f"  Accuracy = (TP + TN) / Total = ({tp} + {tn}) / {len(y_true)} = {accuracy:.4f}")
        print(f"  Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.4f}")
        print(f"  Recall = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.4f}")
        print(f"  F1 = 2 * P * R / (P + R) = 2 * {precision:.4f} * {recall:.4f} / ({precision:.4f} + {recall:.4f}) = {f1:.4f}")
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    if verbose:
        print(f"\n[RESULT]")
        for metric, value in results.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
    
    return results


def aggregate_tokens(tokens, word_vectors, verbose=True):
    """
    Create sentence embedding by averaging word vectors.
    word_vectors is a dict: {word: vector}
    """
    if verbose:
        print(f"\n[SENTENCE EMBEDDING]")
        print(f"Tokens: {tokens}")
    
    vectors = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in word_vectors:
            vectors.append(word_vectors[token_lower])
            if verbose:
                vec = word_vectors[token_lower]
                print(f"  '{token}': found, dim={len(vec)}")
        else:
            if verbose:
                print(f"  '{token}': NOT FOUND, skipped")
    
    if not vectors:
        if verbose:
            print(f"[WARNING] No vectors found!")
        return None
    
    # Average the vectors
    dim = len(vectors[0])
    avg_vec = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    
    if verbose:
        print(f"\n[Step] Average {len(vectors)} vectors of dimension {dim}")
        print(f"[RESULT] Sentence embedding dimension: {len(avg_vec)}")
    
    return avg_vec


def aggregate_sentences(sentence_embeddings, verbose=True):
    """
    Create document embedding by averaging sentence embeddings.
    """
    if verbose:
        print(f"\n[DOCUMENT EMBEDDING]")
        print(f"Number of sentences: {len(sentence_embeddings)}")
    
    valid_embeddings = [e for e in sentence_embeddings if e is not None]
    
    if not valid_embeddings:
        if verbose:
            print(f"[WARNING] No valid embeddings!")
        return None
    
    dim = len(valid_embeddings[0])
    avg_vec = [sum(e[i] for e in valid_embeddings) / len(valid_embeddings) for i in range(dim)]
    
    if verbose:
        print(f"\n[Step] Average {len(valid_embeddings)} sentence embeddings")
        print(f"[RESULT] Document embedding dimension: {len(avg_vec)}")
    
    return avg_vec


def compute_doc_embedding(text, model, verbose=False):
    """
    Compute document embedding by averaging sentence averages.
    Pipeline: Text -> Sentences -> Word Vectors -> Sentence Embeddings -> Document Embedding
    """
    import nltk
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download('punkt', quiet=True)
        sentences = nltk.sent_tokenize(text)
        
    if not sentences:
        return None
    
    sentence_embeddings = []
    for sent in sentences:
        tokens = prepare_text(sent, verbose=False).split()
        if not tokens:
            continue
        # Use existing aggregate_tokens logic but suppressed verbosity for cleaner output
        sent_vec = aggregate_tokens(tokens, model, verbose=False)
        if sent_vec is not None:
            sentence_embeddings.append(sent_vec)
            
    if not sentence_embeddings:
        return None
        
    return aggregate_sentences(sentence_embeddings, verbose=verbose)


def load_word_vectors(path, fallback_model='glove-wiki-gigaword-50', verbose=True):
    """
    Load KeyedVectors from path or download fallback model.
    """
    import os
    from gensim.models import KeyedVectors
    import gensim.downloader as api
    
    if os.path.exists(path):
        if verbose:
            print(f"Loading local model from {path}...")
        return KeyedVectors.load_word2vec_format(path, binary=True)
    elif fallback_model:
        if verbose:
            print(f"File '{path}' not found. Downloading fallback model '{fallback_model}'...")
        return api.load(fallback_model)
    else:
        raise FileNotFoundError(f"Model file {path} not found and no fallback specified.")


def find_nearest(target_vec, all_vectors, labels, k=5, verbose=True):
    """
    Find k nearest neighbors by cosine similarity.
    """
    if verbose:
        print(f"\n[FINDING {k} NEAREST NEIGHBORS]")
        print(f"Number of candidates: {len(all_vectors)}")
    
    similarities = []
    for i, vec in enumerate(all_vectors):
        sim = vector_similarity(target_vec, vec, verbose=False)
        similarities.append((labels[i], sim))
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top k (excluding self if similarity is 1.0)
    top_k = []
    for label, sim in similarities:
        if len(top_k) < k:
            top_k.append((label, sim))
    
    if verbose:
        print(f"\n[RESULT] Top {k} most similar:")
        for rank, (label, sim) in enumerate(top_k, 1):
            print(f"  {rank}. {label}: {sim:.4f}")
    
    return top_k


# =============================================================================
# SECTION 9: COMPUTATION GRAPHS (Forward/Backward Pass)
# =============================================================================

def forward_pass(inputs, operations, verbose=True):
    """
    Simple computation graph forward pass.
    operations: list of (op_type, operand_indices) or (op_type, operand_indices, params)
    Returns: final output and cache for backward pass
    """
    if verbose:
        print(f"\n[FORWARD PASS]")
        print(f"Inputs: {inputs}")
    
    values = list(inputs)
    cache = {'inputs': inputs, 'operations': operations, 'intermediates': []}
    
    for i, op in enumerate(operations):
        op_type = op[0]
        operands = op[1]
        params = op[2] if len(op) > 2 else None
        
        op_values = [values[idx] for idx in operands]
        
        if op_type == 'add':
            result = sum(op_values)
            if verbose:
                print(f"[Op {i}] ADD({op_values}) = {result}")
        elif op_type == 'mul':
            result = 1
            for v in op_values:
                result *= v
            if verbose:
                print(f"[Op {i}] MUL({op_values}) = {result}")
        elif op_type == 'sub':
            result = op_values[0] - op_values[1]
            if verbose:
                print(f"[Op {i}] SUB({op_values}) = {result}")
        elif op_type == 'square':
            result = op_values[0] ** 2
            if verbose:
                print(f"[Op {i}] SQUARE({op_values[0]}) = {result}")
        elif op_type == 'sigmoid':
            result = 1 / (1 + math.exp(-op_values[0]))
            if verbose:
                print(f"[Op {i}] SIGMOID({op_values[0]}) = {result:.4f}")
        elif op_type == 'relu':
            result = max(0, op_values[0])
            if verbose:
                print(f"[Op {i}] RELU({op_values[0]}) = {result}")
        else:
            result = op_values[0]  # identity
        
        cache['intermediates'].append({
            'op_type': op_type,
            'inputs': op_values,
            'output': result
        })
        values.append(result)
    
    if verbose:
        print(f"\n[RESULT] Output = {values[-1]}")
    
    return values[-1], cache


def backward_pass(output_grad, cache, verbose=True):
    """
    Compute gradients via backpropagation.
    Uses cached values from forward pass.
    """
    if verbose:
        print(f"\n[BACKWARD PASS]")
        print(f"Output gradient: {output_grad}")
    
    n_inputs = len(cache['inputs'])
    grads = [0.0] * n_inputs
    
    # Work backwards through operations
    intermediates = cache['intermediates']
    current_grad = output_grad
    
    for i in range(len(intermediates) - 1, -1, -1):
        inter = intermediates[i]
        op_type = inter['op_type']
        op_inputs = inter['inputs']
        
        if verbose:
            print(f"\n[Op {i} backward] {op_type}")
            print(f"  Incoming gradient: {current_grad}")
        
        if op_type == 'add':
            # Gradient flows equally to all inputs
            local_grads = [current_grad] * len(op_inputs)
            if verbose:
                print(f"  Gradients to inputs: {local_grads}")
        elif op_type == 'mul':
            # Product rule
            local_grads = []
            for j, inp in enumerate(op_inputs):
                other = 1
                for k, v in enumerate(op_inputs):
                    if k != j:
                        other *= v
                local_grads.append(current_grad * other)
            if verbose:
                print(f"  Gradients to inputs: {local_grads}")
        elif op_type == 'sub':
            local_grads = [current_grad, -current_grad]
            if verbose:
                print(f"  Gradients to inputs: {local_grads}")
        elif op_type == 'square':
            local_grads = [2 * op_inputs[0] * current_grad]
            if verbose:
                print(f"  d/dx(x^2) = 2x = 2*{op_inputs[0]} = {2*op_inputs[0]}")
                print(f"  Gradient: {local_grads[0]}")
        elif op_type == 'sigmoid':
            s = inter['output']
            local_grads = [s * (1 - s) * current_grad]
            if verbose:
                print(f"  sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))")
                print(f"  = {s:.4f} * {1-s:.4f} = {s*(1-s):.4f}")
                print(f"  Gradient: {local_grads[0]:.4f}")
        elif op_type == 'relu':
            local_grads = [current_grad if op_inputs[0] > 0 else 0]
            if verbose:
                print(f"  relu'(x) = 1 if x > 0 else 0")
                print(f"  Gradient: {local_grads[0]}")
        else:
            local_grads = [current_grad]
        
        # For now, just accumulate to first inputs (simplified)
        for j, g in enumerate(local_grads):
            if j < len(grads):
                grads[j] += g
    
    if verbose:
        print(f"\n[RESULT] Input gradients: {grads}")
    
    return grads
