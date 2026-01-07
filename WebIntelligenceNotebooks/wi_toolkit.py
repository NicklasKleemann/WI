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
        # Canonical Porter Stemmer rules for each step
        STEP_RULES = {
            '1a': [('sses', 'ss'), ('ies', 'i'), ('ss', 'ss'), ('s', '')],
            '1b': [('eed', 'ee'), ('ed', ''), ('ing', '')],
            '1c': [('y', 'i')],
            '2': [
                ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'), ('anci', 'ance'),
                ('izer', 'ize'), ('abli', 'able'), ('alli', 'al'), ('entli', 'ent'),
                ('eli', 'e'), ('ousli', 'ous'), ('ization', 'ize'), ('ation', 'ate'),
                ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
                ('ousness', 'ous'), ('aliti', 'al'), ('iviti', 'ive'), ('biliti', 'ble')
            ],
            '3': [
                ('icate', 'ic'), ('ative', ''), ('alize', 'al'), ('iciti', 'ic'),
                ('ical', 'ic'), ('ful', ''), ('ness', '')
            ],
            '4': [
                ('al', ''), ('ance', ''), ('ence', ''), ('er', ''), ('ic', ''),
                ('able', ''), ('ible', ''), ('ant', ''), ('ement', ''), ('ment', ''),
                ('ent', ''), ('ion', ''), ('ou', ''), ('ism', ''), ('ate', ''),
                ('iti', ''), ('ous', ''), ('ive', ''), ('ize', '')
            ],
            '5a': [('e', '')],
            '5b': [('ll', 'l')]
        }
        
        def _find_applied_rule(self, step_name, prev_stem, new_stem):
            """Find which canonical rule was applied based on the transformation."""
            if prev_stem == new_stem:
                return None
                
            rules = self.STEP_RULES.get(step_name, [])
            for old_suffix, new_suffix in rules:
                # Check if prev_stem ends with old_suffix and new_stem ends with new_suffix
                if prev_stem.endswith(old_suffix):
                    # Verify the transformation matches
                    base = prev_stem[:-len(old_suffix)] if old_suffix else prev_stem
                    expected_new = base + new_suffix
                    if expected_new == new_stem:
                        return f"(m>0) {old_suffix.upper()} -> {new_suffix.upper() if new_suffix else '∅'}"
            
            # Fallback: deduce rule from suffix change
            i = 0
            min_len = min(len(prev_stem), len(new_stem))
            while i < min_len and prev_stem[i] == new_stem[i]:
                i += 1
            old_suffix = prev_stem[i:]
            new_suffix = new_stem[i:]
            return f"{old_suffix.upper()} -> {new_suffix.upper() if new_suffix else '∅'}"
        
        def stem(self, word):
            stem = word.lower()
            print(f"Original: {word}")
            
            def _apply_step(name, func, current_stem):
                prev = current_stem
                m = self._measure(current_stem) if hasattr(self, '_measure') else "?"
                
                new_stem = func(current_stem)
                
                if new_stem != prev:
                    rule = self._find_applied_rule(name, prev, new_stem)
                    print(f"Step {name:<3} : {prev:<15} -> {new_stem:<15} (m={m}, Rule: {rule})")
                else:
                    print(f"Step {name:<3} : {prev:<15} -> {new_stem:<15} (m={m})")
                return new_stem

            stem = _apply_step("1a", self._step1a, stem)
            stem = _apply_step("1b", self._step1b, stem)
            stem = _apply_step("1c", self._step1c, stem)
            stem = _apply_step("2", self._step2, stem)
            stem = _apply_step("3", self._step3, stem)
            stem = _apply_step("4", self._step4, stem)
            stem = _apply_step("5a", self._step5a, stem)
            stem = _apply_step("5b", self._step5b, stem)
            
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
    Shows the full DP matrix and traces back the operations.
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
        
        # Traceback to find the operations
        print(f"\n[Step 5] Traceback - Operations to transform '{s1}' → '{s2}':")
        operations = _traceback_operations(dp, s1, s2)
        current = s1
        for step_num, (op_type, details, result) in enumerate(operations, 1):
            print(f"  Step {step_num}: {op_type:12} {details:30} → '{result}'")
            current = result
        
        if not operations:
            print(f"  No operations needed - strings are identical!")
        
        print(f"\n[RESULT] Levenshtein distance = {dp[m][n]}")
    
    return dp[m][n]


# Alias for standard name
levenshtein_distance = edit_distance


def _traceback_operations(dp, s1, s2):
    """
    Trace back through the DP matrix to find the sequence of operations.
    Returns a list of (operation_type, details, resulting_string) tuples.
    """
    operations = []
    i, j = len(s1), len(s2)
    
    # Build the result by working backwards, then reverse
    # We'll track what the string looks like at each step
    current = list(s1)
    trace = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            # Characters match - no operation needed
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            # Substitution
            trace.append(('SUBSTITUTE', i-1, s1[i-1], s2[j-1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            # Insertion
            trace.append(('INSERT', i, s2[j-1]))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            # Deletion
            trace.append(('DELETE', i-1, s1[i-1]))
            i -= 1
        else:
            # Fallback (shouldn't happen with correct DP)
            break
    
    # Reverse to get operations in forward order
    trace.reverse()
    
    # Now apply operations in forward order to show the transformation
    current = list(s1)
    offset = 0  # Track position shifts due to insertions/deletions
    
    for op in trace:
        if op[0] == 'SUBSTITUTE':
            pos, old_char, new_char = op[1], op[2], op[3]
            actual_pos = pos + offset
            current[actual_pos] = new_char
            details = f"'{old_char}' → '{new_char}' at position {pos}"
            operations.append((op[0], details, ''.join(current)))
        elif op[0] == 'INSERT':
            pos, char = op[1], op[2]
            actual_pos = pos + offset
            current.insert(actual_pos, char)
            offset += 1
            details = f"'{char}' at position {pos}"
            operations.append((op[0], details, ''.join(current)))
        elif op[0] == 'DELETE':
            pos, char = op[1], op[2]
            actual_pos = pos + offset
            del current[actual_pos]
            offset -= 1
            details = f"'{char}' from position {pos}"
            operations.append((op[0], details, ''.join(current)))
    
    return operations


def hamming_distance(s1, s2, verbose=True):
    """
    Hamming distance - number of positions with different characters.
    Strings must be same length.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[HAMMING DISTANCE]")
        print(f"{'='*60}")
        print(f"\nFormula: H(s1, s2) = number of positions where s1[i] ≠ s2[i]")
        print(f"\nInput:")
        print(f"  String 1: '{s1}' (length {len(s1)})")
        print(f"  String 2: '{s2}' (length {len(s2)})")
    
    if len(s1) != len(s2):
        if verbose:
            print(f"\n[ERROR] Strings must have equal length!")
        raise ValueError("Strings must have equal length for Hamming distance")
    
    if verbose:
        print(f"\n[Step 1] Align strings and compare position by position:")
        print(f"")
        print(f"  Position:  {' '.join(str(i) for i in range(len(s1)))}")
        print(f"  String 1:  {' '.join(s1)}")
        print(f"  String 2:  {' '.join(s2)}")
        print(f"")
    
    distance = 0
    mismatches = []
    for i, (c1, c2) in enumerate(zip(s1, s2)):
        match = c1 == c2
        if not match:
            distance += 1
            mismatches.append(i)
        if verbose:
            symbol = "✓ (match)" if match else "✗ (mismatch)"
            print(f"  Position {i}: '{c1}' vs '{c2}' → {symbol}")
    
    if verbose:
        print(f"\n[Step 2] Count mismatches:")
        print(f"  Mismatch positions: {mismatches if mismatches else 'none'}")
        print(f"  Total mismatches: {distance}")
        print(f"\n{'='*60}")
        print(f"[RESULT] Hamming distance = {distance}")
        print(f"{'='*60}")
    
    return distance


# Alias for backwards compatibility
binary_distance = hamming_distance


def jaccard_similarity(set1, set2, verbose=True):
    """
    Jaccard similarity: |A ∩ B| / |A ∪ B|
    Works with any iterables (converts to sets).
    """
    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[JACCARD SIMILARITY]")
        print(f"{'='*60}")
        print(f"\nFormula: J(A,B) = |A ∩ B| / |A ∪ B|")
        print(f"\nInput:")
        print(f"  Set A: {sorted(set1) if all(isinstance(x, (int, float, str)) for x in set1) else set1}")
        print(f"  Set B: {sorted(set2) if all(isinstance(x, (int, float, str)) for x in set2) else set2}")
    
    intersection = set1 & set2
    union = set1 | set2
    
    if verbose:
        print(f"\n[Step 1] Find intersection (A ∩ B):")
        print(f"  Elements in BOTH sets: {sorted(intersection) if intersection else '∅ (empty)'}")
        print(f"  |A ∩ B| = {len(intersection)}")
        
        print(f"\n[Step 2] Find union (A ∪ B):")
        print(f"  Elements in EITHER set: {sorted(union) if union else '∅ (empty)'}")
        print(f"  |A ∪ B| = {len(union)}")
    
    if len(union) == 0:
        if verbose:
            print(f"\n[RESULT] Both sets empty, Jaccard similarity = 1.0 (by convention)")
        return 1.0
    
    jaccard = len(intersection) / len(union)
    
    if verbose:
        print(f"\n[Step 3] Apply formula:")
        print(f"  J(A,B) = |A ∩ B| / |A ∪ B|")
        print(f"  J(A,B) = {len(intersection)} / {len(union)}")
        print(f"  J(A,B) = {jaccard:.4f}")
        print(f"\n{'='*60}")
        print(f"[RESULT] Jaccard similarity = {jaccard:.4f}")
        print(f"{'='*60}")
    
    return jaccard


# Alias for backwards compatibility
overlap_coefficient = jaccard_similarity


def jaccard_distance(set1, set2, verbose=True):
    """
    Jaccard distance: 1 - Jaccard similarity.
    Works with any iterables (converts to sets).
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[JACCARD DISTANCE]")
        print(f"{'='*60}")
        print(f"\nFormula: d(A,B) = 1 - J(A,B) = 1 - |A ∩ B| / |A ∪ B|")
        print(f"\n[Step 1] First compute Jaccard similarity:")
    
    similarity = jaccard_similarity(set1, set2, verbose=verbose)
    distance = 1 - similarity
    
    if verbose:
        print(f"\n[Step 2] Compute distance:")
        print(f"  d(A,B) = 1 - J(A,B)")
        print(f"  d(A,B) = 1 - {similarity:.4f}")
        print(f"  d(A,B) = {distance:.4f}")
        print(f"\n{'='*60}")
        print(f"[RESULT] Jaccard distance = {distance:.4f}")
        print(f"{'='*60}")
    
    return distance


def euclidean_distance(v1, v2, verbose=True):
    """
    Euclidean distance between two vectors.
    d(A,B) = sqrt(sum((a_i - b_i)^2))
    Works with lists, tuples, or dicts.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[EUCLIDEAN DISTANCE]")
        print(f"{'='*60}")
        print(f"\nFormula: d(A,B) = √Σ(aᵢ - bᵢ)²")
    
    # Handle dict input
    if isinstance(v1, dict) and isinstance(v2, dict):
        all_keys = sorted(set(v1.keys()) | set(v2.keys()))
        v1_list = [v1.get(k, 0) for k in all_keys]
        v2_list = [v2.get(k, 0) for k in all_keys]
        if verbose:
            print(f"\nInput (as dicts, keys: {all_keys}):")
            print(f"  Vector A: {v1}")
            print(f"  Vector B: {v2}")
    else:
        v1_list, v2_list = list(v1), list(v2)
        if verbose:
            print(f"\nInput:")
            print(f"  Vector A: {v1_list}")
            print(f"  Vector B: {v2_list}")
    
    n = len(v1_list)
    if len(v1_list) != len(v2_list):
        raise ValueError("Vectors must have the same dimension")
    
    if verbose:
        print(f"  Dimensions: {n}")
        print(f"\n[Step 1] Compute difference for each dimension:")
    
    differences = []
    squared_diffs = []
    for i, (a, b) in enumerate(zip(v1_list, v2_list)):
        diff = a - b
        sq_diff = diff ** 2
        differences.append(diff)
        squared_diffs.append(sq_diff)
        if verbose:
            print(f"  d{i+1} = a{i+1} - b{i+1} = {a} - {b} = {diff}")
    
    if verbose:
        print(f"\n[Step 2] Square each difference:")
        for i, (diff, sq) in enumerate(zip(differences, squared_diffs)):
            print(f"  d{i+1}² = ({diff})² = {sq:.4f}")
    
    sum_sq = sum(squared_diffs)
    
    if verbose:
        print(f"\n[Step 3] Sum of squared differences:")
        sq_str = ' + '.join(f"{sq:.4f}" for sq in squared_diffs)
        print(f"  Σ(dᵢ)² = {sq_str}")
        print(f"  Σ(dᵢ)² = {sum_sq:.4f}")
    
    distance = math.sqrt(sum_sq)
    
    if verbose:
        print(f"\n[Step 4] Take square root:")
        print(f"  d(A,B) = √{sum_sq:.4f}")
        print(f"  d(A,B) = {distance:.4f}")
        print(f"\n{'='*60}")
        print(f"[RESULT] Euclidean distance = {distance:.4f}")
        print(f"{'='*60}")
    
    return distance


# =============================================================================
# SECTION 3: VECTOR SPACE MODELS
# =============================================================================

def term_counts(documents, verbose=True):
    """
    Bag of Words / Count Vectorizer.
    Counts the frequency of each term in each document.
    Returns (doc_counts, vocab) where doc_counts is a list of Counter objects.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[BAG OF WORDS / COUNT VECTORIZER]")
        print(f"{'='*60}")
        print(f"\nConcept: Represent each document as a vector of word counts")
        print(f"\nInput: {len(documents)} documents")
    
    vocab = set()
    doc_counts = []
    
    if verbose:
        print(f"\n[Step 1] Tokenize each document and count terms:")
    
    for i, doc in enumerate(documents):
        tokens = doc.lower().split() if isinstance(doc, str) else doc
        counts = Counter(tokens)
        doc_counts.append(counts)
        vocab.update(tokens)
        if verbose:
            preview = doc[:50] + "..." if len(str(doc)) > 50 else doc
            print(f"  Doc {i}: '{preview}'")
            print(f"         Tokens: {len(tokens)}, Unique: {len(counts)}")
            top3 = counts.most_common(3)
            print(f"         Top 3: {top3}")
    
    vocab = sorted(vocab)
    
    if verbose:
        print(f"\n[Step 2] Build vocabulary (all unique terms):")
        print(f"  Total unique terms: {len(vocab)}")
        if len(vocab) <= 20:
            print(f"  Vocabulary: {vocab}")
        else:
            print(f"  First 20: {vocab[:20]}...")
        
        print(f"\n{'='*60}")
        print(f"[RESULT] {len(doc_counts)} document vectors, {len(vocab)} vocabulary size")
        print(f"{'='*60}")
    
    return doc_counts, vocab


# Aliases for discoverability
bag_of_words = term_counts
count_vectorizer = term_counts


def build_td_matrix(documents, verbose=True):
    """
    Build term-document matrix.
    Rows = terms, Columns = documents.
    Each cell [t][d] = count of term t in document d.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[TERM-DOCUMENT MATRIX]")
        print(f"{'='*60}")
        print(f"\nConcept: Matrix where rows=terms, columns=documents")
        print(f"         Cell [term][doc] = frequency of term in document")
    
    doc_counts, vocab = term_counts(documents, verbose=False)
    
    matrix = {}
    for term in vocab:
        matrix[term] = [counts.get(term, 0) for counts in doc_counts]
    
    if verbose:
        print(f"\n[Step 1] Extract vocabulary: {len(vocab)} unique terms")
        print(f"[Step 2] Create matrix: {len(vocab)} rows x {len(documents)} columns")
        
        print(f"\n[Step 3] Term-Document Matrix:")
        header = "Term".ljust(15) + "  ".join(f"D{i}" for i in range(len(documents)))
        print(header)
        print("-" * len(header))
        display_terms = vocab[:30] if len(vocab) > 30 else vocab
        for term in display_terms:
            row = term.ljust(15) + "  ".join(str(v).rjust(2) for v in matrix[term])
            print(row)
        if len(vocab) > 30:
            print(f"  ... ({len(vocab) - 30} more terms)")
        
        print(f"\n{'='*60}")
        print(f"[RESULT] Term-Document matrix: {len(vocab)} terms x {len(documents)} docs")
        print(f"{'='*60}")
    
    return matrix, vocab


# Alias
term_document_matrix = build_td_matrix


def compute_tf(document, verbose=True):
    """
    Compute Term Frequency (TF) for a single document.
    TF(t,d) = count(t in d) / total terms in d
    
    Returns dict mapping each term to its TF value.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[TERM FREQUENCY (TF)]")
        print(f"{'='*60}")
        print(f"\nFormula: TF(t,d) = count(t in d) / total_terms_in_d")
        print(f"         (How often does term t appear in document d?)")
    
    tokens = document.lower().split() if isinstance(document, str) else list(document)
    counts = Counter(tokens)
    total = len(tokens)
    
    if verbose:
        preview = str(document)[:60] + "..." if len(str(document)) > 60 else document
        print(f"\nInput document: '{preview}'")
        print(f"Total tokens: {total}")
    
    tf = {}
    
    if verbose:
        print(f"\n[Step 1] Count each term:")
    
    for term in sorted(counts.keys()):
        count = counts[term]
        tf[term] = count / total if total > 0 else 0
        if verbose:
            print(f"  '{term}': count={count}, TF = {count}/{total} = {tf[term]:.4f}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[RESULT] TF computed for {len(tf)} unique terms")
        print(f"{'='*60}")
    
    return tf


# Alias
term_frequency = compute_tf


def compute_df(documents, verbose=True):
    """
    Compute Document Frequency (DF) for each term in a corpus.
    DF(t) = number of documents containing term t
    
    Returns dict mapping each term to its DF value.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[DOCUMENT FREQUENCY (DF)]")
        print(f"{'='*60}")
        print(f"\nFormula: DF(t) = number of documents containing term t")
        print(f"         (In how many documents does term t appear?)")
        print(f"\nInput: {documents}")
    
    doc_counts, vocab = term_counts(documents, verbose=False)
    
    df = {}
    
    if verbose:
        print(f"\n[Step 1] For each term, count documents containing it:")
    
    for term in sorted(vocab):
        doc_count = sum(1 for counts in doc_counts if term in counts)
        df[term] = doc_count
        if verbose:
            docs_with_term = [i for i, counts in enumerate(doc_counts) if term in counts]
            print(f"  '{term}': appears in docs {docs_with_term}, DF = {doc_count}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[RESULT] DF computed for {len(df)} terms across {len(documents)} docs")
        print(f"{'='*60}")
    
    return df


# Alias
document_frequency = compute_df


def compute_idf(documents, verbose=True):
    """
    Compute Inverse Document Frequency (IDF) for each term in a corpus.
    IDF(t) = log(N / DF(t)) where N = total number of documents
    
    Rare terms get higher IDF (more discriminative).
    Common terms get lower IDF (less discriminative).
    
    Returns dict mapping each term to its IDF value.
    """
    N = len(documents)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[INVERSE DOCUMENT FREQUENCY (IDF)]")
        print(f"{'='*60}")
        print(f"\nFormula: IDF(t) = log(N / DF(t))")
        print(f"         where N = total documents = {N}")
        print(f"\nIntuition: Rare terms -> high IDF (important)")
        print(f"           Common terms -> low IDF (less important)")
    
    df = compute_df(documents, verbose=False)
    
    idf = {}
    
    if verbose:
        print(f"\n[Step 1] Compute IDF for each term:")
        print(f"{'Term':<15} {'DF':>4} {'N/DF':>8} {'IDF=log(N/DF)':>12}")
        print("-" * 45)
    
    for term in sorted(df.keys()):
        df_val = df[term]
        ratio = N / df_val if df_val > 0 else 0
        idf[term] = math.log(ratio) if ratio > 0 else 0
        if verbose:
            print(f"  {term:<13} {df_val:>4} {ratio:>8.2f} {idf[term]:>12.4f}")
    
    if verbose:
        # Show interpretation
        sorted_by_idf = sorted(idf.items(), key=lambda x: x[1], reverse=True)
        print(f"\n[Step 2] Interpretation:")
        print(f"  Most discriminative (highest IDF): {sorted_by_idf[:3]}")
        print(f"  Least discriminative (lowest IDF): {sorted_by_idf[-3:]}")
        
        print(f"\n{'='*60}")
        print(f"[RESULT] IDF computed for {len(idf)} terms")
        print(f"{'='*60}")
    
    return idf


# Alias
inverse_document_frequency = compute_idf


def compute_tfidf(documents, verbose=True):
    """
    Compute TF-IDF scores for a corpus.
    TF-IDF(t,d) = TF(t,d) * IDF(t)
    
    Combines term frequency (local importance) with inverse document frequency
    (global importance across corpus).
    
    Returns (tfidf_vectors, vocab, idf_values).
    """
    N = len(documents)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[TF-IDF: TERM FREQUENCY - INVERSE DOCUMENT FREQUENCY]")
        print(f"{'='*60}")
        print(f"\nFormula: TF-IDF(t,d) = TF(t,d) * IDF(t)")
        print(f"         where TF(t,d) = count(t,d) / total_terms_in_d")
        print(f"               IDF(t)  = log(N / DF(t))")
        print(f"\nIntuition: High TF-IDF = term frequent in this doc, rare in corpus")
        print(f"\nInput: {N} documents")
    
    doc_counts, vocab = term_counts(documents, verbose=True)
    
    # Step 1: Compute DF
    if verbose:
        print(f"\n[Step 1] Compute Document Frequency (DF):")
    
    df = {}
    for term in vocab:
        df[term] = sum(1 for counts in doc_counts if term in counts)
    
    if verbose:
        for term in sorted(vocab)[:10]:
            print(f"  DF('{term}') = {df[term]}")
        if len(vocab) > 10:
            print(f"  ... ({len(vocab) - 10} more terms)")
    
    # Step 2: Compute IDF
    if verbose:
        print(f"\n[Step 2] Compute Inverse Document Frequency (IDF):")
    
    idf = {}
    for term in vocab:
        idf[term] = math.log(N / df[term]) if df[term] > 0 else 0
    
    if verbose:
        for term in sorted(vocab)[:10]:
            print(f"  IDF('{term}') = log({N}/{df[term]}) = {idf[term]:.4f}")
        if len(vocab) > 10:
            print(f"  ... ({len(vocab) - 10} more terms)")
    
    # Step 3: Compute TF for each document
    if verbose:
        print(f"\n[Step 3] Compute TF-IDF for each document:")
    
    tfidf = []
    for i, counts in enumerate(doc_counts):
        doc_len = sum(counts.values())
        doc_tfidf = {}
        
        if verbose:
            print(f"\n  Document {i} ({doc_len} tokens):")
        
        for term in vocab:
            count = counts.get(term, 0)
            tf = count / doc_len if doc_len > 0 else 0
            tfidf_val = tf * idf[term]
            doc_tfidf[term] = tfidf_val
            
            if verbose and count > 0:
                print(f"    '{term}': TF={count}/{doc_len}={tf:.4f}, IDF={idf[term]:.4f}, TF-IDF={tfidf_val:.4f}")
        
        tfidf.append(doc_tfidf)
    
    if verbose:
        print(f"\n[Step 4] Summary - Top TF-IDF terms per document:")
        for i, doc_tfidf in enumerate(tfidf):
            top_terms = sorted(doc_tfidf.items(), key=lambda x: x[1], reverse=True)[:5]
            terms_str = ", ".join(f"{t}:{s:.3f}" for t, s in top_terms if s > 0)
            print(f"  Doc {i}: {terms_str}")
        
        print(f"\n{'='*60}")
        print(f"[RESULT] TF-IDF computed for {len(documents)} docs, {len(vocab)} terms")
        print(f"{'='*60}")
    
    return tfidf, vocab, idf


# Alias
tfidf = compute_tfidf


def cosine_similarity(v1, v2, verbose=True):
    """
    Cosine similarity between two vectors (as dicts or lists).
    cos(A,B) = (A·B) / (||A|| × ||B||)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[COSINE SIMILARITY]")
        print(f"{'='*60}")
        print(f"\nFormula: cos(θ) = (A·B) / (||A|| × ||B||)")
        print(f"         where A·B = Σ(aᵢ × bᵢ)  and  ||A|| = √Σ(aᵢ)²")
    
    # Handle dict input
    if isinstance(v1, dict) and isinstance(v2, dict):
        all_keys = sorted(set(v1.keys()) | set(v2.keys()))
        v1_list = [v1.get(k, 0) for k in all_keys]
        v2_list = [v2.get(k, 0) for k in all_keys]
        if verbose:
            print(f"\nInput (as dicts, keys: {all_keys}):")
            print(f"  Vector A: {v1}")
            print(f"  Vector B: {v2}")
    else:
        v1_list, v2_list = list(v1), list(v2)
        if verbose:
            print(f"\nInput:")
            print(f"  Vector A: {v1_list}")
            print(f"  Vector B: {v2_list}")
    
    n = len(v1_list)
    
    # Dot product step by step
    if verbose:
        print(f"\n[Step 1] Compute dot product (A·B):")
    
    products = []
    for i, (a, b) in enumerate(zip(v1_list, v2_list)):
        prod = a * b
        products.append(prod)
        if verbose:
            print(f"  a{i+1} × b{i+1} = {a} × {b} = {prod:.4f}")
    
    dot = sum(products)
    if verbose:
        prod_str = ' + '.join(f"{p:.4f}" for p in products)
        print(f"  A·B = {prod_str}")
        print(f"  A·B = {dot:.4f}")
    
    # Magnitude A step by step
    if verbose:
        print(f"\n[Step 2] Compute magnitude ||A||:")
    
    sq_a = [a * a for a in v1_list]
    sum_sq_a = sum(sq_a)
    mag1 = math.sqrt(sum_sq_a)
    
    if verbose:
        for i, (a, sq) in enumerate(zip(v1_list, sq_a)):
            print(f"  a{i+1}² = {a}² = {sq:.4f}")
        print(f"  Σ(aᵢ)² = {sum_sq_a:.4f}")
        print(f"  ||A|| = √{sum_sq_a:.4f} = {mag1:.4f}")
    
    # Magnitude B step by step
    if verbose:
        print(f"\n[Step 3] Compute magnitude ||B||:")
    
    sq_b = [b * b for b in v2_list]
    sum_sq_b = sum(sq_b)
    mag2 = math.sqrt(sum_sq_b)
    
    if verbose:
        for i, (b, sq) in enumerate(zip(v2_list, sq_b)):
            print(f"  b{i+1}² = {b}² = {sq:.4f}")
        print(f"  Σ(bᵢ)² = {sum_sq_b:.4f}")
        print(f"  ||B|| = √{sum_sq_b:.4f} = {mag2:.4f}")
    
    if mag1 == 0 or mag2 == 0:
        if verbose:
            print(f"\n[RESULT] Zero vector detected, similarity = 0")
        return 0.0
    
    similarity = dot / (mag1 * mag2)
    
    if verbose:
        print(f"\n[Step 4] Apply formula:")
        print(f"  cos(θ) = (A·B) / (||A|| × ||B||)")
        print(f"  cos(θ) = {dot:.4f} / ({mag1:.4f} × {mag2:.4f})")
        print(f"  cos(θ) = {dot:.4f} / {mag1 * mag2:.4f}")
        print(f"  cos(θ) = {similarity:.4f}")
        print(f"\n{'='*60}")
        print(f"[RESULT] Cosine similarity = {similarity:.4f}")
        print(f"{'='*60}")
    
    return similarity


# Alias for backwards compatibility
vector_similarity = cosine_similarity


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
    An n-gram is a contiguous sequence of n items from a text.
    """
    if isinstance(tokens, str):
        tokens = tokens.split()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[N-GRAM EXTRACTION]")
        print(f"{'='*60}")
        print(f"\nDefinition: An n-gram is a sequence of n consecutive tokens")
        print(f"  - Unigram (n=1): single words")
        print(f"  - Bigram (n=2): pairs of words")
        print(f"  - Trigram (n=3): triples of words")
        print(f"\nInput tokens: {tokens}")
        print(f"N = {n} ({['unigram', 'bigram', 'trigram', f'{n}-gram'][min(n-1, 3)]})")
    
    ngrams = []
    
    if verbose:
        print(f"\n[Step 1] Slide window of size {n} across tokens:")
    
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
        if verbose:
            window = ' '.join(f"[{tokens[j]}]" if i <= j < i+n else tokens[j] for j in range(len(tokens)))
            print(f"  Position {i}: {window} -> {ngram}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[RESULT] {len(ngrams)} {n}-grams generated")
        print(f"{'='*60}")
    
    return ngrams


def ngram_counts(corpus, n, verbose=True):
    """
    Count n-gram frequencies in a corpus.
    Adds <s> start tokens and </s> end tokens for language modeling.
    
    Returns (counts, context_counts) where:
    - counts: Counter of n-gram frequencies
    - context_counts: Counter of (n-1)-gram context frequencies
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[N-GRAM COUNTING]")
        print(f"{'='*60}")
        print(f"\nConcept: Count how often each n-gram appears in the corpus")
        print(f"         Also count contexts (n-1 grams) for probability calculation")
        print(f"\nInput: {len(corpus)} sentences, n = {n}")
    
    counts = Counter()
    context_counts = Counter()
    
    if verbose:
        print(f"\n[Step 1] Process each sentence (with <s> and </s> markers):")
    
    for i, sent in enumerate(corpus):
        tokens = sent.lower().split() if isinstance(sent, str) else sent
        # Add start/end tokens
        padded = ['<s>'] * (n - 1) + tokens + ['</s>']
        
        if verbose:
            print(f"\n  Sentence {i}: \"{sent}\"")
            print(f"  Padded: {padded}")
        
        ngrams = extract_ngrams(padded, n, verbose=False)
        
        if verbose:
            print(f"  N-grams extracted: {ngrams}")
        
        for ng in ngrams:
            counts[ng] += 1
            context = ng[:-1]
            context_counts[context] += 1
    
    if verbose:
        print(f"\n[Step 2] N-gram frequency table:")
        print(f"  {'N-gram':<30} {'Count':>6}")
        print(f"  {'-'*36}")
        for ng, c in counts.most_common():
            print(f"  {str(ng):<30} {c:>6}")
        
        print(f"\n[Step 3] Context frequency table:")
        print(f"  {'Context':<25} {'Count':>6}")
        print(f"  {'-'*31}")
        for ctx, c in context_counts.most_common():
            print(f"  {str(ctx):<25} {c:>6}")
        
        print(f"\n{'='*60}")
        print(f"[RESULT] {len(counts)} unique n-grams, {len(context_counts)} unique contexts")
        print(f"{'='*60}")
    
    return counts, context_counts


# Alias for discoverability
count_ngrams = ngram_counts


def estimate_prob(ngram, counts, context_counts, verbose=True):
    """
    Maximum Likelihood Estimation (MLE) for n-gram probability.
    
    Formula: P(word | context) = count(context + word) / count(context)
    
    This is the probability of seeing 'word' given we've seen 'context'.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[MAXIMUM LIKELIHOOD ESTIMATION (MLE)]")
        print(f"{'='*60}")
        print(f"\nFormula: P(w | context) = C(context, w) / C(context)")
        print(f"         where C(x) = count of x in training corpus")
    
    context = ngram[:-1]
    word = ngram[-1]
    
    ngram_count = counts.get(ngram, 0)
    context_count = context_counts.get(context, 0)
    
    if verbose:
        print(f"\nInput n-gram: {ngram}")
        print(f"\n[Step 1] Split into context and target word:")
        print(f"  Context: {context}")
        print(f"  Target word: '{word}'")
        
        print(f"\n[Step 2] Look up counts in training data:")
        print(f"  C({ngram}) = {ngram_count}")
        print(f"  C({context}) = {context_count}")
    
    if context_count == 0:
        prob = 0.0
        if verbose:
            print(f"\n[Step 3] Context never seen in training data!")
            print(f"  P('{word}' | {context}) = 0")
    else:
        prob = ngram_count / context_count
        if verbose:
            print(f"\n[Step 3] Apply MLE formula:")
            print(f"  P('{word}' | {context}) = C({ngram}) / C({context})")
            print(f"  P('{word}' | {context}) = {ngram_count} / {context_count}")
            print(f"  P('{word}' | {context}) = {prob:.4f}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[RESULT] P('{word}' | {context}) = {prob:.4f}")
        print(f"{'='*60}")
    
    return prob


# Aliases for discoverability
maximum_likelihood = estimate_prob
mle_probability = estimate_prob


def show_ngram_probabilities(context, counts, context_counts, top_k=10, verbose=True):
    """
    Show probability distribution of next words given a context.
    Displays a visual bar chart of the most likely continuations.
    
    Args:
        context: tuple of words, e.g., ('the',) for bigram or ('the', 'cat') for trigram
        counts: n-gram counts from ngram_counts()
        context_counts: context counts from ngram_counts()
        top_k: show top k most likely words
    """
    if isinstance(context, str):
        context = tuple(context.split())
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[NEXT WORD PROBABILITY DISTRIBUTION]")
        print(f"{'='*60}")
        print(f"\nContext: {context}")
    
    context_count = context_counts.get(context, 0)
    
    if context_count == 0:
        if verbose:
            print(f"\n[ERROR] Context '{context}' not found in training data!")
        return {}
    
    if verbose:
        print(f"Context count: {context_count}")
        print(f"\n[Step 1] Find all n-grams starting with this context:")
    
    # Find all ngrams that start with this context
    next_words = {}
    for ngram, count in counts.items():
        if ngram[:-1] == context:
            word = ngram[-1]
            prob = count / context_count
            next_words[word] = (count, prob)
    
    # Sort by probability
    sorted_words = sorted(next_words.items(), key=lambda x: x[1][1], reverse=True)
    
    if verbose:
        print(f"\n[Step 2] Compute P(w | context) = C(context, w) / C(context):")
        print(f"\n  {'Next Word':<15} {'Count':>6} {'Probability':>12} {'Visualization'}")
        print(f"  {'-'*60}")
        
        max_prob = sorted_words[0][1][1] if sorted_words else 1.0
        
        for word, (count, prob) in sorted_words[:top_k]:
            bar_length = int(30 * prob / max_prob) if max_prob > 0 else 0
            bar = '█' * bar_length
            print(f"  '{word}'".ljust(15) + f"{count:>6}" + f"{prob:>12.4f}  " + bar)
        
        if len(sorted_words) > top_k:
            print(f"  ... and {len(sorted_words) - top_k} more words")
        
        # Show that probabilities sum to 1
        total_prob = sum(p for _, p in next_words.values())
        print(f"\n[Step 3] Verify probabilities sum to 1:")
        print(f"  Sum of all P(w | context) = {total_prob:.4f}")
        
        print(f"\n{'='*60}")
        print(f"[RESULT] {len(next_words)} possible next words for context {context}")
        print(f"{'='*60}")
    
    return {w: p for w, (c, p) in next_words.items()}


# Aliases
likelihood_distribution = show_ngram_probabilities
next_word_probabilities = show_ngram_probabilities


def ngram_probability_matrix(counts, context_counts, verbose=True):
    """
    Display the full n-gram probability matrix P(word | context).
    
    Shows a matrix where:
    - Rows = contexts (previous n-1 words)
    - Columns = next words
    - Cells = P(column | row)
    
    Works for bigrams (single word contexts) and trigrams (2-word contexts).
    
    Args:
        counts: n-gram counts from ngram_counts(corpus, n)
        context_counts: context counts from ngram_counts()
    
    Returns:
        dict: {context: {word: probability}}
    """
    # Extract all unique contexts and words
    contexts = sorted(set(context_counts.keys()), key=lambda x: str(x))
    words = sorted(set(ng[-1] for ng in counts.keys()))
    
    # Determine n from context length
    sample_ctx = next(iter(context_counts.keys()), ())
    n = len(sample_ctx) + 1  # context is (n-1)-gram
    
    # Build probability matrix
    prob_matrix = {}
    for ctx in contexts:
        ctx_count = context_counts.get(ctx, 0)
        prob_matrix[ctx] = {}
        for word in words:
            ngram = ctx + (word,)
            ngram_count = counts.get(ngram, 0)
            if ctx_count > 0:
                prob_matrix[ctx][word] = ngram_count / ctx_count
            else:
                prob_matrix[ctx][word] = 0.0
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"[{n}-GRAM PROBABILITY MATRIX]")
        print(f"{'='*70}")
        print(f"\nFormula: P(word | context) = C(context, word) / C(context)")
        print(f"\nMatrix layout:")
        print(f"  - Rows = context ({n-1}-gram)")
        print(f"  - Columns = next word")
        print(f"  - Cell = P(next | context)")
        
        # Format contexts for display
        def fmt_ctx(ctx):
            return " ".join(ctx) if len(ctx) > 1 else ctx[0]
        
        context_strs = [fmt_ctx(ctx) for ctx in contexts]
        
        # Determine column width
        col_width = max(8, max(len(w) for w in words) + 1) if words else 8
        row_label_width = max(len(c) for c in context_strs) + 2 if context_strs else 10
        
        # Print header
        print(f"\n{'':>{row_label_width}}", end="")
        for word in words:
            print(f"{word:>{col_width}}", end="")
        print()
        
        # Print separator
        print(f"{'':>{row_label_width}}" + "-" * (col_width * len(words)))
        
        # Print each row
        for ctx, ctx_str in zip(contexts, context_strs):
            print(f"{ctx_str:>{row_label_width}}", end="")
            for word in words:
                prob = prob_matrix[ctx].get(word, 0)
                if prob > 0:
                    print(f"{prob:>{col_width}.2f}", end="")
                else:
                    print(f"{'--':>{col_width}}", end="")
            print()
        
        print(f"\n{'='*70}")
        print(f"[RESULT] {len(contexts)} contexts × {len(words)} words")
        print(f"{'='*70}")
    
    return prob_matrix


# Aliases
probability_matrix = ngram_probability_matrix
bigram_probability_matrix = ngram_probability_matrix


def sequence_probability(tokens, counts, context_counts, n, verbose=True):
    """
    Calculate probability of a sentence using the chain rule.
    
    Formula: P(w1, w2, ..., wm) = ∏ P(wi | w(i-n+1)...w(i-1))
    
    This decomposes the joint probability into a product of conditional probabilities.
    """
    if isinstance(tokens, str):
        tokens = tokens.lower().split()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[SENTENCE PROBABILITY (Chain Rule)]")
        print(f"{'='*60}")
        print(f"\nFormula: P(w1, w2, ..., wm) = ∏ P(wi | context)")
        print(f"         Using {n}-gram model")
        print(f"\nInput sentence: \"{' '.join(tokens)}\"")
    
    # Add padding
    padded = ['<s>'] * (n - 1) + tokens + ['</s>']
    
    if verbose:
        print(f"\n[Step 1] Add start/end markers:")
        print(f"  Padded: {padded}")
        print(f"\n[Step 2] Decompose into conditional probabilities:")
    
    log_prob = 0.0
    prob_product = 1.0
    probabilities = []
    
    for i in range(n - 1, len(padded)):
        ngram = tuple(padded[i - n + 1:i + 1])
        context = ngram[:-1]
        word = ngram[-1]
        p = estimate_prob(ngram, counts, context_counts, verbose=False)
        probabilities.append((context, word, p))
        
        if verbose:
            print(f"  P('{word}' | {context}) = {p:.4f}")
        
        if p > 0:
            log_prob += math.log(p)
            prob_product *= p
        else:
            log_prob = float('-inf')
            prob_product = 0.0
            if verbose:
                print(f"  [WARNING] Zero probability! Sentence impossible under this model.")
            break
    
    if verbose:
        print(f"\n[Step 3] Multiply all probabilities:")
        prob_strs = [f"{p:.4f}" for _, _, p in probabilities]
        print(f"  P(sentence) = {' × '.join(prob_strs)}")
        print(f"  P(sentence) = {prob_product:.10f}")
        
        if prob_product > 0:
            print(f"\n[Step 4] Log probability (more numerically stable):")
            print(f"  log P(sentence) = {log_prob:.4f}")
        
        print(f"\n{'='*60}")
        print(f"[RESULT] P(sentence) = {prob_product:.10f}")
        if prob_product > 0:
            print(f"[RESULT] Log P(sentence) = {log_prob:.4f}")
        print(f"{'='*60}")
    
    return prob_product, log_prob


# Aliases
sentence_probability = sequence_probability
probability_of_sentence = sequence_probability


def apply_smoothing(counts, context_counts, vocab_size, k=1, verbose=True):
    """
    Add-k (Laplace) smoothing for n-gram probabilities.
    
    Formula: P_smooth(w | context) = (C(context, w) + k) / (C(context) + k × V)
    
    This prevents zero probabilities for unseen n-grams by adding k to all counts.
    When k=1, this is called "Laplace smoothing" or "add-one smoothing".
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[ADD-{k} SMOOTHING (Laplace Smoothing)]")
        print(f"{'='*60}")
        print(f"\nFormula: P_smooth(w | context) = (C(context,w) + k) / (C(context) + k×V)")
        print(f"\nParameters:")
        print(f"  k (smoothing constant) = {k}")
        print(f"  V (vocabulary size) = {vocab_size}")
        print(f"\nIntuition: Add k 'pseudo-counts' to every possible n-gram")
        print(f"           This prevents zero probabilities for unseen n-grams")
    
    def smoothed_prob(ngram, inner_verbose=True):
        context = ngram[:-1]
        word = ngram[-1]
        ngram_count = counts.get(ngram, 0)
        context_count = context_counts.get(context, 0)
        
        numerator = ngram_count + k
        denominator = context_count + k * vocab_size
        prob = numerator / denominator
        
        if verbose and inner_verbose:
            print(f"\n  Computing P_smooth('{word}' | {context}):")
            print(f"    C({ngram}) = {ngram_count}")
            print(f"    C({context}) = {context_count}")
            print(f"    P_smooth = ({ngram_count} + {k}) / ({context_count} + {k}×{vocab_size})")
            print(f"    P_smooth = {numerator} / {denominator}")
            print(f"    P_smooth = {prob:.6f}")
        
        return prob
    
    return smoothed_prob


# Alias
laplace_smoothing = apply_smoothing


def compute_perplexity(test_tokens, counts, context_counts, n, vocab_size=None, smoothing_k=0, verbose=True):
    """
    Compute perplexity of a test sequence.
    
    Formula: PP = 2^(-1/N × Σ log2 P(wi | context))
    
    Perplexity measures how well a language model predicts the test data.
    Lower perplexity = better model. Can be interpreted as the average
    number of equally likely choices the model is uncertain between.
    """
    if isinstance(test_tokens, str):
        test_tokens = test_tokens.lower().split()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[PERPLEXITY CALCULATION]")
        print(f"{'='*60}")
        print(f"\nFormula: PP = 2^(-1/N × Σ log2 P(wi | context))")
        print(f"\nInterpretation:")
        print(f"  - Lower perplexity = model predicts test data better")
        print(f"  - PP ≈ average branching factor (choices at each step)")
        print(f"\nTest sequence: \"{' '.join(test_tokens)}\"")
        print(f"Using {n}-gram model")
        if smoothing_k > 0:
            print(f"With add-{smoothing_k} smoothing (V = {vocab_size})")
    
    # Add padding
    padded = ['<s>'] * (n - 1) + test_tokens + ['</s>']
    N = len(padded) - (n - 1)  # number of predictions
    
    if verbose:
        print(f"\n[Step 1] Prepare sequence:")
        print(f"  Padded: {padded}")
        print(f"  N (number of predictions) = {N}")
    
    log_prob_sum = 0.0
    
    if smoothing_k > 0 and vocab_size:
        smooth_fn = apply_smoothing(counts, context_counts, vocab_size, smoothing_k, verbose=False)
    
    if verbose:
        print(f"\n[Step 2] Compute log2 probability for each prediction:")
        print(f"  {'N-gram':<30} {'P':>10} {'log2(P)':>10}")
        print(f"  {'-'*50}")
    
    for i in range(n - 1, len(padded)):
        ngram = tuple(padded[i - n + 1:i + 1])
        
        if smoothing_k > 0 and vocab_size:
            p = smooth_fn(ngram, inner_verbose=False)
        else:
            p = estimate_prob(ngram, counts, context_counts, verbose=False)
        
        if p > 0:
            log2_p = math.log2(p)
            log_prob_sum += log2_p
            if verbose:
                print(f"  {str(ngram):<30} {p:>10.6f} {log2_p:>10.4f}")
        else:
            log_prob_sum = float('-inf')
            if verbose:
                print(f"  {str(ngram):<30} {'0':>10} {'-inf':>10}")
            break
    
    if log_prob_sum == float('-inf'):
        perplexity = float('inf')
    else:
        avg_log_prob = log_prob_sum / N
        perplexity = 2 ** (-avg_log_prob)
    
    if verbose:
        print(f"\n[Step 3] Calculate perplexity:")
        print(f"  Sum of log2(P) = {log_prob_sum:.4f}")
        print(f"  Average = {log_prob_sum:.4f} / {N} = {log_prob_sum/N:.4f}")
        print(f"  PP = 2^(-{log_prob_sum/N:.4f}) = 2^{-log_prob_sum/N:.4f}")
        print(f"  PP = {perplexity:.4f}")
        
        print(f"\n{'='*60}")
        print(f"[RESULT] Perplexity = {perplexity:.4f}")
        print(f"{'='*60}")
    
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
