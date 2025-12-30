# Web Intelligence Toolkit - Quick Reference

## Summary
Comprehensive toolkit for Web Intelligence exam preparation with **all 24 tests passing**.

## Files
- `wi_toolkit.py` - Main toolkit (~1750 lines)
- `test_wi_toolkit.py` - Unit tests (24 tests)

---

## Quick Reference

### Section 1: Text Processing
```python
from wi_toolkit import prepare_text, split_tokens, filter_common, reduce_stem, clean_pipeline

stems = clean_pipeline("The running cats are jumping!")
# Shows all steps: lowercase -> remove punctuation -> tokenize -> filter stopwords -> stem
```

### Section 2: Distance Metrics
```python
from wi_toolkit import edit_distance, binary_distance, overlap_coefficient

edit_distance("kitten", "sitting")      # Levenshtein = 3
binary_distance("karolin", "kathrin")   # Hamming = 3
overlap_coefficient({1,2,3}, {2,3,4})   # Jaccard = 0.5
```

### Section 3: Vector Space
```python
from wi_toolkit import term_counts, build_td_matrix, compute_tfidf, vector_similarity

docs = ["cat sat mat", "dog ran fast"]
tfidf, vocab, idf = compute_tfidf(docs)
similarity = vector_similarity(tfidf[0], tfidf[1])
```

### Section 4: N-grams & Language Models
```python
from wi_toolkit import extract_ngrams, ngram_counts, apply_smoothing, compute_perplexity

corpus = ["win money now", "cheap products"]
counts, context_counts = ngram_counts(corpus, n=2)
smooth_fn = apply_smoothing(counts, context_counts, vocab_size=10, k=1)
perplexity = compute_perplexity("cheap money", counts, context_counts, n=2)
```

### Section 5: Inverted Index
```python
from wi_toolkit import build_index, query_index

docs = ["scientist won award", "actor won prize"]
index = build_index(docs, ["Doc1", "Doc2"])
results = query_index(index, "scientist award", op='AND')
```

### Section 6: Neural Gates
```python
from wi_toolkit import gate_and, gate_or, gate_xor, show_truth_table

gate_and(1, 1)  # = 1
gate_or(0, 1)   # = 1
gate_xor(1, 1)  # = 0 (shows 2-layer network)
show_truth_table(gate_xor, "XOR")
```

### Section 7: Graph Analytics
```python
from wi_toolkit import degree_centrality, closeness_centrality, pagerank

adj = [[0,1,1], [1,0,1], [1,1,0]]  # Triangle
degree_centrality(adj)
pagerank(adj, damping=0.85)
```

### Section 8: Evaluation & Embeddings
```python
from wi_toolkit import split_data, eval_metrics, aggregate_tokens

y_true = [1, 1, 0, 0]
y_pred = [1, 0, 0, 1]
metrics = eval_metrics(y_true, y_pred)  # Shows accuracy, precision, recall, F1
```

### Section 9: Computation Graphs
```python
from wi_toolkit import forward_pass, backward_pass

inputs = [2, 3]
ops = [('mul', [0, 1])]  # 2 * 3 = 6
output, cache = forward_pass(inputs, ops)
grads = backward_pass(1.0, cache)  # Backprop
```

---

## Exam Usage Tips
1. All functions print step-by-step output by default
2. Set `verbose=False` for clean results
3. Run `python test_wi_toolkit.py` to verify everything works before exam
