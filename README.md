# Preliminary Experiment of Embedding Models for Code-Assistance

### Collect problems
```
# crawl data (or download *.zip)
$ python crawl.py
```

### Preprocess
```
# find english problem
$ python find_english_problems.py

# deduplicate same problems in problems_desc_only (no duplication in problems)
$ python deduplicate_problems.py
```

### Analysis
```
# compute embeddings
$ python compute_embeddings.py

# plot difficulty of english problem
$ python plot_difficulty.py
```
