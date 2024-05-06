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

### Compute embeddings
```
# GPT4AllEmbeddings
$ python compute_embeddings.py --homedir english_problems_desc_only_deduplicate --savedir GPT4AllEmbeddings --embedding GPT4AllEmbeddings

# HuggingFaceEmbeddings
$ python compute_embeddings.py --homedir english_problems_desc_only_deduplicate --savedir HuggingFaceEmbeddings --embedding HuggingFaceEmbeddings

# HuggingFaceBgeEmbeddings
$ python compute_embeddings.py --homedir english_problems_desc_only_deduplicate --savedir HuggingFaceBgeEmbeddings --embedding HuggingFaceBgeEmbeddings

# HuggingFaceInstructEmbeddings
$ python compute_embeddings.py --homedir english_problems_desc_only_deduplicate --savedir HuggingFaceInstructEmbeddings --embedding HuggingFaceInstructEmbeddings
```

### Analysis
```
# plot difficulty of english problem
$ python plot_difficulty.py
```
