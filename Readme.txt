There are two scripts:
- training.py: contains code for training SCIBERT model first by MLM then finetuning to BIOSSES-Dataset. Models saved are under output directory(This requires HEAVY GPU)
- cosine_similarity.py: contains code for finding cosine similarity. This script uses the model: SciBert_finetuning_BIOSSES
Install the following packages with these commands:
pip install transformers
pip install sentence-transformers
pip install plotly


