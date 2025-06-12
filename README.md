
# Multimodal Recommendation System

This project implements a **multimodal movie recommendation system** using:
- Collaborative Filtering (CF)
- CF + BERT (text overview embeddings)
- CF + BERT + ResNet (poster image embeddings)

It includes modular training, evaluation, and hyperparameter search across fusion strategies.

---

## Project Structure

| File | Purpose |
|------|---------|
| `cf_model.py` | Classic CF model using Surprise SVD |
| `cf_bert_fusion_model.py` | CF + BERT model (text-based fusion) |
| `cf_bert_resnet_fusion_model.py` | CF + BERT + ResNet model (text + image fusion) |
| `evaluation.py` | Evaluation metrics: RMSE and MRR@10 by user type |
| `grid_search.py` | Grid search over alpha fusion weights |
| `main.py` | Main training + evaluation script |
| `README.md` | This documentation |

---

## How to Use

### 1. Prepare Your Data

You should prepare the following variables:

- `train_df`, `test_df`: Pandas DataFrames with columns:
  ```
  userId, movieId, rating, user_type
  ```
- `movie_bert_dict`: `dict[movieId] → 384-dim np.ndarray` from BERT
- `movie_poster_dict`: `dict[movieId] → 2048-dim np.ndarray` from ResNet
- `user2idx`, `item2idx`: ID-to-index mappings (optional for embeddings)

> These must be prepared in your own preprocessing pipeline (not included here).

---

### 2. Run the Main Pipeline

```python
from main import main
main(train_df, test_df, movie_bert_dict, movie_poster_dict, user2idx, item2idx)
```

This will:
- Train the CF model
- Train the CF + BERT model
- Perform `grid_search_bert_alpha` to find optimal fusion weight
- Train the CF + BERT + ResNet model
- Perform `grid_search_bert_resnet_alpha` to find best triple fusion weights

---

## Metrics

For both fusion strategies, two metrics are reported by user type:
- **RMSE** (Root Mean Squared Error)
- **MRR@10** (Mean Reciprocal Rank, top-10)

You can use these to compare models and validate improvement.

---

## Dependencies

Install the following packages:

```bash
pip install torch transformers scikit-learn pandas numpy surprise tqdm
```

---

## Maintainer

Developed by **Jersey** as part of a multimodal ML project at Copenhagen University.

