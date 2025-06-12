
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class BERTMLP(nn.Module):
    def __init__(self, bert_dim, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        input_dim = bert_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(input_dim, hd))
            layers.append(nn.ReLU())
            input_dim = hd
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()

class FusionDataset(Dataset):
    def __init__(self, df, movie_bert_dict):
        self.user_ids = df['userId'].values
        self.movie_ids = df['movieId'].values
        self.ratings = df['rating'].values.astype(np.float32)
        self.movie_bert_dict = movie_bert_dict

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        mid = self.movie_ids[idx]
        emb = self.movie_bert_dict.get(mid, np.zeros(384))
        return uid, mid, torch.tensor(emb, dtype=torch.float32), torch.tensor(self.ratings[idx])

def train_nlp_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for _, _, bert_emb, rating in tqdm(loader, desc="Train NLP"):
        bert_emb, rating = bert_emb.to(device), rating.to(device)
        optimizer.zero_grad()
        pred = model(bert_emb)
        loss = loss_fn(pred, rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(rating)
    return (total_loss / len(loader.dataset)) ** 0.5

def fusion_predict(cf_model, nlp_model, user_ids, item_ids, user2idx, item2idx, movie_bert_dict, alpha, device):
    cf_scores = cf_model.predict_batch(user_ids, item_ids)
    bert_embs = np.stack([movie_bert_dict.get(mid, np.zeros(384)) for mid in item_ids])
    bert_embs_tensor = torch.tensor(bert_embs, dtype=torch.float32).to(device)
    nlp_scores = nlp_model(bert_embs_tensor).detach().cpu().numpy()
    return alpha * cf_scores + (1 - alpha) * nlp_scores

def eval_fusion_rmse_by_user_type(cf_model, nlp_model, test_df, user2idx, item2idx, movie_bert_dict, alpha, device="cuda"):
    results = {}
    for utype in ['regular', 'new']:
        subset = test_df[test_df['user_type'] == utype]
        if len(subset) == 0:
            results[f'{utype}_rmse'] = None
            continue
        user_ids = subset['userId'].values
        item_ids = subset['movieId'].values
        y_true = subset['rating'].values
        y_pred = fusion_predict(cf_model, nlp_model, user_ids, item_ids, user2idx, item2idx, movie_bert_dict, alpha, device)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        results[f'{utype}_rmse'] = rmse
    return results

def eval_fusion_mrr_by_user_type(cf_model, nlp_model, test_df, user2idx, item2idx, movie_bert_dict,
                                  alpha, device="cuda", k=10):
    results = {}
    candidate_movie_ids = np.array(test_df['movieId'].unique())
    for utype in ['regular', 'new']:
        subset = test_df[test_df['user_type'] == utype]
        if len(subset) == 0:
            results[f'{utype}_mrr'] = None
            continue
        user_ids = subset['userId'].unique()
        mrr = 0
        for uid in user_ids:
            gt_items = set(subset[subset['userId'] == uid]['movieId'])
            candidates = [mid for mid in candidate_movie_ids if mid not in gt_items]
            rec_scores = fusion_predict(cf_model, nlp_model,
                                        [uid] * len(candidates), candidates,
                                        user2idx, item2idx, movie_bert_dict,
                                        alpha, device)
            sorted_items = [x for _, x in sorted(zip(rec_scores, candidates), reverse=True)]
            for rank, iid in enumerate(sorted_items[:k], 1):
                if iid in gt_items:
                    mrr += 1 / rank
                    break
        results[f'{utype}_mrr'] = mrr / len(user_ids)
    return results
