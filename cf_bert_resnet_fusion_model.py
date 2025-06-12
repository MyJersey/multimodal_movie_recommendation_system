
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class TwoBranchMLP(nn.Module):
    def __init__(self, review_dim, poster_dim, hidden_dim=128):
        super().__init__()
        self.review_mlp = nn.Sequential(
            nn.Linear(review_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.poster_mlp = nn.Sequential(
            nn.Linear(poster_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, review_emb, poster_emb):
        review_score = self.review_mlp(review_emb)
        poster_score = self.poster_mlp(poster_emb)
        return (review_score + poster_score).squeeze()

class FusionDataset3(Dataset):
    def __init__(self, df, movie_bert_dict, movie_poster_dict):
        self.user_ids = df['userId'].values
        self.movie_ids = df['movieId'].values
        self.ratings = df['rating'].values.astype(np.float32)
        self.movie_bert_dict = movie_bert_dict
        self.movie_poster_dict = movie_poster_dict

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        mid = self.movie_ids[idx]
        review_emb = self.movie_bert_dict.get(mid, np.zeros(384))
        poster_emb = self.movie_poster_dict.get(mid, np.zeros(2048))
        return uid, mid, torch.tensor(review_emb, dtype=torch.float32), torch.tensor(poster_emb, dtype=torch.float32), torch.tensor(self.ratings[idx])

def fusion3_predict(cf_model, model, user_ids, item_ids,
                    user2idx, item2idx, movie_bert_dict, movie_poster_dict,
                    alpha_cf=0.7, alpha_review=0.15, alpha_poster=0.15, device='cuda', mini_batch_size=4096):
    preds = []
    n = len(user_ids)
    for i in range(0, n, mini_batch_size):
        uid_batch = user_ids[i:i + mini_batch_size]
        mid_batch = item_ids[i:i + mini_batch_size]
        cf_scores = cf_model.predict_batch(uid_batch, mid_batch)
        review_embs = torch.tensor([movie_bert_dict.get(mid, np.zeros(384)) for mid in mid_batch], dtype=torch.float32).to(device)
        poster_embs = torch.tensor([movie_poster_dict.get(mid, np.zeros(2048)) for mid in mid_batch], dtype=torch.float32).to(device)
        with torch.no_grad():
            fusion_scores = model(review_embs, poster_embs).cpu().numpy()
        final_scores = alpha_cf * cf_scores + alpha_review * fusion_scores + alpha_poster * fusion_scores
        preds.extend(final_scores)
    return np.array(preds)

def train_fusion3_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for _, _, review_emb, poster_emb, rating in tqdm(loader, desc="Train Fusion3"):
        review_emb, poster_emb, rating = review_emb.to(device), poster_emb.to(device), rating.to(device)
        optimizer.zero_grad()
        pred = model(review_emb, poster_emb)
        loss = loss_fn(pred, rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(rating)
    return (total_loss / len(loader.dataset)) ** 0.5
