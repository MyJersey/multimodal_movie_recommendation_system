
from cf_model import CFRecommender
from cf_bert_fusion_model import BERTMLP, FusionDataset, train_nlp_one_epoch, eval_fusion_rmse_by_user_type, eval_fusion_mrr_by_user_type
from cf_bert_resnet_fusion_model import TwoBranchMLP, FusionDataset3, train_fusion3_one_epoch
from evaluation import eval_fusion3_rmse_by_user_type, eval_fusion3_mrr_by_user_type
from grid_search import grid_search_bert_alpha, grid_search_bert_resnet_alpha

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

def fusion_collate_fn(batch):
    user_ids, movie_ids, bert_embs, ratings = zip(*batch)
    return list(user_ids), list(movie_ids), torch.stack(bert_embs), torch.tensor(ratings)

def fusion3_collate_fn(batch):
    user_ids, movie_ids, review_embs, poster_embs, ratings = zip(*batch)
    return list(user_ids), list(movie_ids), torch.stack(review_embs), torch.stack(poster_embs), torch.tensor(ratings)

def main(train_df, test_df, movie_bert_dict, movie_poster_dict, user2idx, item2idx):
    # Train CF Model
    cf_model = CFRecommender()
    cf_model.fit(train_df)

    # Train BERT model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_dim = len(next(iter(movie_bert_dict.values())))
    nlp_model = BERTMLP(bert_dim).to(device)
    optimizer = torch.optim.Adam(nlp_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    train_ds = FusionDataset(train_df, movie_bert_dict)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, collate_fn=fusion_collate_fn)

    for epoch in range(5):
        rmse = train_nlp_one_epoch(nlp_model, train_loader, optimizer, loss_fn, device)
        print(f"[BERT Epoch {epoch+1}] Train RMSE: {rmse:.4f}")

    # Search best alpha for CF+BERT
    bert_results, best_alpha_bert, best_score_bert = grid_search_bert_alpha(
        cf_model, nlp_model, test_df, user2idx, item2idx, movie_bert_dict, device=device
    )
    print(f"Best alpha for CF+BERT: {best_alpha_bert}, score: {best_score_bert:.4f}")

    # Train BERT + ResNet model
    review_dim = bert_dim
    poster_dim = len(next(iter(movie_poster_dict.values())))
    fusion_model = TwoBranchMLP(review_dim, poster_dim).to(device)
    optimizer2 = torch.optim.Adam(fusion_model.parameters(), lr=1e-3)
    train_ds3 = FusionDataset3(train_df, movie_bert_dict, movie_poster_dict)
    train_loader3 = DataLoader(train_ds3, batch_size=1024, shuffle=True, collate_fn=fusion3_collate_fn)

    for epoch in range(5):
        rmse = train_fusion3_one_epoch(fusion_model, train_loader3, optimizer2, loss_fn, device)
        print(f"[Fusion Epoch {epoch+1}] Train RMSE: {rmse:.4f}")

    # Search best alpha for CF+BERT+ResNet
    resnet_results, best_alpha_resnet, best_score_resnet = grid_search_bert_resnet_alpha(
        cf_model, fusion_model, test_df, user2idx, item2idx,
        movie_bert_dict, movie_poster_dict, device=device
    )
    print(f"Best alphas for CF+BERT+ResNet: {best_alpha_resnet}, score: {best_score_resnet:.4f}")

if __name__ == "__main__":
    print("This is a main entry point. Please import and call main(...) with prepared data.")
