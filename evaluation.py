
import numpy as np
import pandas as pd

def eval_fusion3_rmse_by_user_type(cf_model, fusion_model, test_df, user2idx, item2idx,
                                   movie_bert_dict, movie_poster_dict,
                                   alpha_cf=0.7, alpha_review=0.15, alpha_poster=0.15,
                                   device="cuda", batch_size=2048):
    results = {}
    for utype in ['regular', 'new']:
        subset = test_df[test_df['user_type'] == utype]
        if len(subset) == 0:
            results[f'{utype}_rmse'] = None
            continue
        user_ids = subset['userId'].values
        item_ids = subset['movieId'].values
        y_true = subset['rating'].values
        from cf_bert_resnet_fusion_model import fusion3_predict
        y_pred = fusion3_predict(cf_model, fusion_model, user_ids, item_ids,
                                 user2idx, item2idx, movie_bert_dict, movie_poster_dict,
                                 alpha_cf, alpha_review, alpha_poster, device, batch_size)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        results[f'{utype}_rmse'] = rmse
    return results

def eval_fusion3_mrr_by_user_type(cf_model, fusion_model, test_df, user2idx, item2idx,
                                  movie_bert_dict, movie_poster_dict,
                                  alpha_cf=0.7, alpha_review=0.15, alpha_poster=0.15,
                                  device="cuda", k=10):
    results = {}
    candidate_movie_ids = np.array(test_df['movieId'].unique())
    from cf_bert_resnet_fusion_model import fusion3_predict

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
            scores = fusion3_predict(cf_model, fusion_model,
                                     [uid] * len(candidates), candidates,
                                     user2idx, item2idx, movie_bert_dict, movie_poster_dict,
                                     alpha_cf, alpha_review, alpha_poster, device)
            sorted_items = [x for _, x in sorted(zip(scores, candidates), reverse=True)]
            for rank, iid in enumerate(sorted_items[:k], 1):
                if iid in gt_items:
                    mrr += 1 / rank
                    break
        results[f'{utype}_mrr'] = mrr / len(user_ids)
    return results
