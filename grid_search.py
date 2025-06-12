
import numpy as np
from itertools import product
from evaluation import eval_fusion3_rmse_by_user_type, eval_fusion3_mrr_by_user_type

def grid_search_bert_alpha(cf_model, nlp_model, test_df,
                           user2idx, item2idx, movie_bert_dict,
                           device="cuda", k=10,
                           alphas=np.arange(0, 1.01, 0.05),
                           lambda_mrr=0.7, mrr_scale=2.0):
    from cf_bert_fusion_model import fusion_predict
    results = []
    best_score = -np.inf
    best_alpha = None

    for alpha in alphas:
        mrr_res = nlp_model.eval_fusion_mrr_by_user_type(
            cf_model, nlp_model, test_df, user2idx, item2idx, movie_bert_dict,
            alpha, device, k
        )
        rmse_res = nlp_model.eval_fusion_rmse_by_user_type(
            cf_model, nlp_model, test_df, user2idx, item2idx, movie_bert_dict,
            alpha, device
        )

        mrr = np.mean([v for v in mrr_res.values() if v is not None])
        rmse = np.mean([v for v in rmse_res.values() if v is not None])
        score = lambda_mrr * mrr + (1 - lambda_mrr) * (1 - rmse / mrr_scale)

        results.append({'alpha': alpha, 'mrr': mrr, 'rmse': rmse, 'score': score})
        if score > best_score:
            best_score = score
            best_alpha = alpha

    return results, best_alpha, best_score

def grid_search_bert_resnet_alpha(cf_model, fusion_model, test_df,
                                  user2idx, item2idx, movie_bert_dict, movie_poster_dict,
                                  device="cuda", k=10,
                                  alpha_range=np.arange(0, 1.01, 0.1),
                                  lambda_mrr=0.7, mrr_scale=4.0):
    results = []
    best_score = -np.inf
    best_alpha = None

    combos = [
        (a, b, 1 - a - b)
        for a, b in product(alpha_range, repeat=2)
        if 0 <= a + b <= 1
    ]

    for alpha_cf, alpha_review, alpha_poster in combos:
        mrr_res = eval_fusion3_mrr_by_user_type(
            cf_model, fusion_model, test_df,
            user2idx, item2idx, movie_bert_dict, movie_poster_dict,
            alpha_cf, alpha_review, alpha_poster, device, k
        )
        rmse_res = eval_fusion3_rmse_by_user_type(
            cf_model, fusion_model, test_df,
            user2idx, item2idx, movie_bert_dict, movie_poster_dict,
            alpha_cf, alpha_review, alpha_poster, device
        )

        mrr = np.mean([v for v in mrr_res.values() if v is not None])
        rmse = np.mean([v for v in rmse_res.values() if v is not None])
        score = lambda_mrr * mrr + (1 - lambda_mrr) * (1 - rmse / mrr_scale)

        results.append({
            'alpha_cf': alpha_cf,
            'alpha_review': alpha_review,
            'alpha_poster': alpha_poster,
            'mrr': mrr,
            'rmse': rmse,
            'score': score
        })

        if score > best_score:
            best_score = score
            best_alpha = (alpha_cf, alpha_review, alpha_poster)

    return results, best_alpha, best_score
