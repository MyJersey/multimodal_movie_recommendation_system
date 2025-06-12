
from surprise import SVD, Dataset, Reader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class CFRecommender:
    def __init__(self, n_factors=50, n_epochs=20, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.algo = None
        self.user_hist = None
        self.movie_ids = None

    def fit(self, df):
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.algo = SVD(n_factors=self.n_factors, n_epochs=self.n_epochs, random_state=self.random_state)
        self.algo.fit(trainset)

        self.user_hist = df.groupby('userId')['movieId'].apply(set).to_dict()
        self.movie_ids = set(df['movieId'].unique())

    def predict(self, user_id, item_id):
        return self.algo.predict(user_id, item_id).est

    def predict_batch(self, user_ids, item_ids):
        return np.array([self.predict(u, i) for u, i in zip(user_ids, item_ids)])

    def recommend(self, user_ids, N=10, candidate_movie_ids=None, return_score=False):
        if candidate_movie_ids is None:
            candidate_movie_ids = self.movie_ids

        recommendations = {}
        for uid in user_ids:
            seen = self.user_hist.get(uid, set())
            candidates = list(candidate_movie_ids - seen)
            preds = [(iid, self.predict(uid, iid)) for iid in candidates]
            preds.sort(key=lambda x: x[1], reverse=True)
            top_n = preds[:N]
            if return_score:
                recommendations[uid] = top_n
            else:
                recommendations[uid] = [iid for iid, _ in top_n]
        return recommendations

    def rmse(self, test_df):
        preds = [self.predict(row['userId'], row['movieId']) for _, row in test_df.iterrows()]
        return mean_squared_error(test_df['rating'], preds, squared=False)

    def mrr_at_k(self, test_df, recommendations, k=10):
        mrr = 0
        for uid, group in test_df.groupby('userId'):
            true_items = set(group['movieId'])
            recs = [iid for iid, _ in recommendations.get(uid, [])][:k]
            for rank, iid in enumerate(recs, 1):
                if iid in true_items:
                    mrr += 1 / rank
                    break
        return mrr / len(test_df['userId'].unique())
