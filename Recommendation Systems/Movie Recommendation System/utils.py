"""
Movie Recommendation System — Utility Functions
Collaborative filtering on the MovieLens (small) explicit-rating dataset.
Two models are compared against a popularity baseline:
  - Item-based CF   : cosine similarity between items on mean-centred ratings
  - Matrix factorization : TruncatedSVD latent factors
Evaluation: RMSE (rating prediction) + Precision@k / Recall@k (top-k ranking).

This is an unsupervised / self-supervised setup — there is no external label;
we hold out known ratings and measure how well each model recovers them.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


# Column mapping for this dataset (other explicit projects reuse this file by
# editing only these three names + the loaders below).
UCOL, ICOL, RCOL = "userId", "movieId", "rating"


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_ratings(path="data/ratings.csv"):
    """Load ratings as a tidy frame with columns user / item / rating."""
    df = pd.read_csv(path)
    return df.rename(columns={UCOL: "user", ICOL: "item", RCOL: "rating"})[["user", "item", "rating"]]


def load_items(path="data/movies.csv", id_col="movieId", name_col="title"):
    """Load an item-id → title lookup Series."""
    items = pd.read_csv(path)
    return items.set_index(id_col)[name_col]


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def filter_popular(df, max_users=2000, max_items=2000, min_user=5, min_item=5):
    """
    Keep the most active users and most-rated items (for a tractable, dense-enough
    matrix), after dropping users/items with too few ratings. Returns filtered df.
    """
    vc_i = df["item"].value_counts()
    df = df[df["item"].isin(vc_i[vc_i >= min_item].index)]
    vc_u = df["user"].value_counts()
    df = df[df["user"].isin(vc_u[vc_u >= min_user].index)]
    top_items = df["item"].value_counts().head(max_items).index
    top_users = df["user"].value_counts().head(max_users).index
    return df[df["item"].isin(top_items) & df["user"].isin(top_users)]


def build_user_item(df):
    """Pivot a tidy ratings frame to a (users × items) matrix; unrated = NaN."""
    return df.pivot_table(index="user", columns="item", values="rating")


def train_test_split_ratings(df, test_size=0.2, seed=42):
    """
    Random hold-out of individual ratings. Rows whose user or item is absent from
    the training side are moved back to train so every test pair is predictable.
    """
    rng = np.random.RandomState(seed)
    mask = rng.rand(len(df)) < test_size
    train, test = df[~mask].copy(), df[mask].copy()
    train_users, train_items = set(train["user"]), set(train["item"])
    movable = ~test["user"].isin(train_users) | ~test["item"].isin(train_items)
    train = pd.concat([train, test[movable]])
    test = test[~movable]
    return train, test


def sparsity(matrix):
    """Fraction of empty cells in the user-item matrix."""
    return float(np.isnan(matrix.values).mean())


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class PopularityBaseline:
    """Predict each item's global mean rating (ignores the user)."""
    def fit(self, train_matrix):
        self.item_means = train_matrix.mean(axis=0)
        self.global_mean = float(np.nanmean(train_matrix.values))
        return self

    def predict(self, user, item):
        return self.item_means.get(item, self.global_mean)

    def score_all(self, user):
        return self.item_means.fillna(self.global_mean)


class ItemBasedCF:
    """Item-item cosine similarity on mean-centred ratings; weighted-average predict."""
    def fit(self, train_matrix):
        self.matrix = train_matrix
        self.user_means = train_matrix.mean(axis=1)
        centred = train_matrix.sub(self.user_means, axis=0).fillna(0.0)
        sim = cosine_similarity(centred.T)
        self.sim = pd.DataFrame(sim, index=train_matrix.columns, columns=train_matrix.columns)
        self.global_mean = float(np.nanmean(train_matrix.values))
        return self

    def predict(self, user, item, k=30):
        if user not in self.matrix.index or item not in self.sim.columns:
            return self.global_mean
        rated = self.matrix.loc[user].dropna()
        if rated.empty:
            return self.user_means.get(user, self.global_mean)
        sims = self.sim.loc[item, rated.index]
        top = sims.sort_values(ascending=False).head(k)
        top = top[top > 0]
        if top.sum() == 0:
            return self.user_means.get(user, self.global_mean)
        centred = rated[top.index] - self.user_means[user]
        return float(self.user_means[user] + (top * centred).sum() / top.sum())

    def score_all(self, user):
        """Vectorised predicted rating for every item (positive similarities only)."""
        items = self.matrix.columns
        if user not in self.matrix.index:
            return pd.Series(self.global_mean, index=items)
        rated = self.matrix.loc[user].dropna()
        um = self.user_means.get(user, self.global_mean)
        if rated.empty:
            return pd.Series(um, index=items)
        centred = (rated - um).values
        simpos = np.clip(self.sim.loc[:, rated.index].values, 0, None)
        num = simpos @ centred
        denom = simpos.sum(axis=1)
        pred = np.where(denom > 0, um + num / np.where(denom == 0, 1, denom), um)
        return pd.Series(pred, index=items)


class SVDRecommender:
    """TruncatedSVD matrix factorization on the mean-centred, zero-filled matrix."""
    def fit(self, train_matrix, n_components=20, seed=42):
        self.user_means = train_matrix.mean(axis=1)
        centred = train_matrix.sub(self.user_means, axis=0).fillna(0.0)
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)
        self.U = self.svd.fit_transform(centred)
        self.V = self.svd.components_
        self.recon = pd.DataFrame(self.U @ self.V, index=train_matrix.index,
                                  columns=train_matrix.columns).add(self.user_means, axis=0)
        self.global_mean = float(np.nanmean(train_matrix.values))
        return self

    def predict(self, user, item):
        if user in self.recon.index and item in self.recon.columns:
            return float(self.recon.loc[user, item])
        return self.global_mean

    def score_all(self, user):
        if user in self.recon.index:
            return self.recon.loc[user]
        return pd.Series(self.global_mean, index=self.recon.columns)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def rmse_on_test(model, test_df, **kw):
    """Root mean squared error of predicted vs actual on held-out rating pairs."""
    preds = np.array([model.predict(u, i, **kw) if kw else model.predict(u, i)
                      for u, i in zip(test_df["user"], test_df["item"])])
    actual = test_df["rating"].values
    return float(np.sqrt(np.mean((preds - actual) ** 2)))


def precision_recall_at_k(model, train_matrix, test_df, k=10, rel_threshold=4.0):
    """
    Ranking quality via top-k. For each user with held-out 'relevant' items
    (test rating >= rel_threshold), score every unseen item (vectorised), take the
    top-k, and measure precision@k and recall@k; returns the mean over users.
    """
    rel_by_user = (test_df[test_df["rating"] >= rel_threshold]
                   .groupby("user")["item"].apply(set).to_dict())
    precs, recs = [], []
    for user, relevant in rel_by_user.items():
        if user not in train_matrix.index:
            continue
        scores = model.score_all(user).copy()
        seen = train_matrix.loc[user].dropna().index
        scores = scores.drop(index=seen, errors="ignore")
        topk = set(scores.sort_values(ascending=False).head(k).index)
        hits = len(topk & relevant)
        precs.append(hits / k)
        recs.append(hits / len(relevant))
    return {"precision@k": round(np.mean(precs), 4), "recall@k": round(np.mean(recs), 4),
            "k": k, "n_users_evaluated": len(precs)}


def recommend(model, train_matrix, user, items_lookup=None, n=10):
    """Top-n unseen-item recommendations for a user, with optional title lookup."""
    if user not in train_matrix.index:
        return pd.DataFrame()
    scores = model.score_all(user).copy()
    seen = train_matrix.loc[user].dropna().index
    scores = scores.drop(index=seen, errors="ignore").sort_values(ascending=False).head(n)
    out = pd.DataFrame({"item": scores.index, "score": scores.values.round(3)})
    if items_lookup is not None and not out.empty:
        out.insert(1, "title", out["item"].map(items_lookup))
    return out
