"""
Music Recommendation System — Utility Functions
Implicit-feedback collaborative filtering on the Last.fm (hetrec2011) dataset:
the signal is how many times a user played an artist (no explicit ratings).

Because feedback is implicit there is no rating to predict, so we evaluate with
ranking metrics only — Precision@k / Recall@k on held-out interactions.
Two models are compared against a popularity baseline:
  - Item-based CF       : cosine similarity between artists on the play matrix
  - Matrix factorization : TruncatedSVD latent factors
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

UCOL, ICOL, WCOL = "userID", "artistID", "weight"


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_interactions(path="data/user_artists.csv"):
    """Load implicit interactions as user / item / weight (play count)."""
    df = pd.read_csv(path)
    return df.rename(columns={UCOL: "user", ICOL: "item", WCOL: "weight"})[["user", "item", "weight"]]


def load_items(path="data/artists.csv", id_col="id", name_col="name"):
    """Load item-id → name lookup Series."""
    return pd.read_csv(path).set_index(id_col)[name_col]


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def filter_popular(df, max_users=2000, max_items=2000, min_user=5, min_item=5):
    """Keep active users and popular items (tractable, dense-enough matrix)."""
    vc_i = df["item"].value_counts()
    df = df[df["item"].isin(vc_i[vc_i >= min_item].index)]
    vc_u = df["user"].value_counts()
    df = df[df["user"].isin(vc_u[vc_u >= min_user].index)]
    top_items = df["item"].value_counts().head(max_items).index
    top_users = df["user"].value_counts().head(max_users).index
    return df[df["item"].isin(top_items) & df["user"].isin(top_users)]


def build_matrix(df, log=True):
    """
    Pivot to a (users × items) implicit matrix; missing = 0. Play counts span several
    orders of magnitude, so we log1p-compress them into a confidence weight by default.
    """
    m = df.pivot_table(index="user", columns="item", values="weight", fill_value=0.0)
    if log:
        m = np.log1p(m)
    return m


def train_test_split_interactions(df, test_size=0.2, seed=42, min_items=2):
    """
    Per-user hold-out for ranking: for each user with >= min_items interactions, move a
    `test_size` fraction of their items to the test set (kept out of the train matrix).
    """
    rng = np.random.RandomState(seed)
    train_parts, test_parts = [], []
    for user, grp in df.groupby("user"):
        if len(grp) < min_items:
            train_parts.append(grp); continue
        n_test = max(1, int(round(len(grp) * test_size)))
        test_idx = rng.choice(grp.index, size=min(n_test, len(grp) - 1), replace=False)
        test_parts.append(grp.loc[test_idx])
        train_parts.append(grp.drop(index=test_idx))
    return pd.concat(train_parts), pd.concat(test_parts)


def sparsity(matrix):
    return float((matrix.values == 0).mean())


# ─────────────────────────────────────────────
# Models (implicit)
# ─────────────────────────────────────────────

class PopularityBaseline:
    """Score every item by its total interaction weight (same ranking for all users)."""
    def fit(self, matrix):
        self.item_pop = matrix.sum(axis=0)
        return self

    def score_all(self, user):
        return self.item_pop


class ItemBasedCF:
    """Item-item cosine similarity on the implicit matrix; score = sims · user vector."""
    def fit(self, matrix):
        self.matrix = matrix
        sim = cosine_similarity(matrix.T.values)
        np.fill_diagonal(sim, 0.0)
        self.sim = pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)
        return self

    def score_all(self, user):
        if user not in self.matrix.index:
            return pd.Series(0.0, index=self.matrix.columns)
        user_vec = self.matrix.loc[user].values
        return pd.Series(self.sim.values @ user_vec, index=self.matrix.columns)


class SVDRecommender:
    """TruncatedSVD latent-factor reconstruction of the implicit matrix."""
    def fit(self, matrix, n_components=30, seed=42):
        self.matrix = matrix
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)
        U = self.svd.fit_transform(matrix.values)
        self.recon = pd.DataFrame(U @ self.svd.components_, index=matrix.index, columns=matrix.columns)
        return self

    def score_all(self, user):
        if user in self.recon.index:
            return self.recon.loc[user]
        return pd.Series(0.0, index=self.recon.columns)


# ─────────────────────────────────────────────
# Evaluation (ranking only — implicit feedback)
# ─────────────────────────────────────────────

def precision_recall_at_k(model, train_matrix, test_df, k=10):
    """
    For each user, 'relevant' = their held-out items. Score every item not in train,
    take the top-k, and average precision@k / recall@k over users.
    """
    rel_by_user = test_df.groupby("user")["item"].apply(set).to_dict()
    precs, recs = [], []
    for user, relevant in rel_by_user.items():
        if user not in train_matrix.index:
            continue
        scores = model.score_all(user).copy()
        seen = train_matrix.loc[user]
        seen = seen[seen > 0].index
        scores = scores.drop(index=seen, errors="ignore")
        relevant = relevant & set(scores.index)
        if not relevant:
            continue
        topk = set(scores.sort_values(ascending=False).head(k).index)
        hits = len(topk & relevant)
        precs.append(hits / k)
        recs.append(hits / len(relevant))
    return {"precision@k": round(np.mean(precs), 4), "recall@k": round(np.mean(recs), 4),
            "k": k, "n_users_evaluated": len(precs)}


def recommend(model, train_matrix, user, items_lookup=None, n=10):
    """Top-n recommended items the user hasn't interacted with."""
    if user not in train_matrix.index:
        return pd.DataFrame()
    scores = model.score_all(user).copy()
    seen = train_matrix.loc[user]
    seen = seen[seen > 0].index
    scores = scores.drop(index=seen, errors="ignore").sort_values(ascending=False).head(n)
    out = pd.DataFrame({"item": scores.index, "score": scores.values.round(3)})
    if items_lookup is not None and not out.empty:
        out.insert(1, "name", out["item"].map(items_lookup))
    return out
