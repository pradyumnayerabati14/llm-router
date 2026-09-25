"""Routers: every one maps a prompt to P(the strong model is needed).

A router only ever sees the query. It never sees a model's answer, because at
serving time no answer exists yet - that is the whole point of routing (section
3.1 of the paper). The routing rule itself lives in ``metrics.route``:

    send to the strong model  <=>  P(strong wins | query) >= alpha
"""

from __future__ import annotations

import numpy as np


class RandomRouter:
    """Baseline: ignore the query, score at random.

    Not a straw man. Combined with a threshold it sends a controllable fraction
    of traffic to the strong model, which is exactly what a company would do
    without a router ("send 30% of traffic to GPT-4"). Every learned router has
    to beat this to be worth anything.
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomRouter:
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.rng.random(len(X))


class SimilarityWeightedRouter:
    """Similarity-weighted Bradley-Terry ranking (paper section 4.2, eq. 9-10).

    No training. For a new query we look at the labelled training queries,
    weight each by how similar it is to the new one, and take the weighted
    average of their labels.

    Why a weighted average is the Bradley-Terry answer: with only two players
    the model has a single free parameter d = xi_strong - xi_weak, and the
    weighted log-likelihood in eq. 10 is maximised at

        sigmoid(d) = sum(w_i * y_i) / sum(w_i)

    so the win probability *is* the weighted mean label. Cheap to reason about,
    but inference cost grows with the training set, which is why the paper
    reports it as the slowest router (Table 7: 2.9 requests/second).
    """

    name = "sw_ranking"

    def __init__(self, gamma: float = 10.0, top_k: int = 64) -> None:
        self.gamma = gamma  # exponential weighting; the paper uses gamma = 10
        self.top_k = top_k  # only the k nearest neighbours vote

    def fit(self, X: np.ndarray, y: np.ndarray) -> SimilarityWeightedRouter:
        self.X = X
        self.y = y
        return self

    def predict_proba(self, X: np.ndarray, batch: int = 512) -> np.ndarray:
        out = np.empty(len(X), dtype=np.float32)
        for start in range(0, len(X), batch):
            chunk = X[start : start + batch]
            sims = chunk @ self.X.T  # cosine similarity (vectors are normalised)
            # Scale by the best match in the batch, as in eq. 9, so the weights
            # do not collapse when every neighbour is only loosely related.
            sims = sims / np.clip(sims.max(axis=1, keepdims=True), 1e-6, None)
            idx = np.argpartition(-sims, self.top_k, axis=1)[:, : self.top_k]
            near = np.take_along_axis(sims, idx, axis=1)
            weights = self.gamma ** (1.0 + near)
            labels = self.y[idx]
            out[start : start + batch] = (weights * labels).sum(1) / weights.sum(1)
        return out


class LogisticRouter:
    """Logistic regression on frozen embeddings, trained with eq. 10's loss.

    This stands in for the paper's BERT and Llama-3-8B classifiers. Those
    fine-tune the whole encoder on 2x24GB and 8xA100 GPUs respectively; here the
    encoder stays frozen and we train only the final layer, which takes seconds
    on a laptop. Same idea, a small fraction of the capacity.

    Trained on soft targets (y = 1.0 / 0.5 / 0.0) with binary cross-entropy, so
    a tie pulls the prediction toward 0.5 instead of being thrown away.
    """

    name = "logistic"

    def __init__(self, epochs: int = 30, lr: float = 1e-2, weight_decay: float = 1e-4) -> None:
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRouter:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        yb = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)

        self.layer = torch.nn.Linear(X.shape[1], 1).to(device)
        opt = torch.optim.AdamW(self.layer.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self.layer(xb), yb)
            loss.backward()
            opt.step()
        self.final_loss = float(loss.detach())
        self.device = device
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32, device=self.device)
            return torch.sigmoid(self.layer(xb)).squeeze(1).cpu().numpy()

    def save(self, path) -> None:
        import torch

        torch.save({"state_dict": self.layer.state_dict(), "dim": self.layer.in_features}, path)

    @classmethod
    def load(cls, path) -> LogisticRouter:
        import torch

        blob = torch.load(path, map_location="cpu")
        obj = cls()
        obj.layer = torch.nn.Linear(blob["dim"], 1)
        obj.layer.load_state_dict(blob["state_dict"])
        obj.device = "cpu"
        return obj
