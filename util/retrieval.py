import numpy as np


class RetrievalIndex:
    def __init__(self, embeddings, labels, diagnoses, ids):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=int)
        self.diagnoses = np.asarray(diagnoses)
        self.ids = np.asarray(ids)

    def __len__(self):
        return len(self.ids)

    def search(self, queries, k):
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        sims = queries @ self.embeddings.T
        k_eff = min(int(k), len(self))
        order = np.argsort(-sims, axis=1, kind='stable')[:, :k_eff]
        return order, np.take_along_axis(sims, order, axis=1)

    def neighbours(self, query, k):
        order, sims = self.search(query, k)
        return [
            {'id': str(self.ids[i]), 'diagnosis': str(self.diagnoses[i]), 'label': int(self.labels[i]), 'similarity': float(s)}
            for i, s in zip(order[0], sims[0])
        ]


def majority_label(labels, similarities):
    labels = np.asarray(labels, dtype=int)
    similarities = np.asarray(similarities, dtype=float)
    counts = np.bincount(labels, minlength=2)
    if counts[0] == counts[1]:
        return int(similarities[labels == 1].sum() > similarities[labels == 0].sum())
    return int(np.argmax(counts))


def class_fraction(neighbours, label):
    if not neighbours:
        return float('nan')
    return float(np.mean([n['label'] == label for n in neighbours]))


def precision_at_k(index, queries, query_labels, k):
    order, _ = index.search(queries, k)
    hits = index.labels[order] == np.asarray(query_labels, dtype=int)[:, None]
    return float(hits.mean()) if hits.size else float('nan')
