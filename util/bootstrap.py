import numpy as np
import pandas as pd


def summarize(point, replicates, ci):
    alpha = (100.0 - ci) / 2.0
    if isinstance(point, dict):
        return {k: summarize(point[k], [r.get(k, float('nan')) for r in replicates], ci) for k in point}
    reps = np.asarray([r for r in replicates], dtype=float)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0 or point is None or not np.isfinite(point):
        return {'point': float(point) if point is not None else float('nan'), 'lo': float('nan'), 'hi': float('nan'), 'n_valid': int(reps.size)}
    return {
        'point': float(point),
        'lo': float(np.percentile(reps, alpha)),
        'hi': float(np.percentile(reps, 100.0 - alpha)),
        'n_valid': int(reps.size),
    }


class ClusterBootstrap:
    def __init__(self, clusters, n_boot, seed, ci=95.0):
        self.uniques = pd.Index(sorted(set(pd.Series(clusters).astype(str))))
        self.n_boot = int(n_boot)
        self.ci = float(ci)
        groups = np.array([u.split(':')[0] for u in self.uniques])
        rng = np.random.default_rng(seed)
        self.multiplicity = np.zeros((self.n_boot, len(self.uniques)), dtype=np.float64)
        for group in np.unique(groups):
            idx = np.flatnonzero(groups == group)
            m = len(idx)
            self.multiplicity[:, idx] = rng.multinomial(m, np.full(m, 1.0 / m), size=self.n_boot)

    @property
    def n_clusters(self):
        return len(self.uniques)

    def codes(self, clusters):
        codes = self.uniques.get_indexer(pd.Series(clusters).astype(str))
        if (codes < 0).any():
            raise KeyError('Cluster outside the bootstrap universe')
        return codes

    def run(self, fn):
        point = fn(np.ones(len(self.uniques), dtype=np.float64))
        replicates = [fn(self.multiplicity[b]) for b in range(self.n_boot)]
        return summarize(point, replicates, self.ci)
