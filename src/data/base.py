import numpy as np
from tqdm.auto import tqdm
import iisignature
from joblib import Parallel, delayed

import utils


class Price(object):
    """Base class for data (i.e. price models)."""

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _sig(path, order):
        return np.r_[1., iisignature.sig(utils.transform(path), order)]


    def generate(self):
        """Generate a sample path."""

        raise NotImplementedError("Generator not implemented")

    def _generate(self, seed):
        np.random.seed(seed)
        return self.generate()

    def _generate_paths(self, n_paths=1000):
        paths = Parallel(n_jobs=-1)(delayed(self._generate)(seed) \
                                    for seed in tqdm(range(n_paths), desc="Building paths"))

        return paths

    def build(self, *args, n_paths=1000, order=6, **kwargs):
        """Builds paths and ES."""


        # Create paths
        paths = self._generate_paths(*args, n_paths=n_paths, **kwargs)

        signals = None
        if isinstance(paths[0], tuple):
            signals = [path[0] for path in paths]
            paths = [path[1] for path in paths]


        # Compute signatures
        sigs = Parallel(n_jobs=-1)(delayed(self._sig)(path, order) \
                                   for path in tqdm(paths, desc="Computing signatures"))

        # Compute ES
        ES = np.mean(sigs, axis=0)

        if signals is None:
            return np.array(paths), ES
        else:
            return np.array(signals), np.array(paths), ES
