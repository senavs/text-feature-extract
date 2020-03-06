from typing import Sequence, KeysView
from collections import Counter

import numpy as np

from dtypes.base import BaseDType


class BagOfWordSequenced(BaseDType):

    def __init__(self, n_vocabulary: int = None):
        self._n_vocabulary = n_vocabulary
        self._vocabulary = {}

    def fit(self, x: Sequence[str]):
        counter = Counter()

        for text in x:
            counter.update(text.split())

        for index, (word, _) in enumerate(reversed(counter.most_common()), start=1):
            self._vocabulary[word] = index

    def predict(self, x: Sequence[str]) -> Sequence[Sequence[int]]:
        output = np.zeros((len(x), self.n_vocabulary), dtype=np.int32)

        for i, sequence in enumerate(x):
            for j, word in zip(range(self.n_vocabulary), sequence.split()):
                output[i, j] = self.vocabulary_indexed.get(word, 0)

        return output

    def fit_predict(self, x: Sequence[str]) -> Sequence[Sequence[int]]:
        self.fit(x)
        output = self.predict(x)
        return output

    @property
    def n_vocabulary(self) -> int:
        return self._n_vocabulary if self._n_vocabulary else len(self._vocabulary)

    @property
    def vocabulary(self) -> KeysView:
        return self._vocabulary.keys()

    @property
    def vocabulary_indexed(self) -> dict:
        return self._vocabulary
