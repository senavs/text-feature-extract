import abc
from typing import Sequence


class BaseDType(abc.ABC):

    @abc.abstractmethod
    def fit(self, x: Sequence[str]): ...

    @abc.abstractmethod
    def predict(self, x: Sequence[str]) -> Sequence[Sequence[int]]: ...

    @abc.abstractmethod
    def fit_predict(self, x: Sequence[str]) -> Sequence[Sequence[int]]: ...

    def __repr__(self):
        return f'<class {type(self).__name__}>'
