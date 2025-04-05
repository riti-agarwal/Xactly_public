from abc import ABC, abstractmethod

class SamplingStrategyBase(ABC):
    def __init__(self, itemCollection) -> None:
        # ItemCollection is a mongo db collection object, which you can use to make queries with
        super().__init__()

    @abstractmethod
    def getKClosestItems(self, query: str, trendingItems: list[str], previouslyBoughtItems: list[str]) -> list[str]:
        """
        Function that takes in a single query, a list of trending items and a list of previously bought items, and 
        in turn returns a list of itemIds that are relevant to the search.
        """
        raise NotImplementedError
