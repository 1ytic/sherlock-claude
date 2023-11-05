import requests
from typing import List, Optional
from haystack import Document
from haystack.nodes.search_engine.base import SearchEngine


class BraveEngine(SearchEngine):
    """
    Search engine using Brave API.
    """

    def __init__(self, api_key: str, top_k: Optional[int] = 10):
        """
        :param api_key: API key for the Brave API.
        :param top_k: Number of documents to return.
        For example, you can set 'num' to 20 to increase the number of search results.
        """
        super().__init__()
        self.api_key = api_key
        self.top_k = top_k

    def search(self, query: str, **kwargs) -> List[Document]:
        """
        :param query: Query string.
        :param kwargs: Additional parameters passed to the Brave API, such as top_k.
        :return: List[Document]
        """
        top_k = kwargs.pop("top_k", self.top_k)

        url = f"https://api.search.brave.com/res/v1/web/search?count={top_k}&q={query}"

        headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'Content-Type': 'application/json',
            'X-Subscription-Token': self.api_key
        }

        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Error while querying {self.__class__.__name__}: {response.text}")

        documents = []

        try:
            content = response.json()
            for position, result in enumerate(content["web"]["results"]):
                result["position"] = position
                result["link"] = result["url"]
                document = Document.from_dict(result, field_map={"description": "content"})
                documents.append(document)
        except Exception as e:
            return []        

        return self.score_results(documents, False)
