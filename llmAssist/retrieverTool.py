import json
import os

import requests
from langchain.tools import tool
#Agent 사용 시 사용
class GetRetriever:
    @tool("Get Retriever Tool")
    def RetrieverSet(self) -> str:
        # Agent가 Tool을 사용 하기 위한 설명
        """
        Retrieve relevant documents or data based on self.retriever.invoke(query).
                
        Returns:
            str: The documents or information associated with the performed.
        """
        return GetRetriever.search(self.query)
    def __init__(self, retriever=None, query=None):
        self.retriever = retriever
        self.query = query

    def search(self):
        doc = self.retriever.invoke(self.query)
        result = ''
        for text in doc:
            result += text.page_content
        return f"\nSearch result: {result}\n"