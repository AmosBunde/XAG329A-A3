from typing import Dict, Any

from xag329a_a3.api_manager import APIManager
from xag329a_a3.multi_lm_agent import MultiLMAgent


# Subclassing MultiLMAgent means `generate`, `decompose_query`, and
# `iterative_refine` are available here as building blocks for `research`.
class DeepResearchAgent(MultiLMAgent):
    def __init__(self, api_manager: APIManager):
        super().__init__(api_manager)

    def research(self, query: str) -> Dict[str, Any]:
        """
        Conducts deep research on a given query.

        Args:
            query: Complex research question to investigate

        Returns:
            Dictionary containing:
            - report: str, synthesized findings with citations
            - sources: List[str], list of source URLs or references
        """
        ### TODO: YOUR CODE STARTS HERE
        ### TODO: YOUR CODE ENDS HERE
