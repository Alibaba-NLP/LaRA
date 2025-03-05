from typing import Optional, List, Mapping, Any, Dict
from abc import ABC, abstractmethod
import os
import logging
import json
from tqdm import tqdm
import concurrent.futures
import time

from llama_index.core.schema import NodeWithScore, BaseNode, MetadataMode
from llama_index.core.indices.query.schema import QueryBundle

from query_engine import BaseQueryEngine
from llama_index.core import Settings

def multi_call(call_func, inp):
    while True:
        try:
            return call_func(*inp)
        except Exception as e:
            time.sleep(5)
            print(f"Error: {e}, retrying...")
            continue

class BaseSearcher(ABC):
    show_progress = True

    @abstractmethod
    def load_query_engine(self, nodes: List[BaseNode]) -> BaseQueryEngine:
        """
        Load the query engine from nodes.
        """
        pass

    def __init__(self, config, nodes):
        self.remove_if_exists = config.get("remove_if_exists", False)
        self.thread_num = config.get("thread_num", 1)
        self.excluded_embed_metadata_keys = config.get('excluded_embed_metadata_keys', None)
        self.excluded_llm_metadata_keys = config.get('excluded_llm_metadata_keys', None)
        self.text_template = config.get('text_template', None)
        self.metadata_template = config.get('metadata_template', None)
        self.metadata_seperator = config.get('metadata_seperator', None)
        Settings.llm = None
        self.nodes = nodes
        self.query_engine = None

    def process(self, query):
        self.query_engine = self.load_query_engine(self.nodes)
        recall_results_list = []
        query_bundle = QueryBundle(query_str=query)
        recall_results = multi_call(self.query_engine.retrieve, [query_bundle])
        return recall_results

    def nodes2dict(self, nodes: NodeWithScore) -> List[Dict[str, Any]]:
        resp_dict = {
            "response": None,
            "source_nodes": [],
            "metadata": None
        }
        for node in nodes:
            resp_dict["source_nodes"].append(node.to_dict())
        return resp_dict