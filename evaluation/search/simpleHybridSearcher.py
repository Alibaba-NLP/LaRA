import re
import json
import os
from typing import Optional, List, Mapping, Any, Dict

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import RecursiveRetriever, AutoMergingRetriever
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter, get_leaf_nodes
from llama_index.core.schema import IndexNode, NodeWithScore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.storage.docstore import SimpleDocumentStore

from search.baseSearcher import BaseSearcher
from search.simpleHybridRetriever import SimpleHybridRetriever
from query_engine import RetrieverQueryEngine

class Small2big(BaseNodePostprocessor):

    all_nodes_dict: dict

    @classmethod
    def class_name(cls) -> str:
        return "Small2big"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        reordered_nodes: List[NodeWithScore] = []
        for i, node in enumerate(nodes):
            node2add = NodeWithScore(node=self.all_nodes_dict[node.node_id], score=node.score)
            reordered_nodes.append(node2add)
        return reordered_nodes

class AutoMerge(BaseNodePostprocessor):
    auto_merge_retriever: AutoMergingRetriever
    auto_merge_topk: int

    @classmethod
    def class_name(cls) -> str:
        return "AutoMerge"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        initial_nodes = nodes[:self.auto_merge_topk]
        tail_nodes = nodes[self.auto_merge_topk:]

        cur_nodes, is_changed = self.auto_merge_retriever._try_merging(initial_nodes)
        while is_changed:
            cur_nodes, is_changed = self.auto_merge_retriever._try_merging(cur_nodes)

        # sort by similarity
        cur_nodes.sort(key=lambda x: x.get_score(), reverse=True)
        return cur_nodes + tail_nodes

class SimpleHybridSearcher(BaseSearcher):
    def __init__(self, config, nodes):
        self.rerank_size = config["rerank_size"]
        self.vector_ratio = config["vector_ratio"]
        self.embed_model_name = config['embed_model_name']
        self.rerank_model = config["rerank_model"]
        self.regenerate_emb = config.get("regenerate_emb", False)
        self.use_async = config.get("use_async", False)
        self.recursive_retrieve = config.get("recursive_retrieve", None)
        self.recursive_rerank = config.get("recursive_rerank", None)
        self.enable_auto_merge = config.get("enable_auto_merge", False)
        self.auto_merge_ratio = config.get("auto_merge_ratio", 0.5)
        self.auto_merge_topk = config.get("auto_merge_topk", 10)
        self.all_nodes_dict = None
        self.auto_merge_retriever = None
        super(SimpleHybridSearcher, self).__init__(config, nodes)

    
    def load_query_engine(self, nodes):
        """
        Load the query engine from nodes.
        """
        retriever = self.load_retriever(nodes)
        node_postprocessors = self.load_node_postprocessors()

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=node_postprocessors
        )
        return query_engine

    def load_node_postprocessors(self):
        from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
        reranker = FlagEmbeddingReranker(
                    top_n=self.rerank_size,
                    model=self.rerank_model,
                    use_fp16=False
        )
        if self.recursive_rerank:
            return [reranker, Small2big(all_nodes_dict=self.all_nodes_dict)]
        elif self.enable_auto_merge:
            return [reranker, AutoMerge(auto_merge_retriever=self.auto_merge_retriever, auto_merge_topk=self.auto_merge_topk)]
        else:
            return [reranker]

    def load_retriever(self, nodes):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name, embed_batch_size=10, max_length=512)
        if self.regenerate_emb:
            print('Regenerating embeddings')
            new_nodes = []
            for node in nodes:
                node.embedding = None
                new_nodes.append(node)
            nodes = new_nodes
        vector_recall_num = int(self.rerank_size * self.vector_ratio)
        bm25_recall_num = self.rerank_size - vector_recall_num
        if self.enable_auto_merge:
            docstore = SimpleDocumentStore()
            # insert nodes into docstore
            docstore.add_documents(nodes)
            # define storage context (will include vector store by default too)
            storage_context = StorageContext.from_defaults(docstore=docstore)
            leaf_nodes = get_leaf_nodes(nodes)
            nodes = leaf_nodes
        if vector_recall_num > 0:
            if self.recursive_retrieve:
                sub_node_parsers = [
                    SentenceSplitter(chunk_size=c, chunk_overlap=0) for c in self.recursive_retrieve
                ]
                all_nodes = []
                for base_node in nodes:
                    for n in sub_node_parsers:
                        sub_nodes = n.get_nodes_from_documents([base_node])
                        sub_inodes = [
                            IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                        ]
                        all_nodes.extend(sub_inodes)
                    # also add original node to node
                    original_node = IndexNode.from_text_node(base_node, base_node.node_id)
                    all_nodes.append(original_node)
                all_nodes_dict = {n.node_id: n for n in all_nodes}
                vector_index = VectorStoreIndex(all_nodes, embed_model=embed_model, show_progress=self.show_progress, use_async=self.use_async, insert_batch_size=2048)
                vector_retriever_chunk = vector_index.as_retriever(similarity_top_k=vector_recall_num)
                vector_retriever = RecursiveRetriever(
                    "vector",
                    retriever_dict={"vector": vector_retriever_chunk},
                    node_dict=all_nodes_dict,
                )
            elif self.recursive_rerank:
                sub_node_parsers = [
                    SentenceSplitter(chunk_size=c, chunk_overlap=0) for c in self.recursive_rerank
                ]
                all_nodes = []
                all_nodes_dict = {}
                for base_node in nodes:
                    for n in sub_node_parsers:
                        sub_nodes = n.get_nodes_from_documents([base_node])
                        sub_inodes = [
                            IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                        ]
                        all_nodes.extend(sub_inodes)
                        for n in sub_inodes:
                            all_nodes_dict[n.node_id] = base_node
                self.all_nodes_dict = all_nodes_dict
                vector_index = VectorStoreIndex(all_nodes, embed_model=embed_model, show_progress=self.show_progress, use_async=self.use_async, insert_batch_size=2048)
                vector_retriever = vector_index.as_retriever(similarity_top_k=vector_recall_num)
                nodes = all_nodes
            else:
                vector_index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=self.show_progress, use_async=self.use_async, insert_batch_size=2048)
                vector_retriever = vector_index.as_retriever(similarity_top_k=vector_recall_num)
        else:
            vector_retriever = None

        if bm25_recall_num > 0:
            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=bm25_recall_num)
        else:
            bm25_retriever = None
        hybrid_retriever = SimpleHybridRetriever(vector_retriever, bm25_retriever)

        if self.enable_auto_merge:
            self.auto_merge_retriever = AutoMergingRetriever(hybrid_retriever, storage_context, simple_ratio_thresh=self.auto_merge_ratio, verbose=False)
        return hybrid_retriever

