from llama_index.core.retrievers import BaseRetriever

class SimpleHybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        if self.vector_retriever:
            vector_nodes = self.vector_retriever.retrieve(query, **kwargs)
        else:
            vector_nodes = []
        if self.bm25_retriever:
            bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        else:
            bm25_nodes = []

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes