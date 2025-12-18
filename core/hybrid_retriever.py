"""
Hybrid Retriever: Combines Vector RAG + GraphRAG
"""

from typing import List, Dict
from langchain_core.documents import Document

class HybridGraphRetriever:
    """
    Combines traditional vector similarity search with knowledge graph traversal
    for more comprehensive legal context retrieval.
    """
    
    def __init__(self, vectorstore, knowledge_graph):
        self.vectorstore = vectorstore
        self.kg = knowledge_graph
    
    def get_relevant_documents(self, query: str, domain: str = None, k: int = 4) -> List[Document]:
        """
        Retrieve documents using hybrid approach:
        1. Vector similarity (traditional RAG)
        2. Knowledge graph expansion (GraphRAG)
        3. Merge and rank results
        """
        
        # STEP 1: Vector Retrieval (Traditional RAG)
        search_kwargs = {"k": k}
        if domain and domain != "General":
            search_kwargs["filter"] = {"domain": domain}
        
        vector_retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        vector_docs = vector_retriever.get_relevant_documents(query)
        
        # STEP 2: Knowledge Graph Expansion (GraphRAG)
        # Extract entities from the retrieved documents
        graph_context = self.kg.get_graph_context(query, domain)
        
        # Get related cases from knowledge graph
        related_cases = []
        for doc in vector_docs[:2]:  # Check top 2 docs for case names
            # Simple entity extraction - look for "v" pattern (e.g., "Williams v Roffey")
            import re
            case_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
            matches = re.findall(case_pattern, doc.page_content)
            for match in matches:
                case_name = f"{match[0]} v {match[1]}"
                related = self.kg.get_related_cases(case_name, max_depth=2)
                related_cases.extend(related)
        
        # STEP 3: Retrieve documents for related cases
        graph_docs = []
        for related_case in set(related_cases[:3]):  # Limit to top 3 related cases
            # Search for documents mentioning the related case
            case_search = self.vectorstore.similarity_search(related_case, k=1)
            graph_docs.extend(case_search)
        
        # STEP 4: Merge and deduplicate
        all_docs = vector_docs + graph_docs
        
        # Deduplicate based on content similarity
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            content_hash = hash(doc.page_content[:200])  # Hash first 200 chars
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # STEP 5: Add graph context as metadata
        if graph_context and unique_docs:
            # Add graph context to the first document's metadata
            unique_docs[0].metadata["graph_context"] = graph_context
        
        return unique_docs[:k+2]  # Return slightly more docs due to graph expansion