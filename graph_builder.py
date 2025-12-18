"""
GraphRAG Knowledge Graph Builder for UK Contract Law
Extracts entities and relationships from legal documents to build a knowledge graph
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import networkx as nx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
import pickle

class LegalKnowledgeGraph:
    def __init__(self, model_name: str = "uk-lawyer"):
        self.graph = nx.DiGraph()
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        self.entity_types = [
            "CASE", "DOCTRINE", "LEGAL_PRINCIPLE", "STATUTE", 
            "PARTY", "COURT", "JUDGE", "LEGAL_TEST"
        ]
        self.relation_types = [
            "OVERRULES", "DISTINGUISHES", "APPLIES", "CITES",
            "ESTABLISHES_PRINCIPLE", "CREATES_EXCEPTION", "REFINES",
            "RELATED_TO", "SUPPORTS", "CONTRADICTS"
        ]
        
    def extract_entities_and_relations(self, text: str, domain: str) -> Dict:
        """Use regex pattern matching to extract legal entities and relationships from text."""
        
        entities = []
        relationships = []
        
        # Pattern 1: Extract case names (e.g., "Williams v Roffey Bros [1991]")
        import re
        case_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:\[(\d{4})\])?'
        case_matches = re.findall(case_pattern, text)
        
        for match in case_matches[:5]:  # Limit to 5 cases per chunk
            case_name = f"{match[0]} v {match[1]}"
            year = match[2] if match[2] else None
            entities.append({
                "name": case_name,
                "type": "CASE",
                "year": year
            })
        
        # Pattern 2: Extract doctrines and principles
        doctrine_keywords = [
            "Promissory Estoppel", "Economic Duress", "Consideration",
            "Misrepresentation", "Mutual Mistake", "Offer and Acceptance",
            "Offer & Acceptance", "Contractual Terms", "Undue Influence",
            "Mistake", "Duress", "Estoppel"
        ]
        
        text_lower = text.lower()
        for keyword in doctrine_keywords:
            if keyword.lower() in text_lower:
                entities.append({
                    "name": keyword,
                    "type": "DOCTRINE"
                })
        
        # Pattern 3: Extract legal tests and principles
        test_keywords = [
            "practical benefit", "shield not a sword", "IRAC",
            "reasonable person test", "objective test", "subjective test",
            "part payment", "bargain", "legitimate pressure"
        ]
        
        for keyword in test_keywords:
            if keyword.lower() in text_lower:
                entities.append({
                    "name": keyword.title(),
                    "type": "LEGAL_TEST"
                })
        
        # Pattern 4: Extract relationships between cases
        # Look for relationship indicators
        relationship_patterns = {
            "distinguishes": r'(?:distinguished|distinguishing)\s+([A-Z][a-z]+\s+v\s+[A-Z][a-z]+)',
            "overrules": r'(?:overruled|overruling)\s+([A-Z][a-z]+\s+v\s+[A-Z][a-z]+)',
            "applies": r'(?:applied|applying)\s+([A-Z][a-z]+\s+v\s+[A-Z][a-z]+)',
            "follows": r'(?:followed|following)\s+([A-Z][a-z]+\s+v\s+[A-Z][a-z]+)',
        }
        
        # Extract first case as source
        if case_matches:
            source_case = f"{case_matches[0][0]} v {case_matches[0][1]}"
            
            for relation_type, pattern in relationship_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                for target in matches[:2]:  # Limit relationships
                    relationships.append({
                        "source": source_case,
                        "target": target,
                        "relation": relation_type.upper()
                    })
            
            # Add doctrine establishment relationships
            for entity in entities:
                if entity["type"] == "DOCTRINE":
                    relationships.append({
                        "source": source_case,
                        "target": entity["name"],
                        "relation": "APPLIES"
                    })
        
        return {
            "entities": entities,
            "relationships": relationships
        }
    
    def add_to_graph(self, extraction: Dict, domain: str, source_doc: str):
        """Add extracted entities and relationships to the knowledge graph."""
        
        # Add entities as nodes
        for entity in extraction.get("entities", []):
            node_id = entity["name"]
            self.graph.add_node(
                node_id,
                type=entity.get("type", "UNKNOWN"),
                domain=domain,
                source=source_doc,
                year=entity.get("year"),
                metadata=entity
            )
        
        # Add relationships as edges
        for rel in extraction.get("relationships", []):
            self.graph.add_edge(
                rel["source"],
                rel["target"],
                relation=rel["relation"],
                domain=domain
            )
    
    def build_from_pdfs(self, data_folder: str = "./legal_docs"):
        """Build knowledge graph from all PDFs in the data folder."""
        
        print("🏗️ Building Legal Knowledge Graph...")
        
        if not os.path.exists(data_folder):
            print(f"❌ Folder not found: {data_folder}")
            return
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=400
        )
        
        # Walk through all PDFs
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    domain_name = os.path.basename(root)
                    if domain_name == "legal_docs":
                        domain_name = "General"
                    
                    print(f"\n📄 Processing [{domain_name}] {file}...")
                    
                    try:
                        # Load PDF
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        
                        # Split into chunks
                        splits = text_splitter.split_documents(docs)
                        
                        # Process first few chunks (to avoid overwhelming)
                        for i, chunk in enumerate(splits[:5]):  # Limit to 5 chunks per PDF
                            print(f"  Processing chunk {i+1}/5...")
                            extraction = self.extract_entities_and_relations(
                                chunk.page_content, 
                                domain_name
                            )
                            self.add_to_graph(extraction, domain_name, file)
                            
                    except Exception as e:
                        print(f"  ⚠️ Error processing {file}: {e}")
        
        print(f"\n✅ Knowledge Graph Built!")
        print(f"   Nodes: {self.graph.number_of_nodes()}")
        print(f"   Edges: {self.graph.number_of_edges()}")
    
    def get_related_cases(self, case_name: str, max_depth: int = 2) -> List[str]:
        """Get all cases related to a given case within max_depth hops."""
        
        if case_name not in self.graph:
            return []
        
        # Use BFS to find related nodes
        related = []
        visited = set()
        queue = [(case_name, 0)]
        
        while queue:
            node, depth = queue.pop(0)
            if depth > max_depth or node in visited:
                continue
            
            visited.add(node)
            if node != case_name and self.graph.nodes[node].get('type') == 'CASE':
                related.append(node)
            
            # Add neighbors
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return related
    
    def get_doctrine_cases(self, doctrine: str) -> List[str]:
        """Get all cases that apply or establish a specific doctrine."""
        
        cases = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if node_data.get('type') == 'CASE':
                # Check if this case relates to the doctrine
                for neighbor in self.graph.neighbors(node):
                    if doctrine.lower() in neighbor.lower():
                        edge_data = self.graph.get_edge_data(node, neighbor)
                        if edge_data and edge_data.get('relation') in ['APPLIES', 'ESTABLISHES_PRINCIPLE']:
                            cases.append(node)
                            break
        
        return cases
    
    def get_graph_context(self, query: str, domain: str = None) -> str:
        """Get relevant context from the knowledge graph for a query."""
        
        # Extract potential entity mentions from query
        context_parts = []
        
        # Find matching nodes
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            
            # Check if node matches query or domain
            if (any(word.lower() in node.lower() for word in query.split()) or 
                (domain and node_data.get('domain') == domain)):
                
                # Add node info
                context_parts.append(f"\n**{node}** ({node_data.get('type', 'UNKNOWN')})")
                
                # Add relationships
                for target in self.graph.neighbors(node):
                    edge_data = self.graph.get_edge_data(node, target)
                    relation = edge_data.get('relation', 'RELATED_TO')
                    context_parts.append(f"  → {relation}: {target}")
        
        if not context_parts:
            return ""
        
        return "**Knowledge Graph Context:**\n" + "\n".join(context_parts[:20])  # Limit output
    
    def save(self, filepath: str = "legal_knowledge_graph.pkl"):
        """Save the knowledge graph to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"💾 Saved knowledge graph to {filepath}")
    
    def load(self, filepath: str = "legal_knowledge_graph.pkl"):
        """Load the knowledge graph from disk."""
        try:
            with open(filepath, 'rb') as f:
                self.graph = pickle.load(f)
            print(f"📂 Loaded knowledge graph from {filepath}")
            print(f"   Nodes: {self.graph.number_of_nodes()}")
            print(f"   Edges: {self.graph.number_of_edges()}")
        except FileNotFoundError:
            print(f"⚠️ No saved graph found at {filepath}")
    
    def visualize_subgraph(self, center_node: str, depth: int = 2, output_file: str = "graph.png"):
        """Visualize a subgraph around a specific node."""
        try:
            import matplotlib.pyplot as plt
            
            # Get subgraph
            nodes = {center_node}
            for _ in range(depth):
                new_nodes = set()
                for node in nodes:
                    new_nodes.update(self.graph.neighbors(node))
                    new_nodes.update(self.graph.predecessors(node))
                nodes.update(new_nodes)
            
            subgraph = self.graph.subgraph(nodes)
            
            # Draw
            plt.figure(figsize=(15, 10))
            pos = nx.spring_layout(subgraph, k=2, iterations=50)
            
            # Color by type
            colors = []
            for node in subgraph.nodes():
                node_type = subgraph.nodes[node].get('type', 'UNKNOWN')
                color_map = {
                    'CASE': 'lightblue',
                    'DOCTRINE': 'lightgreen',
                    'LEGAL_PRINCIPLE': 'yellow',
                    'LEGAL_TEST': 'orange'
                }
                colors.append(color_map.get(node_type, 'gray'))
            
            nx.draw(subgraph, pos, node_color=colors, with_labels=True, 
                   node_size=3000, font_size=8, font_weight='bold',
                   edge_color='gray', arrows=True)
            
            plt.title(f"Legal Knowledge Graph: {center_node}")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {output_file}")
            
        except ImportError:
            print("⚠️ matplotlib not installed. Run: pip install matplotlib")


# Test/Build Script
if __name__ == "__main__":
    kg = LegalKnowledgeGraph()
    
    # Try to load existing graph
    kg.load()
    
    # If no graph exists, build from PDFs
    if kg.graph.number_of_nodes() == 0:
        kg.build_from_pdfs("./legal_docs")
        kg.save()
    
    # Example queries
    print("\n" + "="*80)
    print("EXAMPLE GRAPH QUERIES")
    print("="*80)
    
    # Find related cases
    related = kg.get_related_cases("Williams v Roffey Bros")
    if related:
        print(f"\nCases related to Williams v Roffey Bros:")
        for case in related:
            print(f"  • {case}")
    
    # Get context for a query
    context = kg.get_graph_context("promissory estoppel rent reduction", "Promissory Estoppel")
    print(f"\n{context}")