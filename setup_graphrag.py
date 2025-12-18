#!/usr/bin/env python3
"""
Setup script for GraphRAG Legal System
Builds the knowledge graph from your legal PDFs
"""

from graph_builder import LegalKnowledgeGraph
import os

def main():
    print("="*80)
    print("GraphRAG Legal Knowledge System - Setup")
    print("="*80)
    
    # Check if legal_docs folder exists
    if not os.path.exists("./legal_docs"):
        print("\n❌ Error: ./legal_docs folder not found")
        print("Please create the folder and add your PDF files organized by domain:")
        print("  legal_docs/")
        print("    ├── Promissory Estoppel/")
        print("    │   └── cases.pdf")
        print("    ├── Contractual Terms/")
        print("    │   └── cases.pdf")
        print("    └── ...")
        return
    
    # Build knowledge graph
    print("\n📊 Building Legal Knowledge Graph...")
    print("This will take several minutes on first run.\n")
    
    kg = LegalKnowledgeGraph()
    
    # Check if graph already exists
    if os.path.exists("legal_knowledge_graph.pkl"):
        response = input("Knowledge graph already exists. Rebuild? (y/n): ")
        if response.lower() != 'y':
            print("Loading existing graph...")
            kg.load()
            print("\n✅ Setup complete!")
            return
    
    # Build from PDFs
    kg.build_from_pdfs("./legal_docs")
    kg.save()
    
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*80)
    print(f"Total Entities: {kg.graph.number_of_nodes()}")
    print(f"Total Relationships: {kg.graph.number_of_edges()}")
    
    # Show sample entities by type
    entity_types = {}
    for node in kg.graph.nodes():
        node_type = kg.graph.nodes[node].get('type', 'UNKNOWN')
        entity_types[node_type] = entity_types.get(node_type, 0) + 1
    
    print("\nEntity Breakdown:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count}")
    
    # Show sample relationships
    print("\nSample Relationships:")
    relation_types = {}
    for edge in list(kg.graph.edges(data=True))[:10]:
        source, target, data = edge
        relation = data.get('relation', 'UNKNOWN')
        relation_types[relation] = relation_types.get(relation, 0) + 1
        print(f"  {source} --[{relation}]--> {target}")
    
    print("\n✅ Setup complete! You can now run:")
    print("   streamlit run GUI_GraphRAG.py")
    
    # Optional: Generate visualization
    try:
        # Find a central case for visualization
        cases = [n for n in kg.graph.nodes() if kg.graph.nodes[n].get('type') == 'CASE']
        if cases:
            print(f"\n📊 Generating visualization for: {cases[0]}")
            kg.visualize_subgraph(cases[0], depth=2, output_file="knowledge_graph_sample.png")
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

if __name__ == "__main__":
    main()