#!/usr/bin/env python3
"""
Test script for hybrid search implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database_interactions import QueryVectorDB
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_hybrid_search():
    """Test the hybrid search with the actual application code"""
    print("=== TESTING HYBRID SEARCH ===")
    
    # Initialize the QueryVectorDB with the aquelarre database
    query_db = QueryVectorDB.get_instance("aquelarre-distiluse-base-multilang")
    
    # Test queries
    test_queries = [
        "Aliento de Salamandra",
        "Halitus Salamandrae"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing query: '{query[:50]}...' ---")
        
        try:
            # This should use the hybrid search implementation
            contexts, metadata_list = query_db.search(query, k=5, score_threshold=0.1)
            
            print(f"Found {len(contexts)} contexts:")
            for i, (context, metadata) in enumerate(zip(contexts, metadata_list)):
                similarity = metadata.get('similarity_score', 'N/A')
                salamandra_found = "Aliento de Salamandra" in context or "salamandra" in context.lower()
                marker = "🎯" if salamandra_found else "  "
                print(f"{marker} {i+1}. Similarity: {similarity} - {context[:80]}...")
                
                if salamandra_found:
                    print(f"     FULL MATCH: {context[:200]}...")
                    
        except Exception as e:
            print(f"Error: {e}")
            
        print("-" * 80)
    
    # Cleanup
    query_db.cleanup()

if __name__ == "__main__":
    test_hybrid_search() 