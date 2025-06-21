#!/usr/bin/env python3
"""
Debug script to test vector search directly in PostgreSQL
"""

import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
import yaml
from pathlib import Path

def load_config():
    """Load configuration"""
    config_path = Path("../config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_vector_search():
    """Test vector search directly"""
    print("=== DEBUGGING VECTOR SEARCH ===")
    
    # Load config
    config = load_config()
    pg_config = config.get('postgresql', {})
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=pg_config.get('host', 'localhost'),
        port=pg_config.get('port', 5432),
        user=pg_config.get('user', 'postgres'),
        password=pg_config.get('password', ''),
        database=pg_config.get('database', 'vectordb')
    )
    
    # Load the same embedding model
    model_path = config.get('EMBEDDING_MODEL_NAME')
    print(f"Loading model from: {model_path}")
    model = SentenceTransformer(model_path)
    
    # Test queries
    queries = [
        "Aliento de Salamandra",
        "salamandra",
        "gota sangre salamandra",
        "maleficio llama",
        "Halitus Salamandrae"
    ]
    
    with conn.cursor() as cur:
        for query in queries:
            print(f"\n--- Testing query: '{query}' ---")
            
            # Generate query embedding
            query_embedding = model.encode(query).tolist()
            print(f"Query embedding length: {len(query_embedding)}")
            
            # Test different similarity thresholds
            thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            
            for threshold in thresholds:
                cur.execute("""
                    SELECT text, 1 - (embedding <=> %s::vector) as similarity
                    FROM vectors_aquelarre_distiluse_base_multilang
                    WHERE 1 - (embedding <=> %s::vector) > %s
                    ORDER BY similarity DESC
                    LIMIT 3;
                """, (query_embedding, query_embedding, threshold))
                
                results = cur.fetchall()
                print(f"Threshold {threshold}: {len(results)} results")
                
                for i, (text, similarity) in enumerate(results):
                    print(f"  {i+1}. Similarity: {similarity:.4f}")
                    if "salamandra" in text.lower():
                        print(f"     🎯 FOUND SALAMANDRA! Text preview: {text[:100]}...")
                    else:
                        print(f"     Text preview: {text[:50]}...")
            
            # Also test without any threshold to see all similarities
            print("--- Top 10 results without threshold ---")
            cur.execute("""
                SELECT text, 1 - (embedding <=> %s::vector) as similarity
                FROM vectors_aquelarre_distiluse_base_multilang
                ORDER BY similarity DESC
                LIMIT 10;
            """, (query_embedding,))
            
            all_results = cur.fetchall()
            for i, (text, similarity) in enumerate(all_results):
                salamandra_found = "salamandra" in text.lower() or "Aliento de Salamandra" in text
                marker = "🎯" if salamandra_found else "  "
                print(f"{marker} {i+1}. Similarity: {similarity:.4f} - {text[:60]}...")
                
            # Check if Salamandra entries exist at all
            cur.execute("""
                SELECT COUNT(*) FROM vectors_aquelarre_distiluse_base_multilang
                WHERE text ILIKE '%salamandra%';
            """)
            salamandra_count = cur.fetchone()[0]
            print(f"Total entries containing 'salamandra': {salamandra_count}")
            
            print("\n" + "="*60)
    
    conn.close()

if __name__ == "__main__":
    test_vector_search() 