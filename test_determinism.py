#!/usr/bin/env python3
"""Test script to verify deterministic responses."""
import sys
sys.path.insert(0, '/home/hammad/Desktop/enigma/hybrid_chat_test')

from hybrid_chat import pinecone_query, fetch_graph_context, build_prompt, call_chat

def test_determinism(query, num_runs=3):
    print(f"Testing determinism for query: '{query}'")
    print(f"Running {num_runs} times...\n")
    
    responses = []
    
    for i in range(num_runs):
        print(f"Run {i+1}:")
        matches = pinecone_query(query, top_k=5)
        match_ids = [m["id"] for m in matches]
        print(f"  Match IDs: {match_ids}")
        
        graph_facts = fetch_graph_context(match_ids)
        print(f"  Graph facts count: {len(graph_facts)}")
        
        # Check first few fact IDs
        fact_ids = [f['target_id'] for f in graph_facts[:5]]
        print(f"  First 5 fact target IDs: {fact_ids}")
        
        prompt = build_prompt(query, matches, graph_facts)
        answer = call_chat(prompt)
        responses.append(answer)
        print(f"  Response preview: {answer[:100]}...\n")
    
    # Compare responses
    print("="*60)
    print("DETERMINISM CHECK:")
    print("="*60)
    
    all_same = all(r == responses[0] for r in responses)
    
    if all_same:
        print("✅ SUCCESS: All responses are IDENTICAL")
        print("\nFull response:")
        print(responses[0])
    else:
        print("❌ FAILURE: Responses differ!")
        for i, resp in enumerate(responses):
            print(f"\n--- Response {i+1} ---")
            print(resp)
            print()

if __name__ == "__main__":
    test_query = "plan a romantic trip to vietnam"
    test_determinism(test_query, num_runs=2)
