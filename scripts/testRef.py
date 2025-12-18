#!/usr/bin/env python3
"""
Test script for the QueryRefiner to verify domain classification
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.refiner import QueryRefiner

# Test queries for different domains
test_cases = [
    {
        "query": "landlord promised to lower rent but now wants it back",
        "expected_domain": "Promissory Estoppel"
    },
    {
        "query": "I have a commercial lease. During pandemic, landlord emailed saying I only pay 40% rent while shop closed. I relied on this and spent savings on stock. Now landlord demands I pay back the 60% discounted rent.",
        "expected_domain": "Promissory Estoppel"
    },
    {
        "query": "The seller lied about the car's mileage before I signed the contract",
        "expected_domain": "Misrepresentation"
    },
    {
        "query": "Does the exclusion clause in my contract protect the supplier?",
        "expected_domain": "Contractual Terms"
    },
    {
        "query": "We both thought we were buying the same painting, but it was a different one",
        "expected_domain": "Mistake- Mutual mistake"
    },
    {
        "query": "I made an offer but they sent a counter-offer. Is there a binding contract?",
        "expected_domain": "Offer & Acceptance"
    }
]

def test_refiner():
    print("=" * 80)
    print("TESTING QUERY REFINER")
    print("=" * 80)
    
    try:
        refiner = QueryRefiner()
        print("\n✅ Refiner initialized successfully\n")
    except Exception as e:
        print(f"\n❌ Failed to initialize refiner: {e}\n")
        return
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"TEST {i}: {test['expected_domain']}")
        print(f"{'─' * 80}")
        print(f"Query: {test['query'][:100]}...")
        
        try:
            # Try to get both domain and refined query
            result = refiner.refine(test['query'])
            
            # Check what we got back
            if isinstance(result, tuple) and len(result) == 2:
                domains, refined_text = result
                print(f"\n✅ Returned 2 values (correct format)")
                print(f"   Domains: {domains}")
                print(f"   Expected: [{test['expected_domain']}]")
                print(f"   Match: {'✅' if test['expected_domain'] in domains else '❌'}")
                print(f"   Refined: {refined_text[:100]}...")
            else:
                print(f"\n❌ Unexpected return format: {type(result)}")
                print(f"   Value: {result}")
                
        except Exception as e:
            print(f"\n❌ Error during refinement: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    test_refiner()