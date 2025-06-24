# check.py - Fixed version
from qdrant_rag_system import qdrant_rag
import asyncio

async def test_search():
    await qdrant_rag.initialize()
    
    # Test different search queries
    queries = [
        'towing service',
        'membership plans', 
        'battery service',
        'pricing costs'
    ]
    
    for query in queries:
        print(f'\n🔍 Testing: {query}')
        results = await qdrant_rag.search(query, limit=3)
        for i, result in enumerate(results):
            score = result["score"]
            text = result["text"]
            print(f'  {i+1}. Score: {score:.3f}')
            print(f'     Text: {text[:150]}...')
    
    await qdrant_rag.close()

if __name__ == "__main__":
    asyncio.run(test_search())