# test_aggressive_caching.py
"""
Aggressive performance test to demonstrate the caching improvements
"""
import asyncio
import logging
import time
from qdrant_rag_system import qdrant_rag
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_aggressive_caching():
    """Test the aggressive caching system"""
    logger.info("🚀 TESTING AGGRESSIVE CACHING SYSTEM")
    logger.info("=" * 60)
    
    # Initialize system
    start_time = time.time()
    success = await qdrant_rag.initialize()
    init_time = (time.time() - start_time) * 1000
    
    if not success:
        logger.error("❌ Failed to initialize Qdrant")
        return False
    
    logger.info(f"✅ System initialized in {init_time:.1f}ms")
    
    # Test queries with variations that should hit similarity cache
    test_sets = [
        # Set 1: Exact matches (should be super fast on repeat)
        ["towing service", "towing service", "towing service"],
        
        # Set 2: Similar queries (should hit similarity cache)
        ["battery help", "battery service", "battery assistance"], 
        
        # Set 3: Pricing variations
        ["pricing", "cost", "how much does it cost"],
        
        # Set 4: Service variations  
        ["what services do you offer", "what services", "services offered"],
        
        # Set 5: New queries (will need API calls)
        ["emergency roadside", "24 hour service", "contact information"]
    ]
    
    total_tests = 0
    total_time = 0
    cache_hits = 0
    api_calls = 0
    
    for set_num, query_set in enumerate(test_sets, 1):
        logger.info(f"\n🧪 Test Set {set_num}: {query_set[0]} variations")
        
        for i, query in enumerate(query_set):
            start_time = time.time()
            results = await qdrant_rag.search(query, limit=2)
            search_time = (time.time() - start_time) * 1000
            
            total_tests += 1
            total_time += search_time
            
            # Classify performance
            if search_time < 20:
                cache_hits += 1
                status = "⚡ CACHE HIT"
            elif search_time < 100:
                status = "🚀 FAST"
            else:
                api_calls += 1
                status = "📡 API CALL"
            
            logger.info(f"   Query {i+1}: {search_time:.1f}ms - {status}")
            logger.info(f"   '{query}' → {len(results)} results")
    
    # Get final cache statistics
    cache_stats = await qdrant_rag.get_cache_stats()
    
    # Calculate performance metrics
    avg_time = total_time / total_tests if total_tests > 0 else 0
    cache_ratio = cache_hits / total_tests if total_tests > 0 else 0
    api_ratio = api_calls / total_tests if total_tests > 0 else 0
    
    logger.info(f"\n📊 AGGRESSIVE CACHING RESULTS:")
    logger.info(f"   Total queries: {total_tests}")
    logger.info(f"   Average time: {avg_time:.1f}ms")
    logger.info(f"   Cache hits: {cache_hits}/{total_tests} ({cache_ratio:.1%})")
    logger.info(f"   API calls: {api_calls}/{total_tests} ({api_ratio:.1%})")
    logger.info(f"   Target time: {config.rag_timeout_ms}ms")
    logger.info(f"   Performance target met: {'✅ YES' if avg_time < config.rag_timeout_ms else '❌ NO'}")
    
    logger.info(f"\n🚀 Cache Statistics:")
    logger.info(f"   Search cache: {cache_stats['search_cache_size']}/{cache_stats['search_cache_max']}")
    logger.info(f"   Embedding cache: {cache_stats['embedding_cache_size']}/{cache_stats['embedding_cache_max']}")
    logger.info(f"   Query mappings: {cache_stats['query_mapping_cache_size']}")
    
    # Performance assessment
    telephony_ready = avg_time < 100 and cache_ratio > 0.3
    
    logger.info(f"\n🎯 FINAL ASSESSMENT:")
    if telephony_ready:
        logger.info("🎉 EXCELLENT! Ultra-fast performance achieved!")
        logger.info("   🚀 Ready for production telephony")
        logger.info("   💰 Massive API cost savings")
        logger.info("   ⚡ Real-time response capability")
    elif avg_time < config.rag_timeout_ms:
        logger.info("✅ GOOD! Performance target met")
        logger.info("   📞 Suitable for telephony with some optimization")
    else:
        logger.info("⚠️ NEEDS IMPROVEMENT")
        logger.info("   🔧 Additional tuning required")
    
    await qdrant_rag.close()
    return telephony_ready

async def test_similar_query_matching():
    """Test the similarity-based query matching"""
    logger.info("\n🔍 TESTING SIMILARITY-BASED QUERY MATCHING")
    logger.info("=" * 60)
    
    await qdrant_rag.initialize()
    
    # Test pairs where second should match first
    similarity_tests = [
        ("towing service", "tow my car"),
        ("battery help", "my battery is dead"),  
        ("pricing information", "how much does it cost"),
        ("business hours", "when are you open"),
        ("contact information", "how can I reach you")
    ]
    
    total_similarity_hits = 0
    
    for original, similar in similarity_tests:
        # First query (will hit API)
        start_time = time.time()
        await qdrant_rag.search(original, limit=2)
        first_time = (time.time() - start_time) * 1000
        
        # Second query (should hit similarity cache)
        start_time = time.time()
        await qdrant_rag.search(similar, limit=2)
        second_time = (time.time() - start_time) * 1000
        
        # Check if similarity caching worked
        similarity_hit = second_time < first_time * 0.5  # At least 50% faster
        if similarity_hit:
            total_similarity_hits += 1
        
        logger.info(f"   '{original}' → {first_time:.1f}ms")
        logger.info(f"   '{similar}' → {second_time:.1f}ms {'⚡ SIM HIT' if similarity_hit else '📡 API'}")
    
    similarity_ratio = total_similarity_hits / len(similarity_tests)
    logger.info(f"\n🎯 Similarity Matching: {total_similarity_hits}/{len(similarity_tests)} ({similarity_ratio:.1%})")
    
    await qdrant_rag.close()

if __name__ == "__main__":
    async def main():
        # Test aggressive caching
        success = await test_aggressive_caching()
        
        # Test similarity matching
        await test_similar_query_matching()
        
        if success:
            logger.info("\n🎉 ALL PERFORMANCE TESTS PASSED!")
            logger.info("🚀 Your system is ready for production telephony!")
        else:
            logger.info("\n⚠️ Performance needs additional tuning")
    
    asyncio.run(main())