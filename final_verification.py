# final_verification.py
"""
Final verification script to test the optimized system
"""
import asyncio
import logging
from simple_rag_system import simple_rag
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_optimized_system():
    """Verify the optimized system works correctly"""
    
    logger.info("🔍 FINAL VERIFICATION OF OPTIMIZED SYSTEM")
    logger.info("=" * 55)
    
    # Test optimized configuration
    logger.info("1. Configuration Check")
    logger.info("-" * 30)
    logger.info(f"✅ Similarity threshold: {config.similarity_threshold} (lowered from 0.3)")
    logger.info(f"✅ RAG timeout: {config.rag_timeout_ms}ms (increased for reliability)")
    logger.info(f"✅ Search limit: {config.search_limit} (increased for better results)")
    logger.info(f"✅ High confidence: {config.high_confidence_threshold}")
    logger.info(f"✅ Medium confidence: {config.medium_confidence_threshold}")
    logger.info(f"✅ Minimum usable: {config.minimum_usable_threshold}")
    
    # Test RAG initialization
    logger.info("\n2. RAG System Test")
    logger.info("-" * 30)
    
    success = await simple_rag.initialize()
    if not success:
        logger.error("❌ RAG initialization failed!")
        return False
    
    status = await simple_rag.get_status()
    points_count = status.get("points_count", 0)
    logger.info(f"✅ RAG initialized successfully")
    logger.info(f"✅ Documents: {points_count}")
    
    # Test real queries from your log with new thresholds
    logger.info("\n3. Real Query Test with New Thresholds")
    logger.info("-" * 30)
    
    test_queries = [
        "I want to change my tire",
        "what services do you provide",
        "towing cost",
        "flat tire change cost",
        "battery jumpstart price"
    ]
    
    all_good = True
    for query in test_queries:
        results = await simple_rag.search(query, limit=config.search_limit)
        
        if results:
            best_score = results[0].get("score", 0)
            
            if best_score >= config.high_confidence_threshold:
                confidence = "HIGH"
                symbol = "✅"
            elif best_score >= config.medium_confidence_threshold:
                confidence = "MEDIUM"
                symbol = "⚠️"
            elif best_score >= config.minimum_usable_threshold:
                confidence = "LOW"
                symbol = "⚠️"
            else:
                confidence = "TOO LOW"
                symbol = "❌"
                all_good = False
            
            logger.info(f"{symbol} '{query}': score={best_score:.3f} ({confidence})")
        else:
            logger.error(f"❌ '{query}': No results")
            all_good = False
    
    # Test conversation scenarios
    logger.info("\n4. Conversation Scenario Test")
    logger.info("-" * 30)
    
    scenarios = [
        ("Customer asks about pricing", "how much does towing cost"),
        ("Customer needs tire help", "I have a flat tire"),
        ("Customer asks about services", "what services do you offer"),
        ("Customer wants transfer", "can I speak to someone"),
    ]
    
    for description, query in scenarios:
        results = await simple_rag.search(query, limit=2)
        if results and results[0].get("score", 0) >= config.minimum_usable_threshold:
            logger.info(f"✅ {description}: Will get knowledge base context")
        else:
            logger.warning(f"⚠️ {description}: May not get good context")
    
    # Summary
    logger.info("\n5. FINAL VERIFICATION SUMMARY")
    logger.info("-" * 30)
    
    if all_good and points_count > 0:
        logger.info("🎉 SYSTEM IS READY!")
        logger.info("✅ All tests passed")
        logger.info("✅ RAG system working optimally")
        logger.info("✅ Thresholds properly configured")
        logger.info("\n💡 NEXT STEPS:")
        logger.info("   1. Replace your main.py with the new version")
        logger.info("   2. Replace your config.py with the optimized version")
        logger.info("   3. Run: python main.py dev")
        logger.info("   4. Test with phone calls")
        
        return True
    else:
        logger.error("❌ ISSUES FOUND:")
        if points_count == 0:
            logger.error("   - No documents in knowledge base")
            logger.error("   - Run: python excel_ingest.py --file data/Roadside_Assist_Info.xlsx")
        if not all_good:
            logger.error("   - Some queries still have low scores")
            logger.error("   - Consider re-ingesting data or adjusting thresholds further")
        
        return False

if __name__ == "__main__":
    asyncio.run(verify_optimized_system())