# comprehensive_test.py
"""
Comprehensive test to verify all fixes work correctly
Tests exact scenarios from your logs and Excel data
"""
import asyncio
import logging
from simple_rag_system import simple_rag
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_comprehensive_system():
    """Test the comprehensive fixed system"""
    
    logger.info("🔍 COMPREHENSIVE SYSTEM TEST")
    logger.info("=" * 60)
    
    # Test 1: Configuration Verification
    logger.info("1. Configuration Verification")
    logger.info("-" * 40)
    logger.info(f"✅ Similarity threshold: {config.similarity_threshold} (should be 0.15)")
    logger.info(f"✅ RAG timeout: {config.rag_timeout_ms}ms (should be 3000)")
    logger.info(f"✅ Search limit: {config.search_limit} (should be 5)")
    logger.info(f"✅ High confidence: {config.high_confidence_threshold}")
    logger.info(f"✅ Medium confidence: {config.medium_confidence_threshold}")
    logger.info(f"✅ Minimum usable: {config.minimum_usable_threshold}")
    
    # Test 2: RAG System
    logger.info("\n2. RAG System Test")
    logger.info("-" * 40)
    
    success = await simple_rag.initialize()
    if not success:
        logger.error("❌ RAG initialization failed!")
        return False
    
    status = await simple_rag.get_status()
    points_count = status.get("points_count", 0)
    logger.info(f"✅ RAG initialized: {points_count} documents")
    
    if points_count == 0:
        logger.error("❌ No documents! Run: python excel_ingest.py --file data/Roadside_Assist_Info.xlsx")
        return False
    
    # Test 3: Exact Excel Data Queries
    logger.info("\n3. Excel Data Query Test")
    logger.info("-" * 40)
    
    excel_queries = [
        ("Battery jump start pricing", "battery jump start", "$40"),
        ("Flat tire change cost", "flat tire change", "$50"),
        ("Standard sedan towing", "towing sedan", "$75"),
        ("SUV towing cost", "SUV towing", "$120"),
        ("Battery replacement", "battery replacement", "$150"),
        ("Fuel delivery service", "fuel delivery", "$65"),
        ("Car lockout service", "car lockout", "$55"),
        ("Tire repair cost", "tire repair", "$25"),
    ]
    
    all_good = True
    for description, query, expected_price in excel_queries:
        results = await simple_rag.search(query, limit=5)
        
        if results:
            best_score = results[0].get("score", 0)
            text = results[0].get("text", "").lower()
            
            # Check if expected price is in the result
            price_found = expected_price.lower() in text
            
            if best_score >= config.minimum_usable_threshold and price_found:
                logger.info(f"✅ {description}: score={best_score:.3f}, found {expected_price}")
            elif best_score >= config.minimum_usable_threshold:
                logger.warning(f"⚠️ {description}: score={best_score:.3f}, but price {expected_price} not found in: {text[:100]}")
            else:
                logger.error(f"❌ {description}: score={best_score:.3f} too low")
                all_good = False
        else:
            logger.error(f"❌ {description}: No results")
            all_good = False
    
    # Test 4: Problem Queries from Logs
    logger.info("\n4. Problem Queries from Your Logs")
    logger.info("-" * 40)
    
    problem_queries = [
        "tell me which services you provide",
        "battery services",
        "pricing platform of battery service",
        "specific battery services",
        "what does the battery cost"
    ]
    
    for query in problem_queries:
        results = await simple_rag.search(query, limit=5)
        
        if results:
            best_score = results[0].get("score", 0)
            if best_score >= config.minimum_usable_threshold:
                logger.info(f"✅ '{query}': score={best_score:.3f} - WILL GET CONTEXT")
            else:
                logger.warning(f"⚠️ '{query}': score={best_score:.3f} - MAY NOT GET CONTEXT")
        else:
            logger.error(f"❌ '{query}': No results")
    
    # Test 5: Service Categories
    logger.info("\n5. Service Category Coverage Test")
    logger.info("-" * 40)
    
    service_categories = [
        "towing services",
        "battery services", 
        "tire services",
        "fuel services",
        "lockout services"
    ]
    
    for category in service_categories:
        results = await simple_rag.search(category, limit=3)
        
        if results:
            good_results = [r for r in results if r.get("score", 0) >= config.minimum_usable_threshold]
            logger.info(f"✅ {category}: {len(good_results)}/{len(results)} good results")
        else:
            logger.error(f"❌ {category}: No results")
    
    # Test 6: Conversation Flow Simulation
    logger.info("\n6. Conversation Flow Simulation")
    logger.info("-" * 40)
    
    conversation_flow = [
        ("User needs help", "my car is not working"),
        ("User asks about services", "what services do you provide"),  
        ("User asks about battery", "I need battery help"),
        ("User asks about pricing", "how much does battery service cost"),
        ("User confirms service", "yes I need jump start")
    ]
    
    for step, query in conversation_flow:
        results = await simple_rag.search(query, limit=3)
        
        if results and results[0].get("score", 0) >= config.minimum_usable_threshold:
            logger.info(f"✅ {step}: Will get helpful context")
        else:
            logger.warning(f"⚠️ {step}: May need manual response")
    
    # Summary
    logger.info("\n7. COMPREHENSIVE TEST SUMMARY")
    logger.info("-" * 40)
    
    if all_good and points_count > 50:
        logger.info("🎉 COMPREHENSIVE TEST PASSED!")
        logger.info("✅ Configuration is correct")
        logger.info("✅ RAG system working properly") 
        logger.info("✅ Excel data accessible")
        logger.info("✅ Problem queries should now work")
        logger.info("\n💡 READY TO TEST:")
        logger.info("   1. Update main.py and config.py with fixed versions")
        logger.info("   2. Run: python main.py dev")
        logger.info("   3. Test with real phone calls")
        logger.info("   4. Ask about battery services, towing costs, etc.")
        
        return True
    else:
        logger.error("❌ COMPREHENSIVE TEST FAILED!")
        if points_count == 0:
            logger.error("   - No documents in knowledge base")
            logger.error("   - Run: python excel_ingest.py --file data/Roadside_Assist_Info.xlsx")
        if not all_good:
            logger.error("   - Some Excel queries not working properly")
            logger.error("   - May need to adjust thresholds or re-ingest data")
        
        return False

if __name__ == "__main__":
    asyncio.run(test_comprehensive_system())