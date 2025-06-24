# pricing_diagnostic.py
"""
Diagnostic script to analyze pricing issues in your RAG system
"""
import asyncio
import logging
from qdrant_rag_system import qdrant_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_pricing_queries():
    """Test various pricing-related queries to see what the RAG system returns"""
    
    logger.info("🔍 TESTING PRICING QUERIES IN RAG SYSTEM")
    logger.info("=" * 60)
    
    # Initialize RAG system
    if not await qdrant_rag.initialize():
        logger.error("❌ Failed to initialize RAG system")
        return
    
    # Test queries that should return pricing information
    pricing_queries = [
        "towing cost",
        "towing price", 
        "long distance towing cost",
        "25 kilometers towing price",
        "15 miles towing cost",
        "pricing for towing service",
        "how much does towing cost",
        "towing rates",
        "base cost towing",
        "cost under 50 miles",
        "cost over 50 miles",
        "15 dollar towing",
        "75 dollar towing"
    ]
    
    results_summary = {}
    
    for query in pricing_queries:
        try:
            logger.info(f"\n🔍 Query: '{query}'")
            results = await qdrant_rag.search(query, limit=3)
            
            if results:
                for i, result in enumerate(results):
                    score = result["score"]
                    text = result["text"]
                    logger.info(f"   Result {i+1} (Score: {score:.3f}): {text[:150]}...")
                    
                    # Extract any pricing information
                    import re
                    prices = re.findall(r'\$\d+(?:\.\d{2})?', text)
                    if prices:
                        logger.info(f"   💰 Prices found: {prices}")
                        results_summary[query] = {
                            "score": score,
                            "prices": prices,
                            "text": text[:200]
                        }
                
                if not results:
                    logger.info(f"   ❌ No results found")
            else:
                logger.info(f"   ❌ No results found")
                
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
    
    # Analyze the results
    logger.info(f"\n📊 PRICING ANALYSIS SUMMARY")
    logger.info("=" * 60)
    
    all_prices = []
    for query, data in results_summary.items():
        prices = data["prices"]
        all_prices.extend(prices)
        logger.info(f"Query: '{query}' -> Prices: {prices} (Score: {data['score']:.3f})")
    
    # Find unique prices
    unique_prices = list(set(all_prices))
    logger.info(f"\n💰 All unique prices found: {unique_prices}")
    
    if len(unique_prices) > 3:
        logger.warning(f"⚠️ TOO MANY DIFFERENT PRICES! This explains the confusion.")
        logger.warning(f"⚠️ Your knowledge base has conflicting pricing information")
    
    await qdrant_rag.close()

async def analyze_knowledge_base_content():
    """Analyze what's actually stored in your knowledge base"""
    
    logger.info(f"\n📚 ANALYZING KNOWLEDGE BASE CONTENT")
    logger.info("=" * 60)
    
    if not await qdrant_rag.initialize():
        logger.error("❌ Failed to initialize RAG system")
        return
    
    try:
        # Get collection info
        from qdrant_client import QdrantClient
        from config import config
        
        client = QdrantClient(url=config.qdrant_url)
        collection_info = client.get_collection(config.qdrant_collection_name)
        
        logger.info(f"📊 Collection: {collection_info.points_count} documents")
        
        # Sample some documents to see the content
        points, _ = client.scroll(
            collection_name=config.qdrant_collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False
        )
        
        logger.info(f"\n📄 SAMPLE DOCUMENTS:")
        for i, point in enumerate(points[:5]):
            text = point.payload.get("text", "")
            source = point.payload.get("source", "unknown")
            doc_type = point.payload.get("type", "unknown")
            
            logger.info(f"\nDocument {i+1}:")
            logger.info(f"   Source: {source}")
            logger.info(f"   Type: {doc_type}")
            logger.info(f"   Text: {text[:200]}...")
            
            # Look for pricing in this document
            import re
            prices = re.findall(r'\$\d+(?:\.\d{2})?', text)
            if prices:
                logger.info(f"   💰 Prices in this doc: {prices}")
        
        client.close()
        
    except Exception as e:
        logger.error(f"❌ Error analyzing knowledge base: {e}")
    
    await qdrant_rag.close()

async def main():
    """Run comprehensive pricing diagnostic"""
    
    logger.info("🩺 PRICING DIAGNOSTIC TOOL")
    logger.info("=" * 60)
    logger.info("This will help identify why your agent gives inconsistent pricing")
    
    try:
        # Test pricing queries
        await test_pricing_queries()
        
        # Analyze knowledge base content
        await analyze_knowledge_base_content()
        
        logger.info(f"\n🎯 DIAGNOSTIC COMPLETE")
        logger.info("=" * 60)
        logger.info("💡 RECOMMENDATIONS:")
        logger.info("1. Check your Excel file for conflicting pricing information")
        logger.info("2. Ensure consistent pricing structure across all documents")
        logger.info("3. Remove or update conflicting price entries")
        logger.info("4. Re-ingest your Excel file after fixing pricing")
        logger.info("5. Test the system again with the diagnostic script")
        
    except Exception as e:
        logger.error(f"❌ Diagnostic failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())