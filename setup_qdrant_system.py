# setup_qdrant_system.py
"""
Setup script for Qdrant RAG system - No sample data
"""
import asyncio
import logging
from pathlib import Path

from qdrant_rag_system import qdrant_rag
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_qdrant_system():
    """Test the Qdrant RAG system"""
    try:
        # Initialize system
        success = await qdrant_rag.initialize()
        if not success:
            logger.error("❌ Failed to initialize Qdrant system")
            return False
        
        # Check if collection has data
        from qdrant_client import QdrantClient
        client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
        
        try:
            collection_info = client.get_collection(config.qdrant_collection_name)
            point_count = collection_info.points_count
            logger.info(f"📊 Collection has {point_count} documents")
            
            if point_count > 0:
                # Test a simple query
                results = await qdrant_rag.search("information")
                if results:
                    logger.info(f"✅ Search test successful - Found {len(results)} results")
                else:
                    logger.warning("⚠️ Search test returned no results")
            else:
                logger.info("📄 Collection is empty - ready for data ingestion")
                
        except Exception as e:
            logger.info("📁 Collection doesn't exist yet - will be created during ingestion")
        
        await qdrant_rag.close()
        logger.info("✅ Qdrant system test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Qdrant system test failed: {e}")
        return False

async def main():
    """Main setup function"""
    logger.info("🔧 Setting up Qdrant RAG System")
    
    # Ensure data directory exists
    config.data_dir.mkdir(exist_ok=True)
    logger.info(f"📁 Data directory: {config.data_dir.absolute()}")
    
    # Test system
    await test_qdrant_system()
    
    logger.info("🎉 Qdrant RAG System setup completed!")
    logger.info("📋 Next steps:")
    logger.info("1. Place your PDF/Excel files in the 'data' directory")
    logger.info("2. Run: python qdrant_data_ingestion.py --directory data")
    logger.info("3. Start the agent: python ultra_fast_qdrant_agent.py dev")

if __name__ == "__main__":
    asyncio.run(main())