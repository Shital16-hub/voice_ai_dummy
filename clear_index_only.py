# clear_index_only.py
"""
Simple script to ONLY clear/delete the Qdrant collection
"""
import asyncio
import logging
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def clear_index_only():
    """Just delete the collection - that's it!"""
    try:
        # Connect to Qdrant
        client = QdrantClient(url="http://localhost:6333")
        
        # Delete the collection
        try:
            client.delete_collection("telephony_knowledge")
            logger.info("✅ Deleted collection: telephony_knowledge")
        except Exception as e:
            logger.info(f"Collection didn't exist or already deleted: {e}")
        
        client.close()
        logger.info("✅ Index cleared successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error clearing index: {e}")

if __name__ == "__main__":
    asyncio.run(clear_index_only())