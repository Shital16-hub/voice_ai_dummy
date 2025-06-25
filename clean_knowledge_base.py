# clean_knowledge_base.py
"""
Knowledge Base Cleanup Script for LiveKit Voice Agent
Cleans existing data from Qdrant vector database
"""
import asyncio
import logging
import sys
import argparse
from pathlib import Path

# Import your existing RAG system
from simple_rag_system import simple_rag
from config import config

# Try to import Qdrant client directly for advanced operations
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance
    QDRANT_CLIENT_AVAILABLE = True
except ImportError:
    QDRANT_CLIENT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBaseCleaner:
    """Clean and reset Qdrant knowledge base"""
    
    def __init__(self):
        self.qdrant_client = None
        
    async def initialize_qdrant_client(self) -> bool:
        """Initialize direct Qdrant client for cleanup operations"""
        try:
            if not QDRANT_CLIENT_AVAILABLE:
                logger.warning("⚠️ Direct Qdrant client not available, using simple_rag only")
                return False
            
            logger.info("🔧 Initializing direct Qdrant client...")
            self.qdrant_client = QdrantClient(
                url=config.qdrant_url,
                timeout=10
            )
            
            # Test connection
            collections = await asyncio.to_thread(self.qdrant_client.get_collections)
            logger.info(f"✅ Connected to Qdrant with {len(collections.collections)} collections")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Qdrant client: {e}")
            return False
    
    async def get_collection_info(self) -> dict:
        """Get current collection information"""
        try:
            logger.info("📊 Getting collection information...")
            
            # First try simple_rag
            status = await simple_rag.get_status()
            if status.get("status") == "ready":
                logger.info(f"✅ Collection '{config.qdrant_collection_name}' status via simple_rag:")
                logger.info(f"   Points count: {status.get('points_count', 'unknown')}")
                logger.info(f"   Cache size: {status.get('cache_size', 'unknown')}")
                return status
            
            # Try direct client if available
            if self.qdrant_client:
                try:
                    collection_info = await asyncio.to_thread(
                        self.qdrant_client.get_collection,
                        config.qdrant_collection_name
                    )
                    logger.info(f"✅ Collection '{config.qdrant_collection_name}' status via direct client:")
                    logger.info(f"   Points count: {collection_info.points_count}")
                    logger.info(f"   Vector size: {collection_info.config.params.vectors.size}")
                    logger.info(f"   Distance metric: {collection_info.config.params.vectors.distance}")
                    
                    return {
                        "status": "ready",
                        "points_count": collection_info.points_count,
                        "vector_size": collection_info.config.params.vectors.size,
                        "distance_metric": collection_info.config.params.vectors.distance
                    }
                except Exception as e:
                    logger.warning(f"⚠️ Could not get collection info via direct client: {e}")
            
            return {"status": "unknown"}
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection info: {e}")
            return {"status": "error", "error": str(e)}
    
    async def list_all_collections(self) -> None:
        """List all collections in Qdrant"""
        try:
            logger.info("📋 Listing all collections...")
            
            if self.qdrant_client:
                collections = await asyncio.to_thread(self.qdrant_client.get_collections)
                
                if collections.collections:
                    logger.info(f"Found {len(collections.collections)} collections:")
                    for collection in collections.collections:
                        logger.info(f"   📁 {collection.name}")
                else:
                    logger.info("No collections found")
            else:
                logger.warning("⚠️ Direct client not available, cannot list collections")
                
        except Exception as e:
            logger.error(f"❌ Failed to list collections: {e}")
    
    async def clear_collection_data(self) -> bool:
        """Clear all data from the collection without deleting the collection"""
        try:
            logger.info(f"🧹 Clearing data from collection: {config.qdrant_collection_name}")
            
            # Get initial count
            initial_info = await self.get_collection_info()
            initial_count = initial_info.get("points_count", 0)
            
            if initial_count == 0:
                logger.info("✅ Collection is already empty")
                return True
            
            logger.info(f"📊 Current collection has {initial_count} documents")
            
            if self.qdrant_client:
                # Use direct client for efficient clearing
                logger.info("🔧 Using direct client to clear collection...")
                
                # Delete all points in the collection
                await asyncio.to_thread(
                    self.qdrant_client.delete,
                    collection_name=config.qdrant_collection_name,
                    points_selector=True  # Select all points
                )
                
                # Wait a moment for the operation to complete
                await asyncio.sleep(2)
                
                # Verify clearing
                final_info = await self.get_collection_info()
                final_count = final_info.get("points_count", 0)
                
                if final_count == 0:
                    logger.info(f"✅ Successfully cleared {initial_count} documents from collection")
                    return True
                else:
                    logger.warning(f"⚠️ Collection still has {final_count} documents after clearing")
                    return False
            else:
                logger.warning("⚠️ Direct client not available, cannot clear collection efficiently")
                logger.info("💡 Try: docker restart qdrant-container or check Qdrant connection")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to clear collection: {e}")
            return False
    
    async def delete_collection(self) -> bool:
        """Delete the entire collection"""
        try:
            logger.warning(f"🗑️ DELETING ENTIRE COLLECTION: {config.qdrant_collection_name}")
            logger.warning("⚠️ This will permanently remove all data and the collection structure!")
            
            if self.qdrant_client:
                # Check if collection exists
                try:
                    await asyncio.to_thread(
                        self.qdrant_client.get_collection,
                        config.qdrant_collection_name
                    )
                except Exception:
                    logger.info("✅ Collection doesn't exist - nothing to delete")
                    return True
                
                # Delete the collection
                await asyncio.to_thread(
                    self.qdrant_client.delete_collection,
                    config.qdrant_collection_name
                )
                
                logger.info(f"✅ Collection '{config.qdrant_collection_name}' deleted successfully")
                return True
            else:
                logger.error("❌ Direct client not available, cannot delete collection")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete collection: {e}")
            return False
    
    async def recreate_collection(self) -> bool:
        """Recreate the collection with proper configuration"""
        try:
            logger.info(f"🔧 Recreating collection: {config.qdrant_collection_name}")
            
            if self.qdrant_client:
                # Create new collection with proper vector configuration
                await asyncio.to_thread(
                    self.qdrant_client.create_collection,
                    collection_name=config.qdrant_collection_name,
                    vectors_config=VectorParams(
                        size=config.embedding_dimensions,  # 1536 for text-embedding-3-small
                        distance=Distance.COSINE
                    )
                )
                
                logger.info(f"✅ Collection '{config.qdrant_collection_name}' recreated successfully")
                logger.info(f"   Vector size: {config.embedding_dimensions}")
                logger.info(f"   Distance metric: COSINE")
                return True
            else:
                logger.error("❌ Direct client not available, cannot recreate collection")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to recreate collection: {e}")
            return False
    
    async def reset_simple_rag_cache(self) -> None:
        """Reset simple_rag cache"""
        try:
            logger.info("🧹 Clearing simple_rag cache...")
            
            # Initialize simple_rag to reset its state
            await simple_rag.initialize()
            
            # Clear any internal caches
            if hasattr(simple_rag, 'cache'):
                simple_rag.cache.clear()
                logger.info("✅ simple_rag cache cleared")
            
            if hasattr(simple_rag, 'quick_responses'):
                simple_rag.quick_responses.clear()
                logger.info("✅ simple_rag quick responses cleared")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not reset simple_rag cache: {e}")

async def main():
    """Main cleanup function"""
    parser = argparse.ArgumentParser(description="Clean Qdrant knowledge base")
    parser.add_argument("--action", "-a", 
                       choices=["info", "clear", "delete", "reset"], 
                       default="info",
                       help="Action to perform: info (show info), clear (clear data), delete (delete collection), reset (delete and recreate)")
    parser.add_argument("--force", "-f", action="store_true", 
                       help="Skip confirmation prompts")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("🧹 Knowledge Base Cleanup Tool")
        logger.info(f"🎯 Target collection: {config.qdrant_collection_name}")
        logger.info(f"🔗 Qdrant URL: {config.qdrant_url}")
        
        # Create cleaner
        cleaner = KnowledgeBaseCleaner()
        
        # Initialize connections
        logger.info("\n🔧 Initializing connections...")
        qdrant_direct = await cleaner.initialize_qdrant_client()
        simple_rag_ready = await simple_rag.initialize()
        
        logger.info(f"   Direct Qdrant client: {'✅ Ready' if qdrant_direct else '⚠️ Not available'}")
        logger.info(f"   Simple RAG system: {'✅ Ready' if simple_rag_ready else '⚠️ Not ready'}")
        
        if args.action == "info":
            logger.info("\n📊 Collection Information:")
            await cleaner.list_all_collections()
            await cleaner.get_collection_info()
            
        elif args.action == "clear":
            logger.info("\n🧹 Clearing collection data...")
            
            # Show current status
            await cleaner.get_collection_info()
            
            # Confirm action
            if not args.force:
                response = input(f"\n⚠️ This will clear ALL data from '{config.qdrant_collection_name}'. Continue? [y/N]: ")
                if response.lower() != 'y':
                    logger.info("❌ Operation cancelled")
                    return
            
            success = await cleaner.clear_collection_data()
            if success:
                await cleaner.reset_simple_rag_cache()
                logger.info("✅ Knowledge base cleared successfully!")
                logger.info("💡 You can now run: python excel_ingest.py --file data/your_file.xlsx")
            else:
                logger.error("❌ Failed to clear knowledge base")
                sys.exit(1)
                
        elif args.action == "delete":
            logger.info("\n🗑️ Deleting collection...")
            
            # Show current status
            await cleaner.get_collection_info()
            
            # Confirm action
            if not args.force:
                response = input(f"\n⚠️ This will PERMANENTLY DELETE the collection '{config.qdrant_collection_name}'. Continue? [y/N]: ")
                if response.lower() != 'y':
                    logger.info("❌ Operation cancelled")
                    return
            
            success = await cleaner.delete_collection()
            if success:
                await cleaner.reset_simple_rag_cache()
                logger.info("✅ Collection deleted successfully!")
                logger.info("💡 Run with --action reset to recreate, or let ingestion recreate it")
            else:
                logger.error("❌ Failed to delete collection")
                sys.exit(1)
                
        elif args.action == "reset":
            logger.info("\n🔄 Resetting collection (delete and recreate)...")
            
            # Show current status
            await cleaner.get_collection_info()
            
            # Confirm action
            if not args.force:
                response = input(f"\n⚠️ This will DELETE and RECREATE '{config.qdrant_collection_name}'. Continue? [y/N]: ")
                if response.lower() != 'y':
                    logger.info("❌ Operation cancelled")
                    return
            
            # Delete
            delete_success = await cleaner.delete_collection()
            if not delete_success:
                logger.error("❌ Failed to delete collection")
                sys.exit(1)
            
            # Recreate
            create_success = await cleaner.recreate_collection()
            if not create_success:
                logger.error("❌ Failed to recreate collection")
                sys.exit(1)
            
            # Reset cache
            await cleaner.reset_simple_rag_cache()
            
            logger.info("✅ Collection reset successfully!")
            logger.info("💡 You can now run: python excel_ingest.py --file data/your_file.xlsx")
        
        logger.info("\n🎯 Cleanup completed!")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Cleanup cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())