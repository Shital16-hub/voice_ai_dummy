# enhanced_clean_qdrant.py
"""
Enhanced Qdrant Data Cleaning Script
Provides multiple cleaning options with safety checks and detailed reporting
"""
import asyncio
import argparse
import logging
from typing import Optional, List, Dict, Any
import requests
from pathlib import Path

from qdrant_rag_system import qdrant_rag
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantDataCleaner:
    """Enhanced Qdrant data cleaning with multiple options"""
    
    def __init__(self):
        self.client = None
        self.collection_name = config.qdrant_collection_name
    
    async def initialize(self) -> bool:
        """Initialize Qdrant connection"""
        try:
            # Check if Qdrant is running
            if not await self._check_qdrant_health():
                logger.error("❌ Qdrant is not running. Start it with: docker-compose up -d")
                return False
            
            # Initialize RAG system
            success = await qdrant_rag.initialize()
            if success:
                self.client = qdrant_rag.client
                logger.info("✅ Connected to Qdrant")
                return True
            else:
                logger.error("❌ Failed to initialize Qdrant RAG system")
                return False
                
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            return False
    
    async def _check_qdrant_health(self) -> bool:
        """Check if Qdrant Docker container is healthy"""
        try:
            response = requests.get(f"{config.qdrant_url}/", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    async def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed collection information"""
        try:
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                self.collection_name
            )
            
            # Get collection statistics
            stats = {
                "name": self.collection_name,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "vectors_config": collection_info.config.params.vectors,
                "status": collection_info.status
            }
            
            return stats
            
        except Exception as e:
            logger.warning(f"⚠️ Collection {self.collection_name} doesn't exist or error: {e}")
            return None
    
    async def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = await asyncio.to_thread(self.client.get_collections)
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.error(f"❌ Error listing collections: {e}")
            return []
    
    async def get_sample_points(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample points from collection"""
        try:
            # Scroll through points to get samples - using correct qdrant client method
            points, _ = await asyncio.to_thread(
                self.client.scroll,
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            samples = []
            for point in points:
                samples.append({
                    "id": str(point.id),
                    "payload": point.payload
                })
            
            return samples
            
        except Exception as e:
            logger.error(f"❌ Error getting sample points: {e}")
            return []
    
    async def delete_collection(self, confirm: bool = False) -> bool:
        """Delete entire collection"""
        if not confirm:
            logger.error("❌ Collection deletion requires confirmation")
            return False
        
        try:
            await asyncio.to_thread(
                self.client.delete_collection,
                self.collection_name
            )
            logger.info(f"✅ Deleted collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting collection: {e}")
            return False
    
    async def delete_points_by_filter(self, filter_condition: Dict[str, Any]) -> bool:
        """Delete points matching specific filter"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Convert filter to Qdrant format
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    ) for key, value in filter_condition.items()
                ]
            )
            
            result = await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=qdrant_filter
            )
            
            logger.info(f"✅ Deleted points matching filter: {filter_condition}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting points by filter: {e}")
            return False
    
    async def delete_points_by_source(self, source_file: str) -> bool:
        """Delete all points from a specific source file"""
        filter_condition = {"source": source_file}
        return await self.delete_points_by_filter(filter_condition)
    
    async def delete_points_by_type(self, doc_type: str) -> bool:
        """Delete all points of a specific document type"""
        filter_condition = {"type": doc_type}
        return await self.delete_points_by_filter(filter_condition)
    
    async def clear_all_points(self, confirm: bool = False) -> bool:
        """Clear all points but keep collection structure"""
        if not confirm:
            logger.error("❌ Clearing all points requires confirmation")
            return False
        
        try:
            # Get collection info first
            info = await self.get_collection_info()
            if not info:
                logger.warning("⚠️ Collection doesn't exist")
                return True
            
            points_count = info["points_count"]
            if points_count == 0:
                logger.info("✅ Collection is already empty")
                return True
            
            # Delete collection and recreate it
            await asyncio.to_thread(
                self.client.delete_collection,
                self.collection_name
            )
            
            # Recreate collection with same configuration
            await qdrant_rag._setup_collection()
            
            logger.info(f"✅ Cleared {points_count} points from collection")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error clearing points: {e}")
            return False
    
    async def cleanup_orphaned_data(self) -> bool:
        """Clean up any orphaned or corrupted data"""
        try:
            # Optimize collection
            await asyncio.to_thread(
                self.client.update_collection,
                collection_name=self.collection_name,
                optimizer_config={
                    "deleted_threshold": 0.1,
                    "vacuum_min_vector_number": 100
                }
            )
            
            logger.info("✅ Optimized collection and cleaned orphaned data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
            return False
    
    async def show_detailed_status(self):
        """Show detailed collection status"""
        logger.info("📊 QDRANT COLLECTION STATUS")
        logger.info("=" * 50)
        
        # Collection info
        info = await self.get_collection_info()
        if info:
            logger.info(f"Collection: {info['name']}")
            logger.info(f"Points: {info['points_count']}")
            logger.info(f"Segments: {info['segments_count']}")
            logger.info(f"Indexed vectors: {info['indexed_vectors_count']}")
            logger.info(f"Status: {info['status']}")
            
            # Sample points
            if info['points_count'] > 0:
                logger.info("\n📄 SAMPLE DATA:")
                samples = await self.get_sample_points(3)
                for i, sample in enumerate(samples, 1):
                    payload = sample['payload']
                    logger.info(f"  {i}. ID: {sample['id'][:8]}...")
                    logger.info(f"     Type: {payload.get('type', 'unknown')}")
                    logger.info(f"     Source: {payload.get('source', 'unknown')}")
                    logger.info(f"     Text: {payload.get('text', '')[:80]}...")
        else:
            logger.info("Collection doesn't exist or is empty")
        
        # All collections
        collections = await self.list_collections()
        logger.info(f"\n📂 ALL COLLECTIONS: {collections}")
    
    async def close(self):
        """Close connections"""
        if qdrant_rag.ready:
            await qdrant_rag.close()

async def main():
    """Main cleaning function with command line options"""
    parser = argparse.ArgumentParser(description="Enhanced Qdrant Data Cleaner")
    parser.add_argument("--action", choices=[
        "status", "clear-all", "delete-collection", "delete-by-source", 
        "delete-by-type", "cleanup", "interactive"
    ], default="interactive", help="Cleaning action to perform")
    
    parser.add_argument("--source", type=str, help="Source file to delete (for delete-by-source)")
    parser.add_argument("--type", type=str, help="Document type to delete (for delete-by-type)")
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    cleaner = QdrantDataCleaner()
    
    try:
        # Initialize
        if not await cleaner.initialize():
            logger.error("❌ Failed to initialize cleaner")
            return
        
        # Execute action
        if args.action == "status":
            await cleaner.show_detailed_status()
            
        elif args.action == "clear-all":
            if args.confirm or input("⚠️ Clear ALL points? (yes/no): ").lower() == "yes":
                await cleaner.clear_all_points(confirm=True)
            else:
                logger.info("❌ Operation cancelled")
                
        elif args.action == "delete-collection":
            if args.confirm or input("⚠️ Delete ENTIRE collection? (yes/no): ").lower() == "yes":
                await cleaner.delete_collection(confirm=True)
            else:
                logger.info("❌ Operation cancelled")
                
        elif args.action == "delete-by-source":
            if not args.source:
                logger.error("❌ --source required for delete-by-source")
                return
            await cleaner.delete_points_by_source(args.source)
            
        elif args.action == "delete-by-type":
            if not args.type:
                logger.error("❌ --type required for delete-by-type")
                return
            await cleaner.delete_points_by_type(args.type)
            
        elif args.action == "cleanup":
            await cleaner.cleanup_orphaned_data()
            
        elif args.action == "interactive":
            await interactive_mode(cleaner)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Operation cancelled by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        await cleaner.close()

async def interactive_mode(cleaner: QdrantDataCleaner):
    """Interactive mode for step-by-step cleaning"""
    while True:
        print("\n🧹 QDRANT DATA CLEANER - INTERACTIVE MODE")
        print("=" * 50)
        print("1. Show collection status")
        print("2. Clear all points (keep collection)")
        print("3. Delete entire collection")
        print("4. Delete by source file")
        print("5. Delete by document type")
        print("6. Cleanup orphaned data")
        print("7. Exit")
        
        try:
            choice = input("\nChoose an option (1-7): ").strip()
            
            if choice == "1":
                await cleaner.show_detailed_status()
                
            elif choice == "2":
                confirm = input("⚠️ Clear ALL points? This will remove all data! (yes/no): ")
                if confirm.lower() == "yes":
                    await cleaner.clear_all_points(confirm=True)
                else:
                    print("❌ Operation cancelled")
                    
            elif choice == "3":
                confirm = input("⚠️ Delete ENTIRE collection? This cannot be undone! (yes/no): ")
                if confirm.lower() == "yes":
                    await cleaner.delete_collection(confirm=True)
                else:
                    print("❌ Operation cancelled")
                    
            elif choice == "4":
                source = input("Enter source file path to delete: ").strip()
                if source:
                    await cleaner.delete_points_by_source(source)
                else:
                    print("❌ No source file specified")
                    
            elif choice == "5":
                doc_type = input("Enter document type to delete (json_entry, text_chunk, etc.): ").strip()
                if doc_type:
                    await cleaner.delete_points_by_type(doc_type)
                else:
                    print("❌ No document type specified")
                    
            elif choice == "6":
                await cleaner.cleanup_orphaned_data()
                
            elif choice == "7":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-7.")
                
        except KeyboardInterrupt:
            print("\n🛑 Exiting...")
            break
        except Exception as e:
            logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())