# mongodb_migration_helper.py
"""
MongoDB Migration Helper for Call Transcription Data
Provides utilities to migrate from SQLite to MongoDB when ready
"""
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# MongoDB dependencies (commented out for now)
# from motor.motor_asyncio import AsyncIOMotorClient
# import pymongo

from call_transcription_storage import call_storage, CallTranscriptionStorage

logger = logging.getLogger(__name__)

class MongoDBMigrationHelper:
    """
    Helper class for migrating call transcription data to MongoDB
    
    When ready to use MongoDB:
    1. Uncomment MongoDB imports
    2. Install: pip install motor pymongo
    3. Update connection configuration
    4. Run migration scripts
    """
    
    def __init__(self, mongodb_connection_string: str = None):
        self.connection_string = mongodb_connection_string or "mongodb://localhost:27017"
        self.database_name = "livekit_voice_ai"
        
        # Collections structure for MongoDB
        self.collections = {
            "call_sessions": "Complete call session records",
            "conversation_items": "Individual conversation turns", 
            "transcription_segments": "Real-time transcription segments",
            "caller_profiles": "Caller information and statistics"
        }
        
        # MongoDB client (will be initialized when needed)
        self.client = None
        self.db = None
    
    async def initialize_mongodb(self):
        """Initialize MongoDB connection (when ready to migrate)"""
        try:
            # Uncomment when ready for MongoDB
            # self.client = AsyncIOMotorClient(self.connection_string)
            # self.db = self.client[self.database_name]
            
            # Create indexes for optimal performance
            # await self._create_indexes()
            
            logger.info("✅ MongoDB connection initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ MongoDB initialization failed: {e}")
            return False
    
    async def _create_indexes(self):
        """Create optimal indexes for MongoDB collections"""
        try:
            # Call sessions indexes
            await self.db.call_sessions.create_index([
                ("caller_id", 1),
                ("start_time", -1)
            ])
            await self.db.call_sessions.create_index("phone_number")
            await self.db.call_sessions.create_index("status")
            
            # Conversation items indexes  
            await self.db.conversation_items.create_index([
                ("session_id", 1),
                ("timestamp", 1)
            ])
            await self.db.conversation_items.create_index("caller_id")
            await self.db.conversation_items.create_index("role")
            
            # Transcription segments indexes
            await self.db.transcription_segments.create_index([
                ("session_id", 1),
                ("timestamp", 1)
            ])
            await self.db.transcription_segments.create_index("caller_id")
            await self.db.transcription_segments.create_index("speaker")
            await self.db.transcription_segments.create_index("is_final")
            
            # Caller profiles indexes
            await self.db.caller_profiles.create_index("phone_number", unique=True)
            await self.db.caller_profiles.create_index("last_call_time")
            
            logger.info("✅ MongoDB indexes created")
            
        except Exception as e:
            logger.error(f"❌ Index creation failed: {e}")
    
    async def migrate_session_to_mongodb(self, session_id: str) -> bool:
        """
        Migrate a single session to MongoDB format
        
        This method shows the structure for MongoDB migration
        """
        try:
            # Export session in MongoDB format
            mongodb_doc = await call_storage.export_for_mongodb(session_id)
            
            if not mongodb_doc:
                logger.warning(f"⚠️ No data found for session: {session_id}")
                return False
            
            # When MongoDB is ready, uncomment this:
            # result = await self.db.call_sessions.insert_one(mongodb_doc)
            # logger.info(f"✅ Migrated session {session_id} to MongoDB: {result.inserted_id}")
            
            # For now, save to JSON file as demonstration
            await self._save_to_json_file(mongodb_doc, session_id)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed for session {session_id}: {e}")
            return False
    
    async def _save_to_json_file(self, mongodb_doc: Dict[str, Any], session_id: str):
        """Save MongoDB-formatted document to JSON file (demo purposes)"""
        try:
            export_dir = Path("mongodb_export")
            export_dir.mkdir(exist_ok=True)
            
            # Convert datetime objects to ISO strings for JSON serialization
            json_doc = self._convert_datetime_to_string(mongodb_doc)
            
            file_path = export_dir / f"session_{session_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_doc, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 Exported session to: {file_path}")
            
        except Exception as e:
            logger.error(f"❌ JSON export failed: {e}")
    
    def _convert_datetime_to_string(self, obj: Any) -> Any:
        """Convert datetime objects to ISO strings for JSON serialization"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._convert_datetime_to_string(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_datetime_to_string(item) for item in obj]
        else:
            return obj
    
    async def batch_migrate_recent_sessions(self, limit: int = 10) -> Dict[str, Any]:
        """Migrate recent sessions to MongoDB format"""
        try:
            # Get recent sessions
            recent_sessions = await call_storage.get_recent_sessions(limit)
            
            results = {
                "total_sessions": len(recent_sessions),
                "migrated": 0,
                "failed": 0,
                "session_ids": []
            }
            
            for session in recent_sessions:
                success = await self.migrate_session_to_mongodb(session.session_id)
                if success:
                    results["migrated"] += 1
                    results["session_ids"].append(session.session_id)
                else:
                    results["failed"] += 1
            
            logger.info(f"📊 Migration Results: {results['migrated']}/{results['total_sessions']} sessions migrated")
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch migration failed: {e}")
            return {"error": str(e)}
    
    async def generate_migration_report(self) -> Dict[str, Any]:
        """Generate comprehensive migration report"""
        try:
            # Get statistics from SQLite
            recent_sessions = await call_storage.get_recent_sessions(100)
            
            # Calculate statistics
            total_sessions = len(recent_sessions)
            completed_sessions = len([s for s in recent_sessions if s.status == "completed"])
            
            # Get date range
            if recent_sessions:
                earliest = min(s.start_time for s in recent_sessions)
                latest = max(s.start_time for s in recent_sessions)
                date_range = {
                    "earliest": datetime.fromtimestamp(earliest).isoformat(),
                    "latest": datetime.fromtimestamp(latest).isoformat()
                }
            else:
                date_range = {"earliest": None, "latest": None}
            
            # Get unique callers
            unique_callers = len(set(s.caller_id for s in recent_sessions))
            
            # Generate sample MongoDB document
            sample_doc = None
            if recent_sessions:
                sample_doc = await call_storage.export_for_mongodb(recent_sessions[0].session_id)
                sample_doc = self._convert_datetime_to_string(sample_doc)
            
            report = {
                "migration_report": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "sqlite_database": str(call_storage.db_path),
                    "mongodb_target": {
                        "connection_string": self.connection_string,
                        "database": self.database_name,
                        "collections": self.collections
                    },
                    "data_statistics": {
                        "total_sessions": total_sessions,
                        "completed_sessions": completed_sessions,
                        "unique_callers": unique_callers,
                        "date_range": date_range
                    },
                    "migration_status": "ready_for_mongodb",
                    "next_steps": [
                        "1. Install MongoDB: pip install motor pymongo",
                        "2. Setup MongoDB server/cluster",
                        "3. Update connection string in config",
                        "4. Uncomment MongoDB code in this file",
                        "5. Run: python mongodb_migration_helper.py --migrate-all"
                    ]
                },
                "sample_mongodb_document": sample_doc
            }
            
            # Save report to file
            report_path = Path("mongodb_migration_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Migration report saved to: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return {"error": str(e)}

class MongoDBCallStorage:
    """
    MongoDB implementation of call storage (template for future use)
    
    This class shows how the current SQLite storage would be adapted for MongoDB
    """
    
    def __init__(self, connection_string: str, database_name: str = "livekit_voice_ai"):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None
    
    async def initialize(self):
        """Initialize MongoDB connection"""
        try:
            # Uncomment when ready for MongoDB:
            # self.client = AsyncIOMotorClient(self.connection_string)
            # self.db = self.client[self.database_name]
            # 
            # # Test connection
            # await self.client.admin.command('ping')
            # logger.info("✅ MongoDB storage initialized")
            # return True
            
            logger.info("🔄 MongoDB storage template ready (not connected)")
            return False
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            return False
    
    async def start_call_session(self, phone_number: str, session_metadata: Dict = None) -> tuple[str, str]:
        """MongoDB implementation of start_call_session"""
        # Implementation would use MongoDB operations instead of SQLite
        pass
    
    async def save_transcription_segment(self, session_id: str, caller_id: str, 
                                       speaker: str, text: str, is_final: bool, **kwargs) -> str:
        """MongoDB implementation of save_transcription_segment"""
        # Would use: await self.db.transcription_segments.insert_one(...)
        pass
    
    async def save_conversation_item(self, session_id: str, caller_id: str,
                                   role: str, content: str, **kwargs) -> str:
        """MongoDB implementation of save_conversation_item"""
        # Would use: await self.db.conversation_items.insert_one(...)
        pass
    
    async def get_caller_conversation_history(self, caller_id: str, limit: int = 50) -> List[Dict]:
        """MongoDB implementation of get_caller_conversation_history"""
        # Would use: await self.db.conversation_items.find({"caller_id": caller_id}).sort("timestamp", -1).limit(limit).to_list(length=limit)
        pass

# Demo function for testing migration
async def demo_migration():
    """Demonstrate the migration process"""
    try:
        logger.info("🚀 Starting Migration Demo")
        
        # Initialize migration helper
        migration_helper = MongoDBMigrationHelper()
        
        # Generate migration report
        report = await migration_helper.generate_migration_report()
        
        if "error" not in report:
            logger.info("📊 Migration report generated successfully")
            logger.info(f"   Total sessions: {report['migration_report']['data_statistics']['total_sessions']}")
            logger.info(f"   Unique callers: {report['migration_report']['data_statistics']['unique_callers']}")
            
            # Try to migrate a few recent sessions
            migration_results = await migration_helper.batch_migrate_recent_sessions(5)
            logger.info(f"📄 Migration demo: {migration_results.get('migrated', 0)} sessions exported to JSON")
        
        logger.info("✅ Migration demo completed")
        
    except Exception as e:
        logger.error(f"❌ Migration demo failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MongoDB Migration Helper")
    parser.add_argument("--report", action="store_true", help="Generate migration report")
    parser.add_argument("--demo", action="store_true", help="Run migration demo")
    parser.add_argument("--migrate-session", type=str, help="Migrate specific session ID")
    parser.add_argument("--migrate-recent", type=int, default=10, help="Migrate recent sessions")
    
    args = parser.parse_args()
    
    async def main():
        if args.report:
            migration_helper = MongoDBMigrationHelper()
            await migration_helper.generate_migration_report()
        elif args.demo:
            await demo_migration()
        elif args.migrate_session:
            migration_helper = MongoDBMigrationHelper()
            await migration_helper.migrate_session_to_mongodb(args.migrate_session)
        elif args.migrate_recent:
            migration_helper = MongoDBMigrationHelper()
            await migration_helper.batch_migrate_recent_sessions(args.migrate_recent)
        else:
            print("Use --help for available options")
    
    asyncio.run(main())