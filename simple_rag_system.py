# fixed_simple_rag_system.py - PURE EXCEL RAG
"""
FIXED: Simple RAG System that ONLY uses Excel data
NO hardcoded responses, NO fallbacks
"""
import asyncio
import logging
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import openai

from config import config

logger = logging.getLogger(__name__)

class FixedSimpleRAGSystem:
    """
    FIXED: Simple RAG system that ONLY returns Excel data
    NO hardcoded responses, NO fallbacks
    """
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.openai_client: Optional[openai.AsyncOpenAI] = None
        self.ready = False
        
        # Simple cache for frequent queries
        self.cache = {}
        self.max_cache_size = 50
        
        # REMOVED: All quick_responses and fallbacks
        # System must use Excel data ONLY
        
    async def initialize(self) -> bool:
        """Initialize with strict Excel-only mode"""
        try:
            start_time = time.time()
            
            # Step 1: Check Qdrant availability
            if not self._check_qdrant_health():
                logger.error("❌ Qdrant not available at " + config.qdrant_url)
                logger.info("💡 Start Qdrant: docker-compose up -d")
                return False
            
            # Step 2: Initialize clients
            self.client = QdrantClient(
                url=config.qdrant_url,
                timeout=10  # Increased timeout
            )
            
            self.openai_client = openai.AsyncOpenAI(
                api_key=config.openai_api_key,
                timeout=10.0  # Increased timeout
            )
            
            # Step 3: Ensure collection exists
            await self._ensure_collection_exists()
            
            # Step 4: Verify Excel data exists
            data_exists = await self._verify_excel_data()
            
            if not data_exists:
                logger.error("❌ CRITICAL: No Excel data found in knowledge base!")
                logger.error("💡 Run: python quick_ingest.py --file data/your_excel_file.xlsx")
                return False
            
            self.ready = True
            elapsed = (time.time() - start_time) * 1000
            
            logger.info(f"✅ PURE Excel RAG system ready in {elapsed:.1f}ms")
            return True
                
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
            return False
    
    def _check_qdrant_health(self) -> bool:
        """Quick health check"""
        try:
            import requests
            response = requests.get(f"{config.qdrant_url}/", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    async def _ensure_collection_exists(self):
        """Ensure collection exists, create if needed"""
        try:
            collections = await asyncio.to_thread(self.client.get_collections)
            collection_names = [col.name for col in collections.collections]
            
            if config.qdrant_collection_name not in collection_names:
                logger.info(f"📂 Creating collection: {config.qdrant_collection_name}")
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=config.qdrant_collection_name,
                    vectors_config=VectorParams(
                        size=1536,
                        distance=Distance.COSINE
                    )
                )
                logger.info("✅ Collection created")
            else:
                logger.info(f"✅ Collection exists: {config.qdrant_collection_name}")
                
        except Exception as e:
            logger.error(f"❌ Collection setup failed: {e}")
            raise
    
    async def _verify_excel_data(self) -> bool:
        """Verify Excel data exists in collection"""
        try:
            info = await asyncio.to_thread(
                self.client.get_collection,
                config.qdrant_collection_name
            )
            
            points_count = info.points_count
            logger.info(f"📊 Collection has {points_count} Excel documents")
            
            if points_count > 0:
                # Test search to verify data is accessible
                test_results = await self.search("price", limit=1)
                if test_results:
                    logger.info("✅ Excel data is accessible and searchable")
                    return True
                else:
                    logger.warning("⚠️ Excel data exists but not searchable")
                    return False
            else:
                logger.error("❌ No Excel data found in collection")
                return False
                
        except Exception as e:
            logger.error(f"❌ Excel data verification failed: {e}")
            return False
    
    async def search(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """
        PURE Excel search - NO fallbacks
        """
        if not self.ready:
            logger.warning("⚠️ RAG system not ready - no Excel data available")
            return []
        
        try:
            # Check cache first
            cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()[:16]
            if cache_key in self.cache:
                logger.debug("📚 Using cached Excel result")
                return self.cache[cache_key]
            
            # Create embedding
            start_time = time.time()
            embedding = await self._create_embedding(query)
            if not embedding:
                logger.warning("⚠️ Failed to create embedding - no Excel search possible")
                return []
            
            # Search Excel data
            search_result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.search,
                    collection_name=config.qdrant_collection_name,
                    query_vector=embedding,
                    limit=limit,
                    score_threshold=0.15  # Lower threshold for more results
                ),
                timeout=5.0  # Increased timeout
            )
            
            # Format results
            results = []
            for point in search_result:
                text = point.payload.get("text", "")
                
                # Keep full text for accurate pricing info
                if len(text) > 300:  # Only truncate if very long
                    text = text[:297] + "..."
                
                results.append({
                    "id": str(point.id),
                    "text": text,
                    "score": float(point.score),
                    "metadata": {k: v for k, v in point.payload.items() if k != "text"}
                })
            
            # Cache results
            if len(self.cache) >= self.max_cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = results
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"🔍 Excel search completed in {elapsed:.1f}ms, found {len(results)} results")
            
            return results
            
        except asyncio.TimeoutError:
            logger.warning("⏰ Excel search timeout - NO fallback")
            return []
        except Exception as e:
            logger.error(f"❌ Excel search error: {e} - NO fallback")
            return []
    
    async def _create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding with better error handling"""
        try:
            # Clean and prepare text
            clean_text = text.strip()
            if not clean_text:
                logger.warning("⚠️ Empty text provided for embedding")
                return None
                
            logger.debug(f"Creating embedding for: {clean_text[:100]}...")
            
            response = await asyncio.wait_for(
                self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=clean_text[:1000]  # Truncate for speed
                ),
                timeout=15.0  # Increased timeout
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"✅ Created embedding of size {len(embedding)}")
            return embedding
            
        except asyncio.TimeoutError:
            logger.error("⏰ Embedding creation timeout")
            return None
        except Exception as e:
            logger.error(f"❌ Embedding creation failed: {e}")
            return None
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add Excel documents to the collection"""
        try:
            logger.info(f"📝 Adding {len(documents)} Excel documents...")
            points = []
            
            for i, doc in enumerate(documents):
                try:
                    # Create embedding
                    logger.debug(f"Processing Excel document {i+1}/{len(documents)}")
                    embedding = await self._create_embedding(doc["text"])
                    if not embedding:
                        logger.warning(f"⚠️ Failed to create embedding for document {i+1}")
                        continue
                    
                    point = PointStruct(
                        id=doc["id"],
                        vector=embedding,
                        payload={
                            "text": doc["text"],
                            **doc.get("metadata", {})
                        }
                    )
                    points.append(point)
                    
                    # Log progress every 5 documents
                    if (i + 1) % 5 == 0:
                        logger.info(f"   📝 Processed {i+1}/{len(documents)} Excel documents")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing Excel document {i+1}: {e}")
                    continue
            
            if points:
                logger.info(f"📤 Uploading {len(points)} Excel documents to Qdrant...")
                
                # Upload in batches for reliability
                batch_size = 10
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    logger.debug(f"Uploading batch {i//batch_size + 1}")
                    
                    await asyncio.to_thread(
                        self.client.upsert,
                        collection_name=config.qdrant_collection_name,
                        points=batch
                    )
                
                logger.info(f"✅ Successfully added {len(points)} Excel documents")
                
                # Verify upload
                await asyncio.sleep(1)  # Give Qdrant time to index
                status = await self.get_status()
                actual_count = status.get("points_count", 0)
                logger.info(f"🔍 Verification: Collection now has {actual_count} total Excel documents")
                
                return True
            else:
                logger.warning("⚠️ No valid Excel documents to add")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to add Excel documents: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        if not self.ready:
            return {"status": "not_ready", "cache_size": len(self.cache)}
        
        try:
            info = await asyncio.to_thread(
                self.client.get_collection,
                config.qdrant_collection_name
            )
            
            return {
                "status": "ready",
                "points_count": info.points_count,
                "cache_size": len(self.cache),
                "excel_only": True  # Indicates pure Excel mode
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def close(self):
        """Clean up resources"""
        try:
            if self.client:
                self.client.close()
            logger.info("✅ PURE Excel RAG system closed")
        except Exception as e:
            logger.error(f"❌ Error closing RAG system: {e}")

# Global instance - PURE Excel mode
simple_rag = FixedSimpleRAGSystem()