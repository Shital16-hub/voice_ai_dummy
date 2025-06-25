# simple_rag_system.py - SIMPLIFIED & RELIABLE RAG
"""
Simplified RAG System for LiveKit Voice Agents
Based on LiveKit official examples and best practices

Key Features:
1. Fast initialization (< 2 seconds)
2. Reliable Excel data retrieval
3. Proper error handling
4. Context injection that works
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

class SimpleRAGSystem:
    """
    Simplified RAG system optimized for reliability over speed
    """
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.openai_client: Optional[openai.AsyncOpenAI] = None
        self.ready = False
        
        # Simple cache for frequent queries
        self.cache = {}
        self.max_cache_size = 50
        
        # Pre-loaded responses for common queries - REMOVED
        # All responses must come from Excel knowledge base only
        self.quick_responses = {}
        
    async def initialize(self) -> bool:
        """Simple, reliable initialization"""
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
                timeout=5
            )
            
            self.openai_client = openai.AsyncOpenAI(
                api_key=config.openai_api_key,
                timeout=5.0
            )
            
            # Step 3: Ensure collection exists
            await self._ensure_collection_exists()
            
            # Step 4: Test search capability
            test_success = await self._test_search()
            
            self.ready = test_success
            elapsed = (time.time() - start_time) * 1000
            
            if self.ready:
                logger.info(f"✅ Simple RAG system ready in {elapsed:.1f}ms")
                return True
            else:
                logger.warning(f"⚠️ RAG system partially ready in {elapsed:.1f}ms")
                return False
                
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
            return False
    
    def _check_qdrant_health(self) -> bool:
        """Quick health check"""
        try:
            import requests
            response = requests.get(f"{config.qdrant_url}/", timeout=2)
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
    
    async def _test_search(self) -> bool:
        """Test search functionality - but allow empty collections"""
        try:
            # Get collection info
            info = await asyncio.to_thread(
                self.client.get_collection,
                config.qdrant_collection_name
            )
            
            points_count = info.points_count
            logger.info(f"📊 Collection has {points_count} documents")
            
            if points_count > 0:
                # Test a simple search if we have data
                test_results = await self.search("test query", limit=1)
                logger.info(f"🔍 Search test: {'✅ Working' if test_results else '⚠️ No results'}")
            else:
                logger.info("📝 Collection is empty but ready to accept documents")
            
            # System is ready even with empty collection
            return True
                
        except Exception as e:
            logger.error(f"❌ Search test failed: {e}")
            return False
    
    async def search(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """
        Simple, reliable search with fallbacks
        """
        if not self.ready:
            logger.warning("⚠️ RAG system not ready - no knowledge base data available")
            return []
        
        try:
            # Check cache first
            cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()[:16]
            if cache_key in self.cache:
                logger.debug("📚 Using cached result")
                return self.cache[cache_key]
            
            # Check quick responses - DISABLED to force knowledge base usage
            # All responses must come from Excel data only
            # No hardcoded responses allowed
            
            # Perform vector search
            start_time = time.time()
            
            # Create embedding
            embedding = await self._create_embedding(query)
            if not embedding:
                logger.warning("⚠️ Failed to create embedding - no knowledge base search possible")
                return []
            
            # Search
            search_result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.search,
                    collection_name=config.qdrant_collection_name,
                    query_vector=embedding,
                    limit=limit,
                    score_threshold=0.2
                ),
                timeout=2.0
            )
            
            # Format results
            results = []
            for point in search_result:
                text = point.payload.get("text", "")
                if len(text) > 150:  # Keep concise for voice
                    text = text[:147] + "..."
                
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
            logger.info(f"🔍 Search completed in {elapsed:.1f}ms, found {len(results)} results")
            
            return results
            
        except asyncio.TimeoutError:
            logger.warning("⏰ Search timeout - NO fallback, knowledge base only")
            return []
        except Exception as e:
            logger.error(f"❌ Search error: {e} - NO fallback, knowledge base only")
            return []
    
    async def _create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding with error handling"""
        try:
            # Clean and prepare text
            clean_text = text.strip()
            if not clean_text:
                logger.warning("⚠️ Empty text provided for embedding")
                return None
                
            logger.debug(f"Creating embedding for text: {clean_text[:100]}...")
            
            response = await asyncio.wait_for(
                self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=clean_text[:1000]  # Truncate for speed
                ),
                timeout=10.0  # Increased timeout for embedding creation
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"✅ Successfully created embedding of size {len(embedding)}")
            return embedding
            
        except asyncio.TimeoutError:
            logger.error("⏰ Embedding creation timeout")
            return None
        except Exception as e:
            logger.error(f"❌ Embedding creation failed: {e}")
            return None
    
    def _get_fallback_response(self, query: str) -> List[Dict[str, Any]]:
        """Return empty result when search fails - NO hardcoded responses"""
        logger.warning(f"⚠️ No knowledge base results for: {query}")
        logger.info("💡 This query will be handled without knowledge base context")
        
        # Return empty list - let the LLM handle it without knowledge base data
        return []
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the collection"""
        try:
            logger.info(f"📝 Starting to add {len(documents)} documents...")
            points = []
            
            for i, doc in enumerate(documents):
                try:
                    # Create embedding
                    logger.debug(f"Creating embedding for document {i+1}/{len(documents)}")
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
                        logger.info(f"   📝 Processed {i+1}/{len(documents)} documents")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing document {i+1}: {e}")
                    continue
            
            if points:
                logger.info(f"📤 Uploading {len(points)} documents to Qdrant...")
                
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
                
                logger.info(f"✅ Successfully added {len(points)} documents")
                
                # Verify upload
                await asyncio.sleep(1)  # Give Qdrant time to index
                status = await self.get_status()
                actual_count = status.get("points_count", 0)
                logger.info(f"🔍 Verification: Collection now has {actual_count} total documents")
                
                return True
            else:
                logger.warning("⚠️ No valid documents to add (all failed embedding creation)")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
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
                "quick_responses": len(self.quick_responses)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def close(self):
        """Clean up resources"""
        try:
            if self.client:
                self.client.close()
            logger.info("✅ Simple RAG system closed")
        except Exception as e:
            logger.error(f"❌ Error closing RAG system: {e}")

# Global instance - replace your existing qdrant_rag
simple_rag = SimpleRAGSystem()

# For backward compatibility
qdrant_rag = simple_rag