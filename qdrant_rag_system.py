# qdrant_rag_system.py - ULTRA-LOW LATENCY VERSION
"""
Ultra-Fast Qdrant RAG System optimized for <500ms response times
Based on LiveKit performance best practices
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import uuid
import requests
import hashlib

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, 
    FieldCondition, MatchValue, SearchParams, OptimizersConfigDiff,
    HnswConfigDiff
)
import openai

from config import config

logger = logging.getLogger(__name__)

class UltraFastQdrantRAG:
    """
    Ultra-fast Qdrant RAG system optimized for telephony with <500ms response times
    """
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.aclient: Optional[AsyncQdrantClient] = None
        self.openai_client: Optional[openai.AsyncOpenAI] = None
        self.ready = False
        
        # AGGRESSIVE CACHING for ultra-low latency
        self.search_cache = {}
        self.embedding_cache = {}
        self.max_cache_size = 50  # Smaller for memory efficiency
        self.max_embedding_cache_size = 200
        
        # Pre-computed embeddings for instant responses
        self.instant_responses = {}
        
    async def initialize(self) -> bool:
        """Initialize with ultra-fast settings"""
        try:
            start_time = time.time()
            
            # Quick health check
            if not await self._quick_health_check():
                logger.error("❌ Qdrant not available")
                return False
            
            # Initialize clients with minimal timeouts
            await self._init_ultra_fast_clients()
            
            # Setup collection with speed optimizations
            await self._setup_speed_optimized_collection()
            
            # Pre-warm cache with telephony queries
            await self._instant_cache_warmup()
            
            self.ready = True
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"🚀 ULTRA-FAST Qdrant RAG initialized in {elapsed:.1f}ms")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ultra-fast RAG initialization failed: {e}")
            return False
    
    async def _quick_health_check(self) -> bool:
        """Lightning-fast health check"""
        try:
            response = requests.get(f"{config.qdrant_url}/", timeout=1)
            return response.status_code == 200
        except:
            return False
    
    async def _init_ultra_fast_clients(self):
        """Initialize clients with minimal latency settings"""
        self.client = QdrantClient(
            url=config.qdrant_url,
            timeout=2,  # Very short timeout
            prefer_grpc=False  # HTTP is often faster for small operations
        )
        
        self.aclient = AsyncQdrantClient(
            url=config.qdrant_url,
            timeout=2
        )
        
        self.openai_client = openai.AsyncOpenAI(
            api_key=config.openai_api_key,
            timeout=3.0  # Short timeout for embeddings
        )
        
        logger.info("⚡ Ultra-fast clients initialized")
    
    async def _setup_speed_optimized_collection(self):
        """Setup collection optimized for minimal search latency"""
        try:
            collection_name = config.qdrant_collection_name
            
            collections = await asyncio.to_thread(self.client.get_collections)
            collection_exists = any(col.name == collection_name for col in collections.collections)
            
            if not collection_exists:
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=1536,
                        distance=Distance.COSINE,
                        # ULTRA-FAST HNSW settings
                        hnsw_config=HnswConfigDiff(
                            m=8,                    # Reduced for speed
                            ef_construct=64,        # Reduced for speed
                            full_scan_threshold=1000,  # Lower threshold
                            max_indexing_threads=1,
                        )
                    ),
                    # Speed-optimized storage
                    optimizers_config=OptimizersConfigDiff(
                        default_segment_number=1,
                        max_segment_size=10000,    # Smaller segments
                        memmap_threshold=0,
                        indexing_threshold=1000,   # Index sooner
                        flush_interval_sec=2,      # Faster flushing
                        max_optimization_threads=1
                    )
                )
                logger.info("⚡ Speed-optimized collection created")
            else:
                logger.info("⚡ Using existing speed-optimized collection")
                
        except Exception as e:
            logger.error(f"❌ Collection setup failed: {e}")
            raise
    
    async def _instant_cache_warmup(self):
        """Pre-compute embeddings for instant responses"""
        instant_queries = {
            "towing": "Towing service available 24/7. Base rate $75 plus $3/mile.",
            "battery": "Jump start service $25. Battery replacement available.",
            "tire": "Tire change service $35. Spare tire installation included.",
            "pricing": "Service rates: Towing $75+, Jump start $25, Tire change $35.",
            "emergency": "Emergency service available 24/7. Priority dispatch.",
            "location": "Please provide your exact location for fastest service.",
            "cost": "Standard rates apply. Exact quote provided on arrival.",
            "help": "Roadside assistance available for towing, battery, and tire services."
        }
        
        start_time = time.time()
        for query, response in instant_queries.items():
            try:
                # Pre-compute embedding
                embedding = await self._create_embedding_ultra_fast(query)
                self.instant_responses[query] = {
                    "embedding": embedding,
                    "response": response,
                    "score": 1.0
                }
            except Exception as e:
                logger.warning(f"Failed to pre-compute for '{query}': {e}")
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"⚡ Instant cache warmed in {elapsed:.1f}ms with {len(self.instant_responses)} responses")
    
    async def _create_embedding_ultra_fast(self, text: str) -> List[float]:
        """Ultra-fast embedding creation with aggressive caching"""
        cache_key = hashlib.md5(text.lower().strip().encode()).hexdigest()
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            response = await asyncio.wait_for(
                self.openai_client.embeddings.create(
                    model="text-embedding-3-small",  # Fastest model
                    input=text[:1000]  # Truncate for speed
                ),
                timeout=2.0  # Very short timeout
            )
            
            embedding = response.data[0].embedding
            
            # Cache management
            if len(self.embedding_cache) >= self.max_embedding_cache_size:
                # Remove oldest 20%
                old_keys = list(self.embedding_cache.keys())[:int(self.max_embedding_cache_size * 0.2)]
                for key in old_keys:
                    del self.embedding_cache[key]
            
            self.embedding_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Ultra-fast embedding failed: {e}")
            # Return a default embedding to avoid complete failure
            return [0.0] * 1536
    
    def _check_instant_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Check for instant pre-computed responses"""
        query_lower = query.lower().strip()
        
        # Direct keyword matches for instant responses
        for keyword, response_data in self.instant_responses.items():
            if keyword in query_lower:
                return [{
                    "id": f"instant_{keyword}",
                    "text": response_data["response"],
                    "score": response_data["score"],
                    "metadata": {"source": "instant_response", "type": "cached"}
                }]
        
        return None
    
    async def search(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """Ultra-fast search with <500ms target"""
        if not self.ready:
            return []
        
        try:
            start_time = time.time()
            
            # STEP 1: Check instant responses first (0-5ms)
            instant_result = self._check_instant_response(query)
            if instant_result:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"⚡ INSTANT response in {elapsed:.1f}ms")
                return instant_result
            
            # STEP 2: Check search cache (5-10ms)
            cache_key = f"{query.lower().strip()}_{limit or 1}"
            if cache_key in self.search_cache:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"⚡ CACHED response in {elapsed:.1f}ms")
                return self.search_cache[cache_key]
            
            # STEP 3: Ultra-fast vector search (200-500ms)
            query_embedding = await self._create_embedding_ultra_fast(query)
            
            search_result = await asyncio.wait_for(
                self.aclient.search(
                    collection_name=config.qdrant_collection_name,
                    query_vector=query_embedding,
                    limit=1,  # Always limit to 1 for speed
                    score_threshold=0.2,  # Lower threshold for more results
                    search_params=SearchParams(
                        hnsw_ef=16,  # Very low for maximum speed
                        exact=False  # Approximate search for speed
                    )
                ),
                timeout=0.3  # 300ms timeout
            )
            
            # Format results
            results = []
            for point in search_result:
                text = point.payload.get("text", "")
                if len(text) > 100:  # Truncate for telephony
                    text = text[:97] + "..."
                
                results.append({
                    "id": str(point.id),
                    "text": text,
                    "score": float(point.score),
                    "metadata": {k: v for k, v in point.payload.items() if k != "text"}
                })
            
            # Cache results
            if len(self.search_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.search_cache))
                del self.search_cache[oldest_key]
            
            self.search_cache[cache_key] = results
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"⚡ ULTRA-FAST search in {elapsed:.1f}ms, found {len(results)} results")
            return results
            
        except asyncio.TimeoutError:
            elapsed = (time.time() - start_time) * 1000
            logger.warning(f"⚠️ Search timeout after {elapsed:.1f}ms - using fallback")
            
            # Return a generic helpful response on timeout
            return [{
                "id": "timeout_fallback",
                "text": "I can help you with towing, battery, or tire services. What do you need?",
                "score": 0.5,
                "metadata": {"source": "timeout_fallback"}
            }]
            
        except Exception as e:
            logger.error(f"❌ Ultra-fast search error: {e}")
            return []
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents with speed optimizations"""
        try:
            points = []
            
            # Process in small batches for speed
            batch_size = 10
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                for doc in batch:
                    embedding = await self._create_embedding_ultra_fast(doc["text"])
                    point_id = str(uuid.uuid4())
                    
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": doc["text"][:500],  # Limit text size for speed
                            "original_id": doc["id"],
                            **doc.get("metadata", {})
                        }
                    )
                    points.append(point)
                
                # Insert batch
                await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=config.qdrant_collection_name,
                    points=points[-len(batch):]
                )
            
            logger.info(f"⚡ Added {len(points)} documents with speed optimization")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "search_cache_size": len(self.search_cache),
            "embedding_cache_size": len(self.embedding_cache),
            "instant_responses": len(self.instant_responses),
            "ready": self.ready
        }
    
    async def close(self):
        """Clean up resources"""
        try:
            if self.client:
                self.client.close()
            if self.aclient:
                await self.aclient.close()
            logger.info("⚡ Ultra-fast RAG system closed")
        except Exception as e:
            logger.error(f"❌ Error closing: {e}")

# Global ultra-fast instance
qdrant_rag = UltraFastQdrantRAG()