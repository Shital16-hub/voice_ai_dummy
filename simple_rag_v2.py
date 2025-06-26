# simple_rag_v2.py - SIMPLIFIED RAG SYSTEM
"""
Simplified RAG System based on LiveKit examples
Much simpler, more reliable, and follows LiveKit patterns
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient

from config import config

logger = logging.getLogger(__name__)

class SimplifiedRAGSystem:
    """
    Simplified RAG system using LlamaIndex patterns like LiveKit examples
    Much simpler than the previous over-engineered version
    """
    
    def __init__(self):
        self.index: Optional[VectorStoreIndex] = None
        self.ready = False
        
        # Simple cache for frequent queries
        self.cache = {}
        self.max_cache_size = 100
        
    async def initialize(self) -> bool:
        """Simple initialization following LiveKit patterns"""
        try:
            start_time = time.time()
            logger.info("🔧 Initializing simplified RAG system...")
            
            # Configure LlamaIndex settings
            Settings.embed_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=config.openai_api_key
            )
            Settings.llm = OpenAI(
                model="gpt-4o-mini",
                api_key=config.openai_api_key
            )
            
            # Initialize Qdrant clients (both sync and async)
            from qdrant_client import QdrantClient, AsyncQdrantClient
            
            sync_client = QdrantClient(
                url=config.qdrant_url,
                timeout=10
            )
            
            async_client = AsyncQdrantClient(
                url=config.qdrant_url,
                timeout=10
            )
            
            # Create vector store with both clients
            vector_store = QdrantVectorStore(
                client=sync_client,
                aclient=async_client,
                collection_name=config.qdrant_collection_name
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Try to load from existing vector store or create empty index
            try:
                # Try to create index from existing vector store
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context
                )
            except Exception as e:
                logger.info(f"Creating new empty index: {e}")
                # Create empty index if no existing data
                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=storage_context
                )
            
            # Test the index with a simple query
            try:
                retriever = self.index.as_retriever(similarity_top_k=1)
                test_results = await retriever.aretrieve("test")
                logger.info(f"📊 Test query returned {len(test_results)} results")
            except Exception as e:
                logger.info(f"Test query failed (normal for empty index): {e}")
            
            elapsed = (time.time() - start_time) * 1000
            self.ready = True
            
            logger.info(f"✅ Simplified RAG ready in {elapsed:.1f}ms")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
            return False
    
    async def retrieve_context(self, query: str, max_results: int = 3) -> str:
        """
        Retrieve context for a query - simplified approach
        Returns formatted context string ready for LLM injection
        """
        if not self.ready or not self.index:
            logger.warning("⚠️ RAG system not ready")
            return ""
        
        try:
            # Check cache first
            cache_key = query.lower().strip()[:50]
            if cache_key in self.cache:
                logger.debug("📚 Cache hit")
                return self.cache[cache_key]
            
            # Retrieve documents using LlamaIndex
            retriever = self.index.as_retriever(
                similarity_top_k=max_results,
                # Use a lower threshold for better recall
                # LlamaIndex handles scoring internally
            )
            
            start_time = time.time()
            nodes = await asyncio.wait_for(
                retriever.aretrieve(query),
                timeout=3.0  # Increased timeout to 3 seconds for more reliable results
            )
            
            search_time = (time.time() - start_time) * 1000
            
            if not nodes:
                logger.warning("⚠️ No documents retrieved")
                return ""
            
            # Format context for voice response
            context_parts = []
            for node in nodes:
                # Get the text content
                content = node.text.strip()
                
                # Clean for voice
                content = self._clean_for_voice(content)
                
                if content and len(content) > 20:
                    context_parts.append(content)
                
                # Limit total context length
                if len(' '.join(context_parts)) > config.max_response_length * 2:
                    break
            
            if context_parts:
                context = ' | '.join(context_parts[:2])  # Use top 2 results
                
                # Cache the result
                self.cache[cache_key] = context
                if len(self.cache) > self.max_cache_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                
                logger.info(f"✅ Retrieved context in {search_time:.1f}ms")
                return context
            else:
                logger.warning("⚠️ No usable content found")
                return ""
                
        except asyncio.TimeoutError:
            logger.warning(f"⏰ RAG timeout after {config.rag_timeout_ms}ms")
            return ""
        except Exception as e:
            logger.error(f"❌ RAG error: {e}")
            return ""
    
    def _clean_for_voice(self, content: str) -> str:
        """Clean content for voice response"""
        if not content:
            return ""
        
        # Remove formatting
        content = content.replace("Q:", "").replace("A:", "")
        content = content.replace("•", "").replace("-", "").replace("*", "")
        content = content.replace("\n", " ").replace("\t", " ")
        
        # Remove multiple spaces
        while "  " in content:
            content = content.replace("  ", " ")
        
        # Keep it concise for voice
        if len(content) > config.max_response_length:
            sentences = content.split('.')
            if len(sentences) > 1:
                content = sentences[0] + "."
            else:
                content = content[:config.max_response_length] + "..."
        
        return content.strip()
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the index - simplified"""
        try:
            logger.info(f"📝 Adding {len(documents)} documents...")
            
            # Convert to LlamaIndex Document format
            from llama_index.core import Document
            
            llama_docs = []
            for doc in documents:
                llama_doc = Document(
                    text=doc["text"],
                    metadata=doc.get("metadata", {}),
                    doc_id=doc.get("id")
                )
                llama_docs.append(llama_doc)
            
            # Add documents to index in batches for better performance
            batch_size = 10
            for i in range(0, len(llama_docs), batch_size):
                batch = llama_docs[i:i + batch_size]
                logger.info(f"📤 Processing batch {i//batch_size + 1}/{(len(llama_docs)-1)//batch_size + 1}")
                
                for doc in batch:
                    # Use insert_nodes instead of insert for better control
                    self.index.insert(doc)
                
                # Small delay between batches to avoid overwhelming the system
                await asyncio.sleep(0.1)
            
            logger.info(f"✅ Added {len(llama_docs)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        if not self.ready:
            return {"status": "not_ready"}
        
        try:
            # Get some basic stats
            return {
                "status": "ready",
                "cache_size": len(self.cache),
                "index_ready": self.index is not None
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Global instance
simplified_rag = SimplifiedRAGSystem()