# test_embeddings.py - DEBUG OPENAI EMBEDDINGS
"""
Test script to verify OpenAI embeddings are working
Run this before ingesting data to ensure API key is valid
"""
import asyncio
import logging
import openai
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_openai_embeddings():
    """Test OpenAI embeddings API"""
    logger.info("🧪 Testing OpenAI Embeddings API...")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ OPENAI_API_KEY not found in environment variables")
        logger.info("💡 Add to .env file: OPENAI_API_KEY=sk-your-key-here")
        return False
    
    if not api_key.startswith("sk-"):
        logger.error("❌ Invalid OPENAI_API_KEY format (should start with 'sk-')")
        return False
    
    logger.info(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test embedding creation
    try:
        client = openai.AsyncOpenAI(api_key=api_key)
        
        test_text = "This is a test for roadside assistance towing service pricing"
        logger.info(f"🔍 Testing embedding for: '{test_text}'")
        
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=test_text
        )
        
        embedding = response.data[0].embedding
        logger.info(f"✅ Embedding created successfully!")
        logger.info(f"   Model: text-embedding-3-small")
        logger.info(f"   Embedding size: {len(embedding)}")
        logger.info(f"   Sample values: {embedding[:5]}")
        
        # Test with Excel-like content
        excel_content = "Service: Towing; Price: $75 base rate; Description: 24/7 emergency towing service"
        logger.info(f"🔍 Testing Excel-like content: '{excel_content}'")
        
        response2 = await client.embeddings.create(
            model="text-embedding-3-small",
            input=excel_content
        )
        
        embedding2 = response2.data[0].embedding
        logger.info(f"✅ Excel content embedding created!")
        logger.info(f"   Embedding size: {len(embedding2)}")
        
        return True
        
    except openai.AuthenticationError:
        logger.error("❌ OpenAI API authentication failed")
        logger.info("💡 Check your API key at https://platform.openai.com/api-keys")
        return False
    except openai.RateLimitError:
        logger.error("❌ OpenAI API rate limit exceeded")
        logger.info("💡 Wait a moment and try again")
        return False
    except openai.APIConnectionError:
        logger.error("❌ OpenAI API connection failed")
        logger.info("💡 Check your internet connection")
        return False
    except Exception as e:
        logger.error(f"❌ OpenAI API error: {e}")
        return False

async def test_qdrant_connection():
    """Test Qdrant connection"""
    logger.info("🧪 Testing Qdrant connection...")
    
    try:
        import requests
        response = requests.get("http://localhost:6333/", timeout=5)
        
        if response.status_code == 200:
            logger.info("✅ Qdrant is running and accessible")
            
            # Get collections
            collections_response = requests.get("http://localhost:6333/collections", timeout=5)
            if collections_response.status_code == 200:
                collections = collections_response.json()
                logger.info(f"📂 Found {len(collections.get('result', {}).get('collections', []))} collections")
                return True
            else:
                logger.warning("⚠️ Could not get collections info")
                return True
        else:
            logger.error(f"❌ Qdrant returned status {response.status_code}")
            return False
            
    except requests.ConnectionError:
        logger.error("❌ Cannot connect to Qdrant at http://localhost:6333")
        logger.info("💡 Start Qdrant: docker-compose up -d")
        return False
    except Exception as e:
        logger.error(f"❌ Qdrant connection error: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("🚀 Running pre-ingestion tests...")
    logger.info("=" * 50)
    
    # Test OpenAI
    openai_ok = await test_openai_embeddings()
    logger.info("")
    
    # Test Qdrant
    qdrant_ok = await test_qdrant_connection()
    logger.info("")
    
    # Summary
    logger.info("📊 TEST RESULTS:")
    logger.info(f"   OpenAI API: {'✅ Working' if openai_ok else '❌ Failed'}")
    logger.info(f"   Qdrant:     {'✅ Working' if qdrant_ok else '❌ Failed'}")
    logger.info("")
    
    if openai_ok and qdrant_ok:
        logger.info("🎉 All tests passed! Ready to ingest Excel data.")
        logger.info("💡 Next step: python ingest_excel_data.py --file data/your_file.xlsx")
    else:
        logger.error("❌ Some tests failed. Fix the issues above before proceeding.")
        
        if not openai_ok:
            logger.info("🔧 OpenAI fixes:")
            logger.info("   1. Get API key from https://platform.openai.com/api-keys")
            logger.info("   2. Add to .env: OPENAI_API_KEY=sk-your-key-here")
            
        if not qdrant_ok:
            logger.info("🔧 Qdrant fixes:")
            logger.info("   1. Start Qdrant: docker-compose up -d")
            logger.info("   2. Wait 30 seconds for startup")
            logger.info("   3. Check: curl http://localhost:6333")

if __name__ == "__main__":
    asyncio.run(main())