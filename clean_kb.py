from qdrant_rag_system import qdrant_rag
import asyncio
async def clear_collection():
    await qdrant_rag.initialize()
    try:
        await qdrant_rag.client.delete_collection('telephony_knowledge')
        print('✅ Cleared old PDF data')
    except:
        print('Collection already empty or doesn\'t exist')
    await qdrant_rag.close()
asyncio.run(clear_collection())