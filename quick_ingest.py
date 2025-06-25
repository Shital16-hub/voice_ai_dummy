# quick_ingest.py - SIMPLE & WORKING EXCEL INGESTION
"""
Quick Excel ingestion using your existing working qdrant_rag_system
This bypasses the initialization issues by using your proven RAG system
"""
import asyncio
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any
import uuid

# Use your existing working RAG system
from qdrant_rag_system import qdrant_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def quick_ingest_excel(file_path: str) -> bool:
    """Quick and simple Excel ingestion"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            return False
        
        logger.info(f"📄 Processing Excel file: {file_path.name}")
        
        # Initialize the working RAG system
        logger.info("🔧 Initializing RAG system...")
        success = await qdrant_rag.initialize()
        if not success:
            logger.error("❌ Failed to initialize RAG system")
            return False
        
        logger.info("✅ RAG system initialized successfully")
        
        # Process Excel file
        documents = []
        excel_file = pd.ExcelFile(file_path)
        
        for sheet_name in excel_file.sheet_names:
            logger.info(f"   📋 Processing sheet: {sheet_name}")
            
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Process each row
            for index, row in df.iterrows():
                # Create content from all columns
                content_parts = []
                for col, val in row.items():
                    if pd.notna(val) and str(val).lower() not in ['nan', 'none', '']:
                        content_parts.append(f"{col}: {val}")
                
                if content_parts:
                    content = "; ".join(content_parts)
                    doc_id = f"{file_path.stem}_{sheet_name}_row_{index}"
                    
                    document = {
                        "id": doc_id,
                        "text": content,
                        "metadata": {
                            "source": file_path.name,
                            "sheet": sheet_name,
                            "row": index + 1,
                            "type": "excel_row"
                        }
                    }
                    documents.append(document)
            
            logger.info(f"   ✅ Extracted {len(documents)} entries from {sheet_name}")
        
        if not documents:
            logger.warning("⚠️ No valid documents extracted from Excel file")
            return False
        
        logger.info(f"📤 Adding {len(documents)} documents to knowledge base...")
        
        # Add documents using the working RAG system
        success = await qdrant_rag.add_documents(documents)
        
        if success:
            logger.info(f"🎉 Successfully ingested {len(documents)} documents")
            
            # Test search
            logger.info("🧪 Testing search functionality...")
            test_results = await qdrant_rag.search("service price", limit=1)
            if test_results:
                logger.info(f"✅ Search test successful: {test_results[0]['text'][:100]}...")
            else:
                logger.warning("⚠️ Search test returned no results")
            
            return True
        else:
            logger.error("❌ Failed to add documents to knowledge base")
            return False
            
    except Exception as e:
        logger.error(f"❌ Excel ingestion failed: {e}")
        return False
    finally:
        # Close the RAG system
        await qdrant_rag.close()

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Excel ingestion")
    parser.add_argument("--file", required=True, help="Excel file to process")
    args = parser.parse_args()
    
    success = await quick_ingest_excel(args.file)
    
    if success:
        logger.info("🎉 Excel ingestion completed successfully!")
        logger.info("💡 Next steps:")
        logger.info("   1. Test: python check_knowledge_base.py --full-report")
        logger.info("   2. Run agent: python main_fixed_rag.py dev")
    else:
        logger.error("❌ Excel ingestion failed!")

if __name__ == "__main__":
    asyncio.run(main())