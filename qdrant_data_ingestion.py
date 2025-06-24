# qdrant_data_ingestion.py
"""
Enhanced Qdrant Data Ingestion Script
Supports PDF, Excel, JSON, and Text files
"""
import asyncio
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

# PDF and Excel processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import pandas as pd
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

from qdrant_rag_system import qdrant_rag
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedQdrantIngestion:
    """Enhanced data ingestion for PDF and Excel files"""
    
    def __init__(self):
        self.supported_formats = ['.json', '.txt', '.md']
        if PDF_AVAILABLE:
            self.supported_formats.extend(['.pdf'])
        if EXCEL_AVAILABLE:
            self.supported_formats.extend(['.xlsx', '.xls', '.csv'])
    
    async def ingest_directory(self, directory: Path, recursive: bool = True) -> bool:
        """Ingest all supported files from directory"""
        try:
            if not directory.exists():
                logger.error(f"❌ Directory not found: {directory}")
                return False
            
            # Initialize RAG system
            if not await qdrant_rag.initialize():
                logger.error("❌ Failed to initialize Qdrant RAG system")
                return False
            
            documents = []
            pattern = "**/*" if recursive else "*"
            
            for file_path in directory.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    logger.info(f"📄 Processing: {file_path.name}")
                    file_docs = await self._process_file(file_path)
                    documents.extend(file_docs)
                    logger.info(f"   ✅ Extracted {len(file_docs)} chunks")
            
            if not documents:
                logger.warning("⚠️ No documents found to ingest")
                return False
            
            # Add to Qdrant
            success = await qdrant_rag.add_documents(documents)
            
            if success:
                logger.info(f"🎉 Successfully ingested {len(documents)} documents")
                return True
            else:
                logger.error("❌ Failed to add documents to Qdrant")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {e}")
            return False
        finally:
            await qdrant_rag.close()
    
    async def _process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process individual file based on extension"""
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.json':
                return await self._process_json(file_path)
            elif suffix in ['.txt', '.md']:
                return await self._process_text(file_path)
            elif suffix == '.pdf' and PDF_AVAILABLE:
                return await self._process_pdf(file_path)
            elif suffix in ['.xlsx', '.xls'] and EXCEL_AVAILABLE:
                return await self._process_excel(file_path)
            elif suffix == '.csv' and EXCEL_AVAILABLE:
                return await self._process_csv(file_path)
            else:
                logger.warning(f"⚠️ Unsupported or missing library for: {file_path.suffix}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return []
    
    async def _process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process PDF file"""
        documents = []
        
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text().strip()
                if text:
                    # Split page into chunks if too long
                    chunks = self._chunk_text(text)
                    for chunk_num, chunk in enumerate(chunks):
                        documents.append({
                            "id": f"{file_path.stem}_page_{page_num}_chunk_{chunk_num}",
                            "text": chunk,
                            "metadata": {
                                "source": str(file_path),
                                "page": page_num + 1,
                                "type": "pdf_chunk",
                                "filename": file_path.name
                            }
                        })
        
        return documents
    
    async def _process_excel(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process Excel file"""
        documents = []
        
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Process each row
            for index, row in df.iterrows():
                # Create content from all columns
                content_parts = []
                for col, val in row.items():
                    if pd.notna(val):
                        content_parts.append(f"{col}: {val}")
                
                if content_parts:
                    content = "; ".join(content_parts)
                    documents.append({
                        "id": f"{file_path.stem}_{sheet_name}_row_{index}",
                        "text": content,
                        "metadata": {
                            "source": str(file_path),
                            "sheet": sheet_name,
                            "row": index + 1,
                            "type": "excel_row",
                            "filename": file_path.name
                        }
                    })
        
        return documents
    
    async def _process_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process CSV file"""
        documents = []
        
        df = pd.read_csv(file_path)
        
        for index, row in df.iterrows():
            # Create content from all columns
            content_parts = []
            for col, val in row.items():
                if pd.notna(val):
                    content_parts.append(f"{col}: {val}")
            
            if content_parts:
                content = "; ".join(content_parts)
                documents.append({
                    "id": f"{file_path.stem}_row_{index}",
                    "text": content,
                    "metadata": {
                        "source": str(file_path),
                        "row": index + 1,
                        "type": "csv_row",
                        "filename": file_path.name
                    }
                })
        
        return documents
    
    async def _process_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process JSON file"""
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            for key, value in data.items():
                documents.append({
                    "id": f"{file_path.stem}_{key}",
                    "text": str(value),
                    "metadata": {
                        "source": str(file_path),
                        "category": key,
                        "type": "json_entry",
                        "filename": file_path.name
                    }
                })
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    content = item.get('content') or item.get('text') or str(item)
                    title = item.get('title') or item.get('name') or f"item_{i}"
                else:
                    content = str(item)
                    title = f"item_{i}"
                
                documents.append({
                    "id": f"{file_path.stem}_{i}",
                    "text": content,
                    "metadata": {
                        "source": str(file_path),
                        "category": title,
                        "type": "json_list_item",
                        "filename": file_path.name
                    }
                })
        
        return documents
    
    async def _process_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process text/markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            return []
        
        chunks = self._chunk_text(content)
        documents = []
        
        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{file_path.stem}_chunk_{i}",
                "text": chunk,
                "metadata": {
                    "source": str(file_path),
                    "category": file_path.stem,
                    "type": "text_chunk",
                    "filename": file_path.name
                }
            })
        
        return documents
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text for optimal telephony performance"""
        if len(text) <= config.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + config.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > start + config.chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - config.chunk_overlap
        
        return [c for c in chunks if c.strip()]

async def main():
    """Main ingestion function"""
    parser = argparse.ArgumentParser(description="Ingest PDF, Excel, and other files into Qdrant")
    parser.add_argument("--directory", type=str, default="data", help="Directory to process")
    parser.add_argument("--recursive", action="store_true", help="Process directory recursively")
    parser.add_argument("--file", type=str, help="Process single file")
    
    args = parser.parse_args()
    
    # Check dependencies
    missing_deps = []
    if not PDF_AVAILABLE:
        missing_deps.append("PyPDF2 (for PDF files)")
    if not EXCEL_AVAILABLE:
        missing_deps.append("pandas (for Excel/CSV files)")
    
    if missing_deps:
        logger.warning(f"⚠️ Missing optional dependencies: {', '.join(missing_deps)}")
        logger.info("Install with: pip install PyPDF2 pandas")
    
    ingestion = EnhancedQdrantIngestion()
    
    if args.file:
        # Process single file
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            exit(1)
        
        logger.info(f"📄 Processing single file: {file_path}")
        if not await qdrant_rag.initialize():
            logger.error("❌ Failed to initialize Qdrant")
            exit(1)
        
        documents = await ingestion._process_file(file_path)
        if documents:
            success = await qdrant_rag.add_documents(documents)
            if success:
                logger.info(f"✅ Successfully processed {file_path.name}")
            else:
                logger.error("❌ Failed to add documents")
        await qdrant_rag.close()
    else:
        # Process directory
        directory = Path(args.directory)
        success = await ingestion.ingest_directory(directory, args.recursive)
        
        if success:
            logger.info("🎉 Data ingestion completed successfully!")
        else:
            logger.error("❌ Data ingestion failed!")
            exit(1)

if __name__ == "__main__":
    asyncio.run(main())