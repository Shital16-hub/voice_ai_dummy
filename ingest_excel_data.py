# ingest_excel_data.py - OPTIMIZED FOR EXCEL FILES
"""
Excel Data Ingestion for RAG System
Optimized for roadside assistance Excel spreadsheets

Usage:
    python ingest_excel_data.py --file data/roadside_services.xlsx
    python ingest_excel_data.py --directory data
"""
import asyncio
import argparse
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional  # Added Optional here
import uuid

from simple_rag_system import simple_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExcelRAGIngestion:
    """Optimized Excel ingestion for roadside assistance data"""
    
    def __init__(self):
        self.processed_documents = []
        
    async def ingest_excel_file(self, file_path: Path) -> bool:
        """Ingest a single Excel file optimized for roadside assistance"""
        try:
            logger.info(f"📄 Processing Excel file: {file_path.name}")
            
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            documents = []
            
            for sheet_name in excel_file.sheet_names:
                logger.info(f"   📋 Processing sheet: {sheet_name}")
                
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheet_docs = self._process_sheet(df, sheet_name, file_path.name)
                documents.extend(sheet_docs)
                
                logger.info(f"   ✅ Extracted {len(sheet_docs)} entries from {sheet_name}")
            
            if documents:
                # Initialize RAG system first
                logger.info("🔧 Initializing RAG system...")
                init_success = await simple_rag.initialize()
                
                # Check if system is usable (even if not fully ready)
                if not init_success and not simple_rag.client:
                    logger.error("❌ Failed to initialize RAG system - no client available")
                    return False
                
                if init_success:
                    logger.info("✅ RAG system fully ready")
                else:
                    logger.info("⚠️ RAG system partially ready - proceeding with ingestion")
                
                logger.info(f"📤 Adding {len(documents)} documents to knowledge base...")
                
                # Add documents with better error handling
                try:
                    success = await simple_rag.add_documents(documents)
                    if success:
                        logger.info(f"🎉 Successfully ingested {len(documents)} documents from {file_path.name}")
                        self.processed_documents.extend(documents)
                        
                        # Verify documents were added
                        status = await simple_rag.get_status()
                        points_count = status.get("points_count", 0)
                        logger.info(f"✅ Verification: Knowledge base now has {points_count} documents")
                        
                        return True
                    else:
                        logger.error("❌ Failed to add documents to RAG system")
                        return False
                except Exception as e:
                    logger.error(f"❌ Error adding documents: {e}")
                    return False
            else:
                logger.warning("⚠️ No valid documents extracted from Excel file")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return False
    
    def _process_sheet(self, df: pd.DataFrame, sheet_name: str, filename: str) -> List[Dict[str, Any]]:
        """Process a single Excel sheet with roadside assistance optimization"""
        documents = []
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Auto-detect content structure
        content_strategy = self._detect_content_strategy(df)
        logger.info(f"   🔍 Detected strategy: {content_strategy}")
        
        for index, row in df.iterrows():
            try:
                doc = self._create_document_from_row(
                    row, index, sheet_name, filename, content_strategy
                )
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"   ⚠️ Error processing row {index}: {e}")
                continue
        
        return documents
    
    def _detect_content_strategy(self, df: pd.DataFrame) -> str:
        """Detect the best strategy for processing this Excel sheet"""
        columns = [str(col).lower() for col in df.columns]
        
        # Strategy 1: Q&A format
        if any('question' in col for col in columns) and any('answer' in col for col in columns):
            return "qa_format"
        
        # Strategy 2: Service catalog
        if any(word in ' '.join(columns) for word in ['service', 'price', 'rate', 'cost']):
            return "service_catalog"
        
        # Strategy 3: Policy/procedure
        if any(word in ' '.join(columns) for word in ['policy', 'procedure', 'rule', 'guideline']):
            return "policy_format"
        
        # Strategy 4: Contact/location info
        if any(word in ' '.join(columns) for word in ['phone', 'address', 'location', 'contact']):
            return "contact_info"
        
        # Default: row-based
        return "row_based"
    
    def _create_document_from_row(
        self, 
        row: pd.Series, 
        index: int, 
        sheet_name: str, 
        filename: str, 
        strategy: str
    ) -> Optional[Dict[str, Any]]:  # Changed return type to Optional
        """Create a document from a row based on detected strategy"""
        
        if strategy == "qa_format":
            return self._create_qa_document(row, index, sheet_name, filename)
        elif strategy == "service_catalog":
            return self._create_service_document(row, index, sheet_name, filename)
        elif strategy == "policy_format":
            return self._create_policy_document(row, index, sheet_name, filename)
        elif strategy == "contact_info":
            return self._create_contact_document(row, index, sheet_name, filename)
        else:
            return self._create_row_based_document(row, index, sheet_name, filename)
    
    def _create_qa_document(self, row: pd.Series, index: int, sheet_name: str, filename: str) -> Optional[Dict[str, Any]]:
        """Create document from Q&A format"""
        question_cols = [col for col in row.index if 'question' in str(col).lower()]
        answer_cols = [col for col in row.index if 'answer' in str(col).lower()]
        
        if not question_cols or not answer_cols:
            return None
        
        question = str(row[question_cols[0]]) if pd.notna(row[question_cols[0]]) else ""
        answer = str(row[answer_cols[0]]) if pd.notna(row[answer_cols[0]]) else ""
        
        if not question or not answer or question.lower() in ['nan', 'none']:
            return None
        
        # Combine Q&A for better searchability
        content = f"Q: {question} A: {answer}"
        
        return {
            "id": f"{filename}_{sheet_name}_qa_{index}",
            "text": content,
            "metadata": {
                "source": filename,
                "sheet": sheet_name,
                "type": "qa_pair",
                "question": question,
                "answer": answer,
                "row": index + 1
            }
        }
    
    def _create_service_document(self, row: pd.Series, index: int, sheet_name: str, filename: str) -> Dict[str, Any]:
        """Create document from service catalog format"""
        # Look for service name
        service_cols = [col for col in row.index if any(word in str(col).lower() 
                      for word in ['service', 'name', 'title', 'type'])]
        
        # Look for pricing
        price_cols = [col for col in row.index if any(word in str(col).lower() 
                     for word in ['price', 'cost', 'rate', 'fee', 'charge'])]
        
        # Look for description
        desc_cols = [col for col in row.index if any(word in str(col).lower() 
                    for word in ['description', 'detail', 'info', 'note'])]
        
        content_parts = []
        service_name = ""
        
        # Extract service name
        if service_cols:
            service_name = str(row[service_cols[0]]) if pd.notna(row[service_cols[0]]) else ""
            if service_name and service_name.lower() not in ['nan', 'none']:
                content_parts.append(f"Service: {service_name}")
        
        # Extract pricing
        for col in price_cols:
            value = row[col]
            if pd.notna(value) and str(value).lower() not in ['nan', 'none']:
                content_parts.append(f"{col}: {value}")
        
        # Extract description
        for col in desc_cols:
            value = row[col]
            if pd.notna(value) and str(value).lower() not in ['nan', 'none']:
                content_parts.append(f"{col}: {value}")
        
        # Add any remaining important columns
        for col, value in row.items():
            if (col not in service_cols + price_cols + desc_cols and 
                pd.notna(value) and str(value).lower() not in ['nan', 'none']):
                content_parts.append(f"{col}: {value}")
        
        if not content_parts:
            return None
        
        content = "; ".join(content_parts)
        
        return {
            "id": f"{filename}_{sheet_name}_service_{index}",
            "text": content,
            "metadata": {
                "source": filename,
                "sheet": sheet_name,
                "type": "service_info",
                "service_name": service_name,
                "row": index + 1
            }
        }
    
    def _create_policy_document(self, row: pd.Series, index: int, sheet_name: str, filename: str) -> Optional[Dict[str, Any]]:
        """Create document from policy/procedure format"""
        content_parts = []
        
        for col, value in row.items():
            if pd.notna(value) and str(value).lower() not in ['nan', 'none']:
                content_parts.append(f"{col}: {value}")
        
        if not content_parts:
            return None
        
        content = "; ".join(content_parts)
        
        return {
            "id": f"{filename}_{sheet_name}_policy_{index}",
            "text": content,
            "metadata": {
                "source": filename,
                "sheet": sheet_name,
                "type": "policy_info",
                "row": index + 1
            }
        }
    
    def _create_contact_document(self, row: pd.Series, index: int, sheet_name: str, filename: str) -> Optional[Dict[str, Any]]:
        """Create document from contact/location format"""
        content_parts = []
        
        for col, value in row.items():
            if pd.notna(value) and str(value).lower() not in ['nan', 'none']:
                content_parts.append(f"{col}: {value}")
        
        if not content_parts:
            return None
        
        content = "; ".join(content_parts)
        
        return {
            "id": f"{filename}_{sheet_name}_contact_{index}",
            "text": content,
            "metadata": {
                "source": filename,
                "sheet": sheet_name,
                "type": "contact_info",
                "row": index + 1
            }
        }
    
    def _create_row_based_document(self, row: pd.Series, index: int, sheet_name: str, filename: str) -> Optional[Dict[str, Any]]:
        """Create document from generic row data"""
        content_parts = []
        
        for col, value in row.items():
            if pd.notna(value) and str(value).lower() not in ['nan', 'none']:
                content_parts.append(f"{col}: {value}")
        
        if not content_parts:
            return None
        
        content = "; ".join(content_parts)
        
        return {
            "id": f"{filename}_{sheet_name}_row_{index}",
            "text": content,
            "metadata": {
                "source": filename,
                "sheet": sheet_name,
                "type": "row_data",
                "row": index + 1
            }
        }
    
    async def ingest_directory(self, directory: Path) -> bool:
        """Ingest all Excel files from a directory"""
        try:
            excel_files = list(directory.glob("*.xlsx")) + list(directory.glob("*.xls"))
            
            if not excel_files:
                logger.warning(f"⚠️ No Excel files found in {directory}")
                return False
            
            logger.info(f"📁 Found {len(excel_files)} Excel files")
            
            success_count = 0
            for file_path in excel_files:
                if await self.ingest_excel_file(file_path):
                    success_count += 1
            
            logger.info(f"🎉 Successfully processed {success_count}/{len(excel_files)} Excel files")
            logger.info(f"📊 Total documents ingested: {len(self.processed_documents)}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error processing directory: {e}")
            return False
    
    def print_summary(self):
        """Print ingestion summary"""
        if not self.processed_documents:
            logger.info("No documents were processed")
            return
        
        # Count by type
        type_counts = {}
        source_counts = {}
        
        for doc in self.processed_documents:
            doc_type = doc["metadata"].get("type", "unknown")
            source = doc["metadata"].get("source", "unknown")
            
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info("📊 INGESTION SUMMARY:")
        logger.info(f"   Total documents: {len(self.processed_documents)}")
        logger.info("   By type:")
        for doc_type, count in type_counts.items():
            logger.info(f"     {doc_type}: {count}")
        logger.info("   By source:")
        for source, count in source_counts.items():
            logger.info(f"     {source}: {count}")

async def main():
    """Main ingestion function"""
    parser = argparse.ArgumentParser(description="Ingest Excel files for RAG system")
    parser.add_argument("--file", type=str, help="Single Excel file to process")
    parser.add_argument("--directory", type=str, default="data", help="Directory containing Excel files")
    parser.add_argument("--test", action="store_true", help="Test RAG system after ingestion")
    
    args = parser.parse_args()
    
    ingestion = ExcelRAGIngestion()
    
    try:
        if args.file:
            # Process single file
            file_path = Path(args.file)
            if not file_path.exists():
                logger.error(f"❌ File not found: {file_path}")
                return
            
            success = await ingestion.ingest_excel_file(file_path)
            
        else:
            # Process directory
            directory = Path(args.directory)
            if not directory.exists():
                logger.error(f"❌ Directory not found: {directory}")
                return
            
            success = await ingestion.ingest_directory(directory)
        
        # Print summary
        ingestion.print_summary()
        
        # Test RAG system if requested
        if args.test and success:
            await test_rag_system()
        
        # Close RAG system
        await simple_rag.close()
        
        if success:
            logger.info("🎉 Excel ingestion completed successfully!")
        else:
            logger.error("❌ Excel ingestion failed!")
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")

async def test_rag_system():
    """Test the RAG system with common queries"""
    logger.info("🧪 Testing RAG system...")
    
    test_queries = [
        "towing prices",
        "battery service cost",
        "tire change rates",
        "emergency service",
        "business hours"
    ]
    
    for query in test_queries:
        try:
            results = await simple_rag.search(query, limit=1)
            if results:
                result = results[0]
                logger.info(f"   ✅ '{query}' -> {result['text'][:100]}...")
            else:
                logger.warning(f"   ⚠️ '{query}' -> No results")
        except Exception as e:
            logger.error(f"   ❌ '{query}' -> Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())