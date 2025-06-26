# excel_ingest_improved.py
"""
Improved Excel ingestion for simplified RAG system
Uses LlamaIndex Document format directly
"""
import asyncio
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import argparse

from llama_index.core import Document
from simple_rag_v2 import simplified_rag
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedExcelIngestor:
    """Simplified Excel ingestor using LlamaIndex Document format"""
    
    def __init__(self, excel_file_path: str):
        self.excel_file_path = Path(excel_file_path)
        self.documents = []
        
    def read_excel_file(self) -> Dict[str, pd.DataFrame]:
        """Read all sheets from Excel file"""
        try:
            logger.info(f"📖 Reading Excel file: {self.excel_file_path}")
            
            if not self.excel_file_path.exists():
                raise FileNotFoundError(f"Excel file not found: {self.excel_file_path}")
            
            # Read all sheets
            excel_data = pd.read_excel(
                self.excel_file_path,
                sheet_name=None,
                engine='openpyxl'
            )
            
            logger.info(f"✅ Successfully read {len(excel_data)} sheets")
            for sheet_name, df in excel_data.items():
                logger.info(f"   📋 {sheet_name}: {len(df)} rows, {len(df.columns)} columns")
            
            return excel_data
            
        except Exception as e:
            logger.error(f"❌ Failed to read Excel file: {e}")
            raise
    
    def process_services_sheet(self, df: pd.DataFrame) -> List[Document]:
        """Process Services sheet into LlamaIndex Documents"""
        documents = []
        
        logger.info("🔧 Processing Services sheet...")
        
        for idx, row in df.iterrows():
            try:
                # Skip empty rows
                if pd.isna(row.get('Service Type')) or pd.isna(row.get('Service Name')):
                    continue
                
                service_type = str(row['Service Type']).strip()
                service_name = str(row['Service Name']).strip()
                description = str(row['Description']).strip() if not pd.isna(row.get('Description')) else ""
                base_cost = str(row['Base Cost']).strip() if not pd.isna(row.get('Base Cost')) else ""
                additional_details = str(row['Additional Details']).strip() if not pd.isna(row.get('Additional Details')) else ""
                
                # Create comprehensive text content
                text_parts = []
                
                # Main service description
                text_parts.append(f"Service: {service_name}")
                text_parts.append(f"Category: {service_type}")
                
                if description:
                    text_parts.append(f"Description: {description}")
                
                if base_cost:
                    text_parts.append(f"Price: {base_cost}")
                
                if additional_details:
                    text_parts.append(f"Additional Information: {additional_details}")
                
                text_content = ". ".join(text_parts)
                
                # Create LlamaIndex Document
                doc = Document(
                    text=text_content,
                    metadata={
                        "sheet": "Services",
                        "service_type": service_type,
                        "service_name": service_name,
                        "base_cost": base_cost,
                        "document_type": "service_info",
                        "source": "excel_roadside_assistance"
                    },
                    doc_id=f"service_{idx}_{service_name.lower().replace(' ', '_')}"
                )
                
                documents.append(doc)
                logger.debug(f"   ✅ Processed: {service_name}")
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing row {idx}: {e}")
                continue
        
        logger.info(f"✅ Services sheet: {len(documents)} documents created")
        return documents
    
    def process_membership_sheet(self, df: pd.DataFrame) -> List[Document]:
        """Process Membership Plans sheet into LlamaIndex Documents"""
        documents = []
        
        logger.info("👥 Processing Membership Plans sheet...")
        
        for idx, row in df.iterrows():
            try:
                # Skip empty rows
                if pd.isna(row.get('Plan Name')):
                    continue
                
                plan_name = str(row['Plan Name']).strip()
                annual_cost = str(row['Annual Cost']).strip() if not pd.isna(row.get('Annual Cost')) else ""
                towing_distance = str(row['Towing Distance']).strip() if not pd.isna(row.get('Towing Distance')) else ""
                jump_starts = str(row['Jump-Starts']).strip() if not pd.isna(row.get('Jump-Starts')) else ""
                tire_changes = str(row['Tire Changes']).strip() if not pd.isna(row.get('Tire Changes')) else ""
                fuel_delivery = str(row['Fuel Delivery']).strip() if not pd.isna(row.get('Fuel Delivery')) else ""
                rental_discount = str(row['Rental Discount']).strip() if not pd.isna(row.get('Rental Discount')) else ""
                additional_benefits = str(row['Additional Benefits']).strip() if not pd.isna(row.get('Additional Benefits')) else ""
                
                # Create comprehensive membership plan content
                text_parts = []
                text_parts.append(f"Membership Plan: {plan_name}")
                
                if annual_cost:
                    text_parts.append(f"Annual Cost: {annual_cost}")
                
                if towing_distance:
                    text_parts.append(f"Towing Coverage: {towing_distance}")
                
                if jump_starts:
                    text_parts.append(f"Jump-Start Services: {jump_starts}")
                
                if tire_changes:
                    text_parts.append(f"Tire Changes: {tire_changes}")
                
                if fuel_delivery:
                    text_parts.append(f"Fuel Delivery: {fuel_delivery}")
                
                if rental_discount:
                    text_parts.append(f"Rental Discount: {rental_discount}")
                
                if additional_benefits:
                    text_parts.append(f"Additional Benefits: {additional_benefits}")
                
                text_content = ". ".join(text_parts)
                
                # Create LlamaIndex Document
                doc = Document(
                    text=text_content,
                    metadata={
                        "sheet": "Membership Plans",
                        "plan_name": plan_name,
                        "annual_cost": annual_cost,
                        "document_type": "membership_plan",
                        "source": "excel_roadside_assistance"
                    },
                    doc_id=f"membership_{idx}_{plan_name.lower().replace(' ', '_')}"
                )
                
                documents.append(doc)
                logger.debug(f"   ✅ Processed: {plan_name}")
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing row {idx}: {e}")
                continue
        
        logger.info(f"✅ Membership Plans sheet: {len(documents)} documents created")
        return documents
    
    def process_company_info_sheet(self, df: pd.DataFrame) -> List[Document]:
        """Process Company Info sheet into LlamaIndex Documents"""
        documents = []
        
        logger.info("🏢 Processing Company Info sheet...")
        
        for idx, row in df.iterrows():
            try:
                # Skip empty rows
                if pd.isna(row.get('Category')) or pd.isna(row.get('Detail')):
                    continue
                
                category = str(row['Category']).strip()
                detail = str(row['Detail']).strip()
                value = str(row['Value']).strip() if not pd.isna(row.get('Value')) else ""
                
                # Create searchable content
                text_content = f"{category} - {detail}: {value}"
                
                # Create LlamaIndex Document
                doc = Document(
                    text=text_content,
                    metadata={
                        "sheet": "Company Info",
                        "category": category,
                        "detail": detail,
                        "value": value,
                        "document_type": "company_info",
                        "source": "excel_roadside_assistance"
                    },
                    doc_id=f"company_{idx}_{category.lower().replace(' ', '_')}_{detail.lower().replace(' ', '_')}"
                )
                
                documents.append(doc)
                logger.debug(f"   ✅ Processed: {category} - {detail}")
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing row {idx}: {e}")
                continue
        
        logger.info(f"✅ Company Info sheet: {len(documents)} documents created")
        return documents
    
    def process_all_sheets(self) -> List[Document]:
        """Process all Excel sheets into LlamaIndex Documents"""
        try:
            excel_data = self.read_excel_file()
            all_documents = []
            
            # Process each sheet
            for sheet_name, df in excel_data.items():
                logger.info(f"\n📊 Processing sheet: {sheet_name}")
                
                if sheet_name.lower() in ['services', 'service']:
                    documents = self.process_services_sheet(df)
                elif sheet_name.lower() in ['membership plans', 'membership', 'plans']:
                    documents = self.process_membership_sheet(df)
                elif sheet_name.lower() in ['company info', 'company', 'info']:
                    documents = self.process_company_info_sheet(df)
                else:
                    logger.warning(f"⚠️ Unknown sheet type: {sheet_name}, skipping...")
                    continue
                
                all_documents.extend(documents)
            
            logger.info(f"\n✅ Total documents created: {len(all_documents)}")
            
            # Log document type distribution
            doc_types = {}
            for doc in all_documents:
                doc_type = doc.metadata.get("document_type", "unknown")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            logger.info("📊 Document distribution:")
            for doc_type, count in doc_types.items():
                logger.info(f"   {doc_type}: {count}")
            
            self.documents = all_documents
            return all_documents
            
        except Exception as e:
            logger.error(f"❌ Failed to process Excel sheets: {e}")
            raise
    
    async def ingest_to_rag(self) -> bool:
        """Ingest processed documents to simplified RAG system"""
        try:
            if not self.documents:
                logger.error("❌ No documents to ingest. Run process_all_sheets() first.")
                return False
            
            logger.info(f"🚀 Starting RAG ingestion of {len(self.documents)} documents...")
            
            # Initialize simplified RAG system
            logger.info("🔧 Initializing simplified RAG system...")
            success = await simplified_rag.initialize()
            
            if not success:
                logger.error("❌ Failed to initialize RAG system")
                return False
            
            # Convert to the format expected by our simplified system
            doc_dicts = []
            for doc in self.documents:
                doc_dict = {
                    "id": doc.doc_id,
                    "text": doc.text,
                    "metadata": doc.metadata
                }
                doc_dicts.append(doc_dict)
            
            # Ingest documents
            logger.info("📤 Adding documents to RAG system...")
            start_time = time.time()
            
            success = await simplified_rag.add_documents(doc_dicts)
            
            if success:
                ingest_time = time.time() - start_time
                logger.info(f"✅ Ingestion completed in {ingest_time:.2f} seconds")
                
                # Verify ingestion
                status = await simplified_rag.get_status()
                logger.info(f"📊 RAG system status: {status}")
                
                logger.info("🎉 Excel knowledge base successfully ingested!")
                return True
            else:
                logger.error("❌ Document ingestion failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ingestion error: {e}")
            return False
    
    async def test_search(self, test_queries: List[str] = None) -> None:
        """Test search functionality with sample queries"""
        if test_queries is None:
            test_queries = [
                "towing cost",
                "membership plans", 
                "business hours",
                "battery jumpstart",
                "tire change service",
                "fuel delivery"
            ]
        
        logger.info("🔍 Testing search functionality...")
        
        for query in test_queries:
            try:
                logger.info(f"\n🔍 Testing query: '{query}'")
                context = await simplified_rag.retrieve_context(query, max_results=2)
                
                if context:
                    logger.info(f"   ✅ Found context: {context[:100]}...")
                else:
                    logger.warning(f"   ⚠️ No context found for: {query}")
            except Exception as e:
                logger.error(f"   ❌ Search error: {e}")

async def main():
    """Main ingestion function"""
    parser = argparse.ArgumentParser(description="Ingest Excel knowledge base into simplified RAG")
    parser.add_argument("--file", "-f", required=True, help="Path to Excel file")
    parser.add_argument("--test", "-t", action="store_true", help="Run search tests after ingestion")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("🚀 Improved Excel Knowledge Base Ingestion Started")
        logger.info(f"📁 Excel file: {args.file}")
        logger.info(f"🎯 Target: Simplified RAG System")
        logger.info(f"🔗 Qdrant URL: {config.qdrant_url}")
        
        # Create ingestor
        ingestor = ImprovedExcelIngestor(args.file)
        
        # Process Excel file
        logger.info("\n📊 Step 1: Processing Excel file...")
        documents = ingestor.process_all_sheets()
        
        # Ingest to simplified RAG
        logger.info("\n🚀 Step 2: Ingesting to simplified RAG...")
        success = await ingestor.ingest_to_rag()
        
        if success:
            logger.info("\n✅ INGESTION COMPLETED SUCCESSFULLY!")
            
            # Run tests if requested
            if args.test:
                logger.info("\n🧪 Step 3: Running search tests...")
                await ingestor.test_search()
            
            logger.info("\n🎯 Your voice agent is now ready with simplified RAG system!")
            logger.info("💡 Run your agent with: python main_improved.py dev")
            
        else:
            logger.error("\n❌ INGESTION FAILED!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Ingestion cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    import time
    asyncio.run(main())