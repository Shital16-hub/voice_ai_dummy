# excel_ingest.py
"""
Excel Knowledge Base Ingestion Script for LiveKit Voice Agent
Ingests roadside assistance Excel data into Qdrant vector database
"""
import asyncio
import logging
import sys
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import argparse

# Import your existing RAG system
from simple_rag_system import simple_rag
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExcelIngestor:
    """Excel to Qdrant ingestion system"""
    
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
                sheet_name=None,  # Read all sheets
                engine='openpyxl'
            )
            
            logger.info(f"✅ Successfully read {len(excel_data)} sheets")
            for sheet_name, df in excel_data.items():
                logger.info(f"   📋 {sheet_name}: {len(df)} rows, {len(df.columns)} columns")
            
            return excel_data
            
        except Exception as e:
            logger.error(f"❌ Failed to read Excel file: {e}")
            raise
    
    def process_services_sheet(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process Services sheet into documents"""
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
                
                # Create comprehensive text content for better search
                content_parts = [
                    f"Service: {service_name}",
                    f"Type: {service_type}",
                    f"Description: {description}",
                    f"Cost: {base_cost}",
                    f"Details: {additional_details}"
                ]
                
                # Filter out empty parts
                content_parts = [part for part in content_parts if not part.endswith(": ")]
                content = ". ".join(content_parts)
                
                # Create multiple searchable variations for better RAG performance
                variations = [
                    # Main service document
                    {
                        "text": content,
                        "metadata": {
                            "sheet": "Services",
                            "service_type": service_type,
                            "service_name": service_name,
                            "base_cost": base_cost,
                            "document_type": "service_info"
                        }
                    },
                    
                    # Pricing-focused document
                    {
                        "text": f"{service_name} pricing: {base_cost}. {additional_details}",
                        "metadata": {
                            "sheet": "Services", 
                            "service_type": service_type,
                            "service_name": service_name,
                            "base_cost": base_cost,
                            "document_type": "pricing_info"
                        }
                    },
                    
                    # Service type focused document
                    {
                        "text": f"{service_type} service options: {service_name}. {description}. Starting at {base_cost}",
                        "metadata": {
                            "sheet": "Services",
                            "service_type": service_type,
                            "service_name": service_name,
                            "base_cost": base_cost,
                            "document_type": "service_category"
                        }
                    }
                ]
                
                documents.extend(variations)
                logger.debug(f"   ✅ Processed: {service_name}")
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing row {idx}: {e}")
                continue
        
        logger.info(f"✅ Services sheet: {len(documents)} documents created from {len(df)} rows")
        return documents
    
    def process_membership_sheet(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process Membership Plans sheet into documents"""
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
                content_parts = [
                    f"Membership Plan: {plan_name}",
                    f"Annual Cost: {annual_cost}",
                    f"Towing Coverage: {towing_distance}",
                    f"Jump-Start Services: {jump_starts}",
                    f"Tire Changes: {tire_changes}",
                    f"Fuel Delivery: {fuel_delivery}",
                    f"Rental Discount: {rental_discount}",
                    f"Additional Benefits: {additional_benefits}"
                ]
                
                # Filter out empty parts
                content_parts = [part for part in content_parts if not part.endswith(": ")]
                content = ". ".join(content_parts)
                
                # Create searchable variations
                variations = [
                    # Main membership document
                    {
                        "text": content,
                        "metadata": {
                            "sheet": "Membership Plans",
                            "plan_name": plan_name,
                            "annual_cost": annual_cost,
                            "document_type": "membership_plan"
                        }
                    },
                    
                    # Pricing-focused document
                    {
                        "text": f"{plan_name} membership costs {annual_cost} per year. Includes {towing_distance} towing, {jump_starts} jump-starts, {tire_changes} tire changes.",
                        "metadata": {
                            "sheet": "Membership Plans",
                            "plan_name": plan_name,
                            "annual_cost": annual_cost,
                            "document_type": "membership_pricing"
                        }
                    },
                    
                    # Benefits-focused document
                    {
                        "text": f"{plan_name} plan benefits: {additional_benefits}. Rental discount: {rental_discount}",
                        "metadata": {
                            "sheet": "Membership Plans",
                            "plan_name": plan_name,
                            "annual_cost": annual_cost,
                            "document_type": "membership_benefits"
                        }
                    }
                ]
                
                documents.extend(variations)
                logger.debug(f"   ✅ Processed: {plan_name}")
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing row {idx}: {e}")
                continue
        
        logger.info(f"✅ Membership Plans sheet: {len(documents)} documents created from {len(df)} rows")
        return documents
    
    def process_company_info_sheet(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process Company Info sheet into documents"""
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
                content = f"{category} {detail}: {value}"
                
                # Create variations for better search
                variations = [
                    # Main info document
                    {
                        "text": content,
                        "metadata": {
                            "sheet": "Company Info",
                            "category": category,
                            "detail": detail,
                            "value": value,
                            "document_type": "company_info"
                        }
                    }
                ]
                
                # Add specific search-friendly variations for important info
                if "hours" in detail.lower():
                    variations.append({
                        "text": f"Service hours: {value}. We are available {value}",
                        "metadata": {
                            "sheet": "Company Info",
                            "category": category,
                            "detail": detail,
                            "value": value,
                            "document_type": "business_hours"
                        }
                    })
                
                if "phone" in detail.lower() or "contact" in detail.lower():
                    variations.append({
                        "text": f"Contact information: {detail} {value}",
                        "metadata": {
                            "sheet": "Company Info",
                            "category": category,
                            "detail": detail,
                            "value": value,
                            "document_type": "contact_info"
                        }
                    })
                
                documents.extend(variations)
                logger.debug(f"   ✅ Processed: {category} - {detail}")
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing row {idx}: {e}")
                continue
        
        logger.info(f"✅ Company Info sheet: {len(documents)} documents created from {len(df)} rows")
        return documents
    
    def generate_document_id(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate unique document ID"""
        # Create ID from text and key metadata
        content_for_id = f"{text}_{metadata.get('sheet', '')}_{metadata.get('document_type', '')}"
        return hashlib.md5(content_for_id.encode()).hexdigest()
    
    def process_all_sheets(self) -> List[Dict[str, Any]]:
        """Process all Excel sheets into documents"""
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
            
            # Add unique IDs to all documents
            for doc in all_documents:
                doc["id"] = self.generate_document_id(doc["text"], doc["metadata"])
            
            logger.info(f"\n✅ Total documents created: {len(all_documents)}")
            
            # Log document type distribution
            doc_types = {}
            for doc in all_documents:
                doc_type = doc["metadata"].get("document_type", "unknown")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            logger.info("📊 Document distribution:")
            for doc_type, count in doc_types.items():
                logger.info(f"   {doc_type}: {count}")
            
            self.documents = all_documents
            return all_documents
            
        except Exception as e:
            logger.error(f"❌ Failed to process Excel sheets: {e}")
            raise
    
    async def ingest_to_qdrant(self) -> bool:
        """Ingest processed documents to Qdrant"""
        try:
            if not self.documents:
                logger.error("❌ No documents to ingest. Run process_all_sheets() first.")
                return False
            
            logger.info(f"🚀 Starting Qdrant ingestion of {len(self.documents)} documents...")
            
            # Initialize RAG system
            logger.info("🔧 Initializing Qdrant RAG system...")
            success = await simple_rag.initialize()
            
            if not success:
                logger.error("❌ Failed to initialize RAG system")
                return False
            
            # Get initial status
            initial_status = await simple_rag.get_status()
            initial_count = initial_status.get("points_count", 0)
            logger.info(f"📊 Current collection size: {initial_count} documents")
            
            # Ingest documents
            logger.info("📤 Adding documents to Qdrant...")
            start_time = time.time()
            
            success = await simple_rag.add_documents(self.documents)
            
            if success:
                ingest_time = time.time() - start_time
                logger.info(f"✅ Ingestion completed in {ingest_time:.2f} seconds")
                
                # Verify ingestion
                final_status = await simple_rag.get_status()
                final_count = final_status.get("points_count", 0)
                new_documents = final_count - initial_count
                
                logger.info(f"📊 Verification:")
                logger.info(f"   Initial documents: {initial_count}")
                logger.info(f"   Final documents: {final_count}")
                logger.info(f"   Added documents: {new_documents}")
                
                if new_documents > 0:
                    logger.info("🎉 Excel knowledge base successfully ingested!")
                    return True
                else:
                    logger.warning("⚠️ No new documents were added")
                    return False
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
                results = await simple_rag.search(query, limit=2)
                
                if results:
                    for i, result in enumerate(results):
                        score = result.get("score", 0)
                        text = result.get("text", "")[:100] + "..." if len(result.get("text", "")) > 100 else result.get("text", "")
                        logger.info(f"   Result {i+1} (score: {score:.3f}): {text}")
                else:
                    logger.warning(f"   ⚠️ No results found for: {query}")
            except Exception as e:
                logger.error(f"   ❌ Search error: {e}")

async def main():
    """Main ingestion function"""
    parser = argparse.ArgumentParser(description="Ingest Excel knowledge base into Qdrant")
    parser.add_argument("--file", "-f", required=True, help="Path to Excel file")
    parser.add_argument("--test", "-t", action="store_true", help="Run search tests after ingestion")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("🚀 Excel Knowledge Base Ingestion Started")
        logger.info(f"📁 Excel file: {args.file}")
        logger.info(f"🎯 Target collection: {config.qdrant_collection_name}")
        logger.info(f"🔗 Qdrant URL: {config.qdrant_url}")
        
        # Create ingestor
        ingestor = ExcelIngestor(args.file)
        
        # Process Excel file
        logger.info("\n📊 Step 1: Processing Excel file...")
        documents = ingestor.process_all_sheets()
        
        # Ingest to Qdrant
        logger.info("\n🚀 Step 2: Ingesting to Qdrant...")
        success = await ingestor.ingest_to_qdrant()
        
        if success:
            logger.info("\n✅ INGESTION COMPLETED SUCCESSFULLY!")
            
            # Run tests if requested
            if args.test:
                logger.info("\n🧪 Step 3: Running search tests...")
                await ingestor.test_search()
            
            logger.info("\n🎯 Your voice agent is now ready with Excel knowledge base!")
            logger.info("💡 Run your agent with: python main.py dev")
            
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
    asyncio.run(main())