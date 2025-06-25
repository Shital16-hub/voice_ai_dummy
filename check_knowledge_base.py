# check_knowledge_base.py - VERIFY EXCEL DATA IS PROPERLY INDEXED
"""
Knowledge Base Verification Tool
Ensures your Excel data is properly indexed and searchable

Usage:
    python check_knowledge_base.py --test-queries
    python check_knowledge_base.py --show-data
    python check_knowledge_base.py --search "towing prices"
"""
import asyncio
import argparse
import logging
from typing import List, Dict, Any
import json

from simple_rag_system import simple_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBaseChecker:
    """Verify that Excel data is properly indexed and searchable"""
    
    def __init__(self):
        self.test_queries = [
            "towing service pricing",
            "battery jump start cost",
            "tire change rates", 
            "emergency service fees",
            "business hours",
            "contact information",
            "service area coverage",
            "payment methods",
            "membership benefits",
            "warranty information"
        ]
    
    async def check_system_status(self) -> bool:
        """Check if the RAG system is working"""
        logger.info("🔍 Checking RAG system status...")
        
        try:
            # Initialize system
            success = await simple_rag.initialize()
            if not success:
                logger.error("❌ RAG system failed to initialize")
                return False
            
            # Get status
            status = await simple_rag.get_status()
            logger.info(f"📊 System Status: {status}")
            
            # Check if we have data
            points_count = status.get("points_count", 0)
            if points_count == 0:
                logger.error("❌ No documents found in knowledge base!")
                logger.info("💡 Run: python ingest_excel_data.py --file your_excel_file.xlsx")
                return False
            
            logger.info(f"✅ Knowledge base has {points_count} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ System check failed: {e}")
            return False
    
    async def test_search_queries(self) -> Dict[str, Any]:
        """Test common search queries"""
        logger.info("🧪 Testing search queries...")
        
        results = {
            "total_queries": len(self.test_queries),
            "successful_queries": 0,
            "failed_queries": 0,
            "query_results": {}
        }
        
        for query in self.test_queries:
            try:
                search_results = await simple_rag.search(query, limit=2)
                
                if search_results:
                    results["successful_queries"] += 1
                    results["query_results"][query] = {
                        "status": "success",
                        "results_count": len(search_results),
                        "best_score": search_results[0].get("score", 0),
                        "preview": search_results[0].get("text", "")[:100] + "..."
                    }
                    logger.info(f"   ✅ '{query}' -> {len(search_results)} results (score: {search_results[0].get('score', 0):.3f})")
                else:
                    results["failed_queries"] += 1
                    results["query_results"][query] = {
                        "status": "no_results", 
                        "results_count": 0
                    }
                    logger.warning(f"   ⚠️ '{query}' -> No results")
                    
            except Exception as e:
                results["failed_queries"] += 1
                results["query_results"][query] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"   ❌ '{query}' -> Error: {e}")
        
        success_rate = (results["successful_queries"] / results["total_queries"]) * 100
        logger.info(f"📊 Search Success Rate: {success_rate:.1f}% ({results['successful_queries']}/{results['total_queries']})")
        
        return results
    
    async def show_sample_data(self, limit: int = 10):
        """Show sample documents from the knowledge base"""
        logger.info(f"📄 Showing sample data (limit: {limit})...")
        
        try:
            # Search with a broad query to get sample results
            sample_queries = ["service", "price", "cost", "available", "contact"]
            
            all_samples = []
            for query in sample_queries:
                results = await simple_rag.search(query, limit=2)
                all_samples.extend(results)
                if len(all_samples) >= limit:
                    break
            
            # Remove duplicates and limit
            seen_ids = set()
            unique_samples = []
            for sample in all_samples:
                if sample["id"] not in seen_ids:
                    unique_samples.append(sample)
                    seen_ids.add(sample["id"])
                if len(unique_samples) >= limit:
                    break
            
            if unique_samples:
                logger.info(f"📋 Found {len(unique_samples)} sample documents:")
                for i, doc in enumerate(unique_samples, 1):
                    metadata = doc.get("metadata", {})
                    source = metadata.get("source", "unknown")
                    doc_type = metadata.get("type", "unknown")
                    
                    logger.info(f"   {i}. ID: {doc['id']}")
                    logger.info(f"      Source: {source}")
                    logger.info(f"      Type: {doc_type}")
                    logger.info(f"      Score: {doc.get('score', 0):.3f}")
                    logger.info(f"      Text: {doc['text'][:150]}...")
                    logger.info("")
            else:
                logger.warning("⚠️ No sample documents found")
                
        except Exception as e:
            logger.error(f"❌ Error showing sample data: {e}")
    
    async def search_specific_query(self, query: str, limit: int = 3):
        """Search for a specific query and show detailed results"""
        logger.info(f"🔍 Searching for: '{query}'")
        
        try:
            results = await simple_rag.search(query, limit=limit)
            
            if results:
                logger.info(f"✅ Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    metadata = result.get("metadata", {})
                    logger.info(f"   Result {i}:")
                    logger.info(f"     Score: {result.get('score', 0):.3f}")
                    logger.info(f"     Source: {metadata.get('source', 'unknown')}")
                    logger.info(f"     Type: {metadata.get('type', 'unknown')}")
                    logger.info(f"     Text: {result['text']}")
                    logger.info("")
            else:
                logger.warning(f"⚠️ No results found for: '{query}'")
                logger.info("💡 This means:")
                logger.info("   - Your Excel file may not contain relevant data for this query")
                logger.info("   - The data may not be indexed properly")
                logger.info("   - Try checking your Excel file content")
                
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
    
    async def analyze_data_coverage(self):
        """Analyze what types of data are in the knowledge base"""
        logger.info("📊 Analyzing data coverage...")
        
        try:
            # Search for different types of content
            analysis_queries = {
                "pricing": ["price", "cost", "rate", "fee", "charge", "$"],
                "services": ["service", "towing", "battery", "tire", "jump", "repair"],
                "contact": ["phone", "address", "contact", "location", "hours"],
                "policy": ["policy", "coverage", "warranty", "terms", "conditions"],
                "emergency": ["emergency", "urgent", "24/7", "available", "dispatch"]
            }
            
            coverage_results = {}
            
            for category, keywords in analysis_queries.items():
                category_results = []
                for keyword in keywords:
                    results = await simple_rag.search(keyword, limit=1)
                    if results:
                        category_results.extend(results)
                
                # Remove duplicates
                unique_results = []
                seen_ids = set()
                for result in category_results:
                    if result["id"] not in seen_ids:
                        unique_results.append(result)
                        seen_ids.add(result["id"])
                
                coverage_results[category] = {
                    "documents_found": len(unique_results),
                    "keywords_with_results": len([kw for kw in keywords if any(kw.lower() in r["text"].lower() for r in unique_results)])
                }
                
                logger.info(f"   {category.title()}: {len(unique_results)} documents found")
            
            # Summary
            total_categories = len(analysis_queries)
            covered_categories = len([cat for cat, data in coverage_results.items() if data["documents_found"] > 0])
            
            logger.info(f"📈 Coverage Summary: {covered_categories}/{total_categories} categories have data")
            
            if covered_categories == 0:
                logger.warning("⚠️ No data found for any category!")
                logger.info("💡 Possible issues:")
                logger.info("   - Excel file not ingested properly")
                logger.info("   - Excel file doesn't contain expected data")
                logger.info("   - Column names/structure not recognized")
            
            return coverage_results
            
        except Exception as e:
            logger.error(f"❌ Data coverage analysis failed: {e}")
            return {}
    
    def generate_report(self, test_results: Dict[str, Any], coverage_results: Dict[str, Any]):
        """Generate a comprehensive report"""
        logger.info("📋 KNOWLEDGE BASE REPORT")
        logger.info("=" * 60)
        
        # Test Results Summary
        success_rate = (test_results["successful_queries"] / test_results["total_queries"]) * 100
        logger.info(f"🧪 Query Test Results:")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        logger.info(f"   Successful: {test_results['successful_queries']}")
        logger.info(f"   Failed: {test_results['failed_queries']}")
        logger.info("")
        
        # Coverage Summary
        if coverage_results:
            logger.info("📊 Data Coverage:")
            for category, data in coverage_results.items():
                status = "✅" if data["documents_found"] > 0 else "❌"
                logger.info(f"   {status} {category.title()}: {data['documents_found']} documents")
            logger.info("")
        
        # Recommendations
        logger.info("💡 Recommendations:")
        if success_rate < 50:
            logger.info("   ⚠️ Low success rate - check Excel data quality")
            logger.info("   ⚠️ Consider re-ingesting Excel files")
        elif success_rate < 80:
            logger.info("   📈 Moderate success rate - some queries may need adjustment")
        else:
            logger.info("   ✅ Good success rate - knowledge base is working well")
        
        if not any(data["documents_found"] > 0 for data in coverage_results.values()):
            logger.info("   ❌ No data found - run Excel ingestion script")
        
        logger.info("")
        logger.info("🔧 Next Steps:")
        logger.info("   1. If results are poor, check your Excel file content")
        logger.info("   2. Re-run ingestion: python ingest_excel_data.py --file your_file.xlsx")
        logger.info("   3. Test specific queries: python check_knowledge_base.py --search 'your query'")

async def main():
    """Main checker function"""
    parser = argparse.ArgumentParser(description="Check knowledge base status")
    parser.add_argument("--test-queries", action="store_true", help="Test common queries")
    parser.add_argument("--show-data", action="store_true", help="Show sample data")
    parser.add_argument("--search", type=str, help="Search for specific query")
    parser.add_argument("--analyze", action="store_true", help="Analyze data coverage")
    parser.add_argument("--full-report", action="store_true", help="Generate full report")
    parser.add_argument("--limit", type=int, default=10, help="Limit for sample data")
    
    args = parser.parse_args()
    
    checker = KnowledgeBaseChecker()
    
    try:
        # Always check system status first
        if not await checker.check_system_status():
            logger.error("❌ System check failed - cannot proceed")
            return
        
        test_results = {}
        coverage_results = {}
        
        if args.search:
            # Search specific query
            await checker.search_specific_query(args.search)
            
        elif args.show_data:
            # Show sample data
            await checker.show_sample_data(args.limit)
            
        elif args.test_queries:
            # Test queries
            test_results = await checker.test_search_queries()
            
        elif args.analyze:
            # Analyze coverage
            coverage_results = await checker.analyze_data_coverage()
            
        elif args.full_report:
            # Generate full report
            test_results = await checker.test_search_queries()
            coverage_results = await checker.analyze_data_coverage()
            checker.generate_report(test_results, coverage_results)
            
        else:
            # Default: run basic checks
            logger.info("🔧 Running basic checks...")
            await checker.show_sample_data(5)
            test_results = await checker.test_search_queries()
            if test_results["successful_queries"] < test_results["total_queries"] / 2:
                logger.warning("⚠️ Consider running full analysis: python check_knowledge_base.py --full-report")
        
        # Close system
        await simple_rag.close()
        
    except Exception as e:
        logger.error(f"❌ Checker error: {e}")

if __name__ == "__main__":
    asyncio.run(main())