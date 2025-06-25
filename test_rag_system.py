# test_rag_system.py
"""
Comprehensive RAG System Test Script
Tests all aspects of the RAG system used in main.py
"""
import asyncio
import logging
import time
import json
from typing import List, Dict, Any
from pathlib import Path

# Import the same modules as main.py
from simple_rag_system import simple_rag
from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystemTester:
    """Comprehensive RAG system tester"""
    
    def __init__(self):
        self.test_results = {}
        
    async def run_all_tests(self):
        """Run all RAG system tests"""
        logger.info("🧪 STARTING COMPREHENSIVE RAG SYSTEM TESTS")
        logger.info("=" * 60)
        
        # Test 1: Basic Initialization
        await self.test_initialization()
        
        # Test 2: Collection Status
        await self.test_collection_status()
        
        # Test 3: Basic Search Functionality
        await self.test_basic_search()
        
        # Test 4: Real User Queries (from your log)
        await self.test_real_user_queries()
        
        # Test 5: Different Query Types
        await self.test_query_types()
        
        # Test 6: Score Analysis
        await self.test_score_analysis()
        
        # Test 7: Cache Testing
        await self.test_cache_functionality()
        
        # Test 8: Performance Testing
        await self.test_performance()
        
        # Generate Report
        self.generate_test_report()
        
    async def test_initialization(self):
        """Test RAG system initialization"""
        logger.info("\n🔧 TEST 1: RAG System Initialization")
        logger.info("-" * 40)
        
        try:
            start_time = time.time()
            success = await simple_rag.initialize()
            init_time = (time.time() - start_time) * 1000
            
            self.test_results["initialization"] = {
                "success": success,
                "time_ms": init_time,
                "ready": simple_rag.ready
            }
            
            logger.info(f"✅ Initialization: {'SUCCESS' if success else 'FAILED'}")
            logger.info(f"   Time: {init_time:.1f}ms")
            logger.info(f"   Ready: {simple_rag.ready}")
            
            if not success:
                logger.error("❌ CRITICAL: RAG system failed to initialize!")
                logger.error("💡 Check: docker-compose up -d")
                logger.error("💡 Check: OpenAI API key in .env")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            self.test_results["initialization"] = {"success": False, "error": str(e)}
            return False
    
    async def test_collection_status(self):
        """Test collection status and document count"""
        logger.info("\n📊 TEST 2: Collection Status")
        logger.info("-" * 40)
        
        try:
            status = await simple_rag.get_status()
            
            self.test_results["collection_status"] = status
            
            logger.info(f"Status: {status.get('status', 'unknown')}")
            logger.info(f"Documents: {status.get('points_count', 0)}")
            logger.info(f"Cache size: {status.get('cache_size', 0)}")
            
            points_count = status.get("points_count", 0)
            
            if points_count == 0:
                logger.error("❌ CRITICAL: No documents in knowledge base!")
                logger.error("💡 Run: python excel_ingest.py --file data/Roadside_Assist_Info.xlsx")
                return False
            elif points_count < 50:
                logger.warning(f"⚠️ Low document count: {points_count}")
                logger.warning("💡 Consider re-ingesting your Excel file")
            else:
                logger.info(f"✅ Good document count: {points_count}")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Status check error: {e}")
            self.test_results["collection_status"] = {"error": str(e)}
            return False
    
    async def test_basic_search(self):
        """Test basic search functionality"""
        logger.info("\n🔍 TEST 3: Basic Search Functionality")
        logger.info("-" * 40)
        
        basic_queries = [
            "test",
            "service",
            "cost",
            "price",
            "towing",
            "battery"
        ]
        
        search_results = {}
        
        for query in basic_queries:
            try:
                logger.info(f"Testing query: '{query}'")
                start_time = time.time()
                results = await simple_rag.search(query, limit=3)
                search_time = (time.time() - start_time) * 1000
                
                search_results[query] = {
                    "results_count": len(results),
                    "search_time_ms": search_time,
                    "results": results[:2] if results else []  # Store top 2 for analysis
                }
                
                if results:
                    best_score = results[0].get("score", 0)
                    logger.info(f"   ✅ Found {len(results)} results (best score: {best_score:.3f}) in {search_time:.1f}ms")
                    
                    # Show top result content
                    top_result = results[0]
                    text_preview = top_result.get("text", "")[:100] + "..." if len(top_result.get("text", "")) > 100 else top_result.get("text", "")
                    logger.info(f"   📄 Top result: {text_preview}")
                else:
                    logger.warning(f"   ⚠️ No results found")
                    
            except Exception as e:
                logger.error(f"   ❌ Search error: {e}")
                search_results[query] = {"error": str(e)}
        
        self.test_results["basic_search"] = search_results
        
        # Analyze results
        successful_searches = len([r for r in search_results.values() if "error" not in r and r.get("results_count", 0) > 0])
        logger.info(f"\n📊 Basic Search Summary:")
        logger.info(f"   Successful searches: {successful_searches}/{len(basic_queries)}")
        
        return successful_searches > len(basic_queries) // 2
    
    async def test_real_user_queries(self):
        """Test with real queries from your log file"""
        logger.info("\n👤 TEST 4: Real User Queries (From Your Log)")
        logger.info("-" * 40)
        
        # Extracted from your log
        real_queries = [
            "I want to change my tire. Uh, there is a flat tire.",
            "Then what other services you provide?",
            "Can you please state me in detail, uh, the stop services report?",
            "Disposable shop services?",
            "If I want to talk my vehicle, then",
            "towing cost",
            "flat tire change cost",
            "roadside assistance services",
            "battery jumpstart",
            "membership plans"
        ]
        
        real_query_results = {}
        
        for query in real_queries:
            try:
                logger.info(f"\nTesting real query: '{query[:50]}...'")
                start_time = time.time()
                results = await simple_rag.search(query, limit=3)
                search_time = (time.time() - start_time) * 1000
                
                real_query_results[query] = {
                    "results_count": len(results),
                    "search_time_ms": search_time,
                    "best_score": results[0].get("score", 0) if results else 0,
                    "results": results[:1] if results else []
                }
                
                if results:
                    best_score = results[0].get("score", 0)
                    logger.info(f"   ✅ Found {len(results)} results (score: {best_score:.3f}) in {search_time:.1f}ms")
                    
                    # Show what the agent would get
                    top_result = results[0]
                    text_preview = top_result.get("text", "")[:150] + "..." if len(top_result.get("text", "")) > 150 else top_result.get("text", "")
                    logger.info(f"   📄 Agent would receive: {text_preview}")
                    
                    # Check if score is good enough for agent
                    if best_score >= 0.3:
                        logger.info(f"   ✅ Good score - agent would use this")
                    elif best_score >= 0.2:
                        logger.warning(f"   ⚠️ Low score - agent might combine results")
                    else:
                        logger.warning(f"   ❌ Very low score - agent might not use this")
                else:
                    logger.warning(f"   ⚠️ No results - agent would have no context")
                    
            except Exception as e:
                logger.error(f"   ❌ Search error: {e}")
                real_query_results[query] = {"error": str(e)}
        
        self.test_results["real_queries"] = real_query_results
        
        # Analyze results
        good_results = len([r for r in real_query_results.values() if r.get("best_score", 0) >= 0.3])
        okay_results = len([r for r in real_query_results.values() if 0.2 <= r.get("best_score", 0) < 0.3])
        poor_results = len([r for r in real_query_results.values() if r.get("best_score", 0) < 0.2])
        
        logger.info(f"\n📊 Real Query Analysis:")
        logger.info(f"   Good results (≥0.3): {good_results}")
        logger.info(f"   Okay results (0.2-0.3): {okay_results}")
        logger.info(f"   Poor results (<0.2): {poor_results}")
        
        return good_results + okay_results > poor_results
    
    async def test_query_types(self):
        """Test different types of queries"""
        logger.info("\n🎯 TEST 5: Different Query Types")
        logger.info("-" * 40)
        
        query_types = {
            "pricing": [
                "how much does towing cost",
                "tire change price",
                "membership plan cost",
                "battery jumpstart fee"
            ],
            "services": [
                "what services do you offer",
                "roadside assistance options",
                "emergency services",
                "24 hour service"
            ],
            "company_info": [
                "business hours",
                "service hours",
                "company information",
                "contact information"
            ],
            "specific_issues": [
                "flat tire help",
                "dead battery",
                "locked out of car",
                "out of gas"
            ]
        }
        
        type_results = {}
        
        for query_type, queries in query_types.items():
            logger.info(f"\nTesting {query_type} queries:")
            type_results[query_type] = {}
            
            for query in queries:
                try:
                    results = await simple_rag.search(query, limit=2)
                    best_score = results[0].get("score", 0) if results else 0
                    
                    type_results[query_type][query] = {
                        "results_count": len(results),
                        "best_score": best_score
                    }
                    
                    logger.info(f"   '{query}': {len(results)} results (score: {best_score:.3f})")
                    
                except Exception as e:
                    logger.error(f"   ❌ Error with '{query}': {e}")
                    type_results[query_type][query] = {"error": str(e)}
        
        self.test_results["query_types"] = type_results
        return True
    
    async def test_score_analysis(self):
        """Analyze score distribution and thresholds"""
        logger.info("\n📈 TEST 6: Score Analysis")
        logger.info("-" * 40)
        
        test_queries = [
            "towing service cost",
            "flat tire change",
            "battery jumpstart price",
            "membership plan benefits",
            "business hours",
            "emergency roadside assistance"
        ]
        
        all_scores = []
        score_results = {}
        
        for query in test_queries:
            try:
                results = await simple_rag.search(query, limit=5)
                scores = [r.get("score", 0) for r in results]
                all_scores.extend(scores)
                
                score_results[query] = {
                    "scores": scores,
                    "max_score": max(scores) if scores else 0,
                    "min_score": min(scores) if scores else 0,
                    "avg_score": sum(scores) / len(scores) if scores else 0
                }
                
                logger.info(f"'{query}': max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}" if scores else f"'{query}': no results")
                
            except Exception as e:
                logger.error(f"Error with '{query}': {e}")
        
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            max_score = max(all_scores)
            min_score = min(all_scores)
            
            logger.info(f"\n📊 Overall Score Statistics:")
            logger.info(f"   Average score: {avg_score:.3f}")
            logger.info(f"   Max score: {max_score:.3f}")
            logger.info(f"   Min score: {min_score:.3f}")
            
            # Recommend threshold
            if avg_score > 0.5:
                logger.info(f"   ✅ Good average score - current threshold (0.2) is appropriate")
            elif avg_score > 0.3:
                logger.info(f"   ⚠️ Moderate average score - consider lowering threshold to 0.15")
            else:
                logger.warning(f"   ❌ Low average score - consider re-ingesting data or lowering threshold to 0.1")
        
        self.test_results["score_analysis"] = score_results
        return True
    
    async def test_cache_functionality(self):
        """Test caching functionality"""
        logger.info("\n💾 TEST 7: Cache Functionality")
        logger.info("-" * 40)
        
        test_query = "towing service cost"
        
        # First search (no cache)
        logger.info(f"First search for: '{test_query}'")
        start_time = time.time()
        results1 = await simple_rag.search(test_query, limit=2)
        first_time = (time.time() - start_time) * 1000
        
        # Second search (should use cache)
        logger.info(f"Second search for: '{test_query}' (should be cached)")
        start_time = time.time()
        results2 = await simple_rag.search(test_query, limit=2)
        second_time = (time.time() - start_time) * 1000
        
        cache_working = second_time < first_time * 0.5  # Second should be much faster
        
        logger.info(f"   First search: {first_time:.1f}ms")
        logger.info(f"   Second search: {second_time:.1f}ms")
        logger.info(f"   Cache working: {'✅ YES' if cache_working else '❌ NO'}")
        
        self.test_results["cache"] = {
            "first_time_ms": first_time,
            "second_time_ms": second_time,
            "cache_working": cache_working
        }
        
        return cache_working
    
    async def test_performance(self):
        """Test performance under load"""
        logger.info("\n⚡ TEST 8: Performance Testing")
        logger.info("-" * 40)
        
        queries = [
            "towing cost",
            "flat tire",
            "battery service",
            "membership plans",
            "business hours"
        ]
        
        # Test concurrent searches
        logger.info("Testing concurrent searches...")
        start_time = time.time()
        
        tasks = [simple_rag.search(query, limit=2) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(queries)
        
        errors = [r for r in results if isinstance(r, Exception)]
        successful = len(results) - len(errors)
        
        logger.info(f"   Total time: {total_time:.1f}ms")
        logger.info(f"   Average per query: {avg_time:.1f}ms")
        logger.info(f"   Successful: {successful}/{len(queries)}")
        logger.info(f"   Errors: {len(errors)}")
        
        performance_good = avg_time < 1000 and len(errors) == 0
        
        self.test_results["performance"] = {
            "total_time_ms": total_time,
            "avg_time_ms": avg_time,
            "successful": successful,
            "errors": len(errors),
            "performance_good": performance_good
        }
        
        return performance_good
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("\n📋 COMPREHENSIVE TEST REPORT")
        logger.info("=" * 60)
        
        # Overall assessment
        issues = []
        recommendations = []
        
        # Check initialization
        if not self.test_results.get("initialization", {}).get("success", False):
            issues.append("RAG system failed to initialize")
            recommendations.append("Check Qdrant connection and OpenAI API key")
        
        # Check document count
        points_count = self.test_results.get("collection_status", {}).get("points_count", 0)
        if points_count == 0:
            issues.append("No documents in knowledge base")
            recommendations.append("Run: python excel_ingest.py --file data/Roadside_Assist_Info.xlsx")
        elif points_count < 50:
            issues.append(f"Low document count: {points_count}")
            recommendations.append("Consider re-ingesting Excel file with more comprehensive data")
        
        # Check search performance
        real_query_results = self.test_results.get("real_queries", {})
        if real_query_results:
            good_results = len([r for r in real_query_results.values() if r.get("best_score", 0) >= 0.3])
            total_results = len(real_query_results)
            if good_results < total_results * 0.5:
                issues.append(f"Poor search results: only {good_results}/{total_results} queries had good scores")
                recommendations.append("Consider lowering similarity threshold or improving data quality")
        
        # Check cache
        if not self.test_results.get("cache", {}).get("cache_working", False):
            issues.append("Cache not working effectively")
            recommendations.append("Cache might need debugging")
        
        # Generate summary
        logger.info("🎯 SUMMARY:")
        if not issues:
            logger.info("✅ ALL TESTS PASSED - RAG system is working correctly!")
        else:
            logger.info("❌ ISSUES FOUND:")
            for issue in issues:
                logger.info(f"   - {issue}")
        
        if recommendations:
            logger.info("\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                logger.info(f"   - {rec}")
        
        # Configuration suggestions
        logger.info("\n⚙️ CONFIGURATION SUGGESTIONS:")
        
        score_analysis = self.test_results.get("score_analysis", {})
        if score_analysis:
            all_scores = []
            for query_data in score_analysis.values():
                if "scores" in query_data:
                    all_scores.extend(query_data["scores"])
            
            if all_scores:
                avg_score = sum(all_scores) / len(all_scores)
                if avg_score < 0.3:
                    logger.info(f"   - Consider lowering similarity threshold from {config.similarity_threshold} to 0.15")
                if avg_score < 0.2:
                    logger.info(f"   - Consider lowering similarity threshold from {config.similarity_threshold} to 0.1")
        
        performance = self.test_results.get("performance", {})
        avg_time = performance.get("avg_time_ms", 0)
        if avg_time > 1000:
            logger.info(f"   - Consider increasing RAG timeout from {config.rag_timeout_ms}ms to {int(avg_time * 1.5)}ms")
        
        # Save detailed report
        report_path = Path("rag_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        logger.info(f"\n📄 Detailed report saved to: {report_path}")

async def main():
    """Run RAG system tests"""
    logger.info("🧪 RAG SYSTEM TESTING TOOL")
    logger.info("Testing the same RAG system used in main.py")
    logger.info(f"Qdrant URL: {config.qdrant_url}")
    logger.info(f"Collection: {config.qdrant_collection_name}")
    logger.info(f"Current threshold: {config.similarity_threshold}")
    
    tester = RAGSystemTester()
    await tester.run_all_tests()
    
    logger.info("\n🎯 Testing complete!")
    logger.info("💡 Use this information to debug your voice agent issues")

if __name__ == "__main__":
    asyncio.run(main())