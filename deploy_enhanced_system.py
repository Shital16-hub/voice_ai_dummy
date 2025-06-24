# deploy_enhanced_system.py
"""
Deployment script for Enhanced Multi-Agent Voice AI System
Handles initialization, validation, and startup of all components
"""
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import argparse

from enhanced_config import enhanced_config, validate_enhanced_config
from conversation_memory_manager import memory_manager
from qdrant_rag_system import qdrant_rag

# Setup logging
logging.basicConfig(
    level=getattr(logging, enhanced_config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(enhanced_config.logs_dir / "deployment.log")
    ]
)
logger = logging.getLogger(__name__)

class SystemDeployment:
    """Manages deployment of the enhanced voice AI system"""
    
    def __init__(self):
        self.deployment_start_time = time.time()
        self.components_status = {
            "config": False,
            "directories": False,
            "qdrant": False,
            "memory_system": False,
            "knowledge_base": False,
            "docker_services": False
        }
        
    async def deploy_system(self, mode: str = "development") -> bool:
        """Deploy the complete enhanced system"""
        
        logger.info("🚀 Starting Enhanced Multi-Agent Voice AI System Deployment")
        logger.info(f"📋 Deployment Mode: {mode}")
        logger.info(f"🎯 Features: Multi-Agent, Conversation Memory, Dynamic Knowledge")
        
        try:
            # Step 1: Validate configuration
            await self._validate_configuration()
            
            # Step 2: Setup directories and environment
            await self._setup_environment()
            
            # Step 3: Initialize Docker services
            await self._setup_docker_services()
            
            # Step 4: Initialize Qdrant and knowledge base
            await self._setup_knowledge_system()
            
            # Step 5: Initialize memory system
            await self._setup_memory_system()
            
            # Step 6: Validate all components
            await self._validate_components()
            
            # Step 7: Start the appropriate agent
            await self._start_agent_system(mode)
            
            deployment_time = (time.time() - self.deployment_start_time) * 1000
            logger.info(f"✅ System deployment completed in {deployment_time:.1f}ms")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            await self._cleanup_on_failure()
            return False
    
    async def _validate_configuration(self):
        """Validate system configuration"""
        logger.info("📋 Validating configuration...")
        
        try:
            validate_enhanced_config()
            self.components_status["config"] = True
            logger.info("✅ Configuration validated")
            
            # Log enabled features
            features = []
            if enhanced_config.enable_multi_agent:
                features.append("Multi-Agent")
            if enhanced_config.enable_conversation_memory:
                features.append("Memory")
            if enhanced_config.natural_conversation_flow:
                features.append("Natural Flow")
            if enhanced_config.enable_conversation_analytics:
                features.append("Analytics")
                
            logger.info(f"🎯 Enabled Features: {', '.join(features)}")
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise
    
    async def _setup_environment(self):
        """Setup directories and environment"""
        logger.info("📁 Setting up environment...")
        
        try:
            # Create necessary directories
            enhanced_config.ensure_directories()
            
            # Verify write permissions
            test_file = enhanced_config.logs_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            
            self.components_status["directories"] = True
            logger.info("✅ Environment setup completed")
            
        except Exception as e:
            logger.error(f"❌ Environment setup failed: {e}")
            raise
    
    async def _setup_docker_services(self):
        """Setup Docker services (Qdrant)"""
        logger.info("🐳 Setting up Docker services...")
        
        try:
            # Check if Docker is available
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("⚠️ Docker not available, skipping container setup")
                return
            
            # Check if Qdrant is running
            try:
                import requests
                response = requests.get(f"{enhanced_config.qdrant_url}/", timeout=2)
                if response.status_code == 200:
                    logger.info("✅ Qdrant container already running")
                else:
                    await self._start_qdrant_container()
            except:
                await self._start_qdrant_container()
            
            self.components_status["docker_services"] = True
            
        except Exception as e:
            logger.error(f"❌ Docker services setup failed: {e}")
            raise
    
    async def _start_qdrant_container(self):
        """Start Qdrant Docker container"""
        logger.info("🚀 Starting Qdrant container...")
        
        try:
            # Check if docker-compose exists
            compose_file = Path("docker-compose.yml")
            if compose_file.exists():
                result = subprocess.run(["docker-compose", "up", "-d"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("✅ Qdrant started with docker-compose")
                else:
                    logger.error(f"❌ Docker compose failed: {result.stderr}")
                    raise Exception(f"Docker compose failed: {result.stderr}")
            else:
                # Start Qdrant manually
                docker_cmd = [
                    "docker", "run", "-d",
                    "--name", "qdrant_voice_ai",
                    "-p", "6333:6333",
                    "-p", "6334:6334",
                    "-v", f"{enhanced_config.qdrant_storage_dir}:/qdrant/storage",
                    "qdrant/qdrant:latest"
                ]
                
                result = subprocess.run(docker_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("✅ Qdrant container started manually")
                else:
                    logger.error(f"❌ Docker run failed: {result.stderr}")
                    raise Exception(f"Docker run failed: {result.stderr}")
            
            # Wait for Qdrant to be ready
            await self._wait_for_qdrant()
            
        except Exception as e:
            logger.error(f"❌ Failed to start Qdrant container: {e}")
            raise
    
    async def _wait_for_qdrant(self, max_attempts: int = 30):
        """Wait for Qdrant to be ready"""
        import requests
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{enhanced_config.qdrant_url}/", timeout=2)
                if response.status_code == 200:
                    logger.info("✅ Qdrant is ready")
                    return
            except:
                pass
            
            if attempt < max_attempts - 1:
                logger.info(f"⏳ Waiting for Qdrant... ({attempt + 1}/{max_attempts})")
                await asyncio.sleep(2)
        
        raise Exception("Qdrant failed to start within timeout period")
    
    async def _setup_knowledge_system(self):
        """Setup Qdrant and knowledge base"""
        logger.info("📚 Setting up knowledge system...")
        
        try:
            # Initialize Qdrant RAG system
            success = await qdrant_rag.initialize()
            if not success:
                raise Exception("Failed to initialize Qdrant RAG system")
            
            # Check if knowledge base has data
            try:
                from qdrant_client import QdrantClient
                client = QdrantClient(url=enhanced_config.qdrant_url)
                collection_info = client.get_collection(enhanced_config.qdrant_collection_name)
                
                if collection_info.points_count > 0:
                    logger.info(f"✅ Knowledge base ready with {collection_info.points_count} documents")
                else:
                    logger.warning("⚠️ Knowledge base is empty")
                    logger.info("💡 Run: python qdrant_data_ingestion.py --directory data")
                    
            except Exception:
                logger.warning("⚠️ Knowledge base collection not found")
                logger.info("💡 Run: python qdrant_data_ingestion.py --directory data")
            
            self.components_status["knowledge_base"] = True
            self.components_status["qdrant"] = True
            
        except Exception as e:
            logger.error(f"❌ Knowledge system setup failed: {e}")
            raise
    
    async def _setup_memory_system(self):
        """Setup conversation memory system"""
        logger.info("🧠 Setting up memory system...")
        
        try:
            if enhanced_config.enable_conversation_memory:
                # Initialize memory database
                memory_manager._init_database()
                
                # Test memory system
                test_session = memory_manager.create_session("test_session")
                await memory_manager.add_turn("test_session", "user", "test message")
                
                # Cleanup test session
                if "test_session" in memory_manager.active_memories:
                    del memory_manager.active_memories["test_session"]
                
                logger.info("✅ Conversation memory system ready")
            else:
                logger.info("ℹ️ Conversation memory disabled")
            
            self.components_status["memory_system"] = True
            
        except Exception as e:
            logger.error(f"❌ Memory system setup failed: {e}")
            raise
    
    async def _validate_components(self):
        """Validate all system components"""
        logger.info("✅ Validating system components...")
        
        failed_components = [
            component for component, status in self.components_status.items() 
            if not status
        ]
        
        if failed_components:
            raise Exception(f"Failed components: {', '.join(failed_components)}")
        
        # Test knowledge search
        try:
            results = await qdrant_rag.search("test search", limit=1)
            logger.info(f"🔍 Knowledge search test: {'✅ Working' if results is not None else '⚠️ No results'}")
        except Exception as e:
            logger.warning(f"⚠️ Knowledge search test failed: {e}")
        
        logger.info("✅ All components validated")
    
    async def _start_agent_system(self, mode: str):
        """Start the appropriate agent system"""
        logger.info(f"🎙️ Starting agent system in {mode} mode...")
        
        if mode == "single":
            logger.info("🤖 Starting single enhanced agent")
            logger.info("💡 Run: python enhanced_conversational_agent.py")
        elif mode == "multi":
            logger.info("🎭 Starting multi-agent orchestrator")
            logger.info("💡 Run: python multi_agent_orchestrator.py")
        else:
            logger.info("🔧 Development mode - manual start required")
            logger.info("💡 Options:")
            logger.info("   Single Agent: python enhanced_conversational_agent.py")
            logger.info("   Multi-Agent: python multi_agent_orchestrator.py")
    
    async def _cleanup_on_failure(self):
        """Cleanup resources on deployment failure"""
        logger.info("🧹 Cleaning up after failure...")
        
        try:
            # Close Qdrant connections
            if qdrant_rag.ready:
                await qdrant_rag.close()
            
            # Stop Docker containers if we started them
            subprocess.run(["docker", "stop", "qdrant_voice_ai"], 
                         capture_output=True, text=True)
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    def print_deployment_summary(self):
        """Print deployment summary"""
        print("\n" + "="*60)
        print("🎉 ENHANCED VOICE AI SYSTEM DEPLOYMENT SUMMARY")
        print("="*60)
        
        print(f"📊 Components Status:")
        for component, status in self.components_status.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component.replace('_', ' ').title()}")
        
        print(f"\n🎯 System Features:")
        if enhanced_config.enable_multi_agent:
            print("   🎭 Multi-Agent Architecture")
        if enhanced_config.enable_conversation_memory:
            print("   🧠 Conversation Memory")
        if enhanced_config.natural_conversation_flow:
            print("   💬 Natural Conversation Flow")
        if enhanced_config.auto_context_injection:
            print("   🔄 Auto Context Injection")
        
        print(f"\n📞 Agent Options:")
        print("   🤖 Single Agent: python enhanced_conversational_agent.py")
        print("   🎭 Multi-Agent: python multi_agent_orchestrator.py")
        
        print(f"\n📚 Knowledge Base:")
        print("   📄 Ingest Data: python qdrant_data_ingestion.py --directory data")
        print("   🔧 Clean Data: python enhanced_clean_qdrant.py")
        print("   📊 Performance: python performance_monitor.py")
        
        print(f"\n🔧 Utilities:")
        print("   📋 Config Check: python enhanced_config.py")
        print("   🧪 Test System: python test_optimized_agent.py")
        
        print("="*60)

async def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy Enhanced Voice AI System")
    parser.add_argument("--mode", choices=["development", "single", "multi"], 
                       default="development", help="Deployment mode")
    parser.add_argument("--skip-docker", action="store_true", 
                       help="Skip Docker container setup")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate configuration")
    
    args = parser.parse_args()
    
    deployment = SystemDeployment()
    
    try:
        if args.validate_only:
            await deployment._validate_configuration()
            print("✅ Configuration validation completed")
            return
        
        success = await deployment.deploy_system(args.mode)
        
        if success:
            deployment.print_deployment_summary()
            print("\n🎉 System ready for voice AI operations!")
        else:
            print("\n❌ Deployment failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Deployment error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())