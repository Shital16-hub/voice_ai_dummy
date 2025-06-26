# fix_openai_import.py
"""
Quick fix script to test the corrected OpenAI integration
"""
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_openai_fixed():
    """Test the fixed OpenAI integration"""
    logger.info("🔧 Testing fixed OpenAI integration...")
    
    try:
        from config import config
        import openai  # Direct import instead of livekit.plugins.openai
        
        if not config.openai_api_key:
            logger.error("❌ No OpenAI API key configured")
            return False
        
        client = openai.AsyncOpenAI(api_key=config.openai_api_key)
        
        # Test simple completion
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'OpenAI integration test successful'"}],
                max_tokens=10
            ),
            timeout=10.0
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"✅ OpenAI integration working: {result}")
        
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ OpenAI API timeout")
        return False
    except Exception as e:
        logger.error(f"❌ OpenAI integration failed: {e}")
        return False

async def test_conversation_analysis():
    """Test conversation history analysis"""
    logger.info("🧠 Testing conversation analysis...")
    
    try:
        from config import config
        import openai
        
        client = openai.AsyncOpenAI(api_key=config.openai_api_key)
        
        # Sample conversation history
        sample_history = """[2024-12-19] user: Hi, my battery is dead and I need a jump start
[2024-12-19] agent: I can help with that. What's your location?
[2024-12-19] user: I'm at 123 Main Street
[2024-12-19] agent: Perfect. I'll send a technician with a jump start service for $40
[2024-12-19] user: That sounds good, thank you
[2024-12-19] agent: You're welcome! The technician should arrive in about 20 minutes"""
        
        context_prompt = f"""Analyze this customer's previous roadside assistance call history and extract key context for a personalized greeting:

Previous conversations:
{sample_history}

Extract:
1. Previous services used (towing, battery, tire, etc.)
2. Vehicle information mentioned
3. Common issues or patterns
4. Service satisfaction indicators

Provide a brief, professional summary (2-3 sentences) that would help a dispatcher provide personalized service."""

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": context_prompt}],
                max_tokens=150,
                temperature=0.1
            ),
            timeout=10.0
        )
        
        analysis = response.choices[0].message.content.strip()
        logger.info(f"✅ History analysis result:")
        logger.info(f"   {analysis}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Conversation analysis failed: {e}")
        return False

async def test_greeting_generation():
    """Test personalized greeting generation"""
    logger.info("👋 Testing greeting generation...")
    
    try:
        from config import config
        import openai
        
        client = openai.AsyncOpenAI(api_key=config.openai_api_key)
        
        history_context = "Customer previously used battery jump start service. Vehicle was at 123 Main Street. Customer was satisfied with the service and responded positively."
        
        greeting_prompt = f"""You are Mark, a professional roadside assistance dispatcher. Generate a warm, personalized greeting for a returning customer based on their history.

Customer context:
- Phone: +1555123456
- Previous calls: 2
- History summary: {history_context}

Generate a natural, warm greeting that:
1. Welcomes them back
2. Shows you remember their previous interactions
3. Asks how you can help today
4. Keeps it under 30 words for phone clarity

Generate only the greeting text, no explanations."""

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": greeting_prompt}],
                max_tokens=80,
                temperature=0.3
            ),
            timeout=10.0
        )
        
        greeting = response.choices[0].message.content.strip().strip('"\'')
        logger.info(f"✅ Generated personalized greeting:")
        logger.info(f"   {greeting}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Greeting generation failed: {e}")
        return False

async def main():
    """Run all OpenAI integration tests"""
    logger.info("🚀 OPENAI INTEGRATION FIX TEST")
    logger.info("=" * 50)
    
    tests = [
        ("Basic OpenAI Connection", test_openai_fixed),
        ("Conversation Analysis", test_conversation_analysis),
        ("Greeting Generation", test_greeting_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}")
        logger.info("-" * 30)
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("✅ OpenAI integration is working correctly")
        logger.info("\n💡 NEXT STEPS:")
        logger.info("   1. Run: python test_history_integration.py")
        logger.info("   2. If successful, run: python main_multiagent_improved_with_history.py dev")
        logger.info("   3. Test with returning caller: +1555123456")
    else:
        logger.error("\n❌ SOME TESTS FAILED!")
        logger.error("🔧 TROUBLESHOOTING:")
        logger.error("   1. Check OpenAI API key in .env file")
        logger.error("   2. Verify internet connection")
        logger.error("   3. Ensure OpenAI package is installed: pip install openai")
        logger.error("   4. Check OpenAI API quota/billing")

if __name__ == "__main__":
    asyncio.run(main())