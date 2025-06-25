# fixed_main.py - PURE RAG WITHOUT FALLBACKS
"""
FIXED: Pure RAG system that ONLY uses Excel data
NO hardcoded responses, NO fallbacks - Excel data or nothing
Based on LiveKit LlamaIndex RAG example patterns
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from livekit import api, agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    JobProcess
)
from livekit.plugins import deepgram, openai, elevenlabs, silero

# Fixed turn detector import
try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    TURN_DETECTOR_AVAILABLE = True
except ImportError:
    TURN_DETECTOR_AVAILABLE = False

# Use simple RAG system for reliability
from simple_rag_system import simple_rag
from config import config
from call_transcription_storage import call_storage

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class CallData:
    """Call data structure"""
    session_id: Optional[str] = None
    caller_id: Optional[str] = None
    phone_number: Optional[str] = None
    caller_name: Optional[str] = None
    location: Optional[str] = None
    vehicle_year: Optional[str] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_color: Optional[str] = None
    service_type: Optional[str] = None
    issue_description: Optional[str] = None
    is_returning_caller: bool = False
    previous_calls_count: int = 0
    gathered_info: Dict[str, bool] = field(default_factory=lambda: {
        "name": False, "phone": False, "location": False, 
        "vehicle": False, "service": False
    })

class PureRAGAgent(Agent):
    """
    PURE RAG Agent - ONLY uses Excel knowledge base
    NO hardcoded responses, NO fallbacks
    """
    
    def __init__(self, call_data: CallData):
        self.call_data = call_data
        self.rag_cache = {}
        self.last_rag_lookup = 0
        
        instructions = self._build_instructions()
        super().__init__(instructions=instructions)
    
    def _build_instructions(self) -> str:
        """Build context-aware instructions - NO hardcoded pricing"""
        base_instructions = """You are Mark, a professional roadside assistance operator.

CRITICAL: You provide roadside assistance services directly. You ARE the service provider.

YOUR EXCEL KNOWLEDGE BASE:
- Contains ALL pricing, services, and policy information
- ALWAYS use search_knowledge_base() for ANY pricing questions
- NEVER give generic responses - only use Excel data
- If Excel search fails, say "Let me check our current rates and get back to you"

CONVERSATION FLOW:
1. Collect customer information step by step
2. Use search_knowledge_base() for ALL service questions
3. Provide EXACT information from Excel spreadsheet only
4. Route to specialists when needed

INFORMATION GATHERING:
1. Full name
2. Phone number  
3. Vehicle location (complete address)
4. Vehicle details (year, make, model)
5. Service needed

Use gather_caller_information() to store each piece.

CRITICAL RULES:
- NO generic pricing (like "$25-35")
- NO fallback responses
- ONLY use Excel spreadsheet data
- If knowledge base fails, admit it and offer callback

Keep responses under 30 words for phone clarity."""

        if self.call_data.is_returning_caller:
            base_instructions += f"""

RETURNING CALLER:
- Previous calls: {self.call_data.previous_calls_count}
- Welcome them back: "Welcome back! I see you've called us before."
"""
        
        return base_instructions
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """
        PURE RAG injection - NO fallbacks
        """
        try:
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 3:
                return
            
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_rag_lookup < 1.0:
                return
            
            # Check if this should trigger RAG
            if self._should_use_rag(user_text):
                self.last_rag_lookup = current_time
                
                try:
                    rag_context = await self._perform_rag_lookup(user_text)
                    if rag_context:
                        turn_ctx.add_message(
                            role="system",
                            content=f"[EXCEL_DATA]: {rag_context}\n\nThis is EXACT data from your Excel knowledge base. Use this information to answer the customer's question. Do NOT add any other information."
                        )
                        logger.info(f"✅ Excel data injected: {user_text[:50]}...")
                    else:
                        # NO fallback - just log that no data was found
                        logger.warning(f"⚠️ No Excel data found for: {user_text[:50]}...")
                        
                except Exception as e:
                    logger.warning(f"⚠️ RAG lookup failed: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Error in RAG context injection: {e}")
    
    def _should_use_rag(self, user_text: str) -> bool:
        """Determine if query should trigger RAG lookup"""
        rag_keywords = [
            "cost", "price", "how much", "fee", "charge", "rate", "pricing",
            "service", "towing", "battery", "tire", "jumpstart", "jump start",
            "coverage", "policy", "hours", "available", "time", "plan",
            "help", "assist", "options", "what do you offer", "services"
        ]
        user_lower = user_text.lower()
        return any(keyword in user_lower for keyword in rag_keywords)
    
    async def _perform_rag_lookup(self, query: str) -> Optional[str]:
        """
        PURE Excel data lookup - NO fallbacks
        """
        try:
            # Check cache first
            cache_key = query.lower().strip()[:100]
            if cache_key in self.rag_cache:
                return self.rag_cache[cache_key]
            
            # Use simple_rag for Excel data search
            results = await asyncio.wait_for(
                simple_rag.search(query, limit=2),
                timeout=3.0
            )
            
            if results and len(results) > 0:
                # Format the best result from Excel
                best_result = results[0]
                context = self._format_excel_result(best_result["text"])
                
                # Cache successful result
                self.rag_cache[cache_key] = context
                if len(self.rag_cache) > 20:
                    oldest_key = next(iter(self.rag_cache))
                    del self.rag_cache[oldest_key]
                
                logger.info(f"📊 Excel data found (score: {best_result.get('score', 0):.3f})")
                return context
            else:
                logger.warning("🔍 No Excel data found for query - NO fallback used")
                return None
                
        except asyncio.TimeoutError:
            logger.warning("⏰ Excel data lookup timeout - NO fallback used")
            return None
        except Exception as e:
            logger.error(f"❌ Excel data lookup error: {e} - NO fallback used")
            return None
    
    def _format_excel_result(self, raw_text: str) -> str:
        """Format Excel result for context injection"""
        if not raw_text:
            return ""
        
        # Clean and format Excel data
        cleaned = raw_text.strip()
        for char in ["•", "-", "*", "\n", "\t"]:
            cleaned = cleaned.replace(char, " ")
        
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")
        
        # Keep concise for voice but preserve pricing
        if len(cleaned) > 200:
            sentences = cleaned.split(".")
            result = sentences[0].strip()
            if len(result) < 50 and len(sentences) > 1:
                result += ". " + sentences[1].strip()
        else:
            result = cleaned
        
        return result.strip()

    @function_tool()
    async def gather_caller_information(
        self, 
        context: RunContext[CallData],
        name: str = None,
        phone: str = None,
        location: str = None,
        vehicle_year: str = None,
        vehicle_make: str = None,
        vehicle_model: str = None,
        vehicle_color: str = None,
        issue: str = None,
        service_needed: str = None
    ) -> str:
        """Store caller information"""
        
        updates = []
        
        if name:
            context.userdata.caller_name = name.strip()
            context.userdata.gathered_info["name"] = True
            updates.append(f"name: {name}")
            
        if phone:
            context.userdata.phone_number = phone.strip()
            context.userdata.gathered_info["phone"] = True
            updates.append(f"phone: {phone}")
            
        if location:
            context.userdata.location = location.strip()
            context.userdata.gathered_info["location"] = True
            updates.append(f"location: {location}")
            
        if vehicle_year:
            context.userdata.vehicle_year = vehicle_year
        if vehicle_make:
            context.userdata.vehicle_make = vehicle_make
        if vehicle_model:
            context.userdata.vehicle_model = vehicle_model
        if vehicle_color:
            context.userdata.vehicle_color = vehicle_color
            
        if any([vehicle_year, vehicle_make, vehicle_model, vehicle_color]):
            context.userdata.gathered_info["vehicle"] = True
            vehicle_info = f"{vehicle_year or ''} {vehicle_make or ''} {vehicle_model or ''} {vehicle_color or ''}".strip()
            updates.append(f"vehicle: {vehicle_info}")
            
        if issue:
            context.userdata.issue_description = issue
        if service_needed:
            context.userdata.service_type = service_needed
            context.userdata.gathered_info["service"] = True
            updates.append(f"service: {service_needed}")
        
        logger.info(f"📝 Stored info: {updates}")
        
        gathered = context.userdata.gathered_info
        if all([gathered["name"], gathered["phone"], gathered["location"], gathered["vehicle"], gathered["service"]]):
            return "Perfect! I have all the information I need. Let me get you an exact quote from our system."
        else:
            missing = [key for key, value in gathered.items() if not value]
            next_questions = {
                "name": "Could you tell me your full name please?",
                "phone": "What's a good phone number to reach you?",
                "location": "Where is your vehicle located? Please give me the complete address.",
                "vehicle": "What's the year, make, and model of your vehicle?",
                "service": "What type of service do you need today?"
            }
            next_missing = missing[0] if missing else None
            question = next_questions.get(next_missing, "Let me get some more information.")
            return question

    @function_tool()
    async def search_knowledge_base(
        self, 
        context: RunContext[CallData],
        query: str
    ) -> str:
        """
        PURE Excel knowledge base search - NO fallbacks allowed
        """
        try:
            logger.info(f"🔍 Searching Excel knowledge base: {query}")
            
            # Search Excel data with longer timeout
            results = await asyncio.wait_for(
                simple_rag.search(query, limit=3),
                timeout=5.0
            )
            
            if not results:
                logger.warning("⚠️ No data found in Excel knowledge base")
                # NO fallback - be honest about data availability
                return "I don't have that specific information in my current database. Let me get someone who can access our complete pricing system to help you."
            
            # Format results from Excel data ONLY
            response_parts = []
            for result in results[:2]:
                if result.get("score", 0) >= 0.2:
                    formatted = self._format_excel_result(result["text"])
                    if formatted and formatted not in response_parts:
                        response_parts.append(formatted)
            
            if response_parts:
                response = " ".join(response_parts)
                logger.info(f"📊 Excel data search successful - returned {len(response_parts)} results")
                return response
            else:
                logger.warning("⚠️ Excel data found but formatting failed")
                return "I found information in our database but need to verify the details. Let me connect you with someone who can provide the exact pricing."
                
        except Exception as e:
            logger.error(f"❌ Excel database search error: {e}")
            # NO fallback - admit system issue
            return "I'm having trouble accessing our pricing database right now. Let me connect you with someone who can provide current pricing information."

    @function_tool()
    async def route_to_towing_specialist(self, context: RunContext[CallData]) -> Agent:
        """Route to towing specialist"""
        logger.info("🔄 ROUTING TO TOWING SPECIALIST")
        return TowingSpecialistAgent(context.userdata)

    @function_tool()
    async def route_to_battery_specialist(self, context: RunContext[CallData]) -> Agent:
        """Route to battery specialist"""
        logger.info("🔄 ROUTING TO BATTERY SPECIALIST")
        return BatterySpecialistAgent(context.userdata)

class TowingSpecialistAgent(Agent):
    """Towing specialist with PURE Excel access"""
    
    def __init__(self, customer_data: CallData):
        self.customer_data = customer_data
        
        instructions = f"""You are a TOWING SPECIALIST.

CUSTOMER INFO:
- Name: {customer_data.caller_name}
- Phone: {customer_data.phone_number}
- Location: {customer_data.location}
- Vehicle: {customer_data.vehicle_year or ''} {customer_data.vehicle_make or ''} {customer_data.vehicle_model or ''}

YOUR JOB:
- Ask where they want the vehicle towed
- Use search_knowledge_base() for EXACT pricing from Excel
- Provide accurate quotes from Excel data ONLY
- NO generic pricing - Excel data only

CRITICAL: Only use search_knowledge_base() for pricing. NO hardcoded rates."""
        
        super().__init__(instructions=instructions)

    async def on_enter(self):
        """Greet customer with their information"""
        name = self.customer_data.caller_name or "there"
        location = self.customer_data.location or "your location"
        vehicle = f"{self.customer_data.vehicle_year or ''} {self.customer_data.vehicle_make or ''} {self.customer_data.vehicle_model or ''}".strip()
        
        greeting = f"Hi {name}, I'm your towing specialist. I have you at {location} with your {vehicle}. Where would you like it towed to?"
        
        await self.session.generate_reply(instructions=f"Say exactly: '{greeting}'")

    @function_tool()
    async def search_knowledge_base(self, context: RunContext[CallData], query: str) -> str:
        """Search Excel for towing info - NO fallbacks"""
        try:
            enhanced_query = f"towing service rates pricing {query}"
            results = await asyncio.wait_for(simple_rag.search(enhanced_query, limit=2), timeout=3.0)
            
            if results and results[0].get("score", 0) >= 0.2:
                return results[0]["text"][:200]
            
            # NO fallback pricing
            return "I need to look up the exact towing rates in our current system. Let me get you the most accurate pricing."
        except Exception:
            return "I need to verify our current towing rates. Let me get the exact pricing for you."

class BatterySpecialistAgent(Agent):
    """Battery specialist with PURE Excel access"""
    
    def __init__(self, customer_data: CallData):
        self.customer_data = customer_data
        super().__init__(instructions="You are a BATTERY SPECIALIST. Use search_knowledge_base() for EXACT pricing from Excel ONLY.")

    async def on_enter(self):
        name = self.customer_data.caller_name or "there"
        await self.session.generate_reply(
            instructions=f"Greet: 'Hi {name}, I'm your battery specialist. What battery problems are you experiencing?'"
        )

    @function_tool()
    async def search_knowledge_base(self, context: RunContext[CallData], query: str) -> str:
        """Search Excel for battery service info - NO fallbacks"""
        try:
            enhanced_query = f"battery jumpstart service pricing {query}"
            results = await asyncio.wait_for(simple_rag.search(enhanced_query, limit=1), timeout=3.0)
            return results[0]["text"][:200] if results else "Let me check our current battery service rates in the system."
        except:
            return "Let me verify our battery service pricing in our database."

async def identify_caller_with_history(ctx: JobContext) -> CallData:
    """Identify caller and load history"""
    try:
        participant = await ctx.wait_for_participant()
        
        if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            logger.warning("⚠️ No SIP participant found")
            return CallData()
        
        # Extract phone number
        phone_number = "unknown"
        phone_attrs = ["sip.phoneNumber", "sip.from_number", "sip.caller_number"]
        
        for attr in phone_attrs:
            if attr in participant.attributes:
                phone_number = participant.attributes[attr]
                break
        
        logger.info(f"📞 Incoming call from: {phone_number}")
        
        # Start call session
        session_id, caller_id = await call_storage.start_call_session(
            phone_number=phone_number,
            session_metadata={"participant_identity": participant.identity}
        )
        
        # Check caller history
        caller_profile = await call_storage.get_caller_by_phone(phone_number)
        
        is_returning = False
        previous_calls = 0
        
        if caller_profile and caller_profile.total_calls > 0:
            is_returning = True
            previous_calls = caller_profile.total_calls
            logger.info(f"🔄 Returning caller: {previous_calls} previous calls")
        else:
            logger.info("✨ New caller detected")
        
        # Create call data
        call_data = CallData()
        call_data.session_id = session_id
        call_data.caller_id = caller_id
        call_data.phone_number = phone_number
        call_data.is_returning_caller = is_returning
        call_data.previous_calls_count = previous_calls
        
        return call_data
        
    except Exception as e:
        logger.error(f"❌ Error identifying caller: {e}")
        return CallData()

def prewarm(proc: JobProcess):
    """Prewarm function to load models early"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """
    PURE RAG entrypoint - NO fallbacks allowed
    """
    
    logger.info("🚀 PURE RAG Voice Agent Starting - NO FALLBACKS")
    logger.info("📊 ONLY Excel knowledge base data will be used")
    
    await ctx.connect()
    logger.info("✅ Connected to room")
    
    # Initialize RAG with STRICT timeout
    try:
        rag_start = time.time()
        logger.info("🔧 Initializing Excel knowledge base (STRICT mode)...")
        
        # STRICT initialization - must succeed
        success = await asyncio.wait_for(simple_rag.initialize(), timeout=10.0)
        rag_time = (time.time() - rag_start) * 1000
        
        if success:
            status = await simple_rag.get_status()
            points_count = status.get("points_count", 0)
            logger.info(f"✅ Excel knowledge base initialized in {rag_time:.1f}ms")
            logger.info(f"📊 Knowledge base has {points_count} documents")
            
            if points_count == 0:
                logger.error("❌ CRITICAL: Excel knowledge base is EMPTY!")
                logger.error("💡 Run: python quick_ingest.py --file data/your_excel_file.xlsx")
                logger.error("💡 Agent will NOT work without Excel data!")
                return
        else:
            logger.error("❌ CRITICAL: Excel knowledge base initialization FAILED!")
            logger.error("💡 Check Qdrant: docker-compose up -d")
            logger.error("💡 Agent cannot start without knowledge base!")
            return
            
    except asyncio.TimeoutError:
        logger.error("❌ CRITICAL: Excel knowledge base initialization TIMEOUT!")
        logger.error("💡 Check Qdrant connection and restart")
        return
    except Exception as e:
        logger.error(f"❌ CRITICAL: Excel knowledge base error: {e}")
        return
    
    # Identify caller
    call_data = await identify_caller_with_history(ctx)
    
    # Create session
    session_params = {
        "vad": ctx.proc.userdata["vad"],
        "stt": deepgram.STT(model="nova-2-general", language="en-US"),
        "llm": openai.LLM(model="gpt-4o-mini", temperature=0.1),
        "userdata": call_data
    }
    
    # TTS setup
    try:
        session_params["tts"] = elevenlabs.TTS(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            voice_settings=elevenlabs.VoiceSettings(
                stability=0.7, similarity_boost=0.8, style=0.0, speed=0.9
            ),
            model="eleven_turbo_v2_5",
        )
        logger.info("✅ Using ElevenLabs TTS")
    except Exception as e:
        logger.warning(f"⚠️ ElevenLabs TTS failed, using OpenAI: {e}")
        session_params["tts"] = openai.TTS(voice="alloy")
    
    # Add turn detection if available
    if TURN_DETECTOR_AVAILABLE:
        session_params["turn_detection"] = MultilingualModel()
        logger.info("✅ Using turn detection")
    
    session = AgentSession[CallData](**session_params)
    
    # Create PURE RAG agent
    initial_agent = PureRAGAgent(call_data)
    
    # Start session
    await session.start(
        agent=initial_agent,
        room=ctx.room
    )
    
    # Generate greeting
    if call_data.is_returning_caller:
        greeting = "Say: 'Welcome back! I see you've called us before. How can I help you today?'"
    else:
        greeting = "Say: 'Roadside assistance, this is Mark, how can I help you today?'"
    
    await session.generate_reply(instructions=greeting)
    
    logger.info("✅ PURE RAG Agent Ready - NO FALLBACKS")
    logger.info(f"📞 Session ID: {call_data.session_id}")
    logger.info(f"👤 Caller ID: {call_data.caller_id}")
    logger.info(f"📱 Phone: {call_data.phone_number}")
    logger.info(f"🔄 Returning: {call_data.is_returning_caller}")
    logger.info("🎯 ONLY Excel knowledge base data will be used")

if __name__ == "__main__":
    try:
        logger.info("🎙️ Starting PURE RAG Voice Agent - NO FALLBACKS")
        logger.info("📊 Excel knowledge base ONLY - NO hardcoded responses")
        logger.info("🔧 Based on LiveKit LlamaIndex RAG patterns")
        
        worker_options = WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="my-telephony-agent"
        )
        
        cli.run_app(worker_options)
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)