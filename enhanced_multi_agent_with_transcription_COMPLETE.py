# enhanced_multi_agent_with_transcription_FIXED.py
"""
FIXED Enhanced Multi-Agent System with Call Transcription
SOLUTION: Fixed the RAG lookup error by properly handling the context injection
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
    get_job_context
)
from livekit.plugins import deepgram, openai, elevenlabs, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from qdrant_rag_system import qdrant_rag
from config import config
from call_transcription_storage import call_storage, CallTranscriptionStorage

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class EnhancedCallData:
    """Enhanced call data with transcription tracking"""
    # Basic call info
    session_id: Optional[str] = None
    caller_id: Optional[str] = None
    phone_number: Optional[str] = None
    
    # Caller information
    caller_name: Optional[str] = None
    location: Optional[str] = None
    vehicle_year: Optional[str] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_color: Optional[str] = None
    service_type: Optional[str] = None
    issue_description: Optional[str] = None
    urgency_level: str = "normal"
    
    # Call progress tracking
    call_stage: str = "greeting"
    gathered_info: Dict[str, bool] = field(default_factory=lambda: {
        "name": False,
        "phone": False, 
        "location": False,
        "vehicle": False,
        "service": False
    })
    
    # History tracking
    is_returning_caller: bool = False
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    previous_calls_count: int = 0

class TranscriptionHandler:
    """Handles transcription events with LiveKit 1.1 patterns"""
    
    def __init__(self, storage: CallTranscriptionStorage):
        self.storage = storage
        
    def setup_transcription_handlers(
        self, 
        session: AgentSession, 
        call_data: EnhancedCallData
    ):
        """Setup transcription handlers using correct LiveKit 1.1 syntax"""
        
        @session.on("user_input_transcribed")
        def on_user_transcribed(event):
            asyncio.create_task(self._handle_user_transcription(event, call_data))
        
        @session.on("conversation_item_added")
        def on_conversation_item_added(event):
            asyncio.create_task(self._handle_conversation_item(event, call_data))
        
        logger.info("✅ Transcription handlers setup completed")
    
    async def _handle_user_transcription(self, event, call_data: EnhancedCallData):
        """Handle user speech transcription"""
        try:
            if call_data.session_id and call_data.caller_id and hasattr(event, 'transcript'):
                await self.storage.save_transcription_segment(
                    session_id=call_data.session_id,
                    caller_id=call_data.caller_id,
                    speaker="user",
                    text=event.transcript,
                    is_final=getattr(event, 'is_final', True),
                    confidence=getattr(event, 'confidence', None)
                )
                
                if getattr(event, 'is_final', True):
                    logger.info(f"👤 User: {event.transcript}")
                    
        except Exception as e:
            logger.error(f"❌ Error saving user transcription: {e}")
    
    async def _handle_conversation_item(self, event, call_data: EnhancedCallData):
        """Handle complete conversation turns"""
        try:
            if call_data.session_id and call_data.caller_id and hasattr(event, 'item'):
                item = event.item
                await self.storage.save_conversation_item(
                    session_id=call_data.session_id,
                    caller_id=call_data.caller_id,
                    role=item.role,
                    content=getattr(item, 'text_content', '') or getattr(item, 'content', ''),
                    interrupted=getattr(item, 'interrupted', False),
                    metadata={
                        "call_stage": call_data.call_stage,
                        "urgency_level": call_data.urgency_level
                    }
                )
                
                logger.info(f"💬 {item.role}: {getattr(item, 'text_content', '')[:100]}...")
                
        except Exception as e:
            logger.error(f"❌ Error saving conversation item: {e}")

class RAGEnhancedAgent(Agent):
    """FIXED: Base agent class with proper RAG patterns"""
    
    def __init__(self, instructions: str, rag_context_prefix: str = ""):
        super().__init__(instructions=instructions)
        self.rag_context_prefix = rag_context_prefix
        self.rag_cache = {}
        self.last_rag_query = ""
        self.rag_processing = False  # FIXED: Add processing flag
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """
        FIXED: Proper RAG pattern with error handling
        """
        try:
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 3 or self.rag_processing:
                return
            
            # Set processing flag to prevent concurrent RAG calls
            self.rag_processing = True
            
            try:
                # Perform RAG lookup based on user's message
                rag_context = await self._perform_rag_lookup(user_text)
                
                if rag_context:
                    # FIXED: Properly create and add context message
                    try:
                        # Add context as system message
                        turn_ctx.add_message(
                            role="system",
                            content=f"[RELEVANT_CONTEXT]: {rag_context}"
                        )
                        logger.info(f"✅ RAG context injected for: {user_text[:50]}...")
                    except Exception as ctx_error:
                        logger.error(f"❌ Context injection error: {ctx_error}")
                        
            except Exception as rag_error:
                logger.debug(f"🔍 RAG lookup - low confidence or no results: {rag_error}")
            finally:
                self.rag_processing = False
                
        except Exception as e:
            logger.error(f"❌ RAG lookup error: {e}")
            self.rag_processing = False
    
    async def _perform_rag_lookup(self, query: str) -> Optional[str]:
        """FIXED: Perform intelligent RAG lookup with proper error handling"""
        try:
            # Normalize query for caching
            normalized_query = query.lower().strip()
            cache_key = f"{self.rag_context_prefix}_{normalized_query[:100]}"
            
            # Check cache first
            if cache_key in self.rag_cache:
                logger.debug("📚 Using cached RAG result")
                return self.rag_cache[cache_key]
            
            # Enhance query with context prefix
            enhanced_query = f"{self.rag_context_prefix} {query}" if self.rag_context_prefix else query
            
            # Perform RAG search with timeout
            results = await asyncio.wait_for(
                qdrant_rag.search(enhanced_query, limit=2),
                timeout=0.8
            )
            
            if results and results[0]["score"] >= 0.25:
                # Format result for injection
                rag_context = self._format_rag_context(results[0]["text"], query)
                
                # Cache successful result
                self.rag_cache[cache_key] = rag_context
                
                # Limit cache size
                if len(self.rag_cache) > 50:
                    oldest_key = next(iter(self.rag_cache))
                    del self.rag_cache[oldest_key]
                
                logger.info(f"✅ RAG lookup successful (score: {results[0]['score']:.3f})")
                return rag_context
            else:
                logger.debug(f"🔍 RAG lookup - low confidence or no results")
                return None
                
        except asyncio.TimeoutError:
            logger.warning("⏰ RAG lookup timeout")
            return None
        except Exception as e:
            logger.error(f"❌ RAG lookup error: {e}")
            return None
    
    def _format_rag_context(self, raw_text: str, original_query: str) -> str:
        """Format RAG result for context injection"""
        # Clean the text
        cleaned = raw_text.strip()
        cleaned = cleaned.replace("•", "").replace("-", "").replace("*", "")
        cleaned = cleaned.replace("\n", " ").replace("\t", " ")
        
        # Remove multiple spaces
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")
        
        # Extract most relevant sentence
        sentences = [s.strip() for s in cleaned.split('.') if len(s.strip()) > 15]
        
        if sentences:
            # Find sentence most relevant to query
            query_words = set(original_query.lower().split())
            
            best_sentence = sentences[0]
            best_score = 0
            
            for sentence in sentences[:3]:
                sentence_words = set(sentence.lower().split())
                relevance = len(query_words.intersection(sentence_words))
                
                if relevance > best_score:
                    best_score = relevance
                    best_sentence = sentence
            
            # Keep concise for context injection
            if len(best_sentence) > 120:
                best_sentence = best_sentence[:117] + "..."
            
            return best_sentence
        
        # Fallback
        return cleaned[:100].strip() + ("..." if len(cleaned) > 100 else "")

class EnhancedDispatcherAgent(RAGEnhancedAgent):
    """Enhanced dispatcher with FIXED RAG capabilities"""
    
    def __init__(self, call_data: EnhancedCallData):
        self.call_data = call_data
        
        instructions = self._build_instructions()
        super().__init__(instructions=instructions, rag_context_prefix="roadside assistance general")
    
    def _build_instructions(self) -> str:
        base_instructions = """You are Mark, a professional roadside assistance dispatcher.

TASK: Collect customer information step by step, then route to specialists.

INFORMATION GATHERING ORDER (ONE at a time):
1. Full name - use gather_caller_information(name="John Smith")
2. Phone number - use gather_caller_information(phone="555-1234") 
3. Vehicle location - use gather_caller_information(location="123 Main St")
4. Vehicle details - use gather_caller_information(vehicle_year="2020", vehicle_make="Honda")
5. Service type - use gather_caller_information(service_needed="towing")

ROUTING DECISIONS (ONLY after ALL info is collected):
- Towing needs → Use route_to_towing_specialist()
- Battery issues → Use route_to_battery_specialist() 
- Tire problems → Use route_to_tire_specialist()

CONTEXT USAGE:
- Use relevant context from knowledge base automatically injected based on user queries
- Provide accurate information about services, pricing, and policies
- Always confirm important details

Keep responses under 25 words for phone clarity."""
        
        if self.call_data.is_returning_caller:
            context_info = f"""

🔄 RETURNING CALLER:
- Previous calls: {self.call_data.previous_calls_count}
- Phone: {self.call_data.phone_number}
- Welcome them back: "Welcome back! I see you've called us before."
"""
            base_instructions += context_info
        
        return base_instructions

    async def on_enter(self):
        """Enhanced greeting"""
        if self.call_data.is_returning_caller:
            greeting = f"Welcome back! I see you've called us before. How can I help you today?"
        else:
            greeting = "Roadside assistance, this is Mark, how can I help you today?"
        
        await self.session.generate_reply(instructions=f"Say exactly: '{greeting}'")

    @function_tool()
    async def gather_caller_information(
        self, 
        context: RunContext[EnhancedCallData],
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
        """Store caller information - same as working version"""
        
        updates = []
        if name:
            context.userdata.caller_name = name
            context.userdata.gathered_info["name"] = True
            updates.append(f"name: {name}")
            
        if phone:
            context.userdata.phone_number = phone
            context.userdata.gathered_info["phone"] = True
            updates.append(f"phone: {phone}")
            
        if location:
            context.userdata.location = location
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
        
        logger.info(f"📝 Updated caller info: {updates}")
        
        # Check completion
        gathered = context.userdata.gathered_info
        if all([gathered["name"], gathered["phone"], gathered["location"], gathered["vehicle"], gathered["service"]]):
            return "Perfect! I have all the information I need. Let me connect you with our specialist who can help you."
        else:
            missing = [key for key, value in gathered.items() if not value]
            next_questions = {
                "name": "Could you please provide your full name?",
                "phone": "Could you provide a good phone number where we can reach you?",
                "location": "What is the exact location of your vehicle? Please provide the complete address.",
                "vehicle": "Could you tell me the year, make, and model of your vehicle?",
                "service": "What type of service do you need today?"
            }
            next_missing = missing[0] if missing else None
            return next_questions.get(next_missing, "Let me get the remaining information I need.")

    @function_tool()
    async def route_to_towing_specialist(self, context: RunContext[EnhancedCallData]) -> Agent:
        """Route to towing specialist with RAG capabilities"""
        logger.info("🔄 TRANSFERRING TO TOWING SPECIALIST")
        return EnhancedTowingSpecialistAgent(context.userdata)

    @function_tool()
    async def route_to_battery_specialist(self, context: RunContext[EnhancedCallData]) -> Agent:
        """Route to battery specialist"""
        logger.info("🔄 TRANSFERRING TO BATTERY SPECIALIST")
        return EnhancedBatterySpecialistAgent(context.userdata)

    @function_tool()
    async def route_to_tire_specialist(self, context: RunContext[EnhancedCallData]) -> Agent:
        """Route to tire specialist"""
        logger.info("🔄 TRANSFERRING TO TIRE SPECIALIST")
        return EnhancedTireSpecialistAgent(context.userdata)

class EnhancedTowingSpecialistAgent(RAGEnhancedAgent):
    """RAG-powered towing specialist with FIXED patterns"""
    
    def __init__(self, customer_data: EnhancedCallData):
        self.customer_data = customer_data
        
        context_summary = f"""Customer: {customer_data.caller_name or 'Unknown'}
Phone: {customer_data.phone_number or 'Unknown'}
Location: {customer_data.location or 'Unknown'}
Vehicle: {customer_data.vehicle_year or ''} {customer_data.vehicle_make or ''} {customer_data.vehicle_model or ''}"""
        
        instructions = f"""You are a TOWING SPECIALIST for roadside assistance.

CUSTOMER INFORMATION (already collected):
{context_summary}

YOUR ROLE:
- Provide accurate quotes using knowledge base information automatically injected
- Handle special vehicle requirements based on available data
- Give realistic ETAs based on current service information
- Use context from knowledge base to answer pricing and policy questions

CRITICAL: You have access to relevant knowledge base information through context injection.
Use this information to provide accurate, specific answers about rates, policies, and services.

Keep responses professional and under 40 words for phone clarity."""
        
        super().__init__(instructions=instructions, rag_context_prefix="towing service rates")

    async def on_enter(self):
        """Enhanced greeting with context"""
        vehicle_info = f"{self.customer_data.vehicle_year or ''} {self.customer_data.vehicle_make or ''} {self.customer_data.vehicle_model or ''}".strip()
        location = self.customer_data.location or "your location"
        name = self.customer_data.caller_name or "there"
        
        if self.customer_data.is_returning_caller:
            greeting = f"Hi {name}, welcome back! I'm your towing specialist. I see you need towing for your {vehicle_info} at {location}. Where would you like it towed to?"
        else:
            greeting = f"Hi {name}, I'm your towing specialist. I see you need towing for your {vehicle_info} at {location}. Where would you like it towed to?"
        
        await self.session.generate_reply(instructions=f"Say exactly: '{greeting}'")

class EnhancedBatterySpecialistAgent(RAGEnhancedAgent):
    """RAG-powered battery specialist"""
    
    def __init__(self, customer_data: EnhancedCallData):
        self.customer_data = customer_data
        
        instructions = """You are a BATTERY SPECIALIST for roadside assistance.

Customer information has been collected. Focus on:
- Battery symptoms and diagnosis using knowledge base information
- Jump start vs replacement recommendations from available data
- Service pricing and scheduling based on current rates

Use automatically injected context to provide accurate service information."""
        
        super().__init__(instructions=instructions, rag_context_prefix="battery jumpstart service")

    async def on_enter(self):
        """Enhanced greeting"""
        name = self.customer_data.caller_name or "there"
        greeting_context = "welcome back! " if self.customer_data.is_returning_caller else ""
        
        await self.session.generate_reply(
            instructions=f"Greet: 'Hi {name}, {greeting_context}I'm your battery specialist. I have your info. What battery problems are you experiencing?'"
        )

class EnhancedTireSpecialistAgent(RAGEnhancedAgent):
    """RAG-powered tire specialist"""
    
    def __init__(self, customer_data: EnhancedCallData):
        self.customer_data = customer_data
        
        instructions = """You are a TIRE SPECIALIST for roadside assistance.

Customer information has been collected. Focus on:
- Tire damage assessment using knowledge base information
- Spare tire availability and options from service data
- Repair vs replacement recommendations based on available information

Use automatically injected context to provide accurate service details."""
        
        super().__init__(instructions=instructions, rag_context_prefix="tire service repair")

    async def on_enter(self):
        """Enhanced greeting"""
        name = self.customer_data.caller_name or "there"
        greeting_context = "welcome back! " if self.customer_data.is_returning_caller else ""
        
        await self.session.generate_reply(
            instructions=f"Greet: 'Hi {name}, {greeting_context}I'm your tire specialist. I have your info. What's the tire problem?'"
        )

async def create_enhanced_session(userdata: EnhancedCallData) -> AgentSession[EnhancedCallData]:
    """Create session with optimized configuration"""
    
    session = AgentSession[EnhancedCallData](
        stt=deepgram.STT(
            model="nova-2-general",
            language="en-US",
        ),
        
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.1,
        ),
        
        tts=elevenlabs.TTS(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            voice_settings=elevenlabs.VoiceSettings(
                stability=0.6,
                similarity_boost=0.7,
                style=0.0,
                speed=1.0
            ),
            model="eleven_turbo_v2_5",
        ),
        
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        
        allow_interruptions=True,
        min_interruption_duration=0.4,
        min_endpointing_delay=0.6,
        max_endpointing_delay=2.5,
        
        userdata=userdata
    )
    
    return session

async def identify_caller_and_restore_context(ctx: JobContext) -> EnhancedCallData:
    """Identify caller using correct ParticipantKind enum"""
    try:
        participant = await ctx.wait_for_participant()
        
        # Use correct ParticipantKind enum value
        if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            logger.warning("⚠️ No SIP participant found")
            return EnhancedCallData()
        
        # Extract caller information
        phone_number = participant.attributes.get("sip.from_number", "unknown")
        call_id = participant.attributes.get("sip.call_id", "unknown")
        
        logger.info(f"📞 Incoming call from: {phone_number}")
        
        # Start call session and check if returning caller
        session_id, caller_id = await call_storage.start_call_session(
            phone_number=phone_number,
            session_metadata={
                "call_id": call_id,
                "sip_to_number": participant.attributes.get("sip.to_number"),
                "trunk_id": participant.attributes.get("sip.trunk_id")
            }
        )
        
        # Check caller history
        caller_profile = await call_storage.get_caller_by_phone(phone_number)
        is_returning = False
        previous_calls = 0
        
        if caller_profile and caller_profile.total_calls > 1:
            is_returning = True
            previous_calls = caller_profile.total_calls - 1
            logger.info(f"🔄 Returning caller: {previous_calls} previous calls")
        else:
            logger.info("✨ New caller detected")
        
        # Create call data
        call_data = EnhancedCallData()
        call_data.session_id = session_id
        call_data.caller_id = caller_id
        call_data.phone_number = phone_number
        call_data.is_returning_caller = is_returning
        call_data.previous_calls_count = previous_calls
        
        return call_data
        
    except Exception as e:
        logger.error(f"❌ Error identifying caller: {e}")
        return EnhancedCallData()

async def entrypoint(ctx: JobContext):
    """FIXED: Enhanced entrypoint with proper RAG error handling"""
    
    logger.info("🚀 COMPLETE Enhanced Multi-Agent System with LiveKit 1.1 RAG")
    logger.info("✅ Using on_user_turn_completed for optimal RAG performance")
    
    await ctx.connect()
    
    # Initialize systems
    asyncio.create_task(qdrant_rag.initialize())
    
    # Identify caller and restore context
    call_data = await identify_caller_and_restore_context(ctx)
    
    # Create session
    session = await create_enhanced_session(call_data)
    
    # Setup transcription handlers
    transcription_handler = TranscriptionHandler(call_storage)
    transcription_handler.setup_transcription_handlers(session, call_data)
    
    # Create initial agent with RAG capabilities
    initial_agent = EnhancedDispatcherAgent(call_data)
    
    # Start session
    await session.start(
        agent=initial_agent,
        room=ctx.room
    )
    
    logger.info("✅ Enhanced system ready with LiveKit 1.1 RAG patterns")
    logger.info(f"📞 Session ID: {call_data.session_id}")
    logger.info(f"👤 Caller ID: {call_data.caller_id}")
    logger.info(f"🔄 Returning Caller: {call_data.is_returning_caller}")
    logger.info("🧠 RAG: Using on_user_turn_completed for optimal performance")

if __name__ == "__main__":
    try:
        logger.info("🎙️ Starting FIXED Enhanced Multi-Agent System")
        logger.info("🎯 Features: LiveKit 1.1 RAG + Multi-agent + Full transcription")
        logger.info("🔧 FIXED: Proper RAG error handling and context injection")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)