# enhanced_multi_agent_with_transcription_FIXED.py
"""
FIXED Enhanced Multi-Agent System with Call Transcription
Based on working patterns from fixed_multi_agent_orchestrator.py
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
    """Enhanced call data with transcription tracking - SAME STRUCTURE AS WORKING VERSION"""
    # Basic call info
    session_id: Optional[str] = None
    caller_id: Optional[str] = None
    phone_number: Optional[str] = None
    
    # Caller information (SAME AS WORKING VERSION)
    caller_name: Optional[str] = None
    location: Optional[str] = None
    vehicle_year: Optional[str] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_color: Optional[str] = None
    service_type: Optional[str] = None
    issue_description: Optional[str] = None
    urgency_level: str = "normal"
    
    # Call progress tracking (SAME AS WORKING VERSION)
    call_stage: str = "greeting"
    gathered_info: Dict[str, bool] = field(default_factory=lambda: {
        "name": False,
        "phone": False, 
        "location": False,
        "vehicle": False,
        "service": False
    })
    
    # NEW: History tracking for transcription
    is_returning_caller: bool = False
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    previous_calls_count: int = 0

class TranscriptionHandler:
    """FIXED: Handles transcription events using correct LiveKit 1.1 patterns"""
    
    def __init__(self, storage: CallTranscriptionStorage):
        self.storage = storage
        
    def setup_transcription_handlers(
        self, 
        session: AgentSession, 
        call_data: EnhancedCallData
    ):
        """FIXED: Setup transcription handlers using correct LiveKit 1.1 syntax"""
        
        # FIXED: Use correct event handler syntax for LiveKit 1.1
        @session.on("user_input_transcribed")
        def on_user_transcribed(event):
            """Handle user speech transcription"""
            # Create task for async operation
            asyncio.create_task(self._handle_user_transcription(event, call_data))
        
        @session.on("conversation_item_added")
        def on_conversation_item_added(event):
            """Handle complete conversation turns"""
            # Create task for async operation
            asyncio.create_task(self._handle_conversation_item(event, call_data))
        
        logger.info("✅ FIXED transcription handlers setup")
    
    async def _handle_user_transcription(self, event, call_data: EnhancedCallData):
        """Handle user speech transcription (async)"""
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
        """Handle complete conversation turns (async)"""
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

class EnhancedDispatcherAgent(Agent):
    """FIXED: Dispatcher agent using working patterns from fixed_multi_agent_orchestrator.py"""
    
    def __init__(self, call_data: EnhancedCallData):
        self.call_data = call_data
        
        # Build context-aware instructions
        instructions = self._build_instructions()
        super().__init__(instructions=instructions)
    
    def _build_instructions(self) -> str:
        """Build instructions based on caller history - SAME PATTERN AS WORKING VERSION"""
        base_instructions = """You are Mark, a professional roadside assistance dispatcher.

TASK: Collect customer information step by step, then route to specialists.

INFORMATION GATHERING ORDER (ONE at a time using gather_caller_information):
1. Full name - use gather_caller_information(name="John Smith")
2. Phone number - use gather_caller_information(phone="555-1234") 
3. Vehicle location - use gather_caller_information(location="123 Main St")
4. Vehicle details - use gather_caller_information(vehicle_year="2020", vehicle_make="Honda")
5. Service type - use gather_caller_information(service_needed="towing")

ROUTING DECISIONS (ONLY after ALL info is collected):
- Towing needs → Use route_to_towing_specialist()
- Battery issues → Use route_to_battery_specialist() 
- Tire problems → Use route_to_tire_specialist()

CONVERSATION STYLE:
- Ask for ONE piece of information at a time
- Be empathetic: "I'm sorry to hear about that"
- Confirm details: "Just to confirm, you said..."
- Keep responses under 25 words for phone clarity

CRITICAL: Only call routing functions AFTER you have collected all required information."""
        
        # Add caller context if returning caller
        if self.call_data.is_returning_caller:
            context_info = f"""

🔄 RETURNING CALLER DETECTED:
- Previous calls: {self.call_data.previous_calls_count}
- Phone number: {self.call_data.phone_number}
- Welcome them back warmly: "Welcome back! I see you've called us before."
- Reference their history when appropriate
"""
            base_instructions += context_info
        
        return base_instructions

    async def on_enter(self):
        """Called when this agent becomes active - SAME PATTERN AS WORKING VERSION"""
        if self.call_data.is_returning_caller:
            greeting = f"Welcome back! I see you've called us before. How can I help you today?"
        else:
            greeting = "Roadside assistance, this is Mark, how can I help you today?"
        
        await self.session.generate_reply(
            instructions=f"Say exactly: '{greeting}'"
        )

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
        """SAME FUNCTION AS WORKING VERSION"""
        
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
        
        # Check completion and provide next step - SAME LOGIC AS WORKING VERSION
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
        """FIXED: Route to towing specialist - MUST return Agent instance"""
        logger.info("🔄 TRANSFERRING TO TOWING SPECIALIST")
        return TowingSpecialistAgent(context.userdata)

    @function_tool()
    async def route_to_battery_specialist(self, context: RunContext[EnhancedCallData]) -> Agent:
        """Route to battery specialist"""
        logger.info("🔄 TRANSFERRING TO BATTERY SPECIALIST")
        return BatterySpecialistAgent(context.userdata)

    @function_tool()
    async def route_to_tire_specialist(self, context: RunContext[EnhancedCallData]) -> Agent:
        """Route to tire specialist"""
        logger.info("🔄 TRANSFERRING TO TIRE SPECIALIST")
        return TireSpecialistAgent(context.userdata)

class TowingSpecialistAgent(Agent):
    """SAME AS WORKING VERSION"""
    
    def __init__(self, customer_data: EnhancedCallData):
        self.customer_data = customer_data
        
        context_summary = f"""Customer: {customer_data.caller_name or 'Unknown'}
Phone: {customer_data.phone_number or 'Unknown'}
Location: {customer_data.location or 'Unknown'}
Vehicle: {customer_data.vehicle_year or ''} {customer_data.vehicle_make or ''} {customer_data.vehicle_model or ''}"""
        
        super().__init__(
            instructions=f"""You are a TOWING SPECIALIST for roadside assistance.

CUSTOMER INFORMATION (already collected by dispatcher):
{context_summary}

CRITICAL: DO NOT ask for name, phone, location, or vehicle info again! You already have it.

YOUR ROLE:
- Assess towing destination requirements  
- Provide distance-based pricing quotes
- Handle special vehicle needs (AWD, low clearance, etc.)
- Arrange service scheduling and ETA

Use search_knowledge for current towing rates and policies.
Keep responses professional and under 30 words for phone clarity."""
        )

    async def on_enter(self):
        """Greet with existing context"""
        vehicle_info = f"{self.customer_data.vehicle_year or ''} {self.customer_data.vehicle_make or ''} {self.customer_data.vehicle_model or ''}".strip()
        location = self.customer_data.location or "your location"
        name = self.customer_data.caller_name or "there"
        
        if self.customer_data.is_returning_caller:
            greeting = f"Hi {name}, welcome back! I'm your towing specialist. I see you need towing for your {vehicle_info} at {location}. Where would you like it towed to?"
        else:
            greeting = f"Hi {name}, I'm your towing specialist. I see you need towing for your {vehicle_info} at {location}. Where would you like it towed to?"
        
        await self.session.generate_reply(instructions=f"Say exactly: '{greeting}'")

    @function_tool()
    async def search_knowledge(self, context: RunContext[EnhancedCallData], query: str) -> str:
        """Search towing knowledge base with improved response"""
        try:
            enhanced_query = f"towing service rates pricing {query}"
            results = await asyncio.wait_for(qdrant_rag.search(enhanced_query, limit=1), timeout=0.5)
            if results and results[0]["score"] >= 0.2:
                return results[0]["text"][:120]
            
            # Provide helpful default towing information
            return "For a 10km tow, our standard rate is $75 hookup plus $3.50 per mile. Total would be approximately $96. ETA is 30-45 minutes. Would you like me to arrange this service?"
        except Exception:
            return "For local towing up to 10km, our rate is typically $75 hookup plus mileage. Shall I arrange the service?"

class BatterySpecialistAgent(Agent):
    """SAME PATTERN AS WORKING VERSION"""
    
    def __init__(self, customer_data: EnhancedCallData):
        self.customer_data = customer_data
        
        super().__init__(
            instructions="""You are a BATTERY SPECIALIST for roadside assistance.

The customer information has already been collected. Focus on:
- Battery symptoms and diagnosis  
- Jump start vs replacement recommendations
- Service pricing and scheduling"""
        )

    async def on_enter(self):
        """Greet with context"""
        name = self.customer_data.caller_name or "there"
        greeting_context = "welcome back! " if self.customer_data.is_returning_caller else ""
        
        await self.session.generate_reply(
            instructions=f"Greet the customer: 'Hi {name}, {greeting_context}I'm your battery specialist. I have your info. What battery problems are you experiencing?'"
        )

class TireSpecialistAgent(Agent):
    """SAME PATTERN AS WORKING VERSION"""
    
    def __init__(self, customer_data: EnhancedCallData):
        self.customer_data = customer_data
        
        super().__init__(
            instructions="""You are a TIRE SPECIALIST for roadside assistance.

The customer information has already been collected. Focus on:
- Tire damage assessment
- Spare tire availability
- Repair vs replacement options"""
        )

    async def on_enter(self):
        """Greet with context"""
        name = self.customer_data.caller_name or "there"
        greeting_context = "welcome back! " if self.customer_data.is_returning_caller else ""
        
        await self.session.generate_reply(
            instructions=f"Greet the customer: 'Hi {name}, {greeting_context}I'm your tire specialist. I have your info. What's the tire problem?'"
        )

async def create_enhanced_session(userdata: EnhancedCallData) -> AgentSession[EnhancedCallData]:
    """SAME SESSION CREATION AS WORKING VERSION"""
    
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
    """FIXED: Identify caller using correct ParticipantKind enum"""
    try:
        # Wait for participant
        participant = await ctx.wait_for_participant()
        
        # FIXED: Use correct ParticipantKind enum value
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
    """FIXED: Entrypoint using working patterns"""
    
    logger.info("🚀 FIXED Enhanced Multi-Agent System with Call Transcription")
    logger.info("✅ Using working patterns from fixed_multi_agent_orchestrator.py")
    
    await ctx.connect()
    
    # Initialize systems - SAME AS WORKING VERSION
    asyncio.create_task(qdrant_rag.initialize())
    
    # Identify caller and restore context
    call_data = await identify_caller_and_restore_context(ctx)
    
    # Create session - SAME PATTERN AS WORKING VERSION
    session = await create_enhanced_session(call_data)
    
    # FIXED: Setup transcription handlers
    transcription_handler = TranscriptionHandler(call_storage)
    transcription_handler.setup_transcription_handlers(session, call_data)
    
    # Create initial agent with caller context
    initial_agent = EnhancedDispatcherAgent(call_data)
    
    # Start session - SAME AS WORKING VERSION
    await session.start(
        agent=initial_agent,
        room=ctx.room
    )
    
    logger.info("✅ Enhanced multi-agent system ready with WORKING patterns + transcription")
    logger.info(f"📞 Session ID: {call_data.session_id}")
    logger.info(f"👤 Caller ID: {call_data.caller_id}")
    logger.info(f"🔄 Returning Caller: {call_data.is_returning_caller}")

if __name__ == "__main__":
    try:
        logger.info("🎙️ Starting FIXED Enhanced Multi-Agent System with Transcription")
        logger.info("🎯 Features: Working agent handoffs + full call transcription + caller recognition")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)