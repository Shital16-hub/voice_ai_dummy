# main.py - FINAL WORKING VERSION based on Official LiveKit Examples
"""
FINAL WORKING Enhanced Multi-Agent System with Call Transcription
SOLUTION: Based on official LiveKit examples from GitHub

Key Fixes:
1. Use simple AgentSession configuration like official examples
2. Fixed STT configuration based on basic_agent.py
3. Proper function tool handling and agent flow
4. Correct session start pattern from examples
5. Better conversation flow without complex validation

Based on: https://github.com/livekit/agents/blob/main/examples/voice_agents/basic_agent.py
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
    get_job_context,
    JobProcess
)
from livekit.plugins import deepgram, openai, elevenlabs, silero

# Import turn detector based on official examples
try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    TURN_DETECTOR_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Using MultilingualModel turn detection")
except ImportError:
    TURN_DETECTOR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Turn detector not available")

from qdrant_rag_system import qdrant_rag
from config import config
from call_transcription_storage import call_storage, CallTranscriptionStorage

load_dotenv()
logger = logging.getLogger(__name__)

def normalize_phone_number(phone: str) -> str:
    """Normalize phone number for consistent comparison"""
    if not phone or phone == "unknown":
        return "unknown"
    
    digits_only = ''.join(filter(str.isdigit, phone))
    
    if len(digits_only) == 11 and digits_only.startswith('1'):
        return f"+1{digits_only[1:]}"
    elif len(digits_only) == 10:
        return f"+1{digits_only}"
    elif len(digits_only) > 10:
        return f"+{digits_only}"
    else:
        return phone

@dataclass
class CallData:
    """Simple call data structure like official examples"""
    # Basic call info
    session_id: Optional[str] = None
    caller_id: Optional[str] = None
    phone_number: Optional[str] = None
    
    # Customer information
    caller_name: Optional[str] = None
    location: Optional[str] = None
    vehicle_year: Optional[str] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_color: Optional[str] = None
    service_type: Optional[str] = None
    issue_description: Optional[str] = None
    
    # Call progress
    is_returning_caller: bool = False
    previous_calls_count: int = 0
    call_stage: str = "greeting"
    
    # Information gathering progress
    gathered_info: Dict[str, bool] = field(default_factory=lambda: {
        "name": False,
        "phone": False, 
        "location": False,
        "vehicle": False,
        "service": False
    })

class TranscriptionHandler:
    """Simple transcription handler"""
    
    def __init__(self, storage: CallTranscriptionStorage):
        self.storage = storage
        
    def setup_transcription_handlers(
        self, 
        session: AgentSession, 
        call_data: CallData
    ):
        """Setup transcription handlers"""
        
        @session.on("user_speech_committed")
        def on_user_speech_committed(event):
            asyncio.create_task(self._handle_user_transcription(event, call_data))
        
        @session.on("agent_speech_committed")
        def on_agent_speech_committed(event):
            asyncio.create_task(self._handle_agent_speech(event, call_data))
        
        logger.info("✅ Transcription handlers setup completed")
    
    async def _handle_user_transcription(self, event, call_data: CallData):
        """Handle user speech transcription"""
        try:
            if call_data.session_id and call_data.caller_id:
                text = getattr(event, 'text', '') or getattr(event, 'transcript', '')
                confidence = getattr(event, 'confidence', None)
                
                if text:
                    await self.storage.save_transcription_segment(
                        session_id=call_data.session_id,
                        caller_id=call_data.caller_id,
                        speaker="user",
                        text=text,
                        is_final=True,
                        confidence=confidence
                    )
                    
                    conf_str = f" (conf: {confidence:.2f})" if confidence else ""
                    logger.info(f"🎙️ User: {text}{conf_str}")
                    
        except Exception as e:
            logger.error(f"❌ Error saving user transcription: {e}")
    
    async def _handle_agent_speech(self, event, call_data: CallData):
        """Handle agent speech"""
        try:
            if call_data.session_id and call_data.caller_id:
                text = getattr(event, 'text', '') or getattr(event, 'content', '')
                if text:
                    await self.storage.save_conversation_item(
                        session_id=call_data.session_id,
                        caller_id=call_data.caller_id,
                        role="assistant",
                        content=text,
                        metadata={
                            "call_stage": call_data.call_stage,
                            "is_returning_caller": call_data.is_returning_caller
                        }
                    )
                    logger.debug(f"🤖 Agent: {text[:100]}...")
                    
        except Exception as e:
            logger.error(f"❌ Error saving conversation item: {e}")

class DispatcherAgent(Agent):
    """Simple dispatcher agent based on official examples"""
    
    def __init__(self, call_data: CallData):
        self.call_data = call_data
        
        # Build instructions based on caller history
        instructions = self._build_instructions()
        
        super().__init__(instructions=instructions)
    
    def _build_instructions(self) -> str:
        """Build simple, clear instructions"""
        
        base_instructions = """You are Mark, a professional roadside assistance dispatcher.

GOAL: Collect customer information step by step, then route to specialists.

CONVERSATION STYLE:
- Be patient and understanding
- Speak clearly for phone calls
- Ask for ONE piece of information at a time
- If you don't understand, ask them to repeat
- Keep responses under 20 words

INFORMATION GATHERING (in order):
1. Full name
2. Phone number  
3. Vehicle location (complete address)
4. Vehicle details (year, make, model)
5. Service needed

Use gather_caller_information() to store each piece of information.

ROUTING (only after ALL info collected):
- Towing → route_to_towing_specialist()
- Battery → route_to_battery_specialist() 
- Tire → route_to_tire_specialist()"""
        
        # Add returning caller context
        if self.call_data.is_returning_caller:
            context_info = f"""

RETURNING CALLER:
- Previous calls: {self.call_data.previous_calls_count}
- Phone: {self.call_data.phone_number}
- Welcome them back warmly"""
            base_instructions += context_info
        
        return base_instructions

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
        """Store caller information simply and reliably"""
        
        updates = []
        
        # Store name
        if name:
            context.userdata.caller_name = name.strip()
            context.userdata.gathered_info["name"] = True
            updates.append(f"name: {name}")
            
        # Store phone
        if phone:
            context.userdata.phone_number = phone.strip()
            context.userdata.gathered_info["phone"] = True
            updates.append(f"phone: {phone}")
            
        # Store location
        if location:
            context.userdata.location = location.strip()
            context.userdata.gathered_info["location"] = True
            updates.append(f"location: {location}")
            
        # Store vehicle info
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
            
        # Store service info
        if issue:
            context.userdata.issue_description = issue
        if service_needed:
            context.userdata.service_type = service_needed
            context.userdata.gathered_info["service"] = True
            updates.append(f"service: {service_needed}")
        
        logger.info(f"📝 Stored info: {updates}")
        
        # Check if all information is collected
        gathered = context.userdata.gathered_info
        
        if all([gathered["name"], gathered["phone"], gathered["location"], gathered["vehicle"], gathered["service"]]):
            logger.info("✅ ALL INFORMATION COLLECTED - Ready to route")
            return "Perfect! I have all the information I need. Let me connect you to our specialist now."
        else:
            # Ask for next missing piece
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
            logger.info(f"❓ Asking for: {next_missing}")
            return question

    @function_tool()
    async def route_to_towing_specialist(self, context: RunContext[CallData]) -> Agent:
        """Route to towing specialist"""
        logger.info("🔄 ROUTING TO TOWING SPECIALIST")
        context.userdata.call_stage = "towing_specialist"
        return TowingSpecialistAgent(context.userdata)

    @function_tool()
    async def route_to_battery_specialist(self, context: RunContext[CallData]) -> Agent:
        """Route to battery specialist"""
        logger.info("🔄 ROUTING TO BATTERY SPECIALIST")
        context.userdata.call_stage = "battery_specialist"
        return BatterySpecialistAgent(context.userdata)

    @function_tool()
    async def route_to_tire_specialist(self, context: RunContext[CallData]) -> Agent:
        """Route to tire specialist"""
        logger.info("🔄 ROUTING TO TIRE SPECIALIST")
        context.userdata.call_stage = "tire_specialist"
        return TireSpecialistAgent(context.userdata)

class TowingSpecialistAgent(Agent):
    """Towing specialist agent"""
    
    def __init__(self, customer_data: CallData):
        self.customer_data = customer_data
        
        instructions = f"""You are a TOWING SPECIALIST for roadside assistance.

CUSTOMER INFO:
- Name: {customer_data.caller_name}
- Phone: {customer_data.phone_number}
- Location: {customer_data.location}
- Vehicle: {customer_data.vehicle_year or ''} {customer_data.vehicle_make or ''} {customer_data.vehicle_model or ''}

YOUR JOB:
- Ask where they want the vehicle towed
- Provide pricing and ETA
- Arrange the service

Keep responses short and clear for phone calls."""
        
        super().__init__(instructions=instructions)

    async def on_enter(self):
        """Greet customer with their information"""
        name = self.customer_data.caller_name or "there"
        location = self.customer_data.location or "your location"
        vehicle = f"{self.customer_data.vehicle_year or ''} {self.customer_data.vehicle_make or ''} {self.customer_data.vehicle_model or ''}".strip()
        
        greeting = f"Hi {name}, I'm your towing specialist. I have you at {location} with your {vehicle}. Where would you like it towed to?"
        
        await self.session.generate_reply(instructions=f"Say exactly: '{greeting}'")
        logger.info(f"🚛 Towing specialist ready for {name}")

class BatterySpecialistAgent(Agent):
    """Battery specialist agent"""
    
    def __init__(self, customer_data: CallData):
        self.customer_data = customer_data
        
        instructions = """You are a BATTERY SPECIALIST for roadside assistance.

Customer information has been collected. Focus on:
- Battery diagnosis
- Jump start vs replacement
- Service pricing"""
        
        super().__init__(instructions=instructions)

    async def on_enter(self):
        """Greet customer"""
        name = self.customer_data.caller_name or "there"
        location = self.customer_data.location or "your location"
        
        greeting = f"Hi {name}, I'm your battery specialist. I have you at {location}. What battery problems are you experiencing?"
        
        await self.session.generate_reply(instructions=f"Say exactly: '{greeting}'")
        logger.info(f"🔋 Battery specialist ready for {name}")

class TireSpecialistAgent(Agent):
    """Tire specialist agent"""
    
    def __init__(self, customer_data: CallData):
        self.customer_data = customer_data
        
        instructions = """You are a TIRE SPECIALIST for roadside assistance.

Customer information has been collected. Focus on:
- Tire damage assessment
- Spare tire options
- Repair vs replacement"""
        
        super().__init__(instructions=instructions)

    async def on_enter(self):
        """Greet customer"""
        name = self.customer_data.caller_name or "there"
        location = self.customer_data.location or "your location"
        
        greeting = f"Hi {name}, I'm your tire specialist. I have you at {location}. What's the tire problem?"
        
        await self.session.generate_reply(instructions=f"Say exactly: '{greeting}'")
        logger.info(f"🛞 Tire specialist ready for {name}")

async def identify_caller_with_history(ctx: JobContext) -> CallData:
    """Identify caller and load history"""
    try:
        participant = await ctx.wait_for_participant()
        
        if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            logger.warning("⚠️ No SIP participant found")
            return CallData()
        
        # Extract phone number
        phone_number = "unknown"
        
        phone_attributes = [
            "sip.phoneNumber",
            "sip.from_number", 
            "sip.caller_number",
            "sip.from",
            "sip.fromNumber",
            "sip.trunkPhoneNumber"
        ]
        
        for attr in phone_attributes:
            if attr in participant.attributes:
                phone_number = participant.attributes[attr]
                logger.info(f"📞 Found phone number in {attr}: {phone_number}")
                break
        
        normalized_phone = normalize_phone_number(phone_number)
        
        call_id = participant.attributes.get("sip.callIDFull", 
                  participant.attributes.get("sip.callID", "unknown"))
        
        logger.info(f"📞 Incoming call from: {phone_number} (normalized: {normalized_phone})")
        
        # Start call session
        session_id, caller_id = await call_storage.start_call_session(
            phone_number=phone_number,
            session_metadata={
                "call_id": call_id,
                "normalized_phone": normalized_phone,
                "participant_identity": participant.identity
            }
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

# Prewarm function like official examples
def prewarm(proc: JobProcess):
    """Prewarm function to load models early"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main entrypoint based on official LiveKit examples"""
    
    logger.info("🚀 WORKING Enhanced Multi-Agent System Starting")
    logger.info("📞 Based on official LiveKit examples for reliability")
    
    # Add context fields like official examples
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    await ctx.connect()
    logger.info("✅ Connected to room")
    
    # Initialize RAG with timeout
    try:
        await asyncio.wait_for(qdrant_rag.initialize(), timeout=3.0)
        logger.info("✅ RAG system initialized")
    except:
        logger.warning("⚠️ RAG system timeout - continuing without RAG")
    
    # Identify caller
    call_data = await identify_caller_with_history(ctx)
    
    # Create session like official examples
    session_params = {
        "vad": ctx.proc.userdata["vad"],  # Use prewarmed VAD
        # STT configuration based on basic_agent.py
        "stt": deepgram.STT(
            model="nova-3", 
            language="en-US"  # Fixed language instead of "multi"
        ),
        "llm": openai.LLM(model="gpt-4o-mini"),
        "tts": elevenlabs.TTS(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            voice_settings=elevenlabs.VoiceSettings(
                stability=0.7,
                similarity_boost=0.8,
                style=0.0,
                speed=0.9
            ),
            model="eleven_turbo_v2_5",
        ),
        "userdata": call_data
    }
    
    # Add turn detection if available
    if TURN_DETECTOR_AVAILABLE:
        session_params["turn_detection"] = MultilingualModel()
        logger.info("✅ Using turn detection model")
    
    session = AgentSession[CallData](**session_params)
    
    # Setup transcription handlers
    transcription_handler = TranscriptionHandler(call_storage)
    transcription_handler.setup_transcription_handlers(session, call_data)
    
    # Create initial agent
    initial_agent = DispatcherAgent(call_data)
    
    # Start session like official examples
    await session.start(
        agent=initial_agent,
        room=ctx.room
    )
    
    # Generate initial greeting like official examples
    if call_data.is_returning_caller:
        greeting_instruction = "Say: 'Welcome back! I see you've called us before. How can I help you today?'"
    else:
        greeting_instruction = "Say: 'Roadside assistance, this is Mark, how can I help you today?'"
    
    await session.generate_reply(instructions=greeting_instruction)
    
    logger.info("✅ WORKING Enhanced System Ready")
    logger.info(f"📞 Session ID: {call_data.session_id}")
    logger.info(f"👤 Caller ID: {call_data.caller_id}")
    logger.info(f"📱 Phone: {call_data.phone_number}")
    logger.info(f"🔄 Returning: {call_data.is_returning_caller} ({call_data.previous_calls_count} prev calls)")
    logger.info("🎯 Based on official LiveKit examples for maximum reliability")

if __name__ == "__main__":
    try:
        logger.info("🎙️ Starting WORKING Enhanced Multi-Agent System")
        logger.info("📞 Based on official LiveKit GitHub examples")
        logger.info("🔧 Simplified and reliable configuration")
        
        worker_options = WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,  # Add prewarm like official examples
            agent_name="my-telephony-agent"
        )
        
        cli.run_app(worker_options)
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)