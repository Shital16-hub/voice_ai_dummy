# main_improved.py - SIMPLIFIED WITH LLAMAINDEX RAG
"""
Improved main.py using simplified RAG system based on LiveKit examples
Preserves all your existing features but with much simpler RAG implementation
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
    JobProcess,
    llm
)
from livekit.plugins import deepgram, openai, elevenlabs, silero

# Turn detector import
try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    TURN_DETECTOR_AVAILABLE = True
except ImportError:
    TURN_DETECTOR_AVAILABLE = False

# Import systems - using simplified RAG
from simple_rag_v2 import simplified_rag
from config import config
from call_transcription_storage import call_storage

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class CallData:
    """Enhanced call data structure - preserved from your original"""
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
    transfer_requested: bool = False
    information_complete: bool = False
    gathered_info: Dict[str, bool] = field(default_factory=lambda: {
        "name": False, "phone": False, "location": False, 
        "vehicle": False, "service": False
    })

class ImprovedRAGAgent(Agent):
    """
    Improved agent using simplified RAG system
    Following LiveKit patterns but preserving all your features
    """
    
    def __init__(self, call_data: CallData):
        self.call_data = call_data
        self.conversation_turns = 0
        
        instructions = self._build_instructions()
        super().__init__(instructions=instructions)
    
    def _build_instructions(self) -> str:
        """Build instructions - preserved from your original with RAG context"""
        base_instructions = """You are Mark, a professional roadside assistance operator.

CRITICAL: You provide roadside assistance services directly. You ARE the service provider.

YOUR KNOWLEDGE BASE CONTAINS:
- Towing services: Standard Sedan ($75), SUV/Truck ($120), Motorcycle ($60), Long Distance ($150), etc.
- Battery services: Jump-Start ($40), Battery Replacement ($150+), Battery Testing ($30)
- Tire services: Flat Tire Change ($50), Tire Repair ($25), Tire Inflation ($20)
- Fuel services: Fuel Delivery ($65), Wrong Fuel ($125)
- Lockout services: Car Lockout ($55), Key Replacement ($45)

CONVERSATION RULES:
- ALWAYS use search_knowledge_base() for pricing and service questions
- NEVER auto-transfer unless customer explicitly asks for "human agent" or "transfer me"
- Keep responses under 40 words for phone clarity
- Be helpful and provide exact pricing from knowledge base

INFORMATION GATHERING ORDER:
1. Customer's full name
2. Phone number for callback  
3. Exact vehicle location (complete address)
4. Vehicle details (year, make, model)
5. Service needed and provide exact pricing

TRANSFER POLICY:
- ONLY transfer if customer explicitly says: "transfer me", "human agent", "speak to someone"
- ALWAYS provide information first from knowledge base
- Ask for confirmation before transferring

Use search_knowledge_base() for ANY questions about services, pricing, or company information."""

        if self.call_data.is_returning_caller:
            base_instructions += f"""

RETURNING CALLER:
- Previous calls: {self.call_data.previous_calls_count}
- Welcome them back professionally
"""
        
        return base_instructions
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """
        SIMPLIFIED context injection using LlamaIndex retrieval pattern
        Much simpler than the previous over-engineered version
        """
        try:
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 2:
                return
            
            self.conversation_turns += 1
            
            # Skip for explicit transfers
            transfer_phrases = ["transfer me", "human agent", "speak to a person"]
            if any(phrase in user_text.lower() for phrase in transfer_phrases):
                return
            
            # Check if this looks like a question that needs knowledge base
            if self._needs_knowledge_context(user_text):
                # Use simplified RAG to get context
                context = await simplified_rag.retrieve_context(user_text, max_results=2)
                
                if context:
                    # Inject context following LiveKit pattern
                    turn_ctx.add_message(
                        role="system",
                        content=f"KNOWLEDGE BASE CONTEXT: {context}\n\nUse this information to answer the customer's question accurately."
                    )
                    logger.info(f"✅ Context injected for: {user_text[:50]}...")
            
            # Add conversation guidance - preserved from your original
            self._inject_conversation_guidance(turn_ctx, user_text)
                    
        except Exception as e:
            logger.error(f"❌ Error in context injection: {e}")
    
    def _needs_knowledge_context(self, user_text: str) -> bool:
        """Determine if we need knowledge base context - simplified logic"""
        user_lower = user_text.lower()
        
        # Service-related keywords
        service_keywords = [
            "cost", "price", "how much", "fee", "charge", "rate", "pricing",
            "service", "services", "towing", "battery", "tire", "jumpstart",
            "fuel", "gas", "lockout", "locked", "flat", "dead", "replacement",
            "what do you", "do you offer", "what services", "tell me", "provide",
            "available", "help", "assist", "need", "problem", "issue",
            "hours", "business", "company", "contact", "phone",
            "membership", "plan", "plans", "coverage"
        ]
        
        # Skip simple responses
        simple_responses = ["yes", "no", "okay", "ok", "hello", "hi", "thanks"]
        if len(user_text.split()) <= 2 and any(simple in user_lower for simple in simple_responses):
            return False
        
        return any(keyword in user_lower for keyword in service_keywords)
    
    def _inject_conversation_guidance(self, turn_ctx: ChatContext, user_text: str):
        """Inject conversation flow guidance - preserved from your original"""
        guidance_parts = []
        
        # Check for explicit transfer requests
        transfer_keywords = ["transfer", "human", "person", "someone else", "specialist", "agent"]
        if any(keyword in user_text.lower() for keyword in transfer_keywords):
            self.call_data.transfer_requested = True
            guidance_parts.append("TRANSFER REQUEST: Customer explicitly asked for transfer")
        
        # Check information completeness
        gathered = self.call_data.gathered_info
        if not all(gathered.values()):
            missing = [key for key, value in gathered.items() if not value]
            guidance_parts.append(f"INFO NEEDED: Still need {', '.join(missing)}")
        else:
            self.call_data.information_complete = True
            guidance_parts.append("INFO COMPLETE: All customer information collected")
        
        if guidance_parts:
            turn_ctx.add_message(
                role="system",
                content=f"CONVERSATION STATUS: {' | '.join(guidance_parts)}"
            )

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
        """Enhanced information gathering - preserved from your original"""
        
        updates = []
        
        # Process each field with validation
        if name:
            context.userdata.caller_name = name.strip()
            context.userdata.gathered_info["name"] = True
            updates.append(f"name: {name}")
            
        if phone:
            # Clean phone number
            clean_phone = ''.join(filter(str.isdigit, phone))
            if len(clean_phone) >= 10:
                context.userdata.phone_number = clean_phone
                context.userdata.gathered_info["phone"] = True
                updates.append(f"phone: {clean_phone}")
            
        if location:
            context.userdata.location = location.strip()
            context.userdata.gathered_info["location"] = True
            updates.append(f"location: {location}")
        
        # Vehicle information
        vehicle_updated = False
        if vehicle_year:
            context.userdata.vehicle_year = vehicle_year.strip()
            vehicle_updated = True
        if vehicle_make:
            context.userdata.vehicle_make = vehicle_make.strip()
            vehicle_updated = True
        if vehicle_model:
            context.userdata.vehicle_model = vehicle_model.strip()
            vehicle_updated = True
        if vehicle_color:
            context.userdata.vehicle_color = vehicle_color.strip()
            vehicle_updated = True
            
        if vehicle_updated:
            context.userdata.gathered_info["vehicle"] = True
            vehicle_info = f"{context.userdata.vehicle_year or ''} {context.userdata.vehicle_make or ''} {context.userdata.vehicle_model or ''} {context.userdata.vehicle_color or ''}".strip()
            updates.append(f"vehicle: {vehicle_info}")
        
        # Service information
        if issue:
            context.userdata.issue_description = issue.strip()
        if service_needed:
            context.userdata.service_type = service_needed.strip()
            context.userdata.gathered_info["service"] = True
            updates.append(f"service: {service_needed}")
        
        # Log to call storage - preserved from your original
        if updates and context.userdata.session_id:
            try:
                await call_storage.save_conversation_item(
                    session_id=context.userdata.session_id,
                    caller_id=context.userdata.caller_id,
                    role="agent",
                    content=f"Information recorded: {', '.join(updates)}",
                    metadata={"type": "information_gathering"}
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to log conversation: {e}")
        
        logger.info(f"📝 Information updated: {updates}")
        
        # Check completeness and guide next steps
        gathered = context.userdata.gathered_info
        if all(gathered.values()):
            context.userdata.information_complete = True
            return "Perfect! I have all your information. Let me search for the exact service details and pricing from our system."
        else:
            # Guide to next needed information
            if not gathered["name"]:
                return "Could you tell me your full name please?"
            elif not gathered["phone"]:
                return "What's a good phone number to reach you?"
            elif not gathered["location"]:
                return "Where is your vehicle located? Please give me the complete address."
            elif not gathered["vehicle"]:
                return "What's the year, make, and model of your vehicle?"
            elif not gathered["service"]:
                return "What type of service do you need today?"
            else:
                return "Let me get a bit more information to help you better."

    @function_tool()
    async def search_knowledge_base(
        self, 
        context: RunContext[CallData],
        query: str
    ) -> str:
        """
        SIMPLIFIED knowledge base search using LlamaIndex patterns
        Much simpler than the previous over-engineered version
        """
        try:
            logger.info(f"🔍 Knowledge base search: {query}")
            
            # Use simplified RAG system
            context_text = await simplified_rag.retrieve_context(query, max_results=3)
            
            if context_text:
                logger.info("📊 Knowledge base search successful")
                return context_text
            else:
                logger.warning("⚠️ No relevant information found in knowledge base")
                return "I don't have specific information about that in my knowledge base right now. Let me help you with what I can, or would you like me to connect you with a specialist who can provide detailed information?"
                
        except Exception as e:
            logger.error(f"❌ Knowledge base search error: {e}")
            return "I'm having trouble accessing our information system right now. Let me try to help you directly, or I can connect you with someone who can assist."

    @function_tool()
    async def request_transfer_to_human(
        self, 
        context: RunContext[CallData],
        reason: str = "Customer requested human assistance"
    ) -> str:
        """Request transfer - preserved from your original"""
        logger.info(f"💬 Transfer requested: {reason}")
        
        context.userdata.transfer_requested = True
        customer_name = context.userdata.caller_name or "there"
        
        return f"I understand you'd like to speak with someone else, {customer_name}. Would you like me to connect you with one of our specialists? Just say 'yes, transfer me' to confirm."

    @function_tool()
    async def execute_transfer_to_human(
        self, 
        context: RunContext[CallData],
        confirmed: bool = True
    ) -> str:
        """Execute transfer - preserved from your original"""
        if not confirmed:
            return "Just let me know if you'd like me to transfer you by saying 'yes, transfer me'."
        
        try:
            logger.info("🔄 EXECUTING CONFIRMED TRANSFER")
            
            from livekit.agents import get_job_context
            job_ctx = get_job_context()
            
            # Find SIP participant
            sip_participant = None
            for participant in job_ctx.room.remote_participants.values():
                if str(participant.kind) == "3":
                    sip_participant = participant
                    break
            
            if not sip_participant:
                return "I'm having trouble with the transfer right now. Let me continue helping you directly."
            
            # Prepare handoff information
            data = context.userdata
            handoff_info = {
                "caller_name": data.caller_name,
                "phone_number": data.phone_number,
                "location": data.location,
                "vehicle": f"{data.vehicle_year or ''} {data.vehicle_make or ''} {data.vehicle_model or ''}".strip(),
                "service_type": data.service_type,
                "issue": data.issue_description
            }
            
            # Log transfer
            if data.session_id:
                try:
                    await call_storage.save_conversation_item(
                        session_id=data.session_id,
                        caller_id=data.caller_id,
                        role="agent",
                        content="Transferring to human agent - customer confirmed",
                        metadata={"type": "confirmed_transfer", "handoff_info": handoff_info}
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log transfer: {e}")
            
            logger.info(f"🔄 Transfer details: {handoff_info}")
            
            # Execute transfer
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=sip_participant.identity,
                transfer_to=config.transfer_sip_address,
                play_dialtone=True,
            )
            
            await job_ctx.api.sip.transfer_sip_participant(transfer_request)
            
            return f"Connecting you to a specialist now, {data.caller_name or 'there'}. Please hold."
            
        except Exception as e:
            logger.error(f"❌ Transfer failed: {e}")
            return "I'm having trouble with the transfer. Let me continue helping you with your service request."

async def identify_caller_with_history(ctx: JobContext) -> CallData:
    """Enhanced caller identification - preserved from your original"""
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
        
        # Create enhanced call data
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
    """Prewarm function - preserved from your original"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """
    IMPROVED entrypoint with simplified RAG system
    Preserves all your features but uses much simpler RAG implementation
    """
    
    logger.info("🚀 IMPROVED RAG Voice Agent Starting")
    logger.info("🔧 Using simplified LlamaIndex-based RAG system")
    
    await ctx.connect()
    logger.info("✅ Connected to room")
    
    # Initialize simplified RAG system
    try:
        rag_start = time.time()
        logger.info("🔧 Initializing simplified RAG system...")
        
        success = await simplified_rag.initialize()
        rag_time = (time.time() - rag_start) * 1000
        
        if success:
            status = await simplified_rag.get_status()
            logger.info(f"✅ Simplified RAG ready in {rag_time:.1f}ms")
            logger.info(f"📊 RAG status: {status}")
        else:
            logger.error("❌ CRITICAL: RAG system failed to initialize!")
            logger.error("💡 Check: docker-compose up -d")
            logger.error("💡 Check: OpenAI API key")
            logger.error("💡 Agent will work but without knowledge base")
            
    except Exception as e:
        logger.error(f"❌ RAG initialization error: {e}")
        success = False
    
    # Identify caller and load history - preserved from your original
    call_data = await identify_caller_with_history(ctx)
    
    # Create session with enhanced STT - preserved from your original
    session_params = {
        "vad": ctx.proc.userdata["vad"],
        
        # Enhanced STT configuration
        "stt": deepgram.STT(
            model="nova-3",
            language="en-US",
            smart_format=True,
            punctuate=True,
            profanity_filter=False,
            numerals=True,
            interim_results=True,
        ),
        
        "llm": openai.LLM(
            model="gpt-4o-mini", 
            temperature=0.1
        ),
        "userdata": call_data
    }
    
    # Enhanced TTS setup - preserved from your original
    try:
        session_params["tts"] = elevenlabs.TTS(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            voice_settings=elevenlabs.VoiceSettings(
                stability=0.7, 
                similarity_boost=0.8, 
                style=0.0, 
                speed=0.9
            ),
            model="eleven_turbo_v2_5",
        )
        logger.info("✅ Using ElevenLabs TTS")
    except Exception as e:
        logger.warning(f"⚠️ ElevenLabs TTS failed, using OpenAI: {e}")
        session_params["tts"] = openai.TTS(voice="alloy")
    
    # Add turn detection if available - preserved from your original
    if TURN_DETECTOR_AVAILABLE:
        session_params["turn_detection"] = MultilingualModel()
        logger.info("✅ Using semantic turn detection")
    
    # Create session
    session = AgentSession[CallData](**session_params)
    
    # Create improved RAG agent
    improved_agent = ImprovedRAGAgent(call_data)
    
    # Start session
    await session.start(
        agent=improved_agent,
        room=ctx.room
    )
    
    # Generate contextual greeting - preserved from your original
    if call_data.is_returning_caller:
        greeting = "Say: 'Welcome back! I see you've called us before. How can I help you today?'"
    else:
        greeting = "Say: 'Roadside assistance, this is Mark, how can I help you today?'"
    
    await session.generate_reply(instructions=greeting)
    
    # Log final status with improved configuration
    logger.info("✅ IMPROVED RAG AGENT READY")
    logger.info(f"📞 Session ID: {call_data.session_id}")
    logger.info(f"👤 Caller ID: {call_data.caller_id}")
    logger.info(f"📱 Phone: {call_data.phone_number}")
    logger.info(f"🔄 Returning: {call_data.is_returning_caller}")
    logger.info(f"📊 RAG System: {'✅ Active' if success else '⚠️ Disabled'}")
    logger.info("🚫 Auto-transfer: DISABLED (only on explicit request)")
    logger.info("✅ Enhanced STT with better transcription")
    logger.info("✅ Simplified RAG with LlamaIndex patterns")

if __name__ == "__main__":
    try:
        logger.info("🎙️ Starting IMPROVED RAG Voice Agent")
        logger.info("📊 Using simplified LlamaIndex-based RAG system")
        logger.info("🔧 Following LiveKit RAG patterns for reliability")
        
        worker_options = WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="my-telephony-agent"
        )
        
        cli.run_app(worker_options)
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)