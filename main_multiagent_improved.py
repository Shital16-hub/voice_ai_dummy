# main_multiagent_improved.py
"""
Enhanced Multi-Agent System with Simplified RAG
Combines the multi-agent architecture with our improved LlamaIndex-based RAG system
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
class EnhancedCallData:
    """Enhanced call data with transcription tracking - preserving all features"""
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
    transfer_requested: bool = False
    information_complete: bool = False

class TranscriptionHandler:
    """Handles transcription events with LiveKit patterns"""
    
    def __init__(self, storage):
        self.storage = storage
        
    def setup_transcription_handlers(
        self, 
        session: AgentSession, 
        call_data: EnhancedCallData
    ):
        """Setup transcription handlers"""
        
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

class ImprovedRAGAgent(Agent):
    """
    Base agent class with simplified RAG system using LlamaIndex patterns
    Much simpler than the previous over-engineered version
    """
    
    def __init__(self, instructions: str, rag_context_prefix: str = ""):
        super().__init__(instructions=instructions)
        self.rag_context_prefix = rag_context_prefix
        self.rag_cache = {}
        self.rag_processing = False
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """
        SIMPLIFIED RAG pattern using our improved LlamaIndex system
        Much cleaner than the previous over-engineered approach
        """
        try:
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 3 or self.rag_processing:
                return
            
            self.rag_processing = True
            
            try:
                # Check if this looks like a question that needs knowledge base
                if self._needs_knowledge_context(user_text):
                    # Use our simplified RAG system to get context
                    context = await self._get_rag_context(user_text)
                    
                    if context:
                        # Inject context following LiveKit pattern
                        turn_ctx.add_message(
                            role="system",
                            content=f"KNOWLEDGE BASE CONTEXT: {context}\n\nUse this information to answer the customer's question accurately."
                        )
                        logger.info(f"✅ RAG context injected for: {user_text[:50]}...")
                        
            except Exception as rag_error:
                logger.debug(f"🔍 RAG lookup failed: {rag_error}")
            finally:
                self.rag_processing = False
                
        except Exception as e:
            logger.error(f"❌ RAG processing error: {e}")
            self.rag_processing = False
    
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
    
    async def _get_rag_context(self, query: str) -> Optional[str]:
        """Get RAG context using simplified system"""
        try:
            # Check cache first
            cache_key = f"{self.rag_context_prefix}_{query.lower().strip()[:50]}"
            if cache_key in self.rag_cache:
                logger.debug("📚 Using cached RAG result")
                return self.rag_cache[cache_key]
            
            # Enhance query with context prefix
            enhanced_query = f"{self.rag_context_prefix} {query}" if self.rag_context_prefix else query
            
            # Use our simplified RAG system
            context = await simplified_rag.retrieve_context(enhanced_query, max_results=2)
            
            if context:
                # Cache successful result
                self.rag_cache[cache_key] = context
                
                # Limit cache size
                if len(self.rag_cache) > 50:
                    oldest_key = next(iter(self.rag_cache))
                    del self.rag_cache[oldest_key]
                
                logger.info("✅ RAG context retrieved successfully")
                return context
            else:
                logger.debug("🔍 No relevant context found")
                return None
                
        except Exception as e:
            logger.error(f"❌ RAG context error: {e}")
            return None

class EnhancedDispatcherAgent(ImprovedRAGAgent):
    """Enhanced dispatcher with simplified RAG capabilities"""
    
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
- General questions → Use search_knowledge_base()

KNOWLEDGE BASE USAGE:
- Relevant context from your Excel knowledge base is automatically injected
- Use this context to provide accurate information about services, pricing, and policies
- Always search knowledge base for specific questions using search_knowledge_base()

TRANSFER POLICY:
- ONLY transfer to human if customer explicitly says: "transfer me", "human agent", "speak to someone"
- Use execute_transfer_to_human() only when explicitly requested

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
        """Store caller information - preserved from original system"""
        
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
        
        # Log to call storage
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
        
        # Check completion and guide next steps
        gathered = context.userdata.gathered_info
        if all(gathered.values()):
            context.userdata.information_complete = True
            return "Perfect! I have all the information I need. Let me connect you with our specialist who can help you."
        else:
            # Guide to next needed information
            if not gathered["name"]:
                return "Could you please provide your full name?"
            elif not gathered["phone"]:
                return "Could you provide a good phone number where we can reach you?"
            elif not gathered["location"]:
                return "What is the exact location of your vehicle? Please provide the complete address."
            elif not gathered["vehicle"]:
                return "Could you tell me the year, make, and model of your vehicle?"
            elif not gathered["service"]:
                return "What type of service do you need today?"
            else:
                return "Let me get the remaining information I need."

    @function_tool()
    async def search_knowledge_base(
        self, 
        context: RunContext[EnhancedCallData],
        query: str
    ) -> str:
        """
        Search knowledge base using simplified RAG system
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

    @function_tool()
    async def execute_transfer_to_human(
        self, 
        context: RunContext[EnhancedCallData],
        confirmed: bool = True
    ) -> str:
        """Execute transfer to human - preserved from original"""
        if not confirmed:
            return "Just let me know if you'd like me to transfer you by saying 'yes, transfer me'."
        
        try:
            logger.info("🔄 EXECUTING CONFIRMED TRANSFER")
            
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

class EnhancedTowingSpecialistAgent(ImprovedRAGAgent):
    """RAG-powered towing specialist with simplified system"""
    
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
- Use search_knowledge_base() for specific towing questions

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

    @function_tool()
    async def search_knowledge_base(
        self, 
        context: RunContext[EnhancedCallData],
        query: str
    ) -> str:
        """Search knowledge base for towing-specific information"""
        try:
            logger.info(f"🔍 Towing specialist searching: {query}")
            
            # Use simplified RAG system with towing context
            enhanced_query = f"towing service {query}"
            context_text = await simplified_rag.retrieve_context(enhanced_query, max_results=2)
            
            if context_text:
                return context_text
            else:
                return "I don't have specific towing information about that. Let me get you connected with our dispatch team for detailed towing information."
                
        except Exception as e:
            logger.error(f"❌ Towing knowledge search error: {e}")
            return "I'm having trouble accessing towing information right now. Let me connect you with dispatch."

class EnhancedBatterySpecialistAgent(ImprovedRAGAgent):
    """RAG-powered battery specialist with simplified system"""
    
    def __init__(self, customer_data: EnhancedCallData):
        self.customer_data = customer_data
        
        instructions = """You are a BATTERY SPECIALIST for roadside assistance.

Customer information has been collected. Focus on:
- Battery symptoms and diagnosis using knowledge base information
- Jump start vs replacement recommendations from available data
- Service pricing and scheduling based on current rates
- Use search_knowledge_base() for specific battery questions

Use automatically injected context to provide accurate service information."""
        
        super().__init__(instructions=instructions, rag_context_prefix="battery jumpstart service")

    async def on_enter(self):
        """Enhanced greeting"""
        name = self.customer_data.caller_name or "there"
        greeting_context = "welcome back! " if self.customer_data.is_returning_caller else ""
        
        await self.session.generate_reply(
            instructions=f"Greet: 'Hi {name}, {greeting_context}I'm your battery specialist. I have your info. What battery problems are you experiencing?'"
        )

    @function_tool()
    async def search_knowledge_base(
        self, 
        context: RunContext[EnhancedCallData],
        query: str
    ) -> str:
        """Search knowledge base for battery-specific information"""
        try:
            logger.info(f"🔍 Battery specialist searching: {query}")
            
            # Use simplified RAG system with battery context
            enhanced_query = f"battery service {query}"
            context_text = await simplified_rag.retrieve_context(enhanced_query, max_results=2)
            
            if context_text:
                return context_text
            else:
                return "I don't have specific battery information about that. Let me provide general battery assistance or connect you with our technical team."
                
        except Exception as e:
            logger.error(f"❌ Battery knowledge search error: {e}")
            return "I'm having trouble accessing battery information right now. Let me help with general battery assistance."

class EnhancedTireSpecialistAgent(ImprovedRAGAgent):
    """RAG-powered tire specialist with simplified system"""
    
    def __init__(self, customer_data: EnhancedCallData):
        self.customer_data = customer_data
        
        instructions = """You are a TIRE SPECIALIST for roadside assistance.

Customer information has been collected. Focus on:
- Tire damage assessment using knowledge base information
- Spare tire availability and options from service data
- Repair vs replacement recommendations based on available information
- Use search_knowledge_base() for specific tire questions

Use automatically injected context to provide accurate service details."""
        
        super().__init__(instructions=instructions, rag_context_prefix="tire service repair")

    async def on_enter(self):
        """Enhanced greeting"""
        name = self.customer_data.caller_name or "there"
        greeting_context = "welcome back! " if self.customer_data.is_returning_caller else ""
        
        await self.session.generate_reply(
            instructions=f"Greet: 'Hi {name}, {greeting_context}I'm your tire specialist. I have your info. What's the tire problem?'"
        )

    @function_tool()
    async def search_knowledge_base(
        self, 
        context: RunContext[EnhancedCallData],
        query: str
    ) -> str:
        """Search knowledge base for tire-specific information"""
        try:
            logger.info(f"🔍 Tire specialist searching: {query}")
            
            # Use simplified RAG system with tire context
            enhanced_query = f"tire service {query}"
            context_text = await simplified_rag.retrieve_context(enhanced_query, max_results=2)
            
            if context_text:
                return context_text
            else:
                return "I don't have specific tire information about that. Let me provide general tire assistance or connect you with our technical team."
                
        except Exception as e:
            logger.error(f"❌ Tire knowledge search error: {e}")
            return "I'm having trouble accessing tire information right now. Let me help with general tire assistance."

async def create_enhanced_session(userdata: EnhancedCallData) -> AgentSession[EnhancedCallData]:
    """Create session with optimized configuration - preserved from original"""
    
    session_params = {
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
            temperature=0.1,
        ),
        
        "vad": silero.VAD.load(),
        "userdata": userdata,
        
        # Optimized timing for telephony
        "allow_interruptions": True,
        "min_interruption_duration": 0.4,
        "min_endpointing_delay": 0.6,
        "max_endpointing_delay": 2.5,
    }
    
    # Enhanced TTS setup
    try:
        session_params["tts"] = elevenlabs.TTS(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            voice_settings=elevenlabs.VoiceSettings(
                stability=0.6,
                similarity_boost=0.7,
                style=0.0,
                speed=1.0
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
        logger.info("✅ Using semantic turn detection")
    
    session = AgentSession[EnhancedCallData](**session_params)
    return session

async def identify_caller_and_restore_context(ctx: JobContext) -> EnhancedCallData:
    """Identify caller - preserved from original with correct enum usage"""
    try:
        participant = await ctx.wait_for_participant()
        
        if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            logger.warning("⚠️ No SIP participant found")
            return EnhancedCallData()
        
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

def prewarm(proc: JobProcess):
    """Prewarm function to load models early"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """
    Enhanced Multi-Agent entrypoint with simplified RAG system
    Combines multi-agent architecture with our improved LlamaIndex-based RAG
    """
    
    logger.info("🚀 ENHANCED MULTI-AGENT SYSTEM with SIMPLIFIED RAG")
    logger.info("🔧 Using LlamaIndex-based RAG + Multi-agent architecture")
    
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
            logger.error("💡 Multi-agent system will work but without knowledge base")
            
    except Exception as e:
        logger.error(f"❌ RAG initialization error: {e}")
        success = False
    
    # Identify caller and restore context
    call_data = await identify_caller_and_restore_context(ctx)
    
    # Create enhanced session
    session = await create_enhanced_session(call_data)
    
    # Setup transcription handlers
    transcription_handler = TranscriptionHandler(call_storage)
    transcription_handler.setup_transcription_handlers(session, call_data)
    
    # Create initial dispatcher agent with simplified RAG capabilities
    initial_agent = EnhancedDispatcherAgent(call_data)
    
    # Start session
    await session.start(
        agent=initial_agent,
        room=ctx.room
    )
    
    # Log final status
    logger.info("✅ ENHANCED MULTI-AGENT SYSTEM READY")
    logger.info(f"📞 Session ID: {call_data.session_id}")
    logger.info(f"👤 Caller ID: {call_data.caller_id}")
    logger.info(f"📱 Phone: {call_data.phone_number}")
    logger.info(f"🔄 Returning: {call_data.is_returning_caller}")
    logger.info(f"📊 RAG System: {'✅ Active' if success else '⚠️ Disabled'}")
    logger.info("🚫 Auto-transfer: DISABLED (only on explicit request)")
    logger.info("✅ Enhanced STT with better transcription")
    logger.info("✅ Simplified RAG with LlamaIndex patterns")
    logger.info("🎯 Multi-agent routing: Dispatcher → Specialists")
    logger.info("📝 Full call transcription and history tracking")

if __name__ == "__main__":
    try:
        logger.info("🎙️ Starting ENHANCED MULTI-AGENT SYSTEM")
        logger.info("📊 Features: Multi-agent + Simplified RAG + Full transcription")
        logger.info("🔧 Using simplified LlamaIndex-based RAG system")
        logger.info("🎯 Agent flow: Dispatcher → Towing/Battery/Tire Specialists")
        
        worker_options = WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="my-telephony-agent"
        )
        
        cli.run_app(worker_options)
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)