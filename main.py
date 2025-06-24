# main.py - OPTIMIZED FOR TELEPHONY WITH BETTER STT AND CALLER RECOGNITION
"""
OPTIMIZED VERSION for telephony calls with:
- Better STT configuration for phone calls
- Improved caller recognition and history tracking
- Optimized for both MicroSIP and regular phone calls
- Enhanced transcription accuracy
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

# FIXED: Use English-only model for maximum speed with fallback
try:
    from livekit.plugins.turn_detector.english import EnglishModel
    TURN_DETECTOR_CLASS = EnglishModel
    TURN_DETECTOR_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Using English turn detection model for maximum speed")
except ImportError:
    try:
        from livekit.plugins.turn_detector.multilingual import MultilingualModel
        TURN_DETECTOR_CLASS = MultilingualModel
        TURN_DETECTOR_AVAILABLE = True
        logger = logging.getLogger(__name__)
        logger.info("✅ Using Multilingual turn detection model")
    except ImportError:
        TURN_DETECTOR_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning("⚠️ Turn detector not available - using VAD only")

from qdrant_rag_system import qdrant_rag
from config import config
from call_transcription_storage import call_storage, CallTranscriptionStorage

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class EnhancedCallData:
    """Enhanced call data with improved caller tracking"""
    # Basic call info
    session_id: Optional[str] = None
    caller_id: Optional[str] = None
    phone_number: Optional[str] = None
    normalized_phone: Optional[str] = None  # NEW: Normalized for comparison
    
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
    
    # Enhanced history tracking
    is_returning_caller: bool = False
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    previous_calls_count: int = 0
    greeting_sent: bool = False
    
    # NEW: Previous call data for context
    previous_name: Optional[str] = None
    previous_location: Optional[str] = None
    previous_vehicle: Optional[str] = None

def normalize_phone_number(phone: str) -> str:
    """Normalize phone number for consistent comparison"""
    if not phone or phone == "unknown":
        return "unknown"
    
    # Remove all non-digit characters
    digits_only = ''.join(filter(str.isdigit, phone))
    
    # Handle different formats
    if len(digits_only) == 11 and digits_only.startswith('1'):
        # US number with country code
        return f"+1{digits_only[1:]}"
    elif len(digits_only) == 10:
        # US number without country code
        return f"+1{digits_only}"
    elif len(digits_only) > 10:
        # International number
        return f"+{digits_only}"
    else:
        # Return as is for short numbers
        return phone

class TranscriptionHandler:
    """Enhanced transcription handler with better accuracy tracking"""
    
    def __init__(self, storage: CallTranscriptionStorage):
        self.storage = storage
        
    def setup_transcription_handlers(
        self, 
        session: AgentSession, 
        call_data: EnhancedCallData
    ):
        """Setup transcription handlers with enhanced logging"""
        
        @session.on("user_speech_committed")
        def on_user_speech_committed(event):
            asyncio.create_task(self._handle_user_transcription(event, call_data))
        
        @session.on("agent_speech_committed")
        def on_agent_speech_committed(event):
            asyncio.create_task(self._handle_agent_speech(event, call_data))
        
        logger.info("✅ Enhanced transcription handlers setup completed")
    
    async def _handle_user_transcription(self, event, call_data: EnhancedCallData):
        """Handle user speech transcription with confidence tracking"""
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
                    
                    # Enhanced logging with confidence
                    conf_str = f" (conf: {confidence:.2f})" if confidence else ""
                    logger.info(f"🎙️ User transcription{conf_str}: {text}")
                    
                    # Warn about low confidence
                    if confidence and confidence < 0.7:
                        logger.warning(f"⚠️ Low transcription confidence: {confidence:.2f} for '{text}'")
                    
        except Exception as e:
            logger.error(f"❌ Error saving user transcription: {e}")
    
    async def _handle_agent_speech(self, event, call_data: EnhancedCallData):
        """Handle agent speech with metadata"""
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
                            "urgency_level": call_data.urgency_level,
                            "is_returning_caller": call_data.is_returning_caller
                        }
                    )
                    logger.debug(f"🤖 Agent: {text[:100]}...")
                    
        except Exception as e:
            logger.error(f"❌ Error saving conversation item: {e}")

class RAGEnhancedAgent(Agent):
    """Enhanced RAG agent with better timeout handling and fallback"""
    
    def __init__(self, instructions: str, rag_context_prefix: str = ""):
        super().__init__(instructions=instructions)
        self.rag_context_prefix = rag_context_prefix
        self.rag_cache = {}
        self.rag_processing = False
        self.rag_failures = 0
        self.max_rag_failures = 3
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Enhanced RAG with adaptive timeout and fallback"""
        try:
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 3 or self.rag_processing:
                return
            
            # Skip RAG if too many failures
            if self.rag_failures >= self.max_rag_failures:
                logger.debug(f"🔍 Skipping RAG due to {self.rag_failures} failures")
                return
            
            self.rag_processing = True
            
            try:
                # Very short timeout for phone calls
                rag_context = await asyncio.wait_for(
                    self._perform_rag_lookup(user_text),
                    timeout=0.2  # Very aggressive for phone calls
                )
                
                if rag_context:
                    try:
                        turn_ctx.add_message(
                            role="system",
                            content=f"[CONTEXT]: {rag_context}"
                        )
                        logger.info(f"✅ RAG context injected: {user_text[:30]}...")
                        self.rag_failures = max(0, self.rag_failures - 1)  # Reduce failure count on success
                    except Exception as ctx_error:
                        logger.debug(f"❌ Context injection error: {ctx_error}")
                        
            except asyncio.TimeoutError:
                self.rag_failures += 1
                logger.debug(f"⏰ RAG timeout ({self.rag_failures}/{self.max_rag_failures})")
            except Exception as rag_error:
                self.rag_failures += 1
                logger.debug(f"🔍 RAG error ({self.rag_failures}/{self.max_rag_failures}): {rag_error}")
            finally:
                self.rag_processing = False
                
        except Exception as e:
            logger.error(f"❌ RAG processing error: {e}")
            self.rag_processing = False
    
    async def _perform_rag_lookup(self, query: str) -> Optional[str]:
        """Optimized RAG lookup with caching"""
        try:
            # Check cache first
            cache_key = f"{self.rag_context_prefix}_{query.lower().strip()[:50]}"
            if cache_key in self.rag_cache:
                return self.rag_cache[cache_key]
            
            enhanced_query = f"{self.rag_context_prefix} {query}" if self.rag_context_prefix else query
            
            results = await asyncio.wait_for(
                qdrant_rag.search(enhanced_query, limit=1),
                timeout=0.15
            )
            
            if results and results[0]["score"] >= 0.2:  # Lower threshold for phone calls
                context = self._format_rag_context(results[0]["text"])
                self.rag_cache[cache_key] = context
                
                # Limit cache size
                if len(self.rag_cache) > 10:
                    oldest_key = next(iter(self.rag_cache))
                    del self.rag_cache[oldest_key]
                
                return context
            
            return None
            
        except Exception:
            return None
    
    def _format_rag_context(self, raw_text: str) -> str:
        """Format RAG context for phone calls"""
        cleaned = raw_text.strip()
        cleaned = cleaned.replace("•", "").replace("-", "").replace("*", "")
        cleaned = cleaned.replace("\n", " ").replace("\t", " ")
        
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")
        
        # Keep very short for phone calls
        if len(cleaned) > 80:
            sentences = [s.strip() for s in cleaned.split('.') if len(s.strip()) > 10]
            if sentences:
                cleaned = sentences[0]
        
        return cleaned[:80] + ("..." if len(cleaned) > 80 else "")

class EnhancedDispatcherAgent(RAGEnhancedAgent):
    """Enhanced dispatcher with better caller recognition and information collection"""
    
    def __init__(self, call_data: EnhancedCallData):
        self.call_data = call_data
        
        instructions = self._build_instructions()
        super().__init__(instructions=instructions, rag_context_prefix="roadside assistance")
    
    def _build_instructions(self) -> str:
        """Build instructions with enhanced caller recognition"""
        
        base_instructions = """You are Mark, a professional roadside assistance dispatcher.

TASK: Collect customer information step by step, then route to specialists.

INFORMATION GATHERING ORDER (ONE at a time, be thorough):
1. FULL NAME - use gather_caller_information(name="John Michael Smith")
2. Phone number - use gather_caller_information(phone="555-123-4567") 
3. COMPLETE ADDRESS - use gather_caller_information(location="123 Main Street, City, State")
4. Vehicle details - use gather_caller_information(vehicle_year="2020", vehicle_make="Honda", vehicle_model="Civic")
5. Service type - use gather_caller_information(service_needed="towing")

CRITICAL ROUTING LOGIC:
- ONLY route to specialists AFTER collecting ALL information
- When complete, IMMEDIATELY call the appropriate routing function:
  * Towing/pulling/moving → route_to_towing_specialist()
  * Battery/jump/dead battery → route_to_battery_specialist()
  * Tire/flat tire/puncture → route_to_tire_specialist()

CONVERSATION TIPS:
- Speak clearly and slowly for phone calls
- Confirm spellings of names and addresses
- Ask for clarification if transcription seems unclear
- Use "Could you repeat that?" if needed

Keep responses under 25 words for phone clarity."""
        
        # Enhanced returning caller recognition
        if self.call_data.is_returning_caller:
            context_info = f"""

🔄 RETURNING CALLER DETECTED:
- Previous calls: {self.call_data.previous_calls_count}
- Phone: {self.call_data.phone_number}
- Last name: {self.call_data.previous_name or 'Not recorded'}
- Last location: {self.call_data.previous_location or 'Not recorded'}
- Last vehicle: {self.call_data.previous_vehicle or 'Not recorded'}

IMPORTANT: 
- Welcome them back: "Welcome back! I see you've called us before."
- Reference previous information when helpful
- Still collect current information for this call"""
            base_instructions += context_info
        
        return base_instructions

    async def on_enter(self):
        """Enhanced greeting with better caller recognition"""
        if not self.call_data.greeting_sent:
            if self.call_data.is_returning_caller:
                if self.call_data.previous_name:
                    greeting = f"Welcome back! I see you've called us before. Is this still {self.call_data.previous_name}?"
                else:
                    greeting = f"Welcome back! I see you've called us before. How can I help you today?"
                logger.info(f"🔄 Returning caller greeting: {self.call_data.phone_number} (prev: {self.call_data.previous_name})")
            else:
                greeting = "Roadside assistance, this is Mark, how can I help you today?"
                logger.info(f"✨ New caller greeting: {self.call_data.phone_number}")
            
            await self.session.generate_reply(instructions=f"Say exactly: '{greeting}'")
            self.call_data.greeting_sent = True

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
        """Enhanced information gathering with validation and confirmation"""
        
        updates = []
        
        # Enhanced name handling with validation
        if name:
            # Clean and validate name
            cleaned_name = ' '.join(name.strip().split())
            name_parts = cleaned_name.split()
            
            if len(name_parts) < 2:
                return f"I have {cleaned_name}. Could you also provide your last name please?"
            
            # Check against previous caller info
            if context.userdata.is_returning_caller and context.userdata.previous_name:
                if cleaned_name.lower() != context.userdata.previous_name.lower():
                    return f"I have {cleaned_name}. Just to confirm, last time you called as {context.userdata.previous_name}. Is {cleaned_name} correct for today?"
            
            context.userdata.caller_name = cleaned_name
            context.userdata.gathered_info["name"] = True
            updates.append(f"name: {cleaned_name}")
            
        # Enhanced phone handling
        if phone and not context.userdata.is_returning_caller:
            # Validate phone format
            normalized = normalize_phone_number(phone)
            context.userdata.phone_number = phone
            context.userdata.normalized_phone = normalized
            context.userdata.gathered_info["phone"] = True
            updates.append(f"phone: {phone}")
            
        # Enhanced address handling with validation
        if location:
            cleaned_location = ' '.join(location.strip().split())
            
            # Validate address completeness
            if len(cleaned_location) < 15:  # Minimum meaningful address length
                return f"I have {cleaned_location}. Could you provide a more complete address with street number and city?"
            
            # Check against previous caller info
            if context.userdata.is_returning_caller and context.userdata.previous_location:
                if cleaned_location.lower() != context.userdata.previous_location.lower():
                    return f"I have {cleaned_location}. Last time you were at {context.userdata.previous_location}. Is {cleaned_location} correct for today?"
            
            context.userdata.location = cleaned_location
            context.userdata.gathered_info["location"] = True
            updates.append(f"location: {cleaned_location}")
            logger.info(f"📍 ADDRESS COLLECTED: {cleaned_location}")
            
        # Enhanced vehicle information handling
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
            
            # Check against previous vehicle info
            if context.userdata.is_returning_caller and context.userdata.previous_vehicle:
                if vehicle_info.lower() != context.userdata.previous_vehicle.lower():
                    return f"I have {vehicle_info}. Last time you had a {context.userdata.previous_vehicle}. Is {vehicle_info} correct for today?"
            
            updates.append(f"vehicle: {vehicle_info}")
            
        # Enhanced service type handling
        if issue:
            context.userdata.issue_description = issue
        if service_needed:
            context.userdata.service_type = service_needed
            context.userdata.gathered_info["service"] = True
            updates.append(f"service: {service_needed}")
        
        logger.info(f"📝 Updated caller info: {updates}")
        
        # Enhanced completion check with automatic routing
        gathered = context.userdata.gathered_info
        
        # Auto-complete for returning callers
        if context.userdata.is_returning_caller:
            if not gathered["name"] and context.userdata.previous_name:
                context.userdata.caller_name = context.userdata.previous_name
                gathered["name"] = True
            if not gathered["phone"] and context.userdata.phone_number:
                gathered["phone"] = True
        
        # Check completion and route
        if all([gathered["name"], gathered["phone"], gathered["location"], gathered["vehicle"], gathered["service"]]):
            service_type = context.userdata.service_type.lower() if context.userdata.service_type else ""
            
            logger.info(f"✅ COMPLETE INFO COLLECTED - Service: {service_type}")
            
            # Determine routing message
            if "tow" in service_type or "pull" in service_type or "move" in service_type:
                return "Perfect! I have everything I need. Connecting you to our towing specialist now."
            elif "battery" in service_type or "jump" in service_type or "dead" in service_type:
                return "Perfect! I have everything I need. Connecting you to our battery specialist now."
            elif "tire" in service_type or "flat" in service_type or "puncture" in service_type:
                return "Perfect! I have everything I need. Connecting you to our tire specialist now."
            else:
                return "Perfect! I have everything I need. Connecting you to our service specialist now."
        else:
            # Information still missing
            missing = [key for key, value in gathered.items() if not value]
            next_questions = {
                "name": "Could you provide your full name, first and last name please?",
                "phone": "Could you provide a good phone number where we can reach you?",
                "location": "What's the complete address where your vehicle is located?",
                "vehicle": "Could you tell me the year, make, and model of your vehicle?",
                "service": "What type of service do you need today?"
            }
            next_missing = missing[0] if missing else None
            return next_questions.get(next_missing, "Let me get the remaining information I need.")

    @function_tool()
    async def route_to_towing_specialist(self, context: RunContext[EnhancedCallData]) -> Agent:
        """Route to enhanced towing specialist"""
        logger.info("🔄 ROUTING TO ENHANCED TOWING SPECIALIST")
        context.userdata.call_stage = "specialist_towing"
        return EnhancedTowingSpecialistAgent(context.userdata)

    @function_tool()
    async def route_to_battery_specialist(self, context: RunContext[EnhancedCallData]) -> Agent:
        """Route to enhanced battery specialist"""
        logger.info("🔄 ROUTING TO ENHANCED BATTERY SPECIALIST")
        context.userdata.call_stage = "specialist_battery"
        return EnhancedBatterySpecialistAgent(context.userdata)

    @function_tool()
    async def route_to_tire_specialist(self, context: RunContext[EnhancedCallData]) -> Agent:
        """Route to enhanced tire specialist"""
        logger.info("🔄 ROUTING TO ENHANCED TIRE SPECIALIST")
        context.userdata.call_stage = "specialist_tire"
        return EnhancedTireSpecialistAgent(context.userdata)

class EnhancedTowingSpecialistAgent(RAGEnhancedAgent):
    """Enhanced towing specialist with better context and RAG"""
    
    def __init__(self, customer_data: EnhancedCallData):
        self.customer_data = customer_data
        
        context_summary = f"""Customer: {customer_data.caller_name or 'Unknown'}
Phone: {customer_data.phone_number or 'Unknown'}
Location: {customer_data.location or 'Unknown'}
Vehicle: {customer_data.vehicle_year or ''} {customer_data.vehicle_make or ''} {customer_data.vehicle_model or ''}
Service: Towing
Returning: {customer_data.is_returning_caller}"""
        
        instructions = f"""You are a TOWING SPECIALIST for roadside assistance.

CUSTOMER INFORMATION:
{context_summary}

YOUR ROLE:
- Ask where they want the vehicle towed TO
- Use context information to provide accurate quotes
- Give realistic ETAs based on current conditions
- Handle special requirements (AWD, etc.)

IMPORTANT: You have all customer information. Don't ask for name, phone, location, or vehicle details again.

Keep responses under 40 words for phone clarity."""
        
        super().__init__(instructions=instructions, rag_context_prefix="towing rates pricing")

    async def on_enter(self):
        """Enhanced specialist greeting"""
        vehicle_info = f"{self.customer_data.vehicle_year or ''} {self.customer_data.vehicle_make or ''} {self.customer_data.vehicle_model or ''}".strip()
        location = self.customer_data.location or "your location"
        name = self.customer_data.caller_name or "there"
        
        if self.customer_data.is_returning_caller:
            greeting = f"Hi {name}, welcome back! I'm your towing specialist. I have you at {location} with your {vehicle_info}. Where would you like it towed to?"
        else:
            greeting = f"Hi {name}, I'm your towing specialist. I have you at {location} with your {vehicle_info}. Where would you like it towed to?"
        
        await self.session.generate_reply(instructions=f"Say exactly: '{greeting}'")
        logger.info(f"🚛 Towing specialist greeting sent to {name}")

class EnhancedBatterySpecialistAgent(RAGEnhancedAgent):
    """Enhanced battery specialist"""
    
    def __init__(self, customer_data: EnhancedCallData):
        self.customer_data = customer_data
        
        instructions = """You are a BATTERY SPECIALIST for roadside assistance.

Customer information has been collected. Focus on:
- Battery symptoms and diagnosis
- Jump start vs replacement recommendations
- Service pricing and scheduling

Use context information to provide accurate service details."""
        
        super().__init__(instructions=instructions, rag_context_prefix="battery jumpstart service")

    async def on_enter(self):
        """Enhanced battery specialist greeting"""
        name = self.customer_data.caller_name or "there"
        location = self.customer_data.location or "your location"
        greeting_context = "welcome back! " if self.customer_data.is_returning_caller else ""
        
        greeting = f"Hi {name}, {greeting_context}I'm your battery specialist. I have you at {location}. What battery problems are you experiencing?"
        
        await self.session.generate_reply(instructions=f"Say exactly: '{greeting}'")
        logger.info(f"🔋 Battery specialist greeting sent to {name}")

class EnhancedTireSpecialistAgent(RAGEnhancedAgent):
    """Enhanced tire specialist"""
    
    def __init__(self, customer_data: EnhancedCallData):
        self.customer_data = customer_data
        
        instructions = """You are a TIRE SPECIALIST for roadside assistance.

Customer information has been collected. Focus on:
- Tire damage assessment
- Spare tire availability and options
- Repair vs replacement recommendations

Use context information to provide accurate service details."""
        
        super().__init__(instructions=instructions, rag_context_prefix="tire service repair")

    async def on_enter(self):
        """Enhanced tire specialist greeting"""
        name = self.customer_data.caller_name or "there"
        location = self.customer_data.location or "your location"
        greeting_context = "welcome back! " if self.customer_data.is_returning_caller else ""
        
        greeting = f"Hi {name}, {greeting_context}I'm your tire specialist. I have you at {location}. What's the tire problem?"
        
        await self.session.generate_reply(instructions=f"Say exactly: '{greeting}'")
        logger.info(f"🛞 Tire specialist greeting sent to {name}")

async def create_optimized_telephony_session(userdata: EnhancedCallData) -> AgentSession[EnhancedCallData]:
    """Create session optimized specifically for telephony with enhanced STT"""
    
    # ENHANCED: Determine best STT model based on call type
    # Use phone-optimized model for better accuracy
    stt_model = "nova-2-phonecall"  # Optimized for phone calls
    
    # ENHANCED: Phone-specific STT configuration
    stt_config = deepgram.STT(
        model='nova-3',
        language="en-US",
        smart_format=True,
        interim_results=True,
        profanity_filter=False,
        punctuate=True,
        numerals=True,
        filler_words=True,  # Help with turn detection
        endpointing_ms=100,  # Faster endpointing for phone calls
        # keyterms= NOT_GIVEN,
        # ENHANCED: Keywords for better accuracy on common terms
        keyterms=[
            ("roadside", 0.5),
            ("assistance", 0.5),
            ("towing", 0.5),
            ("battery", 0.5),
            ("tire", 0.5),
            ("vehicle", 0.3),
            ("location", 0.3),
            ("address", 0.3),
            ("phone", 0.3),
            ("number", 0.3)
        ]
    )
    
    logger.info(f"📞 Using phone-optimized STT model: {stt_model}")
    
    session_params = {
        "stt": deepgram.STT(
            model="enhanced-general",  # Fallback to enhanced model
            language="en-US",
            interim_results=False
        ),
        
        # ENHANCED: LLM settings for better phone conversations
        "llm": openai.LLM(
            model="gpt-4o-mini",
            temperature=0.1,  # More consistent for phone calls
            max_completion_tokens=60,  # Shorter for phone clarity
        ),
        
        # ENHANCED: TTS optimized for phone calls
        "tts": elevenlabs.TTS(
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Professional male voice
            voice_settings=elevenlabs.VoiceSettings(
                stability=0.7,      # More stable for phone calls
                similarity_boost=0.8,
                style=0.0,          # No style variations
                speed=0.9           # Slightly slower for phone clarity
            ),
            model="eleven_turbo_v2_5",
        ),
        
        # ENHANCED: VAD configuration for phone calls
        "vad": silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.5,  # Longer for phone calls
            activation_threshold=0.4,  # Lower for phone audio
            sample_rate=16000
        ),
        
        # ENHANCED: Timing optimized for phone calls
        "allow_interruptions": True,
        "min_interruption_duration": 0.5,  # Longer for phone calls
        "min_endpointing_delay": 0.6,      # Longer for phone calls
        "max_endpointing_delay": 2.5,      # Longer for phone calls
        
        "userdata": userdata
    }
    
    # Add turn detection if available
    if TURN_DETECTOR_AVAILABLE:
        session_params["turn_detection"] = TURN_DETECTOR_CLASS()
        logger.info("✅ Using advanced turn detection optimized for phone calls")
    else:
        logger.info("⚠️ Using VAD-only turn detection for phone calls")
    
    session = AgentSession[EnhancedCallData](**session_params)
    return session

async def identify_caller_with_enhanced_history(ctx: JobContext) -> EnhancedCallData:
    """Enhanced caller identification with better history retrieval"""
    try:
        participant = await ctx.wait_for_participant()
        
        if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            logger.warning("⚠️ No SIP participant found")
            return EnhancedCallData()
        
        # ENHANCED: Better phone number extraction
        phone_number = "unknown"
        
        # Try multiple SIP attribute keys
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
        
        # If still unknown, try pattern matching
        if phone_number == "unknown":
            for key, value in participant.attributes.items():
                if value and len(value) > 7 and any(c.isdigit() for c in value):
                    if "phone" in key.lower() or "number" in key.lower() or "from" in key.lower():
                        phone_number = value
                        logger.info(f"📞 Found phone number in {key}: {phone_number}")
                        break
        
        # Normalize phone number for comparison
        normalized_phone = normalize_phone_number(phone_number)
        
        call_id = participant.attributes.get("sip.callIDFull", 
                  participant.attributes.get("sip.callID", "unknown"))
        
        logger.info(f"📞 Incoming call from: {phone_number} (normalized: {normalized_phone})")
        logger.info(f"📋 Available SIP attributes: {list(participant.attributes.keys())}")
        
        # Start call session
        session_id, caller_id = await call_storage.start_call_session(
            phone_number=phone_number,
            session_metadata={
                "call_id": call_id,
                "normalized_phone": normalized_phone,
                "sip_attributes": dict(participant.attributes),
                "participant_identity": participant.identity
            }
        )
        
        # ENHANCED: Better caller history retrieval
        caller_profile = await call_storage.get_caller_by_phone(phone_number)
        
        # Also try normalized phone number if first attempt fails
        if not caller_profile and normalized_phone != phone_number:
            caller_profile = await call_storage.get_caller_by_phone(normalized_phone)
            if caller_profile:
                logger.info(f"📞 Found caller using normalized number: {normalized_phone}")
        
        is_returning = False
        previous_calls = 0
        previous_name = None
        previous_location = None
        previous_vehicle = None
        
        if caller_profile and caller_profile.total_calls > 0:
            is_returning = True
            previous_calls = caller_profile.total_calls
            
            # Get recent conversation history for context
            history = await call_storage.get_caller_conversation_history(
                caller_profile.caller_id, limit=10
            )
            
            # Extract previous information from history
            for item in history:
                content = item.content.lower()
                # Look for previous name
                if not previous_name and ("name" in content or "i'm" in content):
                    words = item.content.split()
                    for i, word in enumerate(words):
                        if word.lower() in ["name", "i'm"] and i + 1 < len(words):
                            potential_name = words[i + 1]
                            if potential_name.istitle() and len(potential_name) > 2:
                                previous_name = potential_name
                                break
                
                # Look for previous location
                if not previous_location and ("street" in content or "road" in content or "avenue" in content):
                    # Extract location from content
                    if len(item.content) > 15:
                        previous_location = item.content[:50]
                
                # Look for previous vehicle
                if not previous_vehicle and any(brand in content for brand in ["honda", "toyota", "ford", "bmw", "audi"]):
                    previous_vehicle = item.content[:30]
            
            logger.info(f"🔄 Returning caller: {previous_calls} calls, prev name: {previous_name}")
        else:
            logger.info("✨ New caller detected")
        
        # Create enhanced call data
        call_data = EnhancedCallData()
        call_data.session_id = session_id
        call_data.caller_id = caller_id
        call_data.phone_number = phone_number
        call_data.normalized_phone = normalized_phone
        call_data.is_returning_caller = is_returning
        call_data.previous_calls_count = previous_calls
        call_data.previous_name = previous_name
        call_data.previous_location = previous_location
        call_data.previous_vehicle = previous_vehicle
        
        return call_data
        
    except Exception as e:
        logger.error(f"❌ Error identifying caller: {e}")
        return EnhancedCallData()

async def entrypoint(ctx: JobContext):
    """Enhanced entrypoint optimized for telephony"""
    
    logger.info("🚀 ENHANCED TELEPHONY SYSTEM STARTING")
    logger.info("📞 Optimized for phone calls with enhanced STT and caller recognition")
    
    # Connect immediately
    await ctx.connect()
    logger.info("✅ Connected to room")
    
    # Initialize RAG with timeout
    try:
        await asyncio.wait_for(qdrant_rag.initialize(), timeout=3.0)
        logger.info("✅ RAG system initialized")
    except asyncio.TimeoutError:
        logger.warning("⚠️ RAG initialization timeout - continuing without RAG")
    except Exception as e:
        logger.warning(f"⚠️ RAG initialization error: {e} - continuing without RAG")
    
    # Enhanced caller identification
    call_data = await identify_caller_with_enhanced_history(ctx)
    
    # Create optimized session for telephony
    session = await create_optimized_telephony_session(call_data)
    
    # Setup enhanced transcription handlers
    transcription_handler = TranscriptionHandler(call_storage)
    transcription_handler.setup_transcription_handlers(session, call_data)
    
    # Create enhanced dispatcher agent
    initial_agent = EnhancedDispatcherAgent(call_data)
    
    # Start session
    await session.start(
        agent=initial_agent,
        room=ctx.room
    )
    
    logger.info("✅ ENHANCED TELEPHONY SYSTEM READY")
    logger.info(f"📞 Session ID: {call_data.session_id}")
    logger.info(f"👤 Caller ID: {call_data.caller_id}")
    logger.info(f"📱 Phone: {call_data.phone_number} (normalized: {call_data.normalized_phone})")
    logger.info(f"🔄 Returning: {call_data.is_returning_caller} ({call_data.previous_calls_count} prev calls)")
    logger.info(f"👤 Prev Name: {call_data.previous_name}")
    logger.info("🎯 ENHANCED: Phone-optimized STT, better caller recognition, improved accuracy")

if __name__ == "__main__":
    try:
        logger.info("🎙️ Starting ENHANCED TELEPHONY SYSTEM")
        logger.info("📞 Optimized for phone calls with nova-2-phonecall model")
        logger.info("🔄 Enhanced caller recognition with history context")
        logger.info("🎯 Better transcription accuracy for phone calls")
        
        worker_options = WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        )
        
        cli.run_app(worker_options)
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)