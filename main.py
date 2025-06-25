# main.py - FINAL FIXED VERSION
"""
FINAL FIXED VERSION - Addresses all issues from logs:
1. Better STT configuration 
2. Fixed RAG search patterns
3. Proper config usage
4. Enhanced error handling
Based on LiveKit examples and your specific issues
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

# Turn detector import
try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    TURN_DETECTOR_AVAILABLE = True
except ImportError:
    TURN_DETECTOR_AVAILABLE = False

# Import systems
from simple_rag_system import simple_rag
from config import config
from call_transcription_storage import call_storage

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class CallData:
    """Enhanced call data structure"""
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

class FinalFixedRAGAgent(Agent):
    """
    FINAL FIXED Agent - addresses all issues from logs
    """
    
    def __init__(self, call_data: CallData):
        self.call_data = call_data
        self.rag_context_cache = {}
        self.conversation_turns = 0
        self.last_rag_lookup = 0
        
        instructions = self._build_fixed_instructions()
        super().__init__(instructions=instructions)
    
    def _build_fixed_instructions(self) -> str:
        """Build fixed instructions based on your Excel data"""
        base_instructions = """You are Mark, a professional roadside assistance operator.

CRITICAL: You provide roadside assistance services directly. You ARE the service provider.

YOUR EXCEL KNOWLEDGE BASE CONTAINS:
- Towing services: Standard Sedan ($75), SUV/Truck ($120), Motorcycle ($60), Long Distance ($150), etc.
- Battery services: Jump-Start ($40), Battery Replacement ($150+), Battery Testing ($30)
- Tire services: Flat Tire Change ($50), Tire Repair ($25), Tire Inflation ($20)
- Fuel services: Fuel Delivery ($65), Wrong Fuel ($125)
- Lockout services: Car Lockout ($55), Key Replacement ($45)

CONVERSATION RULES:
- ALWAYS use search_knowledge_base() for pricing and service questions
- NEVER auto-transfer unless customer explicitly asks for "human agent" or "transfer me"
- Keep responses under 40 words for phone clarity
- Be helpful and provide exact pricing from Excel data

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
        FIXED context injection with better search patterns
        """
        try:
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 2:
                return
            
            self.conversation_turns += 1
            
            # Rate limiting with shorter intervals
            current_time = time.time()
            if current_time - self.last_rag_lookup < 0.3:  # Reduced for better responsiveness
                return
            
            # Enhanced RAG trigger logic
            if self._should_inject_rag_context(user_text):
                self.last_rag_lookup = current_time
                
                rag_context = await self._get_enhanced_rag_context(user_text)
                if rag_context:
                    # Inject context in proper format
                    turn_ctx.add_message(
                        role="system",
                        content=f"KNOWLEDGE BASE CONTEXT (from your Excel roadside assistance data):\n{rag_context}\n\nUse this EXACT information to answer the customer's question. This is from your official pricing and service database."
                    )
                    logger.info(f"✅ RAG context injected for: {user_text[:50]}...")
                else:
                    logger.warning(f"⚠️ No RAG context found for: {user_text[:50]}...")
            
            # Add conversation guidance
            self._inject_conversation_guidance(turn_ctx, user_text)
                    
        except Exception as e:
            logger.error(f"❌ Error in context injection: {e}")
    
    def _should_inject_rag_context(self, user_text: str) -> bool:
        """Enhanced logic to determine RAG injection"""
        user_lower = user_text.lower()
        
        # Always inject for service-related queries
        service_triggers = [
            # Pricing keywords
            "cost", "price", "how much", "fee", "charge", "rate", "pricing", "rates",
            # Service keywords  
            "service", "services", "towing", "battery", "tire", "jumpstart", "jump start",
            "fuel", "gas", "lockout", "locked", "flat", "dead", "replacement",
            # Question patterns
            "what do you", "do you offer", "what services", "tell me", "provide",
            "available", "help", "assist", "need", "problem", "issue",
            # Company info
            "hours", "business", "company", "contact", "phone",
            # Membership
            "membership", "plan", "plans", "coverage"
        ]
        
        # Don't inject for simple responses (unless asking for info)
        simple_responses = ["yes", "no", "okay", "ok", "hello", "hi", "thanks", "thank you"]
        
        # Skip if it's just a simple response
        if len(user_text.split()) <= 2 and any(simple in user_lower for simple in simple_responses):
            return False
        
        return any(trigger in user_lower for trigger in service_triggers)
    
    async def _get_enhanced_rag_context(self, query: str) -> Optional[str]:
        """FIXED RAG context retrieval"""
        try:
            # Check cache first
            cache_key = query.lower().strip()[:80]
            if cache_key in self.rag_context_cache:
                logger.debug("📚 Using RAG cache")
                return self.rag_context_cache[cache_key]
            
            # Enhanced query processing for your Excel data
            search_queries = self._create_targeted_queries(query)
            
            all_results = []
            for search_query in search_queries[:3]:  # Try top 3 variants
                try:
                    logger.debug(f"Searching with query: {search_query}")
                    results = await asyncio.wait_for(
                        simple_rag.search(search_query, limit=config.search_limit),
                        timeout=config.rag_timeout_ms / 1000.0
                    )
                    if results:
                        all_results.extend(results)
                        logger.debug(f"Found {len(results)} results for: {search_query}")
                    else:
                        logger.debug(f"No results for: {search_query}")
                except Exception as e:
                    logger.debug(f"Search failed for '{search_query}': {e}")
                    continue
            
            if not all_results:
                logger.warning("⚠️ No RAG results found for any query variant")
                return None
            
            # Process results using FIXED thresholds
            context = self._process_rag_results_fixed(all_results)
            
            # Cache successful results
            if context:
                self.rag_context_cache[cache_key] = context
                if len(self.rag_context_cache) > 100:  # Larger cache
                    oldest_key = next(iter(self.rag_context_cache))
                    del self.rag_context_cache[oldest_key]
            
            return context
            
        except Exception as e:
            logger.error(f"❌ RAG context error: {e}")
            return None
    
    def _create_targeted_queries(self, query: str) -> List[str]:
        """Create targeted queries for your Excel data"""
        queries = [query]
        query_lower = query.lower()
        
        # Map to your exact Excel content
        if "battery" in query_lower or "jump" in query_lower or "dead" in query_lower:
            queries.extend([
                "battery jump start",
                "battery replacement", 
                "battery testing",
                "dead battery assistance"
            ])
        elif "tire" in query_lower or "flat" in query_lower:
            queries.extend([
                "flat tire change",
                "tire repair",
                "tire inflation",
                "spare tire"
            ])
        elif "tow" in query_lower:
            queries.extend([
                "towing standard sedan",
                "towing SUV truck",
                "long distance towing",
                "emergency towing"
            ])
        elif "fuel" in query_lower or "gas" in query_lower:
            queries.extend([
                "fuel delivery",
                "wrong fuel",
                "emergency gas"
            ])
        elif "lock" in query_lower:
            queries.extend([
                "car lockout",
                "key replacement",
                "vehicle entry"
            ])
        elif "service" in query_lower and ("what" in query_lower or "which" in query_lower):
            queries.extend([
                "roadside assistance services",
                "available services",
                "service options"
            ])
        elif "cost" in query_lower or "price" in query_lower:
            queries.extend([
                "service pricing",
                "rates charges",
                "cost fees"
            ])
        
        return queries
    
    def _process_rag_results_fixed(self, results: List[Dict]) -> Optional[str]:
        """FIXED result processing using proper thresholds"""
        if not results:
            return None
        
        # Remove duplicates and sort by score
        seen_texts = set()
        unique_results = []
        for result in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
            text = result.get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_results.append(result)
                if len(unique_results) >= 5:  # Keep more results
                    break
        
        if not unique_results:
            return None
        
        best_result = unique_results[0]
        best_score = best_result.get("score", 0)
        
        logger.info(f"📊 Best RAG score: {best_score:.3f}")
        
        # Use FIXED thresholds from config
        if best_score >= config.high_confidence_threshold:
            # High confidence - use single best result
            return self._format_excel_content(best_result["text"])
        
        elif best_score >= config.medium_confidence_threshold:
            # Medium confidence - combine top results
            combined_parts = []
            for result in unique_results[:3]:  # Top 3
                if result.get("score", 0) >= config.medium_confidence_threshold:
                    formatted = self._format_excel_content(result["text"])
                    if formatted and len(formatted) > 20:
                        combined_parts.append(formatted)
            
            if combined_parts:
                return " | ".join(combined_parts)
            else:
                return self._format_excel_content(best_result["text"])
        
        elif best_score >= config.minimum_usable_threshold:
            # Lower confidence but still usable
            return self._format_excel_content(best_result["text"])
        
        else:
            # Score too low
            logger.warning(f"⚠️ RAG score too low: {best_score:.3f} < {config.minimum_usable_threshold}")
            return None
    
    def _format_excel_content(self, raw_text: str) -> str:
        """Format Excel content for voice conversation"""
        if not raw_text:
            return ""
        
        # Clean up the text
        cleaned = raw_text.strip()
        
        # Remove formatting characters
        for char in ["•", "-", "*", "\n", "\t"]:
            cleaned = cleaned.replace(char, " ")
        
        # Remove multiple spaces
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")
        
        # Optimize for voice - prioritize pricing information
        if len(cleaned) > config.max_response_length:
            sentences = cleaned.split(".")
            
            # Prioritize sentences with pricing and service information
            priority_sentences = []
            other_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in ["$", "cost", "price", "service", "fee"]):
                    priority_sentences.append(sentence)
                elif len(sentence) > 15:
                    other_sentences.append(sentence)
            
            # Build result with priority content first
            result_parts = priority_sentences[:2] + other_sentences[:1]  # More priority content
            result = ". ".join(result_parts)
            if result and not result.endswith("."):
                result += "."
        else:
            result = cleaned
        
        return result.strip()
    
    def _inject_conversation_guidance(self, turn_ctx: ChatContext, user_text: str):
        """Inject conversation flow guidance"""
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
        
        # Add guidance if any
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
        """Enhanced information gathering with better validation"""
        
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
        """FIXED knowledge base search with proper error handling"""
        try:
            logger.info(f"🔍 Knowledge base search: {query}")
            
            # Use the same enhanced search as context injection
            rag_context = await self._get_enhanced_rag_context(query)
            
            if rag_context:
                logger.info("📊 Knowledge base search successful")
                return rag_context
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
        """Request transfer (doesn't auto-transfer) - FIXED"""
        logger.info(f"💬 Transfer requested: {reason}")
        
        context.userdata.transfer_requested = True
        customer_name = context.userdata.caller_name or "there"
        
        # Provide helpful response and ask for confirmation
        return f"I understand you'd like to speak with someone else, {customer_name}. Would you like me to connect you with one of our specialists? Just say 'yes, transfer me' to confirm."

    @function_tool()
    async def execute_transfer_to_human(
        self, 
        context: RunContext[CallData],
        confirmed: bool = True
    ) -> str:
        """Execute transfer only when confirmed - FIXED"""
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
    """Enhanced caller identification"""
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
    """Prewarm function to load models early"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """
    FINAL FIXED entrypoint with enhanced STT and RAG
    """
    
    logger.info("🚀 FINAL FIXED RAG Voice Agent Starting")
    logger.info("🔧 Enhanced STT, RAG, and conversation flow")
    
    await ctx.connect()
    logger.info("✅ Connected to room")
    
    # Initialize RAG with FIXED error handling
    try:
        rag_start = time.time()
        logger.info("🔧 Initializing FIXED RAG system...")
        
        # Initialize with proper retries
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            try:
                success = await asyncio.wait_for(simple_rag.initialize(), timeout=20.0)
                if success:
                    break
                else:
                    logger.warning(f"RAG initialization attempt {attempt + 1} failed")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
            except asyncio.TimeoutError:
                logger.warning(f"RAG initialization attempt {attempt + 1} timed out")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
        
        rag_time = (time.time() - rag_start) * 1000
        
        if success:
            status = await simple_rag.get_status()
            points_count = status.get("points_count", 0)
            logger.info(f"✅ FIXED RAG system initialized in {rag_time:.1f}ms")
            logger.info(f"📊 Knowledge base: {points_count} documents")
            
            if points_count == 0:
                logger.error("❌ CRITICAL: Knowledge base is EMPTY!")
                logger.error("💡 Run: python excel_ingest.py --file data/Roadside_Assist_Info.xlsx")
                logger.error("💡 Agent will work but responses will be generic")
            elif points_count < 20:
                logger.warning(f"⚠️ Low document count: {points_count}")
                logger.warning("💡 Consider adding more data to knowledge base")
            else:
                logger.info(f"✅ Good knowledge base size: {points_count} documents")
        else:
            logger.error("❌ CRITICAL: RAG system failed to initialize!")
            logger.error("💡 Check: docker-compose up -d")
            logger.error("💡 Check: OpenAI API key")
            logger.error("💡 Agent will work but without knowledge base")
            
    except Exception as e:
        logger.error(f"❌ RAG initialization error: {e}")
        success = False
    
    # Identify caller and load history
    call_data = await identify_caller_with_history(ctx)
    
    # Create FIXED session with enhanced STT
    session_params = {
        "vad": ctx.proc.userdata["vad"],
        
        # ENHANCED STT configuration based on LiveKit examples
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
    
    # Enhanced TTS setup
    try:
        session_params["tts"] = elevenlabs.TTS(
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Professional voice
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
    
    # Add turn detection if available
    if TURN_DETECTOR_AVAILABLE:
        session_params["turn_detection"] = MultilingualModel()
        logger.info("✅ Using semantic turn detection")
    
    # Create session
    session = AgentSession[CallData](**session_params)
    
    # Create FIXED RAG agent
    fixed_agent = FinalFixedRAGAgent(call_data)
    
    # Start session
    await session.start(
        agent=fixed_agent,
        room=ctx.room
    )
    
    # Generate contextual greeting
    if call_data.is_returning_caller:
        greeting = "Say: 'Welcome back! I see you've called us before. How can I help you today?'"
    else:
        greeting = "Say: 'Roadside assistance, this is Mark, how can I help you today?'"
    
    await session.generate_reply(instructions=greeting)
    
    # Log final status with FIXED configuration
    logger.info("✅ FINAL FIXED RAG AGENT READY")
    logger.info(f"📞 Session ID: {call_data.session_id}")
    logger.info(f"👤 Caller ID: {call_data.caller_id}")
    logger.info(f"📱 Phone: {call_data.phone_number}")
    logger.info(f"🔄 Returning: {call_data.is_returning_caller}")
    logger.info(f"📊 RAG System: {'✅ Active' if success else '⚠️ Disabled'}")
    logger.info(f"🎯 FIXED Config: threshold={config.similarity_threshold}, timeout={config.rag_timeout_ms}ms")
    logger.info("🚫 Auto-transfer: DISABLED (only on explicit request)")
    logger.info("✅ Enhanced STT with better transcription")
    logger.info("✅ Fixed RAG with targeted Excel data search")

if __name__ == "__main__":
    try:
        logger.info("🎙️ Starting FINAL FIXED RAG Voice Agent")
        logger.info("📊 Fixed all issues from logs analysis")
        logger.info("🔧 Enhanced STT, RAG, and conversation flow")
        logger.info(f"⚙️ Fixed similarity threshold: {config.similarity_threshold}")
        logger.info(f"⚙️ Fixed RAG timeout: {config.rag_timeout_ms}ms")
        logger.info(f"⚙️ Enhanced search limit: {config.search_limit}")
        
        worker_options = WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="my-telephony-agent"
        )
        
        cli.run_app(worker_options)
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)