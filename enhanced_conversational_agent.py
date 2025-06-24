# enhanced_conversational_agent.py - FIXED VERSION
"""
Enhanced LiveKit 1.0 Voice Agent with Natural Conversation Flow
FIXED: Updated to use correct turn detector imports
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

from dotenv import load_dotenv
from livekit import api, agents
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
# FIXED: Import the correct turn detector classes
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from qdrant_rag_system import qdrant_rag
from config import config

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class CallData:
    """Shared conversation state across the entire call"""
    caller_name: Optional[str] = None
    phone_number: Optional[str] = None
    location: Optional[str] = None
    vehicle_year: Optional[str] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_color: Optional[str] = None
    service_type: Optional[str] = None
    issue_description: Optional[str] = None
    urgency_level: str = "normal"  # normal, urgent, emergency
    call_stage: str = "greeting"  # greeting, gathering_info, diagnosing, quoting, confirming, transferring
    conversation_history: List[str] = field(default_factory=list)
    gathered_info: Dict[str, bool] = field(default_factory=lambda: {
        "name": False,
        "phone": False, 
        "location": False,
        "vehicle": False,
        "service": False
    })

class EnhancedRoadsideAgent(Agent):
    """Enhanced conversational agent that mimics natural human call flow from transcripts"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions=self._get_dynamic_instructions()
        )
        self.call_start_time = time.time()
        self.last_user_input_time = None
        self.conversation_memory = []
        self.knowledge_cache = {}  # Cache frequent lookups
        
    def _get_dynamic_instructions(self) -> str:
        """Dynamic instructions based on actual call transcript patterns"""
        return """You are Mark, a professional roadside assistance operator. Follow the natural conversation flow from our call transcripts.

CONVERSATION STYLE (based on real calls):
- Start: "Roadside assistance, this is Mark, how can I help?"
- Be empathetic: "I'm sorry to hear about that" when they describe problems
- Confirm details: "Just to confirm, you said..." or "So your vehicle is at..."
- Use natural transitions: "Got it" "Thanks for that" "Perfect"
- Ask clarifying questions: "Could you tell me..." "What's the exact location..."

INFORMATION GATHERING SEQUENCE (follow this order):
1. First get their FULL NAME
2. Then get PHONE NUMBER for callback
3. Get EXACT LOCATION (street address, city, landmarks)
4. Get VEHICLE details (year, make, model, color if mentioned)
5. Understand the PROBLEM and SERVICE NEEDED
6. Search knowledge base for service options and pricing
7. Confirm details and arrange service

CONVERSATION PATTERNS FROM TRANSCRIPTS:
- "Could you please provide your full name?"
- "Could you also provide a good phone number where we can reach you?"
- "What is the exact location of your vehicle? Please provide the full street address, city, and any nearby landmarks"
- "Could you tell me the year, make, and model of your vehicle?"
- "What type of service do you need today?"
- "Just to confirm..." (always confirm important details)

ALWAYS:
- Use the search_knowledge function for ANY service, pricing, or policy questions
- Never provide hardcoded pricing - always search the knowledge base
- Keep responses under 30 words for phone clarity
- Confirm addresses by repeating them back
- Be patient if they can't hear or need to repeat information"""

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Enhanced context injection with conversation memory and dynamic knowledge"""
        try:
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 3:
                return
                
            self.last_user_input_time = time.time()
            
            # Add to conversation memory
            self.conversation_memory.append(f"User: {user_text}")
            if len(self.conversation_memory) > 12:  # Keep conversation context
                self.conversation_memory = self.conversation_memory[-12:]
            
            # Analyze conversation stage and what we still need
            stage_context = await self._analyze_conversation_stage(user_text, turn_ctx)
            
            # Update dynamic instructions based on progress
            await self._update_context_awareness(turn_ctx, stage_context)
            
            # Auto-inject knowledge for service-related queries
            if await self._should_search_knowledge(user_text):
                await self._inject_knowledge_context(user_text, turn_ctx)
                    
        except Exception as e:
            logger.error(f"Error in conversation context: {e}")

    async def _analyze_conversation_stage(self, user_text: str, turn_ctx: ChatContext) -> Dict[str, Any]:
        """Analyze conversation progress and what information is still needed"""
        user_lower = user_text.lower()
        
        # Detect urgency indicators
        urgency_keywords = ["emergency", "urgent", "stranded", "highway", "dangerous", "stuck", "can't move", "unsafe"]
        is_urgent = any(keyword in user_lower for keyword in urgency_keywords)
        
        # Detect service type mentions
        service_indicators = {
            "towing": ["tow", "towed", "towing", "pull", "move my car", "won't start", "dead", "broken down"],
            "battery": ["battery", "dead battery", "jump", "jumpstart", "won't start", "no power", "died"],
            "tire": ["tire", "flat tire", "puncture", "rim", "wheel", "flat"],
            "fuel": ["gas", "fuel", "empty", "ran out", "out of gas", "no gas"],
            "lockout": ["locked out", "keys", "locked", "can't get in", "keys inside"]
        }
        
        detected_services = []
        for service, keywords in service_indicators.items():
            if any(keyword in user_lower for keyword in keywords):
                detected_services.append(service)
        
        # Detect information provided in this turn
        info_detected = {
            "name": any(word.istitle() and len(word) > 2 for word in user_text.split()),
            "phone": any(char.isdigit() for char in user_text) and len([c for c in user_text if c.isdigit()]) >= 7,
            "location": any(indicator in user_lower for indicator in ["street", "road", "avenue", "boulevard", "highway", "exit", "mile", "address"]),
            "vehicle": any(brand in user_lower for brand in ["honda", "toyota", "ford", "chevy", "bmw", "audi", "mercedes", "nissan", "hyundai", "kia", "jeep", "dodge"]) or any(year in user_text for year in ["20", "19"]),
            "service_request": len(detected_services) > 0
        }
        
        return {
            "urgency": "urgent" if is_urgent else "normal",
            "detected_services": detected_services,
            "info_provided": info_detected,
            "conversation_length": len(self.conversation_memory),
            "needs_clarification": "hello" in user_lower or "can you hear" in user_lower
        }

    async def _update_context_awareness(self, turn_ctx: ChatContext, stage_info: Dict[str, Any]) -> None:
        """Update agent instructions based on conversation progress"""
        context_msg = "CURRENT CALL STATUS:\n"
        
        if stage_info["urgency"] == "urgent":
            context_msg += "⚠️ URGENT SITUATION - Express empathy and prioritize assistance\n"
            
        if stage_info["detected_services"]:
            context_msg += f"🔧 Service mentioned: {', '.join(stage_info['detected_services'])}\n"
            
        if stage_info["needs_clarification"]:
            context_msg += "🎧 AUDIO ISSUE - User may have trouble hearing, speak clearly\n"
            
        # Progress tracking
        if stage_info["conversation_length"] < 2:
            context_msg += "📞 OPENING - Give professional greeting and ask how to help\n"
        elif not stage_info["info_provided"]["name"]:
            context_msg += "👤 NEED NAME - 'Could you please provide your full name?'\n"
        elif not stage_info["info_provided"]["phone"]:
            context_msg += "📱 NEED PHONE - 'Could you also provide a good phone number where we can reach you?'\n"
        elif not stage_info["info_provided"]["location"]:
            context_msg += "📍 NEED LOCATION - 'What is the exact location of your vehicle? Please provide the full street address, city, and any nearby landmarks'\n"
        elif not stage_info["info_provided"]["vehicle"]:
            context_msg += "🚗 NEED VEHICLE - 'Could you tell me the year, make, and model of your vehicle?'\n"
        elif not stage_info["info_provided"]["service_request"]:
            context_msg += "🔧 NEED SERVICE - 'What type of service do you need today?'\n"
        else:
            context_msg += "✅ INFO COMPLETE - Search knowledge base and provide service options\n"
            
        turn_ctx.add_message(role="system", content=context_msg)

    async def _should_search_knowledge(self, user_text: str) -> bool:
        """Determine if we should automatically search knowledge base"""
        search_triggers = [
            "cost", "price", "how much", "fee", "charge", "payment",
            "coverage", "member", "membership", "plan", "policy",
            "service", "help", "assist", "available", "offer",
            "tow", "battery", "tire", "fuel", "lockout", "jump",
            "business hours", "when", "time", "how long", "wait"
        ]
        user_lower = user_text.lower()
        return any(trigger in user_lower for trigger in search_triggers)

    async def _inject_knowledge_context(self, user_text: str, turn_ctx: ChatContext) -> None:
        """Inject relevant knowledge from Excel database"""
        try:
            # Check cache first
            cache_key = user_text.lower().strip()[:50]
            if cache_key in self.knowledge_cache:
                context = self.knowledge_cache[cache_key]
                turn_ctx.add_message(role="system", content=f"[Knowledge]: {context}")
                logger.info("📚 Used cached knowledge")
                return

            # Search knowledge base
            results = await asyncio.wait_for(
                qdrant_rag.search(user_text, limit=3),
                timeout=0.8
            )
            
            if results and results[0]["score"] >= 0.25:  # Lower threshold for better coverage
                # Combine multiple relevant results for comprehensive context
                knowledge_context = []
                for result in results[:2]:  # Use top 2 results
                    if result["score"] >= 0.25:
                        formatted = self._format_knowledge_for_voice(result["text"])
                        if formatted and formatted not in knowledge_context:
                            knowledge_context.append(formatted)
                
                if knowledge_context:
                    combined_context = " | ".join(knowledge_context)
                    
                    # Cache for future use
                    self.knowledge_cache[cache_key] = combined_context
                    if len(self.knowledge_cache) > 50:  # Limit cache size
                        oldest_key = next(iter(self.knowledge_cache))
                        del self.knowledge_cache[oldest_key]
                    
                    turn_ctx.add_message(role="system", content=f"[Knowledge]: {combined_context}")
                    logger.info(f"📚 Knowledge injected (score: {results[0]['score']:.3f})")
                    
        except Exception as e:
            logger.debug(f"Knowledge injection failed: {e}")

    def _format_knowledge_for_voice(self, raw_text: str) -> str:
        """Format knowledge base content for natural voice delivery"""
        if not raw_text:
            return ""
            
        # Clean up common formatting
        cleaned = raw_text.strip()
        cleaned = cleaned.replace("Q:", "").replace("A:", "")
        cleaned = cleaned.replace("•", "").replace("-", "").replace("*", "")
        cleaned = cleaned.replace("\n", " ").replace("\t", " ")
        
        # Remove multiple spaces
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")
        
        # Take meaningful content
        sentences = [s.strip() for s in cleaned.split('.') if len(s.strip()) > 15]
        if sentences:
            # Return first substantial sentence, keeping it concise for voice
            result = sentences[0].strip()
            if len(result) > 150:  # Truncate if too long for voice
                result = result[:147] + "..."
            return result
        
        # Fallback to first 100 characters if no good sentences
        return cleaned[:100].strip() if len(cleaned) > 100 else cleaned

    @function_tool()
    async def search_knowledge(
        self, 
        context: RunContext[CallData],
        query: str
    ) -> str:
        """
        Search the dynamic knowledge base for service information, pricing, policies, etc.
        Use this for ANY question about services, costs, coverage, business hours, or policies.
        """
        try:
            logger.info(f"🔍 Knowledge search: {query}")
            
            # Search with multiple related queries for better coverage
            search_queries = [query]
            
            # Add related terms based on query content
            query_lower = query.lower()
            if "cost" in query_lower or "price" in query_lower:
                search_queries.append("pricing fees charges")
            if "tow" in query_lower:
                search_queries.append("towing service rates")
            if "battery" in query_lower:
                search_queries.append("jumpstart battery service")
            if "tire" in query_lower:
                search_queries.append("tire change flat tire")
                
            all_results = []
            for search_query in search_queries[:2]:  # Limit to avoid too many searches
                try:
                    results = await asyncio.wait_for(
                        qdrant_rag.search(search_query, limit=2),
                        timeout=1.0
                    )
                    all_results.extend(results)
                except Exception:
                    continue
            
            if not all_results:
                return "I don't have specific information about that in my knowledge base. Let me transfer you to someone who can provide detailed information."
            
            # Get best results
            best_results = sorted(all_results, key=lambda x: x["score"], reverse=True)[:3]
            
            # Format response for voice
            response_parts = []
            for result in best_results:
                if result["score"] >= 0.2:  # Lower threshold for comprehensive answers
                    formatted = self._format_knowledge_for_voice(result["text"])
                    if formatted and formatted not in response_parts:
                        response_parts.append(formatted)
            
            if response_parts:
                response = " | ".join(response_parts)
                logger.info(f"📊 Knowledge found (best score: {best_results[0]['score']:.3f})")
                return response
            else:
                return "I don't have detailed information about that. Let me connect you with someone who can provide specific details."
                
        except Exception as e:
            logger.error(f"❌ Knowledge search error: {e}")
            return "I'm having trouble accessing that information right now. Let me transfer you to someone who can help."

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
        """Store caller information as it's gathered during the natural conversation flow"""
        
        # Update the shared call data
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
            
        # Add to conversation memory
        if updates:
            self.conversation_memory.append(f"Agent: Recorded - {', '.join(updates)}")
        
        logger.info(f"📝 Updated caller info: {context.userdata}")
        
        # Check if we have enough info to proceed
        gathered = context.userdata.gathered_info
        if all([gathered["name"], gathered["phone"], gathered["location"], gathered["vehicle"], gathered["service"]]):
            return "Perfect! I have all the information I need. Let me search for the best service options and pricing for you."
        else:
            missing = [key for key, value in gathered.items() if not value]
            return f"Got it. I still need to get your {', '.join(missing)} to complete your service request."

    @function_tool()
    async def confirm_service_details(
        self, 
        context: RunContext[CallData]
    ) -> str:
        """Confirm all service details before arranging service - mimics transcript pattern"""
        
        data = context.userdata
        
        # Build vehicle description
        vehicle_parts = [data.vehicle_year, data.vehicle_make, data.vehicle_model, data.vehicle_color]
        vehicle_desc = " ".join([part for part in vehicle_parts if part])
        
        confirmation = f"""Let me confirm your service request:
        
Name: {data.caller_name or 'Not provided'}
Phone: {data.phone_number or 'Not provided'}
Location: {data.location or 'Not provided'}
Vehicle: {vehicle_desc or 'Not fully specified'}
Service needed: {data.service_type or 'Not specified'}
Issue: {data.issue_description or 'Not described'}

Is this all correct?"""
        
        context.userdata.call_stage = "confirming"
        
        # Add to conversation memory
        self.conversation_memory.append("Agent: Confirming all service details")
        
        return confirmation

    @function_tool()
    async def transfer_to_dispatcher(
        self, 
        context: RunContext[CallData],
        reason: str = "Complete service arrangement"
    ) -> str:
        """Transfer call to dispatcher following transcript pattern"""
        
        try:
            job_ctx = get_job_context()
            
            # Find SIP participant
            sip_participant = None
            for participant in job_ctx.room.remote_participants.values():
                if str(participant.kind) == "3":  # SIP participant
                    sip_participant = participant
                    break
                    
            if not sip_participant:
                return "I'm having trouble with the transfer. Let me continue helping you directly."
            
            # Prepare detailed handoff information
            data = context.userdata
            vehicle_info = f"{data.vehicle_year or ''} {data.vehicle_make or ''} {data.vehicle_model or ''} {data.vehicle_color or ''}".strip()
            
            handoff_summary = f"""Service Request Summary:
Customer: {data.caller_name or 'Unknown'}
Phone: {data.phone_number or 'TBD'}
Location: {data.location or 'TBD'}
Vehicle: {vehicle_info or 'TBD'}
Service: {data.service_type or 'General assistance'}
Issue: {data.issue_description or 'See notes'}
Urgency: {data.urgency_level}
Call Duration: {time.time() - self.call_start_time:.0f} seconds"""
            
            logger.info(f"🔄 Transferring call:\n{handoff_summary}")
            
            # Inform caller about transfer using transcript language
            await context.session.generate_reply(
                instructions=f"Say: 'I'm going to transfer you to a dispatcher who will provide further instructions and an estimated time of arrival. Please hold.' Keep it professional and brief like in our call transcripts."
            )
            
            await asyncio.sleep(2)  # Allow message to complete
            
            # Execute transfer
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=sip_participant.identity,
                transfer_to=config.transfer_sip_address,
                play_dialtone=True,
            )
            
            await job_ctx.api.sip.transfer_sip_participant(transfer_request)
            
            return f"Successfully transferred {data.caller_name or 'caller'} to dispatcher for service arrangement"
            
        except Exception as e:
            logger.error(f"❌ Transfer failed: {e}")
            return "I'm having trouble with the transfer. Let me continue helping you with your service request."

    @function_tool()
    async def handle_emergency_situation(
        self, 
        context: RunContext[CallData]
    ) -> str:
        """Handle urgent/emergency situations with immediate priority"""
        
        context.userdata.urgency_level = "emergency"
        context.userdata.call_stage = "emergency"
        
        logger.warning("🚨 Emergency situation detected")
        
        # Immediate empathetic response
        await context.session.generate_reply(
            instructions="Express immediate concern and empathy. Ask if they are in immediate danger and need emergency services (911). If safe, prioritize getting their exact location immediately for emergency roadside assistance.",
            allow_interruptions=True
        )
        
        return "Emergency protocols activated. Prioritizing immediate safety and location information."

    @function_tool()
    async def handle_audio_issues(
        self, 
        context: RunContext[CallData]
    ) -> str:
        """Handle audio/connection issues like in the transcripts"""
        
        await context.session.generate_reply(
            instructions="Speak slowly and clearly. Say something like 'Can you hear me now? I want to make sure we have a good connection so I can help you.' Be patient and speak distinctly.",
            allow_interruptions=True
        )
        
        return "Addressing audio/connection issues with caller"

async def create_enhanced_session(userdata: CallData) -> AgentSession[CallData]:
    """Create optimized session for natural conversation matching transcript quality"""
    
    session = AgentSession[CallData](
        # Enhanced STT for better accuracy - matching transcript quality
        stt=deepgram.STT(
            model="nova-2-general",
            language="en-US",
            smart_format=True,  # Better punctuation and formatting
            profanity_filter=False,  # Allow natural speech
            numerals=True,  # Convert numbers properly
        ),
        
        # Optimized LLM for conversation flow
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.2,  # More consistent, professional responses
        ),
        
        # Professional TTS voice (like "Mark" in transcripts)
        tts=elevenlabs.TTS(
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Professional male voice
            voice_settings=elevenlabs.VoiceSettings(
                stability=0.7,      # More stable for professional calls
                similarity_boost=0.8,
                style=0.1,          # Less dramatic, more professional
                speed=0.95          # Slightly slower for clarity
            ),
            model="eleven_turbo_v2_5",  # Fastest model for low latency
        ),
        
        # Enhanced turn detection for natural conversation flow
        vad=silero.VAD.load(),
        # FIXED: Use MultilingualModel instead of EOUModel
        turn_detection=MultilingualModel(),  # Semantic end-of-utterance detection
        
        # Natural conversation timing (based on transcript analysis)
        allow_interruptions=True,
        min_interruption_duration=0.6,  # Allow natural interruptions
        min_endpointing_delay=0.8,      # Natural pause handling
        max_endpointing_delay=4.0,      # Allow time for people to think
        
        # FIXED: userdata goes in constructor, not start() method
        userdata=userdata
    )
    
    return session

async def entrypoint(ctx: JobContext):
    """Enhanced entrypoint matching professional call center standards"""
    
    logger.info("🚀 Enhanced Roadside Assistance Agent Starting")
    logger.info("📋 Features: Dynamic knowledge base, natural conversation flow, context retention")
    
    await ctx.connect()
    
    # Initialize systems in parallel
    init_start = time.time()
    init_tasks = [
        qdrant_rag.initialize()
    ]
    
    rag_ready = await asyncio.gather(*init_tasks)
    rag_ready = rag_ready[0]  # Extract boolean from list
    
    # Create enhanced agent with call data tracking
    agent = EnhancedRoadsideAgent()
    
    # Create call data and session
    call_data = CallData()
    session = await create_enhanced_session(call_data)
    
    # FIXED: start() only takes agent and room parameters
    await session.start(
        agent=agent,
        room=ctx.room
    )
    
    init_time = (time.time() - init_start) * 1000
    
    # Professional greeting matching transcript style
    await session.generate_reply(
        instructions="Give the exact greeting from our transcripts: 'Roadside assistance, this is Mark, how can I help?'"
    )
    
    # Log initialization success
    logger.info("✅ Enhanced agent ready for calls")
    logger.info(f"⚡ Initialization time: {init_time:.1f}ms")
    logger.info(f"📚 Knowledge base ready: {rag_ready}")
    logger.info("🎯 Conversation features: Natural flow, context retention, dynamic knowledge")

if __name__ == "__main__":
    try:
        logger.info("🎙️  Starting Enhanced Roadside Assistance Agent")
        logger.info("📊 Using dynamic Excel knowledge base for all service information")
        logger.info("🗣️  Natural conversation flow based on call transcript analysis")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)