from dotenv import load_dotenv
import os

from livekit.plugins import google

from livekit import agents, api
from livekit.agents import (
    Agent, 
    AgentSession, 
    RoomInputOptions, 
    RunContext,
    function_tool,
    get_job_context
)
from livekit.plugins import (
    openai,
    elevenlabs,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import asyncio
import logging

load_dotenv()

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. You can help users with their queries.
            
            If a user explicitly asks to speak to a human agent, requests human support, says they want to talk to a person,
            or if you cannot help them with their request, offer to transfer them to a human agent. 
            
            Always confirm with the user before transferring the call by saying something like:
            "I'd be happy to transfer you to a human agent. Would you like me to do that now?"
            
            Wait for their confirmation before calling the transfer function."""
        )

    @function_tool()
    async def transfer_call(self, ctx: RunContext):
        """Enhanced transfer function with better debugging and timeout handling"""
        
        transfer_to = "sip:voiceai@sip.linphone.org"
        
        # Get the current job context
        job_ctx = get_job_context()
        
        # Enhanced logging for debugging
        logger.info(f"=== TRANSFER CALL INITIATED ===")
        logger.info(f"Room: {job_ctx.room.name}")
        logger.info(f"Total remote participants: {len(job_ctx.room.remote_participants)}")
        
        # Find the SIP participant
        sip_participant = None
        for participant in job_ctx.room.remote_participants.values():
            logger.info(f"Found participant: {participant.identity}, kind: {participant.kind}")
            if str(participant.kind) == "3" or "sip_" in participant.identity.lower():
                sip_participant = participant
                logger.info(f"✅ Found SIP participant: {participant.identity}")
                break
        
        if not sip_participant:
            logger.error("❌ No SIP participants found!")
            await ctx.session.generate_reply(
                instructions="I'm sorry, I couldn't find any active participants to transfer. Please try calling again."
            )
            return "Could not find any participant to transfer. Please try again."
        
        participant_identity = sip_participant.identity
        logger.info(f"🔄 Will transfer participant: {participant_identity} to SIP: {transfer_to}")
        
        # Inform the user about the transfer with instructions
        await ctx.session.generate_reply(
            instructions="""I'm connecting you to a human agent now. The transfer will begin in just a moment. 
            If you hear ringing, the agent should answer automatically. Please stay on the line."""
        )
        
        # Wait for the message to complete
        await asyncio.sleep(2)
        
        try:
            # Execute the SIP transfer with detailed logging
            logger.info(f"🚀 Starting SIP transfer request...")
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=participant_identity,
                transfer_to=transfer_to,
                play_dialtone=True,
            )
            
            # Start the transfer
            logger.info(f"📞 Executing transfer_sip_participant...")
            start_time = asyncio.get_event_loop().time()
            
            # Try with 30 second timeout (in case auto-answer delay is configured)
            await asyncio.wait_for(
                job_ctx.api.sip.transfer_sip_participant(transfer_request),
                timeout=30.0
            )
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            logger.info(f"✅ SIP Transfer completed successfully in {duration:.2f} seconds!")
            logger.info(f"   From: {participant_identity}")
            logger.info(f"   To: {transfer_to}")
            logger.info(f"   Room: {job_ctx.room.name}")
            
            return "Call transfer completed successfully to human agent"
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ Transfer timeout after 30 seconds")
            await ctx.session.generate_reply(
                instructions="""I'm having trouble connecting to our human agent. The call is reaching them, 
                but their phone isn't automatically answering. They may need to check their auto-answer settings. 
                Would you like me to try again, or would you prefer to call back later?"""
            )
            return "Transfer timed out - auto-answer not responding. Check Linphone settings."
                    
        except Exception as e:
            logger.error(f"❌ Error transferring call: {e}")
            await ctx.session.generate_reply(
                instructions="I apologize, but I'm having trouble transferring your call right now. Please try again in a moment."
            )
            return f"Transfer failed: {str(e)}"

    @function_tool()
    async def check_transfer_availability(self, ctx: RunContext):
        """Check if human agents are available for transfer"""
        logger.info("Checking human agent availability")
        return "Human agents are available for transfer. Would you like me to connect you to one?"

    @function_tool()
    async def get_business_hours(self, ctx: RunContext):
        """Provide information about when human agents are available"""
        return "Our human agents are available 24/7. I can transfer you to speak with someone right now if you'd like."


async def entrypoint(ctx: agents.JobContext):
    """Main entry point for the voice agent"""
    
    # Enhanced logging
    logger.info(f"=== AGENT SESSION STARTING ===")
    logger.info(f"Room: {ctx.room.name}")
    logger.info(f"Agent: my-telephony-agent")
    
    # ✅ TELEPHONY-OPTIMIZED Configuration
    google_project = os.getenv("GOOGLE_CLOUD_PROJECT", "my-tts-project-458404")
    google_location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")  
    google_language = os.getenv("GOOGLE_STT_LANGUAGE", "en-US")
    
    logger.info(f"Google STT Config - Project: {google_project}, Location: {google_location}")
    
    session = AgentSession(
        # ✅ WORKING Google STT Configuration for Telephony
        stt=google.STT(
            # Use latest_long for better telephony audio processing
            model="latest_long",                   # ✅ More reliable than "chirp" for telephony
            languages=[google_language],           # ✅ Use "languages" (plural) parameter  
            location=google_location,              # ✅ Specify your location (us-central1)
            spoken_punctuation=False,              # ✅ Disable spoken punctuation
            interim_results=True,                  # ✅ Enable interim results for responsiveness
            detect_language=False,                 # ✅ Disable language detection (single language)
            punctuate=True,                       # ✅ Enable automatic punctuation
            sample_rate=16000,                    # ✅ Standard telephony sample rate
        ),
        
        # ✅ LLM: OpenAI configuration  
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.3,
        ),
        
        # ✅ TTS: OpenAI configuration (reliable for telephony)
        tts=openai.TTS(
            model="tts-1",           
            voice="nova",            
        ),
        
        # ✅ VAD: Essential for telephony - detects when user is speaking
        vad=silero.VAD.load(
            # More sensitive settings for telephony
            # min_silence_duration_ms=500,   # Wait 500ms of silence before stopping
            # min_speech_duration_ms=100,    # Minimum 100ms of speech to trigger
        ),
        
        # ✅ Turn Detection: Critical for telephony conversations
        turn_detection=MultilingualModel(),
    )

    # Start the session
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # ✅ Telephony-specific noise cancellation
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )

    # Connect to the room
    await ctx.connect()
    logger.info("✅ Agent connected to room successfully")

    # ✅ Add delay to ensure audio stream is ready for telephony
    await asyncio.sleep(1)

    # Generate initial greeting
    await session.generate_reply(
        instructions="""Give a brief, clear greeting for a phone call. Say: "Hello! I'm your AI assistant. How can I help you today?" Keep it short and speak clearly for phone audio quality."""
    )
    
    logger.info("✅ Initial greeting sent")


if __name__ == "__main__":
    # Start the agent
    logger.info("🚀 Starting TELEPHONY-OPTIMIZED Voice Agent")
    logger.info("📞 Transfer destination: sip:voiceai@sip.linphone.org")
    
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="my-telephony-agent"
    ))