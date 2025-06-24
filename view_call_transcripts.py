# view_call_transcripts.py
"""
Script to view and export call transcripts from your system
"""
import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from call_transcription_storage import call_storage

async def view_recent_calls():
    """View recent call sessions"""
    print("📞 RECENT CALL SESSIONS")
    print("=" * 60)
    
    sessions = await call_storage.get_recent_sessions(10)
    
    for session in sessions:
        duration = f"{session.duration_seconds:.1f}s" if session.duration_seconds else "ongoing"
        start_time = datetime.fromtimestamp(session.start_time).strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Session: {session.session_id}")
        print(f"  Phone: {session.phone_number}")
        print(f"  Start: {start_time}")
        print(f"  Duration: {duration}")
        print(f"  Status: {session.status}")
        print(f"  Turns: {session.total_turns}")
        print("-" * 60)

async def view_full_transcript(session_id: str):
    """View complete transcript for a session"""
    print(f"📋 FULL TRANSCRIPT FOR SESSION: {session_id}")
    print("=" * 60)
    
    # Get conversation items (complete turns)
    db_path = call_storage.db_path
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT role, content, timestamp, interrupted 
            FROM conversation_items 
            WHERE session_id = ? 
            ORDER BY timestamp
        """, (session_id,))
        
        items = cursor.fetchall()
        
        if not items:
            print("No transcript found for this session.")
            return
        
        for role, content, timestamp, interrupted in items:
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            interrupted_mark = " [INTERRUPTED]" if interrupted else ""
            
            print(f"[{time_str}] {role.upper()}: {content}{interrupted_mark}")
            print()

async def view_real_time_segments(session_id: str):
    """View real-time transcription segments"""
    print(f"🎙️ REAL-TIME SEGMENTS FOR SESSION: {session_id}")
    print("=" * 60)
    
    db_path = call_storage.db_path
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT speaker, text, timestamp, is_final, confidence 
            FROM transcription_segments 
            WHERE session_id = ? 
            ORDER BY timestamp
        """, (session_id,))
        
        segments = cursor.fetchall()
        
        if not segments:
            print("No segments found for this session.")
            return
        
        for speaker, text, timestamp, is_final, confidence in segments:
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")[:-3]
            final_mark = "FINAL" if is_final else "interim"
            conf_str = f" (conf: {confidence:.2f})" if confidence else ""
            
            print(f"[{time_str}] {speaker.upper()} [{final_mark}]{conf_str}: {text}")

async def export_transcript_to_file(session_id: str, format_type: str = "txt"):
    """Export transcript to file"""
    output_dir = Path("exported_transcripts")
    output_dir.mkdir(exist_ok=True)
    
    if format_type == "txt":
        output_file = output_dir / f"transcript_{session_id}.txt"
        
        with sqlite3.connect(call_storage.db_path) as conn:
            cursor = conn.execute("""
                SELECT role, content, timestamp, interrupted 
                FROM conversation_items 
                WHERE session_id = ? 
                ORDER BY timestamp
            """, (session_id,))
            
            items = cursor.fetchall()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Call Transcript - Session: {session_id}\n")
                f.write("=" * 60 + "\n\n")
                
                for role, content, timestamp, interrupted in items:
                    time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    interrupted_mark = " [INTERRUPTED]" if interrupted else ""
                    
                    f.write(f"[{time_str}] {role.upper()}: {content}{interrupted_mark}\n\n")
    
    elif format_type == "json":
        output_file = output_dir / f"transcript_{session_id}.json"
        
        # Export in MongoDB format
        mongodb_doc = await call_storage.export_for_mongodb(session_id)
        
        if mongodb_doc:
            # Convert datetime objects to strings for JSON
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                return obj
            
            json_doc = convert_datetime(mongodb_doc)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_doc, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Transcript exported to: {output_file}")

async def search_transcripts(keyword: str):
    """Search transcripts for specific keywords"""
    print(f"🔍 SEARCHING TRANSCRIPTS FOR: '{keyword}'")
    print("=" * 60)
    
    db_path = call_storage.db_path
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT ci.session_id, cs.phone_number, ci.role, ci.content, ci.timestamp
            FROM conversation_items ci
            JOIN call_sessions cs ON ci.session_id = cs.session_id
            WHERE ci.content LIKE ?
            ORDER BY ci.timestamp DESC
            LIMIT 20
        """, (f"%{keyword}%",))
        
        results = cursor.fetchall()
        
        if not results:
            print("No results found.")
            return
        
        for session_id, phone_number, role, content, timestamp in results:
            time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"Session: {session_id[:12]}... | Phone: {phone_number}")
            print(f"[{time_str}] {role.upper()}: {content}")
            print("-" * 40)

async def show_caller_history(phone_number: str):
    """Show call history for a specific caller"""
    print(f"📱 CALL HISTORY FOR: {phone_number}")
    print("=" * 60)
    
    caller_profile = await call_storage.get_caller_by_phone(phone_number)
    
    if not caller_profile:
        print("No caller found with that phone number.")
        return
    
    print(f"Caller ID: {caller_profile.caller_id}")
    print(f"Total Calls: {caller_profile.total_calls}")
    print(f"Total Conversation Turns: {caller_profile.total_conversation_turns}")
    print(f"First Call: {datetime.fromtimestamp(caller_profile.first_call_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Last Call: {datetime.fromtimestamp(caller_profile.last_call_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get conversation history
    history = await call_storage.get_caller_conversation_history(caller_profile.caller_id, limit=20)
    
    print("Recent Conversation History:")
    print("-" * 40)
    
    for item in history:
        time_str = datetime.fromtimestamp(item.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{time_str}] {item.role.upper()}: {item.content}")

async def main():
    """Interactive transcript viewer"""
    print("🎙️ CALL TRANSCRIPT VIEWER")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. View recent calls")
        print("2. View full transcript")
        print("3. View real-time segments")
        print("4. Export transcript")
        print("5. Search transcripts")
        print("6. Show caller history")
        print("7. Exit")
        
        choice = input("\nChoose an option (1-7): ").strip()
        
        try:
            if choice == "1":
                await view_recent_calls()
                
            elif choice == "2":
                session_id = input("Enter session ID: ").strip()
                await view_full_transcript(session_id)
                
            elif choice == "3":
                session_id = input("Enter session ID: ").strip()
                await view_real_time_segments(session_id)
                
            elif choice == "4":
                session_id = input("Enter session ID: ").strip()
                format_type = input("Format (txt/json): ").strip().lower()
                if format_type not in ['txt', 'json']:
                    format_type = 'txt'
                await export_transcript_to_file(session_id, format_type)
                
            elif choice == "5":
                keyword = input("Enter search keyword: ").strip()
                await search_transcripts(keyword)
                
            elif choice == "6":
                phone_number = input("Enter phone number: ").strip()
                await show_caller_history(phone_number)
                
            elif choice == "7":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-7.")
                
        except KeyboardInterrupt:
            print("\n🛑 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())