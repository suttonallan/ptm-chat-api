import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory conversation history storage
# Structure: { session_id: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}] }
conversation_history: Dict[str, List[Dict[str, str]]] = {}

def load_system_prompt() -> str:
    """Load system prompt from prompts/system.txt"""
    import os
    # Get the directory of this file, then go up one level to find prompts/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    prompt_path = os.path.join(project_root, "prompts", "system.txt")
    
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are a helpful assistant for Piano Technique Montréal."

async def get_chat_response(
    message: str,
    session_id: str,
    expertise_result: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a chat response using OpenAI GPT-4o
    
    Args:
        message: User's message
        session_id: Session identifier for conversation history
        expertise_result: Optional expertise result from the piano analysis
    
    Returns:
        Assistant's reply
    """
    # Load system prompt
    system_prompt = load_system_prompt()
    
    # Build messages list
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add expertise result to context if available
    if expertise_result:
        expertise_context = f"""
Résultat d'expertise IA disponible :
- Verdict: {expertise_result.get('verdict', 'N/A')}
- Score: {expertise_result.get('score', 'N/A')}/100
- Commentaire expert: {expertise_result.get('commentaire_expert', 'N/A')}

Utilise ces informations pour personnaliser tes réponses et proposer des services adaptés.
"""
        messages.append({"role": "system", "content": expertise_context})
    
    # Get conversation history for this session
    if session_id in conversation_history:
        messages.extend(conversation_history[session_id])
    
    # Add current user message
    messages.append({"role": "user", "content": message})
    
    # Call OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        assistant_reply = response.choices[0].message.content
        
        # Update conversation history
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        conversation_history[session_id].append({"role": "user", "content": message})
        conversation_history[session_id].append({"role": "assistant", "content": assistant_reply})
        
        # Limit history to last 10 exchanges (20 messages) to avoid token limits
        if len(conversation_history[session_id]) > 20:
            conversation_history[session_id] = conversation_history[session_id][-20:]
        
        return assistant_reply
    
    except Exception as e:
        raise Exception(f"Error calling OpenAI API: {str(e)}")
