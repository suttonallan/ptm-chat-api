import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_my_day_prompt() -> str:
    """Load the my_day system prompt from prompts/my_day.txt"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    prompt_path = os.path.join(project_root, "prompts", "my_day.txt")

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Tu génères un résumé de journée pour un technicien de piano."


def _format_entries(entries: List[Dict[str, Any]]) -> str:
    """Format the raw entries into a structured text block for the LLM."""
    lines: list[str] = []
    for entry in entries:
        entry_type = entry.get("type", "appointment")
        lines.append(f"---")
        lines.append(f"Type: {entry_type}")
        if entry.get("time"):
            lines.append(f"Heure: {entry['time']}")
        if entry.get("title"):
            lines.append(f"Titre: {entry['title']}")

        if entry_type == "personal":
            if entry.get("location"):
                lines.append(f"Lieu: {entry['location']}")
            if entry.get("notes"):
                lines.append(f"Notes: {entry['notes']}")
        else:
            # Professional appointment
            if entry.get("client_name"):
                lines.append(f"Client: {entry['client_name']}")
            if entry.get("client_since"):
                lines.append(f"Client depuis: {entry['client_since']}")
            if entry.get("location"):
                lines.append(f"Lieu: {entry['location']}")
            if entry.get("piano"):
                piano = entry["piano"]
                parts = []
                for key in ["brand", "model", "type", "year", "age", "serial"]:
                    if piano.get(key):
                        parts.append(f"{key}: {piano[key]}")
                lines.append(f"Piano: {', '.join(parts)}")
            if entry.get("services"):
                lines.append("Services à effectuer:")
                for svc in entry["services"]:
                    lines.append(f"  - {svc}")
            if entry.get("previous_services"):
                lines.append("Services précédents:")
                for svc in entry["previous_services"]:
                    lines.append(f"  - {svc}")
            if entry.get("environment"):
                env = entry["environment"]
                if env.get("temperature"):
                    lines.append(f"Température: {env['temperature']}")
                if env.get("humidity"):
                    lines.append(f"Humidité: {env['humidity']}")
            if entry.get("tuning_pitch"):
                lines.append(f"Diapason: {entry['tuning_pitch']}")
            if entry.get("notes"):
                lines.append(f"Notes: {entry['notes']}")
            if entry.get("parking"):
                # Intentionally excluded — see prompt rules
                pass

    return "\n".join(lines)


async def generate_my_day(entries: List[Dict[str, Any]]) -> str:
    """
    Generate the daily briefing for the piano technician.

    Args:
        entries: List of appointments and personal events for the day.

    Returns:
        Formatted daily briefing text.
    """
    system_prompt = load_my_day_prompt()
    user_content = _format_entries(entries)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        max_tokens=2000,
    )

    return response.choices[0].message.content
