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

# Modèle de conversation : jamais codé en dur. OPENAI_MODEL (variable Render) l'impose ;
# sinon gpt-4o, et si OpenAI le retire, on choisit le plus récent modèle « gpt-N » général.
_chat_model = None


def _is_model_gone(err: Exception) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in ("model_not_found", "does not exist", "no longer available", "deprecated", "404"))


def _discover_chat_model() -> str:
    import re as _re
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception:
        return "gpt-4o"
    good = [i for i in ids if _re.fullmatch(r"gpt-\d+(\.\d+)?o?", i)]
    if not good:
        return "gpt-4o"
    return sorted(good, key=lambda i: float(_re.findall(r"\d+(?:\.\d+)?", i)[0]), reverse=True)[0]


def get_chat_model(force_discovery: bool = False) -> str:
    global _chat_model
    if force_discovery:
        _chat_model = _discover_chat_model()
    elif not _chat_model:
        _chat_model = os.getenv("OPENAI_MODEL") or "gpt-4o"
    return _chat_model


def load_system_prompt() -> str:
    """Load system prompt from prompts/system.txt"""
    import os
    # Get the directory of this file, then go up one level to find prompts/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    prompt_path = os.path.join(project_root, "prompts", "system.txt")
    
    infos_path = os.path.join(project_root, "prompts", "infos_ptm.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    except FileNotFoundError:
        prompt = "Tu es l'assistant de Piano Technique Montréal. Vouvoie le client et propose d'appeler au 514 344-8008."
    try:
        with open(infos_path, "r", encoding="utf-8") as f:
            infos = "\n".join(l for l in f.read().splitlines() if not l.startswith("#"))
        prompt += "\n\n## INFOS OFFICIELLES PTM\n" + infos.strip()
    except FileNotFoundError:
        pass
    return prompt

async def get_chat_response(
    message: str,
    session_id: str,
    expertise_result: Optional[Dict[str, Any]] = None,
    listing_context: Optional[str] = None,
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
        lines = [
            "## RÉSULTAT D'EXPERTISE PIANO (analyse Gemini)",
            "",
            f"**Marque:** {expertise_result.get('marque_detectee', 'N/A')}",
            f"**Modèle:** {expertise_result.get('modele_detecte', 'N/A')}",
            f"**Âge estimé:** {expertise_result.get('annee_estimee', 'N/A')}",
            f"**Verdict:** {expertise_result.get('verdict', 'N/A')}",
            f"**Type de mécanique:** {expertise_result.get('type_mecanique', 'Standard')}",
            f"**Recommandation d'achat:** {expertise_result.get('recommandation_achat', 'N/A')}",
        ]

        alertes = expertise_result.get("alertes")
        if alertes:
            lines.append(f"**⚠️ ALERTES:** {', '.join(alertes)}")

        historique = expertise_result.get("historique_marque")
        if historique:
            lines.append(f"**Historique de la marque:** {historique}")

        for label, key in [
            ("État général", "etat_general"),
            ("État du clavier", "etat_clavier"),
            ("État du boîtier", "etat_boitier"),
            ("Mécanique visible", "etat_mecanique_visible"),
        ]:
            etat = expertise_result.get(key)
            if etat and isinstance(etat, dict):
                lines.append(f"**{label}:** {etat.get('score', '?')}/10 — {etat.get('description', '')}")

        problemes = expertise_result.get("problemes_detectes")
        if problemes:
            lines.append("**Problèmes détectés:**")
            for p in problemes:
                lines.append(f"  • {p}")

        points = expertise_result.get("points_positifs")
        if points:
            lines.append("**Points positifs:**")
            for p in points:
                lines.append(f"  • {p}")


        travaux = expertise_result.get("travaux_recommandes")
        if travaux:
            lines.append("**Travaux à faire vérifier par un technicien (sans prix) :**")
            for t in travaux:
                lines.append(f"  • {t.get('travail', '')} (priorité: {t.get('priorite', '')})")

        if expertise_result.get("potentiel_restauration"):
            lines.append(f"**Potentiel de restauration:** {expertise_result['potentiel_restauration']}")
        if expertise_result.get("urgence_intervention"):
            lines.append(f"**Urgence d'intervention:** {expertise_result['urgence_intervention']}")
        if expertise_result.get("recommandation_contextuelle"):
            lines.append(f"**Recommandation contextuelle:** {expertise_result['recommandation_contextuelle']}")
        etape = expertise_result.get("prochaine_etape") or expertise_result.get("prochaine_etape_recommandee")
        if etape:
            lines.append(f"**Prochaine étape recommandée:** {etape}")

        lines.append(f"\n**Commentaire expert:** {expertise_result.get('commentaire_expert', 'N/A')}")

        lines.append("")
        lines.append("Présente ce premier avis selon tes consignes « QUAND TU PRÉSENTES UN AVIS PHOTO » : prudent, respectueux, sans aucun prix ni valeur en dollars, sans score, et termine par une seule prochaine étape.")
        photos_recues = expertise_result.get("photos_recues")
        if photos_recues:
            lines.append(f"- Photos reçues du client: {', '.join(photos_recues)}")

        photos_suggerees = expertise_result.get("photos_suggerees")
        if photos_suggerees:
            lines.append(f"- Photos supplémentaires suggérées: {', '.join(photos_suggerees)}")
        else:
            lines.append("- Photos suffisantes, ne PAS en demander d'autres.")

        lines.append("")
        lines.append("Utilise ces informations pour personnaliser tes réponses. Ne demande PAS de photos que le client a déjà envoyées.")

        messages.append({"role": "system", "content": "\n".join(lines)})
    
    # Add listing context if a URL was scraped
    if listing_context:
        messages.append({"role": "system", "content": listing_context})

    # Get conversation history for this session
    if session_id in conversation_history:
        messages.extend(conversation_history[session_id])
    
    # Add current user message
    messages.append({"role": "user", "content": message})
    
    # Allow longer replies when presenting a piano analysis
    token_limit = 1200 if expertise_result else 500

    # Call OpenAI API (modèle choisi automatiquement, rebascule si OpenAI en retire un)
    try:
        try:
            response = client.chat.completions.create(
                model=get_chat_model(),
                messages=messages,
                temperature=0.5,
                max_tokens=token_limit,
            )
        except Exception as e:
            if not _is_model_gone(e):
                raise
            response = client.chat.completions.create(
                model=get_chat_model(force_discovery=True),
                messages=messages,
                temperature=0.5,
                max_tokens=token_limit,
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
