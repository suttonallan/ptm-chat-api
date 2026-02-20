import os
import json
import re
import base64
import logging
from fastapi import HTTPException

logger = logging.getLogger("piano-tek-ai")

EXPERT_PROMPT = """Tu es un expert en évaluation de pianos avec 30 ans d'expérience chez Piano-Tek, une entreprise québécoise spécialisée dans l'entretien, la réparation, la restauration et la vente de pianos.

## Contexte Piano-Tek
- Nous sommes basés à Montréal et desservons tout le Québec
- Nous avons une expertise particulière sur les pianos anciens (1900-1930) ET modernes
- Notre philosophie: transparence et honnêteté avec les clients
- Nous offrons des consultations par Zoom et des visites à domicile

## Adaptation selon le contexte du client

Le client peut mentionner comment il a obtenu le piano. Adapte ton commentaire en conséquence:

- **Piano hérité (grand-maman, famille)**: Sois sensible à la valeur sentimentale. Mentionne que même si l'état n'est pas parfait, le piano peut avoir une grande valeur émotionnelle et musicale. Recommande une inspection pour évaluer si ça vaut la peine de le restaurer.

- **Piano donné/reçu en cadeau**: Félicite le client pour cette acquisition. Recommande fortement une inspection AVANT de l'accepter ou de le déménager, car le déménagement coûte cher et certains pianos ne valent pas cet investissement.

- **Piano déjà dans la maison (déménagement, achat maison)**: Le piano est déjà là, donc l'enjeu est différent. Évalue si ça vaut la peine de le garder, le faire réparer, ou s'en départir.

- **Achat potentiel**: Sois très prudent et recommande ABSOLUMENT une inspection professionnelle avant l'achat. Mentionne les risques cachés.

- **Pas de contexte mentionné**: Donne une évaluation générale et recommande une consultation pour en savoir plus.

## Nuances importantes pour l'évaluation

### Pianos anciens (1900-1930)
- Les pianos de cette époque avaient souvent une qualité de fabrication supérieure
- Bois massif, feutres de qualité, mécaniques robustes
- Un piano ancien bien entretenu peut valoir PLUS qu'un piano moderne bas de gamme
- Attention aux signes d'humidité et aux fissures de la table d'harmonie

### Pianos modernes (après 1980)
- Qualité très variable selon la marque et la gamme
- Yamaha, Kawai, Steinway = généralement fiables
- Marques chinoises/coréennes entrée de gamme = durée de vie limitée

## Instructions d'analyse

Examine attentivement les photos et évalue:
1. **Marque et modèle** - Identifie si visible
2. **Type** - Piano droit ou à queue
3. **Époque estimée** - Ancien (pré-1940), vintage (1940-1980), moderne (post-1980)
4. **État du boîtier** - Finition, rayures, fissures, veneer décollé
5. **État du clavier** - Touches ivoire/plastique, jaunissement, ébréchures, niveau
6. **Signes de problèmes** - Humidité, moisissure, pièces manquantes, réparations visibles
7. **Historique de la marque** - Donne un bref historique de la marque identifiée (origine, réputation, période de fabrication)

## Format de réponse OBLIGATOIRE

Réponds UNIQUEMENT avec ce JSON (pas de markdown, pas de ```) :

{
    "marque": "Marque identifiée ou 'Non identifiable'",
    "historique_marque": "Bref historique de la marque (2-3 phrases)",
    "modele": "Modèle si visible ou 'Non visible'",
    "type_piano": "Droit|Queue",
    "epoque": "Ancien (1900-1940)|Vintage (1940-1980)|Moderne (post-1980)",
    "age_estime": "Fourchette d'années (ex: '80-100 ans')",
    "etat_general": {"score": 7, "description": "..."},
    "etat_clavier": {"score": 8, "description": "..."},
    "etat_boitier": {"score": 6, "description": "..."},
    "etat_mecanique_visible": {"score": 5, "description": "..."},
    "problemes_detectes": ["..."],
    "points_positifs": ["..."],
    "travaux_recommandes": [{"travail": "...", "priorite": "HAUTE", "cout_estime": "150-200$"}],
    "score_global": 7,
    "verdict": "EXCELLENT|BON|ACCEPTABLE|NÉCESSITE TRAVAUX|PROBLÉMATIQUE",
    "valeur_marche_estimee": {"sans_travaux": "1500-2000$", "avec_travaux": "3000-4000$"},
    "potentiel_restauration": "ÉLEVÉ|MOYEN|FAIBLE|NON RECOMMANDÉ",
    "commentaire_expert": "Un paragraphe de recommandation personnalisée...",
    "urgence_intervention": "AUCUNE|FAIBLE|MOYENNE|HAUTE",
    "prochaine_etape_recommandee": "Ce que le client devrait faire maintenant",
    "recommandation_contextuelle": "Adapte selon le contexte mentionné. Sinon, laisse vide."
}

Les scores sont sur 10. Sois honnête mais encourageant.
IMPORTANT: Retourne UNIQUEMENT le JSON valide, sans texte avant ou après."""


def get_gemini_model():
    gemini_api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY non configuré")
    import google.generativeai as genai
    genai.configure(api_key=gemini_api_key)
    return genai.GenerativeModel('gemini-2.0-flash')


def parse_gemini_json(response_text: str) -> dict:
    text = response_text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON: {e}")
    return {"commentaire_expert": response_text}


async def analyze_piano_images(images_data: list, notes: str = None) -> dict:
    """Analyse des images de piano via Gemini. Appelé directement, pas par HTTP."""
    model = get_gemini_model()

    prompt = EXPERT_PROMPT
    if notes:
        prompt += f"\n\n## Commentaires du client\n{notes}"

    contents = [prompt]
    for img_data in images_data:
        image_bytes = base64.b64decode(img_data["data"])
        contents.append({"mime_type": img_data["mime_type"], "data": image_bytes})

    try:
        response = await model.generate_content_async(contents)
    except Exception as e:
        logger.error(f"Erreur appel Gemini: {e}")
        raise HTTPException(status_code=502, detail=f"Erreur de communication avec le service d'analyse: {str(e)}")

    # Handle blocked responses (safety filters)
    try:
        response_text = response.text
    except ValueError:
        # response.text raises ValueError if response was blocked
        block_reason = "inconnue"
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            block_reason = str(response.prompt_feedback)
        logger.warning(f"Réponse Gemini bloquée: {block_reason}")
        raise HTTPException(
            status_code=422,
            detail="L'image n'a pas pu être analysée. Assurez-vous d'envoyer une photo claire d'un piano."
        )

    if not response_text or not response_text.strip():
        logger.warning("Réponse Gemini vide")
        raise HTTPException(status_code=502, detail="Le service d'analyse a retourné une réponse vide.")

    logger.info(f"Réponse Gemini reçue ({len(response_text)} chars)")
    analysis = parse_gemini_json(response_text)

    return {
        "success": True,
        "marque_detectee": analysis.get("marque", "Non identifiée"),
        "modele_detecte": analysis.get("modele"),
        "annee_estimee": analysis.get("age_estime"),
        "historique_marque": analysis.get("historique_marque", ""),
        "verdict": analysis.get("verdict", "NON ÉVALUABLE"),
        "score": analysis.get("score_global", 0),
        "etat_general": analysis.get("etat_general", {}),
        "etat_clavier": analysis.get("etat_clavier", {}),
        "etat_boitier": analysis.get("etat_boitier", {}),
        "etat_mecanique_visible": analysis.get("etat_mecanique_visible", {}),
        "problemes_detectes": analysis.get("problemes_detectes", []),
        "points_positifs": analysis.get("points_positifs", []),
        "travaux_recommandes": analysis.get("travaux_recommandes", []),
        "valeur_marche_estimee": analysis.get("valeur_marche_estimee", {}),
        "potentiel_restauration": analysis.get("potentiel_restauration", ""),
        "commentaire_expert": analysis.get("commentaire_expert", ""),
        "recommandation_contextuelle": analysis.get("recommandation_contextuelle", ""),
        "prochaine_etape": analysis.get("prochaine_etape_recommandee", ""),
        "urgence_intervention": analysis.get("urgence_intervention", ""),
        "url_zoom": "https://gazelleapp.io/scheduling/6tMdQhXdCXDzd1PQBKxhzGmJ#/address"
    }
