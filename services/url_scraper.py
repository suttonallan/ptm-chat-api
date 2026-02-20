import re
import httpx
from bs4 import BeautifulSoup
from typing import Optional, Dict, List

# Regex to find URLs in a message
URL_PATTERN = re.compile(
    r'https?://[^\s<>\"\'\)]+',
    re.IGNORECASE,
)

# Supported classified-ad sites (extend as needed)
SUPPORTED_DOMAINS = [
    "kijiji.ca",
    "facebook.com/marketplace",
    "marketplace.facebook.com",
    "lespac.com",
    "craigslist.org",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-CA,fr;q=0.9,en;q=0.8",
}


def find_urls(text: str) -> List[str]:
    """Extract all URLs from a text string."""
    return URL_PATTERN.findall(text)


def _is_supported(url: str) -> bool:
    """Check if a URL belongs to a supported classified-ad site."""
    lower = url.lower()
    return any(domain in lower for domain in SUPPORTED_DOMAINS)


def _extract_og_images(soup: BeautifulSoup) -> List[str]:
    """Extract Open Graph image URLs from meta tags."""
    images = []
    for tag in soup.find_all("meta", property="og:image"):
        content = tag.get("content", "")
        if content:
            images.append(content)
    return images


def _extract_kijiji(soup: BeautifulSoup, url: str) -> Dict:
    """Extract listing data from a Kijiji page."""
    data: Dict = {"source": "Kijiji", "url": url}

    # Title
    title_tag = soup.find("h1")
    if title_tag:
        data["title"] = title_tag.get_text(strip=True)

    # Price
    price_span = soup.find("span", string=re.compile(r'\$'))
    if not price_span:
        price_span = soup.find(string=re.compile(r'\d+[\s,.]?\d*\s*\$'))
    if price_span:
        data["price"] = price_span.get_text(strip=True) if hasattr(price_span, 'get_text') else str(price_span).strip()

    # Description
    desc_div = soup.find("div", {"itemprop": "description"})
    if not desc_div:
        desc_div = soup.find("div", class_=re.compile(r'description', re.I))
    if desc_div:
        data["description"] = desc_div.get_text(separator="\n", strip=True)[:1500]

    # Location
    loc = soup.find("span", {"itemprop": "address"})
    if not loc:
        loc = soup.find(string=re.compile(r'(Montréal|Montreal|Laval|Longueuil|Québec)', re.I))
    if loc:
        data["location"] = loc.get_text(strip=True) if hasattr(loc, 'get_text') else str(loc).strip()

    # Images
    data["images"] = _extract_og_images(soup)

    return data


def _extract_generic(soup: BeautifulSoup, url: str) -> Dict:
    """Generic extractor for any classified-ad page."""
    data: Dict = {"source": "annonce", "url": url}

    # Title from og:title or <title>
    og_title = soup.find("meta", property="og:title")
    if og_title:
        data["title"] = og_title.get("content", "")
    elif soup.title:
        data["title"] = soup.title.get_text(strip=True)

    # Price from og:price or text
    og_price = soup.find("meta", property="product:price:amount")
    if og_price:
        currency = soup.find("meta", property="product:price:currency")
        data["price"] = f"{og_price.get('content', '')} {currency.get('content', '') if currency else ''}".strip()

    # Description from og:description or meta description
    og_desc = soup.find("meta", property="og:description")
    if not og_desc:
        og_desc = soup.find("meta", attrs={"name": "description"})
    if og_desc:
        data["description"] = og_desc.get("content", "")[:1500]

    # Images
    data["images"] = _extract_og_images(soup)

    return data


async def scrape_listing(url: str) -> Optional[Dict]:
    """
    Fetch and parse a classified-ad URL.
    Returns a dict with title, price, description, images, etc.
    Returns None if the page can't be fetched.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(url, headers=HEADERS)
            resp.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException):
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    if "kijiji.ca" in url.lower():
        return _extract_kijiji(soup, url)
    else:
        return _extract_generic(soup, url)


def format_listing_context(listing: Dict) -> str:
    """Format a scraped listing into a context string for GPT-4o."""
    parts = [f"Le client a partagé une annonce ({listing.get('source', 'web')}) :"]

    if listing.get("title"):
        parts.append(f"Titre : {listing['title']}")
    if listing.get("price"):
        parts.append(f"Prix demandé : {listing['price']}")
    if listing.get("location"):
        parts.append(f"Emplacement : {listing['location']}")
    if listing.get("description"):
        parts.append(f"Description : {listing['description']}")
    if listing.get("images"):
        parts.append(f"Photos dans l'annonce : {len(listing['images'])}")
    parts.append(f"Lien : {listing['url']}")

    return "\n".join(parts)
