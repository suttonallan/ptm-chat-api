import re
import json
import base64
import logging
import httpx
from bs4 import BeautifulSoup
from typing import Optional, Dict, List

logger = logging.getLogger("piano-tek-ai")

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
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-CA,fr;q=0.9,en-CA;q=0.8,en-US;q=0.7,en;q=0.6",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}

# Max images to download and analyze from a listing
MAX_LISTING_IMAGES = 3


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


def _extract_kijiji_images(soup: BeautifulSoup) -> List[str]:
    """
    Extract all listing images from a Kijiji page.
    Tries multiple strategies: og:image, gallery images, JSON-LD, and img tags.
    """
    images: List[str] = []
    seen: set = set()

    def _add(url: str):
        if url and url not in seen and url.startswith("http"):
            seen.add(url)
            images.append(url)

    # 1. OpenGraph images
    for tag in soup.find_all("meta", property="og:image"):
        _add(tag.get("content", ""))

    # 2. JSON-LD structured data (often has image arrays)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            ld_images = data.get("image", [])
            if isinstance(ld_images, str):
                _add(ld_images)
            elif isinstance(ld_images, list):
                for img in ld_images:
                    _add(img if isinstance(img, str) else img.get("url", ""))
        except (json.JSONDecodeError, AttributeError):
            pass

    # 3. Image URLs embedded in script blocks (common in modern Kijiji)
    for script in soup.find_all("script"):
        if script.string:
            # Look for image URLs in JS data
            urls = re.findall(
                r'https?://[^\"\s\'\\]+\.(?:jpg|jpeg|png|webp)',
                script.string,
            )
            for u in urls:
                # Filter to likely listing images (not icons, logos, etc.)
                if any(kw in u.lower() for kw in ["kijiji", "nebula", "classistatic", "images"]):
                    _add(u)

    # 4. Gallery img tags with data-src (lazy loading) or src
    for img in soup.find_all("img"):
        for attr in ("data-src", "src", "data-lazy-src"):
            src = img.get(attr, "")
            if src and any(kw in src.lower() for kw in ["kijiji", "nebula", "classistatic"]):
                _add(src)

    # 5. picture > source tags (srcset)
    for source in soup.find_all("source"):
        srcset = source.get("srcset", "")
        if srcset:
            # srcset can have multiple URLs with size descriptors
            for part in srcset.split(","):
                src_url = part.strip().split()[0]
                if any(kw in src_url.lower() for kw in ["kijiji", "nebula", "classistatic"]):
                    _add(src_url)

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

    # Images — enhanced extraction
    data["images"] = _extract_kijiji_images(soup)

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


async def download_listing_images(image_urls: List[str], max_images: int = MAX_LISTING_IMAGES) -> List[Dict]:
    """
    Download images from URLs and return them as base64-encoded dicts
    suitable for analyze_piano_images().

    Returns list of {"data": "<base64>", "mime_type": "image/jpeg"}.
    """
    images_data: List[Dict] = []
    urls_to_try = image_urls[:max_images + 2]  # Try a few extra in case some fail

    MIME_MAP = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        for img_url in urls_to_try:
            if len(images_data) >= max_images:
                break
            try:
                resp = await client.get(img_url, headers=HEADERS)
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                if "image/" in content_type:
                    mime_type = content_type.split(";")[0].strip()
                else:
                    # Guess from URL extension
                    ext = img_url.rsplit(".", 1)[-1].lower().split("?")[0]
                    mime_type = MIME_MAP.get(ext, "image/jpeg")

                b64 = base64.b64encode(resp.content).decode("utf-8")
                images_data.append({"data": b64, "mime_type": mime_type})
                logger.info(f"Image téléchargée: {img_url[:80]}... ({len(resp.content)} bytes)")
            except Exception as e:
                logger.warning(f"Impossible de télécharger l'image {img_url[:80]}: {e}")
                continue

    return images_data


async def scrape_listing(url: str) -> Optional[Dict]:
    """
    Fetch and parse a classified-ad URL.
    Returns a dict with title, price, description, images, etc.
    Returns None if the page can't be fetched.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(url, headers=HEADERS)
            logger.info(f"Scrape {url[:80]} → HTTP {resp.status_code}")
            resp.raise_for_status()
    except httpx.TimeoutException:
        logger.warning(f"Timeout lors du scraping de {url[:80]}")
        return None
    except httpx.HTTPError as e:
        logger.warning(f"Erreur HTTP lors du scraping de {url[:80]}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    if "kijiji.ca" in url.lower():
        listing = _extract_kijiji(soup, url)
    else:
        listing = _extract_generic(soup, url)

    logger.info(
        f"Scrape résultat: title={listing.get('title', 'N/A')!r}, "
        f"price={listing.get('price', 'N/A')!r}, "
        f"images={len(listing.get('images', []))}, "
        f"description={bool(listing.get('description'))}"
    )
    return listing


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
