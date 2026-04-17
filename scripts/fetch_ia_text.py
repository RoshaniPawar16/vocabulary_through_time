"""
Internet Archive plain-text fetcher.

Tries DjVuTXT first (highest quality OCR layer), falls back to plain .txt.
Returns cleaned text trimmed to [min_chars, max_chars], or None on failure.
"""
import re
import requests
from typing import Optional


# Patterns that indicate page/section headers rather than body text
_NOISE_LINE_RE = re.compile(
    r"^[\s\d\.\-–—]+$"              # lines that are only numbers/punctuation
    r"|^\s{0,4}[IVXLCDM]+\s*$"      # roman numerals alone on a line
)

# Repeated-header detector: if a short line (≤60 chars) appears 4+ times, it's a running header
def _build_repeated_line_set(lines: list[str], max_len: int = 60, min_count: int = 4) -> set[str]:
    from collections import Counter
    short = [ln.strip() for ln in lines if 0 < len(ln.strip()) <= max_len]
    counts = Counter(short)
    return {line for line, count in counts.items() if count >= min_count}


def _clean_ia_text(raw: str, max_chars: int) -> str:
    """
    Remove DjVuTXT boilerplate and OCR noise; join hyphenated line-breaks;
    collapse whitespace; truncate to max_chars.
    """
    # Remove DjVu page-break markers
    text = re.sub(r"\x0c", "\n", raw)                    # form-feed → newline
    text = re.sub(r"\ufffd|\x00", "", text)               # replacement chars / nulls

    # Join soft-hyphen line breaks: "some-\nthing" → "something"
    text = re.sub(r"-\n(\s*)", "", text)

    lines = text.split("\n")

    # Identify and strip running headers/footers
    repeated = _build_repeated_line_set(lines)

    cleaned_lines = []
    for ln in lines:
        stripped = ln.strip()
        if stripped in repeated:
            continue
        if _NOISE_LINE_RE.match(stripped) and len(stripped) < 10:
            continue
        cleaned_lines.append(stripped)

    # Collapse multiple blank lines → single blank
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines))
    result = result.strip()

    # Truncate to max_chars at a word boundary
    if len(result) > max_chars:
        cutoff = result.rfind(" ", 0, max_chars)
        result = result[:cutoff if cutoff > 0 else max_chars]

    return result


def _get_ia_files(session: requests.Session, identifier: str) -> list[dict]:
    """Return list of file dicts from IA metadata API."""
    try:
        r = session.get(f"https://archive.org/metadata/{identifier}", timeout=30)
        if r.status_code == 200:
            return r.json().get("files", [])
    except (requests.RequestException, ValueError):
        pass
    return []


def fetch_ia_text(
    identifier: str,
    min_chars: int = 5_000,
    max_chars: int = 50_000,
) -> Optional[str]:
    """
    Download plain text for an Internet Archive item.

    Strategy:
      1. DjVuTXT via metadata API (finds correct filename, e.g. inner_djvu.txt)
      2. Fallback: {identifier}_djvu.txt (common shorthand)
      3. Fallback: plain .txt files listed in metadata

    Returns cleaned text string if len >= min_chars, else None.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "vocab-through-time-research/1.0"})

    raw: Optional[str] = None
    files = _get_ia_files(session, identifier)

    # --- Strategy 1: DjVuTXT via metadata (correct filename) ---
    djvu_names = [
        f["name"] for f in files
        if f.get("name", "").lower().endswith("_djvu.txt")
    ]
    for fname in djvu_names[:2]:
        url = f"https://archive.org/download/{identifier}/{fname}"
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 200 and len(resp.text) > 1000:
                raw = resp.text
                break
        except requests.RequestException:
            continue

    # --- Strategy 2: DjVuTXT shorthand ---
    if not raw:
        url = f"https://archive.org/download/{identifier}/{identifier}_djvu.txt"
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 200 and len(resp.text) > 1000:
                raw = resp.text
        except requests.RequestException:
            pass

    # --- Strategy 3: plain .txt files ---
    if not raw:
        txt_names = [
            f["name"] for f in files
            if f.get("name", "").endswith(".txt")
            and "djvu" not in f.get("name", "").lower()
            and "meta" not in f.get("name", "").lower()
        ]
        for fname in txt_names[:3]:
            url = f"https://archive.org/download/{identifier}/{fname}"
            try:
                r = session.get(url, timeout=60)
                if r.status_code == 200 and len(r.text) > 1000:
                    raw = r.text
                    break
            except requests.RequestException:
                continue

    if not raw:
        return None

    cleaned = _clean_ia_text(raw, max_chars)
    if len(cleaned) < min_chars:
        return None

    return cleaned


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 'GalaxyMarch1955' is not a valid IA identifier; the real one is below.
    TEST_ID = "galaxymagazine-1955-03"

    print(f"Testing fetch_ia_text('{TEST_ID}', min_chars=5000, max_chars=50000)")
    print("-" * 60)

    # Show raw download size before cleaning
    try:
        import requests as _r
        files_meta = _r.get(f"https://archive.org/metadata/{TEST_ID}", timeout=30).json().get("files", [])
        djvu_name = next((f["name"] for f in files_meta if f.get("name","").endswith("_djvu.txt")), None)
        if djvu_name:
            djvu_url = f"https://archive.org/download/{TEST_ID}/{djvu_name}"
            print(f"DjVuTXT URL: {djvu_url}")
            raw_resp = _r.get(djvu_url, timeout=60)
            raw_size = len(raw_resp.content)
            print(f"Raw download size: {raw_size:,} bytes  (HTTP {raw_resp.status_code})")
        else:
            print("No _djvu.txt file found in metadata")
            raw_size = 0
    except Exception as e:
        print(f"Raw size check failed: {e}")
        raw_size = 0

    result = fetch_ia_text(TEST_ID, min_chars=5000, max_chars=50000)

    if result is None:
        print("RESULT: None (fetch failed or below min_chars)")
    else:
        print(f"Cleaned text length: {len(result):,} chars")
        print(f"min_chars=5000 pass: {'YES' if len(result) >= 5000 else 'NO'}")
        print("\n--- First 200 chars ---")
        print(repr(result[:200]))
        print("\n--- Last 200 chars ---")
        print(repr(result[-200:]))
