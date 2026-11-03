"""Offline evidence extraction module.
USED FOR AUDIT AND DATA REPAIR ONLY. NEVER IMPORTED BY GENERATION.PY.

Implements high-precision extractors using Decimal arithmetic, calendar math,
strict token boundaries, and explicit header/quote separation.
"""

from typing import List, Optional, Tuple, Dict, Any
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import datetime
import re

LEGAL_THREAT_KEYWORDS = [
    r"\brbi\b",
    r"\bombudsman\b",
    r"\bconsumer\s+court\b",
    r"\blawyer\b",
    r"\blegal\s+notice\b",
    r"\blegal\s+action\b",
    r"\bcourt\b",
]

PRIOR_CONTACT_PATTERNS = [
    r"\bthird\s+time\b",
    r"\bsecond\s+time\b",
    r"\b4th\s+time\b|\bfourth\s+time\b",
    r"\bagain\s+and\s+again\b",
    r"\bbaar\s+baar\b",
    r"\bpehle\s+bhi\b",
    r"\bmultiple\s+times\b",
    r"\brepeatedly\b",
    r"\balready\s+contacted\b",
    r"\balready\s+raised\b",
    r"\bno\s+response\s+to\s+my\s+previous\b",
    r"\bprevious\s+complaint\b",
    r"\bfollow(?:ing)?\s+up\s+on\b",
]

HINGLISH_KEYWORDS = [
    r"\bbhai\b",
    r"\bpaise\b",
    r"\bkat\s+gaye\b",
    r"\bnahi\s+aaya\b|\bnahi\s+aaye\b|\bnahi\s+mile\b|\bnahi\s+hua\b",
    r"\bkaro\b",
    r"\bhai\b|\bhain\b",
    r"\bkya\b",
    r"\bhua\b",
    r"\bpehle\b",
    r"\bbaar\b",
    r"\bmera\b|\bmere\b|\bmeri\b",
    r"\bnamaste\b",
    r"\bshukriya\b",
    r"\bdikh\s+raha\b",
    r"\bjaan\s+sakta\b",
    r"\bwapas\b",
    r"\bkaise\b",
    r"\bmadad\b",
]

NP_TXN_PATTERN = re.compile(
    r"\b(?:NP|np)[\s\-_]*(\d{10})\b|\b(?:NP|np)[\s\-_]*(\d{5})[\s\-_]*(\d{5})\b"
)

CHANNEL_KEYWORDS = {
    "upi": [r"\bupi\b", r"\bgpay\b", r"\bphonepe\b", r"\bpaytm\b"],
    "card": [r"\bcredit\s+card\b", r"\bdebit\s+card\b", r"\bvisa\b", r"\bmastercard\b", r"\bcard\s+payment\b", r"\bcard\b"],
    "netbanking": [r"\bnetbanking\b", r"\bnet\s+banking\b", r"\binternet\s+banking\b", r"\bonline\s+banking\b"],
    "wallet": [r"\bwallet\b", r"\bnimbus\s+wallet\b"],
}


def parse_received_date(text: str) -> Optional[datetime.date]:
    """Extract Received date from ticket header: 'Received: YYYY-MM-DD (weekday)'."""
    m = re.search(r"^Received:\s*(\d{4}-\d{2}-\d{2})", text, re.MULTILINE)
    if m:
        try:
            return datetime.date.fromisoformat(m.group(1))
        except ValueError:
            pass
    return None


def extract_customer_body(text: str) -> str:
    """Extract customer message body by stripping headers and quoted lines starting with '>'."""
    # Strip metadata headers: Received:... Source:...
    lines = text.splitlines()
    body_lines = []
    in_header = True

    for line in lines:
        stripped = line.strip()
        if in_header:
            if stripped.startswith("Received:") or stripped.startswith("Source:") or stripped.startswith("Subject:"):
                continue
            if not stripped:
                # Blank line following headers indicates transition to body
                in_header = False
                continue
            # If line doesn't start with known header, transition out
            in_header = False

        # Exclude quoted email chains
        if stripped.startswith(">"):
            continue
        body_lines.append(line)

    return "\n".join(body_lines).strip()


def extract_non_quoted_lines(text: str) -> str:
    """Return text excluding lines starting with '>' (quoted earlier emails)."""
    clean_lines = []
    for line in text.splitlines():
        if line.strip().startswith(">"):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)


def parse_amount_to_paise(val_str: str, scale: Optional[str] = None) -> Optional[int]:
    """Convert amount string with optional scale ('k', 'lakh') to exact integer paise using Decimal."""
    clean_val = val_str.replace(",", "").strip()
    try:
        d = Decimal(clean_val)
    except (InvalidOperation, ValueError):
        return None

    if scale:
        s = scale.lower()
        if s == "k":
            d = d * Decimal(1000)
        elif s == "lakh":
            d = d * Decimal(100000)

    # Convert rupees to paise (multiply by 100)
    paise_decimal = (d * Decimal(100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(paise_decimal)


def find_amount_candidates(text: str) -> List[Dict[str, Any]]:
    """Find disputed amount candidates in customer body.
    Strictly avoids matching header dates (e.g. 2026), phone numbers, or app versions.
    Requires explicit currency prefixes, suffixes, scale notations, or transactional cues.
    """
    body = extract_customer_body(text)
    candidates = []

    # Regex 1: Explicit currency prefix (₹, Rs., INR) followed by number and optional scale
    pat_prefix = re.compile(
        r"(?P<prefix>₹|Rs\.?|INR)\s*(?P<num>[0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*(?P<scale>k|lakh)?\b",
        re.IGNORECASE,
    )
    for m in pat_prefix.finditer(body):
        num_str = m.group("num")
        scale = m.group("scale")
        span = m.span()
        start = max(0, span[0] - 30)
        end = min(len(body), span[1] + 30)
        ctx = body[start:end].lower()

        is_distractor = any(d in ctx for d in ["balance", "account balance", "daily limit", "limit of", "fee of", "fee ₹", "charges of"])
        paise = parse_amount_to_paise(num_str, scale)
        if paise is not None and paise > 0:
            candidates.append({
                "raw": m.group(0),
                "paise": paise,
                "prefix": m.group("prefix"),
                "is_distractor": is_distractor,
                "context": ctx,
                "span": span,
            })

    # Regex 2: Number followed by currency suffix (rupees, rs, paise, /-) or scale (k, lakh)
    pat_suffix = re.compile(
        r"\b(?P<num>[0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*(?P<suffix>rupees?|rs|paise|/-|k|lakh)\b",
        re.IGNORECASE,
    )
    for m in pat_suffix.finditer(body):
        num_str = m.group("num")
        suffix = m.group("suffix").lower()
        span = m.span()
        start = max(0, span[0] - 30)
        end = min(len(body), span[1] + 30)
        ctx = body[start:end].lower()

        scale = None
        if suffix in {"k", "lakh"}:
            scale = suffix

        is_distractor = any(d in ctx for d in ["balance", "account balance", "daily limit", "limit of", "fee of", "charges of"])
        paise = parse_amount_to_paise(num_str, scale)
        if paise is not None and paise > 0:
            # Avoid duplicate spans
            if not any(abs(c["span"][0] - span[0]) < 5 for c in candidates):
                candidates.append({
                    "raw": m.group(0),
                    "paise": paise,
                    "prefix": None,
                    "is_distractor": is_distractor,
                    "context": ctx,
                    "span": span,
                })

    # Sort candidates by appearance in customer body
    candidates.sort(key=lambda x: x["span"][0])
    return candidates


def resolve_relative_date(date_phrase: str, received_date: datetime.date) -> Optional[datetime.date]:
    """Resolve relative date strings strictly anchored to received_date."""
    norm = date_phrase.strip().lower()
    if norm in {"yesterday", "kal", "beete kal"}:
        return received_date - datetime.timedelta(days=1)
    if norm in {"parso", "day before yesterday", "2 days ago", "2 din pehle"}:
        return received_date - datetime.timedelta(days=2)
    m = re.match(r"^(\d+)\s+(?:days?|din)\s+(?:ago|pehle)$", norm)
    if m:
        days = int(m.group(1))
        return received_date - datetime.timedelta(days=days)
    if norm in {"today", "aaj"}:
        return received_date
    return None


def extract_transaction_date(text: str, received_date: Optional[datetime.date]) -> Optional[str]:
    """Extract transaction date from customer body using exact word boundaries and date parsers.
    Never matches dates inside quoted earlier emails.
    """
    body = extract_customer_body(text)
    body_lower = body.lower()

    if received_date is not None:
        # Check relative dates with strict word boundaries
        # Check N days ago / N din pehle
        m_rel = re.search(r"\b(\d+)\s+(?:days?|din)\s+(?:ago|pehle)\b", body_lower)
        if m_rel:
            days = int(m_rel.group(1))
            return (received_date - datetime.timedelta(days=days)).isoformat()

        # Day before yesterday / parso
        if re.search(r"\b(?:parso|day before yesterday|2 days ago|2 din pehle)\b", body_lower):
            return (received_date - datetime.timedelta(days=2)).isoformat()

        # Yesterday / kal (STRICT WORD BOUNDARY: \bkal\b, never inside 'nikal')
        if re.search(r"\b(?:yesterday|kal|beete kal)\b", body_lower):
            return (received_date - datetime.timedelta(days=1)).isoformat()

        if re.search(r"\b(?:today|aaj)\b", body_lower):
            return received_date.isoformat()

    # Check explicit date formats in customer body:
    # 1. YYYY-MM-DD
    m_iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", body)
    if m_iso:
        try:
            d = datetime.date.fromisoformat(m_iso.group(1))
            return d.isoformat()
        except ValueError:
            pass

    # 2. DD-MM-YY or DD/MM/YY
    m_dmy2 = re.search(r"\b(\d{2})[-/](\d{2})[-/](\d{2})\b", body)
    if m_dmy2:
        day, month, year_2d = int(m_dmy2.group(1)), int(m_dmy2.group(2)), int(m_dmy2.group(3))
        full_year = 2000 + year_2d
        try:
            d = datetime.date(full_year, month, day)
            return d.isoformat()
        except ValueError:
            pass

    # 3. DD/MM/YYYY or DD-MM-YYYY
    m_dmy4 = re.search(r"\b(\d{2})[-/](\d{2})[-/](\d{4})\b", body)
    if m_dmy4:
        day, month, year = int(m_dmy4.group(1)), int(m_dmy4.group(2)), int(m_dmy4.group(3))
        try:
            d = datetime.date(year, month, day)
            return d.isoformat()
        except ValueError:
            pass

    return None


def find_canonical_np_txn_id(text: str) -> Optional[str]:
    """Find and canonicalize a NimbusPay transaction ID from customer body.
    Returns 'NP' + 10 digits, or None.
    """
    body = extract_customer_body(text)
    m = NP_TXN_PATTERN.search(body)
    if m:
        if m.group(1):
            return f"NP{m.group(1)}"
        elif m.group(2) and m.group(3):
            return f"NP{m.group(2)}{m.group(3)}"
    return None


def has_legal_threat(text: str) -> bool:
    """Check if the customer body contains explicit legal threats per SCHEMA.md."""
    body = extract_customer_body(text)
    for pat in LEGAL_THREAT_KEYWORDS:
        if re.search(pat, body, re.IGNORECASE):
            return True
    return False


def has_prior_support_contact(text: str) -> bool:
    """Check if customer body states they already contacted support before."""
    body = extract_customer_body(text)
    for pat in PRIOR_CONTACT_PATTERNS:
        if re.search(pat, body, re.IGNORECASE):
            return True
    return False


def is_hinglish(text: str) -> bool:
    """Detect Hinglish (Hindi in Latin script) in customer body."""
    body = extract_customer_body(text)
    matches = 0
    for pat in HINGLISH_KEYWORDS:
        if re.search(pat, body, re.IGNORECASE):
            matches += 1
    return matches >= 2
