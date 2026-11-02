"""Offline evidence extraction module.
USED FOR AUDIT AND DATA REPAIR ONLY. NEVER IMPORTED BY GENERATION.PY.

Implements high-precision extractors using Decimal arithmetic, calendar math,
and strict candidate distinction.
"""

from typing import List, Optional, Tuple, Dict, Any
from decimal import Decimal, InvalidOperation
import datetime
import re

# Regex patterns for evidence detection
CURRENCY_PREFIX = r"(?:₹|Rs\.?|INR)\s*"
AMOUNT_PATTERN = re.compile(
    r"(?:(?P<prefix>₹|Rs\.?|INR)\s*)?(?P<num>[0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*(?P<scale>k|lakh)?\b",
    re.IGNORECASE,
)

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
    """Extract Received date from ticket header: 'Received: 2026-08-14 (Fri)'."""
    m = re.search(r"Received:\s*(\d{4}-\d{2}-\d{2})", text)
    if m:
        try:
            return datetime.date.fromisoformat(m.group(1))
        except ValueError:
            pass
    return None


def extract_non_quoted_lines(text: str) -> str:
    """Return text excluding lines starting with '>' (quoted earlier emails)."""
    clean_lines = []
    for line in text.splitlines():
        if line.strip().startswith(">"):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)


def has_legal_threat(text: str) -> bool:
    """Check if the text contains explicit legal threats per SCHEMA.md."""
    non_quoted = extract_non_quoted_lines(text)
    for pat in LEGAL_THREAT_KEYWORDS:
        if re.search(pat, non_quoted, re.IGNORECASE):
            return True
    return False


def has_prior_support_contact(text: str) -> bool:
    """Check if customer says they already contacted support before."""
    non_quoted = extract_non_quoted_lines(text)
    for pat in PRIOR_CONTACT_PATTERNS:
        if re.search(pat, non_quoted, re.IGNORECASE):
            return True
    return False


def is_hinglish(text: str) -> bool:
    """Detect Hinglish (Hindi in Latin script) in customer text."""
    non_quoted = extract_non_quoted_lines(text)
    # Exclude headers like 'Received: ... Source: ...'
    parts = non_quoted.split("\n\n", 1)
    body = parts[1] if len(parts) > 1 else non_quoted

    matches = 0
    for pat in HINGLISH_KEYWORDS:
        if re.search(pat, body, re.IGNORECASE):
            matches += 1
    return matches >= 2


def find_canonical_np_txn_id(text: str) -> Optional[str]:
    """Find and canonicalize a NimbusPay transaction ID from non-quoted text.
    Returns 'NP' + 10 digits, or None.
    """
    non_quoted = extract_non_quoted_lines(text)
    # Check for NP prefix with 10 digits
    m = NP_TXN_PATTERN.search(non_quoted)
    if m:
        if m.group(1):
            return f"NP{m.group(1)}"
        elif m.group(2) and m.group(3):
            return f"NP{m.group(2)}{m.group(3)}"
    return None


def parse_amount_to_paise(val_str: str, scale: Optional[str] = None) -> Optional[int]:
    """Convert amount string with optional scale ('k', 'lakh') to exact integer paise."""
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
    paise_decimal = d * Decimal(100)
    # Must be integral paise
    return int(paise_decimal)


def find_amount_candidates(text: str) -> List[Dict[str, Any]]:
    """Find disputed amount candidates in non-quoted text, filtering out obvious distractors."""
    non_quoted = extract_non_quoted_lines(text)
    candidates = []

    for m in AMOUNT_PATTERN.finditer(non_quoted):
        prefix = m.group("prefix")
        num_str = m.group("num")
        scale = m.group("scale")
        span = m.span()

        # Context inspection for distractors (balance, limit, fee)
        start = max(0, span[0] - 30)
        end = min(len(non_quoted), span[1] + 30)
        context = non_quoted[start:end].lower()

        is_distractor = False
        if any(d in context for d in ["balance", "account balance", "daily limit", "limit of", "fee of", "fee ₹", "charges of"]):
            # If the context specifically qualifies this number as balance or fee
            if "balance" in context or "limit" in context or "fee" in context:
                is_distractor = True

        paise = parse_amount_to_paise(num_str, scale)
        if paise is not None and paise > 0:
            candidates.append({
                "raw": m.group(0),
                "paise": paise,
                "prefix": prefix,
                "is_distractor": is_distractor,
                "context": context,
            })

    return candidates


def resolve_relative_date(date_phrase: str, received_date: datetime.date) -> Optional[datetime.date]:
    """Resolve relative date strings against received_date."""
    norm = date_phrase.strip().lower()
    if norm in {"yesterday", "kal", "beete kal"}:
        return received_date - datetime.timedelta(days=1)
    if norm in {"parso", "day before yesterday", "2 days ago", "2 din pehle"}:
        return received_date - datetime.timedelta(days=2)
    m = re.match(r"(\d+)\s+(?:days?|din)\s+(?:ago|pehle)", norm)
    if m:
        days = int(m.group(1))
        return received_date - datetime.timedelta(days=days)
    if norm in {"today", "aaj"}:
        return received_date
    return None
