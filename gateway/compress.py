"""Token Killer: shrink a chat prompt before it is routed.

Safe level (always on): collapse runs of spaces and blank lines, strip trailing spaces, drop a message that repeats the one
before it word for word, and drop lines repeated inside one message. Code blocks (``` fences) are left exactly as they are.
Aggressive level (opt-in, "compress": "aggressive" in the request or X-Compress: aggressive): also removes common filler
phrases ("could you please", "I was wondering if", ...). It can change tone, never facts, but it is still the caller's choice.

Savings are estimated at 4 characters per token (the engine's exact count is reported in usage after the answer).
"""
import re

FILLERS = [
    r"\bi was wondering if you could\b", r"\bi was wondering if\b", r"\bcould you please\b", r"\bcan you please\b", r"\bwould you please\b",
    r"\bplease note that\b", r"\bit is important to note that\b", r"\bi would like you to\b", r"\bi want you to\b",
    r"\bas an ai language model,?\b", r"\bthank you in advance\.?", r"\bthanks in advance\.?", r"\bif you don't mind,?\b",
    r"\bbasically,?\b", r"\bactually,?\b", r"\bkind of\b", r"\bsort of\b", r"\bjust\b(?= (?:wanted|need|want))",
]
FILLER_RE = re.compile("|".join(FILLERS), re.IGNORECASE)


def est_tokens(text):
    return (len(text) + 3) // 4


def _split_code(text):
    """[(is_code, part)] so code blocks are never touched."""
    parts, pos = [], 0
    for m in re.finditer(r"```.*?(?:```|$)", text, re.S):
        if m.start() > pos: parts.append((False, text[pos:m.start()]))
        parts.append((True, m.group(0)))
        pos = m.end()
    if pos < len(text): parts.append((False, text[pos:]))
    return parts


def _squeeze(text, aggressive):
    out = []
    for code, part in _split_code(text):
        if code:
            out.append(part); continue
        if aggressive:
            part = FILLER_RE.sub("", part)
        lines, seen = [], set()
        for line in part.split("\n"):
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line and len(line) > 20:   # long lines repeated inside a message are dropped; short ones ("ok", "-") are kept
                if line in seen: continue
                seen.add(line)
            lines.append(line)
        part = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
        if aggressive:
            part = re.sub(r" +([,.;:!?])", r"\1", part)
            part = re.sub(r"^[,;: ]+", "", part, flags=re.M)
        out.append(part)
    return "".join(out).strip()


def compress_messages(messages, aggressive=False):
    """(new messages, tokens before, tokens after)"""
    before = sum(est_tokens(m.get("content") or "") for m in messages if isinstance(m.get("content"), str))
    out = []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, str):   # multimodal parts etc.: untouched
            out.append(m); continue
        nm = dict(m); nm["content"] = _squeeze(c, aggressive)
        if out and out[-1].get("role") == nm.get("role") and out[-1].get("content") == nm["content"]:
            continue   # the same message twice in a row
        out.append(nm)
    after = sum(est_tokens(m.get("content") or "") for m in out if isinstance(m.get("content"), str))
    return out, before, after
