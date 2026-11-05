"""Rule-based FAQ extraction: headings, numbered procedures, definitions and explicit questions."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QA:
    id: int
    question: str
    answer: str
    source: str


GERUND = re.compile(r"^(?P<verb>[A-Za-z]+?)(?:ing)\s+(?P<rest>.+)$")
IRREGULAR = {"chang": "change", "sett": "set", "resett": "reset", "creat": "create", "delet": "delete", "manag": "manage", "us": "use", "mak": "make",
             "export": "export", "add": "add", "updat": "update", "enabl": "enable", "disabl": "disable", "invit": "invite", "remov": "remove", "configur": "configure"}


def _possessive(text: str) -> str:
    return re.sub(r"\byour\b", "my", text, flags=re.I)


def _lower_keep_acronyms(text: str) -> str:
    return " ".join(w if w.isupper() and len(w) > 1 else w.lower() for w in text.split())


def _question_from_heading(h: str, body: str = "") -> str:
    h = h.strip().rstrip(":")
    low = h.lower()
    if h.endswith("?"):
        return h
    if re.match(r"^(how|what|why|when|where|can|do|does|is|are)\b", low):
        return h + "?"
    if (m := GERUND.match(h)) and m["verb"].lower() in IRREGULAR:
        return f"How do I {IRREGULAR[m['verb'].lower()]} {_possessive(_lower_keep_acronyms(m['rest']))}?"
    if low.endswith("s") and not low.endswith("ss"):
        return f"What are the {low}?"
    if re.match(rf"^{re.escape(low)}\s+(is|are)\b", body.strip(), re.I):
        return f"What is {h if h.isupper() else low}?"
    if re.match(r"^(you|admins?|users?|owners?|click|open|go to|navigate)\b", body.strip(), re.I) and len(low.split()) <= 4 and not low.endswith("policy"):
        return f"How does {low} work?"
    return f"What is the {low}?"


def generate(text: str, source: str = "doc") -> list[QA]:
    pairs: list[tuple[str, str]] = []
    sections = re.split(r"^#{2,3}\s+", text, flags=re.M)
    for sec in sections[1:]:
        head, _, body = sec.partition("\n")
        body = body.strip()
        if body:
            answer = re.sub(r"\n(?=\d+\.)", "\n", body)
            pairs.append((_question_from_heading(head, body), " ".join(answer.split()) if not re.search(r"^\d+\.", body, re.M) else "\n".join(l.strip() for l in body.splitlines() if l.strip())))
    # definition sentences outside headed sections ("X is/are/lets ...")
    for para in re.split(r"\n\s*\n", sections[-1]):
        m = re.match(r"^([A-Z][\w\- ]{2,40}?) (is|are) (an? |the )?(.+?\.)\s*(.*)$", para.strip().replace("\n", " "))
        if m and len(m.group(1).split()) <= 4:
            term = m.group(1)
            pairs.append((f"What {'is' if m.group(2) == 'is' else 'are'} {term.lower() if term.split()[0].lower() not in ('api',) else term}?", para.strip().replace("\n", " ")))
    seen, out = set(), []
    for q, a in pairs:
        key = re.sub(r"\W+", " ", q.lower()).strip()
        if key not in seen:
            seen.add(key)
            out.append(QA(len(out) + 1, q, a, source))
    return out


def generate_folder(folder: str | Path) -> list[QA]:
    out: list[QA] = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in (".md", ".txt"):
            for qa in generate(p.read_text(errors="replace"), p.name):
                qa.id = len(out) + 1
                out.append(qa)
    return out
