"""FAQ matching: TF-IDF cosine (question weighted 3x over answer), confidence gates, feedback boosting, gap logging."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .generate import QA

STOP = set("the a an of to in and or for on with is are be by can do does i my me you your how what when where why it this that there any".split())
SYN = {"cost": "price", "pricing": "price", "money": "refund", "reimburse": "refund", "reimbursement": "refund", "cancel": "plan", "upgrade": "plan", "downgrade": "plan",
       "2fa": "two-factor", "mfa": "two-factor", "login": "sign", "signin": "sign", "forgot": "reset", "download": "export", "backup": "export", "browser": "browsers"}


def toks(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()):
        if w in STOP:
            continue
        w = SYN.get(w, w)
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return out


@dataclass
class Reply:
    kind: str                      # answer | suggest | fallback
    text: str
    confidence: float
    pair: QA | None = None
    suggestions: list[QA] = field(default_factory=list)


class FAQBot:
    def __init__(self, pairs: list[QA], threshold: float = 0.45, suggest_at: float = 0.2, log_dir: str | Path | None = None):
        self.pairs, self.threshold, self.suggest_at = pairs, threshold, suggest_at
        self.dir = Path(log_dir) if log_dir else None
        df = Counter(w for p in pairs for w in set(toks(p.question + " " + p.answer)))
        n = len(pairs)
        self.idf = {w: math.log((n + 1) / (c + 0.5)) + 1 for w, c in df.items()}
        self.vecs = [self._vec(toks(p.question) * 3 + toks(p.answer)) for p in pairs]
        fb = self.dir / "feedback.json" if self.dir else None
        self.votes: dict[str, int] = json.loads(fb.read_text()) if fb and fb.exists() else {}

    def _vec(self, toks_: list[str]) -> dict[str, float]:
        tf = Counter(toks_)
        v = {w: (1 + math.log(c)) * self.idf.get(w, 1.0) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def _score(self, q: dict, i: int) -> float:
        v = self.vecs[i]
        s = sum(x * v.get(w, 0.0) for w, x in q.items())
        return s + min(0.1, max(-0.1, 0.03 * self.votes.get(str(self.pairs[i].id), 0)))

    def ask(self, question: str) -> Reply:
        q = self._vec(toks(question))
        ranked = sorted(((self._score(q, i), p) for i, p in enumerate(self.pairs)), key=lambda x: -x[0])
        if not q or not ranked or ranked[0][0] < self.suggest_at:
            self._log("unanswered.jsonl", {"question": question, "best": round(ranked[0][0], 3) if ranked else 0})
            return Reply("fallback", "I do not have an answer for that yet. I have noted it for the team - would you like to talk to a person?", ranked[0][0] if ranked else 0.0)
        best_s, best = ranked[0]
        if best_s >= self.threshold:
            return Reply("answer", best.answer, round(best_s, 2), best)
        self._log("unanswered.jsonl", {"question": question, "best": round(best_s, 3), "near": best.question})
        return Reply("suggest", "I am not sure I understood. Did you mean one of these?", round(best_s, 2), None, [p for s, p in ranked[:3] if s >= self.suggest_at])

    def feedback(self, pair_id: int, helpful: bool) -> None:
        self.votes[str(pair_id)] = self.votes.get(str(pair_id), 0) + (1 if helpful else -1)
        if self.dir:
            (self.dir / "feedback.json").write_text(json.dumps(self.votes))

    def _log(self, name: str, row: dict) -> None:
        if self.dir:
            with open(self.dir / name, "a") as f:
                f.write(json.dumps(row) + "\n")

    def gaps(self) -> list[tuple[str, int]]:
        """Most frequent unanswered questions (content gaps)."""
        p = self.dir / "unanswered.jsonl" if self.dir else None
        if not p or not p.exists():
            return []
        return Counter(json.loads(l)["question"].lower().strip("?! .") for l in p.read_text().splitlines()).most_common()


def save(pairs: list[QA], path: str | Path) -> None:
    Path(path).write_text(json.dumps([p.__dict__ for p in pairs], indent=1))


def load(path: str | Path) -> list[QA]:
    return [QA(**d) for d in json.loads(Path(path).read_text())]
