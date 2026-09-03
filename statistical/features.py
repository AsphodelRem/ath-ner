from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from statistical.common import (
    KNOWN_ATTACHED_SUFFIXES,
    LABELS,
    LexiconEntry,
    Token,
    learn_suffixes,
    script_name,
    word_shape,
)

TERMINAL = ""


@dataclass(frozen=True)
class GazetteerMatch:
    start: int
    end: int
    entry: LexiconEntry


class FeatureExtractor:
    """Produces local, morphology and train-only gazetteer CRF features."""

    def __init__(
        self,
        lexicon: dict[tuple[str, ...], LexiconEntry],
        *,
        gazetteer_min_count: int = 2,
        gazetteer_min_purity: float = 0.9,
        suffix_min_support: int = 4,
        suffix_policy: str = "learned",
    ) -> None:
        self.lexicon = lexicon
        self.gazetteer_min_count = gazetteer_min_count
        self.gazetteer_min_purity = gazetteer_min_purity
        self.suffix_min_support = suffix_min_support
        if suffix_policy not in {"learned", "uzbek_whitelist", "none"}:
            raise ValueError(f"unsupported suffix policy: {suffix_policy}")
        self.suffix_policy = suffix_policy
        self.trie = self._build_trie(lexicon)
        learned = learn_suffixes(lexicon, max_suffix_length=8)
        self.suffixes_by_last: dict[str, list[tuple[str, str, int]]] = {}
        for label in LABELS:
            eligible = [
                (suffix, label, support)
                for suffix, support in learned[label].items()
                if support >= suffix_min_support and suffix.isalpha()
                and (
                    suffix_policy == "learned"
                    or suffix in KNOWN_ATTACHED_SUFFIXES
                )
            ]
            if suffix_policy == "none":
                eligible = []
            for item in sorted(eligible, key=lambda row: (-len(row[0]), -row[2], row[0]))[:200]:
                self.suffixes_by_last.setdefault(item[0][-1], []).append(item)

    @staticmethod
    def _build_trie(lexicon: dict[tuple[str, ...], LexiconEntry]) -> dict[str, Any]:
        trie: dict[str, Any] = {}
        for key, entry in lexicon.items():
            node = trie
            for token in key:
                node = node.setdefault(token, {})
            node[TERMINAL] = entry
        return trie

    def _gazetteer_roles(self, tokens: Sequence[Token]) -> list[dict[str, Any]]:
        candidates: list[GazetteerMatch] = []
        for start in range(len(tokens)):
            node = self.trie
            for end in range(start, min(start + 20, len(tokens))):
                child = node.get(tokens[end].norm)
                if not isinstance(child, dict):
                    break
                node = child
                entry = node.get(TERMINAL)
                if (
                    isinstance(entry, LexiconEntry)
                    and entry.count >= self.gazetteer_min_count
                    and entry.purity >= self.gazetteer_min_purity
                ):
                    candidates.append(GazetteerMatch(start, end + 1, entry))

        selected: list[GazetteerMatch] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (
                -(item.end - item.start),
                -item.entry.purity,
                -item.entry.count,
                item.start,
            ),
        ):
            if any(candidate.start < old.end and old.start < candidate.end for old in selected):
                continue
            selected.append(candidate)

        roles: list[dict[str, Any]] = [{} for _ in tokens]
        for match in selected:
            length = match.end - match.start
            for index in range(match.start, match.end):
                if length == 1:
                    role = "U"
                elif index == match.start:
                    role = "B"
                elif index == match.end - 1:
                    role = "L"
                else:
                    role = "I"
                roles[index] = {
                    "gaz.role": f"{role}-{match.entry.label}",
                    "gaz.label": match.entry.label,
                    "gaz.count": min(6, int(math.log2(match.entry.count + 1))),
                    "gaz.purity": round(match.entry.purity, 1),
                }
        return roles

    def _morphology_features(self, token: Token) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if len(token.norm) < 3:
            return result
        matches = []
        for suffix, label, support in self.suffixes_by_last.get(token.norm[-1], []):
            if token.norm.endswith(suffix) and len(token.norm) > len(suffix) + 1:
                matches.append((suffix, label, support))
        for suffix, label, support in matches[:4]:
            result[f"morph.suffix={label}:{suffix}"] = True
            result[f"morph.label={label}"] = True
            result[f"morph.support={label}"] = min(6, int(math.log2(support + 1)))
        return result

    def transform(self, tokens: Sequence[Token]) -> list[dict[str, Any]]:
        gazetteer = self._gazetteer_roles(tokens)
        result: list[dict[str, Any]] = []
        for index, token in enumerate(tokens):
            norm = token.norm
            features: dict[str, Any] = {
                "bias": 1.0,
                "word.norm": norm,
                "word.shape": word_shape(token.text),
                "word.script": script_name(token.text),
                "word.length": min(len(norm), 24),
                "word.isupper": token.text.isupper(),
                "word.istitle": token.text.istitle(),
                "word.islower": token.text.islower(),
                "word.isdigit": token.text.isdigit(),
                "word.has_digit": any(char.isdigit() for char in token.text),
                "word.has_hyphen": any(char in "-–—" for char in token.text),
                "word.has_apostrophe": "'" in norm,
                "word.is_punct": not any(char.isalnum() for char in token.text),
                **gazetteer[index],
                **self._morphology_features(token),
            }
            for length in range(1, min(5, len(norm)) + 1):
                features[f"prefix{length}"] = norm[:length]
                features[f"suffix{length}"] = norm[-length:]

            if index == 0:
                features["BOS"] = True
            else:
                previous = tokens[index - 1]
                features.update(
                    {
                        "-1:norm": previous.norm,
                        "-1:shape": word_shape(previous.text),
                        "-1:script": script_name(previous.text),
                        "-1:istitle": previous.text.istitle(),
                        "-1+0:norm": f"{previous.norm}|{norm}",
                        "gap.before": token.start > previous.end,
                    }
                )
            if index + 1 == len(tokens):
                features["EOS"] = True
            else:
                following = tokens[index + 1]
                features.update(
                    {
                        "+1:norm": following.norm,
                        "+1:shape": word_shape(following.text),
                        "+1:script": script_name(following.text),
                        "+1:istitle": following.text.istitle(),
                        "0+1:norm": f"{norm}|{following.norm}",
                    }
                )
            if index >= 2:
                features["-2:norm"] = tokens[index - 2].norm
            if index + 2 < len(tokens):
                features["+2:norm"] = tokens[index + 2].norm
            result.append(features)
        return result
