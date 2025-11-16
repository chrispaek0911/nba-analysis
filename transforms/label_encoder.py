"""Utilities to convert play-level label metadata into numeric embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

DEFAULT_SPECIALS = ("<pad>", "<unk>", "<none>")
COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
SHOT_DISTANCE_MAX_FT = 35.0  # roughly logo range


def _to_token(value: Optional[str]) -> str:
    if value is None:
        return "<none>"
    if isinstance(value, str):
        token = value.strip()
        return token.lower() if token else "<none>"
    return str(value)


class Vocabulary:
    """Simple on-disk vocabulary with optional growth."""

    def __init__(
        self,
        name: str,
        path: Optional[Path] = None,
        specials: Iterable[str] = DEFAULT_SPECIALS,
        frozen: bool = False,
    ) -> None:
        self.name = name
        self.path = path
        self.frozen = frozen
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        if path and path.exists():
            self._load()
        else:
            for token in specials:
                self._add(token)

    def _add(self, token: str) -> int:
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx

    def _load(self) -> None:
        assert self.path is not None
        data = json.loads(self.path.read_text())
        for token in data:
            self._add(token)

    def save(self) -> None:
        if not self.path:
            return
        tokens = [self.id_to_token[idx] for idx in sorted(self.id_to_token)]
        self.path.write_text(json.dumps(tokens, ensure_ascii=False, indent=2))

    def encode(self, value: Optional[str]) -> int:
        token = _to_token(value)
        if token in self.token_to_id:
            return self.token_to_id[token]
        if self.frozen:
            return self.token_to_id.get("<unk>", 1)
        return self._add(token)


def _split_dict(
    features: Dict[str, float],
    labels: Dict[str, float],
    aux: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    return features, labels, aux


@dataclass
class PlayLabelEncoder:
    """Turn label_json records into dense numeric features."""

    vocab_dir: Path
    frozen: bool = False
    vocabs: Dict[str, Vocabulary] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.vocab_dir.mkdir(parents=True, exist_ok=True)
        for field in [
            "result_type",
            "shot_event_type",
            "shot_action_type",
            "shot_type",
            "shot_zone_basic",
            "shot_zone_area",
            "shot_zone_range",
            "rebound_type",
            "turnover_type",
            "foul_type",
        ]:
            path = self.vocab_dir / f"{field}.json"
            self.vocabs[field] = Vocabulary(field, path, frozen=self.frozen)

    def save(self) -> None:
        for vocab in self.vocabs.values():
            vocab.save()

    def encode(self, labels: Dict, context: Optional[Dict] = None) -> Dict[str, Dict[str, float]]:
        context = context or {}
        features: Dict[str, float] = {}
        label_targets: Dict[str, float] = {}
        aux_targets: Dict[str, float] = {}

        for feat, lab, aux in [
            self._encode_result(labels),
            self._encode_shot(labels),
            self._encode_rebound(labels),
            self._encode_turnover(labels),
            self._encode_foul(labels, context),
            self._encode_free_throws(labels),
        ]:
            features.update(feat)
            label_targets.update(lab)
            aux_targets.update(aux)
        return {"features": features, "labels": label_targets, "aux_targets": aux_targets}

    def _encode_result(self, labels: Dict) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        result = labels.get("result_type")
        return _split_dict(
            {"events_in_play": float(labels.get("events_in_play") or 0)},
            {"result_type_id": self.vocabs["result_type"].encode(result)},
            {},
        )

    def _encode_shot(self, labels: Dict) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        has_shot = labels.get("shot_event_id") is not None
        distance = labels.get("shot_distance")
        loc_x = labels.get("shot_loc_x")
        loc_y = labels.get("shot_loc_y")
        dist_norm = -1.0
        if distance is not None:
            dist_norm = min(1.0, float(distance) / SHOT_DISTANCE_MAX_FT)

        def _norm_coord(value: Optional[float], max_value: float) -> float:
            if value is None:
                return 0.0
            return float(value) / max_value

        loc_x_norm = _norm_coord(loc_x, COURT_LENGTH_FT)
        loc_y_norm = _norm_coord(loc_y, COURT_WIDTH_FT)
        features = {
            "shot_action_type_id": self.vocabs["shot_action_type"].encode(labels.get("shot_action_type")),
            "shot_event_type_id": self.vocabs["shot_event_type"].encode(labels.get("shot_event_type")),
            "shot_type_id": self.vocabs["shot_type"].encode(labels.get("shot_type")),
            "shot_zone_basic_id": self.vocabs["shot_zone_basic"].encode(labels.get("shot_zone_basic")),
            "shot_zone_area_id": self.vocabs["shot_zone_area"].encode(labels.get("shot_zone_area")),
            "shot_zone_range_id": self.vocabs["shot_zone_range"].encode(labels.get("shot_zone_range")),
            "shot_distance_ft": float(distance or -1),
            "shot_distance_norm": dist_norm,
            "shot_loc_x_norm": loc_x_norm,
            "shot_loc_y_norm": loc_y_norm,
        }
        labels_out = {
            "shot_has_attempt": float(has_shot),
            "shot_made_flag": float(labels.get("shot_made_flag") if has_shot else -1),
        }
        aux_out = {
            "shot_difficulty_score": dist_norm
            + 0.1 * self.vocabs["shot_zone_range"].encode(labels.get("shot_zone_range")),
        }
        return _split_dict(features, labels_out, aux_out)

    def _encode_rebound(self, labels: Dict) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        rebound_type = labels.get("rebound_type")
        mapping = {"off": 1, "def": 0}
        rebound_value = mapping.get((rebound_type or "").lower(), -1)
        features = {"rebound_value": float(rebound_value)}
        labels_out = {
            "has_rebound": float(rebound_type is not None),
            "rebound_player_id": float(labels.get("rebound_player_id") or -1),
        }
        return _split_dict(features, labels_out, {})

    def _encode_turnover(self, labels: Dict) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        turnover = labels.get("turnover_type")
        features = {"turnover_type_id": self.vocabs["turnover_type"].encode(turnover)}
        labels_out = {
            "has_turnover": float(turnover is not None),
            "turnover_player_id": float(labels.get("turnover_player_id") or -1),
            "steal_player_id": float(labels.get("steal_player_id") or -1),
        }
        return _split_dict(features, labels_out, {})

    def _encode_foul(
        self, labels: Dict, context: Dict
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        foul_type = labels.get("foul_type")
        if isinstance(foul_type, float) and np.isnan(foul_type):
            foul_type = None
        foul_token = None
        if foul_type:
            foul_token = str(foul_type).lower()
        is_shooting = float(foul_token in {"shooting", "shooting foul", "shooting_foul"})
        is_blocking = float("block" in foul_token) if foul_token else 0.0
        is_offensive = float(foul_token.startswith("offensive") if foul_token else False)
        offense_fouls = float(context.get("offense_fouls_in_period", 0))
        defense_fouls = float(context.get("defense_fouls_in_period", 0))
        features = {
            "foul_type_id": self.vocabs["foul_type"].encode(foul_type),
            "foul_is_shooting": is_shooting,
            "foul_is_block": is_blocking,
            "foul_is_offensive": is_offensive,
            "offense_fouls_in_period": offense_fouls,
            "defense_fouls_in_period": defense_fouls,
            "offense_in_bonus": float(context.get("offense_in_bonus", 0)),
            "defense_in_bonus": float(context.get("defense_in_bonus", 0)),
        }
        labels_out = {
            "has_foul": float(foul_type is not None),
            "foul_player_id": float(labels.get("foul_player_id") or -1),
        }
        return _split_dict(features, labels_out, {})

    def _encode_free_throws(self, labels: Dict) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        attempts = float(labels.get("free_throw_attempts") or 0)
        made = float(labels.get("free_throw_made") or 0)
        accuracy = made / attempts if attempts > 0 else 0.0
        labels_out = {
            "free_throw_attempts": attempts,
            "free_throw_made": made,
            "free_throw_accuracy": accuracy,
        }
        return _split_dict({}, labels_out, {})
