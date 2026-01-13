#!/usr/bin/env python3
"""PARAM FORGE: interactive terminal UI for multi-provider image generation + receipts.

Usage:
  python scripts/param_forge.py \
    --provider openai --size portrait --n 1 --out outputs/param_forge
  python scripts/param_forge.py --interactive
  python scripts/param_forge.py --defaults

Notes:
- Loads .env from the repo root.
- Uses the bundled provider adapters (OpenAI, Gemini, Imagen, Flux).
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
import getpass
import locale
import os
import shutil
import subprocess
import termios
import tty

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

FIXED_PROMPT = (
    "Generate an image of a California brown pelican riding a bicycle. "
    "The bicycle must have spokes and a correctly shaped bicycle frame. "
    "The pelican must have its characteristic large pouch, and there should be a clear indication of feathers. "
    "The pelican must clearly be pedaling the bicycle. "
    "The image should show the full breeding plumage of the Californian brown pelican."
)
DEFAULT_PROMPTS = [FIXED_PROMPT]
PROVIDER_CHOICES = ["openai", "google", "black forest labs"]
MODEL_CHOICES_BY_PROVIDER = {
    "openai": ["gpt-image-1.5", "gpt-image-1-mini", "gpt-image-1"],
    "gemini": ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"],
    "imagen": ["imagen-4.0-ultra", "imagen-4"],
    "flux": ["flux-2-flex", "flux-2-pro", "flux-2"],
}
SIZE_CHOICES = [
    ("4:5", "4:5 (Instagram feed)"),
    ("9:16", "9:16 (Stories/Reels/TikTok/Shorts)"),
    ("square", "1:1 (Square)"),
    ("1.91:1", "1.91:1 (LinkedIn link preview)"),
    ("16:9", "16:9 (YouTube/landscape)"),
    ("3:4", "3:4 (Portrait)"),
    ("2:3", "2:3 (Photo)"),
    ("portrait", "portrait"),
    ("landscape", "landscape"),
    ("1024x1024", "1024x1024"),
    ("1024x1536", "1024x1536"),
    ("1536x1024", "1536x1024"),
]
OUT_DIR_CHOICES = ["outputs/param_forge", "outputs/param_forge_dated"]


def _provider_display_label(provider: str) -> str:
    provider_key = provider.strip().lower()
    if provider_key == "google":
        return "google (gemini/imagen)"
    if provider_key in {"black forest labs", "bfl", "flux"}:
        return "black forest labs (flux)"
    return provider


def _provider_display_choices() -> list[str]:
    return [_provider_display_label(choice) for choice in PROVIDER_CHOICES]


def _provider_from_display(label: str) -> str:
    mapping = {_provider_display_label(choice): choice for choice in PROVIDER_CHOICES}
    return mapping.get(label, label)

_BANNER = [
    "██████╗  █████╗ ██████╗  █████╗ ███╗   ███╗",
    "██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗ ████║",
    "██████╔╝███████║██████╔╝███████║██╔████╔██║",
    "██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║",
    "██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║",
    "╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝",
    "███████╗  ██████╗  ██████╗   ██████╗  ███████╗",
    "██╔════╝ ██╔═══██╗ ██╔══██╗ ██╔════╝ ██╔════╝",
    "█████╗   ██║   ██║ ██████╔╝ ██║  ███╗█████╗  ",
    "██╔══╝   ██║   ██║ ██╔══██╗ ██║   ██║██╔══╝  ",
    "██║      ╚██████╔╝ ██║  ██║ ╚██████╔╝███████╗",
    "╚═╝       ╚═════╝  ╚═╝  ╚═╝  ╚═════╝ ╚══════╝",
]
_VERSION_CURRENT = ("v0.6.0", "START")
_VERSION_HISTORY = [
    ("v0.5.0", "The Colonel"),
    ("v0.4.0", "Pilot"),
]
_MIN_CURSES_WIDTH = max(40, max(len(line) for line in _BANNER))
_MIN_CURSES_HEIGHT = max(12, len(_BANNER) + 4)


class _CursesFallback(RuntimeError):
    pass


def _find_repo_dotenv() -> Path | None:
    if load_dotenv is None:
        return None
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        dotenv_path = parent / ".env"
        if dotenv_path.exists():
            return dotenv_path
    return None


def _load_repo_dotenv() -> Path | None:
    dotenv_path = _find_repo_dotenv()
    if load_dotenv is None:
        return dotenv_path
    if dotenv_path is not None:
        load_dotenv(dotenv_path=dotenv_path, override=False)
        return dotenv_path
    load_dotenv(override=False)
    return None


def _env_flag(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _retrieval_score_enabled(args: argparse.Namespace | None = None) -> bool:
    raw = os.getenv(RETRIEVAL_SCORE_ENV)
    if raw is not None:
        return _env_flag(RETRIEVAL_SCORE_ENV)
    if args is not None and getattr(args, "defaults", False):
        return False
    return True


def _final_summary_enabled() -> bool:
    raw = os.getenv(FINAL_SUMMARY_ENV)
    if raw is None:
        return True
    return _env_flag(FINAL_SUMMARY_ENV)


def _retrieval_packet_mode() -> str:
    raw = os.getenv(RETRIEVAL_PACKET_ENV, RETRIEVAL_PACKET_DEFAULT).strip().lower()
    if raw == "full":
        return "full"
    return "compact"


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _style(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def _term_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size(fallback=(default, 20)).columns
    except Exception:
        return default


def _init_locale() -> None:
    try:
        locale.setlocale(locale.LC_ALL, "")
    except Exception:
        pass


def _curses_preflight() -> tuple[bool, str | None]:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False, "stdin/stdout is not a TTY"
    term = os.environ.get("TERM", "")
    if term.lower() in {"", "dumb", "unknown"}:
        return False, f"TERM={term or 'unset'}"
    try:
        size = shutil.get_terminal_size(fallback=(0, 0))
        if size.columns and size.lines:
            if size.columns < _MIN_CURSES_WIDTH or size.lines < _MIN_CURSES_HEIGHT:
                return (
                    False,
                    f"terminal too small ({size.columns}x{size.lines}, need "
                    f"{_MIN_CURSES_WIDTH}x{_MIN_CURSES_HEIGHT})",
                )
    except Exception:
        pass
    return True, None


_RAINBOW_COLORS = [
    "red",
    "yellow",
    "green",
    "cyan",
    "blue",
    "magenta",
]


ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-opus-4-5-20251101"
ANTHROPIC_MAX_TOKENS = 1400
ANTHROPIC_THINKING_BUDGET = 1024
ANALYSIS_MAX_CHARS = 500
RECEIPT_ANALYZER_ENV = "RECEIPT_ANALYZER"
ANALYZER_CHOICES = ("anthropic", "openai", "council")
DEFAULT_ANALYZER = "anthropic"
OPENAI_ANALYZER_MODEL = "gpt-5.2"
OPENAI_MAX_OUTPUT_TOKENS = 350
GEMINI_ANALYZER_MODEL = "gemini-3-pro-preview"
OPENAI_STREAM_ENV = "OPENAI_IMAGE_STREAM"
OPENAI_RESPONSES_ENV = "OPENAI_IMAGE_USE_RESPONSES"
OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"
RETRIEVAL_SCORE_ENV = "LLM_RETRIEVAL_SCORE"
RETRIEVAL_PACKET_ENV = "LLM_RETRIEVAL_PACKET"
RETRIEVAL_PACKET_DEFAULT = "compact"
FINAL_SUMMARY_ENV = "FINAL_SUMMARY"
MAX_ROUNDS = 3

_COST_ESTIMATE_CACHE: dict[tuple[str, str | None, str], str] = {}
_PRICING_REFERENCE_CACHE: dict | None = None


def _compose_side_by_side(
    left_path: Path,
    right_path: Path,
    *,
    label_left: str,
    label_right: str,
    out_dir: Path,
) -> Path | None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        return None
    try:
        left = Image.open(left_path).convert("RGB")
        right = Image.open(right_path).convert("RGB")
    except Exception:
        return None
    target_h = min(left.height, right.height)
    if left.height != target_h:
        new_w = max(1, int(left.width * (target_h / left.height)))
        left = left.resize((new_w, target_h), Image.LANCZOS)
    if right.height != target_h:
        new_w = max(1, int(right.width * (target_h / right.height)))
        right = right.resize((new_w, target_h), Image.LANCZOS)
    padding = 20
    label_height = 36
    total_w = left.width + right.width + padding * 3
    total_h = target_h + label_height + padding * 2
    canvas = Image.new("RGB", (total_w, total_h), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    x_left = padding
    x_right = padding * 2 + left.width
    y_img = padding + label_height
    canvas.paste(left, (x_left, y_img))
    canvas.paste(right, (x_right, y_img))
    def _draw_label(text: str, x: int, width: int) -> None:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x + max(0, (width - text_w) // 2)
        text_y = padding + max(0, (label_height - text_h) // 2)
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    _draw_label(label_left, x_left, left.width)
    _draw_label(label_right, x_right, right.width)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"comparison_{stamp}.png"
    canvas.save(out_path)
    return out_path


@dataclass
class _RawMode:
    fd: int
    original: list

    def __enter__(self) -> "_RawMode":
        tty.setraw(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)


def _read_key() -> str:
    ch = sys.stdin.read(1)
    if ch == "\x03":
        raise KeyboardInterrupt
    if ch == "":
        raise KeyboardInterrupt
    if ch != "\x1b":
        return ch
    seq = ""
    for _ in range(5):
        nxt = sys.stdin.read(1)
        if nxt == "":
            break
        seq += nxt
        if nxt.isalpha() or nxt == "~":
            break
    if seq.endswith("D"):
        return "LEFT"
    if seq.endswith("C"):
        return "RIGHT"
    if seq.endswith("A"):
        return "UP"
    if seq.endswith("B"):
        return "DOWN"
    return "ESC"


def _select_from_list(label: str, choices: list[str], default_index: int = 0) -> str:
    idx = default_index
    print(f"{label} (left/right, enter to confirm):")
    while True:
        line = _format_choices_line(
            choices,
            idx,
            max_width=_term_width() - 2,
            color=_supports_color(),
        )
        sys.stdout.write("\r\033[2K" + line)
        sys.stdout.flush()
        key = _read_key()
        if key in {"LEFT", "h", "H", "a", "A"}:
            idx = (idx - 1) % len(choices)
        elif key in {"RIGHT", "l", "L", "d", "D"}:
            idx = (idx + 1) % len(choices)
        elif key in {"\r", "\n"}:
            sys.stdout.write("\r\n")
            return choices[idx]
        elif key in {"q", "Q"}:
            sys.stdout.write("\r\n")
            return choices[default_index]


def _select_int(label: str, default: int, minimum: int = 1, maximum: int = 6) -> int:
    value = default
    print(f"{label} (left/right, enter to confirm):")
    while True:
        sys.stdout.write("\r\033[2K" + f"  [{value}]")
        sys.stdout.flush()
        key = _read_key()
        if key in {"LEFT", "h", "H", "a", "A"}:
            value = max(minimum, value - 1)
        elif key in {"RIGHT", "l", "L", "d", "D"}:
            value = min(maximum, value + 1)
        elif key in {"\r", "\n"}:
            sys.stdout.write("\r\n")
            return value
        elif key in {"q", "Q"}:
            sys.stdout.write("\r\n")
    return default


def _select_text(label: str, default: str) -> str:
    print(f"{label} (enter to confirm; empty = default):")
    buffer = ""
    while True:
        sys.stdout.write("\r\033[2K  " + buffer)
        sys.stdout.flush()
        key = _read_key()
        if key in {"\r", "\n"}:
            sys.stdout.write("\r\n")
            text = buffer.strip()
            return text if text else default
        if key in {"\x7f", "\b"}:
            buffer = buffer[:-1]
            continue
        if key in {"LEFT", "RIGHT", "UP", "DOWN", "ESC"}:
            continue
        if len(key) == 1 and key.isprintable():
            buffer += key


def _model_choices_for(provider: str) -> list[str]:
    provider_key = provider.strip().lower()
    if provider_key == "google":
        return MODEL_CHOICES_BY_PROVIDER.get("gemini", []) + MODEL_CHOICES_BY_PROVIDER.get("imagen", [])
    if provider_key in {"black forest labs", "bfl"}:
        return MODEL_CHOICES_BY_PROVIDER.get("flux", [])
    return MODEL_CHOICES_BY_PROVIDER.get(provider_key, [])


def _size_label_choices() -> list[str]:
    return [label for _, label in SIZE_CHOICES]


def _size_value_from_label(label: str) -> str:
    for value, display in SIZE_CHOICES:
        if label == display:
            return value
    return label


def _normalize_size(value: str) -> str:
    return _size_value_from_label(value)


def _normalize_provider_and_model(provider: str, model: str | None) -> tuple[str, str | None]:
    provider_key = provider.strip().lower()
    if provider_key == "google":
        model_key = (model or "").lower()
        if "imagen" in model_key:
            return "imagen", model
        return "gemini", model
    if provider_key in {"black forest labs", "bfl"}:
        return "flux", model
    return provider_key, model


def _is_openai_gpt_image(provider: str | None, model: str | None) -> bool:
    if (provider or "").strip().lower() != "openai":
        return False
    model_key = (model or "").strip().lower()
    if not model_key:
        return True
    return model_key.startswith("gpt-image")


def _normalize_analyzer(value: str | None) -> str:
    if not value:
        return DEFAULT_ANALYZER
    normalized = value.strip().lower()
    if normalized not in ANALYZER_CHOICES:
        raise RuntimeError(
            f"Unsupported receipt analyzer '{value}'. Use: {', '.join(ANALYZER_CHOICES)}."
        )
    return normalized


def _resolve_receipt_analyzer(value: str | None) -> str:
    if value:
        return _normalize_analyzer(value)
    return _normalize_analyzer(os.getenv(RECEIPT_ANALYZER_ENV))


def _analyzer_display_name(analyzer: str) -> str:
    analyzer_key = analyzer.strip().lower()
    if analyzer_key == "openai":
        return "OpenAI GPT-5.2"
    if analyzer_key == "council":
        return "Council (GPT-5.2 + Claude Opus 4.5 + Gemini 3 Pro)"
    return "Claude Opus 4.5"


def _display_provider_name(provider: object | None) -> str | None:
    if provider is None:
        return None
    key = str(provider).strip().lower()
    if key in {"gemini", "imagen", "google"}:
        return "google"
    if key in {"black forest labs", "bfl", "flux"}:
        return "black forest labs"
    return str(provider)


def _allowed_settings_for_provider(provider: str) -> list[str]:
    if provider == "openai":
        return ["quality", "moderation", "input_fidelity", "output_compression", "use_responses"]
    if provider == "gemini":
        return ["image_size"]
    if provider == "imagen":
        return ["add_watermark", "person_generation"]
    if provider == "flux":
        return [
            "endpoint",
            "url",
            "model",
            "poll_interval",
            "poll_timeout",
            "request_timeout",
            "download_timeout",
            "prompt_upsampling",
            "guidance",
            "steps",
            "safety_tolerance",
        ]
    return []


def _allowed_models_for_provider(provider: str | None, current_model: str | None = None) -> list[str]:
    if not provider:
        return [current_model] if current_model else []
    provider_key, model_key = _normalize_provider_and_model(str(provider), current_model)
    models = list(MODEL_CHOICES_BY_PROVIDER.get(provider_key, []))
    if model_key and model_key not in models:
        models.append(model_key)
    return models


def _model_options_line(provider: str, size: str, current_model: str | None) -> str:
    models = _allowed_models_for_provider(provider, current_model)
    if not models:
        return ""
    items: list[str] = []
    for model in models:
        cost = _estimate_cost_value(provider=provider, model=model, size=size)
        if cost is not None:
            items.append(f"{model} (~${_format_price(cost)})")
        else:
            items.append(str(model))
    return "Model options (same provider): " + ", ".join(items)


def _allowed_settings_for_receipt(receipt: dict, provider: str | None = None) -> list[str]:
    allowed: set[str] = set()
    blocked_top_level = {
        "prompt",
        "mode",
        "out_dir",
        "inputs",
        "user",
        "metadata",
        "stream",
        "partial_images",
        "provider",
    }
    blocked_resolved = {"width", "height", "aspect_ratio"}
    top_level_keys = {"size", "n", "seed", "output_format", "model"}

    provider_key = provider
    if not provider_key and isinstance(receipt, dict):
        resolved_provider = None
        request_provider = None
        resolved = receipt.get("resolved") if isinstance(receipt.get("resolved"), dict) else None
        request = receipt.get("request") if isinstance(receipt.get("request"), dict) else None
        if isinstance(resolved, dict):
            resolved_provider = resolved.get("provider")
        if isinstance(request, dict):
            request_provider = request.get("provider")
        provider_key = resolved_provider or request_provider
    if provider_key:
        provider_key, _ = _normalize_provider_and_model(str(provider_key), None)

    if provider_key == "openai":
        top_level_keys.add("background")

    for key in top_level_keys:
        allowed.add(key)

    if provider_key:
        allowed.update(_allowed_settings_for_provider(provider_key))

    request = receipt.get("request") if isinstance(receipt, dict) else None
    if isinstance(request, dict):
        for key in request.keys():
            if key in blocked_top_level or key == "inputs":
                continue
            if key == "provider_options":
                provider_options = request.get("provider_options")
                if isinstance(provider_options, dict):
                    allowed.update(str(opt_key) for opt_key in provider_options.keys())
                continue
            allowed.add(str(key))

    resolved = receipt.get("resolved") if isinstance(receipt, dict) else None
    if isinstance(resolved, dict):
        provider_params = resolved.get("provider_params")
        if isinstance(provider_params, dict):
            for key in provider_params.keys():
                if key in blocked_resolved:
                    continue
                if key == "image_size" and provider_key not in {"gemini"}:
                    provider_options = (
                        request.get("provider_options") if isinstance(request, dict) else None
                    )
                    if not (isinstance(provider_options, dict) and "image_size" in provider_options):
                        continue
                allowed.add(str(key))

    return sorted({key for key in allowed if key not in blocked_top_level and key})


_TOP_LEVEL_RECOMMENDATION_KEYS = {"size", "n", "seed", "output_format", "background", "model"}
_BLOCKED_RECOMMENDATION_KEYS = {
    "prompt",
    "mode",
    "out_dir",
    "inputs",
    "user",
    "metadata",
    "stream",
    "partial_images",
    "provider",
}


def _filter_unsupported_top_level_recommendations(
    recommendations: list[dict],
    provider: str | None,
) -> list[dict]:
    if not provider:
        return recommendations
    provider_key, _ = _normalize_provider_and_model(str(provider), None)
    if provider_key == "openai":
        return recommendations
    filtered: list[dict] = []
    for rec in recommendations:
        name = str(rec.get("setting_name") or "").strip().lower()
        target = str(rec.get("setting_target") or "provider_options").strip().lower()
        if target in {"request", "top_level", "request_settings", ""} and name == "background":
            continue
        filtered.append(rec)
    return filtered


def _extract_setting_json(text: str) -> tuple[str, object | None]:
    match = re.search(r"<setting_json>(.*?)</setting_json>", text, flags=re.S)
    if not match:
        return text.strip(), None
    raw_json = match.group(1).strip()
    recommendation = None
    try:
        recommendation = json.loads(raw_json)
    except json.JSONDecodeError:
        recommendation = None
    cleaned = (text[: match.start()] + text[match.end():]).strip()
    return cleaned, recommendation


def _parse_recommendation_payload(payload: object) -> tuple[object | None, str | None, bool]:
    if payload is None:
        return None, None, False
    if isinstance(payload, dict):
        if "recommendations" in payload and isinstance(payload.get("recommendations"), list):
            return payload.get("recommendations"), None, False
        if "setting_name" in payload:
            return [payload], None, False
        return None, None, False
    if isinstance(payload, list):
        return payload, None, False
    return None, None, False


def _normalize_recommendations(raw: object) -> list[dict]:
    candidates: list[dict] = []
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = [item for item in raw if isinstance(item, dict)]
    else:
        return []

    cleaned: list[dict] = []
    for rec in candidates:
        name = str(rec.get("setting_name") or "").strip()
        name_lower = name.lower()
        if not name or name_lower in {"none", "null", "no_change", "no change", "n/a"}:
            continue
        if name_lower in _BLOCKED_RECOMMENDATION_KEYS:
            continue
        value = rec.get("setting_value")
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in {"none", "null", "no_change", "no change", "n/a"}:
            continue
        target = rec.get("setting_target")
        if not target:
            target = "request" if name_lower in _TOP_LEVEL_RECOMMENDATION_KEYS else "provider_options"
        if name_lower in _TOP_LEVEL_RECOMMENDATION_KEYS:
            target = "request"
        rec_clean = dict(rec)
        rec_clean["setting_name"] = name
        rec_clean["setting_target"] = str(target).strip()
        cleaned.append(rec_clean)

    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for rec in cleaned:
        key = (
            str(rec.get("setting_target") or "").lower(),
            str(rec.get("setting_name") or "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
    return deduped[:3]


def _coerce_setting_value(value: object) -> object:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if "." in lowered:
                return float(lowered)
            return int(lowered)
        except Exception:
            return lowered
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, list):
        return [_coerce_setting_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k).lower(): _coerce_setting_value(v) for k, v in value.items()}
    return value


def _values_equivalent(left: object, right: object) -> bool:
    left_norm = _coerce_setting_value(left)
    right_norm = _coerce_setting_value(right)
    if isinstance(left_norm, (int, float)) and isinstance(right_norm, (int, float)):
        return float(left_norm) == float(right_norm)
    return left_norm == right_norm


def _filter_noop_recommendations(
    recs: list[dict],
    resolved: dict | None,
    request: dict | None,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> list[dict]:
    filtered: list[dict] = []
    for rec in recs:
        setting_name = str(rec.get("setting_name") or "").strip()
        if not setting_name:
            continue
        setting_target = str(rec.get("setting_target") or "provider_options")
        current_value = _lookup_current_setting_value_with_defaults(
            setting_name,
            setting_target,
            resolved,
            request,
            provider,
            model,
        )
        if current_value is None:
            filtered.append(rec)
            continue
        if _values_equivalent(current_value, rec.get("setting_value")):
            continue
        filtered.append(rec)
    return filtered


def _filter_locked_size_recommendations(
    recs: list[dict],
    locked_size: str | None,
) -> list[dict]:
    if not locked_size:
        return recs
    locked_norm = _normalize_size(str(locked_size))
    filtered: list[dict] = []
    for rec in recs:
        setting_name = str(rec.get("setting_name") or "").strip().lower()
        if setting_name != "size":
            filtered.append(rec)
            continue
        rec_value = rec.get("setting_value")
        rec_norm = _normalize_size(str(rec_value))
        if rec_norm == locked_norm:
            filtered.append(rec)
        # Skip recommendations that change the user-selected size/aspect ratio.
    return filtered


def _filter_model_recommendations(
    recs: list[dict],
    allowed_models: list[str],
) -> list[dict]:
    if not allowed_models:
        return recs
    allowed = {str(model).strip().lower() for model in allowed_models if model}
    filtered: list[dict] = []
    for rec in recs:
        setting_name = str(rec.get("setting_name") or "").strip().lower()
        if setting_name != "model":
            filtered.append(rec)
            continue
        rec_value = str(rec.get("setting_value") or "").strip().lower()
        if rec_value in allowed:
            filtered.append(rec)
    return filtered


def _sanitize_recommendation_rationales(
    recs: list[dict],
    cost_line: str | None,
    *,
    provider: str | None = None,
    size: str | None = None,
    n: int | None = None,
) -> list[dict]:
    if not recs:
        return recs
    base_cost_token = _extract_price_token(cost_line)
    updated: list[dict] = []
    for rec in recs:
        rec_copy = dict(rec)
        rationale = rec_copy.get("rationale")
        if isinstance(rationale, str):
            cost_token = base_cost_token
            setting_name = str(rec_copy.get("setting_name") or "").strip().lower()
            if setting_name == "model" and provider and size:
                try:
                    rec_model = str(rec_copy.get("setting_value") or "").strip()
                    rec_cost_line = _estimate_cost_only(
                        provider=provider,
                        model=rec_model or None,
                        size=size,
                        n=n or 1,
                    )
                    rec_cost_token = _extract_price_token(rec_cost_line)
                    if rec_cost_token:
                        cost_token = rec_cost_token
                except Exception:
                    pass
            normalized = _normalize_rationale_cost(
                rationale,
                cost_token,
                force_simple=setting_name == "model",
            )
            if normalized:
                rec_copy["rationale"] = normalized
            else:
                rec_copy.pop("rationale", None)
        updated.append(rec_copy)
    return updated


def _extract_cost_line(text: str) -> tuple[str, str | None]:
    lines = text.splitlines()
    cost_line = None
    remaining: list[str] = []
    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("COST"):
            cost_line = stripped
            continue
        remaining.append(line)
    return "\n".join(remaining).strip(), cost_line


def _normalize_rec_line(text: str) -> str:
    lines = text.splitlines()
    updated: list[str] = []
    for line in lines:
        if line.strip().upper().startswith("REC:"):
            rec = line.strip()[4:].strip()
            updated.append(f"REC: {rec}")
        else:
            updated.append(line)
    return "\n".join(updated).strip()


def _rec_line_text(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip().upper().startswith("REC:"):
            return line.strip()[4:].strip().lower()
    return None


def _extract_price_token(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"\$[0-9]+(?:\.[0-9]+)?(?:/1K)?", text, flags=re.I)
    if match:
        return match.group(0)
    return None


def _normalize_cost_token(cost_token: str) -> str:
    if "/1k" in cost_token.lower():
        return cost_token
    return f"{cost_token}/1K"


def _rationale_needs_simple_cost(rationale: str) -> bool:
    if re.search(r"\b(both|same|either|vs\.?|versus|compare|compared|equal|each)\b", rationale, flags=re.I):
        return True
    price_hits = re.findall(r"\$[0-9]+(?:\.[0-9]+)?(?:/1K)?", rationale, flags=re.I)
    return len(price_hits) > 1


def _normalize_rationale_cost(
    rationale: str,
    cost_token: str | None,
    *,
    force_simple: bool = False,
) -> str | None:
    trimmed = rationale.strip()
    if not trimmed:
        return None
    contains_cost = bool(re.search(r"\b(cost|save|savings)\b", trimmed, flags=re.I)) or "$" in trimmed
    if not contains_cost:
        return trimmed
    if not cost_token:
        return None
    normalized_token = _normalize_cost_token(cost_token)
    if force_simple or _rationale_needs_simple_cost(trimmed):
        return f"Local cost estimate: {normalized_token} images."
    if "$" in trimmed:
        return re.sub(
            r"\$[0-9]+(?:\.[0-9]+)?(?:/1K)?",
            normalized_token,
            trimmed,
            flags=re.I,
        )
    return f"Local cost estimate: {normalized_token} images."


def _rewrite_rec_line(
    text: str,
    recommendations: list[dict],
) -> str:
    if recommendations:
        parts: list[str] = []
        for rec in recommendations:
            name = str(rec.get("setting_name") or "").strip()
            if not name:
                continue
            value = _format_setting_value(rec.get("setting_value"))
            target = str(rec.get("setting_target") or "provider_options").strip().lower()
            if target == "provider_options":
                parts.append(f"{name}={value}")
            else:
                parts.append(f"{name}={value}")
        rec_line = "REC: " + "; ".join(parts) if parts else "REC: no changes recommended."
    else:
        rec_line = "REC: no changes recommended."
    lines = text.splitlines()
    replaced = False
    for idx, line in enumerate(lines):
        if line.strip().upper().startswith("REC:"):
            lines[idx] = rec_line
            replaced = True
            break
    if not replaced:
        lines.append(rec_line)
    return "\n".join(lines).strip()


def _strip_cost_prefix(cost_line: str | None) -> str | None:
    if not cost_line:
        return None
    stripped = cost_line.strip()
    if stripped.upper().startswith("COST:"):
        stripped = stripped[5:].strip()
    return stripped or None


def _parse_cost_amount(cost_line: str | None) -> float | None:
    if not cost_line:
        return None
    match = re.search(r"\$?([0-9]+(?:\.[0-9]+)?)", cost_line)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except Exception:
        return None
    lowered = cost_line.lower()
    if "/1k" in lowered or "per 1k" in lowered or "per 1000" in lowered:
        return value / 1000.0
    return value


def _analysis_history_line(entry: dict) -> str:
    round_index = entry.get("round")
    settings = entry.get("settings") or "(no settings)"
    elapsed = entry.get("elapsed")
    cost_line = entry.get("cost")
    recs = entry.get("recs")
    accepted = entry.get("accepted")
    parts: list[str] = []
    if round_index is not None:
        parts.append(f"Round {round_index}")
    parts.append(str(settings))
    if isinstance(elapsed, (int, float)):
        parts.append(f"elapsed {elapsed:.1f}s")
    cost_text = _strip_cost_prefix(cost_line)
    if cost_text:
        parts.append(f"cost {cost_text}")
    if recs:
        applied_text = "applied" if accepted else "not applied"
        parts.append(f"recs {recs} ({applied_text})")
    else:
        parts.append("recs none")
    return "; ".join(parts)


def _format_analysis_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = [_analysis_history_line(entry) for entry in history]
    return "\n".join(lines).strip()


def _append_analysis_history(
    history: list[dict],
    *,
    round_index: int,
    settings: dict[str, object] | None,
    recommendations: object,
    accepted: bool,
    elapsed: float | None,
    cost_line: str | None,
    adherence: int | None,
    quality: int | None,
    retrieval: int | None = None,
) -> None:
    summary = _recommendations_summary(recommendations)
    history.append(
        {
            "round": round_index,
            "settings": _format_call_settings_line(settings or {}),
            "elapsed": elapsed,
            "cost": cost_line,
            "recs": summary,
            "accepted": accepted,
            "adherence": adherence,
            "quality": quality,
            "retrieval": retrieval,
        }
    )


def _recommendations_rationale(recommendation: object) -> str | None:
    recs = _normalize_recommendations(recommendation)
    if not recs:
        return None
    rationales: list[str] = []
    seen: set[str] = set()
    for rec in recs:
        rationale = rec.get("rationale")
        if not isinstance(rationale, str):
            continue
        cleaned = rationale.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        rationales.append(cleaned)
        if len(rationales) >= 2:
            break
    if not rationales:
        return None
    joined = " / ".join(rationales)
    return f"Rationale: {joined}"


def _average_score(values: list[int | None]) -> int | None:
    scores = [value for value in values if isinstance(value, (int, float))]
    if not scores:
        return None
    return int(round(sum(scores) / len(scores)))


def _quality_from_history(
    history: list[dict] | None,
    current_quality: int | None = None,
) -> tuple[int | None, int | None]:
    baseline = None
    latest = None
    for entry in history or []:
        value = entry.get("quality")
        if isinstance(value, int):
            if baseline is None:
                baseline = value
            latest = value
    if current_quality is not None:
        latest = current_quality
        if baseline is None:
            baseline = current_quality
    return baseline, latest


def _adherence_from_history(
    history: list[dict] | None,
    current_adherence: int | None = None,
) -> tuple[int | None, int | None]:
    baseline = None
    latest = None
    for entry in history or []:
        value = entry.get("adherence")
        if isinstance(value, int):
            if baseline is None:
                baseline = value
            latest = value
    if current_adherence is not None:
        latest = current_adherence
        if baseline is None:
            baseline = current_adherence
    return baseline, latest


def _retrieval_from_history(
    history: list[dict] | None,
    current_retrieval: int | None = None,
) -> tuple[int | None, int | None]:
    baseline = None
    latest = None
    for entry in history or []:
        value = entry.get("retrieval")
        if isinstance(value, int):
            if baseline is None:
                baseline = value
            latest = value
    if current_retrieval is not None:
        latest = current_retrieval
        if baseline is None:
            baseline = current_retrieval
    return baseline, latest


def _baseline_metrics_from_history(history: list[dict] | None) -> tuple[float | None, str | None]:
    if not history:
        return None, None
    first = history[0]
    elapsed = first.get("elapsed")
    cost_line = first.get("cost")
    baseline_elapsed = float(elapsed) if isinstance(elapsed, (int, float)) else None
    baseline_cost_line = str(cost_line) if isinstance(cost_line, str) else None
    return baseline_elapsed, baseline_cost_line


def _compress_analysis_to_limit(text: str, limit: int, *, analyzer: str | None = None) -> str:
    if len(text) <= limit:
        return text
    prompt = (
        "Shorten the content to <= "
        + str(limit)
        + " characters total, preserving the four lines labeled ADH/UNSET/COST/REC "
        + "and the <setting_json>...</setting_json> block. "
        + "Do not add new content. Keep the JSON valid and unchanged if possible.\n\n"
        + text
    )
    compressed, _ = _call_analyzer(prompt, analyzer=analyzer, enable_web_search=False)
    if len(compressed) <= limit:
        return compressed
    # Second pass if needed
    compressed2, _ = _call_analyzer(
        "Tighten to <= "
        + str(limit)
        + " chars total. Preserve ADH/UNSET/COST/REC labels and <setting_json> block.\n\n"
        + compressed,
        analyzer=analyzer,
        enable_web_search=False,
    )
    if len(compressed2) <= limit:
        return compressed2
    return compressed2


def _load_pricing_reference() -> dict:
    global _PRICING_REFERENCE_CACHE
    if _PRICING_REFERENCE_CACHE is not None:
        return _PRICING_REFERENCE_CACHE
    path = _repo_root() / "docs" / "pricing_reference.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _PRICING_REFERENCE_CACHE = data
            return data
    except Exception:
        pass
    _PRICING_REFERENCE_CACHE = {}
    return _PRICING_REFERENCE_CACHE


def _openai_size_key(size: str) -> str:
    size_key = size.strip().lower()
    if size_key in {"1024x1024", "square", "1:1"}:
        return "1024x1024"
    if size_key in {"1024x1536", "portrait", "3:4", "4:5", "9:16"}:
        return "1024x1536"
    if size_key in {"1536x1024", "landscape", "16:9", "1.91:1", "2:3"}:
        return "1536x1024"
    return "1024x1024"


def _gemini_size_tier(size: str) -> str:
    if "4k" in size.lower() or "4096" in size:
        return "4k"
    return "1k_2k"


def _format_price(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _estimate_cost_only(
    *,
    provider: str,
    model: str | None,
    size: str,
    n: int,
) -> str | None:
    cache_key = (provider, model, size)
    cached = _COST_ESTIMATE_CACHE.get(cache_key)
    if cached:
        return cached
    pricing = _load_pricing_reference()
    provider_key = provider.strip().lower()
    model_key = (model or "").strip().lower()
    cost_line: str | None = None

    def _per_1k(value: float, digits: int = 3) -> str:
        return f"${_format_price(float(value) * 1000.0, digits=digits)}/1K"

    if provider_key == "openai":
        model_key = model_key or "gpt-image-1"
        size_key = _openai_size_key(size)
        model_prices = pricing.get("openai", {}).get(model_key, {})
        size_prices = model_prices.get(size_key) if isinstance(model_prices, dict) else None
        if isinstance(size_prices, dict):
            price = size_prices.get("medium") or size_prices.get("standard") or size_prices.get("low")
            if isinstance(price, (int, float)):
                cost_line = (
                    f"COST: ~{_per_1k(float(price))} images "
                    f"({model_key}, {size_key}, medium)"
                )

    elif provider_key == "gemini":
        model_key = model_key or "gemini-2.5-flash-image"
        model_prices = pricing.get("gemini", {}).get(model_key, {})
        if model_key == "gemini-2.5-flash-image":
            price = model_prices.get("standard") if isinstance(model_prices, dict) else None
            if isinstance(price, (int, float)):
                cost_line = f"COST: ~{_per_1k(float(price))} images ({model_key}, standard)"
        elif model_key == "gemini-3-pro-image-preview":
            tier = _gemini_size_tier(size)
            key = f"standard_{tier}"
            price = model_prices.get(key) if isinstance(model_prices, dict) else None
            if isinstance(price, (int, float)):
                label = "1K/2K" if tier == "1k_2k" else "4K"
                cost_line = f"COST: ~{_per_1k(float(price))} images ({model_key}, {label})"

    elif provider_key == "imagen":
        model_key = model_key or "imagen-4"
        model_prices = pricing.get("imagen", {}).get(model_key, {})
        if isinstance(model_prices, dict):
            price = model_prices.get("standard") or model_prices.get("ultra") or model_prices.get("fast")
            if isinstance(price, (int, float)):
                label = (
                    "ultra"
                    if model_key.endswith("ultra")
                    else "fast"
                    if model_key.endswith("fast")
                    else "standard"
                )
                cost_line = f"COST: ~{_per_1k(float(price))} images ({model_key}, {label})"

    elif provider_key == "flux":
        model_key = model_key or "flux-2"
        model_prices = pricing.get("flux", {}).get(model_key, {})
        if isinstance(model_prices, dict):
            price = model_prices.get("from")
            if isinstance(price, (int, float)):
                cost_line = f"COST: from {_per_1k(float(price))} images ({model_key})"

    if cost_line:
        _COST_ESTIMATE_CACHE[cache_key] = cost_line
    return cost_line


def _apply_recommendation(args: argparse.Namespace, recommendation: object) -> bool:
    recs = _normalize_recommendations(recommendation)
    if not recs:
        return False

    applied = False
    for rec in recs:
        setting_name = rec.get("setting_name")
        if not setting_name:
            continue
        setting_value = rec.get("setting_value")
        if isinstance(setting_value, str):
            lowered = setting_value.strip().lower()
            if lowered in {"true", "false"}:
                setting_value = lowered == "true"
            else:
                try:
                    if "." in lowered:
                        setting_value = float(lowered)
                    else:
                        setting_value = int(lowered)
                except Exception:
                    pass
        setting_target = str(rec.get("setting_target") or "").strip().lower()
        setting_name_lower = str(setting_name).strip().lower()
        if setting_name_lower in {"use_responses", "openai_responses"} and setting_target == "request":
            setting_target = "provider_options"

        if setting_name_lower == "seed" or setting_name_lower in _TOP_LEVEL_RECOMMENDATION_KEYS:
            if setting_name_lower == "seed":
                try:
                    args.seed = int(setting_value)
                    applied = True
                except Exception:
                    continue
            elif setting_name_lower == "n":
                try:
                    args.n = int(setting_value)
                    applied = True
                except Exception:
                    continue
            elif setting_name_lower == "size":
                args.size = _normalize_size(str(setting_value))
                applied = True
            elif setting_name_lower == "model":
                args.model = str(setting_value)
                applied = True
            elif setting_name_lower == "output_format":
                setattr(args, "output_format", str(setting_value))
                applied = True
            elif setting_name_lower == "background":
                setattr(args, "background", str(setting_value))
                applied = True
            continue

        if setting_target not in {"provider_options", "provider", "options", ""}:
            continue
        options = getattr(args, "provider_options", None)
        if not isinstance(options, dict):
            options = {}
            args.provider_options = options
        options[str(setting_name)] = setting_value
        if str(setting_name_lower) in {"use_responses", "openai_responses"} and getattr(args, "provider", "") == "openai":
            args.openai_responses = bool(setting_value)
            if args.openai_responses:
                args.openai_stream = False
        applied = True
    return applied


def _recommendations_summary(recommendation: object) -> str:
    recs = _normalize_recommendations(recommendation)
    if not recs:
        return ""
    parts: list[str] = []
    for rec in recs:
        setting_name = rec.get("setting_name")
        if not setting_name:
            continue
        setting_value = rec.get("setting_value")
        setting_target = str(rec.get("setting_target") or "provider_options")
        if setting_target == "provider_options":
            parts.append(
                f"provider_options.{setting_name}={_format_setting_value(setting_value)}"
            )
        else:
            parts.append(f"{setting_name}={_format_setting_value(setting_value)}")
    return ", ".join(parts)


def _interactive_args_raw(color_override: bool | None = None) -> argparse.Namespace:
    print("PARAM FORGE")
    for line in _version_text_lines():
        print(line)
    print("Test image-gen APIs and capture receipts that help configure calls.")
    try:
        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
    except Exception:
        return _interactive_args_simple()
    with _RawMode(fd, original):
        while True:
            mode = _select_from_list("Mode", ["Explore", "Experiment"], 0)
            if mode.lower() == "experiment":
                print("Experiment mode coming next.")
                continue
            break
        provider = _select_from_list("Provider", _provider_display_choices(), 0)
        provider = _provider_from_display(provider)
        model = _select_from_list("Model", _model_choices_for(provider), 0)
        prompt_text = _select_text("Prompt (press Enter to use default prompt)", DEFAULT_PROMPTS[0])
        size_label = _select_from_list("Size", _size_label_choices(), 0)
        size = _size_value_from_label(size_label)
        n = _select_int("Images per prompt", 1, minimum=1, maximum=4)
        out_choice = _select_from_list("Output dir", OUT_DIR_CHOICES, 0)
    return _build_interactive_namespace(provider, model, prompt_text, size, n, out_choice)


def _interactive_args_simple() -> argparse.Namespace:
    print("PARAM FORGE (simple mode)")
    for line in _version_text_lines():
        print(line)
    print("Type a number and press Enter. Press Enter to accept defaults.")

    while True:
        mode = _prompt_choice("Mode", ["Explore", "Experiment"], 0)
        if mode.lower() == "experiment":
            print("Experiment mode coming next.")
            continue
        break
    provider = _prompt_choice("Provider", _provider_display_choices(), 0)
    provider = _provider_from_display(provider)
    model = _prompt_choice("Model", _model_choices_for(provider), 0)
    prompt_text = _prompt_text("Prompt (press Enter to use default prompt)", DEFAULT_PROMPTS[0])
    size_label = _prompt_choice("Size", _size_label_choices(), 0)
    size = _size_value_from_label(size_label)
    n = _prompt_int("Images per prompt", 1, minimum=1, maximum=4)
    out_choice = _prompt_choice("Output dir", OUT_DIR_CHOICES, 0)
    return _build_interactive_namespace(provider, model, prompt_text, size, n, out_choice)


def _prompt_choice(label: str, choices: list[str], default_index: int = 0) -> str:
    while True:
        print(f"{label}:")
        for idx, choice in enumerate(choices, start=1):
            suffix = " (default)" if idx - 1 == default_index else ""
            print(f"  {idx}. {choice}{suffix}")
        try:
            response = input(f"Select {label} [default {default_index + 1}]: ").strip()
        except EOFError:
            raise KeyboardInterrupt from None
        if response == "":
            return choices[default_index]
        if response.isdigit():
            selected = int(response)
            if 1 <= selected <= len(choices):
                return choices[selected - 1]
        print("Invalid choice. Try again.")


def _prompt_int(label: str, default: int, minimum: int = 1, maximum: int = 6) -> int:
    while True:
        try:
            response = input(f"{label} [{default}]: ").strip()
        except EOFError:
            raise KeyboardInterrupt from None
        if response == "":
            return default
        try:
            value = int(response)
        except ValueError:
            print("Please enter a number.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Enter a number between {minimum} and {maximum}.")


def _prompt_text(label: str, default: str) -> str:
    while True:
        try:
            response = input(f"{label}: ").strip()
        except EOFError:
            raise KeyboardInterrupt from None
        if response == "":
            return default
        return response


def _build_interactive_namespace(
    provider: str,
    model: str | None,
    prompt_text: str,
    size: str,
    n: int,
    out_choice: str,
) -> argparse.Namespace:
    prompt_text = (prompt_text or "").strip()
    prompts = [prompt_text] if prompt_text else list(DEFAULT_PROMPTS)
    out_dir = "outputs/param_forge"
    if out_choice == "outputs/param_forge_dated":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = f"outputs/param_forge/{stamp}"
    resolved_provider, resolved_model = _normalize_provider_and_model(provider, model)
    return argparse.Namespace(
        prompt=prompts,
        provider=resolved_provider,
        size=size,
        n=n,
        out=out_dir,
        model=resolved_model,
        provider_options={},
        seed=None,
        output_format=None,
        background=None,
        analyzer=_resolve_receipt_analyzer(None),
        interactive=True,
        defaults=False,
        no_color=False,
        openai_stream=_env_flag(OPENAI_STREAM_ENV),
        openai_responses=_env_flag(OPENAI_RESPONSES_ENV),
    )


def _interactive_args(color_override: bool | None = None) -> argparse.Namespace:
    try:
        import curses
    except Exception:
        return _interactive_args_raw(color_override=color_override)
    raise RuntimeError("Interactive curses selection must be run inside a curses session.")


def _interactive_args_curses(stdscr, color_override: bool | None = None) -> argparse.Namespace:
    import curses
    _init_locale()
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.keypad(True)
    stdscr.timeout(80)
    try:
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
    except Exception:
        pass
    height, width = stdscr.getmaxyx()
    if width < _MIN_CURSES_WIDTH or height < _MIN_CURSES_HEIGHT:
        raise _CursesFallback(
            f"terminal too small ({width}x{height}, need {_MIN_CURSES_WIDTH}x{_MIN_CURSES_HEIGHT})"
        )
    color_enabled = (color_override is None and curses.has_colors()) or (color_override is True)
    if color_enabled:
        curses.start_color()
        curses.use_default_colors()
        base_colors = [
            curses.COLOR_RED,
            curses.COLOR_YELLOW,
            curses.COLOR_GREEN,
            curses.COLOR_CYAN,
            curses.COLOR_BLUE,
            curses.COLOR_MAGENTA,
        ]
        for idx, color in enumerate(base_colors, start=1):
            curses.init_pair(idx, color, -1)
        highlight_pair = len(base_colors) + 1
        done_pair = highlight_pair + 1
        curses.init_pair(highlight_pair, curses.COLOR_CYAN, -1)
        curses.init_pair(done_pair, curses.COLOR_GREEN, -1)
    else:
        highlight_pair = 0
        done_pair = 0

    mode_idx = 0
    provider_idx = 0
    model_idx = 0
    prompt_text = DEFAULT_PROMPTS[0]
    size_idx = 0
    count_value = 1
    out_idx = 0
    field_idx = 0
    provider_display_choices = _provider_display_choices()

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        provider_selected = field_idx > 1
        if provider_selected:
            model_choices = _model_choices_for(PROVIDER_CHOICES[provider_idx])
        else:
            model_choices = ["(select provider)"]
        size_choices = _size_label_choices()
        if model_idx >= len(model_choices):
            model_idx = 0
        y = 0
        if color_enabled and curses.has_colors():
            try:
                curses.init_pair(99, curses.COLOR_RED, -1)
            except curses.error:
                pass
        for line in _BANNER:
            if y >= height:
                break
            truncated = line[: max(0, width - 1)]
            if color_enabled and curses.has_colors():
                for i, ch in enumerate(truncated):
                    if i >= width - 1:
                        break
                    try:
                        stdscr.addstr(y, i, ch, curses.color_pair(99) | curses.A_BOLD)
                    except curses.error:
                        pass
            else:
                try:
                    stdscr.addstr(y, 0, truncated)
                except curses.error:
                    pass
            y += 1
        version_attr = curses.A_DIM
        for line in _banner_version_lines(width):
            if y >= height:
                break
            try:
                stdscr.addstr(y, 0, line[: max(0, width - 1)], version_attr)
            except curses.error:
                pass
            y += 1
        if y < height - 1:
            y += 1
        y = _draw_choice_line(
            stdscr,
            y,
            "Mode",
            ["Explore", "Experiment"],
            mode_idx,
            field_idx == 0,
            field_idx,
            0,
            highlight_pair,
            done_pair,
            color_enabled,
        )
        hint = (
            "Identify optimal API settings for your goal: lower cost, faster render, etc."
            if mode_idx == 0
            else "Start from a reference image to find the best provider/model/params and export one receipt."
        )
        _safe_addstr(stdscr, y, 4, hint[: max(0, width - 5)], curses.A_BOLD)
        y += 2
        if field_idx >= 1:
            y = _draw_choice_line(
                stdscr,
                y,
                "Provider",
                provider_display_choices,
                provider_idx,
                field_idx == 1,
                field_idx,
                1,
                highlight_pair,
                done_pair,
                color_enabled,
            )
            y = _draw_choice_line(
                stdscr,
                y,
                "Model",
                model_choices,
                model_idx,
                field_idx == 2,
                field_idx,
                2,
                highlight_pair,
                done_pair,
                color_enabled,
            )
            y = _draw_prompt_line(
                stdscr,
                y,
                "Prompt",
                prompt_text,
                DEFAULT_PROMPTS[0],
                field_idx == 3,
                field_idx,
                3,
                highlight_pair,
                done_pair,
                color_enabled,
                hint_text="Press (n) to input custom prompt • Enter: use default prompt",
            )
            y = _draw_choice_column(
                stdscr,
                y,
                "Size",
                size_choices,
                size_idx,
                field_idx == 4,
                field_idx,
                4,
                highlight_pair,
                done_pair,
                color_enabled,
                max_visible=6,
            )
            y = _draw_count_line(
                stdscr,
                y,
                "Images per prompt",
                count_value,
                field_idx == 5,
                field_idx,
                5,
                highlight_pair,
                done_pair,
                color_enabled,
            )
            _draw_choice_line(
                stdscr,
                y,
                "Output dir",
                OUT_DIR_CHOICES,
                out_idx,
                field_idx == 6,
                field_idx,
                6,
                highlight_pair,
                done_pair,
                color_enabled,
            )
        stdscr.refresh()

        key = stdscr.getch()
        if key == -1:
            continue
        if key == curses.KEY_MOUSE:
            try:
                _, _, _, _, bstate = curses.getmouse()
            except curses.error:
                continue
            if field_idx == 4:
                if bstate & curses.BUTTON5_PRESSED:
                    size_idx = (size_idx + 1) % len(size_choices)
                elif bstate & curses.BUTTON4_PRESSED:
                    size_idx = (size_idx - 1) % len(size_choices)
            continue
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt
        if key in (curses.KEY_UP, ord("k"), ord("K")):
            field_idx = max(0, field_idx - 1)
            continue
        if key in (curses.KEY_DOWN, ord("j"), ord("J")):
            if field_idx == 4:
                size_idx = (size_idx + 1) % len(size_choices)
            continue
        if key in (curses.KEY_LEFT, ord("h"), ord("H"), ord("a"), ord("A")):
            if field_idx == 0:
                mode_idx = (mode_idx - 1) % 2
            elif field_idx == 1:
                provider_idx = (provider_idx - 1) % len(PROVIDER_CHOICES)
                model_idx = 0
            elif field_idx == 2:
                model_idx = (model_idx - 1) % len(model_choices)
            elif field_idx == 4:
                size_idx = (size_idx - 1) % len(size_choices)
            elif field_idx == 5:
                count_value = max(1, count_value - 1)
            elif field_idx == 6:
                out_idx = (out_idx - 1) % len(OUT_DIR_CHOICES)
            continue
        if key in (curses.KEY_RIGHT, ord("l"), ord("L"), ord("d"), ord("D")):
            if field_idx == 0:
                mode_idx = (mode_idx + 1) % 2
            elif field_idx == 1:
                provider_idx = (provider_idx + 1) % len(PROVIDER_CHOICES)
                model_idx = 0
            elif field_idx == 2:
                model_idx = (model_idx + 1) % len(model_choices)
            elif field_idx == 4:
                size_idx = (size_idx + 1) % len(size_choices)
            elif field_idx == 5:
                count_value = min(4, count_value + 1)
            elif field_idx == 6:
                out_idx = (out_idx + 1) % len(OUT_DIR_CHOICES)
            continue
        if field_idx == 3 and key in (ord("n"), ord("N")):
            default_prompt = DEFAULT_PROMPTS[0]
            initial_text = ""
            if prompt_text and prompt_text.strip() != default_prompt.strip():
                initial_text = prompt_text
            prompt_text = _prompt_prompt_curses(
                stdscr,
                default_prompt=default_prompt,
                color_enabled=color_enabled,
                initial_text=initial_text,
            )
            field_idx = 4
            continue
        if key in (10, 13, curses.KEY_ENTER, ord("\t")):
            if field_idx == 0:
                if mode_idx == 1:
                    stdscr.erase()
                    height, width = stdscr.getmaxyx()
                    _safe_addstr(stdscr, min(height - 2, 0), 0, "Test mode coming next."[:width])
                    _safe_addstr(
                        stdscr,
                        max(0, height - 1),
                        0,
                        "Press any key to return."[:width],
                        curses.A_DIM,
                    )
                    stdscr.refresh()
                    _wait_for_non_mouse_key(stdscr)
                    continue
                field_idx = 1
                continue
            if field_idx == 3:
                if not prompt_text:
                    prompt_text = DEFAULT_PROMPTS[0]
                field_idx = 4
                continue
            if field_idx < 6:
                field_idx += 1
                continue
            return _build_interactive_namespace(
                PROVIDER_CHOICES[provider_idx],
                _model_choices_for(PROVIDER_CHOICES[provider_idx])[model_idx],
                prompt_text,
                _size_value_from_label(size_choices[size_idx]),
                count_value,
                OUT_DIR_CHOICES[out_idx],
            )


def _run_curses_flow(color_override: bool | None = None) -> int:
    _init_locale()
    try:
        import curses
    except Exception:
        return _run_raw_fallback("curses module unavailable", color_override)

    ok, reason = _curses_preflight()
    if not ok:
        return _run_raw_fallback(reason, color_override)

    result: dict[str, object] = {
        "open_path": None,
        "exit_code": 0,
        "ran": False,
        "error": None,
        "view_ready": False,
        "view_out_dir": None,
    }

    def _curses_flow(stdscr) -> None:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)
        stdscr.timeout(100)
        color_enabled = (color_override is None and curses.has_colors()) or (color_override is True)
        if color_enabled:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_RED, -1)
            curses.init_pair(2, curses.COLOR_CYAN, -1)
            curses.init_pair(3, curses.COLOR_GREEN, -1)
            curses.init_pair(4, curses.COLOR_YELLOW, -1)
        args = _interactive_args_curses(stdscr, color_override=color_override)
        result["ran"] = True
        if args is None:
            result["exit_code"] = 1
            return
        result["view_out_dir"] = Path(args.out)
        args.analyzer = _resolve_receipt_analyzer(getattr(args, "analyzer", None))
        try:
            _load_repo_dotenv()
            _ensure_api_keys(
                args.provider,
                _find_repo_dotenv(),
                allow_prompt=True,
                prompt_func=lambda key, path: _prompt_for_key_curses(stdscr, key, path),
            )
            from forge_image_api import generate, stream  # type: ignore
        except Exception as exc:
            result["exit_code"] = 1
            result["error"] = str(exc)
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            error_line = f"Setup failed: {exc}"
            try:
                stdscr.addstr(0, 0, error_line[: max(0, width - 1)])
            except curses.error:
                pass
            stdscr.refresh()
            try:
                curses.flushinp()
            except curses.error:
                pass
            stdscr.getch()
            return
        if not hasattr(args, "openai_stream"):
            args.openai_stream = _env_flag(OPENAI_STREAM_ENV)
        if not hasattr(args, "openai_responses"):
            args.openai_responses = _env_flag(OPENAI_RESPONSES_ENV)
        def _generate_once(
            round_index: int,
            prev_settings: dict[str, object] | None,
            history_lines: list[str],
            rationale_line: str | None,
            header_lines_override: list[str] | None = None,
        ) -> tuple[
            list[Path],
            list[Path],
            bool,
            int,
            float | None,
            dict[str, object],
            list[str],
            tuple[int | None, int | None],
        ]:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            stdscr.timeout(80)
            y = _draw_banner(stdscr, color_enabled, 1)
            if history_lines:
                for line in history_lines:
                    if y >= height - 2:
                        break
                    _safe_addstr(
                        stdscr,
                        y,
                        0,
                        line[: max(0, width - 1)],
                        curses.A_DIM,
                    )
                    y += 1
                if y < height - 1:
                    y += 1
            openai_stream, _ = _apply_openai_provider_flags(args)
            use_stream = bool(openai_stream and _is_openai_gpt_image(args.provider, args.model))
            current_settings = _capture_call_settings(args)
            y = _render_generation_header(
                stdscr,
                y=y,
                width=width,
                round_index=round_index,
                prev_settings=prev_settings,
                current_settings=current_settings,
                color_enabled=color_enabled,
                rationale_line=rationale_line,
                header_lines_override=header_lines_override,
            )
            if y < height - 1:
                try:
                    stdscr.addstr(
                        y,
                        0,
                        "Generating images... (press Q to cancel)"[: width - 1],
                    )
                except curses.error:
                    pass
                y += 1
            if y < height - 1:
                _safe_addstr(
                    stdscr,
                    y,
                    0,
                    "Prompts (sequence):"[: max(0, width - 1)],
                    curses.A_BOLD,
                )
                y += 1
            stdscr.refresh()

            receipts: list[Path] = []
            images: list[Path] = []
            cancel_requested = False
            last_elapsed: float | None = None
            adherence_scores: list[int | None] = []
            quality_scores: list[int | None] = []
            total_prompts = len(args.prompt)
            prompt_entries: dict[int, dict[str, object]] = {}
            max_prompt_width = max(20, width - 1)
            for idx, prompt in enumerate(args.prompt, start=1):
                lines = _prompt_status_lines(idx, total_prompts, prompt, "pending", max_prompt_width)
                if y + len(lines) >= height - 1:
                    remaining = total_prompts - idx + 1
                    if remaining > 0 and y < height - 1:
                        summary = f"... ({remaining} more prompts; resize terminal to view)"
                        _safe_addstr(
                            stdscr,
                            y,
                            0,
                            _truncate_text(summary, max_prompt_width),
                            curses.A_DIM,
                        )
                    break
                for line in lines:
                    _safe_addstr(
                        stdscr,
                        y,
                        0,
                        line,
                        _prompt_status_attr("pending", color_enabled),
                    )
                    y += 1
                prompt_entries[idx] = {"y": y - len(lines), "prompt": prompt, "lines": len(lines)}
            status_y = min(y + 1, height - 1)
            stdscr.refresh()

            for idx, prompt in enumerate(args.prompt, start=1):
                entry = prompt_entries.get(idx)
                if entry:
                    y_line = int(entry["y"])
                    lines = _prompt_status_lines(
                        idx, total_prompts, prompt, "current", max_prompt_width
                    )
                    for offset, line in enumerate(lines):
                        _safe_addstr(
                            stdscr,
                            y_line + offset,
                            0,
                            line,
                            _prompt_status_attr("current", color_enabled),
                        )
                    stdscr.refresh()

                result_holder: dict[str, object] = {}
                done = threading.Event()
                start = time.monotonic()

                def _run_generate() -> None:
                    try:
                        if use_stream:
                            results: list[object] = []
                            for event in stream(
                                prompt=prompt,
                                provider=args.provider,
                                size=args.size,
                                n=args.n,
                                out_dir=Path(args.out),
                                model=args.model,
                                provider_options=args.provider_options,
                                seed=args.seed,
                                output_format=getattr(args, "output_format", None),
                                background=getattr(args, "background", None),
                            ):
                                if event.type == "error":
                                    raise RuntimeError(event.message or "Streaming failed.")
                                if event.type == "final" and event.result is not None:
                                    results.append(event.result)
                            result_holder["results"] = results
                        else:
                            result_holder["results"] = generate(
                                prompt=prompt,
                                provider=args.provider,
                                size=args.size,
                                n=args.n,
                                out_dir=Path(args.out),
                                model=args.model,
                                provider_options=args.provider_options,
                                seed=args.seed,
                                output_format=getattr(args, "output_format", None),
                                background=getattr(args, "background", None),
                            )
                    except Exception as exc:
                        result_holder["error"] = exc
                    finally:
                        done.set()

                thread = threading.Thread(target=_run_generate, daemon=True)
                thread.start()

                frames = ["|", "/", "-", "\\"]
                frame_idx = 0
                local_cancel = False
                while not done.is_set():
                    elapsed = time.monotonic() - start
                    status = (
                        f"{frames[frame_idx % len(frames)]} Round {round_index} "
                        f"Prompt {idx}/{total_prompts} • Elapsed {elapsed:5.1f}s "
                        "(press Q to cancel)"
                    )
                    _write_status_line(
                        stdscr,
                        status_y,
                        _truncate_text(status, max(20, width - 1)),
                        width,
                    )
                    stdscr.refresh()
                    frame_idx += 1
                    key = stdscr.getch()
                    if key in (ord("q"), ord("Q")):
                        local_cancel = True
                thread.join()

                elapsed = time.monotonic() - start
                last_elapsed = elapsed
                _write_status_line(
                    stdscr,
                    status_y,
                    _truncate_text(f"Done in {elapsed:5.1f}s", max(20, width - 1)),
                    width,
                )
                stdscr.refresh()
                if entry:
                    y_line = int(entry["y"])
                    lines = _prompt_status_lines(
                        idx, total_prompts, prompt, "done", max_prompt_width
                    )
                    for offset, line in enumerate(lines):
                        _safe_addstr(
                            stdscr,
                            y_line + offset,
                            0,
                            line,
                            _prompt_status_attr("done", color_enabled),
                        )
                    stdscr.refresh()

                if "error" in result_holder:
                    result["exit_code"] = 1
                    _safe_addstr(
                        stdscr,
                        min(height - 1, y + 2),
                        0,
                        f"Generation failed: {result_holder['error']}",
                    )
                    stdscr.refresh()
                    stdscr.timeout(-1)
                    stdscr.getch()
                    return receipts, images, True, y + 2, last_elapsed, current_settings, []

                results = result_holder.get("results", [])
                for res in results:
                    receipts.append(Path(res.receipt_path))
                    images.append(Path(res.image_path))
                if results:
                    retrieval_enabled = _retrieval_score_enabled(args)
                    for res_idx, res in enumerate(results, start=1):
                        scoring_label = "Council scoring"
                        if retrieval_enabled:
                            scoring_label = "Council scoring + retrieval"
                        stamp_holder: dict[str, object] = {}
                        stamp_done = threading.Event()

                        def _run_stamp() -> None:
                            try:
                                stamp_holder["scores"] = _apply_snapshot_for_result(
                                    image_path=Path(res.image_path),
                                    receipt_path=Path(res.receipt_path),
                                    prompt=prompt,
                                    elapsed=elapsed,
                                    fallback_settings=_capture_call_settings(args),
                                    retrieval_enabled=retrieval_enabled,
                                )
                            except Exception as exc:
                                stamp_holder["error"] = exc
                            finally:
                                stamp_done.set()

                        threading.Thread(target=_run_stamp, daemon=True).start()
                        frames = ["|", "/", "-", "\\"]
                        frame_idx = 0
                        while not stamp_done.is_set():
                            status = (
                                f"Stamping snapshot {res_idx}/{len(results)} "
                                f"({scoring_label}) {frames[frame_idx % len(frames)]}"
                            )
                            _write_status_line(
                                stdscr,
                                status_y,
                                _truncate_text(status, max(20, width - 1)),
                                width,
                                curses.A_DIM,
                            )
                            stdscr.refresh()
                            frame_idx += 1
                            time.sleep(0.1)
                        scores = stamp_holder.get("scores") or (None, None)
                        adherence_scores.append(scores[0])
                        quality_scores.append(scores[1])

                if local_cancel:
                    cancel_requested = True
                    result["exit_code"] = 1
                    _safe_addstr(
                        stdscr,
                        min(height - 1, y + 2),
                        0,
                        "Cancel requested; stopping after current prompt.",
                    )
                    stdscr.refresh()
                    break

            y += 2
            if y >= height - 2:
                y = min(height - 2, max(0, height - 3))

            if cancel_requested:
                avg_adherence = _average_score(adherence_scores)
                avg_quality = _average_score(quality_scores)
                return (
                    receipts,
                    images,
                    cancel_requested,
                    min(y + 2, height - 2),
                    last_elapsed,
                    current_settings,
                    [],
                    (avg_adherence, avg_quality),
                )

            history_block: list[str] = []
            history_block.extend(
                _build_generation_header_lines(
                    round_index=round_index,
                    prev_settings=prev_settings,
                    current_settings=current_settings,
                    max_width=max_prompt_width,
                )
            )
            history_block.append("Prompts (sequence):")
            for idx, prompt in enumerate(args.prompt, start=1):
                history_block.extend(
                    _prompt_status_lines(
                        idx,
                        total_prompts,
                        prompt,
                        "done",
                        max_prompt_width,
                    )
                )
            if last_elapsed is not None:
                history_block.append(f"Completed in {last_elapsed:.1f}s")
            avg_adherence = _average_score(adherence_scores)
            avg_quality = _average_score(quality_scores)
            return (
                receipts,
                images,
                cancel_requested,
                min(y + 2, height - 2),
                last_elapsed,
                current_settings,
                history_block,
                (avg_adherence, avg_quality),
            )

        auto_analyze_next = False
        stored_goals: list[str] | None = None
        stored_notes: str | None = None
        emphasis_line: str | None = None
        stored_cost: str | None = None
        stored_speed_benchmark: float | None = None
        analysis_history: list[dict] = []
        baseline_settings: dict[str, object] | None = None
        baseline_image: Path | None = None
        compare_baseline: Path | None = None
        compare_round_start: int | None = None
        compare_round_end: int | None = None
        compare_next_open = False
        run_index = 1
        last_call_settings: dict[str, object] | None = None
        history_lines: list[str] = []
        pending_rationale: str | None = None
        while True:
            (
                receipts,
                images,
                cancel_requested,
                prompt_y,
                last_elapsed,
                last_call_settings,
                history_block,
                round_scores,
            ) = _generate_once(
                run_index,
                last_call_settings,
                history_lines,
                pending_rationale,
            )
            if receipts:
                result["view_ready"] = True
            pending_rationale = None
            if history_block:
                if history_lines:
                    history_lines.append("")
                history_lines.extend(history_block)
            if baseline_settings is None and last_call_settings:
                baseline_settings = dict(last_call_settings)
            if baseline_image is None and run_index == 1 and images:
                baseline_image = images[-1]
            stdscr.timeout(-1)
            if cancel_requested:
                height, width = stdscr.getmaxyx()
                _safe_addstr(stdscr, max(0, height - 2), 0, "Cancelled. Press any key to exit."[:width])
                stdscr.refresh()
                _wait_for_non_mouse_key(stdscr)
                return
            if images:
                if compare_next_open and compare_baseline and images[-1].exists():
                    left_label = _round_range_label(compare_round_start, compare_round_end)
                    right_label = f"Round {run_index}"
                    composite = _compose_side_by_side(
                        compare_baseline,
                        images[-1],
                        label_left=left_label,
                        label_right=right_label,
                        out_dir=Path(args.out),
                    )
                    if composite:
                        _open_path(composite)
                        compare_baseline = composite
                        compare_round_end = run_index
                    else:
                        _open_path(images[-1])
                    compare_next_open = False
                else:
                    _open_path(images[-1])
            if receipts:
                if auto_analyze_next:
                    if (
                        stored_goals
                        and "minimize time to render" in stored_goals
                        and stored_speed_benchmark is not None
                        and last_elapsed is not None
                        and last_elapsed < stored_speed_benchmark
                    ):
                        improvement = stored_speed_benchmark - last_elapsed
                        speed_note = (
                            f"Speed improved: {stored_speed_benchmark:.1f}s → {last_elapsed:.1f}s "
                            f"(-{improvement:.1f}s)."
                        )
                        if emphasis_line:
                            emphasis_line = f"{emphasis_line} {speed_note}"
                        else:
                            emphasis_line = speed_note
                    recommendation, cost_line, accepted, stop_recommended = _show_receipt_analysis_curses(
                        stdscr,
                        receipt_path=receipts[-1],
                        provider=args.provider,
                        model=args.model,
                        size=args.size,
                        n=args.n,
                        out_dir=str(args.out),
                        analyzer=getattr(args, "analyzer", None),
                        color_enabled=color_enabled,
                        user_goals=stored_goals,
                        user_notes=stored_notes,
                        analysis_history=analysis_history,
                        emphasis_line=emphasis_line,
                        last_elapsed=last_elapsed,
                        last_cost=stored_cost,
                        benchmark_elapsed=stored_speed_benchmark,
                        round_scores=round_scores,
                    )
                    retrieval_score = _load_retrieval_score(receipts[-1]) if receipts else None
                    _append_analysis_history(
                        analysis_history,
                        round_index=run_index,
                        settings=last_call_settings,
                        recommendations=recommendation,
                        accepted=accepted,
                        elapsed=last_elapsed,
                        cost_line=cost_line,
                        adherence=round_scores[0],
                        quality=round_scores[1],
                        retrieval=retrieval_score,
                    )
                    if cost_line:
                        stored_cost = cost_line.replace("COST:", "").strip()
                    if run_index >= MAX_ROUNDS:
                        if _final_summary_enabled():
                            base_settings = last_call_settings or _capture_call_settings(args)
                            quality_baseline, quality_current = _quality_from_history(analysis_history)
                            adherence_baseline, adherence_current = _adherence_from_history(analysis_history)
                            retrieval_current = _load_retrieval_score(receipts[-1]) if receipts else None
                            retrieval_baseline, retrieval_current = _retrieval_from_history(
                                analysis_history,
                                retrieval_current,
                            )
                            baseline_elapsed, baseline_cost_line = _baseline_metrics_from_history(analysis_history)
                            summary_note = None
                            if recommendation:
                                summary_note = (
                                    "Pending recommendations not applied; summary reflects last tested settings."
                                )
                            _show_final_recommendation_curses(
                                stdscr,
                                settings=base_settings,
                                recommendation=recommendation,
                                user_goals=stored_goals,
                                color_enabled=color_enabled,
                                last_elapsed=last_elapsed,
                                cost_line=cost_line,
                                baseline_elapsed=baseline_elapsed,
                                baseline_cost_line=baseline_cost_line,
                                quality_baseline=quality_baseline,
                                quality_current=quality_current,
                                adherence_baseline=adherence_baseline,
                                adherence_current=adherence_current,
                                retrieval_baseline=retrieval_baseline,
                                retrieval_current=retrieval_current,
                                receipt_path=receipts[-1] if receipts else None,
                                baseline_settings=baseline_settings,
                                force_quality_metrics=bool(stored_goals and "maximize quality of render" in stored_goals),
                                summary_note=summary_note,
                                metrics_settings=base_settings,
                            )
                        return
                    if accepted and recommendation:
                        if _apply_recommendation(args, recommendation):
                            pending_rationale = _recommendations_rationale(recommendation)
                            goals_text = ", ".join(stored_goals) if stored_goals else "your goal"
                            summary = _recommendations_summary(recommendation)
                            if summary:
                                emphasis_line = f"Net effect: {summary} → {goals_text}"
                            else:
                                emphasis_line = f"Net effect: updates applied → {goals_text}"
                            normalized = _normalize_recommendations(recommendation)
                            if len(normalized) == 1:
                                rationale = normalized[0].get("rationale")
                                if isinstance(rationale, str) and rationale.strip():
                                    emphasis_line = f"{emphasis_line}. {rationale.strip()}"
                            if stored_goals and "minimize time to render" in stored_goals:
                                stored_speed_benchmark = last_elapsed
                            if images:
                                if compare_baseline is None:
                                    compare_baseline = images[-1]
                                    compare_round_start = run_index
                                    compare_round_end = run_index
                                compare_next_open = True
                            auto_analyze_next = True
                            run_index += 1
                            continue
                    return
                height, width = stdscr.getmaxyx()
                attr = curses.color_pair(4) | curses.A_BOLD if color_enabled else curses.A_BOLD
                hot_attr = curses.color_pair(1) | curses.A_BOLD if color_enabled else curses.A_BOLD
                prompt_line = "View receipt (v), analyze receipt (a), any other key = exit"
                line_y = max(0, min(prompt_y, height - 2))
                _safe_addstr(stdscr, line_y, 0, prompt_line[:width], attr)
                v_pos = prompt_line.find("(v)")
                a_pos = prompt_line.find("(a)")
                if v_pos != -1 and v_pos + 2 < width:
                    _safe_addstr(stdscr, line_y, v_pos + 1, "v", hot_attr)
                if a_pos != -1 and a_pos + 2 < width:
                    _safe_addstr(stdscr, line_y, a_pos + 1, "a", hot_attr)
                stdscr.refresh()
                key = _wait_for_non_mouse_key(stdscr)
                if key in (ord("v"), ord("V")):
                    result["open_path"] = receipts[-1]
                    return
                if key in (ord("a"), ord("A")):
                    goals_result = _prompt_goal_selection_curses(stdscr, color_enabled)
                    if goals_result is None:
                        return
                    user_goals, user_notes = goals_result
                    stored_goals = user_goals
                    stored_notes = user_notes
                    recommendation, cost_line, accepted, stop_recommended = _show_receipt_analysis_curses(
                        stdscr,
                        receipt_path=receipts[-1],
                        provider=args.provider,
                        model=args.model,
                        size=args.size,
                        n=args.n,
                        out_dir=str(args.out),
                        analyzer=getattr(args, "analyzer", None),
                        color_enabled=color_enabled,
                        user_goals=user_goals,
                        user_notes=user_notes,
                        analysis_history=analysis_history,
                        last_elapsed=last_elapsed,
                        last_cost=stored_cost,
                        benchmark_elapsed=stored_speed_benchmark,
                        round_scores=round_scores,
                    )
                    _append_analysis_history(
                        analysis_history,
                        round_index=run_index,
                        settings=last_call_settings,
                        recommendations=recommendation,
                        accepted=accepted,
                        elapsed=last_elapsed,
                        cost_line=cost_line,
                        adherence=round_scores[0],
                        quality=round_scores[1],
                        retrieval=_load_retrieval_score(receipts[-1]) if receipts else None,
                    )
                    if cost_line:
                        stored_cost = cost_line.replace("COST:", "").strip()
                    if run_index >= MAX_ROUNDS:
                        if _final_summary_enabled():
                            base_settings = last_call_settings or _capture_call_settings(args)
                            quality_baseline, quality_current = _quality_from_history(analysis_history)
                            adherence_baseline, adherence_current = _adherence_from_history(analysis_history)
                            retrieval_current = _load_retrieval_score(receipts[-1]) if receipts else None
                            retrieval_baseline, retrieval_current = _retrieval_from_history(
                                analysis_history,
                                retrieval_current,
                            )
                            baseline_elapsed, baseline_cost_line = _baseline_metrics_from_history(analysis_history)
                            summary_note = None
                            if recommendation:
                                summary_note = (
                                    "Pending recommendations not applied; summary reflects last tested settings."
                                )
                            _show_final_recommendation_curses(
                                stdscr,
                                settings=base_settings,
                                recommendation=recommendation,
                                user_goals=user_goals,
                                color_enabled=color_enabled,
                                last_elapsed=last_elapsed,
                                cost_line=cost_line,
                                baseline_elapsed=baseline_elapsed,
                                baseline_cost_line=baseline_cost_line,
                                quality_baseline=quality_baseline,
                                quality_current=quality_current,
                                adherence_baseline=adherence_baseline,
                                adherence_current=adherence_current,
                                retrieval_baseline=retrieval_baseline,
                                retrieval_current=retrieval_current,
                                receipt_path=receipts[-1] if receipts else None,
                                baseline_settings=baseline_settings,
                                force_quality_metrics=bool(user_goals and "maximize quality of render" in user_goals),
                                summary_note=summary_note,
                                metrics_settings=base_settings,
                            )
                        return
                    if accepted and recommendation:
                        if _apply_recommendation(args, recommendation):
                            pending_rationale = _recommendations_rationale(recommendation)
                            goals_text = ", ".join(user_goals) if user_goals else "your goal"
                            summary = _recommendations_summary(recommendation)
                            if summary:
                                emphasis_line = f"Net effect: {summary} → {goals_text}"
                            else:
                                emphasis_line = f"Net effect: updates applied → {goals_text}"
                            normalized = _normalize_recommendations(recommendation)
                            if len(normalized) == 1:
                                rationale = normalized[0].get("rationale")
                                if isinstance(rationale, str) and rationale.strip():
                                    emphasis_line = f"{emphasis_line}. {rationale.strip()}"
                            auto_analyze_next = True
                            if user_goals and "minimize time to render" in user_goals:
                                stored_speed_benchmark = last_elapsed
                            if images:
                                if compare_baseline is None:
                                    compare_baseline = images[-1]
                                    compare_round_start = run_index
                                    compare_round_end = run_index
                                compare_next_open = True
                            run_index += 1
                            continue
                return
            else:
                height, width = stdscr.getmaxyx()
                _safe_addstr(
                    stdscr,
                    max(0, height - 2),
                    0,
                    "No receipts produced. Press any key to exit."[:width],
                )
                stdscr.refresh()
                _wait_for_non_mouse_key(stdscr)
                return

    try:
        curses.wrapper(_curses_flow)
    except _CursesFallback as exc:
        return _run_raw_fallback(str(exc), color_override)
    except curses.error as exc:
        return _run_raw_fallback(str(exc), color_override)
    except Exception as exc:
        # If curses fails (Terminal app quirks), fall back to raw interactive flow.
        return _run_raw_fallback(str(exc), color_override)
    if not result.get("ran"):
        return _run_raw_fallback("curses flow did not start", color_override)
    if result.get("error"):
        print(f"Setup failed: {result['error']}")
    if result.get("open_path"):
        _open_path(Path(result["open_path"]))  # type: ignore[arg-type]
    if result.get("view_ready") and result.get("view_out_dir"):
        _maybe_open_viewer(Path(result["view_out_dir"]))  # type: ignore[arg-type]
    return int(result.get("exit_code", 0))


def _run_raw_fallback(reason: str | None, color_override: bool | None) -> int:
    if reason:
        reason = reason.strip()
    if reason:
        print(f"Curses UI unavailable ({reason}). Falling back to raw prompts.")
    try:
        args = _interactive_args_raw(color_override=color_override)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 1
    except Exception as exc:
        print(f"Raw prompt failed ({exc}). Falling back to simple prompts.")
        args = _interactive_args_simple()
    exit_code = _run_generation(args)
    if exit_code == 0:
        _maybe_open_viewer(Path(args.out))
    return exit_code


def _safe_addstr(stdscr, y: int, x: int, text: str, attr: int = 0) -> None:
    import curses
    height, width = stdscr.getmaxyx()
    if y < 0 or x < 0 or y >= height or x >= width:
        return
    try:
        stdscr.addstr(y, x, text[: max(0, width - x - 1)], attr)
    except curses.error:
        pass


def _write_status_line(stdscr, y: int, text: str, width: int, attr: int = 0) -> None:
    if width <= 1:
        _safe_addstr(stdscr, y, 0, text, attr)
        return
    padded = text.ljust(max(0, width - 1))
    _safe_addstr(stdscr, y, 0, padded, attr)


def _wait_for_non_mouse_key(stdscr) -> int:
    import curses
    while True:
        key = stdscr.getch()
        if key == curses.KEY_MOUSE:
            try:
                curses.getmouse()
            except Exception:
                pass
            continue
        return key


def _print_lines_to_console(stdscr, title: str, lines: list[str]) -> None:
    import curses
    try:
        curses.def_prog_mode()
        curses.endwin()
    except Exception:
        pass
    print(f"\n{title}")
    if lines:
        for line in lines:
            print(line)
    else:
        print("(no content)")
    try:
        input("\nPress Enter to return...")
    except EOFError:
        pass
    try:
        curses.reset_prog_mode()
        curses.curs_set(0)
        stdscr.clear()
        stdscr.refresh()
    except Exception:
        pass


def _line_text_and_attr(line: object, *, color_enabled: bool) -> tuple[str, int]:
    import curses
    if isinstance(line, tuple) and len(line) == 2 and isinstance(line[0], str):
        text, tag = line
        if tag == "change":
            attr = curses.color_pair(3) | curses.A_BOLD if color_enabled else curses.A_BOLD
            return text, attr
        if tag == "goal":
            attr = curses.color_pair(4) | curses.A_BOLD if color_enabled else curses.A_BOLD
            return text, attr
        if tag == "section":
            return text, curses.A_BOLD
        return text, curses.A_NORMAL
    return str(line), curses.A_NORMAL


def _banner_version_lines(width: int) -> list[str]:
    current_label = f"{_VERSION_CURRENT[0]} — {_VERSION_CURRENT[1]}"
    prev_label = " / ".join(f"{ver} {name}" for ver, name in _VERSION_HISTORY)
    lines = [
        f"{current_label}",
        f"prev: {prev_label}" if prev_label else "prev: (none)",
    ]
    return [line[: max(0, width - 1)] for line in lines]


def _version_text_lines() -> list[str]:
    return _banner_version_lines(10_000)


def _draw_banner(stdscr, color_enabled: bool, color_pair: int = 1) -> int:
    import curses
    height, width = stdscr.getmaxyx()
    y = 0
    for line in _BANNER:
        if y >= height:
            break
        truncated = line[: max(0, width - 1)]
        if color_enabled:
            for i, ch in enumerate(truncated):
                if i >= width - 1:
                    break
                try:
                    stdscr.addstr(y, i, ch, curses.color_pair(color_pair) | curses.A_BOLD)
                except curses.error:
                    pass
        else:
            try:
                stdscr.addstr(y, 0, truncated)
            except curses.error:
                pass
        y += 1
    version_attr = curses.A_DIM
    for line in _banner_version_lines(width):
        if y >= height:
            break
        try:
            stdscr.addstr(y, 0, line[: max(0, width - 1)], version_attr)
        except curses.error:
            pass
        y += 1
    return y


def _wrap_text(text: str, max_width: int) -> list[str]:
    if max_width <= 1:
        return [text]
    lines: list[object] = []
    for raw_line in text.splitlines() or [""]:
        if not raw_line:
            lines.append("")
            continue
        words = raw_line.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            if len(word) > max_width:
                if current:
                    lines.append(current)
                    current = ""
                while len(word) > max_width:
                    lines.append(word[:max_width])
                    word = word[max_width:]
                current = word
                continue
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= max_width:
                current = f"{current} {word}"
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def _wrap_prompt_lines(prefix: str, prompt: str, max_width: int) -> list[str]:
    if max_width <= 1:
        return [f"{prefix}{prompt}"]
    wrapped = _wrap_text(prompt, max_width - len(prefix))
    if not wrapped:
        return [prefix.rstrip()]
    lines = [f"{prefix}{wrapped[0]}"]
    indent = " " * len(prefix)
    lines.extend(f"{indent}{line}" for line in wrapped[1:])
    return lines


def _truncate_text(text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""
    if len(text) <= max_width:
        return text
    if max_width <= 3:
        return text[:max_width]
    return text[: max_width - 3] + "..."


def _prompt_status_line(index: int, total: int, prompt: str, status: str, max_width: int) -> str:
    status_tag = {"pending": "[ ]", "current": "[>]", "done": "[x]"}
    tag = status_tag.get(status, "[ ]")
    line = f"{tag} {index}/{total}: {prompt}"
    return _truncate_text(line, max_width)


def _prompt_status_lines(index: int, total: int, prompt: str, status: str, max_width: int) -> list[str]:
    status_tag = {"pending": "[ ]", "current": "[>]", "done": "[x]"}
    tag = status_tag.get(status, "[ ]")
    prefix = f"{tag} {index}/{total}: "
    if max_width <= len(prefix):
        return [prefix.rstrip()]
    wrapped = _wrap_text(prompt, max_width - len(prefix))
    if not wrapped:
        return [prefix.rstrip()]
    lines = [f"{prefix}{wrapped[0]}"]
    indent = " " * len(prefix)
    lines.extend(f"{indent}{line}" for line in wrapped[1:])
    return lines


def _prompt_status_attr(status: str, color_enabled: bool) -> int:
    import curses
    if status == "current":
        return curses.color_pair(2) | curses.A_BOLD if color_enabled else curses.A_BOLD
    if status == "done":
        return curses.color_pair(3) | curses.A_BOLD if color_enabled else curses.A_NORMAL
    return curses.A_DIM


def _capture_call_settings(args: argparse.Namespace) -> dict[str, object]:
    provider_options = getattr(args, "provider_options", None)
    if not isinstance(provider_options, dict):
        provider_options = {}
    return {
        "provider": getattr(args, "provider", None),
        "model": getattr(args, "model", None),
        "size": getattr(args, "size", None),
        "n": getattr(args, "n", None),
        "seed": getattr(args, "seed", None),
        "output_format": getattr(args, "output_format", None),
        "background": getattr(args, "background", None),
        "provider_options": dict(provider_options),
    }


def _apply_openai_provider_flags(args: argparse.Namespace) -> tuple[bool, bool]:
    openai_stream = bool(getattr(args, "openai_stream", False))
    openai_responses = bool(getattr(args, "openai_responses", False))
    if openai_responses and getattr(args, "provider", "") == "openai":
        provider_options = getattr(args, "provider_options", None)
        if not isinstance(provider_options, dict):
            provider_options = {}
            args.provider_options = provider_options
        provider_options["use_responses"] = True
    if openai_responses:
        openai_stream = False
    return openai_stream, openai_responses


def _format_call_settings_line(settings: dict[str, object]) -> str:
    pairs: list[tuple[str, object]] = []
    for key in ("provider", "model", "size", "n", "seed", "output_format", "background"):
        value = settings.get(key)
        if value is not None:
            pairs.append((key, value))
    line = _format_kv_pairs(pairs)
    provider_options = settings.get("provider_options")
    if isinstance(provider_options, dict) and provider_options:
        options_line = _format_dict_inline(provider_options)
        if line:
            line = f"{line} | provider_options: {options_line}"
        else:
            line = f"provider_options: {options_line}"
    return line or "(no settings)"


def _round_range_label(start: int | None, end: int | None) -> str:
    if start is None or end is None:
        return "Round ?"
    if start == end:
        return f"Round {start}"
    return f"Rounds {start}-{end}"


def _build_generation_header_lines(
    *,
    round_index: int,
    prev_settings: dict[str, object] | None,
    current_settings: dict[str, object],
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    if prev_settings:
        title = f"Round {round_index} (API changes vs Round {round_index - 1})"
    else:
        title = f"Round {round_index} (baseline API call)"
    lines.extend(_wrap_text(title, max_width))
    if prev_settings:
        diffs = _diff_call_settings(prev_settings, current_settings)
        if not diffs:
            diffs = ["Change: none"]
        for diff in diffs:
            prefix = "Change: " if not diff.startswith("Change:") else ""
            lines.extend(_wrap_text(f"{prefix}{diff}", max_width))
    else:
        settings_line = _format_call_settings_line(current_settings)
        lines.extend(_wrap_text(settings_line, max_width))
    return lines


def _estimate_cost_value(
    *,
    provider: str | None,
    model: str | None,
    size: str | None,
) -> float | None:
    if not provider or not size:
        return None
    pricing = _load_pricing_reference()
    provider_key = provider.strip().lower()
    model_key = (model or "").strip().lower()
    if provider_key == "openai":
        model_key = model_key or "gpt-image-1"
        size_key = _openai_size_key(size)
        model_prices = pricing.get("openai", {}).get(model_key, {})
        size_prices = model_prices.get(size_key) if isinstance(model_prices, dict) else None
        if isinstance(size_prices, dict):
            price = size_prices.get("medium") or size_prices.get("standard") or size_prices.get("low")
            if isinstance(price, (int, float)):
                return float(price)
    if provider_key == "gemini":
        model_key = model_key or "gemini-2.5-flash-image"
        model_prices = pricing.get("gemini", {}).get(model_key, {})
        if model_key == "gemini-2.5-flash-image":
            price = model_prices.get("standard") if isinstance(model_prices, dict) else None
            if isinstance(price, (int, float)):
                return float(price)
        if model_key == "gemini-3-pro-image-preview":
            tier = _gemini_size_tier(size)
            key = f"standard_{tier}"
            price = model_prices.get(key) if isinstance(model_prices, dict) else None
            if isinstance(price, (int, float)):
                return float(price)
    if provider_key == "imagen":
        model_key = model_key or "imagen-4"
        model_prices = pricing.get("imagen", {}).get(model_key, {})
        if isinstance(model_prices, dict):
            price = model_prices.get("standard") or model_prices.get("ultra") or model_prices.get("fast")
            if isinstance(price, (int, float)):
                return float(price)
    if provider_key == "flux":
        model_key = model_key or "flux-2"
        model_prices = pricing.get("flux", {}).get(model_key, {})
        if isinstance(model_prices, dict):
            price = model_prices.get("from")
            if isinstance(price, (int, float)):
                return float(price)
    return None


def _format_cost_value(cost: float | None) -> str:
    if cost is None:
        return "N/A"
    per_1k = float(cost) * 1000.0
    return f"${_format_price(per_1k, digits=3)}/1K"


def _display_analysis_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.lstrip()
        prefix = raw_line[: len(raw_line) - len(stripped)]
        upper = stripped.upper()
        if upper.startswith("ADH:"):
            body = stripped[4:].strip()
            lines.append(f"{prefix}Prompt adherence notes: {body}".rstrip())
            continue
        if upper.startswith("UNSET:"):
            body = stripped[6:].strip()
            lines.append(f"{prefix}API settings unchanged: {body}".rstrip())
            continue
        lines.append(raw_line)
    return "\n".join(lines)


def _clamp_score(value: object) -> int | None:
    try:
        score = int(float(value))
    except Exception:
        return None
    if score < 0:
        return 0
    if score > 100:
        return 100
    return score


def _parse_score_payload(text: str) -> tuple[int | None, int | None]:
    adherence = None
    quality = None
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            payload = json.loads(match.group(0))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            adherence = _clamp_score(payload.get("adherence") or payload.get("prompt_adherence"))
            quality = _clamp_score(payload.get("quality"))
    if adherence is None:
        adh_match = re.search(r"adherence[^0-9]{0,10}(\d{1,3})", text, flags=re.I)
        if adh_match:
            adherence = _clamp_score(adh_match.group(1))
    if quality is None:
        qual_match = re.search(r"quality[^0-9]{0,10}(\d{1,3})", text, flags=re.I)
        if qual_match:
            quality = _clamp_score(qual_match.group(1))
    return adherence, quality


_RETRIEVAL_AXIS_KEYS = (
    "text_legibility",
    "captionability",
    "entity_richness",
    "information_density",
    "semantic_novelty",
    "trust_signals",
    "platform_fitness",
)

_RETRIEVAL_AXIS_WEIGHTS: dict[str, float] = {
    "text_legibility": 0.2,
    "captionability": 0.15,
    "entity_richness": 0.15,
    "information_density": 0.15,
    "semantic_novelty": 0.1,
    "trust_signals": 0.15,
    "platform_fitness": 0.1,
}


def _extract_retrieval_json(text: str) -> object | None:
    match = re.search(r"<retrieval_json>(.*?)</retrieval_json>", text, flags=re.S)
    raw = match.group(1).strip() if match else text.strip()
    if "```" in raw:
        fence = re.search(r"```(?:json)?(.*?)```", raw, flags=re.S | re.I)
        if fence:
            raw = fence.group(1).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _extract_axis_scores_from_text(text: str) -> dict[str, int | None]:
    axes: dict[str, int | None] = {}
    for key in _RETRIEVAL_AXIS_KEYS:
        pattern = rf"{re.escape(key)}[^0-9]{{0,10}}(\d{{1,3}})"
        match = re.search(pattern, text, flags=re.I)
        axes[key] = _clamp_score(match.group(1)) if match else None
    return axes


def _clean_text(value: object, max_len: int) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _clean_list(value: object, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        items.append(text)
        if len(items) >= max_items:
            break
    return items


def _clean_claims(value: object, max_items: int) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    claims: list[dict[str, object]] = []
    for entry in value:
        if not isinstance(entry, dict):
            text = _clean_text(entry, 140)
            if text:
                claims.append({"text": text, "confidence": 0.5})
            if len(claims) >= max_items:
                break
            continue
        text = _clean_text(entry.get("text"), 140)
        if not text:
            continue
        confidence_raw = entry.get("confidence")
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        claims.append({"text": text, "confidence": confidence})
        if len(claims) >= max_items:
            break
    return claims


def _normalize_retrieval_axes(value: object) -> dict[str, int | None]:
    axes: dict[str, int | None] = {}
    raw = value if isinstance(value, dict) else {}
    for key in _RETRIEVAL_AXIS_KEYS:
        axes[key] = _clamp_score(raw.get(key))
    return axes


def _compute_retrieval_score(axes: dict[str, int | None]) -> int | None:
    weighted = 0.0
    total = 0.0
    for key, weight in _RETRIEVAL_AXIS_WEIGHTS.items():
        score = axes.get(key)
        if score is None:
            continue
        weighted += score * weight
        total += weight
    if total <= 0:
        return None
    return int(round(weighted / total))


def _compact_retrieval_packet(packet: dict[str, object]) -> dict[str, object]:
    return {
        "alt_text_120": packet.get("alt_text_120"),
        "caption_280": packet.get("caption_280"),
        "entities": packet.get("entities"),
        "claims": packet.get("claims"),
        "questions_answered": packet.get("questions_answered"),
        "flags": packet.get("flags"),
    }


def _build_retrieval_prompt(prompt: str) -> str:
    return (
        "You are evaluating an image for LLM-mediated retrieval and summarization. "
        "Do NOT judge beauty. Focus on clarity, extractability, and usefulness.\n\n"
        "Score each axis 0-100 (integers):\n"
        "- text_legibility\n"
        "- captionability\n"
        "- entity_richness\n"
        "- information_density\n"
        "- semantic_novelty\n"
        "- trust_signals\n"
        "- platform_fitness\n\n"
        "Then compute retrieval_score as a weighted average using weights:\n"
        "text_legibility 0.20, captionability 0.15, entity_richness 0.15, "
        "information_density 0.15, semantic_novelty 0.10, trust_signals 0.15, "
        "platform_fitness 0.10.\n\n"
        "Return JSON with keys: retrieval_score (int), axes (object with axis scores), "
        "alt_text_120, caption_280, ocr_text, entities, claims, questions_answered, flags.\n"
        "Axes object keys: text_legibility, captionability, entity_richness, information_density, "
        "semantic_novelty, trust_signals, platform_fitness.\n\n"
        "Also output a consumption packet:\n"
        "- alt_text_120 (<=120 chars)\n"
        "- caption_280 (<=280 chars)\n"
        "- ocr_text (best effort; empty if none)\n"
        "- entities (<=12 items)\n"
        "- claims (<=5 objects {text, confidence 0-1})\n"
        "- questions_answered (<=5 items)\n"
        "- flags (<=6 items from: unreadable_text, low_contrast_text, ambiguous_subject, "
        "cluttered_layout, artifacted_content, low_trust_layout, thumbnail_failure)\n\n"
        "Return ONLY JSON wrapped in <retrieval_json>...</retrieval_json>.\n\n"
        f"Prompt:\n{prompt}"
    )


def _score_retrieval_with_council(
    *,
    prompt: str,
    image_base64: str | None,
    image_mime: str | None,
) -> dict[str, object] | None:
    if not image_base64 or not image_mime:
        return None
    if not (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY_BACKUP")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    ):
        return None
    prompt_text = _build_retrieval_prompt(prompt)
    try:
        text, _ = _call_council(
            prompt_text,
            enable_web_search=False,
            image_base64=image_base64,
            image_mime=image_mime,
            chair_instructions=[
                "You are the council chair. Synthesize a single JSON response.",
                "Output ONLY <retrieval_json>...</retrieval_json>. No extra text.",
            ],
        )
    except Exception:
        return None
    payload = _extract_retrieval_json(text)
    if not isinstance(payload, dict):
        axes = _extract_axis_scores_from_text(text)
        retrieval_score = _compute_retrieval_score(axes)
        if retrieval_score is None:
            return None
        return {
            "score": retrieval_score,
            "axes": axes,
            "packet": {},
            "packet_mode": "compact",
            "model": "council",
            "raw_text": _clean_text(text, 1200),
        }
    axes_payload = payload.get("axes")
    if not isinstance(axes_payload, dict):
        axes_payload = {key: payload.get(key) for key in _RETRIEVAL_AXIS_KEYS}
    axes = _normalize_retrieval_axes(axes_payload)
    retrieval_score = _clamp_score(payload.get("retrieval_score") or payload.get("score"))
    if retrieval_score is None:
        retrieval_score = _compute_retrieval_score(axes)
    packet: dict[str, object] = {
        "alt_text_120": _clean_text(payload.get("alt_text_120"), 120),
        "caption_280": _clean_text(payload.get("caption_280"), 280),
        "ocr_text": _clean_text(payload.get("ocr_text"), 800),
        "entities": _clean_list(payload.get("entities"), 12),
        "claims": _clean_claims(payload.get("claims"), 5),
        "questions_answered": _clean_list(payload.get("questions_answered"), 5),
        "flags": _clean_list(payload.get("flags"), 6),
    }
    packet_mode = _retrieval_packet_mode()
    stored_packet = packet if packet_mode == "full" else _compact_retrieval_packet(packet)
    return {
        "score": retrieval_score,
        "axes": axes,
        "packet": stored_packet,
        "packet_mode": packet_mode,
        "model": "council",
    }


def _score_image_with_council(
    *,
    prompt: str,
    image_base64: str | None,
    image_mime: str | None,
) -> tuple[int | None, int | None]:
    if not image_base64 or not image_mime:
        return None, None
    if not (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY_BACKUP")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    ):
        return None, None
    score_prompt = (
        "Given the prompt and the generated image, evaluate how a group of appropriate judges would rate "
        "the adherence of the generated image to the given prompt and the quality of the image. "
        "Return ONLY JSON like: {\"adherence\": 0-100, \"quality\": 0-100}. "
        "Use integers. No extra text.\n\n"
        f"Prompt:\n{prompt}"
    )
    try:
        text, _ = _call_council(
            score_prompt,
            enable_web_search=False,
            image_base64=image_base64,
            image_mime=image_mime,
            chair_instructions=[
                "You are the council chair. Synthesize the best final response.",
                "Return ONLY JSON like: {\"adherence\": 0-100, \"quality\": 0-100}. Use integers. No extra text.",
            ],
        )
    except Exception:
        return None, None
    return _parse_score_payload(text)


def _compute_colorfulness(image) -> float:
    pixels = list(image.getdata())
    if not pixels:
        return 0.0
    total = len(pixels)
    step = max(1, total // 50000)
    mean_rg = 0.0
    mean_yb = 0.0
    var_rg = 0.0
    var_yb = 0.0
    count = 0
    for idx in range(0, total, step):
        r, g, b = pixels[idx]
        rg = float(r) - float(g)
        yb = 0.5 * (float(r) + float(g)) - float(b)
        count += 1
        delta_rg = rg - mean_rg
        mean_rg += delta_rg / count
        var_rg += delta_rg * (rg - mean_rg)
        delta_yb = yb - mean_yb
        mean_yb += delta_yb / count
        var_yb += delta_yb * (yb - mean_yb)
    if count <= 1:
        return 0.0
    std_rg = math.sqrt(var_rg / (count - 1))
    std_yb = math.sqrt(var_yb / (count - 1))
    mean_rg_abs = abs(mean_rg)
    mean_yb_abs = abs(mean_yb)
    return math.sqrt(std_rg * std_rg + std_yb * std_yb) + 0.3 * math.sqrt(
        mean_rg_abs * mean_rg_abs + mean_yb_abs * mean_yb_abs
    )


def _compute_image_quality_metrics(image_path: Path) -> dict[str, object]:
    try:
        from PIL import Image, ImageFilter, ImageStat  # type: ignore
    except Exception:
        return {}
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return {}
    max_dim = 512
    if max(image.width, image.height) > max_dim:
        scale = max_dim / max(image.width, image.height)
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(new_size, Image.BICUBIC)
    luma = image.convert("L")
    luma_stats = ImageStat.Stat(luma)
    brightness = float(luma_stats.mean[0])
    contrast = float(luma_stats.stddev[0])
    edges = luma.filter(ImageFilter.FIND_EDGES)
    edge_stats = ImageStat.Stat(edges)
    sharpness = float(edge_stats.mean[0])
    colorfulness = float(_compute_colorfulness(image))
    return {
        "brightness_luma_mean": round(brightness, 2),
        "contrast_luma_std": round(contrast, 2),
        "sharpness_edge_mean": round(sharpness, 2),
        "colorfulness": round(colorfulness, 2),
        "sampled_width": image.width,
        "sampled_height": image.height,
    }


def _image_quality_gates(metrics: dict[str, object]) -> list[str]:
    gates: list[str] = []
    brightness = metrics.get("brightness_luma_mean")
    contrast = metrics.get("contrast_luma_std")
    sharpness = metrics.get("sharpness_edge_mean")
    if isinstance(brightness, (int, float)):
        if brightness < 40:
            gates.append("too_dark")
        elif brightness > 215:
            gates.append("too_bright")
    if isinstance(contrast, (int, float)) and contrast < 20:
        gates.append("low_contrast")
    if isinstance(sharpness, (int, float)) and sharpness < 3:
        gates.append("low_sharpness")
    return gates


def _build_snapshot_lines(
    *,
    elapsed: float | None,
    cost: float | None,
    adherence: int | None,
    quality: int | None,
    retrieval_score: int | None = None,
    include_retrieval: bool = False,
    retrieval_note: str | None = None,
) -> list[str]:
    elapsed_text = "N/A" if elapsed is None else f"{elapsed:.1f}s"
    adherence_text = "N/A" if adherence is None else f"{adherence}/100"
    quality_text = "N/A" if quality is None else f"{quality}/100"
    lines = [
        f"render: {elapsed_text}",
        f"cost: {_format_cost_value(cost)}",
        f"prompt adherence: {adherence_text}",
        f"LLM-rated quality: {quality_text}",
    ]
    if include_retrieval:
        if retrieval_score is None:
            retrieval_text = retrieval_note or "N/A"
        else:
            retrieval_text = f"{retrieval_score}/100"
        lines.append(f"LLM retrieval score: {retrieval_text}")
    return lines


def _snapshot_template_lines(include_retrieval: bool = False) -> list[str]:
    lines = [
        "render: 9999.9s",
        "cost: $99999/1K",
        "prompt adherence: 100/100",
        "LLM-rated quality: 100/100",
    ]
    if include_retrieval:
        lines.append("LLM retrieval score: 100/100")
    return lines


def _measure_text_metrics(draw, font, lines: list[str], line_spacing: int) -> tuple[int, int, int]:
    if not lines:
        return 0, 0, 0
    widths: list[int] = []
    heights: list[int] = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    max_w = max(widths) if widths else 0
    line_height = max(heights) if heights else 0
    total_h = line_height * len(lines) + line_spacing * (len(lines) - 1)
    return max_w, line_height, total_h


def _apply_snapshot_overlay(
    image_path: Path, lines: list[str], *, include_retrieval: bool = False
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        return False
    if not lines:
        return False
    try:
        image = Image.open(image_path).convert("RGBA")
    except Exception:
        return False
    base_font_size = 18
    base_scale = 2.0
    baseline_area = 1024 * 1024
    size_scale = math.sqrt((image.width * image.height) / baseline_area)
    size_scale = max(0.75, min(4.0, size_scale))
    scale_factor = base_scale * size_scale
    target_font_size = int(base_font_size * scale_factor)
    font = None
    is_truetype = False
    selected_font = None
    font_candidates = [
        "Menlo.ttf",
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Menlo-Regular.ttf",
        "/Library/Fonts/Menlo.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "DejaVuSans.ttf",
    ]
    for candidate in font_candidates:
        try:
            font = ImageFont.truetype(candidate, target_font_size)
            is_truetype = True
            selected_font = candidate
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    line_spacing = int(4 * scale_factor)
    padding = int(10 * scale_factor)
    template_lines = _snapshot_template_lines(include_retrieval=include_retrieval)
    if not is_truetype:
        draw = ImageDraw.Draw(image)
        max_w, line_height, total_h = _measure_text_metrics(draw, font, lines, line_spacing)
        template_w, template_height, template_total_h = _measure_text_metrics(
            draw,
            font,
            template_lines,
            line_spacing,
        )
        max_w = max(max_w, template_w)
        line_height = max(line_height, template_height)
        total_h = max(total_h, template_total_h)
        line_heights = [max(10, line_height)] * len(lines)
        box_w = max_w + 10 * 2
        box_h = total_h + 10 * 2
        max_w_allowed = int(image.width * 0.95)
        max_h_allowed = int(image.height * 0.95)
        scale = min(
            scale_factor,
            max_w_allowed / max(1, box_w),
            max_h_allowed / max(1, box_h),
        )
        scale = max(0.5, scale)
        small_overlay = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        small_draw = ImageDraw.Draw(small_overlay)
        small_draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0, 160))
        y = 10
        for line, height in zip(lines, line_heights):
            small_draw.text((10, y), line, fill=(255, 255, 255, 255), font=font)
            y += height + 4
        big_overlay = small_overlay.resize(
            (int(box_w * scale), int(box_h * scale)),
            Image.NEAREST,
        )
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay.paste(big_overlay, (0, 0), big_overlay)
        combined = Image.alpha_composite(image, overlay)
        suffix = image_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            combined.convert("RGB").save(image_path, quality=95)
        else:
            combined.save(image_path)
        return True

    draw = ImageDraw.Draw(image)
    max_w, line_height, total_h = _measure_text_metrics(draw, font, lines, line_spacing)
    template_w, template_height, template_total_h = _measure_text_metrics(
        draw,
        font,
        template_lines,
        line_spacing,
    )
    max_w = max(max_w, template_w)
    line_height = max(line_height, template_height)
    total_h = max(total_h, template_total_h)
    line_heights = [max(10, line_height)] * len(lines)
    box_w = max_w + padding * 2
    box_h = total_h + padding * 2
    max_w_allowed = int(image.width * 0.95)
    max_h_allowed = int(image.height * 0.95)
    if (box_w > max_w_allowed or box_h > max_h_allowed) and selected_font:
        scale = min(
            max_w_allowed / max(1, box_w),
            max_h_allowed / max(1, box_h),
        )
        adjusted_size = max(8, int(target_font_size * scale))
        try:
            font = ImageFont.truetype(selected_font, adjusted_size)
        except Exception:
            font = ImageFont.load_default()
        scale_used = max(0.5, adjusted_size / base_font_size)
        line_spacing = int(4 * scale_used)
        padding = int(10 * scale_used)
        draw = ImageDraw.Draw(image)
        max_w, line_height, total_h = _measure_text_metrics(draw, font, lines, line_spacing)
        template_w, template_height, template_total_h = _measure_text_metrics(
            draw,
            font,
            template_lines,
            line_spacing,
        )
        max_w = max(max_w, template_w)
        line_height = max(line_height, template_height)
        total_h = max(total_h, template_total_h)
        line_heights = [max(10, line_height)] * len(lines)
        box_w = max_w + padding * 2
        box_h = total_h + padding * 2

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0, 160))
    y = padding
    for line, height in zip(lines, line_heights):
        overlay_draw.text((padding, y), line, fill=(255, 255, 255, 255), font=font)
        y += height + line_spacing
    combined = Image.alpha_composite(image, overlay)
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        combined.convert("RGB").save(image_path, quality=95)
    else:
        combined.save(image_path)
    return True


def _write_retrieval_metadata(receipt_path: Path, retrieval: dict[str, object]) -> None:
    try:
        payload = _load_receipt_payload(receipt_path)
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    meta = payload.get("result_metadata")
    if not isinstance(meta, dict):
        meta = {}
        payload["result_metadata"] = meta
    meta["llm_retrieval"] = retrieval
    try:
        receipt_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return


def _write_image_quality_metadata(
    receipt_path: Path,
    metrics: dict[str, object],
    gates: list[str],
) -> None:
    if not metrics:
        return
    try:
        payload = _load_receipt_payload(receipt_path)
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    meta = payload.get("result_metadata")
    if not isinstance(meta, dict):
        meta = {}
        payload["result_metadata"] = meta
    meta["image_quality_metrics"] = {
        "metrics": metrics,
        "gates": gates,
        "version": "v1",
    }
    try:
        receipt_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return


def _write_render_metadata(receipt_path: Path, elapsed: float | None) -> None:
    if elapsed is None:
        return
    try:
        payload = _load_receipt_payload(receipt_path)
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    meta = payload.get("result_metadata")
    if not isinstance(meta, dict):
        meta = {}
        payload["result_metadata"] = meta
    if "render_seconds" not in meta:
        try:
            meta["render_seconds"] = float(elapsed)
        except Exception:
            return
    try:
        receipt_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return


def _write_llm_scores_metadata(
    receipt_path: Path,
    adherence: int | None,
    quality: int | None,
    *,
    model: str = "council",
    version: str = "v1",
) -> None:
    if adherence is None and quality is None:
        return
    try:
        payload = _load_receipt_payload(receipt_path)
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    meta = payload.get("result_metadata")
    if not isinstance(meta, dict):
        meta = {}
        payload["result_metadata"] = meta
    scores: dict[str, object] = {"model": model, "version": version}
    if adherence is not None:
        scores["adherence"] = int(adherence)
    if quality is not None:
        scores["quality"] = int(quality)
    meta["llm_scores"] = scores
    try:
        receipt_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return


def _load_retrieval_score(receipt_path: Path) -> int | None:
    try:
        payload = _load_receipt_payload(receipt_path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    meta = payload.get("result_metadata")
    if not isinstance(meta, dict):
        return None
    retrieval = meta.get("llm_retrieval")
    if not isinstance(retrieval, dict):
        return None
    return _clamp_score(retrieval.get("score"))


def _apply_snapshot_for_result(
    *,
    image_path: Path,
    receipt_path: Path,
    prompt: str,
    elapsed: float | None,
    fallback_settings: dict[str, object],
    retrieval_enabled: bool = True,
) -> tuple[int | None, int | None]:
    provider = fallback_settings.get("provider")
    model = fallback_settings.get("model")
    size = fallback_settings.get("size")
    try:
        receipt = _load_receipt_payload(receipt_path)
        if isinstance(receipt, dict):
            resolved = receipt.get("resolved")
            if isinstance(resolved, dict):
                provider = resolved.get("provider") or provider
                model = resolved.get("model") or model
                size = resolved.get("size") or size
    except Exception:
        pass
    cost_value = _estimate_cost_value(
        provider=str(provider) if provider else None,
        model=str(model) if model else None,
        size=str(size) if size else None,
    )
    _write_render_metadata(receipt_path, elapsed)
    quality_metrics: dict[str, object] = {}
    quality_gates: list[str] = []
    try:
        quality_metrics = _compute_image_quality_metrics(image_path)
        quality_gates = _image_quality_gates(quality_metrics) if quality_metrics else []
    except Exception:
        quality_metrics = {}
        quality_gates = []
    if quality_metrics:
        _write_image_quality_metadata(receipt_path, quality_metrics, quality_gates)
    image_base64 = None
    image_mime = None
    try:
        image_base64, image_mime = _load_image_for_analyzer(image_path)
    except Exception:
        pass
    adherence, quality = _score_image_with_council(
        prompt=prompt,
        image_base64=image_base64,
        image_mime=image_mime,
    )
    _write_llm_scores_metadata(receipt_path, adherence, quality)
    retrieval_payload = None
    retrieval_score = None
    retrieval_note = None
    if retrieval_enabled:
        if quality_gates:
            retrieval_note = f"gated ({', '.join(quality_gates)})"
            retrieval_payload = {
                "score": None,
                "axes": {},
                "packet": {},
                "packet_mode": _retrieval_packet_mode(),
                "model": "quality_gate",
                "gated": True,
                "gate_reasons": quality_gates,
            }
            _write_retrieval_metadata(receipt_path, retrieval_payload)
        else:
            retrieval_payload = _score_retrieval_with_council(
                prompt=prompt,
                image_base64=image_base64,
                image_mime=image_mime,
            )
            if isinstance(retrieval_payload, dict):
                retrieval_score = _clamp_score(retrieval_payload.get("score"))
                _write_retrieval_metadata(receipt_path, retrieval_payload)
    lines = _build_snapshot_lines(
        elapsed=elapsed,
        cost=cost_value,
        adherence=adherence,
        quality=quality,
        retrieval_score=retrieval_score,
        include_retrieval=retrieval_enabled,
        retrieval_note=retrieval_note,
    )
    _apply_snapshot_overlay(image_path, lines, include_retrieval=retrieval_enabled)
    return adherence, quality


def _diff_call_settings(prev: dict[str, object], current: dict[str, object]) -> list[str]:
    diffs: list[str] = []
    prev_resolved = _resolved_request_for_settings(prev)
    curr_resolved = _resolved_request_for_settings(current)
    for key in ("provider", "model", "size", "n", "seed", "output_format", "background"):
        prev_val = prev.get(key)
        curr_val = current.get(key)
        if prev_val != curr_val:
            prev_default = (
                _default_setting_value_for_resolved(prev_resolved, key)
                if prev_val is None
                else None
            )
            curr_default = (
                _default_setting_value_for_resolved(curr_resolved, key)
                if curr_val is None
                else None
            )
            prev_text = _format_setting_value_with_default(prev_val, prev_default)
            curr_text = _format_setting_value_with_default(curr_val, curr_default)
            diffs.append(f"{key}: {prev_text} -> {curr_text}")
    prev_opts = prev.get("provider_options")
    curr_opts = current.get("provider_options")
    if not isinstance(prev_opts, dict):
        prev_opts = {}
    if not isinstance(curr_opts, dict):
        curr_opts = {}
    for key in sorted(set(prev_opts.keys()) | set(curr_opts.keys())):
        prev_val = prev_opts.get(key)
        curr_val = curr_opts.get(key)
        if prev_val != curr_val:
            prev_default = (
                _default_setting_value_for_resolved(prev_resolved, key, setting_target="provider_options")
                if prev_val is None
                else None
            )
            curr_default = (
                _default_setting_value_for_resolved(curr_resolved, key, setting_target="provider_options")
                if curr_val is None
                else None
            )
            diffs.append(
                "provider_options."
                + str(key)
                + ": "
                + _format_setting_value_with_default(prev_val, prev_default)
                + " -> "
                + _format_setting_value_with_default(curr_val, curr_default)
            )
    return diffs


def _render_generation_header(
    stdscr,
    *,
    y: int,
    width: int,
    round_index: int,
    prev_settings: dict[str, object] | None,
    current_settings: dict[str, object],
    color_enabled: bool,
    rationale_line: str | None = None,
    header_lines_override: list[str] | None = None,
) -> int:
    import curses
    header_attr = curses.color_pair(4) | curses.A_BOLD if color_enabled else curses.A_BOLD
    change_attr = curses.color_pair(3) | curses.A_BOLD if color_enabled else curses.A_BOLD
    if header_lines_override is None:
        header_lines = _build_generation_header_lines(
            round_index=round_index,
            prev_settings=prev_settings,
            current_settings=current_settings,
            max_width=max(20, width - 1),
        )
    else:
        header_lines = header_lines_override
    for idx, line in enumerate(header_lines):
        attr = header_attr if idx == 0 else change_attr
        _safe_addstr(stdscr, y, 0, line[: max(0, width - 1)], attr)
        y += 1
        if idx == 0 and rationale_line:
            for wrapped in _wrap_text(rationale_line, max(20, width - 1)):
                _safe_addstr(stdscr, y, 0, wrapped[: max(0, width - 1)], header_attr)
                y += 1
    return y


def _build_validation_header_lines(
    *,
    baseline_settings: dict[str, object],
    current_settings: dict[str, object],
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    lines.extend(_wrap_text("Final validation run (baseline comparison)", max_width))
    baseline_line = f"Baseline: {_format_call_settings_line(baseline_settings)}"
    lines.extend(_wrap_text(baseline_line, max_width))
    current_line = f"Final settings: {_format_call_settings_line(current_settings)}"
    lines.extend(_wrap_text(current_line, max_width))
    return lines


def _format_setting_value(value: object, *, depth: int = 0) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        rendered = ", ".join(_format_setting_value(item, depth=depth + 1) for item in value)
        return rendered or "[]"
    if isinstance(value, dict):
        if depth >= 1:
            return "{...}"
        return _format_dict_inline(value, depth=depth + 1)
    return str(value)


def _resolved_request_for_settings(
    settings: dict[str, object],
) -> tuple[str, str | None, object] | None:
    provider_raw = settings.get("provider")
    if not isinstance(provider_raw, str) or not provider_raw.strip():
        provider_raw = "openai"
    model_raw = settings.get("model")
    if not isinstance(model_raw, str):
        model_raw = None
    provider_key, model_key = _normalize_provider_and_model(provider_raw, model_raw)
    size_value = settings.get("size")
    if not isinstance(size_value, str) or not size_value.strip():
        size_value = "1024x1024"
    n_value = settings.get("n")
    try:
        n_value = int(n_value) if n_value is not None else 1
    except Exception:
        n_value = 1
    try:
        from forge_image_api.core.contracts import ImageRequest  # type: ignore
        from forge_image_api.core.solver import resolve_request  # type: ignore
    except Exception:
        return None
    try:
        request = ImageRequest(prompt=".", size=str(size_value), n=n_value)
        request.output_format = settings.get("output_format")
        request.background = settings.get("background")
        if model_key:
            request.model = model_key
        provider_options = settings.get("provider_options")
        if isinstance(provider_options, dict):
            request.provider_options = dict(provider_options)
        request.seed = settings.get("seed")
        resolved = resolve_request(request, provider_key)
        return provider_key, model_key, resolved
    except Exception:
        return None


def _default_setting_value_for_resolved(
    resolved_data: tuple[str, str | None, object] | None,
    key: str,
    *,
    setting_target: str | None = None,
) -> object | None:
    if not resolved_data:
        return None
    provider_key, model_key, resolved = resolved_data
    if setting_target == "provider_options":
        provider_params = getattr(resolved, "provider_params", None)
        if isinstance(provider_params, dict) and key in provider_params:
            return provider_params.get(key)
        return _default_provider_option(provider_key, model_key, key)
    if key == "provider":
        return getattr(resolved, "provider", None)
    if key == "model":
        return getattr(resolved, "model", None)
    if key == "size":
        return getattr(resolved, "size", None)
    if key == "n":
        return getattr(resolved, "n", None)
    if key == "seed":
        return getattr(resolved, "seed", None)
    if key == "output_format":
        return getattr(resolved, "output_format", None)
    if key == "background":
        return getattr(resolved, "background", None)
    return None


def _default_output_format_for_settings(settings: dict[str, object]) -> str | None:
    resolved_data = _resolved_request_for_settings(settings)
    return _default_setting_value_for_resolved(resolved_data, "output_format")


def _format_setting_value_with_default(value: object, default: object | None) -> str:
    rendered = _format_setting_value(value)
    if value is None and default is not None:
        return f"{rendered} (default: {_format_setting_value(default)})"
    return rendered


def _size_area_estimate(size_value: object) -> float | None:
    if not isinstance(size_value, str):
        return None
    raw = size_value.strip().lower()
    if not raw:
        return None
    if "x" in raw:
        parts = raw.split("x", 1)
        try:
            w = int(float(parts[0].strip()))
            h = int(float(parts[1].strip()))
        except Exception:
            return None
        if w <= 0 or h <= 0:
            return None
        return float(w * h)
    if raw in {"portrait", "tall"}:
        return float(1024 * 1536)
    if raw in {"landscape", "wide"}:
        return float(1536 * 1024)
    if raw in {"square", "1:1"}:
        return float(1024 * 1024)
    if ":" in raw:
        parts = raw.split(":", 1)
        try:
            w = float(parts[0].strip())
            h = float(parts[1].strip())
        except Exception:
            return None
        if w <= 0 or h <= 0:
            return None
        base = 1024.0
        height = base * (h / w)
        return base * height
    return None


def _format_dict_inline(data: dict, *, max_items: int = 8, depth: int = 0) -> str:
    items: list[str] = []
    for idx, key in enumerate(sorted(data.keys())):
        if idx >= max_items:
            items.append("...")
            break
        value = data.get(key)
        items.append(f"{key}={_format_setting_value(value, depth=depth)}")
    return ", ".join(items) if items else "(none)"


def _format_kv_pairs(pairs: list[tuple[str, object]], *, sep: str = " | ") -> str:
    return sep.join(f"{key}={_format_setting_value(value)}" for key, value in pairs)


def _lookup_current_setting_value(
    setting_name: str,
    setting_target: str,
    resolved: dict | None,
    request: dict | None,
) -> object | None:
    if setting_target == "provider_options":
        if isinstance(resolved, dict):
            provider_params = resolved.get("provider_params")
            if isinstance(provider_params, dict) and setting_name in provider_params:
                return provider_params.get(setting_name)
        if isinstance(request, dict):
            provider_options = request.get("provider_options")
            if isinstance(provider_options, dict) and setting_name in provider_options:
                return provider_options.get(setting_name)
    if isinstance(resolved, dict) and setting_name in resolved:
        return resolved.get(setting_name)
    if isinstance(request, dict) and setting_name in request:
        return request.get(setting_name)
    return None


def _flux_default_params(model: str | None) -> dict[str, object]:
    model_key = (model or "").strip().lower()
    if "dev" in model_key:
        return {
            "prompt_upsampling": False,
            "safety_tolerance": 2,
            "steps": 28,
            "guidance": 3,
        }
    return {
        "prompt_upsampling": False,
        "safety_tolerance": 2,
        "steps": 40,
        "guidance": 2.5,
    }


def _default_provider_option(provider: str | None, model: str | None, key: str) -> object | None:
    provider_key = (provider or "").strip().lower()
    if provider_key == "openai" and key == "use_responses":
        return False
    if provider_key in {"black forest labs", "bfl", "flux"}:
        return _flux_default_params(model).get(key)
    return None


def _fill_defaults_for_display(
    data: dict | None,
    *,
    provider: str | None,
    model: str | None,
) -> dict | None:
    if not isinstance(data, dict) or not data:
        return data
    provider_key = (provider or "").strip().lower()
    if provider_key not in {"black forest labs", "bfl", "flux"}:
        return data
    defaults = _flux_default_params(model)
    updated = dict(data)
    for key, value in list(updated.items()):
        if value is None and key in defaults:
            updated[key] = defaults[key]
    return updated


def _lookup_current_setting_value_with_defaults(
    setting_name: str,
    setting_target: str,
    resolved: dict | None,
    request: dict | None,
    provider: str | None,
    model: str | None,
) -> object | None:
    value = _lookup_current_setting_value(setting_name, setting_target, resolved, request)
    if value is not None:
        return value
    if setting_target == "provider_options":
        return _default_provider_option(provider, model, setting_name)
    return None


def _build_receipt_detail_lines(
    receipt: dict,
    recommendation: object,
    *,
    max_width: int,
    return_tags: bool = False,
    stop_reason: str | None = None,
) -> list[object]:
    lines: list[object] = []
    def _append(line: str, tag: str | None = None) -> None:
        if tag and return_tags:
            lines.append((line, tag))
        else:
            lines.append(line)

    def _append_wrapped(text: str, tag: str | None = None) -> None:
        for line in _wrap_text(text, max_width):
            _append(line, tag)

    resolved = receipt.get("resolved") if isinstance(receipt, dict) else None
    request = receipt.get("request") if isinstance(receipt, dict) else None
    provider_value = None
    model_value = None
    if isinstance(resolved, dict):
        provider_value = resolved.get("provider")
        model_value = resolved.get("model")
    if provider_value is None and isinstance(request, dict):
        provider_value = request.get("provider")
    if model_value is None and isinstance(request, dict):
        model_value = request.get("model")

    _append("RENDER SETTINGS (RESOLVED)", "section")
    pairs: list[tuple[str, object]] = []
    for key in ("provider", "model", "size", "n", "output_format", "background", "seed"):
        value = None
        if isinstance(resolved, dict):
            value = resolved.get(key)
        if value is None and isinstance(request, dict):
            value = request.get(key)
        if key == "provider":
            value = _display_provider_name(value)
        if value is not None:
            pairs.append((key, value))
    if isinstance(resolved, dict):
        width = resolved.get("width")
        height = resolved.get("height")
        if width and height:
            pairs.append(("dimensions", f"{width}x{height}"))
    if pairs:
        _append_wrapped(_format_kv_pairs(pairs))
    else:
        _append("(no resolved settings found)")

    if isinstance(resolved, dict):
        provider_params = resolved.get("provider_params")
        provider_params = _fill_defaults_for_display(
            provider_params if isinstance(provider_params, dict) else None,
            provider=provider_value,
            model=model_value,
        )
        if isinstance(provider_params, dict) and provider_params:
            provider_line = f"provider_params: {_format_dict_inline(provider_params)}"
            _append_wrapped(provider_line)
    if isinstance(request, dict):
        provider_options = request.get("provider_options")
        provider_options = _fill_defaults_for_display(
            provider_options if isinstance(provider_options, dict) else None,
            provider=provider_value,
            model=model_value,
        )
        if isinstance(provider_options, dict) and provider_options:
            options_line = f"requested provider_options: {_format_dict_inline(provider_options)}"
            _append_wrapped(options_line)
    warnings = None
    if isinstance(resolved, dict):
        warnings = resolved.get("warnings")
    if not warnings and isinstance(receipt, dict):
        warnings = receipt.get("warnings")
    if isinstance(warnings, list) and warnings:
        warning_line = "warnings: " + "; ".join(str(w) for w in warnings)
        _append_wrapped(warning_line)

    _append("")

    _append("RECOMMENDATIONS", "section")
    recommendations = _normalize_recommendations(recommendation)
    if not recommendations:
        _append("none")
        return lines

    for idx, rec in enumerate(recommendations, start=1):
        if len(recommendations) > 1:
            _append(f"Recommendation {idx}", "section")
        setting_name = str(rec.get("setting_name") or "").strip()
        setting_value = rec.get("setting_value")
        setting_target = str(rec.get("setting_target") or "provider_options")
        detail_pairs = [
            ("setting_name", setting_name),
            ("setting_value", setting_value),
            ("target", setting_target),
        ]
        _append_wrapped(_format_kv_pairs(detail_pairs), "change")
        current_value = _lookup_current_setting_value(setting_name, setting_target, resolved, request)
        if setting_target == "provider_options":
            next_line = (
                f"next render: provider_options.{setting_name}="
                f"{_format_setting_value(setting_value)}"
            )
        else:
            next_line = f"next render: {setting_name}={_format_setting_value(setting_value)}"
        if current_value is not None:
            next_line += f" (was {_format_setting_value(current_value)})"
        _append_wrapped(next_line, "change")
        rationale = rec.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            _append_wrapped(f"rationale: {rationale.strip()}")
        if idx < len(recommendations):
            _append("")
    return lines


def _export_analysis_text(receipt_path: Path, lines: list[str]) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = receipt_path.with_name(f"{receipt_path.stem}-analysis-{stamp}.txt")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path


def _settings_after_recommendation(
    settings: dict[str, object],
    recommendation: object,
) -> dict[str, object]:
    temp_args = argparse.Namespace(**settings)
    if not hasattr(temp_args, "provider_options") or not isinstance(temp_args.provider_options, dict):
        temp_args.provider_options = {}
    _apply_recommendation(temp_args, recommendation)
    return _capture_call_settings(temp_args)


def _settings_from_receipt(
    receipt: dict | None,
    *,
    fallback_provider: str,
    fallback_model: str | None,
    fallback_size: str,
    fallback_n: int,
) -> dict[str, object]:
    resolved = receipt.get("resolved") if isinstance(receipt, dict) else None
    request = receipt.get("request") if isinstance(receipt, dict) else None
    settings: dict[str, object] = {
        "provider": fallback_provider,
        "model": fallback_model,
        "size": fallback_size,
        "n": fallback_n,
        "seed": None,
        "output_format": None,
        "background": None,
        "provider_options": {},
    }
    for key in ("provider", "model", "size", "n", "seed", "output_format", "background"):
        value = None
        if isinstance(resolved, dict):
            value = resolved.get(key)
        if value is None and isinstance(request, dict):
            value = request.get(key)
        if value is not None:
            settings[key] = value
    provider_options = None
    if isinstance(request, dict):
        provider_options = request.get("provider_options")
    if not (isinstance(provider_options, dict) and provider_options):
        if isinstance(resolved, dict):
            provider_options = resolved.get("provider_params")
    if isinstance(provider_options, dict):
        settings["provider_options"] = dict(provider_options)
    return settings


def _build_final_recommendation_lines(
    settings: dict[str, object],
    recommendation: object,
    *,
    user_goals: list[str] | None,
    max_width: int,
    return_tags: bool = False,
    last_elapsed: float | None = None,
    cost_line: str | None = None,
    baseline_settings: dict[str, object] | None = None,
    baseline_elapsed: float | None = None,
    baseline_cost_line: str | None = None,
    quality_baseline: int | None = None,
    quality_current: int | None = None,
    adherence_baseline: int | None = None,
    adherence_current: int | None = None,
    retrieval_baseline: int | None = None,
    retrieval_current: int | None = None,
    force_quality_metrics: bool = False,
    metrics_settings: dict[str, object] | None = None,
) -> list[object]:
    lines: list[object] = []

    def _append(line: str, tag: str | None = None) -> None:
        if tag and return_tags:
            lines.append((line, tag))
        else:
            lines.append(line)

    def _append_wrapped(text: str, tag: str | None = None) -> None:
        for line in _wrap_text(text, max_width):
            _append(line, tag)

    _append("FINAL RECOMMENDED API CALL", "section")
    goals_line = ", ".join(user_goals) if user_goals else "(not specified)"
    _append_wrapped(f"Goals: {goals_line}", "goal")
    if baseline_settings:
        _append_wrapped(
            f"Round 1 baseline (vanilla): {_format_call_settings_line(baseline_settings)}",
            "change",
        )
    recommended_settings = _settings_after_recommendation(settings, recommendation) if recommendation else settings
    metric_settings = metrics_settings or recommended_settings
    _append_wrapped(_format_call_settings_line(recommended_settings))
    baseline_diffs: list[str] = []
    if baseline_settings:
        baseline_diffs = _diff_call_settings(baseline_settings, recommended_settings)
    if recommendation:
        diffs = _diff_call_settings(settings, recommended_settings)
        if not diffs:
            if baseline_diffs:
                diffs = ["Change: none (final settings already applied)"]
            else:
                diffs = ["Change: none"]
        for diff in diffs:
            prefix = "Change: " if not diff.startswith("Change:") else ""
            _append_wrapped(f"{prefix}{diff}", "change")
    else:
        if baseline_diffs:
            _append_wrapped("No changes recommended (final settings already applied).", "change")
        else:
            _append_wrapped("No changes recommended.", "change")

    goals = set(user_goals or [])
    if goals or any(
        value is not None
        for value in (
            baseline_elapsed,
            last_elapsed,
            baseline_cost_line,
            cost_line,
            quality_baseline,
            quality_current,
            adherence_baseline,
            adherence_current,
            retrieval_baseline,
            retrieval_current,
        )
    ):
        reference_settings = baseline_settings or settings
        cost_ref = _estimate_cost_value(
            provider=str(reference_settings.get("provider"))
            if reference_settings.get("provider")
            else None,
            model=str(reference_settings.get("model")) if reference_settings.get("model") else None,
            size=str(reference_settings.get("size")) if reference_settings.get("size") else None,
        )
        if cost_ref is None:
            if baseline_settings and baseline_cost_line:
                cost_ref = _parse_cost_amount(baseline_cost_line)
            else:
                cost_ref = _parse_cost_amount(cost_line)
        cost_target = _estimate_cost_value(
            provider=str(metric_settings.get("provider"))
            if metric_settings.get("provider")
            else None,
            model=str(metric_settings.get("model")) if metric_settings.get("model") else None,
            size=str(metric_settings.get("size")) if metric_settings.get("size") else None,
        )
        if cost_target is None:
            if not _diff_call_settings(metric_settings, settings):
                cost_target = _parse_cost_amount(cost_line) or cost_ref
            elif baseline_settings and not _diff_call_settings(metric_settings, baseline_settings):
                cost_target = cost_ref
        if cost_target is None and cost_ref is not None:
            cost_target = cost_ref

        time_ref = baseline_elapsed if baseline_elapsed is not None else last_elapsed
        time_target = None
        if time_ref is not None:
            if last_elapsed is not None and not _diff_call_settings(metric_settings, settings):
                time_target = last_elapsed
            elif cost_ref is not None and cost_target is not None and cost_ref > 0:
                time_target = time_ref * (cost_target / cost_ref)
            else:
                area_ref = _size_area_estimate(reference_settings.get("size"))
                area_target = _size_area_estimate(metric_settings.get("size"))
                if area_ref and area_target and area_ref > 0:
                    time_target = time_ref * (area_target / area_ref)

        wants_cost = "minimize cost of render" in goals
        wants_time = "minimize time to render" in goals
        wants_quality = "maximize quality of render" in goals
        wants_retrieval = "maximize LLM retrieval score" in goals

        def _fmt_seconds(value: float | None) -> str:
            if value is None:
                return "N/A"
            return f"{value:.1f}s"

        def _fmt_cost(value: float | None) -> str:
            return _format_cost_value(value)

        def _fmt_score(value: int | None) -> str:
            if value is None:
                return "N/A"
            return f"{value}/100"

        def _delta_label_pct(
            delta: float | None,
            *,
            higher_label: str,
            lower_label: str,
            invert_sign: bool = False,
        ) -> str:
            if delta is None:
                return "N/A"
            pct = abs(delta) * 100.0
            if invert_sign:
                sign = "-" if delta > 0 else "+" if delta < 0 else ""
            else:
                sign = "+" if delta > 0 else "-" if delta < 0 else ""
            if pct < 0.5:
                return "flat"
            if pct < 3:
                return f"{sign}{pct:.0f}% (minimal, {higher_label if delta > 0 else lower_label})"
            return f"{sign}{pct:.0f}% ({higher_label if delta > 0 else lower_label})"

        def _delta_label_points(delta: int | None, *, positive_label: str, negative_label: str) -> str:
            if delta is None:
                return "N/A"
            if delta == 0:
                return "flat"
            sign = "+" if delta > 0 else "-"
            abs_delta = abs(delta)
            if abs_delta == 1:
                return f"{sign}1 pt (minimal, {positive_label if delta > 0 else negative_label})"
            return f"{sign}{abs_delta} pts ({positive_label if delta > 0 else negative_label})"

        metrics: list[dict[str, object]] = []

        cost_delta = None
        if cost_ref is not None and cost_target is not None and cost_ref > 0:
            cost_delta = (cost_target - cost_ref) / cost_ref
        metrics.append(
            {
                "key": "cost",
                "label": "Cost",
                "baseline": cost_ref,
                "target": cost_target,
                "delta": cost_delta,
                "line": f"Cost: {_fmt_cost(cost_ref)} → {_fmt_cost(cost_target)}",
                "delta_text": _delta_label_pct(
                    cost_delta,
                    higher_label="higher",
                    lower_label="lower",
                    invert_sign=True,
                ),
                "primary": wants_cost,
            }
        )

        time_delta = None
        if time_ref is not None and time_target is not None and time_ref > 0:
            time_delta = (time_target - time_ref) / time_ref
        metrics.append(
            {
                "key": "time",
                "label": "Render time",
                "baseline": time_ref,
                "target": time_target,
                "delta": time_delta,
                "line": f"Render time: {_fmt_seconds(time_ref)} → {_fmt_seconds(time_target)}",
                "delta_text": _delta_label_pct(
                    time_delta,
                    higher_label="slower",
                    lower_label="faster",
                    invert_sign=True,
                ),
                "primary": wants_time,
            }
        )

        quality_delta = None
        if quality_baseline is not None and quality_current is not None:
            quality_delta = quality_current - quality_baseline
        metrics.append(
            {
                "key": "quality",
                "label": "LLM quality",
                "baseline": quality_baseline,
                "target": quality_current,
                "delta": quality_delta,
                "line": f"LLM quality: {_fmt_score(quality_baseline)} → {_fmt_score(quality_current)}",
                "delta_text": _delta_label_points(quality_delta, positive_label="better", negative_label="worse"),
                "primary": wants_quality,
            }
        )

        adherence_delta = None
        if adherence_baseline is not None and adherence_current is not None:
            adherence_delta = adherence_current - adherence_baseline
        metrics.append(
            {
                "key": "adherence",
                "label": "Prompt adherence",
                "baseline": adherence_baseline,
                "target": adherence_current,
                "delta": adherence_delta,
                "line": f"Prompt adherence: {_fmt_score(adherence_baseline)} → {_fmt_score(adherence_current)}",
                "delta_text": _delta_label_points(adherence_delta, positive_label="better", negative_label="worse"),
                "primary": wants_quality,
            }
        )

        retrieval_delta = None
        if retrieval_baseline is not None and retrieval_current is not None:
            retrieval_delta = retrieval_current - retrieval_baseline
        metrics.append(
            {
                "key": "retrieval",
                "label": "LLM retrieval",
                "baseline": retrieval_baseline,
                "target": retrieval_current,
                "delta": retrieval_delta,
                "line": f"LLM retrieval: {_fmt_score(retrieval_baseline)} → {_fmt_score(retrieval_current)}",
                "delta_text": _delta_label_points(retrieval_delta, positive_label="better", negative_label="worse"),
                "primary": wants_retrieval,
            }
        )

        order = {"time": 0, "cost": 1, "quality": 2, "adherence": 3, "retrieval": 4}
        metrics.sort(
            key=lambda item: (
                0 if item.get("primary") else 1,
                order.get(str(item.get("key")), 9),
            )
        )

        _append_wrapped("Metric changes (baseline → suggested):", "section")
        for metric in metrics:
            line = f"{metric['line']} ({metric['delta_text']})"
            tag = "goal" if metric.get("primary") else None
            _append_wrapped(line, tag)
    return lines


def _show_final_recommendation_curses(
    stdscr,
    *,
    settings: dict[str, object],
    recommendation: object,
    user_goals: list[str] | None,
    color_enabled: bool,
    last_elapsed: float | None = None,
    cost_line: str | None = None,
    baseline_elapsed: float | None = None,
    baseline_cost_line: str | None = None,
    quality_baseline: int | None = None,
    quality_current: int | None = None,
    adherence_baseline: int | None = None,
    adherence_current: int | None = None,
    retrieval_baseline: int | None = None,
    retrieval_current: int | None = None,
    receipt_path: Path | None = None,
    baseline_settings: dict[str, object] | None = None,
    force_quality_metrics: bool = False,
    summary_note: str | None = None,
    metrics_settings: dict[str, object] | None = None,
) -> None:
    import curses
    height, width = stdscr.getmaxyx()
    max_width = max(40, width - 2)
    lines = _build_final_recommendation_lines(
        settings,
        recommendation,
        user_goals=user_goals,
        max_width=max_width,
        return_tags=True,
        last_elapsed=last_elapsed,
        cost_line=cost_line,
        baseline_settings=baseline_settings,
        baseline_elapsed=baseline_elapsed,
        baseline_cost_line=baseline_cost_line,
        quality_baseline=quality_baseline,
        quality_current=quality_current,
        adherence_baseline=adherence_baseline,
        adherence_current=adherence_current,
        retrieval_baseline=retrieval_baseline,
        retrieval_current=retrieval_current,
        force_quality_metrics=force_quality_metrics,
        metrics_settings=metrics_settings,
    )
    if summary_note:
        insert_at = 1 if len(lines) > 1 else len(lines)
        lines.insert(insert_at, (summary_note, "goal"))
    if baseline_settings:
        recommended_settings = _settings_after_recommendation(settings, recommendation) if recommendation else settings
        diff_lines = _diff_call_settings(baseline_settings, recommended_settings)
        if diff_lines:
            lines.append("")
            lines.append(("Differences vs Round 1 baseline (final settings)", "section"))
            for diff in diff_lines:
                prefix = "Change: " if not diff.startswith("Change:") else ""
                for line in _wrap_text(f"{prefix}{diff}", max_width):
                    lines.append((line, "change"))
    footer_line = "Open receipt viewer (o) • Press Q or Enter to exit"
    open_keys = {ord("o"), ord("O")} if receipt_path else None
    hotkeys = None
    if receipt_path:
        hotkeys = {"o": curses.color_pair(1) | curses.A_BOLD if color_enabled else curses.A_BOLD}
    while True:
        action = _render_scrollable_text_with_banner(
            stdscr,
            title_line=f"Final recommendation (max {MAX_ROUNDS} rounds)",
            body_lines=lines,
            footer_line=footer_line if receipt_path else "Press Q or Enter to exit",
            color_enabled=color_enabled,
            footer_attr=curses.A_DIM,
            footer_hotkeys=hotkeys,
            open_keys=open_keys,
        )
        if action == "open" and receipt_path:
            _maybe_open_viewer(receipt_path.parent)
            continue
        break




def _prompt_prompt_curses(
    stdscr,
    *,
    default_prompt: str,
    color_enabled: bool,
    initial_text: str | None = None,
) -> str:
    import curses
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    y = _draw_banner(stdscr, color_enabled, 1)
    title = "Enter a prompt (press Enter to use the default prompt)."
    _safe_addstr(stdscr, min(height - 2, y), 0, title[: max(0, width - 1)], curses.A_BOLD)
    y = min(height - 2, y + 2)
    default_line = _truncate_text(default_prompt, max(0, width - 1))
    if default_line:
        _safe_addstr(
            stdscr,
            min(height - 2, y),
            0,
            f"Default: {default_line}"[: max(0, width - 1)],
            curses.A_DIM,
        )
        y = min(height - 2, y + 2)
    input_y = min(height - 1, y + 1)
    buffer = initial_text or ""
    try:
        curses.curs_set(1)
    except curses.error:
        pass
    stdscr.timeout(-1)
    while True:
        stdscr.move(input_y, 0)
        stdscr.clrtoeol()
        display = buffer
        max_len = max(1, width - 1)
        if len(display) > max_len:
            display = display[-max_len:]
        _safe_addstr(stdscr, input_y, 0, display[: max(0, width - 1)])
        stdscr.refresh()
        key = stdscr.getch()
        if key in (10, 13, curses.KEY_ENTER):
            break
        if key in (curses.KEY_BACKSPACE, 127, 8):
            buffer = buffer[:-1]
            continue
        if 32 <= key <= 126:
            buffer += chr(key)
            continue
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.timeout(80)
    text = buffer.strip()
    return text or default_prompt


def _prompt_freeform_curses(stdscr, prompt: str, *, color_enabled: bool) -> str:
    import curses
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    y = _draw_banner(stdscr, color_enabled, 1)
    if y < height - 1:
        _safe_addstr(stdscr, y, 0, prompt[: max(0, width - 1)], curses.A_BOLD)
        y += 2
    _safe_addstr(stdscr, min(height - 2, y), 0, "Type your notes and press Enter."[: max(0, width - 1)], curses.A_DIM)
    stdscr.refresh()
    curses.echo()
    try:
        entry = stdscr.getstr(min(height - 1, y + 1), 0, max(1, width - 1))
        text = entry.decode("utf-8", errors="ignore").strip()
    except Exception:
        text = ""
    finally:
        curses.noecho()
    return text


def _prompt_goal_selection_curses(stdscr, color_enabled: bool) -> tuple[list[str], str | None] | None:
    import curses
    options = [
        "minimize time to render",
        "minimize cost of render",
        "maximize quality of render",
        "maximize LLM retrieval score",
        "something else",
    ]
    selected = [False] * len(options)
    idx = 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        y = _draw_banner(stdscr, color_enabled, 1)
        title = "Select your goals (space to toggle, Enter to continue)"
        _safe_addstr(stdscr, min(height - 2, y), 0, title[: max(0, width - 1)], curses.A_BOLD)
        y = min(height - 2, y + 2)
        for i, option in enumerate(options):
            if y >= height - 1:
                break
            marker = "[x]" if selected[i] else "[ ]"
            line = f"{marker} {option}"
            attr = curses.A_DIM
            if i == idx:
                attr = curses.color_pair(2) | curses.A_BOLD if color_enabled else curses.A_BOLD
            _safe_addstr(stdscr, y, 2, line[: max(0, width - 3)], attr)
            y += 1
        footer = "Space: toggle • Enter: continue • Q/Esc: cancel"
        _safe_addstr(stdscr, height - 1, 0, footer[: max(0, width - 1)], curses.A_DIM)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            return None
        if key in (curses.KEY_UP, ord("k"), ord("K")):
            idx = (idx - 1) % len(options)
            continue
        if key in (curses.KEY_DOWN, ord("j"), ord("J")):
            idx = (idx + 1) % len(options)
            continue
        if key == ord(" "):
            selected[idx] = not selected[idx]
            continue
        if key in (10, 13, curses.KEY_ENTER):
            if not any(selected):
                selected[idx] = True
            break

    goals = [opt for opt, is_on in zip(options, selected) if is_on]
    notes = None
    if "something else" in goals:
        notes = _prompt_freeform_curses(
            stdscr,
            "Tell us more about your goal:",
            color_enabled=color_enabled,
        )
    return goals, notes


def _prompt_yes_no_curses(stdscr, prompt: str, *, color_enabled: bool) -> bool:
    import curses
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        y = _draw_banner(stdscr, color_enabled, 1)
        if y < height - 2:
            _safe_addstr(stdscr, y, 0, prompt[: max(0, width - 1)], curses.A_BOLD)
            y += 2
        footer = "Press Y to run, N or Esc to cancel"
        _safe_addstr(stdscr, min(height - 1, y), 0, footer[: max(0, width - 1)], curses.A_DIM)
        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("y"), ord("Y")):
            return True
        if key in (ord("n"), ord("N"), 27):
            return False


def _render_scrollable_text(
    stdscr,
    *,
    header_lines: list[str],
    body_lines: list[object],
    footer_line: str,
    color_enabled: bool,
) -> None:
    import curses
    top = 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        y = 0
        for line in header_lines:
            if y >= height - 1:
                break
            _safe_addstr(stdscr, y, 0, line[: max(0, width - 1)], curses.A_BOLD)
            y += 1
        available = max(1, height - y - 1)
        max_top = max(0, len(body_lines) - available)
        visible = body_lines[top : top + available]
        for line in visible:
            if y >= height - 1:
                break
            text, attr = _line_text_and_attr(line, color_enabled=color_enabled)
            _safe_addstr(stdscr, y, 0, text[: max(0, width - 1)], attr)
            y += 1
        if height > 0:
            _safe_addstr(
                stdscr,
                height - 1,
                0,
                footer_line[: max(0, width - 1)],
                curses.A_DIM,
            )
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27, 10, 13):
            break
        if key in (curses.KEY_UP, ord("k"), ord("K")):
            top = max(0, top - 1)
        elif key in (curses.KEY_DOWN, ord("j"), ord("J")):
            top = min(max_top, top + 1)
        elif key in (curses.KEY_PPAGE,):
            top = max(0, top - available)
        elif key in (curses.KEY_NPAGE,):
            top = min(max_top, top + available)
        elif key == curses.KEY_HOME:
            top = 0
        elif key == curses.KEY_END:
            top = max_top


def _render_scrollable_text_with_banner(
    stdscr,
    *,
    title_line: str,
    body_lines: list[object],
    footer_line: str,
    color_enabled: bool,
    emphasis_line: str | None = None,
    footer_attr: int | None = None,
    footer_hotkeys: dict[str, int] | None = None,
    accept_keys: set[int] | None = None,
    open_keys: set[int] | None = None,
    action_keys: dict[int, str] | None = None,
) -> str:
    import curses
    top = 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        y = _draw_banner(stdscr, color_enabled, 1)
        if y < height - 1:
            _safe_addstr(stdscr, y, 0, title_line[: max(0, width - 1)], curses.A_BOLD)
            y += 1
        if emphasis_line and y < height - 1:
            for line in _wrap_text(emphasis_line, max(20, width - 2)):
                if y >= height - 1:
                    break
                _safe_addstr(stdscr, y, 0, line[: max(0, width - 1)], curses.A_BOLD)
                y += 1
        if y < height - 1:
            y += 1
        available = max(1, height - y - 1)
        max_top = max(0, len(body_lines) - available)
        visible = body_lines[top : top + available]
        for line in visible:
            if y >= height - 1:
                break
            text, attr = _line_text_and_attr(line, color_enabled=color_enabled)
            _safe_addstr(stdscr, y, 0, text[: max(0, width - 1)], attr)
            y += 1
        if height > 0:
            attr = footer_attr if footer_attr is not None else curses.A_DIM
            _safe_addstr(
                stdscr,
                height - 1,
                0,
                footer_line[: max(0, width - 1)],
                attr,
            )
            if footer_hotkeys:
                for key_char, key_attr in footer_hotkeys.items():
                    pos = footer_line.find(f"({key_char})")
                    if pos != -1 and pos + 1 < width:
                        _safe_addstr(stdscr, height - 1, pos + 1, key_char, key_attr)
        stdscr.refresh()

        key = stdscr.getch()
        if accept_keys and key in accept_keys:
            return "accept"
        if open_keys and key in open_keys:
            return "open"
        if action_keys and key in action_keys:
            return action_keys[key]
        if key in (ord("q"), ord("Q"), 27, 10, 13):
            return "exit"
        if key in (curses.KEY_UP, ord("k"), ord("K")):
            top = max(0, top - 1)
        elif key in (curses.KEY_DOWN, ord("j"), ord("J")):
            top = min(max_top, top + 1)
        elif key in (curses.KEY_PPAGE,):
            top = max(0, top - available)
        elif key in (curses.KEY_NPAGE,):
            top = min(max_top, top + available)
        elif key == curses.KEY_HOME:
            top = 0
        elif key == curses.KEY_END:
            top = max_top
    return "exit"


def _show_receipt_analysis_curses(
    stdscr,
    *,
    receipt_path: Path,
    provider: str,
    model: str | None,
    size: str,
    n: int,
    out_dir: str,
    analyzer: str | None = None,
    color_enabled: bool,
    user_goals: list[str] | None = None,
    user_notes: str | None = None,
    analysis_history: list[dict] | None = None,
    emphasis_line: str | None = None,
    allow_rerun: bool = True,
    last_elapsed: float | None = None,
    last_cost: str | None = None,
    benchmark_elapsed: float | None = None,
    round_scores: tuple[int | None, int | None] | None = None,
) -> tuple[list[dict] | None, str | None, bool, bool]:
    import curses
    analyzer_key = _normalize_analyzer(analyzer)
    cost_line: str | None = None
    pre_cost_line: str | None = None
    current_round = len(analysis_history or []) + 1
    rounds_left = max(0, MAX_ROUNDS - current_round)
    baseline_elapsed: float | None = None
    baseline_cost_text: str | None = None
    if analysis_history:
        first_entry = analysis_history[0]
        first_elapsed = first_entry.get("elapsed")
        if isinstance(first_elapsed, (int, float)):
            baseline_elapsed = float(first_elapsed)
        baseline_cost_text = _strip_cost_prefix(first_entry.get("cost"))

    if user_goals and "minimize cost of render" in user_goals and not last_cost:
        pre_cost_line = _estimate_cost_only(
            provider=provider,
            model=model,
            size=size,
            n=n,
        )
        if pre_cost_line:
            last_cost = str(pre_cost_line).replace("COST:", "").strip()

    stdscr.erase()
    height, width = stdscr.getmaxyx()
    y = _draw_banner(stdscr, color_enabled, 1)
    if y < height - 1:
        _safe_addstr(
            stdscr,
            y,
            0,
            f"Analyzing receipt with {_analyzer_display_name(analyzer_key)}..."[: max(0, width - 1)],
        )
        y += 1
    if analyzer_key == "council" and y < height - 1:
        note_text = "Note: Council analysis can take a few minutes."
        note_attr = curses.A_DIM if hasattr(curses, "A_DIM") else 0
        _safe_addstr(stdscr, y, 0, note_text[: max(0, width - 1)], note_attr)
        y += 1
    if y < height - 1:
        model_label = model or "(default)"
        provider_line = f"Provider/Model: {_display_provider_name(provider) or provider} • {model_label}"
        _safe_addstr(stdscr, y, 0, provider_line[: max(0, width - 1)], curses.A_BOLD)
        y += 1
    if user_goals and y < height - 1:
        goals_text = "Goals: " + ", ".join(user_goals)
        goals_attr = curses.color_pair(4) | curses.A_BOLD if color_enabled else curses.A_BOLD
        _safe_addstr(stdscr, y, 0, goals_text[: max(0, width - 1)], goals_attr)
        y += 1
    if y < height - 1:
        rounds_text = f"Rounds left: {rounds_left} (of {MAX_ROUNDS})"
        rounds_attr = curses.color_pair(4) | curses.A_BOLD if color_enabled else curses.A_BOLD
        _safe_addstr(stdscr, y, 0, rounds_text[: max(0, width - 1)], rounds_attr)
        y += 1
    if user_goals and "minimize time to render" in user_goals and y < height - 1:
        if baseline_elapsed is not None and last_elapsed is not None:
            speed_text = f"Speed baseline: {baseline_elapsed:.1f}s → latest {last_elapsed:.1f}s"
        elif benchmark_elapsed is not None and last_elapsed is not None:
            speed_text = f"Speed benchmark: {benchmark_elapsed:.1f}s → latest {last_elapsed:.1f}s"
        else:
            speed_text = "Last render: unknown"
            if last_elapsed is not None:
                speed_text = f"Last render: {last_elapsed:.1f}s"
        _safe_addstr(stdscr, y, 0, speed_text[: max(0, width - 1)], curses.A_BOLD)
        y += 1
    if user_goals and "minimize cost of render" in user_goals and y < height - 1:
        cost_text = "Cost benchmark: estimating..."
        if baseline_cost_text:
            cost_text = f"Cost baseline: {baseline_cost_text}"
        elif last_cost:
            cost_text = f"Cost benchmark: {last_cost}"
        _safe_addstr(stdscr, y, 0, cost_text[: max(0, width - 1)], curses.A_BOLD)
        y += 1
    stdscr.refresh()
    result_holder: dict[str, object] = {}
    done = threading.Event()

    def _run_analysis() -> None:
        try:
            history_text = _format_analysis_history(analysis_history or [])
            result_holder["payload"] = _analyze_receipt_payload(
                receipt_path=receipt_path,
                provider=provider,
                model=model,
                size=size,
                n=n,
                out_dir=out_dir,
                analyzer=analyzer_key,
                user_goals=user_goals,
                user_notes=user_notes,
                history_text=history_text,
                history_rounds=len(analysis_history or []),
            )
        except Exception as exc:
            result_holder["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_run_analysis, daemon=True)
    thread.start()
    status_y = min(y, height - 1)
    frames = ["|", "/", "-", "\\"]
    frame_idx = 0
    while not done.is_set():
        status = f"Analyzing… {frames[frame_idx % len(frames)]}"
        _safe_addstr(stdscr, status_y, 0, status[: max(0, width - 1)], curses.A_DIM)
        stdscr.refresh()
        frame_idx += 1
        time.sleep(0.1)
    thread.join()

    if "error" in result_holder:
        exc = result_holder["error"]
        body_lines = _wrap_text(f"Receipt analysis failed: {exc}", max(20, width - 2))
        _render_scrollable_text_with_banner(
            stdscr,
            title_line="Receipt analysis failed",
            body_lines=body_lines,
            footer_line="Press Q or Enter to exit",
            color_enabled=color_enabled,
        )
        return None, pre_cost_line or cost_line, False, False
    analysis, citations, recommendation, cost_line, stop_reason = result_holder.get("payload")  # type: ignore[misc]
    stop_reason = None
    if not cost_line and pre_cost_line:
        cost_line = pre_cost_line
    recommendations = _normalize_recommendations(recommendation)
    if not recommendations:
        recommendation = None
    else:
        recommendation = recommendations
    stop_recommended = False

    comparison_lines: list[str] = []
    if user_goals and "minimize time to render" in user_goals:
        if baseline_elapsed is not None and last_elapsed is not None:
            comparison_lines.append(f"Speed (baseline): {baseline_elapsed:.1f}s → {last_elapsed:.1f}s")
        elif benchmark_elapsed is not None and last_elapsed is not None:
            comparison_lines.append(f"Speed: {benchmark_elapsed:.1f}s → {last_elapsed:.1f}s")
    if user_goals and "minimize cost of render" in user_goals:
        latest_cost = str(cost_line).replace("COST:", "").strip() if cost_line else None
        if baseline_cost_text and latest_cost:
            comparison_lines.append(f"Cost (baseline): {baseline_cost_text} → {latest_cost}")
        elif last_cost and latest_cost:
            comparison_lines.append(f"Cost: {last_cost} → {latest_cost}")
    if comparison_lines:
        side_by_side = "\n".join(comparison_lines)
        if emphasis_line:
            emphasis_line = f"{emphasis_line}\n{side_by_side}"
        else:
            emphasis_line = side_by_side

    detail_lines: list[object] = []
    receipt_payload: dict | None = None
    quality_baseline, quality_current = _quality_from_history(
        analysis_history or [],
        round_scores[1] if round_scores else None,
    )
    adherence_baseline, adherence_current = _adherence_from_history(
        analysis_history or [],
        round_scores[0] if round_scores else None,
    )
    retrieval_current = _load_retrieval_score(receipt_path)
    retrieval_baseline, retrieval_current = _retrieval_from_history(
        analysis_history or [],
        retrieval_current,
    )
    try:
        receipt_payload = _load_receipt_payload(receipt_path)
        detail_lines = _build_receipt_detail_lines(
            receipt_payload,
            recommendation,
            max_width=max(20, width - 2),
            return_tags=True,
            stop_reason=None,
        )
    except Exception as exc:
        detail_lines = _wrap_text(f"Render settings unavailable: {exc}", max(20, width - 2))

    lines: list[str] = []
    if user_goals:
        goals_text = ", ".join(user_goals)
        lines.append((f"User goals: {goals_text}", "goal"))
        lines.append("")
    lines.append((f"Rounds left: {rounds_left} (of {MAX_ROUNDS})", "goal"))
    lines.append("")
    lines.append("MODEL ANALYSIS (LLM)")
    lines.append("")
    if analysis:
        display_analysis = _display_analysis_text(analysis)
        lines.extend(_wrap_text(display_analysis, max(20, width - 2)))
    else:
        lines.append("Receipt analysis returned no content.")
    if detail_lines:
        lines.append("")
        lines.extend(detail_lines)
    if not recommendations:
        summary_settings = _settings_from_receipt(
            receipt_payload,
            fallback_provider=provider,
            fallback_model=model,
            fallback_size=size,
            fallback_n=n,
        )
        baseline_elapsed, baseline_cost_line = _baseline_metrics_from_history(analysis_history or [])
        lines.append("")
        lines.extend(
            _build_final_recommendation_lines(
                summary_settings,
                None,
                user_goals=user_goals,
                max_width=max(20, width - 2),
                return_tags=True,
                last_elapsed=last_elapsed,
                cost_line=cost_line,
                baseline_elapsed=baseline_elapsed,
                baseline_cost_line=baseline_cost_line,
                quality_baseline=quality_baseline,
                quality_current=quality_current,
                adherence_baseline=adherence_baseline,
                adherence_current=adherence_current,
                retrieval_baseline=retrieval_baseline,
                retrieval_current=retrieval_current,
                force_quality_metrics=bool(user_goals and "maximize quality of render" in user_goals),
            )
        )

    rec_label = "recommendations" if recommendations and len(recommendations) > 1 else "recommendation"
    footer_line = (
        "Open receipt (o) • Export text (t) • Print text (p) • "
        f"Accept {rec_label} (y) or quit (q) • "
        "Up/Down/PgUp/PgDn to scroll"
    )
    footer_attr = curses.color_pair(4) | curses.A_BOLD if color_enabled else curses.A_BOLD
    hotkeys = None
    accept_keys = None
    if allow_rerun and recommendations:
        hotkeys = {
            "y": curses.color_pair(1) | curses.A_BOLD if color_enabled else curses.A_BOLD,
            "q": curses.color_pair(1) | curses.A_BOLD if color_enabled else curses.A_BOLD,
            "o": curses.color_pair(1) | curses.A_BOLD if color_enabled else curses.A_BOLD,
            "t": curses.color_pair(1) | curses.A_BOLD if color_enabled else curses.A_BOLD,
            "p": curses.color_pair(1) | curses.A_BOLD if color_enabled else curses.A_BOLD,
        }
        accept_keys = {ord("y"), ord("Y")}
    else:
        hotkeys = {
            "o": curses.color_pair(1) | curses.A_BOLD if color_enabled else curses.A_BOLD,
            "t": curses.color_pair(1) | curses.A_BOLD if color_enabled else curses.A_BOLD,
            "p": curses.color_pair(1) | curses.A_BOLD if color_enabled else curses.A_BOLD,
        }
    open_keys = {ord("o"), ord("O")}
    action_keys = {
        ord("t"): "export",
        ord("T"): "export",
        ord("p"): "print",
        ord("P"): "print",
    }
    while True:
        action = _render_scrollable_text_with_banner(
            stdscr,
            title_line="Receipt analysis",
            body_lines=lines,
            footer_line=(
                footer_line
                if accept_keys
                else "Open receipt (o) • Export text (t) • Print text (p) • Up/Down/PgUp/PgDn to scroll, Q or Enter to exit"
            ),
            color_enabled=color_enabled,
            emphasis_line=emphasis_line,
            footer_attr=footer_attr,
            footer_hotkeys=hotkeys,
            accept_keys=accept_keys,
            open_keys=open_keys,
            action_keys=action_keys,
        )
        if action == "export":
            export_lines: list[str] = []
            if user_goals:
                export_lines.append(f"User goals: {', '.join(user_goals)}")
                export_lines.append("")
            if analysis:
                export_lines.append("MODEL ANALYSIS (LLM)")
                export_lines.append("")
                export_lines.extend(_display_analysis_text(analysis).splitlines())
            else:
                export_lines.append("MODEL ANALYSIS (LLM)")
                export_lines.append("")
                export_lines.append("Receipt analysis returned no content.")
            if receipt_payload is not None:
                export_lines.append("")
                export_lines.extend(
                    [
                        line
                        for line in _build_receipt_detail_lines(
                            receipt_payload,
                            recommendation,
                            max_width=120,
                            stop_reason=None,
                        )
                        if isinstance(line, str)
                    ]
                )
            else:
                export_lines.append("")
                export_lines.extend([str(line) for line in lines])
            try:
                export_path = _export_analysis_text(receipt_path, export_lines)
                _open_path(export_path)
            except Exception:
                pass
            continue
        if action == "print":
            print_lines: list[str] = []
            if user_goals:
                print_lines.append(f"User goals: {', '.join(user_goals)}")
                print_lines.append("")
            print_lines.append("MODEL ANALYSIS (LLM)")
            print_lines.append("")
            if analysis:
                print_lines.extend(_display_analysis_text(analysis).splitlines())
            else:
                print_lines.append("Receipt analysis returned no content.")
            if receipt_payload is not None:
                print_lines.append("")
                for line in _build_receipt_detail_lines(
                    receipt_payload,
                    recommendation,
                    max_width=120,
                    stop_reason=stop_reason,
                ):
                    if isinstance(line, tuple):
                        print_lines.append(line[0])
                    else:
                        print_lines.append(line)
            _print_lines_to_console(stdscr, "Receipt analysis (copyable):", print_lines)
            continue
        if action == "open":
            _open_path(receipt_path)
            continue
        if allow_rerun and recommendations and action == "accept":
            return recommendations, cost_line, True, stop_recommended
        return recommendations, cost_line, False, stop_recommended


def _draw_choice_line(
    stdscr,
    y: int,
    label: str,
    choices: list[str],
    selected_idx: int,
    active: bool,
    current_step: int,
    step_index: int,
    highlight_pair: int,
    done_pair: int,
    color_enabled: bool,
) -> int:
    import curses
    height, width = stdscr.getmaxyx()
    if y >= height - 1:
        return y
    width = max(1, width - 1)
    is_done = step_index < current_step
    label_attr = curses.A_BOLD if active else (curses.A_NORMAL if is_done else curses.A_DIM)
    prefix = "-> " if active else ("✓  " if is_done else "   ")
    try:
        stdscr.addstr(y, 0, f"{prefix}{label}:"[:width], label_attr)
    except curses.error:
        return y + 1
    y += 1
    indices, prefix, suffix = _visible_indices(choices, selected_idx, width, bracket_selected=active)
    x = 4
    if prefix:
        try:
            stdscr.addstr(y, x, "… ")
        except curses.error:
            return y + 2
        x += 2
    for idx, choice_idx in enumerate(indices):
        token = choices[choice_idx]
        if choice_idx == selected_idx:
            if active:
                token = f"[{token}]"
                if color_enabled and highlight_pair:
                    attr = curses.color_pair(highlight_pair) | curses.A_BOLD
                else:
                    attr = curses.A_REVERSE
            elif is_done:
                if color_enabled and done_pair:
                    attr = curses.color_pair(done_pair) | curses.A_BOLD
                else:
                    attr = curses.A_BOLD
            else:
                attr = curses.A_DIM
        else:
            attr = curses.A_DIM
        if y >= height - 1 or x >= width - 1:
            break
        try:
            stdscr.addstr(y, x, token[: max(0, width - x - 1)], attr)
        except curses.error:
            break
        x += len(token)
        if idx < len(indices) - 1:
            if x < width - 1:
                try:
                    stdscr.addstr(y, x, " | "[: max(0, width - x - 1)])
                except curses.error:
                    pass
            x += 3
    if suffix and x < width - 1:
        try:
            stdscr.addstr(y, x, " …")
        except curses.error:
            pass
    return y + 2


def _draw_choice_column(
    stdscr,
    y: int,
    label: str,
    choices: list[str],
    selected_idx: int,
    active: bool,
    current_step: int,
    step_index: int,
    highlight_pair: int,
    done_pair: int,
    color_enabled: bool,
    max_visible: int = 6,
) -> int:
    import curses
    height, width = stdscr.getmaxyx()
    if y >= height - 1:
        return y
    width = max(1, width - 1)
    is_done = step_index < current_step
    label_attr = curses.A_BOLD if active else (curses.A_NORMAL if is_done else curses.A_DIM)
    prefix = "-> " if active else ("✓  " if is_done else "   ")
    try:
        stdscr.addstr(y, 0, f"{prefix}{label}:"[:width], label_attr)
    except curses.error:
        return y + 1
    y += 1

    available = max(1, min(max_visible, height - y - 1))
    total = len(choices)
    if total == 0:
        return y + 1
    start = max(0, min(total - available, selected_idx - available // 2))
    end = min(total, start + available)

    for idx in range(start, end):
        token = choices[idx]
        if idx == selected_idx:
            if active:
                token = f"[{token}]"
                if color_enabled and highlight_pair:
                    attr = curses.color_pair(highlight_pair) | curses.A_BOLD
                else:
                    attr = curses.A_REVERSE
            elif is_done:
                if color_enabled and done_pair:
                    attr = curses.color_pair(done_pair) | curses.A_BOLD
                else:
                    attr = curses.A_BOLD
            else:
                attr = curses.A_DIM
        else:
            attr = curses.A_DIM
        if y >= height - 1:
            break
        _safe_addstr(stdscr, y, 4, token[: max(0, width - 5)], attr)
        y += 1

    return y + 1


def _draw_count_line(
    stdscr,
    y: int,
    label: str,
    value: int,
    active: bool,
    current_step: int,
    step_index: int,
    highlight_pair: int,
    done_pair: int,
    color_enabled: bool,
) -> int:
    import curses
    height, width = stdscr.getmaxyx()
    if y >= height - 1:
        return y
    is_done = step_index < current_step
    label_attr = curses.A_BOLD if active else (curses.A_NORMAL if is_done else curses.A_DIM)
    prefix = "-> " if active else ("✓  " if is_done else "   ")
    try:
        stdscr.addstr(y, 0, f"{prefix}{label}:"[: width - 1], label_attr)
    except curses.error:
        return y + 1
    y += 1
    token = f"[{value}]" if active else str(value)
    if active:
        if color_enabled and highlight_pair:
            attr = curses.color_pair(highlight_pair) | curses.A_BOLD
        else:
            attr = curses.A_REVERSE
    elif is_done:
        if color_enabled and done_pair:
            attr = curses.color_pair(done_pair) | curses.A_BOLD
        else:
            attr = curses.A_BOLD
    else:
        attr = curses.A_DIM
    if y < height - 1:
        try:
            stdscr.addstr(y, 4, token[: max(0, width - 5)], attr)
        except curses.error:
            pass
    return y + 2


def _draw_prompt_line(
    stdscr,
    y: int,
    label: str,
    prompt_text: str,
    default_prompt: str,
    active: bool,
    current_step: int,
    step_index: int,
    highlight_pair: int,
    done_pair: int,
    color_enabled: bool,
    hint_text: str | None = None,
) -> int:
    import curses
    height, width = stdscr.getmaxyx()
    if y >= height - 1:
        return y
    is_done = step_index < current_step
    label_attr = curses.A_BOLD if active else (curses.A_NORMAL if is_done else curses.A_DIM)
    prefix = "-> " if active else ("✓  " if is_done else "   ")
    try:
        stdscr.addstr(y, 0, f"{prefix}{label}:"[: width - 1], label_attr)
    except curses.error:
        return y + 1
    y += 1
    display = prompt_text.strip() if prompt_text else ""
    if display and default_prompt and display == default_prompt.strip():
        display = "(default prompt)"
    if not display:
        display = "(default prompt)"
    token = _truncate_text(display, max(1, width - 5))
    if active:
        if color_enabled and highlight_pair:
            attr = curses.color_pair(highlight_pair) | curses.A_BOLD
        else:
            attr = curses.A_REVERSE
    elif is_done:
        if color_enabled and done_pair:
            attr = curses.color_pair(done_pair) | curses.A_BOLD
        else:
            attr = curses.A_BOLD
    else:
        attr = curses.A_DIM
    _safe_addstr(stdscr, y, 4, token, attr)
    y += 1
    if hint_text and active:
        _safe_addstr(
            stdscr,
            y,
            4,
            _truncate_text(hint_text, max(1, width - 5)),
            curses.A_DIM,
        )
        y += 1
    return y + 1


def _visible_indices(
    choices: list[str],
    selected_idx: int,
    max_width: int,
    *,
    bracket_selected: bool = True,
) -> tuple[list[int], bool, bool]:
    def line_len(indices: list[int], prefix: bool, suffix: bool) -> int:
        tokens = []
        for i in indices:
            if i == selected_idx and bracket_selected:
                tokens.append(f"[{choices[i]}]")
            else:
                tokens.append(choices[i])
        line = "  " + " | ".join(tokens)
        if prefix:
            line = "… " + line
        if suffix:
            line = line + " …"
        return len(line)

    indices = [selected_idx]
    left = selected_idx - 1
    right = selected_idx + 1
    while True:
        grew = False
        if left >= 0 and line_len([left] + indices, True, right < len(choices)) <= max_width:
            indices = [left] + indices
            left -= 1
            grew = True
        if right < len(choices) and line_len(indices + [right], left >= 0, True) <= max_width:
            indices = indices + [right]
            right += 1
            grew = True
        if not grew:
            break
    prefix = indices[0] > 0
    suffix = indices[-1] < len(choices) - 1
    while line_len(indices, prefix, suffix) > max_width and len(indices) > 1:
        if selected_idx - indices[0] > indices[-1] - selected_idx:
            indices = indices[1:]
        else:
            indices = indices[:-1]
        prefix = indices[0] > 0
        suffix = indices[-1] < len(choices) - 1
    return indices, prefix, suffix


def _format_choices_line(choices: list[str], idx: int, max_width: int, color: bool) -> str:
    sel_start = "\x00SEL\x00"
    sel_end = "\x00END\x00"

    def token(i: int) -> str:
        if i == idx:
            return f"{sel_start}[{choices[i]}]{sel_end}"
        return choices[i]

    def line_for(indices: list[int]) -> str:
        return "  " + " | ".join(token(i) for i in indices)

    def visible_len(text: str) -> int:
        return len(text.replace(sel_start, "").replace(sel_end, ""))

    indices = [idx]
    left = idx - 1
    right = idx + 1
    while True:
        grew = False
        if left >= 0:
            candidate = [left] + indices
            if visible_len(line_for(candidate)) <= max_width:
                indices = candidate
                left -= 1
                grew = True
        if right < len(choices):
            candidate = indices + [right]
            if visible_len(line_for(candidate)) <= max_width:
                indices = candidate
                right += 1
                grew = True
        if not grew:
            break

    prefix = "… " if indices[0] > 0 else ""
    suffix = " …" if indices[-1] < len(choices) - 1 else ""
    rendered = line_for(indices)
    line = f"{prefix}{rendered}{suffix}"
    while visible_len(line) > max_width and len(indices) > 1:
        if idx - indices[0] > indices[-1] - idx:
            indices = indices[1:]
        else:
            indices = indices[:-1]
        rendered = line_for(indices)
        prefix = "… " if indices[0] > 0 else ""
        suffix = " …" if indices[-1] < len(choices) - 1 else ""
        line = f"{prefix}{rendered}{suffix}"
    if color and _supports_color():
        line = line.replace(sel_start, "\033[1;36m").replace(sel_end, "\033[0m")
    else:
        line = line.replace(sel_start, "").replace(sel_end, "")
    return line


class _Spinner:
    def __init__(self, message: str, interval: float = 0.1, show_elapsed: bool = False) -> None:
        self.message = message
        self.interval = interval
        self.show_elapsed = show_elapsed
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start: float | None = None

    def start(self) -> None:
        if sys.stdout.isatty():
            self._start = time.monotonic()
            self._thread.start()

    def stop(self) -> None:
        if not sys.stdout.isatty():
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")
        sys.stdout.flush()

    def _run(self) -> None:
        frames = ["|", "/", "-", "\\"]
        index = 0
        while not self._stop.is_set():
            frame = frames[index % len(frames)]
            line = f"\r{self.message} {frame}"
            if self.show_elapsed and self._start is not None:
                elapsed = time.monotonic() - self._start
                line = f"{line}  {elapsed:5.1f}s"
            sys.stdout.write(line)
            sys.stdout.flush()
            time.sleep(self.interval)
            index += 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_env_key(dotenv_path: Path, key: str, value: str) -> None:
    if not dotenv_path.parent.exists():
        dotenv_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if dotenv_path.exists():
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
    new_line = f'{key}="{escaped}"'
    replaced = False
    for idx, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[idx] = new_line
            replaced = True
            break
    if not replaced:
        lines.append(new_line)
    dotenv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(dotenv_path, 0o600)
    except Exception:
        pass


def _prompt_for_key(key: str, dotenv_path: Path | None) -> bool:
    choice = input(f"Set {key} now? [y/N]: ").strip().lower()
    if choice not in {"y", "yes"}:
        return False
    value = getpass.getpass(f"Enter {key}: ").strip()
    if not value:
        return False
    if dotenv_path is None:
        dotenv_path = _repo_root() / ".env"
    save = input(f"Save to {dotenv_path}? [Y/n]: ").strip().lower()
    if save in {"", "y", "yes"}:
        _write_env_key(dotenv_path, key, value)
        print(f"Saved {key} to {dotenv_path}.")
    else:
        print(f"export {key}=\"{value}\"")
    os.environ[key] = value
    return True


def _prompt_for_key_curses(stdscr, key: str, dotenv_path: Path | None) -> bool:
    try:
        import curses
    except Exception:
        return _prompt_for_key(key, dotenv_path)
    try:
        try:
            curses.def_prog_mode()
            curses.endwin()
        except Exception:
            pass
        return _prompt_for_key(key, dotenv_path)
    finally:
        try:
            curses.reset_prog_mode()
            curses.curs_set(0)
            stdscr.clear()
            stdscr.refresh()
        except Exception:
            pass


def _provider_key_hint(provider: str) -> str:
    provider_key = provider.strip().lower()
    if provider_key == "openai":
        return "OPENAI_API_KEY (or OPENAI_API_KEY_BACKUP)"
    if provider_key == "gemini":
        return "GEMINI_API_KEY (or GOOGLE_API_KEY)"
    if provider_key == "flux":
        return "BFL_API_KEY (or FLUX_API_KEY)"
    if provider_key == "imagen":
        return "GOOGLE_API_KEY (or Vertex credentials like GOOGLE_APPLICATION_CREDENTIALS)"
    return "the required API key"


def _ensure_api_keys(
    provider: str, dotenv_path: Path | None, allow_prompt: bool = True, prompt_func=None
) -> None:
    if prompt_func is None:
        prompt_func = _prompt_for_key
    provider = provider.strip().lower()
    if provider == "openai":
        if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_BACKUP"):
            return
        if not allow_prompt:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider.")
        if not prompt_func("OPENAI_API_KEY", dotenv_path):
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider.")
        return
    if provider == "gemini":
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return
        if not allow_prompt:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for Gemini provider.")
        if not prompt_func("GEMINI_API_KEY", dotenv_path):
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for Gemini provider.")
        return
    if provider == "flux":
        if os.getenv("BFL_API_KEY") or os.getenv("FLUX_API_KEY"):
            return
        if not allow_prompt:
            raise RuntimeError("BFL_API_KEY (or FLUX_API_KEY) is required for Flux provider.")
        if not prompt_func("BFL_API_KEY", dotenv_path):
            raise RuntimeError("BFL_API_KEY (or FLUX_API_KEY) is required for Flux provider.")
        return
    if provider == "imagen":
        vertex_present = bool(
            os.getenv("IMAGEN_VERTEX_PROJECT")
            or os.getenv("IMAGEN_VERTEX_SERVICE_ACCOUNT_FILE")
            or os.getenv("IMAGEN_VERTEX_SERVICE_ACCOUNT_JSON")
            or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
        if vertex_present:
            return
        if os.getenv("GOOGLE_API_KEY"):
            return
        if not allow_prompt:
            raise RuntimeError(
                "GOOGLE_API_KEY (or Vertex credentials) is required for Imagen provider."
            )
        if not prompt_func("GOOGLE_API_KEY", dotenv_path):
            raise RuntimeError(
                "GOOGLE_API_KEY (or Vertex credentials) is required for Imagen provider."
            )
        return


def _extract_anthropic_text(payload: dict) -> str:
    content = payload.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(content, dict) and content.get("type") == "text":
        text = content.get("text")
        if isinstance(text, str):
            return text.strip()
    return ""


def _extract_anthropic_text_with_citations(payload: dict) -> tuple[str, list[dict[str, str]]]:
    content = payload.get("content")
    if not isinstance(content, list):
        text = _extract_anthropic_text(payload)
        return text, []
    lines: list[str] = []
    citations: list[dict[str, str]] = []
    citation_index: dict[tuple[str, str], int] = {}
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        markers: list[str] = []
        raw_citations = item.get("citations")
        if isinstance(raw_citations, list):
            for citation in raw_citations:
                if not isinstance(citation, dict):
                    continue
                url = citation.get("url")
                title = citation.get("title") or url or "Source"
                if not isinstance(url, str) or not url:
                    continue
                key = (url, str(title))
                if key not in citation_index:
                    citation_index[key] = len(citations) + 1
                    citations.append(
                        {
                            "index": str(citation_index[key]),
                            "title": str(title),
                            "url": url,
                        }
                    )
                markers.append(f"[{citation_index[key]}]")
        suffix = f" {' '.join(markers)}" if markers else ""
        lines.append(f"{text}{suffix}".strip())
    return "\n".join(lines).strip(), citations


def _call_anthropic(
    prompt: str,
    *,
    model: str = ANTHROPIC_MODEL,
    max_tokens: int = ANTHROPIC_MAX_TOKENS,
    enable_web_search: bool = True,
    enable_thinking: bool = True,
    image_base64: str | None = None,
    image_mime: str | None = None,
) -> tuple[str, list[dict[str, str]]]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required to analyze receipts.")
    content_blocks: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    if image_base64 and image_mime:
        content_blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": image_mime, "data": image_base64},
            }
        )
    body: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": content_blocks,
            }
        ],
    }
    if enable_thinking:
        if max_tokens > ANTHROPIC_THINKING_BUDGET and ANTHROPIC_THINKING_BUDGET >= 1024:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": ANTHROPIC_THINKING_BUDGET,
            }
    if enable_web_search:
        body["tools"] = [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }
        ]
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        ANTHROPIC_ENDPOINT,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Anthropic request failed ({exc.code}): {raw}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Anthropic request failed: {exc.reason}") from exc
    payload = json.loads(raw)
    if isinstance(payload, dict) and payload.get("error"):
        message = payload.get("error", {}).get("message", raw)
        raise RuntimeError(f"Anthropic request failed: {message}")
    if isinstance(payload, dict):
        text, citations = _extract_anthropic_text_with_citations(payload)
        if text:
            return text, citations
    return str(payload).strip(), []


def _extract_openai_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()
    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    text = block.get("text") or block.get("output_text")
                    if isinstance(text, str):
                        parts.append(text)
            elif isinstance(content, str):
                parts.append(content)
        if parts:
            return "\n".join(parts).strip()
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                if parts:
                    return "\n".join(parts).strip()
    return ""


def _extract_gemini_text(payload: dict) -> str:
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            content = candidate.get("content") if isinstance(candidate, dict) else None
            if isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str) and text.strip():
                                return text.strip()
    output = payload.get("text")
    if isinstance(output, str):
        return output.strip()
    return ""


def _openai_request(
    url: str,
    *,
    body: dict[str, object],
    api_key: str,
) -> tuple[dict | None, int | None, str | None]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")
        message = raw
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict) and payload.get("error"):
                message = str(payload.get("error", {}).get("message", raw))
        except Exception:
            pass
        return None, exc.code, message
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"OpenAI response parse failed: {exc}") from exc
    if isinstance(payload, dict) and payload.get("error"):
        message = str(payload.get("error", {}).get("message", raw))
        return None, None, message
    return payload, None, None


def _is_openai_endpoint_error(code: int | None, message: str | None) -> bool:
    if code == 404:
        return True
    if not message:
        return False
    lowered = message.lower()
    return "not found" in lowered or "unknown endpoint" in lowered or "unknown url" in lowered


def _is_openai_image_error(message: str | None) -> bool:
    if not message:
        return False
    lowered = message.lower()
    return (
        "image" in lowered
        and (
            "not supported" in lowered
            or "unsupported" in lowered
            or "invalid" in lowered
            or "vision" in lowered
            or "does not support" in lowered
        )
    )


def _call_openai(
    prompt: str,
    *,
    model: str = OPENAI_ANALYZER_MODEL,
    max_output_tokens: int = OPENAI_MAX_OUTPUT_TOKENS,
    image_base64: str | None = None,
    image_mime: str | None = None,
    enable_web_search: bool = False,
) -> tuple[str, list[dict[str, str]]]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_BACKUP")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to analyze receipts.")
    include_image = bool(image_base64 and image_mime)

    def _responses_body(with_image: bool) -> dict[str, object]:
        content: list[dict[str, object]] = [{"type": "input_text", "text": prompt}]
        if with_image and image_base64 and image_mime:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{image_mime};base64,{image_base64}",
                }
            )
        body: dict[str, object] = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_output_tokens,
        }
        if enable_web_search:
            body["tools"] = [{"type": "web_search"}]
        return body

    def _chat_body(with_image: bool) -> dict[str, object]:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        if with_image and image_base64 and image_mime:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_mime};base64,{image_base64}"},
                }
            )
        body: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_output_tokens,
        }
        return body

    def _attempt(url: str, body: dict[str, object]) -> tuple[str | None, int | None, str | None]:
        payload, code, message = _openai_request(url, body=body, api_key=api_key)
        if payload is not None:
            if isinstance(payload, dict):
                text = _extract_openai_text(payload)
                if text:
                    return text, None, None
            return str(payload).strip(), None, None
        return None, code, message

    text, code, message = _attempt(OPENAI_RESPONSES_ENDPOINT, _responses_body(include_image))
    if text is not None:
        return text, []
    if include_image and _is_openai_image_error(message):
        text, code, message = _attempt(OPENAI_RESPONSES_ENDPOINT, _responses_body(False))
        if text is not None:
            return text, []
    if _is_openai_endpoint_error(code, message):
        if enable_web_search:
            detail = message or "unknown error"
            code_label = f" ({code})" if code else ""
            raise RuntimeError(f"OpenAI request failed{code_label}: {detail}")
        text, code, message = _attempt(OPENAI_CHAT_ENDPOINT, _chat_body(include_image))
        if text is not None:
            return text, []
        if include_image and _is_openai_image_error(message):
            text, code, message = _attempt(OPENAI_CHAT_ENDPOINT, _chat_body(False))
            if text is not None:
                return text, []
    detail = message or "unknown error"
    code_label = f" ({code})" if code else ""
    raise RuntimeError(f"OpenAI request failed{code_label}: {detail}")


def _call_gemini(
    prompt: str,
    *,
    model: str = GEMINI_ANALYZER_MODEL,
    image_base64: str | None = None,
    image_mime: str | None = None,
    enable_web_search: bool = False,
) -> tuple[str, list[dict[str, str]]]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required to analyze receipts with Gemini.")
    parts: list[dict[str, object]] = [{"text": prompt}]
    if image_base64 and image_mime:
        parts.append(
            {
                "inline_data": {
                    "mime_type": image_mime,
                    "data": image_base64,
                }
            }
        )
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseMimeType": "text/plain"},
    }
    if enable_web_search:
        body["tools"] = [{"google_search": {}}]
    endpoint = f"{GEMINI_ENDPOINT}/models/{model}:generateContent?key={api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini request failed ({exc.code}): {raw}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini request failed: {exc.reason}") from exc
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Gemini response parse failed: {exc}") from exc
    if isinstance(payload, dict) and payload.get("error"):
        message = payload.get("error", {}).get("message", raw)
        raise RuntimeError(f"Gemini request failed: {message}")
    if isinstance(payload, dict):
        text = _extract_gemini_text(payload)
        if text:
            return text, []
    return str(payload).strip(), []


def _call_council(
    prompt: str,
    *,
    enable_web_search: bool = True,
    image_base64: str | None = None,
    image_mime: str | None = None,
    fallback_prompt: str | None = None,
    chair_instructions: list[str] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    opinions: list[tuple[str, str]] = []
    errors: list[str] = []
    lock = threading.Lock()
    ready = threading.Event()
    completed = 0
    total = 3
    quorum = 2

    def _mark_done() -> None:
        nonlocal completed
        completed += 1
        if len(opinions) >= quorum or completed >= total:
            ready.set()

    def _record_success(label: str, text: str | None) -> None:
        with lock:
            if text:
                opinions.append((label, text))
            else:
                errors.append(f"{label} returned empty response.")
            _mark_done()

    def _record_error(label: str, exc: Exception) -> None:
        with lock:
            errors.append(f"{label} failed: {exc}")
            _mark_done()

    def _run_openai() -> None:
        try:
            text, _ = _call_openai(
                prompt,
                image_base64=image_base64,
                image_mime=image_mime,
                enable_web_search=enable_web_search,
            )
            _record_success("OpenAI GPT-5.2 analyst", text)
        except Exception as exc:
            _record_error("OpenAI analyst", exc)

    def _run_anthropic() -> None:
        try:
            text, _ = _call_anthropic(
                prompt,
                enable_web_search=enable_web_search,
                image_base64=image_base64,
                image_mime=image_mime,
            )
            _record_success("Claude Opus 4.5 analyst", text)
        except Exception as exc:
            _record_error("Claude analyst", exc)

    def _run_gemini() -> None:
        try:
            text, _ = _call_gemini(
                prompt,
                image_base64=image_base64,
                image_mime=image_mime,
                enable_web_search=enable_web_search,
            )
            _record_success("Gemini 3 Pro analyst", text)
        except Exception as exc:
            _record_error("Gemini analyst", exc)

    threads = [
        threading.Thread(target=_run_openai, daemon=True),
        threading.Thread(target=_run_anthropic, daemon=True),
        threading.Thread(target=_run_gemini, daemon=True),
    ]
    for thread in threads:
        thread.start()

    ready.wait()
    with lock:
        opinions_snapshot = list(opinions)
        errors_snapshot = list(errors)

    if chair_instructions is None:
        chair_instructions = [
            "You are the council chair. Synthesize the best final response.",
            "Follow the original instructions EXACTLY: output ONLY the final response in the required 4-line format plus <setting_json>.",
            "Prefer changes likely to result in a 10x improvement to the metrics attached to the user's stated goals.",
            "Avoid tiny deltas (e.g., steps 20→30) unless they unlock a major improvement.",
            "Honor the user goals.",
    ]
    council_sections: list[str] = [
        *chair_instructions,
        "",
        "Original task:",
        prompt,
    ]
    if opinions_snapshot:
        council_sections.append("")
        council_sections.append("Council feedback:")
        for name, text in opinions_snapshot:
            council_sections.append(f"{name}:\n{text}")
    if errors_snapshot:
        council_sections.append("")
        council_sections.append("Notes:")
        council_sections.extend(errors_snapshot)

    chair_prompt = "\n".join(council_sections).strip()
    try:
        text, _ = _call_openai(
            chair_prompt,
            image_base64=image_base64,
            image_mime=image_mime,
            enable_web_search=enable_web_search,
        )
        return text, []
    except Exception:
        # Fall back to the best available opinion.
        if opinions_snapshot:
            return opinions_snapshot[0][1], []
        raise


def _is_anthropic_rate_limit_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return (
        "rate limit" in lowered
        or "rate_limit" in lowered
        or "too many requests" in lowered
        or "429" in lowered
        or "overloaded" in lowered
    )


def _is_anthropic_image_size_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    if "image" not in lowered:
        return False
    size_hits = (
        "size" in lowered
        or "too large" in lowered
        or "exceed" in lowered
        or "maximum" in lowered
        or "max_image" in lowered
    )
    if not size_hits:
        return False
    return "5mb" in lowered or "5 mb" in lowered or "image" in lowered


def _call_analyzer(
    prompt: str,
    *,
    analyzer: str | None = None,
    enable_web_search: bool = True,
    image_base64: str | None = None,
    image_mime: str | None = None,
    fallback_prompt: str | None = None,
) -> tuple[str, list[dict[str, str]]]:
    analyzer_key = _normalize_analyzer(analyzer)
    if analyzer_key == "openai":
        return _call_openai(
            prompt,
            image_base64=image_base64,
            image_mime=image_mime,
        )
    if analyzer_key == "council":
        return _call_council(
            prompt,
            enable_web_search=enable_web_search,
            image_base64=image_base64,
            image_mime=image_mime,
            fallback_prompt=fallback_prompt,
        )
    try:
        return _call_anthropic(
            prompt,
            enable_web_search=enable_web_search,
            image_base64=image_base64,
            image_mime=image_mime,
        )
    except Exception as exc:
        if _is_anthropic_rate_limit_error(exc) or _is_anthropic_image_size_error(exc):
            openai_prompt = fallback_prompt or prompt
            return _call_openai(
                openai_prompt,
                image_base64=image_base64,
                image_mime=image_mime,
            )
        raise


def _load_receipt_payload(receipt_path: Path) -> dict:
    try:
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to read {receipt_path} ({exc})") from exc


def _guess_mime_from_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png"}:
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix in {".webp"}:
        return "image/webp"
    if suffix in {".gif"}:
        return "image/gif"
    return "image/png"


def _detect_mime(data: bytes, fallback: str) -> str:
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return fallback


def _load_image_for_analyzer(image_path: Path) -> tuple[str, str]:
    data = image_path.read_bytes()
    mime = _detect_mime(data, _guess_mime_from_suffix(image_path))
    encoded = base64.b64encode(data).decode("ascii")
    return encoded, mime


def _build_receipt_analysis_prompt(
    *,
    receipt: dict,
    explicit_fields: dict[str, object],
    target_prompt: str,
    allowed_settings: list[str],
    user_goals: list[str] | None,
    user_notes: str | None,
    history_text: str | None = None,
    model_options_line: str | None = None,
    enable_web_search: bool = True,
    current_round: int = 1,
    max_rounds: int = MAX_ROUNDS,
) -> str:
    top_level_candidates = ("size", "n", "seed", "output_format", "background", "model")
    top_level_allowed = [key for key in top_level_candidates if key in allowed_settings]
    top_level_hint = ", ".join(top_level_allowed) if top_level_allowed else "size, n, seed, output_format, model"
    explicit_lines = "\n".join(f"- {key}: {value}" for key, value in explicit_fields.items())
    allowed_line = ", ".join(allowed_settings) if allowed_settings else "(none)"
    goals_line = ", ".join(user_goals) if user_goals else "(not specified)"
    notes_line = user_notes.strip() if user_notes else ""
    history_block = ""
    if history_text:
        history_block = f"Session history (previous rounds):\n{history_text}\n\n"
    model_block = ""
    if model_options_line:
        model_block = f"{model_options_line}\n"
    summary = {
        "request": receipt.get("request"),
        "resolved": receipt.get("resolved"),
        "provider_request": receipt.get("provider_request"),
        "warnings": receipt.get("warnings"),
        "result_metadata": receipt.get("result_metadata"),
    }
    summary_json = json.dumps(summary, indent=2, ensure_ascii=True)
    web_search_text = ""
    if enable_web_search:
        web_search_text = (
            "Use the web_search tool to find model/provider settings and pricing that best achieve the user's goals. "
            "Base recommendations and cost estimate on documented settings for the selected provider/model. "
        )
    provider_hint = explicit_fields.get("provider")
    provider_key, _ = _normalize_provider_and_model(str(provider_hint), None) if provider_hint else ("", None)
    call_path_text = ""
    if provider_key == "openai":
        call_path_text = (
            "For OpenAI gpt-image models, you may recommend use_responses=true/false to switch between "
            "the direct gpt-image endpoint and the Responses API. "
        )
    remaining_rounds = max(0, max_rounds - current_round)
    return (
        "Simulate many runs of this prompt given the image attached with bleeding-edge vision analysis. "
        "Respond with EXACTLY four lines labeled ADH, UNSET, COST, REC, then a <setting_json> block. "
        "No tables, no markdown, no bullet points. "
        "ADH: brief prompt adherence summary (1 sentence), comparing the image to the prompt as written.\n"
        "UNSET: list 2–4 most important unset params, given the user's selected goals.\n"
        "COST: estimated USD cost per 1K images for this provider/model (short).\n"
        "REC: 1–3 unconventional recommendations aligned to user goals. "
        "If speed is a priority, you may suggest unconventional levers (e.g., output format/filetype, "
        "compression, or size tradeoffs) in addition to standard settings. "
        "Prefer changes likely to result in a 10x improvement to the metrics attached to the user's stated goals "
        "(avoid tiny deltas like steps 20→30); think big and unconventional when it helps meet the goals.\n"
        f"You have at most {max_rounds} total rounds. This is round {current_round}; "
        f"{remaining_rounds} round(s) remain after this. "
        "Plan a coherent testing path and avoid flip-flopping settings (e.g., jpeg→png→jpeg) unless there's a strong reason.\n\n"
        f"Target prompt:\n{target_prompt}\n\n"
        f"Explicitly set in the flow:\n{explicit_lines}\n\n"
        f"User goals (multi-select): {goals_line}\n"
        "If the user selected 'maximize quality of render', treat this as an optimization problem: "
        "prioritize maximizing prompt adherence and LLM-rated image quality, even if it increases cost/time. "
        "However, if the user also selected 'minimize cost of render' or 'minimize time to render', "
        "those goals take precedence and quality becomes secondary.\n"
        "If the user selected 'maximize LLM retrieval score', prioritize legibility, captionability, "
        "and retrieval-focused clarity even if aesthetic quality softens, unless cost/time goals are selected.\n"
        f"User notes: {notes_line or '(none)'}\n\n"
        f"{web_search_text}"
        f"{call_path_text}"
        f"{history_block}"
        f"{model_block}"
        f"Allowed settings for recommendations (API settings for this model): {allowed_line}\n"
        "Do NOT recommend or list settings outside the allowed list.\n"
        "Model changes must stay within the same provider and use the listed model options.\n"
        "Use setting_target='provider_options' for provider-specific options, "
        f"and setting_target='request' for top-level settings like {top_level_hint}.\n\n"
        "At the end, output a JSON array (max 3 items) wrapped in <setting_json>...</setting_json> "
        "with objects that include keys: setting_name, setting_value, setting_target, rationale. "
        "If no safe recommendation, output an empty array.\n\n"
        "Receipt summary JSON:\n"
        f"{summary_json}"
    )


def _build_recommendation_only_prompt(
    *,
    summary_json: str,
    allowed_settings: list[str],
    provider: str,
    model: str | None,
    user_goals: list[str] | None,
    user_notes: str | None,
    history_text: str | None = None,
    model_options_line: str | None = None,
    enable_web_search: bool = True,
    current_round: int = 1,
    max_rounds: int = MAX_ROUNDS,
) -> str:
    top_level_candidates = ("size", "n", "seed", "output_format", "background", "model")
    top_level_allowed = [key for key in top_level_candidates if key in allowed_settings]
    top_level_hint = ", ".join(top_level_allowed) if top_level_allowed else "size, n, seed, output_format, model"
    allowed_line = ", ".join(allowed_settings) if allowed_settings else "(none)"
    model_label = model or "(default)"
    goals_line = ", ".join(user_goals) if user_goals else "(not specified)"
    notes_line = user_notes.strip() if user_notes else ""
    history_block = ""
    if history_text:
        history_block = f"Session history (previous rounds):\n{history_text}\n"
    model_block = ""
    if model_options_line:
        model_block = f"{model_options_line}\n"
    web_search_text = ""
    if enable_web_search:
        web_search_text = (
            "Use the web_search tool to find model/provider settings that best achieve the user's goals. "
            "Base recommendations on documented settings for the selected provider/model.\n"
        )
    provider_key, _ = _normalize_provider_and_model(str(provider), model)
    call_path_text = ""
    if provider_key == "openai":
        call_path_text = (
            "For OpenAI gpt-image models, you may recommend use_responses=true/false to switch between "
            "the direct gpt-image endpoint and the Responses API.\n"
        )
    remaining_rounds = max(0, max_rounds - current_round)
    return (
        "Simulate many runs of this prompt given the image attached with bleeding-edge vision analysis. "
        "Based on the receipt summary, recommend 1–3 API setting changes to improve prompt adherence. "
        "Only choose from the allowed settings list. "
        "Output ONLY a JSON array wrapped in <setting_json>...</setting_json> with objects that include keys: "
        "setting_name, setting_value, setting_target, rationale. If no safe recommendation, output an empty array.\n\n"
        f"Provider: {provider}\nModel: {model_label}\n"
        f"User goals (multi-select): {goals_line}\n"
        "If the user selected 'maximize quality of render', treat this as an optimization problem: "
        "prioritize maximizing prompt adherence and LLM-rated image quality, even if it increases cost/time. "
        "However, if the user also selected 'minimize cost of render' or 'minimize time to render', "
        "those goals take precedence and quality becomes secondary.\n"
        "If the user selected 'maximize LLM retrieval score', prioritize legibility, captionability, "
        "and retrieval-focused clarity even if aesthetic quality softens, unless cost/time goals are selected.\n"
        f"User notes: {notes_line or '(none)'}\n"
        f"{web_search_text}"
        f"{call_path_text}"
        f"{history_block}"
        f"{model_block}"
        "If speed is a priority, consider unconventional levers (e.g., output format/filetype, "
        "compression, or size tradeoffs) but still choose from the allowed list. "
        "Prefer changes likely to result in a 10x improvement to the metrics attached to the user's stated goals "
        "(avoid tiny deltas like steps 20→30); think big and unconventional when it helps meet the goals.\n"
        "If cost/time goals are selected, treat them as primary and only pursue quality improvements "
        "that do not materially worsen cost/time.\n"
        f"You have at most {max_rounds} total rounds. This is round {current_round}; "
        f"{remaining_rounds} round(s) remain after this. "
        "Plan a coherent testing path and avoid flip-flopping settings unless there's a strong reason.\n"
        "Model changes must stay within the same provider and use the listed model options.\n"
        "Use setting_target='provider_options' for provider-specific options, "
        f"and setting_target='request' for top-level settings like {top_level_hint}.\n"
        f"Allowed settings: {allowed_line}\n"
        "Do NOT recommend or list settings outside the allowed list.\n\n"
        f"Receipt summary JSON:\n{summary_json}"
    )


def _analyze_receipt_payload(
    *,
    receipt_path: Path,
    provider: str,
    model: str | None,
    size: str,
    n: int,
    out_dir: str | Path,
    analyzer: str | None = None,
    user_goals: list[str] | None = None,
    user_notes: str | None = None,
    history_text: str | None = None,
    history_rounds: int = 0,
) -> tuple[str, list[dict[str, str]], list[dict] | None, str | None, str | None]:
    analyzer_key = _normalize_analyzer(analyzer)
    receipt = _load_receipt_payload(receipt_path)
    image_path = None
    artifacts = receipt.get("artifacts")
    if isinstance(artifacts, dict):
        raw_path = artifacts.get("image_path")
        if isinstance(raw_path, str):
            image_path = Path(raw_path)
    if image_path is not None and not image_path.exists():
        image_path = None
    explicit_fields = {
        "provider": provider,
        "model": model or "(default)",
        "size": size,
        "n": n,
        "prompt": str(receipt.get("request", {}).get("prompt", "")),
        "out_dir": str(out_dir),
    }
    provider_key, _ = _normalize_provider_and_model(provider, model)
    if provider_key == "openai":
        use_responses = None
        resolved = receipt.get("resolved") if isinstance(receipt.get("resolved"), dict) else None
        request = receipt.get("request") if isinstance(receipt.get("request"), dict) else None
        for source in (
            resolved.get("provider_params") if isinstance(resolved, dict) else None,
            request.get("provider_options") if isinstance(request, dict) else None,
        ):
            if isinstance(source, dict) and "use_responses" in source:
                use_responses = bool(source.get("use_responses"))
                break
        if use_responses is None:
            use_responses = False
        explicit_fields["use_responses"] = use_responses
    target_prompt = FIXED_PROMPT
    allowed_settings = _allowed_settings_for_receipt(receipt, provider)
    allowed_models = _allowed_models_for_provider(provider, model)
    model_options_line = _model_options_line(provider, size, model)
    summary = {
        "request": receipt.get("request"),
        "resolved": receipt.get("resolved"),
        "provider_request": receipt.get("provider_request"),
        "warnings": receipt.get("warnings"),
        "result_metadata": receipt.get("result_metadata"),
    }
    summary_json = json.dumps(summary, indent=2, ensure_ascii=True)
    allow_web_search = analyzer_key in {"anthropic", "council"}
    current_round = max(1, history_rounds + 1)
    prompt = _build_receipt_analysis_prompt(
        receipt=receipt,
        explicit_fields=explicit_fields,
        target_prompt=target_prompt,
        allowed_settings=allowed_settings,
        user_goals=user_goals,
        user_notes=user_notes,
        history_text=history_text,
        model_options_line=model_options_line,
        enable_web_search=allow_web_search,
        current_round=current_round,
        max_rounds=MAX_ROUNDS,
    )
    fallback_prompt = None
    if allow_web_search:
        fallback_prompt = _build_receipt_analysis_prompt(
            receipt=receipt,
            explicit_fields=explicit_fields,
            target_prompt=target_prompt,
            allowed_settings=allowed_settings,
            user_goals=user_goals,
            user_notes=user_notes,
            history_text=history_text,
            model_options_line=model_options_line,
            enable_web_search=False,
            current_round=current_round,
            max_rounds=MAX_ROUNDS,
        )
    image_base64 = None
    image_mime = None
    if image_path is not None:
        try:
            image_base64, image_mime = _load_image_for_analyzer(image_path)
        except Exception as exc:
            raise RuntimeError(f"failed to read image for analysis ({exc})") from exc
    analysis_text, citations = _call_analyzer(
        prompt,
        analyzer=analyzer_key,
        enable_web_search=allow_web_search,
        image_base64=image_base64,
        image_mime=image_mime,
        fallback_prompt=fallback_prompt,
    )
    if len(analysis_text) > ANALYSIS_MAX_CHARS:
        analysis_text = _compress_analysis_to_limit(
            analysis_text,
            ANALYSIS_MAX_CHARS,
            analyzer=analyzer_key,
        )
    analysis_text = _normalize_rec_line(analysis_text)
    cleaned_text, recommendation_payload = _extract_setting_json(analysis_text)
    recs_payload, stop_reason, stop_recommended = _parse_recommendation_payload(recommendation_payload)
    stop_reason = None
    stop_recommended = False
    cleaned_text = _normalize_rec_line(cleaned_text)
    rec_text = _rec_line_text(cleaned_text)
    recommendations = _normalize_recommendations(recs_payload)
    resolved = receipt.get("resolved") if isinstance(receipt, dict) else None
    request = receipt.get("request") if isinstance(receipt, dict) else None
    recommendations = _filter_noop_recommendations(
        recommendations,
        resolved,
        request,
        provider=provider,
        model=model,
    )
    recommendations = _filter_unsupported_top_level_recommendations(recommendations, provider)
    recommendations = _filter_locked_size_recommendations(recommendations, size)
    recommendations = _filter_model_recommendations(recommendations, allowed_models)
    if rec_text and "no change" in rec_text:
        recommendations = []
    cleaned_text, cost_line = _extract_cost_line(cleaned_text)
    local_cost_line = _estimate_cost_only(provider=provider, model=model, size=size, n=n)
    cost_line = local_cost_line
    recommendations = _sanitize_recommendation_rationales(
        recommendations,
        cost_line,
        provider=provider,
        size=size,
        n=n,
    )
    cleaned_text = _rewrite_rec_line(cleaned_text, recommendations)
    if not recommendations:
        fallback_prompt = _build_recommendation_only_prompt(
            summary_json=summary_json,
            allowed_settings=allowed_settings,
            provider=provider,
            model=model,
            user_goals=user_goals,
            user_notes=user_notes,
            history_text=history_text,
            model_options_line=model_options_line,
            enable_web_search=allow_web_search,
            current_round=current_round,
            max_rounds=MAX_ROUNDS,
        )
        fallback_prompt_no_search = None
        if allow_web_search:
            fallback_prompt_no_search = _build_recommendation_only_prompt(
                summary_json=summary_json,
                allowed_settings=allowed_settings,
                provider=provider,
                model=model,
                user_goals=user_goals,
                user_notes=user_notes,
                history_text=history_text,
                model_options_line=model_options_line,
                enable_web_search=False,
                current_round=current_round,
                max_rounds=MAX_ROUNDS,
            )
        fallback_text, _ = _call_analyzer(
            fallback_prompt,
            analyzer=analyzer_key,
            enable_web_search=allow_web_search,
            fallback_prompt=fallback_prompt_no_search,
        )
        _, fallback_payload = _extract_setting_json(fallback_text)
        recs_payload, fallback_stop_reason, fallback_stop = _parse_recommendation_payload(fallback_payload)
        fallback_stop_reason = None
        fallback_stop = False
        recommendations = _normalize_recommendations(recs_payload)
        recommendations = _filter_noop_recommendations(
            recommendations,
            resolved,
            request,
            provider=provider,
            model=model,
        )
        recommendations = _filter_locked_size_recommendations(recommendations, size)
        recommendations = _filter_model_recommendations(recommendations, allowed_models)
        recommendations = _sanitize_recommendation_rationales(
            recommendations,
            cost_line,
            provider=provider,
            size=size,
            n=n,
        )
        cleaned_text = _rewrite_rec_line(cleaned_text, recommendations)
    return cleaned_text, citations, recommendations or None, cost_line, stop_reason


def _analyze_receipt(
    receipt_path: Path,
    *,
    provider: str,
    model: str | None,
    size: str,
    n: int,
    out_dir: str | Path,
    analyzer: str | None = None,
) -> None:
    analyzer_key = _normalize_analyzer(analyzer)
    print(f"\nAnalyzing receipt with {_analyzer_display_name(analyzer_key)}...")
    if analyzer_key == "council":
        print("Note: Council analysis can take a few minutes.")
    try:
        analysis, citations, recommendation, cost_line, stop_reason = _analyze_receipt_payload(
            receipt_path=receipt_path,
            provider=provider,
            model=model,
            size=size,
            n=n,
            out_dir=str(out_dir),
            analyzer=analyzer_key,
        )
    except Exception as exc:
        print(f"Receipt analysis failed: {exc}")
        return
    try:
        receipt_payload = _load_receipt_payload(receipt_path)
        detail_lines = _build_receipt_detail_lines(
            receipt_payload,
            recommendation,
            max_width=max(40, _term_width() - 2),
            stop_reason=None,
        )
        if not recommendation:
            summary_settings = _settings_from_receipt(
                receipt_payload,
                fallback_provider=provider,
                fallback_model=model,
                fallback_size=size,
                fallback_n=n,
            )
            detail_lines.append("")
            detail_lines.extend(
                [
                    line
                    for line in _build_final_recommendation_lines(
                        summary_settings,
                        None,
                        user_goals=None,
                        max_width=max(40, _term_width() - 2),
                        return_tags=False,
                        last_elapsed=None,
                        cost_line=cost_line,
                        baseline_settings=None,
                        baseline_elapsed=None,
                        baseline_cost_line=None,
                        force_quality_metrics=False,
                        retrieval_baseline=None,
                        retrieval_current=None,
                    )
                    if isinstance(line, str)
                ]
            )
    except Exception:
        detail_lines = []
    print("\nMODEL ANALYSIS (LLM):")
    if analysis:
        print(_display_analysis_text(analysis))
    else:
        print("Receipt analysis returned no content.")
    if detail_lines:
        print("\nReceipt settings & recommendations:")
        for line in detail_lines:
            if isinstance(line, str):
                print(line)


def _open_path(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif hasattr(os, "startfile"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass


_VIEWER_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Param Forge Viewer</title>
    <style>
      :root {
        --ink: #121212;
        --muted: #6b645f;
        --paper: #f6f1e8;
        --paper-2: #fbfaf7;
        --card: #ffffff;
        --accent: #ff6b3d;
        --accent-2: #2aa7a1;
        --accent-3: #f2c14e;
        --shadow: rgba(16, 16, 16, 0.12);
        --border: rgba(18, 18, 18, 0.08);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Space Grotesk", "Avenir Next", "Futura", sans-serif;
        color: var(--ink);
        background: radial-gradient(circle at top, #f9ede2 0%, var(--paper) 45%, #f1f7f4 100%);
      }
      .page {
        max-width: 1400px;
        margin: 0 auto;
        padding: 32px 24px 64px;
      }
      header.hero {
        padding: 24px 28px;
        border-radius: 20px;
        background: linear-gradient(120deg, #fff2e8 0%, #f7fff7 50%, #f1f5ff 100%);
        box-shadow: 0 16px 40px -28px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 107, 61, 0.12);
      }
      header.hero h1 {
        margin: 0 0 8px;
        font-size: 28px;
        letter-spacing: -0.02em;
      }
      header.hero .meta {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        font-size: 14px;
        color: var(--muted);
      }
      .pill {
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(18, 18, 18, 0.06);
        border: 1px solid var(--border);
      }
      .toolbar {
        display: grid;
        grid-template-columns: minmax(220px, 1fr) minmax(240px, 1fr) minmax(220px, 1fr);
        gap: 16px;
        margin: 20px 0 12px;
      }
      .panel {
        background: var(--paper-2);
        border-radius: 16px;
        border: 1px solid var(--border);
        padding: 12px 14px;
        box-shadow: 0 10px 24px -18px var(--shadow);
      }
      .panel h3 {
        margin: 0 0 8px;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }
      .panel input[type="text"] {
        width: 100%;
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: #fff;
        font-size: 14px;
      }
      .toggle-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        font-size: 13px;
      }
      .toggle-row label {
        display: flex;
        align-items: center;
        gap: 6px;
        cursor: pointer;
      }
      .filter-list {
        max-height: 160px;
        overflow: auto;
        display: grid;
        grid-template-columns: 1fr;
        gap: 6px;
      }
      .filter-list label {
        display: flex;
        gap: 8px;
        align-items: center;
        font-size: 13px;
      }
      .compare {
        margin: 18px 0 24px;
        padding: 16px;
        background: #fff;
        border-radius: 18px;
        border: 1px solid var(--border);
        box-shadow: 0 18px 30px -26px rgba(0, 0, 0, 0.3);
      }
      .compare h2 {
        margin: 0 0 12px;
        font-size: 16px;
      }
      .compare-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
      }
      .compare-card {
        background: var(--paper-2);
        border-radius: 14px;
        border: 1px solid var(--border);
        padding: 10px;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .compare-card img {
        width: 100%;
        border-radius: 10px;
        object-fit: cover;
      }
      .grid-wrap {
        overflow: auto;
        border-radius: 18px;
        border: 1px solid var(--border);
        background: #fff;
        box-shadow: 0 18px 32px -26px rgba(0, 0, 0, 0.35);
      }
      table.grid {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        min-width: 920px;
      }
      table.grid th,
      table.grid td {
        border-bottom: 1px solid var(--border);
        vertical-align: top;
        padding: 12px;
      }
      table.grid thead th {
        position: sticky;
        top: 0;
        background: #fff;
        z-index: 5;
        text-align: left;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }
      table.grid tbody th {
        position: sticky;
        left: 0;
        background: #fff;
        z-index: 4;
        width: 280px;
        max-width: 280px;
      }
      .prompt-text {
        font-size: 13px;
        line-height: 1.4;
      }
      .prompt-meta {
        margin-top: 8px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }
      .card {
        background: var(--paper-2);
        border-radius: 14px;
        border: 1px solid var(--border);
        overflow: hidden;
        display: flex;
        flex-direction: column;
        gap: 8px;
        min-height: 240px;
        animation: rise 0.4s ease;
      }
      .card.selected {
        border-color: var(--accent);
        box-shadow: 0 12px 28px -18px rgba(255, 107, 61, 0.7);
      }
      .card.winner {
        border-color: var(--accent-2);
        box-shadow: 0 12px 28px -18px rgba(42, 167, 161, 0.7);
      }
      .card img {
        width: 100%;
        height: 180px;
        object-fit: cover;
        display: block;
      }
      .card-body {
        padding: 10px 12px 12px;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .badges {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }
      .badge {
        padding: 4px 8px;
        border-radius: 999px;
        background: rgba(18, 18, 18, 0.08);
        font-size: 11px;
      }
      .badge.accent { background: rgba(255, 107, 61, 0.18); }
      .badge.teal { background: rgba(42, 167, 161, 0.18); }
      .badge.yellow { background: rgba(242, 193, 78, 0.25); }
      .actions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }
      .btn {
        border: none;
        padding: 6px 10px;
        border-radius: 8px;
        background: var(--accent);
        color: #fff;
        font-size: 12px;
        cursor: pointer;
      }
      .btn.ghost {
        background: rgba(18, 18, 18, 0.08);
        color: var(--ink);
      }
      .cell-empty {
        font-size: 12px;
        color: var(--muted);
        padding: 16px;
      }
      .nav-row {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: var(--muted);
      }
      .nav-row button {
        background: transparent;
        border: none;
        cursor: pointer;
        font-size: 12px;
      }
      .toast {
        position: fixed;
        right: 20px;
        top: 20px;
        padding: 10px 14px;
        border-radius: 10px;
        background: rgba(18, 18, 18, 0.92);
        color: #fff;
        font-size: 13px;
        opacity: 0;
        transform: translateY(-10px);
        transition: all 0.2s ease;
        pointer-events: none;
        z-index: 99;
      }
      .toast.show {
        opacity: 1;
        transform: translateY(0);
      }
      @keyframes rise {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @media (max-width: 960px) {
        .toolbar { grid-template-columns: 1fr; }
        table.grid tbody th { width: 220px; max-width: 220px; }
      }
    </style>
  </head>
  <body>
    <div class="page">
      <header class="hero">
        <h1 id="run-title">Param Forge Viewer</h1>
        <div class="meta" id="run-meta"></div>
      </header>
      <div class="toolbar">
        <div class="panel">
          <h3>Search</h3>
          <input id="prompt-search" type="text" placeholder="Filter prompts..." />
        </div>
        <div class="panel">
          <h3>Focus</h3>
          <div class="toggle-row">
            <label><input id="toggle-winners" type="checkbox" /> Winners only</label>
            <label><input id="toggle-flags" type="checkbox" /> Flags only</label>
          </div>
        </div>
        <div class="panel">
          <h3>Models</h3>
          <div class="filter-list" id="column-filters"></div>
        </div>
      </div>
      <section class="compare">
        <h2>Side-by-side compare</h2>
        <div class="compare-grid" id="compare-grid">
          <div class="cell-empty">Pick up to 4 cards to compare.</div>
        </div>
      </section>
      <section class="grid-wrap">
        <table class="grid">
          <thead id="grid-head"></thead>
          <tbody id="grid-body"></tbody>
        </table>
      </section>
    </div>
    <div class="toast" id="toast"></div>
    <script>
      const PF_DATA = __DATA__;
      const state = {
        search: "",
        winnersOnly: false,
        flagsOnly: false,
        visibleColumns: new Set(),
        compare: [],
        winners: {},
        cellCursor: {}
      };
      const variantsById = new Map();
      const cellIndex = {};
      const storageKey = `pf_winners_${PF_DATA.run.id || "default"}`;

      function loadWinners() {
        try {
          const raw = localStorage.getItem(storageKey);
          if (raw) {
            state.winners = JSON.parse(raw);
          }
        } catch (err) {
          state.winners = {};
        }
      }

      function saveWinners() {
        try {
          localStorage.setItem(storageKey, JSON.stringify(state.winners));
        } catch (err) {
          /* ignore */
        }
      }

      function toast(message) {
        const node = document.getElementById("toast");
        node.textContent = message;
        node.classList.add("show");
        setTimeout(() => node.classList.remove("show"), 1400);
      }

      function copyText(text) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(() => toast("Copied."));
          return;
        }
        const textarea = document.createElement("textarea");
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
        toast("Copied.");
      }

      function buildIndex() {
        PF_DATA.variants.forEach((variant) => {
          variantsById.set(variant.id, variant);
          if (!cellIndex[variant.prompt_id]) {
            cellIndex[variant.prompt_id] = {};
          }
          if (!cellIndex[variant.prompt_id][variant.column_key]) {
            cellIndex[variant.prompt_id][variant.column_key] = [];
          }
          cellIndex[variant.prompt_id][variant.column_key].push(variant);
        });
      }

      function formatSeconds(value) {
        if (value === null || value === undefined) return "N/A";
        return `${value.toFixed(1)}s`;
      }

      function formatCost(cost) {
        if (!cost) return "N/A";
        return cost;
      }

      function renderMeta() {
        const meta = document.getElementById("run-meta");
        meta.innerHTML = "";
        const totalImages = PF_DATA.variants.length;
        const totalCost = PF_DATA.variants.reduce((sum, v) => sum + (v.cost_usd || 0), 0);
        const totalLatency = PF_DATA.variants.reduce((sum, v) => sum + (v.render_seconds_per_image || 0), 0);
        const avgLatency = totalImages ? totalLatency / totalImages : null;
        const items = [
          `Run: ${PF_DATA.run.id || "local"}`,
          `Prompts: ${PF_DATA.prompts.length}`,
          `Models: ${PF_DATA.columns.length}`,
          `Images: ${totalImages}`,
          `Est. cost: $${totalCost.toFixed(2)}`,
          `Avg latency: ${avgLatency ? avgLatency.toFixed(1) + "s" : "N/A"}`
        ];
        items.forEach((item) => {
          const pill = document.createElement("div");
          pill.className = "pill";
          pill.textContent = item;
          meta.appendChild(pill);
        });
        const title = document.getElementById("run-title");
        title.textContent = PF_DATA.run.title || "Param Forge Viewer";
      }

      function renderColumnFilters() {
        const container = document.getElementById("column-filters");
        container.innerHTML = "";
        PF_DATA.columns.forEach((column) => {
          const label = document.createElement("label");
          const checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.checked = state.visibleColumns.has(column.key);
          checkbox.addEventListener("change", () => {
            if (checkbox.checked) {
              state.visibleColumns.add(column.key);
            } else {
              state.visibleColumns.delete(column.key);
            }
            renderGrid();
          });
          label.appendChild(checkbox);
          const text = document.createElement("span");
          text.textContent = column.label;
          label.appendChild(text);
          container.appendChild(label);
        });
      }

      function buildBadge(label, className) {
        const badge = document.createElement("span");
        badge.className = `badge ${className || ""}`.trim();
        badge.textContent = label;
        return badge;
      }

      function buildCard(variant, prompt) {
        const card = document.createElement("div");
        card.className = "card";
        if (state.compare.includes(variant.id)) {
          card.classList.add("selected");
        }
        if (state.winners[prompt.id] === variant.id) {
          card.classList.add("winner");
        }
        const image = document.createElement("img");
        image.src = variant.image_src;
        image.alt = variant.prompt || "generated image";
        image.loading = "lazy";
        image.addEventListener("click", () => window.open(variant.image_src, "_blank"));
        card.appendChild(image);

        const body = document.createElement("div");
        body.className = "card-body";

        const navRow = document.createElement("div");
        navRow.className = "nav-row";
        if (variant.variant_total && variant.variant_total > 1) {
          const prev = document.createElement("button");
          prev.textContent = "prev";
          prev.addEventListener("click", () => shiftVariant(prompt.id, variant.column_key, -1));
          const next = document.createElement("button");
          next.textContent = "next";
          next.addEventListener("click", () => shiftVariant(prompt.id, variant.column_key, 1));
          const index = document.createElement("span");
          index.textContent = `${variant.variant_index + 1}/${variant.variant_total}`;
          navRow.appendChild(prev);
          navRow.appendChild(index);
          navRow.appendChild(next);
          body.appendChild(navRow);
        }

        const badges = document.createElement("div");
        badges.className = "badges";
        badges.appendChild(buildBadge(`cost ${formatCost(variant.cost_per_1k)}`, "accent"));
        badges.appendChild(buildBadge(`render ${formatSeconds(variant.render_seconds_per_image)}`, "teal"));
        if (variant.adherence !== null && variant.adherence !== undefined) {
          badges.appendChild(buildBadge(`adh ${variant.adherence}`, "yellow"));
        }
        if (variant.quality !== null && variant.quality !== undefined) {
          badges.appendChild(buildBadge(`qual ${variant.quality}`, "yellow"));
        }
        if (variant.retrieval_score !== null && variant.retrieval_score !== undefined) {
          badges.appendChild(buildBadge(`retr ${variant.retrieval_score}`, "yellow"));
        }
        if (variant.flags && variant.flags.length) {
          badges.appendChild(buildBadge("flag", "accent"));
        }
        body.appendChild(badges);

        const actions = document.createElement("div");
        actions.className = "actions";
        const copyBtn = document.createElement("button");
        copyBtn.className = "btn";
        copyBtn.textContent = "Copy snippet";
        copyBtn.addEventListener("click", () => copyText(variant.snippet));
        const compareBtn = document.createElement("button");
        compareBtn.className = "btn ghost";
        compareBtn.textContent = state.compare.includes(variant.id) ? "Remove" : "Compare";
        compareBtn.addEventListener("click", () => toggleCompare(variant.id));
        const winnerBtn = document.createElement("button");
        winnerBtn.className = "btn ghost";
        winnerBtn.textContent = state.winners[prompt.id] === variant.id ? "Winner" : "Pick winner";
        winnerBtn.addEventListener("click", () => pickWinner(prompt.id, variant.id));
        const receiptBtn = document.createElement("button");
        receiptBtn.className = "btn ghost";
        receiptBtn.textContent = "Receipt";
        receiptBtn.addEventListener("click", () => window.open(variant.receipt_src, "_blank"));
        actions.appendChild(copyBtn);
        actions.appendChild(compareBtn);
        actions.appendChild(winnerBtn);
        actions.appendChild(receiptBtn);
        body.appendChild(actions);

        card.appendChild(body);
        return card;
      }

      function renderGrid() {
        const head = document.getElementById("grid-head");
        const body = document.getElementById("grid-body");
        head.innerHTML = "";
        body.innerHTML = "";
        const visibleColumns = PF_DATA.columns.filter((col) => state.visibleColumns.has(col.key));

        const headRow = document.createElement("tr");
        const corner = document.createElement("th");
        corner.textContent = "Prompt";
        headRow.appendChild(corner);
        visibleColumns.forEach((col) => {
          const th = document.createElement("th");
          th.textContent = col.label;
          headRow.appendChild(th);
        });
        head.appendChild(headRow);

        PF_DATA.prompts.forEach((prompt) => {
          if (state.search && !prompt.text.toLowerCase().includes(state.search)) {
            return;
          }
          const row = document.createElement("tr");
          const promptCell = document.createElement("th");
          const promptText = document.createElement("div");
          promptText.className = "prompt-text";
          promptText.textContent = prompt.text;
          promptCell.appendChild(promptText);
          const promptMeta = document.createElement("div");
          promptMeta.className = "prompt-meta";
          if (state.winners[prompt.id]) {
            promptMeta.appendChild(buildBadge("winner picked", "teal"));
          }
          promptCell.appendChild(promptMeta);
          row.appendChild(promptCell);

          visibleColumns.forEach((column) => {
            const cell = document.createElement("td");
            const variants = (cellIndex[prompt.id] || {})[column.key] || [];
            if (!variants.length) {
              const empty = document.createElement("div");
              empty.className = "cell-empty";
              empty.textContent = "—";
              cell.appendChild(empty);
              row.appendChild(cell);
              return;
            }
            let filtered = variants;
            if (state.flagsOnly) {
              filtered = variants.filter((variant) => variant.flags && variant.flags.length);
            }
            if (state.winnersOnly) {
              filtered = variants.filter((variant) => state.winners[prompt.id] === variant.id);
            }
            if (!filtered.length) {
              const empty = document.createElement("div");
              empty.className = "cell-empty";
              empty.textContent = "filtered";
              cell.appendChild(empty);
              row.appendChild(cell);
              return;
            }
            const key = `${prompt.id}::${column.key}`;
            let idx = state.cellCursor[key] || 0;
            if (idx >= filtered.length) idx = 0;
            state.cellCursor[key] = idx;
            const variant = filtered[idx];
            variant.variant_index = idx;
            variant.variant_total = filtered.length;
            cell.appendChild(buildCard(variant, prompt));
            row.appendChild(cell);
          });
          body.appendChild(row);
        });
      }

      function renderCompare() {
        const grid = document.getElementById("compare-grid");
        grid.innerHTML = "";
        if (!state.compare.length) {
          const empty = document.createElement("div");
          empty.className = "cell-empty";
          empty.textContent = "Pick up to 4 cards to compare.";
          grid.appendChild(empty);
          return;
        }
        state.compare.forEach((id) => {
          const variant = variantsById.get(id);
          if (!variant) return;
          const card = document.createElement("div");
          card.className = "compare-card";
          const img = document.createElement("img");
          img.src = variant.image_src;
          img.alt = variant.prompt || "comparison image";
          card.appendChild(img);
          const badges = document.createElement("div");
          badges.className = "badges";
          badges.appendChild(buildBadge(variant.column_label, "accent"));
          badges.appendChild(buildBadge(`cost ${formatCost(variant.cost_per_1k)}`, "accent"));
          badges.appendChild(buildBadge(`render ${formatSeconds(variant.render_seconds_per_image)}`, "teal"));
          card.appendChild(badges);
          const actions = document.createElement("div");
          actions.className = "actions";
          const copyBtn = document.createElement("button");
          copyBtn.className = "btn";
          copyBtn.textContent = "Copy snippet";
          copyBtn.addEventListener("click", () => copyText(variant.snippet));
          const removeBtn = document.createElement("button");
          removeBtn.className = "btn ghost";
          removeBtn.textContent = "Remove";
          removeBtn.addEventListener("click", () => toggleCompare(id));
          actions.appendChild(copyBtn);
          actions.appendChild(removeBtn);
          card.appendChild(actions);
          grid.appendChild(card);
        });
      }

      function toggleCompare(id) {
        const idx = state.compare.indexOf(id);
        if (idx >= 0) {
          state.compare.splice(idx, 1);
        } else {
          if (state.compare.length >= 4) {
            toast("Compare tray is full.");
            return;
          }
          state.compare.push(id);
        }
        renderCompare();
        renderGrid();
      }

      function pickWinner(promptId, variantId) {
        if (state.winners[promptId] === variantId) {
          delete state.winners[promptId];
          toast("Winner cleared.");
        } else {
          state.winners[promptId] = variantId;
          toast("Winner saved.");
        }
        saveWinners();
        renderGrid();
      }

      function shiftVariant(promptId, columnKey, direction) {
        const key = `${promptId}::${columnKey}`;
        const list = (cellIndex[promptId] || {})[columnKey] || [];
        if (!list.length) return;
        const current = state.cellCursor[key] || 0;
        const next = (current + direction + list.length) % list.length;
        state.cellCursor[key] = next;
        renderGrid();
      }

      function attachHandlers() {
        const search = document.getElementById("prompt-search");
        search.addEventListener("input", (event) => {
          state.search = event.target.value.toLowerCase();
          renderGrid();
        });
        document.getElementById("toggle-winners").addEventListener("change", (event) => {
          state.winnersOnly = event.target.checked;
          renderGrid();
        });
        document.getElementById("toggle-flags").addEventListener("change", (event) => {
          state.flagsOnly = event.target.checked;
          renderGrid();
        });
      }

      function init() {
        PF_DATA.columns.forEach((column) => state.visibleColumns.add(column.key));
        loadWinners();
        buildIndex();
        renderMeta();
        renderColumnFilters();
        renderGrid();
        renderCompare();
        attachHandlers();
      }

      init();
    </script>
  </body>
</html>
"""


def _view_parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _view_seconds_between(start: str | None, end: str | None) -> float | None:
    start_dt = _view_parse_iso(start)
    end_dt = _view_parse_iso(end)
    if not start_dt or not end_dt:
        return None
    try:
        return max(0.0, (end_dt - start_dt).total_seconds())
    except Exception:
        return None


def _view_rel_path(path: Path, base: Path) -> str:
    try:
        rel = os.path.relpath(path, base)
        return Path(rel).as_posix()
    except Exception:
        return Path(path).as_posix()


def _view_build_snippet(receipt: dict) -> str:
    request = receipt.get("request") if isinstance(receipt.get("request"), dict) else {}
    resolved = receipt.get("resolved") if isinstance(receipt.get("resolved"), dict) else {}
    prompt = str(request.get("prompt") or resolved.get("prompt") or "")
    provider = resolved.get("provider") or request.get("provider") or "openai"
    model = resolved.get("model") or request.get("model")
    size = resolved.get("size") or request.get("size") or "1024x1024"
    n = resolved.get("n") or request.get("n") or 1
    seed = resolved.get("seed") or request.get("seed")
    output_format = resolved.get("output_format") or request.get("output_format")
    background = resolved.get("background") or request.get("background")
    mode = request.get("mode") or "generate"
    inputs = request.get("inputs") if isinstance(request.get("inputs"), dict) else {}
    provider_options = {}
    if isinstance(request.get("provider_options"), dict) and request.get("provider_options"):
        provider_options = dict(request.get("provider_options") or {})
    elif isinstance(resolved.get("provider_params"), dict):
        provider_options = dict(resolved.get("provider_params") or {})
    for key in ("size", "output_format", "background", "n", "seed", "model", "prompt"):
        provider_options.pop(key, None)
    provider_options = {k: provider_options[k] for k in sorted(provider_options) if provider_options[k] is not None}

    args: list[str] = [f"prompt={prompt!r}", f"provider={str(provider)!r}"]
    if model:
        args.append(f"model={str(model)!r}")
    if size and str(size) != "1024x1024":
        args.append(f"size={str(size)!r}")
    if n and int(n) != 1:
        args.append(f"n={int(n)}")
    if seed is not None:
        args.append(f"seed={int(seed)}")
    if output_format:
        args.append(f"output_format={str(output_format)!r}")
    if background:
        args.append(f"background={str(background)!r}")
    if provider_options:
        args.append(f"provider_options={provider_options!r}")
    call_name = "generate"
    if mode == "edit":
        call_name = "edit"
        init_image = inputs.get("init_image")
        if init_image:
            args.append(f"init_image={str(init_image)!r}")
        mask = inputs.get("mask")
        if mask:
            args.append(f"mask={str(mask)!r}")

    lines = [
        f"from forge_image_api import {call_name}",
        "",
        f"result = {call_name}(",
    ]
    lines.extend([f"    {item}," for item in args])
    lines.append(")")
    return "\n".join(lines)


def _view_load_manifest(run_dir: Path) -> dict | None:
    manifest_path = run_dir / "run.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _view_collect_receipts(run_dir: Path, manifest: dict | None) -> tuple[list[Path], dict[Path, dict]]:
    receipts: list[Path] = []
    job_lookup: dict[Path, dict] = {}
    if manifest and isinstance(manifest.get("jobs"), list):
        for job in manifest.get("jobs", []):
            artifacts = job.get("artifacts") if isinstance(job, dict) else None
            if not isinstance(artifacts, dict):
                continue
            receipt_paths = artifacts.get("receipt_paths")
            if not isinstance(receipt_paths, list):
                continue
            for item in receipt_paths:
                if not isinstance(item, str):
                    continue
                path = (run_dir / item).resolve()
                receipts.append(path)
                job_lookup[path] = job
    if not receipts:
        receipts = sorted(run_dir.glob("receipt-*.json"))
    return receipts, job_lookup


def _view_build_index(run_path: Path, view_dir: Path) -> dict:
    run_dir = run_path if run_path.is_dir() else run_path.parent
    manifest = _view_load_manifest(run_dir)
    receipts, job_lookup = _view_collect_receipts(run_dir, manifest)

    prompts: list[dict[str, str]] = []
    prompt_by_id: dict[str, str] = {}
    prompt_id_by_text: dict[str, str] = {}

    if manifest and isinstance(manifest.get("jobs"), list):
        for job in manifest.get("jobs", []):
            if not isinstance(job, dict):
                continue
            prompt_id = str(job.get("prompt_id") or "")
            prompt_text = str(job.get("prompt") or "")
            if not prompt_id:
                continue
            if prompt_id in prompt_by_id:
                continue
            prompt_by_id[prompt_id] = prompt_text
            if prompt_text:
                prompt_id_by_text.setdefault(prompt_text, prompt_id)
            prompts.append({"id": prompt_id, "text": prompt_text})

    def _fallback_prompt_id(text: str) -> str:
        existing = prompt_id_by_text.get(text)
        if existing:
            return existing
        next_id = f"p-{len(prompts) + 1:04d}"
        prompt_id_by_text[text] = next_id
        prompt_by_id[next_id] = text
        prompts.append({"id": next_id, "text": text})
        return next_id

    columns: list[dict[str, str]] = []
    column_keys: dict[str, dict[str, str]] = {}

    variants: list[dict[str, object]] = []
    for receipt_path in receipts:
        try:
            receipt = _load_receipt_payload(receipt_path)
        except Exception:
            continue
        if not isinstance(receipt, dict):
            continue
        request = receipt.get("request") if isinstance(receipt.get("request"), dict) else {}
        resolved = receipt.get("resolved") if isinstance(receipt.get("resolved"), dict) else {}
        prompt_text = str(request.get("prompt") or resolved.get("prompt") or "")
        prompt_id = None
        job = job_lookup.get(receipt_path)
        if job and isinstance(job, dict):
            job_prompt = job.get("prompt_id")
            if job_prompt:
                prompt_id = str(job_prompt)
        if not prompt_id:
            prompt_id = _fallback_prompt_id(prompt_text)
        provider = resolved.get("provider") or request.get("provider") or "unknown"
        model = resolved.get("model") or request.get("model") or "default"
        column_key = f"{provider}::{model}"
        if column_key not in column_keys:
            column = {"key": column_key, "provider": str(provider), "model": str(model)}
            column["label"] = f"{provider} / {model}"
            column_keys[column_key] = column
            columns.append(column)
        column_label = column_keys[column_key]["label"]
        artifacts = receipt.get("artifacts") if isinstance(receipt.get("artifacts"), dict) else {}
        image_path = artifacts.get("image_path")
        receipt_path_value = artifacts.get("receipt_path")
        image_path_obj = Path(str(image_path)) if image_path else receipt_path.with_suffix(".png")
        if not image_path_obj.is_absolute():
            image_path_obj = (run_dir / image_path_obj).resolve()
        receipt_path_obj = Path(str(receipt_path_value)) if receipt_path_value else receipt_path
        if not receipt_path_obj.is_absolute():
            receipt_path_obj = (run_dir / receipt_path_obj).resolve()
        image_src = _view_rel_path(image_path_obj, view_dir)
        receipt_src = _view_rel_path(receipt_path_obj, view_dir)
        result_meta = receipt.get("result_metadata") if isinstance(receipt.get("result_metadata"), dict) else {}
        render_seconds = result_meta.get("render_seconds")
        if not isinstance(render_seconds, (int, float)):
            render_seconds = None
        if render_seconds is None and job and isinstance(job, dict):
            render_seconds = _view_seconds_between(job.get("started_at"), job.get("completed_at"))
        size = resolved.get("size") or request.get("size") or "1024x1024"
        n_value = resolved.get("n") or request.get("n") or 1
        try:
            n_value = int(n_value)
        except Exception:
            n_value = 1
        render_seconds_per_image = None
        if isinstance(render_seconds, (int, float)):
            render_seconds_per_image = float(render_seconds) / max(1, n_value)
        cost_value = _estimate_cost_value(
            provider=str(provider) if provider else None,
            model=str(model) if model else None,
            size=str(size) if size else None,
        )
        cost_per_1k = _format_cost_value(cost_value)
        llm_scores = result_meta.get("llm_scores") if isinstance(result_meta.get("llm_scores"), dict) else {}
        adherence = llm_scores.get("adherence")
        quality = llm_scores.get("quality")
        retrieval_payload = result_meta.get("llm_retrieval") if isinstance(result_meta.get("llm_retrieval"), dict) else {}
        retrieval_score = retrieval_payload.get("score")
        quality_metrics = result_meta.get("image_quality_metrics") if isinstance(result_meta.get("image_quality_metrics"), dict) else {}
        quality_gates = quality_metrics.get("gates") if isinstance(quality_metrics.get("gates"), list) else []
        warnings: list[str] = []
        if isinstance(receipt.get("warnings"), list):
            warnings.extend([str(item) for item in receipt.get("warnings") if item])
        if isinstance(resolved.get("warnings"), list):
            warnings.extend([str(item) for item in resolved.get("warnings") if item])
        snippet = _view_build_snippet(receipt)
        variant_id = receipt_path.name
        flags = []
        if isinstance(quality_gates, list):
            flags.extend([str(item) for item in quality_gates if item])
        if warnings:
            flags.extend(warnings)
        variants.append(
            {
                "id": variant_id,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "column_key": column_key,
                "column_label": column_label,
                "provider": provider,
                "model": model,
                "size": size,
                "n": n_value,
                "image_src": image_src,
                "receipt_src": receipt_src,
                "render_seconds": render_seconds,
                "render_seconds_per_image": render_seconds_per_image,
                "cost_usd": cost_value,
                "cost_per_1k": cost_per_1k,
                "adherence": adherence,
                "quality": quality,
                "retrieval_score": retrieval_score,
                "flags": flags,
                "snippet": snippet,
            }
        )

    run_title = "Param Forge • Model Showdown"
    return {
        "run": {
            "id": manifest.get("run_id") if isinstance(manifest, dict) else run_dir.name,
            "title": run_title,
            "path": str(run_dir),
        },
        "prompts": prompts,
        "columns": columns,
        "variants": variants,
    }


def _view_write_html(data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=True).replace("<", "\\u003c")
    html_text = _VIEWER_TEMPLATE.replace("__DATA__", payload)
    out_path.write_text(html_text, encoding="utf-8")


def _maybe_open_viewer(run_dir: Path) -> None:
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.exists():
        return
    has_manifest = (run_dir / "run.json").exists()
    has_receipts = any(run_dir.glob("receipt-*.json"))
    if not has_manifest and not has_receipts:
        return
    out_dir = run_dir / ".param_forge_view"
    data = _view_build_index(run_dir, out_dir)
    if not data.get("variants"):
        return
    out_path = out_dir / "index.html"
    _view_write_html(data, out_path)
    _open_path(out_path)


def _run_view_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="PARAM FORGE: view a receipt run locally.")
    parser.add_argument("path", help="Run folder, run.json, or receipt path")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for viewer (default: <run>/.param_forge_view)",
    )
    parser.add_argument("--no-open", action="store_true", help="Do not open the viewer in a browser.")
    args = parser.parse_args(argv)

    target = Path(args.path).expanduser().resolve()
    run_dir = target if target.is_dir() else target.parent
    out_dir = Path(args.out).expanduser().resolve() if args.out else (run_dir / ".param_forge_view")
    out_path = out_dir / "index.html"
    data = _view_build_index(run_dir, out_dir)
    _view_write_html(data, out_path)
    if not args.no_open:
        _open_path(out_path)
    print(f"Viewer written to {out_path}")
    return 0


def _run_generation(args: argparse.Namespace) -> int:
    _load_repo_dotenv()
    args.analyzer = _resolve_receipt_analyzer(getattr(args, "analyzer", None))
    from forge_image_api import generate, stream

    normalized_provider, normalized_model = _normalize_provider_and_model(args.provider, args.model)
    args.provider = normalized_provider
    args.model = normalized_model
    args.size = _normalize_size(str(args.size))
    if not hasattr(args, "provider_options") or not isinstance(args.provider_options, dict):
        args.provider_options = {}
    if not hasattr(args, "seed"):
        args.seed = None
    if not hasattr(args, "output_format"):
        args.output_format = None
    if not hasattr(args, "background"):
        args.background = None
    if not hasattr(args, "openai_stream"):
        args.openai_stream = _env_flag(OPENAI_STREAM_ENV)
    if not hasattr(args, "openai_responses"):
        args.openai_responses = _env_flag(OPENAI_RESPONSES_ENV)
    openai_stream, openai_responses = _apply_openai_provider_flags(args)
    use_stream = bool(openai_stream and _is_openai_gpt_image(args.provider, args.model))
    allow_prompt = not bool(getattr(args, "defaults", False))
    try:
        _ensure_api_keys(args.provider, _find_repo_dotenv(), allow_prompt=allow_prompt)
    except RuntimeError as exc:
        if not allow_prompt:
            print(f"Setup failed: {exc}")
            print(f"Tip: set {_provider_key_hint(args.provider)} in your environment or .env, then rerun.")
            return 1
        raise
    retrieval_enabled = _retrieval_score_enabled(args)
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    out_dir = Path(args.out).expanduser().resolve()

    if args.defaults:
        print("Running with defaults:")
        print(f"  provider={args.provider} size={args.size} n={args.n} out={out_dir}")
        print(f"  prompts={len(prompts)} (use --interactive to customize)")

    all_receipts: list[Path] = []
    last_image_path: Path | None = None
    for idx, prompt in enumerate(prompts, start=1):
        label = f"Generating ({idx}/{len(prompts)})"
        print(f"{label}: {prompt}")
        spinner = _Spinner(f"{label} in progress", show_elapsed=True)
        start_time = time.monotonic()
        spinner.start()
        stopped = False
        try:
            if use_stream:
                results = []
                for event in stream(
                    prompt=prompt,
                    provider=args.provider,
                    size=args.size,
                    n=args.n,
                    out_dir=out_dir,
                    model=args.model,
                    provider_options=args.provider_options,
                    seed=args.seed,
                    output_format=args.output_format,
                    background=args.background,
                ):
                    if event.type == "error":
                        raise RuntimeError(event.message or "Streaming failed.")
                    if event.type == "final" and event.result is not None:
                        results.append(event.result)
            else:
                results = generate(
                    prompt=prompt,
                    provider=args.provider,
                    size=args.size,
                    n=args.n,
                    out_dir=out_dir,
                    model=args.model,
                    provider_options=args.provider_options,
                    seed=args.seed,
                    output_format=args.output_format,
                    background=args.background,
                )
        except Exception as exc:
            spinner.stop()
            stopped = True
            print(f"Generation failed: {exc}")
            if args.provider == "gemini":
                print(
                    "Tip: Gemini image generation often requires specific model access. "
                    "Try --model gemini-2.5-flash-image or gemini-3-pro-image-preview, "
                    "or switch to --provider openai."
                )
            if args.defaults:
                return 1
            raise
        finally:
            if not stopped:
                spinner.stop()
        elapsed = time.monotonic() - start_time
        for result in results:
            print(result.image_path)
            print(result.receipt_path)
            all_receipts.append(Path(result.receipt_path))
            last_image_path = Path(result.image_path)
            scoring_label = "Council scoring"
            if retrieval_enabled:
                scoring_label = "Council scoring + retrieval"
            print(f"Stamping snapshot ({scoring_label})...")
            _apply_snapshot_for_result(
                image_path=Path(result.image_path),
                receipt_path=Path(result.receipt_path),
                prompt=prompt,
                elapsed=elapsed,
                fallback_settings=_capture_call_settings(args),
                retrieval_enabled=retrieval_enabled,
            )

    if last_image_path and sys.stdin.isatty():
        _open_path(last_image_path)
    if all_receipts and sys.stdin.isatty():
        choice = input("Open last receipt now? [y/N]: ").strip().lower()
        if choice in {"y", "yes"}:
            _open_path(all_receipts[-1])
            _analyze_receipt(
                all_receipts[-1],
                provider=args.provider,
                model=args.model,
                size=args.size,
                n=args.n,
                out_dir=str(out_dir),
                analyzer=args.analyzer,
            )
            input("\nPress Enter to exit.")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "view":
        return _run_view_cli(sys.argv[2:])
    parser = argparse.ArgumentParser(
        description="PARAM FORGE: interactive terminal UI for multi-provider image generation + receipts."
    )
    parser.add_argument("--prompt", action="append", help="Prompt text (repeatable)")
    parser.add_argument("--provider", default="openai", help="Provider name")
    parser.add_argument(
        "--analyzer",
        default=None,
        choices=ANALYZER_CHOICES,
        help="Receipt analyzer provider (default: $RECEIPT_ANALYZER or anthropic)",
    )
    parser.add_argument("--size", default="portrait", help="Size (e.g., portrait, 1024x1024, 16:9)")
    parser.add_argument("--n", type=int, default=1, help="Number of images per prompt")
    parser.add_argument(
        "--output-format",
        dest="output_format",
        default=None,
        help="Optional output format (e.g., jpeg, png, webp)",
    )
    parser.add_argument(
        "--background",
        default=None,
        help="Optional background (e.g., transparent or opaque)",
    )
    parser.add_argument(
        "--out",
        default="outputs/param_forge",
        help="Output directory (default: outputs/param_forge)",
    )
    parser.add_argument("--model", default=None, help="Optional model override")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use interactive selector (arrow keys).",
    )
    parser.add_argument(
        "--defaults",
        action="store_true",
        help="Run with defaults without prompting.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in interactive mode.",
    )
    parser.add_argument(
        "--openai-stream",
        action="store_true",
        help="Use OpenAI streaming for gpt-image models.",
    )
    parser.add_argument(
        "--openai-responses",
        action="store_true",
        help="Use OpenAI Responses API for gpt-image models.",
    )
    args = parser.parse_args()
    if args.analyzer:
        os.environ[RECEIPT_ANALYZER_ENV] = args.analyzer
    if args.openai_stream:
        os.environ[OPENAI_STREAM_ENV] = "1"
    if args.openai_responses:
        os.environ[OPENAI_RESPONSES_ENV] = "1"
    if args.interactive or (len(sys.argv) == 1 and not args.defaults):
        color_override = False if args.no_color else None
        try:
            return _run_curses_flow(color_override=color_override)
        except KeyboardInterrupt:
            print("\nCancelled.")
            return 1
    # When using the interactive flow, avoid Gemini by default unless explicitly chosen.
    if getattr(args, "interactive", False) and args.provider == "gemini":
        print("Gemini provider selected; ensure your model + permissions support image generation.")

    return _run_generation(args)


if __name__ == "__main__":
    raise SystemExit(main())
