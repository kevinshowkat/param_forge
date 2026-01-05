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
    "openai": ["gpt-image-1.5", "gpt-image-1"],
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
    top_level_keys = {"size", "n", "seed", "output_format", "background", "model"}

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
        provider = _select_from_list("Provider", PROVIDER_CHOICES, 0)
        model = _select_from_list("Model", _model_choices_for(provider), 0)
        size_label = _select_from_list("Size", _size_label_choices(), 0)
        size = _size_value_from_label(size_label)
        n = _select_int("Images per prompt", 1, minimum=1, maximum=4)
        out_choice = _select_from_list("Output dir", OUT_DIR_CHOICES, 0)
    return _build_interactive_namespace(provider, model, size, n, out_choice)


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
    provider = _prompt_choice("Provider", PROVIDER_CHOICES, 0)
    model = _prompt_choice("Model", _model_choices_for(provider), 0)
    size_label = _prompt_choice("Size", _size_label_choices(), 0)
    size = _size_value_from_label(size_label)
    n = _prompt_int("Images per prompt", 1, minimum=1, maximum=4)
    out_choice = _prompt_choice("Output dir", OUT_DIR_CHOICES, 0)
    return _build_interactive_namespace(provider, model, size, n, out_choice)


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


def _build_interactive_namespace(
    provider: str,
    model: str | None,
    size: str,
    n: int,
    out_choice: str,
) -> argparse.Namespace:
    prompts = list(DEFAULT_PROMPTS)
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
    size_idx = 0
    count_value = 1
    out_idx = 0
    field_idx = 0

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
            "Repeatable prompt runs with fixed knobs, receipt exports, and side-by-side rounds."
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
                PROVIDER_CHOICES,
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
            y = _draw_choice_column(
                stdscr,
                y,
                "Size",
                size_choices,
                size_idx,
                field_idx == 3,
                field_idx,
                3,
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
                field_idx == 4,
                field_idx,
                4,
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
                field_idx == 5,
                field_idx,
                5,
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
            if field_idx == 3:
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
            if field_idx == 3:
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
            elif field_idx == 3:
                size_idx = (size_idx - 1) % len(size_choices)
            elif field_idx == 4:
                count_value = max(1, count_value - 1)
            elif field_idx == 5:
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
            elif field_idx == 3:
                size_idx = (size_idx + 1) % len(size_choices)
            elif field_idx == 4:
                count_value = min(4, count_value + 1)
            elif field_idx == 5:
                out_idx = (out_idx + 1) % len(OUT_DIR_CHOICES)
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
            if field_idx < 5:
                field_idx += 1
                continue
            return _build_interactive_namespace(
                PROVIDER_CHOICES[provider_idx],
                _model_choices_for(PROVIDER_CHOICES[provider_idx])[model_idx],
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
        args.analyzer = _resolve_receipt_analyzer(getattr(args, "analyzer", None))
        try:
            _load_repo_dotenv()
            _ensure_api_keys(args.provider, _find_repo_dotenv(), allow_prompt=False)
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
                    for res_idx, res in enumerate(results, start=1):
                        status = f"Stamping snapshot {res_idx}/{len(results)} (Council scoring)..."
                        _write_status_line(
                            stdscr,
                            status_y,
                            _truncate_text(status, max(20, width - 1)),
                            width,
                            curses.A_DIM,
                        )
                        stdscr.refresh()
                        scores = _apply_snapshot_for_result(
                            image_path=Path(res.image_path),
                            receipt_path=Path(res.receipt_path),
                            prompt=prompt,
                            elapsed=elapsed,
                            fallback_settings=_capture_call_settings(args),
                        )
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
                    )
                    if cost_line:
                        stored_cost = cost_line.replace("COST:", "").strip()
                    if run_index >= MAX_ROUNDS:
                        base_settings = last_call_settings or _capture_call_settings(args)
                        quality_baseline, quality_current = _quality_from_history(analysis_history)
                        adherence_baseline, adherence_current = _adherence_from_history(analysis_history)
                        baseline_elapsed, baseline_cost_line = _baseline_metrics_from_history(analysis_history)
                        recommended_settings = (
                            _settings_after_recommendation(base_settings, recommendation)
                            if recommendation
                            else base_settings
                        )
                        validation_needed = False
                        if baseline_settings:
                            validation_needed = bool(
                                _diff_call_settings(base_settings, recommended_settings)
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
                            receipt_path=receipts[-1] if receipts else None,
                            baseline_settings=baseline_settings,
                            force_quality_metrics=bool(stored_goals and "maximize quality of render" in stored_goals),
                        )
                        if validation_needed and _prompt_yes_no_curses(
                            stdscr,
                            "Run final validation with the full recommended settings?",
                            color_enabled=color_enabled,
                        ):
                            if recommendation:
                                _apply_recommendation(args, recommendation)
                            validation_settings = _capture_call_settings(args)
                            header_override = _build_validation_header_lines(
                                baseline_settings=baseline_settings or validation_settings,
                                current_settings=validation_settings,
                                max_width=max(20, width - 1),
                            )
                            (
                                val_receipts,
                                val_images,
                                _,
                                _,
                                _,
                                _,
                                _,
                                _,
                            ) = _generate_once(
                                MAX_ROUNDS + 1,
                                None,
                                [],
                                None,
                                header_override,
                            )
                            if baseline_image and val_images:
                                composite = _compose_side_by_side(
                                    baseline_image,
                                    val_images[-1],
                                    label_left="Round 1 baseline",
                                    label_right="Final validation",
                                    out_dir=Path(args.out),
                                )
                                if composite:
                                    _open_path(composite)
                                else:
                                    _open_path(val_images[-1])
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
                    )
                    if cost_line:
                        stored_cost = cost_line.replace("COST:", "").strip()
                    if run_index >= MAX_ROUNDS:
                        base_settings = last_call_settings or _capture_call_settings(args)
                        quality_baseline, quality_current = _quality_from_history(analysis_history)
                        adherence_baseline, adherence_current = _adherence_from_history(analysis_history)
                        baseline_elapsed, baseline_cost_line = _baseline_metrics_from_history(analysis_history)
                        recommended_settings = (
                            _settings_after_recommendation(base_settings, recommendation)
                            if recommendation
                            else base_settings
                        )
                        validation_needed = False
                        if baseline_settings:
                            validation_needed = bool(
                                _diff_call_settings(base_settings, recommended_settings)
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
                            receipt_path=receipts[-1] if receipts else None,
                            baseline_settings=baseline_settings,
                            force_quality_metrics=bool(user_goals and "maximize quality of render" in user_goals),
                        )
                        if validation_needed and _prompt_yes_no_curses(
                            stdscr,
                            "Run final validation with the full recommended settings?",
                            color_enabled=color_enabled,
                        ):
                            if recommendation:
                                _apply_recommendation(args, recommendation)
                            validation_settings = _capture_call_settings(args)
                            header_override = _build_validation_header_lines(
                                baseline_settings=baseline_settings or validation_settings,
                                current_settings=validation_settings,
                                max_width=max(20, width - 1),
                            )
                            (
                                val_receipts,
                                val_images,
                                _,
                                _,
                                _,
                                _,
                                _,
                                _,
                            ) = _generate_once(
                                MAX_ROUNDS + 1,
                                None,
                                [],
                                None,
                                header_override,
                            )
                            if baseline_image and val_images:
                                composite = _compose_side_by_side(
                                    baseline_image,
                                    val_images[-1],
                                    label_left="Round 1 baseline",
                                    label_right="Final validation",
                                    out_dir=Path(args.out),
                                )
                                if composite:
                                    _open_path(composite)
                                else:
                                    _open_path(val_images[-1])
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
    return _run_generation(args)


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
        "You are a strict image evaluator. Given the prompt and the generated image, "
        "rate prompt adherence and overall image quality. "
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


def _build_snapshot_lines(
    *,
    elapsed: float | None,
    cost: float | None,
    adherence: int | None,
    quality: int | None,
) -> list[str]:
    elapsed_text = "N/A" if elapsed is None else f"{elapsed:.1f}s"
    adherence_text = "N/A" if adherence is None else f"{adherence}/100"
    quality_text = "N/A" if quality is None else f"{quality}/100"
    return [
        f"render: {elapsed_text}",
        f"cost: {_format_cost_value(cost)}",
        f"prompt adherence: {adherence_text}",
        f"LLM-rated quality: {quality_text}",
    ]


def _snapshot_template_lines() -> list[str]:
    return [
        "render: 9999.9s",
        "cost: $99999/1K",
        "prompt adherence: 100/100",
        "LLM-rated quality: 100/100",
    ]


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


def _apply_snapshot_overlay(image_path: Path, lines: list[str]) -> bool:
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
    template_lines = _snapshot_template_lines()
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


def _apply_snapshot_for_result(
    *,
    image_path: Path,
    receipt_path: Path,
    prompt: str,
    elapsed: float | None,
    fallback_settings: dict[str, object],
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
    lines = _build_snapshot_lines(
        elapsed=elapsed,
        cost=cost_value,
        adherence=adherence,
        quality=quality,
    )
    _apply_snapshot_overlay(image_path, lines)
    return adherence, quality


def _diff_call_settings(prev: dict[str, object], current: dict[str, object]) -> list[str]:
    diffs: list[str] = []
    for key in ("provider", "model", "size", "n", "seed", "output_format", "background"):
        prev_val = prev.get(key)
        curr_val = current.get(key)
        if prev_val != curr_val:
            diffs.append(
                f"{key}: {_format_setting_value(prev_val)} -> {_format_setting_value(curr_val)}"
            )
    prev_opts = prev.get("provider_options")
    curr_opts = current.get("provider_options")
    if not isinstance(prev_opts, dict):
        prev_opts = {}
    if not isinstance(curr_opts, dict):
        curr_opts = {}
    for key in sorted(set(prev_opts.keys()) | set(curr_opts.keys())):
        if prev_opts.get(key) != curr_opts.get(key):
            diffs.append(
                "provider_options."
                + str(key)
                + ": "
                + _format_setting_value(prev_opts.get(key))
                + " -> "
                + _format_setting_value(curr_opts.get(key))
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
    force_quality_metrics: bool = False,
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
    if goals:
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
            provider=str(recommended_settings.get("provider"))
            if recommended_settings.get("provider")
            else None,
            model=str(recommended_settings.get("model")) if recommended_settings.get("model") else None,
            size=str(recommended_settings.get("size")) if recommended_settings.get("size") else None,
        )
        if cost_target is None:
            if not _diff_call_settings(recommended_settings, settings):
                cost_target = _parse_cost_amount(cost_line) or cost_ref
            elif baseline_settings and not _diff_call_settings(recommended_settings, baseline_settings):
                cost_target = cost_ref
        if cost_target is None and cost_ref is not None:
            cost_target = cost_ref

        time_ref = baseline_elapsed if baseline_elapsed is not None else last_elapsed
        time_target = None
        if time_ref is not None:
            if last_elapsed is not None and not _diff_call_settings(recommended_settings, settings):
                time_target = last_elapsed
            elif cost_ref is not None and cost_target is not None and cost_ref > 0:
                time_target = time_ref * (cost_target / cost_ref)
            else:
                area_ref = _size_area_estimate(reference_settings.get("size"))
                area_target = _size_area_estimate(recommended_settings.get("size"))
                if area_ref and area_target and area_ref > 0:
                    time_target = time_ref * (area_target / area_ref)

        gain_lines: list[str] = []
        wants_cost = "minimize cost of render" in goals
        wants_time = "minimize time to render" in goals
        wants_quality = "maximize quality of render" in goals
        quality_secondary = wants_quality and (wants_cost or wants_time)
        quality_delta_line = None
        adherence_delta_line = None
        if quality_baseline is not None and quality_current is not None and quality_baseline > 0:
            delta = quality_current - quality_baseline
            pct = (delta / quality_baseline) * 100.0
            direction = "increase" if delta >= 0 else "decrease"
            quality_delta_line = f"Quality: {abs(pct):.0f}% {direction}"
        if adherence_baseline is not None and adherence_current is not None and adherence_baseline > 0:
            delta = adherence_current - adherence_baseline
            pct = (delta / adherence_baseline) * 100.0
            direction = "increase" if delta >= 0 else "decrease"
            adherence_delta_line = f"Prompt adherence: {abs(pct):.0f}% {direction}"
        cost_delta_line = None
        if cost_ref is not None and cost_target is not None and cost_ref > 0:
            delta = (cost_target - cost_ref) / cost_ref
            if abs(delta) >= 0.005:
                pct = abs(delta) * 100.0
                direction = "decrease" if delta < 0 else "increase"
                cost_delta_line = f"Cost: {pct:.0f}% {direction}"
        time_delta_line = None
        if time_ref is not None and time_target is not None and time_ref > 0:
            delta = (time_target - time_ref) / time_ref
            if abs(delta) >= 0.005:
                pct = abs(delta) * 100.0
                direction = "decrease" if delta < 0 else "increase"
                time_delta_line = f"Time: {pct:.0f}% {direction}"

        def _maybe_append(line: str | None, *, primary: bool, quality: bool = False) -> None:
            if not line:
                return
            suffix = ""
            if goals and not primary:
                suffix = " (secondary)"
            if quality and quality_secondary:
                suffix = " (secondary)"
            gain_lines.append(f"{line}{suffix}")

        _maybe_append(cost_delta_line, primary=wants_cost)
        _maybe_append(time_delta_line, primary=wants_time)
        if wants_quality:
            _maybe_append(quality_delta_line or ("Quality: N/A (no baseline)" if force_quality_metrics else None), primary=not quality_secondary, quality=True)
            _maybe_append(adherence_delta_line or ("Prompt adherence: N/A (no baseline)" if force_quality_metrics else None), primary=not quality_secondary, quality=True)
        else:
            _maybe_append(quality_delta_line, primary=False, quality=True)
            _maybe_append(adherence_delta_line, primary=False, quality=True)

        if gain_lines:
            _append_wrapped("Expected gains:", "section")
            for line in gain_lines:
                _append_wrapped(line, "goal")
        else:
            _append_wrapped("Expected gains: none (no changes recommended).", "goal")
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
    receipt_path: Path | None = None,
    baseline_settings: dict[str, object] | None = None,
    force_quality_metrics: bool = False,
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
        force_quality_metrics=force_quality_metrics,
    )
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
    footer_line = "Open receipt (o) • Press Q or Enter to exit"
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
            _open_path(receipt_path)
            continue
        break




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


def _ensure_api_keys(provider: str, dotenv_path: Path | None, allow_prompt: bool = True) -> None:
    provider = provider.strip().lower()
    if provider == "openai":
        if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_BACKUP"):
            return
        if not allow_prompt:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider.")
        if not _prompt_for_key("OPENAI_API_KEY", dotenv_path):
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider.")
        return
    if provider == "gemini":
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return
        if not allow_prompt:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for Gemini provider.")
        if not _prompt_for_key("GEMINI_API_KEY", dotenv_path):
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for Gemini provider.")
        return
    if provider == "flux":
        if os.getenv("BFL_API_KEY") or os.getenv("FLUX_API_KEY"):
            return
        if not allow_prompt:
            raise RuntimeError("BFL_API_KEY (or FLUX_API_KEY) is required for Flux provider.")
        if not _prompt_for_key("BFL_API_KEY", dotenv_path):
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
        if not _prompt_for_key("GOOGLE_API_KEY", dotenv_path):
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
            "Prefer step-change recommendations over incremental tweaks and honor the user goals.",
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
        if _is_anthropic_rate_limit_error(exc):
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
        "You are an expert image-generation assistant. An image output is attached. "
        "Respond with EXACTLY four lines labeled ADH, UNSET, COST, REC, then a <setting_json> block. "
        "No tables, no markdown, no bullet points. "
        "Total response INCLUDING <setting_json> must be <=500 characters.\n\n"
        "ADH: brief prompt adherence summary (1 sentence).\n"
        "UNSET: list 2–4 most important unset params (short list).\n"
        "COST: estimated USD cost per 1K images for this provider/model (short).\n"
        "REC: 1–3 short recommendations aligned to user goals. "
        "If speed is a priority, you may suggest unconventional levers (e.g., output format/filetype, "
        "compression, or size tradeoffs) in addition to standard settings. "
        "Prefer step-change recommendations over incremental tweaks (avoid tiny deltas like steps 20→30); "
        "think big and unconventional when it helps meet the goals.\n"
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
        f"User notes: {notes_line or '(none)'}\n\n"
        f"{web_search_text}"
        f"{call_path_text}"
        f"{history_block}"
        f"{model_block}"
        f"Allowed settings for recommendations (API settings for this model): {allowed_line}\n"
        "Model changes must stay within the same provider and use the listed model options.\n"
        "Use setting_target='provider_options' for provider-specific options, "
        "and setting_target='request' for top-level settings like size, n, seed, "
        "output_format, background, or model.\n\n"
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
        "You are an expert image-generation assistant. "
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
        f"User notes: {notes_line or '(none)'}\n"
        f"{web_search_text}"
        f"{call_path_text}"
        f"{history_block}"
        f"{model_block}"
        "If speed is a priority, consider unconventional levers (e.g., output format/filetype, "
        "compression, or size tradeoffs) but still choose from the allowed list. "
        "Prefer step-change recommendations over incremental tweaks (avoid tiny deltas like steps 20→30); "
        "think big and unconventional when it helps meet the goals.\n"
        "If cost/time goals are selected, treat them as primary and only pursue quality improvements "
        "that do not materially worsen cost/time.\n"
        f"You have at most {max_rounds} total rounds. This is round {current_round}; "
        f"{remaining_rounds} round(s) remain after this. "
        "Plan a coherent testing path and avoid flip-flopping settings unless there's a strong reason.\n"
        "Model changes must stay within the same provider and use the listed model options.\n"
        "Use setting_target='provider_options' for provider-specific options, "
        "and setting_target='request' for top-level settings like size, n, seed, "
        "output_format, background, or model.\n"
        f"Allowed settings: {allowed_line}\n\n"
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
    _ensure_api_keys(args.provider, _find_repo_dotenv())
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
            print("Stamping snapshot (Council scoring)...")
            _apply_snapshot_for_result(
                image_path=Path(result.image_path),
                receipt_path=Path(result.receipt_path),
                prompt=prompt,
                elapsed=elapsed,
                fallback_settings=_capture_call_settings(args),
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
