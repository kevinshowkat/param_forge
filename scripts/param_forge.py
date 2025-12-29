#!/usr/bin/env python3
"""Generate Modulette assets for social sharing (Param Forge).

Usage:
  python scripts/param_forge.py \
    --provider openai --size portrait --n 1 --out outputs/param_forge
  python scripts/param_forge.py --interactive
  python scripts/param_forge.py --defaults

Notes:
- Loads .env from the oscillo repo root.
- Imports the local Modulette repo from /Users/mainframe/Desktop/projects/Modulette
  if it's not installed.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
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

DEFAULT_PROMPTS = [
    "Satellite view of a storm forming over a deep blue ocean, swirling cloud bands",
    "Moss-covered forest floor with sunbeams through mist, macro detail",
    "Aurora borealis above a snowy pine ridge, long exposure, crisp night sky",
]
PROMPT_SETS = {
    "Nature set A": DEFAULT_PROMPTS,
    "Nature set B": [
        "Jagged alpine ridge at sunrise, low clouds spilling through the valley",
        "Bioluminescent waves washing onto a dark shoreline under a starry sky",
        "Desert dunes with wind-sculpted ripples and a distant thunderstorm",
    ],
    "Nature set C": [
        "Tropical waterfall hidden in dense jungle, mist catching sunbeams",
        "Frozen lake with geometric cracks beneath clear ice, soft twilight",
        "Volcanic landscape with glowing lava veins and ash in the air",
    ],
}
PROVIDER_CHOICES = ["openai", "gemini", "imagen", "flux"]
SIZE_CHOICES = ["portrait", "square", "landscape", "1024x1024", "1024x1536", "1536x1024", "16:9"]
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
_MIN_CURSES_WIDTH = max(40, max(len(line) for line in _BANNER))
_MIN_CURSES_HEIGHT = max(12, len(_BANNER) + 2)


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


def _ensure_modulette_on_path() -> None:
    try:
        import modulette  # noqa: F401
        return
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    projects_root = repo_root.parent
    modulette_path = projects_root / "Modulette"
    if modulette_path.exists():
        sys.path.insert(0, str(modulette_path))
        return

    raise RuntimeError(
        "Could not import modulette. Install it with 'pip install -e ../Modulette' "
        "or ensure /Users/mainframe/Desktop/projects/Modulette exists."
    )


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


def _interactive_args_raw(color_override: bool | None = None) -> argparse.Namespace:
    print("Param Forge")
    print("Test image-gen APIs and capture receipts that help configure calls.")
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    with _RawMode(fd, original):
        provider = _select_from_list("Provider", PROVIDER_CHOICES, 0)
        size = _select_from_list("Size", SIZE_CHOICES, 0)
        n = _select_int("Images per prompt", 1, minimum=1, maximum=4)
        prompt_set_name = _select_from_list("Prompt set", list(PROMPT_SETS.keys()), 0)
        out_choice = _select_from_list("Output dir", OUT_DIR_CHOICES, 0)
    return _build_interactive_namespace(provider, size, n, prompt_set_name, out_choice)


def _build_interactive_namespace(
    provider: str,
    size: str,
    n: int,
    prompt_set_name: str,
    out_choice: str,
) -> argparse.Namespace:
    prompts = list(PROMPT_SETS[prompt_set_name])
    out_dir = "outputs/param_forge"
    if out_choice == "outputs/param_forge_dated":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = f"outputs/param_forge/{stamp}"
    model = None
    return argparse.Namespace(
        prompt=prompts,
        provider=provider,
        size=size,
        n=n,
        out=out_dir,
        model=model,
        interactive=True,
        defaults=False,
        no_color=False,
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

    provider_idx = 0
    size_idx = 0
    count_value = 1
    prompt_idx = 0
    out_idx = 0
    field_idx = 0

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
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
        if y < height - 1:
            try:
                stdscr.addstr(
                    y,
                    0,
                    "Test image-gen APIs and capture receipts that help configure calls."[: width - 1],
                )
            except curses.error:
                pass
            y += 2
        if y < height - 1:
            try:
                stdscr.addstr(
                    y,
                    0,
                    "Use arrow keys to change, Enter to advance, Q to cancel."[: width - 1],
                )
            except curses.error:
                pass
            y += 2

        y = _draw_choice_line(
            stdscr,
            y,
            "Provider",
            PROVIDER_CHOICES,
            provider_idx,
            field_idx == 0,
            field_idx,
            0,
            highlight_pair,
            done_pair,
            color_enabled,
        )
        y = _draw_choice_line(
            stdscr,
            y,
            "Size",
            SIZE_CHOICES,
            size_idx,
            field_idx == 1,
            field_idx,
            1,
            highlight_pair,
            done_pair,
            color_enabled,
        )
        y = _draw_count_line(
            stdscr,
            y,
            "Images per prompt",
            count_value,
            field_idx == 2,
            field_idx,
            2,
            highlight_pair,
            done_pair,
            color_enabled,
        )
        y = _draw_choice_line(
            stdscr,
            y,
            "Prompt set",
            list(PROMPT_SETS.keys()),
            prompt_idx,
            field_idx == 3,
            field_idx,
            3,
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
            field_idx == 4,
            field_idx,
            4,
            highlight_pair,
            done_pair,
            color_enabled,
        )
        stdscr.refresh()

        key = stdscr.getch()
        if key == -1:
            continue
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt
        if key in (curses.KEY_UP, ord("k"), ord("K")):
            field_idx = max(0, field_idx - 1)
            continue
        if key in (curses.KEY_DOWN, ord("j"), ord("J")):
            field_idx = min(4, field_idx + 1)
            continue
        if key in (curses.KEY_LEFT, ord("h"), ord("H"), ord("a"), ord("A")):
            if field_idx == 0:
                provider_idx = (provider_idx - 1) % len(PROVIDER_CHOICES)
            elif field_idx == 1:
                size_idx = (size_idx - 1) % len(SIZE_CHOICES)
            elif field_idx == 2:
                count_value = max(1, count_value - 1)
            elif field_idx == 3:
                prompt_idx = (prompt_idx - 1) % len(PROMPT_SETS)
            elif field_idx == 4:
                out_idx = (out_idx - 1) % len(OUT_DIR_CHOICES)
            continue
        if key in (curses.KEY_RIGHT, ord("l"), ord("L"), ord("d"), ord("D")):
            if field_idx == 0:
                provider_idx = (provider_idx + 1) % len(PROVIDER_CHOICES)
            elif field_idx == 1:
                size_idx = (size_idx + 1) % len(SIZE_CHOICES)
            elif field_idx == 2:
                count_value = min(4, count_value + 1)
            elif field_idx == 3:
                prompt_idx = (prompt_idx + 1) % len(PROMPT_SETS)
            elif field_idx == 4:
                out_idx = (out_idx + 1) % len(OUT_DIR_CHOICES)
            continue
        if key in (10, 13, curses.KEY_ENTER, ord("\t")):
            if field_idx < 4:
                field_idx += 1
                continue
            return _build_interactive_namespace(
                PROVIDER_CHOICES[provider_idx],
                SIZE_CHOICES[size_idx],
                count_value,
                list(PROMPT_SETS.keys())[prompt_idx],
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

    result: dict[str, object] = {"open_path": None, "exit_code": 0, "ran": False}

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
        args = _interactive_args_curses(stdscr, color_override=color_override)
        result["ran"] = True
        if args is None:
            result["exit_code"] = 1
            return
        try:
            _load_repo_dotenv()
            _ensure_modulette_on_path()
            from modulette import generate  # type: ignore
        except Exception as exc:
            result["exit_code"] = 1
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            error_line = f"Setup failed: {exc}"
            try:
                stdscr.addstr(0, 0, error_line[: max(0, width - 1)])
            except curses.error:
                pass
            stdscr.refresh()
            stdscr.getch()
            return
        stdscr.erase()
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
                        stdscr.addstr(y, i, ch, curses.color_pair(1) | curses.A_BOLD)
                    except curses.error:
                        pass
            else:
                try:
                    stdscr.addstr(y, 0, truncated)
                except curses.error:
                    pass
            y += 1
        if y < height - 1:
            try:
                stdscr.addstr(
                    y,
                    0,
                    "Generating images... (press Q to cancel)"[: width - 1],
                )
            except curses.error:
                pass
            y += 2
        stdscr.refresh()

        receipts: list[Path] = []
        for idx, prompt in enumerate(args.prompt, start=1):
            if y < height - 1:
                line = f"Generating ({idx}/{len(args.prompt)}): {prompt}"
                try:
                    stdscr.addstr(y, 0, line[: width - 1])
                except curses.error:
                    pass
            stdscr.refresh()
            try:
                results = generate(
                    prompt=prompt,
                    provider=args.provider,
                    size=args.size,
                    n=args.n,
                    out_dir=Path(args.out),
                    model=args.model,
                )
            except Exception as exc:
                result["exit_code"] = 1
                if y + 2 < height - 1:
                    try:
                        stdscr.addstr(y + 1, 0, f"Generation failed: {exc}"[: width - 1])
                    except curses.error:
                        pass
                stdscr.refresh()
                stdscr.getch()
                return
            for res in results:
                receipts.append(Path(res.receipt_path))
            y += 2
            if y >= height - 2:
                y = min(height - 2, max(0, height - 3))

        if receipts:
            prompt_line = "Open last receipt now? (o = open, any key = exit)"
            try:
                stdscr.addstr(min(height - 2, y), 0, prompt_line[: width - 1])
            except curses.error:
                pass
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("o"), ord("O")):
                result["open_path"] = receipts[-1]

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
    if result.get("open_path"):
        _open_path(Path(result["open_path"]))  # type: ignore[arg-type]
    return int(result.get("exit_code", 0))


def _run_raw_fallback(reason: str | None, color_override: bool | None) -> int:
    if reason:
        reason = reason.strip()
    if reason:
        print(f"Curses UI unavailable ({reason}). Falling back to raw prompts.")
    args = _interactive_args_raw(color_override=color_override)
    return _run_generation(args)


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
    indices, prefix, suffix = _visible_indices(choices, selected_idx, width)
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
            token = f"[{token}]"
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
        else:
            attr = curses.A_DIM if not is_done else curses.A_NORMAL
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
    token = f"[{value}]"
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


def _visible_indices(choices: list[str], selected_idx: int, max_width: int) -> tuple[list[int], bool, bool]:
    def line_len(indices: list[int], prefix: bool, suffix: bool) -> int:
        tokens = [
            f"[{choices[i]}]" if i == selected_idx else choices[i]
            for i in indices
        ]
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
    def __init__(self, message: str, interval: float = 0.1) -> None:
        self.message = message
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if sys.stdout.isatty():
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
            sys.stdout.write(f"\r{self.message} {frame}")
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


def _ensure_api_keys(provider: str, dotenv_path: Path | None) -> None:
    provider = provider.strip().lower()
    if provider == "openai":
        if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_BACKUP"):
            return
        if not _prompt_for_key("OPENAI_API_KEY", dotenv_path):
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider.")
        return


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
    _ensure_modulette_on_path()

    from modulette import generate

    _ensure_api_keys(args.provider, _find_repo_dotenv())
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    out_dir = Path(args.out).expanduser().resolve()

    if args.defaults:
        print("Running with defaults:")
        print(f"  provider={args.provider} size={args.size} n={args.n} out={out_dir}")
        print(f"  prompts={len(prompts)} (use --interactive to customize)")

    all_receipts: list[Path] = []
    for idx, prompt in enumerate(prompts, start=1):
        label = f"Generating ({idx}/{len(prompts)})"
        print(f"{label}: {prompt}")
        spinner = _Spinner(f"{label} in progress")
        spinner.start()
        stopped = False
        try:
            results = generate(
                prompt=prompt,
                provider=args.provider,
                size=args.size,
                n=args.n,
                out_dir=out_dir,
                model=args.model,
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
        for result in results:
            print(result.image_path)
            print(result.receipt_path)
            all_receipts.append(Path(result.receipt_path))

    if all_receipts and sys.stdin.isatty():
        choice = input("Open last receipt now? [y/N]: ").strip().lower()
        if choice in {"y", "yes"}:
            _open_path(all_receipts[-1])
    return 0
    if provider == "gemini":
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return
        if not _prompt_for_key("GEMINI_API_KEY", dotenv_path):
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for Gemini provider.")
        return
    if provider == "flux":
        if os.getenv("BFL_API_KEY") or os.getenv("FLUX_API_KEY"):
            return
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
        if not _prompt_for_key("GOOGLE_API_KEY", dotenv_path):
            raise RuntimeError(
                "GOOGLE_API_KEY (or Vertex credentials) is required for Imagen provider."
            )
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Param Forge: generate Modulette social share assets.")
    parser.add_argument("--prompt", action="append", help="Prompt text (repeatable)")
    parser.add_argument("--provider", default="openai", help="Provider name")
    parser.add_argument("--size", default="portrait", help="Size (e.g., portrait, 1024x1024, 16:9)")
    parser.add_argument("--n", type=int, default=1, help="Number of images per prompt")
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
    args = parser.parse_args()
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
