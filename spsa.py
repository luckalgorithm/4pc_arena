#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from match import (
    EngineConfig,
    EngineError,
    MatchConfig,
    load_fens,
    load_options,
    play_match,
)


DEFAULT_ALPHA = 0.602
DEFAULT_GAMMA = 0.101
MAX_HISTORY_ENTRIES = 5000
PARAMETER_HISTORY_INTERVAL = 100
MAX_PARAMETER_HISTORY_ENTRIES = 500


@dataclass
class Parameter:
    name: str
    theta: float
    start: float
    minimum: float
    maximum: float
    c: float
    c_end: float
    a: float
    a_end: float
    r_end: float


@dataclass
class Step:
    parameter: str
    flip: int
    c: float
    r: float
    plus: float
    minus: float


def finite_float(value: str, *, field: str, line: int | None = None) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        location = f"line {line} {field}" if line is not None else field
        raise ValueError(f"Invalid SPSA {location}: expected a number") from exc
    if not math.isfinite(number):
        location = f"line {line} {field}" if line is not None else field
        raise ValueError(f"Invalid SPSA {location}: expected a finite number")
    return number


def clip(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def option_value(value: float) -> int:
    return int(round(value))


def parse_parameters(
    path: Path,
    *,
    iterations: int,
    a: float,
    alpha: float,
    gamma: float,
) -> list[Parameter]:
    params: list[Parameter] = []
    raw = path.read_text(encoding="utf-8")
    for line_number, raw_line in enumerate(raw.splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        chunks = [chunk.strip() for chunk in line.split(",")]
        if len(chunks) != 6:
            raise ValueError(
                f"Invalid SPSA line {line_number}: expected "
                "name,start,min,max,c_end,r_end"
            )
        name = chunks[0]
        if not name:
            raise ValueError(f"Invalid SPSA line {line_number}: empty name")
        start = finite_float(chunks[1], field="start", line=line_number)
        minimum = finite_float(chunks[2], field="min", line=line_number)
        maximum = finite_float(chunks[3], field="max", line=line_number)
        c_end = finite_float(chunks[4], field="c_end", line=line_number)
        r_end = finite_float(chunks[5], field="r_end", line=line_number)
        if minimum > maximum:
            raise ValueError(f"Invalid SPSA line {line_number}: min > max")
        if c_end <= 0:
            raise ValueError(f"Invalid SPSA line {line_number}: c_end must be > 0")
        if r_end < 0:
            raise ValueError(f"Invalid SPSA line {line_number}: r_end must be >= 0")
        theta = clip(start, minimum, maximum)
        c = c_end * iterations**gamma
        a_end = r_end * c_end**2
        params.append(
            Parameter(
                name=name,
                theta=theta,
                start=start,
                minimum=minimum,
                maximum=maximum,
                c=c,
                c_end=c_end,
                a=a_end * (a + iterations) ** alpha,
                a_end=a_end,
                r_end=r_end,
            )
        )
    if not params:
        raise ValueError(f"No SPSA parameters found: {path}")
    return params


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def compact_state_history(
    state: dict[str, Any],
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    raw_history = list(state.get("history", []))
    updates_completed = int(state.get("updates_completed", len(raw_history)))

    parameter_history = list(state.get("parameter_history", []))
    if not parameter_history:
        first_update = updates_completed - len(raw_history)
        for index, entry in enumerate(raw_history, 1):
            update_number = first_update + index
            saved_params = entry.get("parameters")
            if (
                update_number % PARAMETER_HISTORY_INTERVAL == 0
                and isinstance(saved_params, list)
            ):
                parameter_history.append(
                    {
                        "iteration": int(entry["iteration"]),
                        "theta": {
                            str(param["name"]): float(param["theta"])
                            for param in saved_params
                        },
                    }
                )

    history = [
        {
            "iteration": int(entry["iteration"]),
            "pairs": int(entry["pairs"]),
            "result": int(entry["result"]),
            "summary": entry["summary"],
        }
        for entry in raw_history[-MAX_HISTORY_ENTRIES:]
    ]
    parameter_history = parameter_history[-MAX_PARAMETER_HISTORY_ENTRIES:]

    state["format"] = 2
    state["updates_completed"] = updates_completed
    state["history"] = history
    state["parameter_history"] = parameter_history
    return updates_completed, history, parameter_history


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_export(target: str, text: str) -> None:
    if target == "-":
        print(text, end="")
    else:
        write_text(Path(target), text)


def restore_parameters(params: list[Parameter], state: dict[str, Any]) -> int:
    by_name = {param.name: param for param in params}
    for saved in state.get("parameters", []):
        name = saved.get("name")
        if name in by_name:
            by_name[name].theta = clip(
                float(saved["theta"]),
                by_name[name].minimum,
                by_name[name].maximum,
            )
    return int(state.get("iteration", 0))


def make_steps(
    params: list[Parameter],
    *,
    iteration: int,
    a: float,
    alpha: float,
    gamma: float,
    rng: random.Random,
) -> list[Step]:
    iter_local = iteration + 1
    steps: list[Step] = []
    for param in params:
        flip = rng.choice((-1, 1))
        c = param.c / iter_local**gamma
        r = param.a / (a + iter_local) ** alpha / c**2
        steps.append(
            Step(
                parameter=param.name,
                flip=flip,
                c=c,
                r=r,
                plus=clip(param.theta + c * flip, param.minimum, param.maximum),
                minus=clip(param.theta - c * flip, param.minimum, param.maximum),
            )
        )
    return steps


def options_from_steps(
    base_options: dict[str, int | bool | str],
    steps: list[Step],
    *,
    plus: bool,
) -> dict[str, int | bool | str]:
    options = dict(base_options)
    for step in steps:
        value = step.plus if plus else step.minus
        options[step.parameter] = option_value(value)
    return options


def update_parameters(params: list[Parameter], steps: list[Step], result: int) -> None:
    by_name = {param.name: param for param in params}
    for step in steps:
        param = by_name[step.parameter]
        delta = step.r * step.c * result * step.flip
        param.theta = clip(param.theta + delta, param.minimum, param.maximum)


def parameter_rows(
    params: list[Parameter],
    *,
    iteration: int,
    a: float,
    alpha: float,
    gamma: float,
) -> list[str]:
    iter_local = iteration + 1
    rows = []
    for param in params:
        c = param.c / iter_local**gamma
        r = param.a / (a + iter_local) ** alpha / c**2
        rows.append(
            f"{param.name},{param.theta:.6g},{param.minimum:.6g},{param.maximum:.6g},"
            f"{c:.6g},{r:.6g}"
        )
    return rows


def cxx_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def rounded_theta(param: dict[str, Any]) -> int:
    value = round(float(param["theta"]))
    return int(clip(value, float(param["minimum"]), float(param["maximum"])))


def export_constexpr_block(params: list[dict[str, Any]], source: str, iteration: Any) -> str:
    values = {str(param["name"]): rounded_theta(param) for param in params}
    indent = "  "

    def value(name: str, default: int) -> int:
        return values.get(name, default)

    def score(mg: int, eg: int) -> str:
        return f"S({mg:>4},{eg:>4})"

    def line(text: str = "") -> str:
        return indent + text if text else ""

    mobility_mg = [
        [-62, -53, -12, -4, 3, 13, 22, 28, 33],
        [-48, -20, 16, 26, 38, 51, 55, 63, 63, 68, 81, 81, 91, 98],
        [-58, -27, -15, -10, -5, -2, 9, 16, 30, 29, 32, 38, 46, 48, 58],
        [-39, -21, 3, 3, 14, 22, 28, 41, 43, 48, 56, 60, 60, 66, 67, 70,
         71, 73, 79, 88, 88, 99, 102, 102, 106, 109, 113, 116],
    ]
    mobility_eg = [
        [-81, -56, -30, -14, 8, 15, 23, 27, 33],
        [-59, -23, -3, 13, 24, 42, 54, 57, 65, 73, 78, 86, 88, 97],
        [-76, -18, 28, 55, 69, 82, 112, 118, 132, 142, 155, 165, 166, 169, 171],
        [-36, -15, 8, 18, 34, 54, 61, 73, 79, 92, 94, 104, 113, 120, 123, 126,
         133, 136, 140, 143, 148, 166, 170, 175, 184, 191, 206, 212],
    ]

    lines = [
        line("#define S(mg, eg) make_score(mg, eg)"),
        "",
        line(f"// Exported from {source} at iteration {iteration}"),
        line("constexpr Score MobilityBonus[4][32] = {"),
    ]
    for piece, (mg_row, eg_row) in enumerate(zip(mobility_mg, mobility_eg)):
        entries = [
            score(
                value(f"MobilityBonusMg[{piece}][{idx}]", mg),
                value(f"MobilityBonusEg[{piece}][{idx}]", eg),
            )
            for idx, (mg, eg) in enumerate(zip(mg_row, eg_row))
        ]
        lines.append(line("  { " + ", ".join(entries) + " },"))
    lines.append(line("};"))
    lines.append("")

    rook_mg = [value("RookOnFileMg[0]", 32), value("RookOnFileMg[1]", 71)]
    rook_eg = [value("RookOnFileEg[0]", 6), value("RookOnFileEg[1]", 38)]
    lines.append(
        line("constexpr Score RookOnFile[] = { "
        + ", ".join(score(mg, eg) for mg, eg in zip(rook_mg, rook_eg))
        + " };")
    )
    lines.append("")

    weights = [0, 0] + [
        value(f"KingAttackWeights[{idx}]", default)
        for idx, default in enumerate([81, 52, 44, 10], 2)
    ]
    lines.append(
        line("constexpr int KingAttackWeights[PIECE_TYPE_NB] = { "
        + ", ".join(str(v) for v in weights)
        + " };")
    )
    lines.append("")

    lines.append(
        line("constexpr Score KingProtector = "
        + score(value("KingProtectorMg", 7), value("KingProtectorEg", 8))
        + ";")
    )
    lines.append(
        line("constexpr Score KingInCheckPenalty = "
        + score(value("KingInCheckPenaltyMg", 250), value("KingInCheckPenaltyEg", 40))
        + ";")
    )
    lines.append("")

    threat_minor_mg = [0, 6, 59, 79, 90, 79, 0, 0]
    threat_minor_eg = [0, 32, 41, 56, 119, 161, 0, 0]
    threat_rook_mg = [0, 3, 38, 38, 0, 51, 0, 0]
    threat_rook_eg = [0, 44, 71, 61, 38, 38, 0, 0]

    def score_array(name: str, mg_defaults: list[int], eg_defaults: list[int]) -> None:
        entries = [
            score(
                value(f"{name}Mg[{idx}]", mg),
                value(f"{name}Eg[{idx}]", eg),
            )
            for idx, (mg, eg) in enumerate(zip(mg_defaults, eg_defaults))
        ]
        lines.append(line(f"constexpr Score {name}[PIECE_TYPE_NB] = {{"))
        lines.append(line("  " + ", ".join(entries)))
        lines.append(line("};"))
        lines.append("")

    score_array("ThreatByMinor", threat_minor_mg, threat_minor_eg)
    score_array("ThreatByRook", threat_rook_mg, threat_rook_eg)

    pawn_mg = [[48, 64], [72, 96], [120, 160]]
    pawn_eg = [[32, 48], [48, 64], [80, 110]]
    lines.append(line("constexpr Score PawnThreat[3][2] = {"))
    for piece_class, (mg_row, eg_row) in enumerate(zip(pawn_mg, pawn_eg)):
        entries = [
            score(
                value(f"PawnThreatMg[{piece_class}][{idx}]", mg),
                value(f"PawnThreatEg[{piece_class}][{idx}]", eg),
            )
            for idx, (mg, eg) in enumerate(zip(mg_row, eg_row))
        ]
        comma = "," if piece_class != len(pawn_mg) - 1 else ""
        lines.append(line("  { " + ", ".join(entries) + " }" + comma))
    lines.append(line("};"))
    lines.append("")

    lines.append(
        line("constexpr Score Hanging = "
        + score(value("HangingMg", 69), value("HangingEg", 36))
        + ";")
    )
    lines.append("")

    scalar_defaults = [
        ("KingAttackCountWeight", 69),
        ("MultiColorAttackPenalty", 150),
        ("WeakKingRingPenalty", 80),
        ("UndefendedKingRingPenalty", 100),
        ("RookCheckPenalty", 1080),
        ("QueenCheckPenalty", 780),
        ("BishopCheckPenalty", 635),
        ("KnightCheckPenalty", 790),
        ("UnsafeCheckPenalty", 148),
        ("NoPawnShieldPenalty", 120),
        ("WeakPawnShieldPenalty", 40),
        ("NoQueenSafetyReduction", 873),
        ("DangerThreshold", 100),
        ("DangerDivisor", 4096),
        ("DangerMaxPenalty", 1000),
        ("SafetyEgDivisor", 16),
        ("MidgameLimit", 30516),
        ("EndgameLimit", 7830),
    ]
    for name, default in scalar_defaults:
        lines.append(line(f"constexpr int {name} = {value(name, default)};"))

    lines.append("")
    lines.extend(
        [
            line("inline Score mobility_bonus(PieceType pt, int mob) {"),
            line("    const int max_idx[] = {8, 13, 14, 27};"),
            line("    return MobilityBonus[pt - KNIGHT][std::min(mob, max_idx[pt - KNIGHT])];"),
            line("}"),
            "",
            line("inline Score rook_on_file_bonus(bool open) {"),
            line("    return RookOnFile[open];"),
            line("}"),
            "",
            line("inline Score king_protector_bonus() {"),
            line("    return KingProtector;"),
            line("}"),
            "",
            line("inline Score king_in_check_penalty() {"),
            line("    return KingInCheckPenalty;"),
            line("}"),
            "",
            line("inline Score threat_by_minor(PieceType pt) {"),
            line("    return ThreatByMinor[pt];"),
            line("}"),
            "",
            line("inline Score threat_by_rook(PieceType pt) {"),
            line("    return ThreatByRook[pt];"),
            line("}"),
            "",
            line("inline Score pawn_threat_bonus(int pieceClass, bool weak) {"),
            line("    return PawnThreat[pieceClass][weak];"),
            line("}"),
            "",
            line("inline Score hanging_bonus() {"),
            line("    return Hanging;"),
            line("}"),
            "",
            line("#undef S"),
        ]
    )
    return "\n".join(lines) + "\n"


def export_state(args: argparse.Namespace) -> int:
    state = load_state(Path(args.state))
    if not state:
        raise ValueError(f"No state found: {args.state}")
    params = state.get("parameters")
    if not isinstance(params, list) or not params:
        raise ValueError(f"State has no parameters: {args.state}")

    params_lines = []
    cxx_lines = [
        "void Tune::read_results() {",
        f"    // Exported from {args.state} at iteration {state.get('iteration', 0)}",
    ]
    for param in params:
        value = rounded_theta(param)
        name = str(param["name"])
        params_lines.append(
            f"{name},{value},{int(float(param['minimum']))},"
            f"{int(float(param['maximum']))},{float(param['c_end']):.6g},"
            f"{float(param['r_end']):.6g}"
        )
        cxx_lines.append(f'    tune_results()["{cxx_string(name)}"] = {value};')
    cxx_lines.append("}")

    if args.params_out:
        write_export(args.params_out, "\n".join(params_lines) + "\n")
    if args.cxx_out:
        write_export(args.cxx_out, "\n".join(cxx_lines) + "\n")
    if args.constexpr_out:
        write_export(
            args.constexpr_out,
            export_constexpr_block(params, args.state, state.get("iteration", 0)),
        )
    wrote_stdout = (
        args.params_out == "-"
        or args.cxx_out == "-"
        or args.constexpr_out == "-"
    )
    if not args.params_out and not args.cxx_out and not args.constexpr_out:
        print("\n".join(cxx_lines))
    elif not wrote_stdout:
        print(
            f"Exported {len(params)} params from iteration "
            f"{state.get('iteration', 0)}"
        )
    return 0


def build_match_config(
    args: argparse.Namespace,
    *,
    plus_options: dict[str, int | bool | str],
    minus_options: dict[str, int | bool | str],
) -> MatchConfig:
    engine = Path(args.engine).resolve()
    arbiter = Path(args.arbiter).resolve() if args.arbiter else None
    return MatchConfig(
        engine1=EngineConfig(engine, plus_options, args.hash, args.threads),
        engine2=EngineConfig(engine, minus_options, args.hash, args.threads),
        arbiter=EngineConfig(arbiter, {}, args.arbiter_hash, 1) if arbiter else None,
        limit_kind=args.limit_kind,
        limit_value=args.limit_value,
        base_time_ms=args.tc,
        increment_ms=args.inc,
        timeout=args.timeout,
        margin_ms=args.margin,
        max_plies=args.max_plies,
        opening_plies=args.opening_plies,
        fens=load_fens(args.fens),
        workers=args.workers,
        show_moves=False,
    )


def result_from_summary(summary: dict[str, Any]) -> int:
    return int(summary["engine1_wins"]) - int(summary["engine2_wins"])


def summarize_step(summary: dict[str, Any]) -> str:
    return (
        f"WDL {summary['engine1_wins']}/"
        f"{summary['draws']}/{summary['engine2_wins']} "
        f"score {100.0 * summary['engine1_score']:.2f}%"
    )


def run_spsa(args: argparse.Namespace) -> int:
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.pairs < 1:
        raise ValueError("--pairs must be at least 1")
    if args.a is None:
        args.a = 0.1 * args.iterations
    if args.a < 0:
        raise ValueError("--A must be >= 0")

    base_options = load_options(args.options, args.option)
    params = parse_parameters(
        Path(args.params),
        iterations=args.iterations,
        a=args.a,
        alpha=args.alpha,
        gamma=args.gamma,
    )
    state_path = Path(args.state)
    state = load_state(state_path) if args.resume else {}
    updates_completed, history, parameter_history = compact_state_history(state)
    start_iteration = restore_parameters(params, state)
    if start_iteration >= args.iterations:
        print("SPSA already complete.")
        return 0

    rng = random.Random(args.seed)
    for _ in range(updates_completed):
        for _param in params:
            rng.choice((-1, 1))

    started_at = state.get(
        "started_at",
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    iteration = start_iteration
    while iteration < args.iterations:
        batch_pairs = min(args.pairs, args.iterations - iteration)
        steps = make_steps(
            params,
            iteration=iteration,
            a=args.a,
            alpha=args.alpha,
            gamma=args.gamma,
            rng=rng,
        )
        plus_options = options_from_steps(base_options, steps, plus=True)
        minus_options = options_from_steps(base_options, steps, plus=False)
        config = build_match_config(
            args,
            plus_options=plus_options,
            minus_options=minus_options,
        )
        match_result = play_match(
            config,
            pairs=batch_pairs,
            seed=args.seed + iteration,
            continue_on_error=args.continue_on_error,
        )
        summary = match_result.summary
        result = result_from_summary(summary)
        update_parameters(params, steps, result)
        iteration += batch_pairs
        updates_completed += 1

        entry = {
            "iteration": iteration,
            "pairs": batch_pairs,
            "result": result,
            "summary": summary,
        }
        history.append(entry)
        if len(history) > MAX_HISTORY_ENTRIES:
            del history[:-MAX_HISTORY_ENTRIES]
        if updates_completed % PARAMETER_HISTORY_INTERVAL == 0:
            parameter_history.append(
                {
                    "iteration": iteration,
                    "theta": {param.name: param.theta for param in params},
                }
            )
            if len(parameter_history) > MAX_PARAMETER_HISTORY_ENTRIES:
                del parameter_history[:-MAX_PARAMETER_HISTORY_ENTRIES]

        state = {
            "format": 2,
            "started_at": started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iteration": iteration,
            "iterations": args.iterations,
            "pairs": args.pairs,
            "updates_completed": updates_completed,
            "seed": args.seed,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "A": args.a,
            "parameters": [asdict(param) for param in params],
            "history": history,
            "parameter_history": parameter_history,
        }
        write_state(state_path, state)

        if iteration % args.progress_interval == 0 or iteration == args.iterations:
            print(
                f"[{iteration}/{args.iterations}] {summarize_step(summary)} "
                f"gradient {result:+d}",
                flush=True,
            )
        if args.param_interval and (
            iteration % args.param_interval == 0 or iteration == args.iterations
        ):
            for row in parameter_rows(
                params,
                iteration=iteration,
                a=args.a,
                alpha=args.alpha,
                gamma=args.gamma,
            ):
                print(row, flush=True)

    return 0


def build_tune_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stockfish-style SPSA tuning against a 4PC UCI engine"
    )
    parser.add_argument("--engine", required=True, help="stockfish_4pc executable")
    parser.add_argument("--params", required=True, help="SPSA parameter file")
    parser.add_argument("--state", default="spsa_state.json", help="resumable state file")
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument(
        "--pairs",
        type=int,
        default=1,
        help="paired openings per SPSA update",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--A", dest="a", type=float)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--options", default="", help="base engine options JSON")
    parser.add_argument("--option", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--arbiter", default="")
    parser.add_argument("--arbiter-hash", type=int, default=16)
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--nodes", type=int)
    limit.add_argument("--depth", type=int)
    limit.add_argument("--movetime", type=int)
    limit.add_argument("--tc", type=int, help="base time in milliseconds")
    parser.add_argument("--inc", type=int, default=0)
    parser.add_argument("--margin", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-plies", type=int, default=1000)
    parser.add_argument("--opening-plies", type=int, default=0)
    parser.add_argument("--fens", default="")
    parser.add_argument(
        "--allow-startpos",
        action="store_true",
        help="allow tuning every pair from startpos; useful only for smoke tests",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1,
        help="print one summary line every N SPSA iterations",
    )
    parser.add_argument(
        "--param-interval",
        type=int,
        default=0,
        help="also print all parameter rows every N SPSA iterations; 0 disables",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def build_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export tuned SPSA values from a state file"
    )
    parser.add_argument("--state", required=True, help="SPSA state JSON")
    parser.add_argument(
        "--cxx-out",
        default="",
        help="write Tune::read_results() replacement snippet",
    )
    parser.add_argument(
        "--params-out",
        default="",
        help="write rounded params file for continuing/testing",
    )
    parser.add_argument(
        "--constexpr-out",
        default="",
        help="write constexpr evaluate.cpp parameter block",
    )
    return parser


def normalize_args(args: argparse.Namespace) -> None:
    if args.tc is not None:
        args.limit_kind = "clock"
        args.limit_value = 0
    elif args.movetime is not None:
        args.limit_kind = "movetime"
        args.limit_value = args.movetime
        args.tc = 0
    elif args.depth is not None:
        args.limit_kind = "depth"
        args.limit_value = args.depth
        args.tc = 0
    else:
        args.limit_kind = "nodes"
        args.limit_value = args.nodes if args.nodes is not None else 10000
        args.tc = 0
    if args.limit_kind == "clock" and args.tc <= 0:
        raise ValueError("--tc must be greater than 0")
    if args.limit_kind != "clock" and args.inc:
        raise ValueError("--inc requires --tc")
    if args.threads < 1 or args.workers < 1:
        raise ValueError("--threads and --workers must be at least 1")
    if not args.fens and args.opening_plies <= 0 and not args.allow_startpos:
        raise ValueError(
            "SPSA needs opening diversity: pass --fens PATH, or use "
            "--allow-startpos only for smoke tests"
        )
    if args.opening_plies > 0 and not args.arbiter:
        raise ValueError(
            "--opening-plies requires --arbiter with legalmoves support for this engine"
        )
    if args.progress_interval < 1:
        raise ValueError("--progress-interval must be at least 1")
    if args.param_interval < 0:
        raise ValueError("--param-interval must be >= 0")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        parser = build_export_parser()
        args = parser.parse_args(sys.argv[2:])
        try:
            return export_state(args)
        except (OSError, ValueError) as exc:
            print(f"spsa export: error: {exc}", file=sys.stderr)
            return 2

    parser = build_tune_parser()
    args = parser.parse_args()
    try:
        normalize_args(args)
        return run_spsa(args)
    except KeyboardInterrupt:
        print(
            "Interrupted; state is saved after each completed SPSA update.",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, TimeoutError, EngineError) as exc:
        print(f"spsa: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
