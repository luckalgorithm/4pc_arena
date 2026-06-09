#!/usr/bin/env python3
"""Run paired matches between two 4-player-chess UCI engines."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TURN_ORDER = ("red", "blue", "yellow", "green")
TEAM_BY_COLOR = {
    "red": "ry",
    "yellow": "ry",
    "blue": "bg",
    "green": "bg",
}


class EngineError(RuntimeError):
    pass


class MatchInterrupted(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineConfig:
    path: Path
    options: dict[str, int | bool | str]
    hash_mb: int
    threads: int


@dataclass(frozen=True)
class StartPosition:
    fen: str | None
    opening_moves: list[str]
    source: str


@dataclass(frozen=True)
class GameTask:
    pair_index: int
    engine1_team: str
    start: StartPosition

    @property
    def game_id(self) -> str:
        return f"pair{self.pair_index:04d}-{self.engine1_team}"


@dataclass
class MatchConfig:
    engine1: EngineConfig
    engine2: EngineConfig
    arbiter: EngineConfig | None
    limit_kind: str
    limit_value: int
    base_time_ms: int
    increment_ms: int
    timeout: float
    margin_ms: int
    max_plies: int
    opening_plies: int
    fens: list[str]
    workers: int
    show_moves: bool
    out: Path | None


@dataclass
class SearchResult:
    bestmove: str | None
    info: dict[str, Any]
    elapsed_ms: int
    result: str | None = None


@dataclass
class MatchResult:
    records: list[dict[str, Any]]
    summary: dict[str, Any]
    engine1_name: str
    engine2_name: str
    interrupted: bool


class ActiveEngines:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engines: set[UciEngine] = set()
        self._stopped = False

    def launch(
        self, engine: UciEngine, command: list[str], **kwargs: Any
    ) -> subprocess.Popen[str]:
        with self._lock:
            if self._stopped:
                raise MatchInterrupted("Match interrupted")
            proc = subprocess.Popen(command, **kwargs)
            self._engines.add(engine)
            return proc

    def remove(self, engine: UciEngine) -> None:
        with self._lock:
            self._engines.discard(engine)

    def close_all(self) -> None:
        with self._lock:
            self._stopped = True
            engines = list(self._engines)
        for engine in engines:
            engine.close(force=True)


class UciEngine:
    def __init__(
        self,
        config: EngineConfig,
        *,
        label: str,
        active_engines: ActiveEngines | None = None,
    ) -> None:
        self.config = config
        self.label = label
        self.name = config.path.name
        self._active_engines = active_engines
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._closed = False
        self._close_lock = threading.Lock()
        popen_args = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.DEVNULL,
            "text": True,
            "bufsize": 1,
            "cwd": str(config.path.parent),
            "env": os.environ.copy(),
        }
        if active_engines is None:
            self.proc = subprocess.Popen([str(config.path)], **popen_args)
        else:
            self.proc = active_engines.launch(self, [str(config.path)], **popen_args)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        try:
            self._reader.start()
            self.send("uci")
            self._wait_for_uciok(30)
            self.send(f"setoption name Hash value {config.hash_mb}")
            self.send(f"setoption name Threads value {config.threads}")
            for name, value in sorted(config.options.items()):
                self.send(f"setoption name {name} value {option_text(value)}")
            self.sync()
        except BaseException:
            self.close(force=True)
            raise

    def __enter__(self) -> UciEngine:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _read_loop(self) -> None:
        assert self.proc.stdout is not None
        try:
            for raw in self.proc.stdout:
                line = raw.strip()
                if line:
                    self._queue.put(line)
        finally:
            self._queue.put(None)

    def send(self, command: str) -> None:
        if self._closed or self.proc.stdin is None:
            raise EngineError(f"{self.label} is closed")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def read_line(self, timeout: float) -> str:
        try:
            line = self._queue.get(timeout=timeout)
        except queue.Empty as exc:
            if self.proc.poll() is not None:
                raise EngineError(
                    f"{self.label} exited with status {self.proc.returncode}"
                ) from exc
            raise TimeoutError(f"{self.label} timed out") from exc
        if line is None:
            raise EngineError(
                f"{self.label} exited with status {self.proc.poll()}"
            )
        return line

    def _wait_for_uciok(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.label} timed out waiting for uciok")
            line = self.read_line(remaining)
            if line.startswith("id name "):
                self.name = line.removeprefix("id name ").strip() or self.name
            if line == "uciok":
                return

    def wait_for(self, prefix: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.label} timed out waiting for {prefix}")
            line = self.read_line(remaining)
            if line.startswith(prefix):
                return line

    def sync(self) -> None:
        self.send("isready")
        self.wait_for("readyok", 30)

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.sync()

    def set_position(self, moves: list[str], fen: str | None) -> None:
        command = f"position fen {fen}" if fen else "position startpos"
        if moves:
            command += " moves " + " ".join(moves)
        self.send(command)

    def legal_moves(self, moves: list[str], fen: str | None) -> list[str]:
        self.set_position(moves, fen)
        self.send("legalmoves")
        parts = self.wait_for("legalmoves", 30).split(maxsplit=1)
        return [] if len(parts) == 1 else [canonical_move(move) for move in parts[1].split()]

    def game_result(
        self,
        moves: list[str],
        fen: str | None,
        timeout: float = 30,
    ) -> str:
        self.set_position(moves, fen)
        self.send("gameresult")
        parts = self.wait_for("gameresult", timeout).split(maxsplit=1)
        return normalize_result(parts[1] if len(parts) == 2 else "unknown")

    def search(
        self,
        moves: list[str],
        fen: str | None,
        go_command: str,
        timeout: float,
    ) -> SearchResult:
        self.set_position(moves, fen)
        started = time.monotonic()
        self.send(go_command)
        deadline = started + timeout
        info: dict[str, Any] = {}
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    self.send("stop")
                except (BrokenPipeError, OSError, EngineError):
                    pass
                raise TimeoutError(f"{self.label} search timed out")
            try:
                line = self.read_line(min(remaining, 1.0))
            except TimeoutError:
                continue
            if line.startswith("info string Game completed."):
                elapsed = int(round((time.monotonic() - started) * 1000))
                return SearchResult(
                    None,
                    info,
                    elapsed,
                    completed_result(line),
                )
            if line.startswith("info "):
                info.update(parse_info(line))
            elif line.startswith("bestmove"):
                parts = line.split()
                move = parts[1] if len(parts) > 1 else None
                if move in (None, "(none)", "0000"):
                    move = None
                elif move is not None:
                    move = canonical_move(move)
                elapsed = int(round((time.monotonic() - started) * 1000))
                return SearchResult(move, info, elapsed)

    def close(self, *, force: bool = False) -> None:
        with self._close_lock:
            if self._closed and not force:
                return
            self._closed = True
        try:
            if force:
                if self.proc.poll() is None:
                    self.proc.terminate()
            else:
                try:
                    if self.proc.stdin is not None:
                        self.proc.stdin.write("quit\n")
                        self.proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                self.proc.wait(timeout=0.5 if force else 2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        finally:
            if self._active_engines is not None:
                self._active_engines.remove(self)


def option_text(value: int | bool | str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def canonical_move(move: str) -> str:
    return move.strip().lower().replace("-", "").replace("=", "")


def normalize_result(result: str) -> str:
    key = result.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if key in {"ongoing", "unfinished", "*"}:
        return "ongoing"
    if key in {"draw", "1/21/2", "1/2-1/2"}:
        return "draw"
    if key in {"winry", "rywin"}:
        return "ry_win"
    if key in {"winbg", "bgwin"}:
        return "bg_win"
    return result.strip().lower()


def completed_result(line: str) -> str:
    lowered = line.lower()
    if "ry won" in lowered:
        return "ry_win"
    if "bg won" in lowered:
        return "bg_win"
    if "stalemate" in lowered or "draw" in lowered:
        return "draw"
    return "unknown"


def parse_info(line: str) -> dict[str, Any]:
    parts = line.split()
    info: dict[str, Any] = {}
    i = 1
    while i < len(parts):
        key = parts[i]
        if key in {"depth", "seldepth", "nodes", "nps", "hashfull", "time"} and i + 1 < len(parts):
            try:
                info[key] = int(parts[i + 1])
            except ValueError:
                pass
            i += 2
        elif key == "score" and i + 2 < len(parts):
            try:
                info["score"] = {"type": parts[i + 1], "value": int(parts[i + 2])}
            except ValueError:
                pass
            i += 3
        elif key == "pv":
            end = i + 1
            while end < len(parts) and parts[end] not in {
                "depth", "seldepth", "nodes", "nps", "hashfull", "time", "score"
            }:
                end += 1
            info["pv"] = parts[i + 1 : end]
            i = end
        else:
            i += 1
    return info


def parse_option_value(value: Any) -> int | bool | str:
    if isinstance(value, (bool, int)):
        return value
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        return text


def load_options(path: str, overrides: list[str]) -> dict[str, int | bool | str]:
    values: dict[str, int | bool | str] = {}
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "values" in raw:
            raw = raw["values"]
        if not isinstance(raw, dict):
            raise ValueError(f"Option file must contain an object: {path}")
        values.update({str(k): parse_option_value(v) for k, v in raw.items()})
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Expected NAME=VALUE, got: {item}")
        name, value = item.split("=", 1)
        values[name] = parse_option_value(value)
    return values


def load_fens(path: str) -> list[str]:
    if not path:
        return []
    fens: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            fens.append(line)
    if not fens:
        raise ValueError(f"No FENs found in {path}")
    return fens


def fen_turn_index(fen: str | None) -> int:
    if not fen:
        return 0
    token = fen.split("-", 1)[0].strip().lower()
    return {"r": 0, "b": 1, "y": 2, "g": 3}.get(token[:1], 0)


def engine_signature(config: EngineConfig) -> dict[str, Any]:
    stat = config.path.stat()
    return {
        "path": str(config.path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "options": config.options,
        "hash_mb": config.hash_mb,
        "threads": config.threads,
    }


def generate_start(
    config: MatchConfig,
    rng: random.Random,
    arbiter: UciEngine | None = None,
) -> StartPosition:
    if not config.fens:
        return StartPosition(None, [], "startpos")
    fen = rng.choice(config.fens)
    moves: list[str] = []
    if config.opening_plies <= 0:
        return StartPosition(fen, moves, "fen")
    if arbiter is None:
        opening_config = config.arbiter or config.engine1
        with UciEngine(opening_config, label="opening-rules") as owned:
            return generate_start(config, rng, owned)
    for _ in range(config.opening_plies):
        legal = arbiter.legal_moves(moves, fen)
        if not legal:
            break
        moves.append(rng.choice(legal))
    return StartPosition(fen, moves, "fen" if fen else "startpos")


def go_command(config: MatchConfig, clocks: dict[str, int]) -> str:
    if config.limit_kind == "clock":
        return (
            f"go wtime {max(0, clocks['ry'])} btime {max(0, clocks['bg'])} "
            f"winc {config.increment_ms} binc {config.increment_ms}"
        )
    return f"go {config.limit_kind} {config.limit_value}"


def winner_for_failure(team: str) -> str:
    return "bg_win" if team == "ry" else "ry_win"


def engine1_score(result: str, engine1_team: str) -> float:
    if result == "draw":
        return 0.5
    if result == "ry_win":
        return 1.0 if engine1_team == "ry" else 0.0
    if result == "bg_win":
        return 1.0 if engine1_team == "bg" else 0.0
    return 0.5


def current_engine_result(
    engine: UciEngine,
    moves: list[str],
    fen: str | None,
    timeout: float,
) -> str:
    try:
        result = engine.game_result(moves, fen, timeout=min(timeout, 1.0))
    except TimeoutError:
        return "unknown"
    return result if result in {"ry_win", "bg_win", "draw"} else "unknown"


def play_game(
    config: MatchConfig,
    task: GameTask,
    stop_event: threading.Event,
    active_engines: ActiveEngines,
) -> dict[str, Any]:
    if stop_event.is_set():
        raise MatchInterrupted("Match interrupted")
    engine1_is_ry = task.engine1_team == "ry"
    ry_config = config.engine1 if engine1_is_ry else config.engine2
    bg_config = config.engine2 if engine1_is_ry else config.engine1
    ry_number = 1 if engine1_is_ry else 2
    bg_number = 2 if engine1_is_ry else 1
    moves = list(task.start.opening_moves)
    searches: list[dict[str, Any]] = []
    clocks = {"ry": config.base_time_ms, "bg": config.base_time_ms}
    start_turn = fen_turn_index(task.start.fen)
    record: dict[str, Any] = {
        "game_id": task.game_id,
        "pair": task.pair_index,
        "engine1_team": task.engine1_team,
        "fen": task.start.fen,
        "opening": list(task.start.opening_moves),
        "start_source": task.start.source,
        "moves": list(moves),
        "result": "draw",
        "engine1_score": 0.5,
        "termination": "unknown",
        "searches": searches,
    }

    with contextlib.ExitStack() as stack:
        ry_engine = stack.enter_context(UciEngine(
            ry_config, label=f"{task.game_id}:RY", active_engines=active_engines
        ))
        bg_engine = stack.enter_context(UciEngine(
            bg_config, label=f"{task.game_id}:BG", active_engines=active_engines
        ))
        arbiter = (
            stack.enter_context(UciEngine(
                config.arbiter,
                label=f"{task.game_id}:arbiter",
                active_engines=active_engines,
            ))
            if config.arbiter is not None
            else None
        )
        record["engine_names"] = {
            "engine1": ry_engine.name if engine1_is_ry else bg_engine.name,
            "engine2": bg_engine.name if engine1_is_ry else ry_engine.name,
        }
        engines = [ry_engine, bg_engine]
        if arbiter is not None:
            engines.append(arbiter)
        for engine in engines:
            engine.new_game()

        while len(moves) < config.max_plies:
            if stop_event.is_set():
                raise MatchInterrupted("Match interrupted")
            color = TURN_ORDER[(start_turn + len(moves)) % 4]
            team = TEAM_BY_COLOR[color]
            engine = ry_engine if team == "ry" else bg_engine
            rules_engine = arbiter or engine
            if arbiter is not None:
                result = arbiter.game_result(moves, task.start.fen)
                if result != "ongoing":
                    record.update({
                        "moves": list(moves),
                        "result": result,
                        "engine1_score": engine1_score(result, task.engine1_team),
                        "termination": "game_result",
                        "final_clocks_ms": clocks,
                    })
                    return record

            legal = rules_engine.legal_moves(moves, task.start.fen)
            if not legal:
                if arbiter is None:
                    try:
                        completed = engine.search(
                            moves,
                            task.start.fen,
                            "go depth 1",
                            min(config.timeout, 30.0),
                        )
                    except TimeoutError:
                        completed = SearchResult(None, {}, 0)
                    if completed.result in {"ry_win", "bg_win", "draw"}:
                        result = completed.result
                        record.update({
                            "moves": list(moves),
                            "result": result,
                            "engine1_score": engine1_score(result, task.engine1_team),
                            "termination": "completed_no_legal_moves",
                            "final_clocks_ms": clocks,
                        })
                        return record
                    result = current_engine_result(
                        engine, moves, task.start.fen, config.timeout
                    )
                    if result in {"ry_win", "bg_win", "draw"}:
                        record.update({
                            "moves": list(moves),
                            "result": result,
                            "engine1_score": engine1_score(result, task.engine1_team),
                            "termination": "current_engine_game_result",
                            "final_clocks_ms": clocks,
                        })
                        return record
                record.update({
                    "moves": list(moves),
                    "result": "draw",
                    "engine1_score": 0.5,
                    "termination": "no_legal_moves",
                    "final_clocks_ms": clocks,
                })
                return record

            engine_number = ry_number if team == "ry" else bg_number

            if len(legal) == 1:
                search = SearchResult(legal[0], {}, 0)
            else:
                try:
                    search = engine.search(
                        moves,
                        task.start.fen,
                        go_command(config, clocks),
                        config.timeout,
                    )
                except TimeoutError:
                    result = winner_for_failure(team)
                    record.update({
                        "moves": list(moves),
                        "result": result,
                        "engine1_score": engine1_score(result, task.engine1_team),
                        "termination": f"engine{engine_number}_timeout",
                        "final_clocks_ms": clocks,
                    })
                    return record

            if search.result in {"ry_win", "bg_win", "draw"}:
                result = search.result
                record.update({
                    "moves": list(moves),
                    "result": result,
                    "engine1_score": engine1_score(result, task.engine1_team),
                    "termination": "engine_reported_result",
                    "final_clocks_ms": clocks,
                })
                return record

            if search.bestmove is None and arbiter is None:
                result = current_engine_result(
                    engine, moves, task.start.fen, config.timeout
                )
                if result in {"ry_win", "bg_win", "draw"}:
                    record.update({
                        "moves": list(moves),
                        "result": result,
                        "engine1_score": engine1_score(result, task.engine1_team),
                        "termination": "current_engine_game_result",
                        "final_clocks_ms": clocks,
                    })
                    return record

            if config.limit_kind == "clock":
                clocks[team] -= search.elapsed_ms
                if clocks[team] < -config.margin_ms:
                    result = winner_for_failure(team)
                    record.update({
                        "moves": list(moves),
                        "result": result,
                        "engine1_score": engine1_score(result, task.engine1_team),
                        "termination": f"engine{engine_number}_time_loss",
                        "final_clocks_ms": clocks,
                    })
                    return record
                clocks[team] += config.increment_ms

            if search.bestmove not in legal:
                result = winner_for_failure(team)
                record.update({
                    "moves": list(moves),
                    "result": result,
                    "engine1_score": engine1_score(result, task.engine1_team),
                    "termination": f"engine{engine_number}_illegal_move",
                    "illegal_move": search.bestmove,
                    "final_clocks_ms": clocks,
                })
                return record

            moves.append(search.bestmove)
            search_record = {
                "ply": len(moves),
                "color": color,
                "team": team,
                "engine": engine_number,
                "move": search.bestmove,
                "elapsed_ms": search.elapsed_ms,
                "clock_ms": clocks[team] if config.limit_kind == "clock" else None,
                **search.info,
            }
            searches.append(search_record)
            if config.show_moves:
                print_move(task.game_id, search_record)

    record.update({
        "moves": list(moves),
        "result": "draw",
        "engine1_score": 0.5,
        "termination": "max_plies",
        "final_clocks_ms": clocks,
    })
    return record


def score_text(info: dict[str, Any]) -> str:
    score = info.get("score")
    if not isinstance(score, dict):
        return "--"
    if score.get("type") == "mate":
        return f"M{score.get('value')}"
    return f"{int(score.get('value', 0)):+d}"


def print_move(game_id: str, info: dict[str, Any]) -> None:
    print(
        f"{game_id} ply {info['ply']:03d} {info['color']:<6} "
        f"E{info['engine']} {info['move']:<6} "
        f"score {score_text(info):>6} depth {info.get('depth', '--'):>3} "
        f"time {info['elapsed_ms']:>6}ms nodes {info.get('nodes', '--')}",
        flush=True,
    )


def summarize(records: list[dict[str, Any]], games_expected: int) -> dict[str, Any]:
    wins1 = sum(float(r.get("engine1_score", 0.5)) == 1.0 for r in records)
    wins2 = sum(float(r.get("engine1_score", 0.5)) == 0.0 for r in records)
    draws = len(records) - wins1 - wins2
    score = (wins1 + draws / 2) / len(records) if records else 0.5
    if score >= 1.0:
        elo = math.inf
    elif score <= 0.0:
        elo = -math.inf
    else:
        elo = 400 * math.log10(score / (1 - score))
    by_id = {str(r["game_id"]): r for r in records}
    pair_scores: list[float] = []
    for pair in range(1, games_expected // 2 + 1):
        ry = by_id.get(f"pair{pair:04d}-ry")
        bg = by_id.get(f"pair{pair:04d}-bg")
        if ry and bg:
            pair_scores.append(
                (float(ry["engine1_score"]) + float(bg["engine1_score"])) / 2
            )
    if len(pair_scores) > 1:
        mean = sum(pair_scores) / len(pair_scores)
        variance = sum((x - mean) ** 2 for x in pair_scores) / (len(pair_scores) - 1)
        error = math.sqrt(variance / len(pair_scores))
        confidence = [max(0.0, mean - 1.96 * error), min(1.0, mean + 1.96 * error)]
    else:
        confidence = None
    terminations: dict[str, int] = {}
    for record in records:
        key = str(record.get("termination", "unknown"))
        terminations[key] = terminations.get(key, 0) + 1
    return {
        "games_completed": len(records),
        "games_expected": games_expected,
        "engine1_wins": wins1,
        "engine2_wins": wins2,
        "draws": draws,
        "engine1_score": score,
        "engine2_score": 1 - score,
        "elo": elo,
        "pairs_completed": len(pair_scores),
        "paired_95ci": confidence,
        "engine1_illegal_moves": terminations.get("engine1_illegal_move", 0),
        "engine2_illegal_moves": terminations.get("engine2_illegal_move", 0),
        "engine1_time_losses": sum(v for k, v in terminations.items() if k == "engine1_time_loss"),
        "engine2_time_losses": sum(v for k, v in terminations.items() if k == "engine2_time_loss"),
        "engine1_timeouts": sum(v for k, v in terminations.items() if k == "engine1_timeout"),
        "engine2_timeouts": sum(v for k, v in terminations.items() if k == "engine2_timeout"),
        "runner_errors": terminations.get("runner_error", 0),
        "terminations": terminations,
    }


def elo_text(elo: float) -> str:
    return (
        "+inf" if elo == math.inf
        else "-inf" if elo == -math.inf
        else f"{elo:+.1f}"
    )


def live_event(record: dict[str, Any], name1: str, name2: str) -> str:
    termination = str(record.get("termination", ""))
    names = {"engine1": name1, "engine2": name2}
    for engine_key, engine_name in names.items():
        if termination == f"{engine_key}_illegal_move":
            move = record.get("illegal_move")
            return f" [{engine_name} reported illegal move: {move}]"
        if termination == f"{engine_key}_time_loss":
            return f" [{engine_name} lost on time]"
        if termination == f"{engine_key}_timeout":
            return f" [{engine_name} timed out]"
    if termination == "runner_error":
        return f" [Runner error: {record.get('error', 'unknown error')}]"
    return ""


def print_summary(
    summary: dict[str, Any],
    name1: str,
    name2: str,
    record: dict[str, Any] | None = None,
) -> None:
    elo = summary["elo"]
    event = live_event(record, name1, name2) if record is not None else ""
    print(
        f"[{summary['games_completed']}/{summary['games_expected']}] "
        f"+{summary['engine1_wins']} ={summary['draws']} "
        f"-{summary['engine2_wins']} | "
        f"{100 * summary['engine1_score']:.2f}% | Elo {elo_text(elo)}"
        f"{event}",
        flush=True,
    )


def print_final_summary(
    summary: dict[str, Any],
    name1: str,
    name2: str,
    *,
    interrupted: bool,
) -> None:
    state = "stopped" if interrupted else "complete"
    print(f"\nMatch {state}: {summary['games_completed']}/{summary['games_expected']} games")
    print(f"Engine 1 ({name1}): {summary['engine1_wins']} wins")
    print(f"Engine 2 ({name2}): {summary['engine2_wins']} wins")
    print(f"Draws: {summary['draws']}")
    print(
        f"Score: {100 * summary['engine1_score']:.2f}% - "
        f"{100 * summary['engine2_score']:.2f}%"
    )
    print(f"Elo: {elo_text(summary['elo'])} from Engine 1's perspective")

    exceptional = [
        (
            "Illegal moves",
            summary["engine1_illegal_moves"],
            summary["engine2_illegal_moves"],
        ),
        (
            "Time losses",
            summary["engine1_time_losses"],
            summary["engine2_time_losses"],
        ),
        (
            "Timeouts",
            summary["engine1_timeouts"],
            summary["engine2_timeouts"],
        ),
    ]
    for label, count1, count2 in exceptional:
        if count1 or count2:
            parts = []
            if count1:
                parts.append(f"{name1}: {count1}")
            if count2:
                parts.append(f"{name2}: {count2}")
            print(f"{label}: {' | '.join(parts)}")
    if summary["runner_errors"]:
        print(f"Runner errors: {summary['runner_errors']}")


def engine_names(
    records: list[dict[str, Any]], engine1: EngineConfig, engine2: EngineConfig
) -> tuple[str, str]:
    for record in reversed(records):
        names = record.get("engine_names")
        if isinstance(names, dict):
            return (
                str(names.get("engine1", engine1.path.name)),
                str(names.get("engine2", engine2.path.name)),
            )
    return engine1.path.name, engine2.path.name


def probe_engine_name(config: EngineConfig, label: str) -> str:
    with UciEngine(config, label=label) as engine:
        return engine.name


def control_text(config: MatchConfig) -> str:
    if config.limit_kind == "clock":
        return f"tc {config.base_time_ms}ms + {config.increment_ms}ms"
    if config.limit_kind == "movetime":
        return f"movetime {config.limit_value}ms"
    return f"{config.limit_kind} {config.limit_value}"


def print_start_banner(
    config: MatchConfig,
    games: int,
    name1: str,
    name2: str,
) -> None:
    print(f"Starting match: {name1} (Engine 1) vs {name2} (Engine 2)", flush=True)
    print(
        f"{games} games | {control_text(config)} | "
        f"{config.workers} workers | max {config.max_plies} plies",
        flush=True,
    )
    print("Results, percentage, and Elo are from Engine 1's perspective.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} line {number}") from exc
        if not isinstance(record, dict) or "game_id" not in record:
            raise ValueError(f"Invalid game record in {path} line {number}")
        records.append(record)
    return records


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def sidecar(path: Path, suffix: str) -> Path:
    return path.with_name(path.name + suffix)


def create_schedule(
    config: MatchConfig,
    pairs: int,
    seed: int,
    path: Path | None,
    resume: bool,
) -> list[StartPosition]:
    if resume and path is not None and path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or len(raw) != pairs:
            raise ValueError(f"Opening schedule does not contain {pairs} pairs")
        return [
            StartPosition(item.get("fen"), list(item["opening_moves"]), item["source"])
            for item in raw
        ]
    rng = random.Random(seed)
    if config.opening_plies and config.fens:
        if config.arbiter is not None:
            with UciEngine(config.arbiter, label="opening-rules") as arbiter:
                starts = [generate_start(config, rng, arbiter) for _ in range(pairs)]
        else:
            starts = []
            with UciEngine(
                config.engine1,
                label="opening-rules-engine1",
            ) as engine1_rules, UciEngine(
                config.engine2,
                label="opening-rules-engine2",
            ) as engine2_rules:
                providers = (engine1_rules, engine2_rules)
                first_provider = rng.randrange(2)
                for pair_index in range(pairs):
                    opening_engine = providers[(first_provider + pair_index) % 2]
                    starts.append(generate_start(config, rng, opening_engine))
    else:
        starts = [generate_start(config, rng) for _ in range(pairs)]
    if path is not None:
        atomic_json(path, [
            {
                "pair": index,
                "fen": start.fen,
                "opening_moves": start.opening_moves,
                "source": start.source,
            }
            for index, start in enumerate(starts, 1)
        ])
    return starts


def validate_engine(config: EngineConfig, label: str) -> None:
    if not config.path.is_file():
        raise ValueError(f"{label} does not exist: {config.path}")
    if not os.access(config.path, os.X_OK):
        raise ValueError(f"{label} is not executable: {config.path}")


def build_tasks(
    starts: list[StartPosition],
    records_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[GameTask]:
    completed = records_by_id or {}
    tasks: list[GameTask] = []
    for pair, start in enumerate(starts, 1):
        team_order = ("ry", "bg") if pair % 2 else ("bg", "ry")
        for team in team_order:
            task = GameTask(pair, team, start)
            if task.game_id not in completed:
                tasks.append(task)
    return tasks


def execute_tasks(
    config: MatchConfig,
    tasks: list[GameTask],
    records_by_id: dict[str, dict[str, Any]],
    *,
    continue_on_error: bool,
    on_record: Any = None,
) -> bool:
    stop_event = threading.Event()
    active_engines = ActiveEngines()

    def run_task(task: GameTask) -> dict[str, Any]:
        return play_game(config, task, stop_event, active_engines)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.workers)
    futures: dict[concurrent.futures.Future[dict[str, Any]], GameTask] = {}
    try:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                if not continue_on_error:
                    raise
                record = {
                    "game_id": task.game_id,
                    "pair": task.pair_index,
                    "engine1_team": task.engine1_team,
                    "fen": task.start.fen,
                    "opening": task.start.opening_moves,
                    "moves": task.start.opening_moves,
                    "result": "draw",
                    "engine1_score": 0.5,
                    "termination": "runner_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            records_by_id[task.game_id] = record
            if on_record is not None:
                on_record(record, records_by_id)
    except KeyboardInterrupt:
        stop_event.set()
        for future in futures:
            future.cancel()
        active_engines.close_all()
        executor.shutdown(wait=True, cancel_futures=True)
        return True
    except BaseException:
        stop_event.set()
        for future in futures:
            future.cancel()
        active_engines.close_all()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
        return False


def play_match(
    config: MatchConfig,
    *,
    pairs: int,
    seed: int,
    continue_on_error: bool = False,
    on_record: Any = None,
) -> MatchResult:
    """Run an in-memory paired match for callers such as tuning tools."""
    for engine, label in (
        (config.engine1, "Engine 1"),
        (config.engine2, "Engine 2"),
    ):
        validate_engine(engine, label)
    if config.arbiter is not None:
        validate_engine(config.arbiter, "Arbiter")

    starts = create_schedule(config, pairs, seed, None, False)
    records_by_id: dict[str, dict[str, Any]] = {}
    interrupted = execute_tasks(
        config,
        build_tasks(starts),
        records_by_id,
        continue_on_error=continue_on_error,
        on_record=on_record,
    )
    records = list(records_by_id.values())
    name1, name2 = engine_names(records, config.engine1, config.engine2)
    summary = summarize(records, pairs * 2)
    summary["engine1_name"] = name1
    summary["engine2_name"] = name2
    return MatchResult(records, summary, name1, name2, interrupted)


def run_match(args: argparse.Namespace) -> int:
    engine1 = EngineConfig(
        Path(args.engine1).resolve(),
        load_options(args.engine1_options, args.engine1_option),
        args.hash1,
        args.threads1,
    )
    engine2 = EngineConfig(
        Path(args.engine2).resolve(),
        load_options(args.engine2_options, args.engine2_option),
        args.hash2,
        args.threads2,
    )
    arbiter = (
        EngineConfig(
            Path(args.arbiter).resolve(),
            load_options(args.arbiter_options, args.arbiter_option),
            args.arbiter_hash,
            1,
        )
        if args.arbiter
        else None
    )
    configs = [(engine1, "Engine 1"), (engine2, "Engine 2")]
    if arbiter is not None:
        configs.append((arbiter, "Arbiter"))
    for config, label in configs:
        validate_engine(config, label)

    out = Path(args.out).resolve() if args.out else None
    config = MatchConfig(
        engine1,
        engine2,
        arbiter,
        args.limit_kind,
        args.limit_value,
        args.tc,
        args.inc,
        args.timeout,
        args.margin,
        args.max_plies,
        args.opening_plies,
        load_fens(args.fens),
        args.workers,
        args.moves,
        out,
    )
    metadata_path = sidecar(out, ".meta.json") if out is not None else None
    schedule_path = sidecar(out, ".schedule.json") if out is not None else None
    summary_path = sidecar(out, ".summary.json") if out is not None else None
    payload = {
        "format": 1,
        "engine1": engine_signature(engine1),
        "engine2": engine_signature(engine2),
        "arbiter": engine_signature(arbiter) if arbiter is not None else "current_engine",
        "pairs": args.pairs,
        "seed": args.seed,
        "limit_kind": args.limit_kind,
        "limit_value": args.limit_value,
        "tc": args.tc,
        "inc": args.inc,
        "margin": args.margin,
        "max_plies": args.max_plies,
        "opening_plies": args.opening_plies,
        "fens": config.fens,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    if args.fresh and out is not None:
        for path in (out, metadata_path, schedule_path, summary_path):
            if path is not None:
                path.unlink(missing_ok=True)
    if out is not None and out.exists() and not args.resume:
        raise ValueError(f"Output exists; use --resume or --fresh: {out}")
    if metadata_path is not None and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint:
            raise ValueError(
                f"Existing match configuration differs: {metadata_path}. "
                "Use the original settings, --fresh, or another --out."
            )
    elif metadata_path is not None:
        atomic_json(metadata_path, {"fingerprint": fingerprint, "config": payload})

    starts = create_schedule(
        config,
        args.pairs,
        args.seed,
        schedule_path,
        out is not None and args.resume and not args.fresh,
    )
    records = (
        load_jsonl(out)
        if out is not None and args.resume and not args.fresh
        else []
    )
    records_by_id = {str(record["game_id"]): record for record in records}
    if records:
        name1, name2 = engine_names(records, engine1, engine2)
    else:
        name1 = probe_engine_name(engine1, "engine1-probe")
        name2 = probe_engine_name(engine2, "engine2-probe")
    print_start_banner(config, args.pairs * 2, name1, name2)
    tasks = build_tasks(starts, records_by_id)

    def on_record(
        record: dict[str, Any],
        current_by_id: dict[str, dict[str, Any]],
    ) -> None:
        if out is not None:
            append_jsonl(out, record)
        current_records = list(current_by_id.values())
        summary = summarize(current_records, args.pairs * 2)
        current_name1, current_name2 = engine_names(
            current_records, engine1, engine2
        )
        summary["engine1_name"] = current_name1
        summary["engine2_name"] = current_name2
        if summary_path is not None:
            atomic_json(summary_path, summary)
        if not args.quiet:
            print_summary(summary, current_name1, current_name2, record)

    interrupted = execute_tasks(
        config,
        tasks,
        records_by_id,
        continue_on_error=args.continue_on_error,
        on_record=on_record,
    )
    if interrupted:
        interrupted_records = list(records_by_id.values())
        summary = summarize(interrupted_records, args.pairs * 2)
        if interrupted_records:
            name1, name2 = engine_names(interrupted_records, engine1, engine2)
        summary["engine1_name"] = name1
        summary["engine2_name"] = name2
        if summary_path is not None:
            atomic_json(summary_path, summary)
        print_final_summary(summary, name1, name2, interrupted=True)
        message = (
            "Completed games are saved and can be resumed."
            if out is not None
            else ""
        )
        if message:
            print(message, file=sys.stderr)
        return 130

    final_records = list(records_by_id.values())
    summary = summarize(final_records, args.pairs * 2)
    if final_records:
        name1, name2 = engine_names(final_records, engine1, engine2)
    summary["engine1_name"] = name1
    summary["engine2_name"] = name2
    if summary_path is not None:
        atomic_json(summary_path, summary)
    print_final_summary(summary, name1, name2, interrupted=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired matches between two 4PC UCI engines"
    )
    parser.add_argument("--engine1", "--e1", required=True, help="first engine executable")
    parser.add_argument("--engine2", "--e2", required=True, help="second engine executable")
    parser.add_argument(
        "--arbiter",
        default="",
        help="dedicated legalmoves/gameresult engine; defaults to the engine currently moving",
    )
    parser.add_argument("--engine1-options", default="", help="engine 1 JSON options")
    parser.add_argument("--engine2-options", default="", help="engine 2 JSON options")
    parser.add_argument("--arbiter-options", default="", help="arbiter JSON options")
    parser.add_argument("--engine1-option", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--engine2-option", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--arbiter-option", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--threads1", "--cores1", type=int, default=1)
    parser.add_argument("--threads2", "--cores2", type=int, default=1)
    parser.add_argument("--hash1", "--mem1", type=int, default=128)
    parser.add_argument("--hash2", "--mem2", type=int, default=128)
    parser.add_argument("--arbiter-hash", type=int, default=16)
    games = parser.add_mutually_exclusive_group()
    games.add_argument("--pairs", type=int, help="paired openings; two games each")
    games.add_argument("--games", type=int, help="total games; must be even")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--nodes", type=int)
    limit.add_argument("--depth", type=int)
    limit.add_argument("--movetime", "--fixed", type=int)
    limit.add_argument("--tc", type=int, help="base time in milliseconds")
    parser.add_argument("--inc", type=int, default=0, help="increment in milliseconds")
    parser.add_argument("--margin", type=int, default=50, help="clock-loss margin in milliseconds")
    parser.add_argument("--timeout", type=float, default=30.0, help="engine response timeout")
    parser.add_argument("--max-plies", "--maxmoves", type=int, default=1000)
    parser.add_argument("--opening-plies", type=int, default=0)
    parser.add_argument("--fens", default="", help="opening FEN file")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--out",
        default="",
        help="optional JSONL output path; enables persistence and resume",
    )
    parser.add_argument("--moves", "--pmoves", action="store_true", help="show move search data")
    parser.add_argument("--quiet", action="store_true", help="only print the final result")
    parser.add_argument("--continue-on-error", "--continue", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fresh", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.games is not None:
        if args.games < 2 or args.games % 2:
            parser.error("--games must be an even number of at least 2")
        args.pairs = args.games // 2
    elif args.pairs is None:
        args.pairs = 1
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
    if args.pairs < 1:
        parser.error("--pairs must be at least 1")
    if args.workers < 1 or args.threads1 < 1 or args.threads2 < 1:
        parser.error("worker and engine thread counts must be at least 1")
    if args.limit_kind == "clock" and args.tc <= 0:
        parser.error("--tc must be greater than 0")
    if args.limit_kind != "clock" and args.inc:
        parser.error("--inc requires --tc")
    if not args.arbiter and (args.arbiter_options or args.arbiter_option):
        parser.error("--arbiter-options and --arbiter-option require --arbiter")
    if (args.fresh or not args.resume) and not args.out:
        parser.error("--fresh and --no-resume require --out")
    try:
        return run_match(args)
    except KeyboardInterrupt:
        message = (
            "Interrupted; completed games are saved and can be resumed."
            if args.out
            else "Interrupted."
        )
        print(message, file=sys.stderr)
        return 130
    except (EngineError, TimeoutError, OSError, ValueError) as exc:
        print(f"match: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
