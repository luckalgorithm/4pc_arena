#!/usr/bin/env python3
# Runs reproducible four-player team-chess matches between UCI engines. Match
# scores are always reported from Engine 1's perspective; position samples use
# the team whose color is to move.
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
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

TURN_ORDER = ("red", "blue", "yellow", "green")
# Red and Yellow share one engine, while Blue and Green share the other.
TEAM_BY_COLOR = {
    "red": "ry",
    "yellow": "ry",
    "blue": "bg",
    "green": "bg",
}
# Four-player UCI assigns a distinct clock command prefix to every color.
UCI_CLOCK_PREFIX = {
    "red": "r",
    "blue": "b",
    "yellow": "y",
    "green": "g",
}
FINAL_RESULTS = {"ry_win", "bg_win", "draw"}
NORMALIZED_ELO_SCALE = 800 / math.log(10)
NORMAL_95_Z = 1.959963984540054
SPRT_REPORT_SEPARATOR = "----------------------------------------------"
# Rule-derived and draw-adjudicated endings are reliable training labels.
# Infrastructure failures must not enter the NNUE corpus.
NNUE_GAME_TERMINATIONS = {
    "game_result",
    "engine_reported_result",
    "no_legal_moves_result",
    "no_legal_moves",
    "max_plies",
}
AUTO_NNUE_OUTPUT = "__auto_nnue_output__"
NNUE_DATA_DEFAULT_GAMES = 50_000
NNUE_DATA_DEFAULT_OPENING_PLIES = 12
NNUE_DATA_DEFAULT_OPENING_NODES = 5_000

# Reports engine startup, protocol, and rule-command failures.
class EngineError(RuntimeError):
    pass


# Separates permanent option incompatibility from retryable engine failures.
class EngineOptionError(EngineError):
    pass

# Signals coordinated shutdown without recording a game as an engine failure.
class MatchInterrupted(RuntimeError):
    pass

# Describes one engine process and the UCI options applied at startup.
@dataclass(frozen=True)
class EngineConfig:
    path: Path
    options: dict[str, int | bool | str]
    hash_mb: int
    threads: int

# Configures the normalized-Elo hypotheses and error bounds for a paired SPRT.
@dataclass(frozen=True)
class SprtConfig:
    elo0: float
    elo1: float
    alpha: float = 0.05
    beta: float = 0.05

# Stores the exact root position and opening moves shared by a scheduled game
# or by both games in a paired opening.
@dataclass(frozen=True)
class StartPosition:
    fen: str | None
    opening_moves: list[str]
    source: str

# Identifies one scheduled game. Paired tasks use the same opening once with
# Engine 1 on each team.
@dataclass(frozen=True)
class GameTask:
    pair_index: int
    engine1_team: str
    start: StartPosition
    paired: bool = True

    @property
    def game_id(self) -> str:
        prefix = "pair" if self.paired else "game"
        return f"{prefix}{self.pair_index:04d}-{self.engine1_team}"

# Collects the validated settings used by both the command-line runner and
# programmatic callers such as the SPSA tuner.
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
    opening_nodes: int = 0
    opening_weights: tuple[float, ...] = (60.0, 30.0, 10.0)
    opening_max_score: int = 1000
    opening_attempts: int = 100
    pgn4_single_line: bool = False
    nnue_output: bool = False
    sprt: SprtConfig | None = None

# Captures the final response to one UCI search, including protocol-level game
# completion or position-rejection details when no move is returned.
@dataclass
class SearchResult:
    bestmove: str | None
    info: dict[str, Any]
    elapsed_ms: int
    result: str | None = None
    position_error: str | None = None

# Retains the option metadata advertised during the UCI handshake. Dictionary
# insertion order preserves the engine's declaration order when values are set.
@dataclass(frozen=True)
class UciOption:
    name: str
    type: str
    default: str | None
    min: int | None
    max: int | None

# Returns completed records and aggregate statistics to in-process match
# clients without requiring JSONL output.
@dataclass
class MatchResult:
    records: list[dict[str, Any]]
    summary: dict[str, Any]
    engine1_name: str
    engine2_name: str
    interrupted: bool

# Tracks completed schedule slots in constant space. A paired entry reserves
# one slot for each Engine 1 team assignment.
class CompletionTracker:
    def __init__(self, entries: int, *, paired: bool) -> None:
        self.entries = entries
        self.paired = paired
        self._completed = bytearray(entries * (2 if paired else 1))

    def _slot(self, index: int, team: str) -> int:
        if index < 1 or index > self.entries:
            raise ValueError(f"Game index is outside the schedule: {index}")
        if not self.paired:
            return index - 1
        if team not in {"ry", "bg"}:
            raise ValueError(f"Invalid Engine 1 team: {team}")
        return (index - 1) * 2 + (0 if team == "ry" else 1)

    def contains(self, index: int, team: str) -> bool:
        return bool(self._completed[self._slot(index, team)])

    def add(self, index: int, team: str) -> bool:
        slot = self._slot(index, team)
        if self._completed[slot]:
            return False
        self._completed[slot] = 1
        return True


# Normalized-Elo GSPRT calculations. The constrained multinomial
# estimate uses deterministic bisection for its secular equation.
def regularized_results(results: list[int]) -> list[float]:
    return [float(value) if value else 1e-3 for value in results]


def distribution_stats(pdf: list[tuple[float, float]]) -> tuple[float, float]:
    total = sum(probability for _, probability in pdf)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Probability distribution does not sum to one")
    mean = sum(value * probability for value, probability in pdf)
    variance = sum(
        probability * (value - mean) ** 2 for value, probability in pdf
    )
    if variance <= 0 or not math.isfinite(variance):
        raise ValueError("Probability distribution has no finite variance")
    return mean, variance


def solve_secular(pdf: list[tuple[float, float]]) -> float:
    values = [value for value, _ in pdf]
    low_value = min(values)
    high_value = max(values)
    if low_value * high_value >= 0:
        raise ValueError("Secular equation support must straddle zero")
    epsilon = 1e-9
    low = -1 / high_value + epsilon
    high = -1 / low_value - epsilon

    def objective(x: float) -> float:
        return sum(
            probability * value / (1 + x * value)
            for value, probability in pdf
        )

    low_result = objective(low)
    high_result = objective(high)
    if low_result < 0 or high_result > 0:
        raise ValueError("Could not bracket secular equation root")
    for _ in range(200):
        midpoint = (low + high) / 2
        result = objective(midpoint)
        if abs(result) <= 1e-13:
            return midpoint
        if result > 0:
            low = midpoint
        else:
            high = midpoint
        if high - low <= 1e-13 * max(1.0, abs(midpoint)):
            return (low + high) / 2
    raise ValueError("Secular equation did not converge")


def mle_t_value(
    empirical: list[tuple[float, float]],
    reference: float,
    target: float,
) -> list[tuple[float, float]]:
    count = len(empirical)
    estimate = [(value, 1 / count) for value, _ in empirical]
    for _ in range(10):
        previous = estimate
        mean, variance = distribution_stats(estimate)
        deviation = math.sqrt(variance)
        adjusted = [
            (
                value
                - reference
                - target
                * deviation
                * (1 + ((mean - value) / deviation) ** 2)
                / 2,
                probability,
            )
            for value, probability in empirical
        ]
        root = solve_secular(adjusted)
        estimate = [
            (
                empirical[index][0],
                empirical[index][1] / (1 + root * adjusted[index][0]),
            )
            for index in range(count)
        ]
        if max(
            abs(previous[index][1] - estimate[index][1])
            for index in range(count)
        ) < 1e-9:
            break
    mean, variance = distribution_stats(estimate)
    actual = (mean - reference) / math.sqrt(variance)
    if not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("Constrained likelihood estimate did not converge")
    return estimate


def normalized_llr(elo0: float, elo1: float, results: list[int]) -> float:
    regularized = regularized_results(results)
    sample_count = sum(regularized)
    pdf = [
        (index / 4, value / sample_count)
        for index, value in enumerate(regularized)
    ]
    scale = math.sqrt(2) / NORMALIZED_ELO_SCALE
    hypotheses = (elo0 * scale, elo1 * scale)
    estimates = [mle_t_value(pdf, 0.5, target) for target in hypotheses]
    jumps = [
        (
            math.log(estimates[1][index][1])
            - math.log(estimates[0][index][1]),
            pdf[index][1],
        )
        for index in range(5)
    ]
    mean, _ = distribution_stats(jumps)
    return sample_count * mean


def logistic_elo(score: float) -> float:
    bounded = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / bounded - 1)


def pentanomial_estimates(
    results: list[int],
) -> tuple[float, float, float, float]:
    regularized = regularized_results(results)
    pair_count = sum(regularized)
    games = 2 * pair_count
    score = sum(value * (index / 2) for index, value in enumerate(regularized)) / games
    pair_mean = 2 * score
    variance = sum(
        value * (index / 2 - pair_mean) ** 2
        for index, value in enumerate(regularized)
    ) / games
    deviation = math.sqrt(variance)
    error = NORMAL_95_Z * deviation / math.sqrt(games)
    elo = logistic_elo(score)
    elo_95 = (logistic_elo(score + error) - logistic_elo(score - error)) / 2
    normalized_elo = (score - 0.5) / deviation * NORMALIZED_ELO_SCALE
    normalized_elo_95 = NORMAL_95_Z / math.sqrt(games) * NORMALIZED_ELO_SCALE
    return elo, elo_95, normalized_elo, normalized_elo_95


class PentanomialSprt:
    def __init__(self, config: SprtConfig) -> None:
        self.config = config
        self.results = [0, 0, 0, 0, 0]
        self.lower_bound = math.log(config.beta / (1 - config.alpha))
        self.upper_bound = math.log((1 - config.beta) / config.alpha)
        self.llr = 0.0
        self.state = "running"

    def add_pair(self, first: float, second: float) -> None:
        if first not in {0.0, 0.5, 1.0} or second not in {0.0, 0.5, 1.0}:
            raise ValueError("SPRT game scores must be 0, 0.5, or 1")
        bucket = int(round(2 * (first + second)))
        self.results[bucket] += 1
        self.llr = normalized_llr(
            self.config.elo0,
            self.config.elo1,
            self.results,
        )
        if self.llr < self.lower_bound:
            self.state = "rejected"
        elif self.llr > self.upper_bound:
            self.state = "accepted"
        else:
            self.state = "running"

    @property
    def pairs_completed(self) -> int:
        return sum(self.results)

    @property
    def terminal(self) -> bool:
        return self.state in {"accepted", "rejected"}

    def snapshot(self, *, final: bool = False) -> dict[str, Any]:
        state = "inconclusive" if final and self.state == "running" else self.state
        estimates = (
            pentanomial_estimates(self.results)
            if self.pairs_completed
            else (None, None, None, None)
        )
        return {
            "model": "normalized",
            "elo0": self.config.elo0,
            "elo1": self.config.elo1,
            "alpha": self.config.alpha,
            "beta": self.config.beta,
            "llr": self.llr,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "state": state,
            "pentanomial": list(self.results),
            "pairs_completed": self.pairs_completed,
            "elo": estimates[0],
            "elo_95": estimates[1],
            "normalized_elo": estimates[2],
            "normalized_elo_95": estimates[3],
        }

# Accumulates WDL, termination counts, Elo, and paired confidence statistics
# without retaining every game record. Pair variance uses Welford's algorithm.
class SummaryAccumulator:
    def __init__(
        self,
        games_expected: int,
        *,
        paired: bool,
        sprt: SprtConfig | None = None,
    ) -> None:
        self.games_expected = games_expected
        self.paired = paired
        self.sprt = PentanomialSprt(sprt) if sprt is not None else None
        self.games_completed = 0
        self.engine1_wins = 0
        self.engine2_wins = 0
        self.draws = 0
        self.terminations: dict[str, int] = {}
        self.pending_pairs: dict[int, float] = {}
        self.pairs_completed = 0
        self.pair_mean = 0.0
        self.pair_m2 = 0.0

    # Incorporates one Engine 1-oriented record. The first half of a pair waits
    # in pending_pairs until its color-swapped partner arrives.
    def add(self, record: dict[str, Any]) -> None:
        result = effective_game_result(record)
        engine1_team = str(record.get("engine1_team", ""))
        score = (
            engine1_score(result, engine1_team)
            if result in FINAL_RESULTS and engine1_team in {"ry", "bg"}
            else float(record.get("engine1_score", 0.5))
        )
        self.games_completed += 1
        if score == 1.0:
            self.engine1_wins += 1
        elif score == 0.0:
            self.engine2_wins += 1
        else:
            self.draws += 1
        termination = str(record.get("termination", "unknown"))
        self.terminations[termination] = self.terminations.get(termination, 0) + 1
        if not self.paired:
            return
        pair = int(record.get("pair", record.get("game", 0)) or 0)
        if pair < 1:
            return
        previous = self.pending_pairs.pop(pair, None)
        if previous is None:
            self.pending_pairs[pair] = score
            return
        pair_score = (previous + score) / 2
        self.pairs_completed += 1
        delta = pair_score - self.pair_mean
        self.pair_mean += delta / self.pairs_completed
        self.pair_m2 += delta * (pair_score - self.pair_mean)
        if self.sprt is not None:
            self.sprt.add_pair(previous, score)

    # Returns a snapshot safe for JSON serialization. The confidence interval
    # is computed over complete pair scores rather than individual games.
    def summary(self, *, final: bool = False) -> dict[str, Any]:
        if self.games_completed:
            score = (
                self.engine1_wins + self.draws / 2
            ) / self.games_completed
        else:
            score = 0.5
        if score >= 1.0:
            elo = math.inf
        elif score <= 0.0:
            elo = -math.inf
        else:
            elo = 400 * math.log10(score / (1 - score))
        confidence = None
        if self.pairs_completed > 1:
            variance = self.pair_m2 / (self.pairs_completed - 1)
            error = math.sqrt(variance / self.pairs_completed)
            confidence = [
                max(0.0, self.pair_mean - 1.96 * error),
                min(1.0, self.pair_mean + 1.96 * error),
            ]
        terminations = dict(self.terminations)
        result = {
            "games_completed": self.games_completed,
            "games_expected": self.games_expected,
            "engine1_wins": self.engine1_wins,
            "engine2_wins": self.engine2_wins,
            "draws": self.draws,
            "engine1_score": score,
            "engine2_score": 1 - score,
            "elo": elo,
            "pairs_completed": self.pairs_completed,
            "paired_95ci": confidence,
            "engine1_illegal_moves": terminations.get("engine1_illegal_move", 0),
            "engine2_illegal_moves": terminations.get("engine2_illegal_move", 0),
            "engine1_time_losses": terminations.get("engine1_time_loss", 0),
            "engine2_time_losses": terminations.get("engine2_time_loss", 0),
            "engine1_timeouts": terminations.get("engine1_timeout", 0),
            "engine2_timeouts": terminations.get("engine2_timeout", 0),
            "runner_errors": terminations.get("runner_error", 0),
            "terminations": terminations,
        }
        if self.sprt is not None:
            sprt = self.sprt.snapshot(final=final)
            result["sprt"] = sprt
            if sprt["elo"] is not None:
                result["elo"] = sprt["elo"]
                result["elo_95"] = sprt["elo_95"]
                result["normalized_elo"] = sprt["normalized_elo"]
                result["normalized_elo_95"] = sprt["normalized_elo_95"]
        return result

# Registers every live subprocess so interruption can stop all workers before
# the executor waits for their threads.
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

# Owns one reusable engine pair and optional arbiter for a single executor
# thread. Reuse avoids a UCI handshake and hash allocation before every game.
class WorkerEngines:
    def __init__(self, config: MatchConfig, active_engines: ActiveEngines) -> None:
        self._stack = contextlib.ExitStack()
        self._closed = False
        try:
            self.engine1 = self._stack.enter_context(UciEngine(
                config.engine1,
                label="worker:engine1",
                active_engines=active_engines,
            ))
            self.engine2 = self._stack.enter_context(UciEngine(
                config.engine2,
                label="worker:engine2",
                active_engines=active_engines,
            ))
            self.arbiter = (
                self._stack.enter_context(UciEngine(
                    config.arbiter,
                    label="worker:arbiter",
                    active_engines=active_engines,
                ))
                if config.arbiter is not None
                else None
            )
        except BaseException:
            self._stack.close()
            raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._stack.close()

    def __enter__(self) -> WorkerEngines:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

# Provides synchronized shutdown for worker-owned engine sets. Individual
# workers remove unhealthy sets before retrying with fresh processes.
class WorkerEngineRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workers: set[WorkerEngines] = set()

    def add(self, engines: WorkerEngines) -> None:
        with self._lock:
            self._workers.add(engines)

    def discard(self, engines: WorkerEngines) -> None:
        with self._lock:
            self._workers.discard(engines)

    def close_all(self) -> None:
        with self._lock:
            workers = list(self._workers)
            self._workers.clear()
        for engines in workers:
            engines.close()

# Wraps one line-oriented UCI subprocess. A reader thread drains stdout into a
# queue while the owning match thread performs synchronous request/response
# operations; one instance is never shared by concurrent games.
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
        self.uci_options: dict[str, UciOption] = {}
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
            available_options = {name.casefold() for name in self.uci_options}
            missing_options = sorted(
                name
                for name in config.options
                if name.casefold() not in available_options
            )
            if missing_options:
                missing = ", ".join(missing_options)
                raise EngineOptionError(
                    f"{self.label} does not expose required UCI option(s): {missing}"
                )
            self.send(f"setoption name Hash value {config.hash_mb}")
            self.send(f"setoption name Threads value {config.threads}")
            requested_by_casefold = {
                name.casefold(): value for name, value in config.options.items()
            }
            # Preserve the engine's declaration order. Tuning builds may defer
            # rebuilding derived tables until their final declared option.
            for name in self.uci_options:
                folded = name.casefold()
                if folded in requested_by_casefold:
                    self.send(
                        f"setoption name {name} value "
                        f"{option_text(requested_by_casefold[folded])}"
                    )
            self.sync()
        except BaseException:
            self.close(force=True)
            raise

    def __enter__(self) -> UciEngine:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # Continuously drains stdout so an engine cannot block on a full pipe. A
    # None sentinel distinguishes process exit from an ordinary timeout.
    def _read_loop(self) -> None:
        assert self.proc.stdout is not None
        try:
            for raw in self.proc.stdout:
                line = raw.strip()
                if not line:
                    continue
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
            elif line.startswith("option name "):
                option = parse_uci_option(line)
                if option is not None:
                    self.uci_options[option.name] = option
            if line == "uciok":
                return

    # Consumes informational output until the requested response arrives.
    # Explicit protocol rejection messages become actionable exceptions.
    def wait_for(self, prefix: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.label} timed out waiting for {prefix}")
            line = self.read_line(remaining)
            if line.startswith("info string invalid command:"):
                raise EngineError(f"{self.label} rejected command: {line}")
            if line.startswith("Unknown command:"):
                raise EngineError(f"{self.label} rejected command: {line}")
            if line.startswith("No such option:"):
                raise EngineOptionError(f"{self.label} rejected option: {line}")
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

    # Uses the four-player engine extension that returns canonical legal moves
    # for the supplied root and move sequence.
    def legal_moves(self, moves: list[str], fen: str | None) -> list[str]:
        self.set_position(moves, fen)
        self.send("legalmoves")
        parts = self.wait_for("legalmoves", 30).split(maxsplit=1)
        return [] if len(parts) == 1 else [canonical_move(move) for move in parts[1].split()]

    # Uses the four-player engine extension that classifies the current game as
    # ongoing, a team win, or a draw.
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

    # Reads the exact root FEN through the engine's diagnostic command. Waiting
    # for Key ensures the complete diagnostic response is removed from stdout.
    def current_fen(self, timeout: float = 30) -> str:
        self.send("d")
        line = self.wait_for("Fen:", timeout)
        fen = line.removeprefix("Fen:").strip()
        self.wait_for("Key:", timeout)
        if not fen:
            raise EngineError(f"{self.label} returned an empty FEN")
        return fen

    # Searches one position and retains the latest fields from its UCI info
    # stream. A timeout requests stop before control returns to the match.
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
            if line.startswith("info string"):
                lowered = line.lower()
                if "illegal move" in lowered or "invalid move" in lowered:
                    elapsed = int(round((time.monotonic() - started) * 1000))
                    return SearchResult(
                        None,
                        info,
                        elapsed,
                        position_error=line.removeprefix("info string").strip(),
                    )
                if any(
                    phrase in lowered
                    for phrase in (
                        "invalid fen",
                        "invalid position",
                        "illegal fen",
                        "illegal position",
                    )
                ):
                    elapsed = int(round((time.monotonic() - started) * 1000))
                    return SearchResult(
                        None,
                        info,
                        elapsed,
                        position_error=line.removeprefix("info string").strip(),
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

    # Collects the final info record for each MultiPV rank. Results are returned
    # in rank order and use the principal variation's first move.
    def search_multipv(
        self,
        moves: list[str],
        fen: str | None,
        go_command: str,
        timeout: float,
    ) -> list[SearchResult]:
        self.set_position(moves, fen)
        started = time.monotonic()
        self.send(go_command)
        deadline = started + timeout
        infos: dict[int, dict[str, Any]] = {}
        bestmove: str | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    self.send("stop")
                except (BrokenPipeError, OSError, EngineError):
                    pass
                raise TimeoutError(f"{self.label} MultiPV search timed out")
            try:
                line = self.read_line(min(remaining, 1.0))
            except TimeoutError:
                continue
            if line.startswith("info string Game completed."):
                return []
            if line.startswith("info "):
                parsed = parse_info(line)
                pv_index = int(parsed.get("multipv", 1))
                infos.setdefault(pv_index, {}).update(parsed)
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) > 1 and parts[1] not in {"(none)", "0000"}:
                    bestmove = canonical_move(parts[1])
                break

        elapsed = int(round((time.monotonic() - started) * 1000))
        results: list[SearchResult] = []
        for pv_index in sorted(infos):
            info = infos[pv_index]
            pv = info.get("pv")
            move = canonical_move(str(pv[0])) if isinstance(pv, list) and pv else None
            if pv_index == 1 and move is None:
                move = bestmove
            if move is not None:
                results.append(SearchResult(move, info, elapsed))
        if not results and bestmove is not None:
            results.append(SearchResult(bestmove, {}, elapsed))
        return results

    # Attempts a graceful UCI shutdown, then escalates to process termination.
    # Cleanup is idempotent so interruption and normal scope exit may overlap.
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
            for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            if (
                hasattr(self, "_reader")
                and self._reader.is_alive()
                and threading.current_thread() is not self._reader
            ):
                self._reader.join(timeout=0.5)

# Converts typed configuration values to the spelling expected by UCI.
def option_text(value: int | bool | str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)

# Parses the subset of a UCI option declaration needed for validation and for
# preserving declaration order. Unknown declaration shapes are ignored.
def parse_uci_option(line: str) -> UciOption | None:
    match = re.match(
        r"^option name (.*?) type (spin|check|combo|button|string)(?: (.*))?$",
        line,
    )
    if match is None:
        return None
    name, option_type, attributes = match.groups()
    attributes = attributes or ""
    default_match = re.search(
        r"(?:^| )default (.*?)(?= (?:min|max|var) |$)",
        attributes,
    )
    min_match = re.search(r"(?:^| )min (-?\d+)(?: |$)", attributes)
    max_match = re.search(r"(?:^| )max (-?\d+)(?: |$)", attributes)
    return UciOption(
        name=name,
        type=option_type,
        default=default_match.group(1) if default_match else None,
        min=int(min_match.group(1)) if min_match else None,
        max=int(max_match.group(1)) if max_match else None,
    )

# Normalizes coordinate and PGN4-like separators before move comparison.
def canonical_move(move: str) -> str:
    return move.strip().lower().replace("-", "").replace("=", "")

# Maps common engine result spellings to the runner's team-oriented vocabulary.
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

# Extracts the result embedded in Stockfish 4PC's completion information line.
def completed_result(line: str) -> str:
    lowered = line.lower()
    if "ry won" in lowered:
        return "ry_win"
    if "bg won" in lowered:
        return "bg_win"
    if "stalemate" in lowered or "draw" in lowered:
        return "draw"
    return "unknown"

# Extracts stable numeric and principal-variation fields from one UCI info line.
# Repeated dictionaries may be merged so later values replace earlier ones.
def parse_info(line: str) -> dict[str, Any]:
    parts = line.split()
    info: dict[str, Any] = {}
    i = 1
    while i < len(parts):
        key = parts[i]
        if key in {"depth", "seldepth", "multipv", "nodes", "nps", "hashfull", "time"} and i + 1 < len(parts):
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
                "depth", "seldepth", "multipv", "nodes", "nps", "hashfull", "time", "score"
            }:
                end += 1
            info["pv"] = parts[i + 1 : end]
            i = end
        else:
            i += 1
    return info

# Preserves integers and booleans from JSON while coercing command-line text to
# those types when its spelling is unambiguous.
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

# Parses positive MultiPV rank weights for argparse.
def parse_opening_weights(value: str) -> tuple[float, ...]:
    try:
        weights = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "opening weights must be comma-separated numbers"
        ) from exc
    if not weights or any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise argparse.ArgumentTypeError("opening weights must all be positive")
    return weights

# Loads a JSON option object, then applies repeated NAME=VALUE overrides in
# command-line order.
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

# Loads nonempty, non-comment opening positions in file order.
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

# Yields deterministic shuffled cycles. A FEN appears at most once per cycle,
# avoiding the clumping caused by independent random choice.
def shuffled_fens(
    fens: list[str],
    entries: int,
    seed: int,
) -> Iterator[str | None]:
    if not fens:
        for _ in range(entries):
            yield None
        return

    rng = random.Random(f"{seed}:fens")
    cycle = list(fens)
    remaining = entries
    while remaining > 0:
        rng.shuffle(cycle)
        cycle_size = min(remaining, len(cycle))
        yield from cycle[:cycle_size]
        remaining -= cycle_size

# Returns the starting color index encoded by a four-player FEN. Startpos and
# malformed turn fields conservatively fall back to Red.
def fen_turn_index(fen: str | None) -> int:
    if not fen:
        return 0
    token = fen.split("-", 1)[0].strip().lower()
    return {"r": 0, "b": 1, "y": 2, "g": 3}.get(token[:1], 0)

# Captures enough executable and option identity to reject incompatible resume
# attempts without hashing a potentially large binary on every invocation.
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

# Builds the stable same-day NNUE filename used by a bare --nnue-output flag.
def default_nnue_output_path(engine1: Path, engine2: Path) -> Path:
    def safe_name(path: Path) -> str:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
        return name or "engine"

    filename = (
        f"{safe_name(engine1)}-vs-{safe_name(engine2)}-"
        f"{date.today().isoformat()}.txt"
    )
    return Path(filename).resolve()

MATCH_METADATA_DEFAULTS: dict[str, Any] = {
    "opening_nodes": 0,
    "opening_weights": [60.0, 30.0, 10.0],
    "opening_max_score": 1000,
    "opening_attempts": 100,
    "pgn4_single_line": False,
    "nnue_output": False,
}

# Adds defaults for fields absent from metadata written by older runner
# versions, allowing compatible runs to resume across schema additions.
def normalized_match_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    for name, default in MATCH_METADATA_DEFAULTS.items():
        normalized.setdefault(name, default)
    return normalized

# Accepts either the current fingerprint or an older payload whose normalized
# settings are equivalent to the requested match.
def metadata_matches_payload(
    metadata: dict[str, Any],
    payload: dict[str, Any],
    fingerprint: str,
) -> bool:
    if metadata.get("fingerprint") == fingerprint:
        return True
    saved = metadata.get("config")
    return (
        isinstance(saved, dict)
        and normalized_match_payload(saved) == normalized_match_payload(payload)
    )

# Clones an engine configuration with the requested MultiPV width while leaving
# the caller's option dictionary unchanged.
def opening_search_config(config: EngineConfig, multipv: int) -> EngineConfig:
    options = {
        name: value
        for name, value in config.options.items()
        if name.casefold() != "multipv"
    }
    options["MultiPV"] = multipv
    return EngineConfig(config.path, options, config.hash_mb, config.threads)

# Rejects mate scores and centipawn evaluations outside the configured opening
# balance window. A zero threshold disables centipawn rejection only.
def opening_score_is_extreme(info: dict[str, Any], max_score: int) -> bool:
    if max_score <= 0:
        return False
    score = info.get("score")
    if not isinstance(score, dict):
        return False
    if score.get("type") == "mate":
        return True
    try:
        return abs(int(score["value"])) > max_score
    except (KeyError, TypeError, ValueError):
        return False

# Generates one shared start position. Uniform mode samples legal moves; guided
# mode samples weighted MultiPV ranks and retries openings whose final score is
# too extreme.
def generate_start(
    config: MatchConfig,
    rng: random.Random,
    fen: str | None,
    opening_engine: UciEngine | None = None,
    rules_engine: UciEngine | None = None,
) -> StartPosition:
    if config.opening_plies <= 0:
        moves: list[str] = []
        return StartPosition(fen, moves, "fen" if fen else "startpos")
    if opening_engine is None:
        opening_config = config.arbiter or config.engine1
        if config.opening_nodes > 0:
            opening_config = opening_search_config(
                config.engine1, len(config.opening_weights)
            )
        with UciEngine(opening_config, label="opening-rules") as owned:
            return generate_start(config, rng, fen, owned)

    attempts = config.opening_attempts if config.opening_nodes > 0 else 1
    for _ in range(attempts):
        moves = []
        rejected = False
        for _ply in range(config.opening_plies):
            if config.opening_nodes <= 0:
                rules = rules_engine or opening_engine
                legal = rules.legal_moves(moves, fen)
                if not legal:
                    break
                moves.append(rng.choice(legal))
                continue

            legal = rules_engine.legal_moves(moves, fen) if rules_engine else None
            if legal == []:
                rejected = True
                break
            ranked = opening_engine.search_multipv(
                moves,
                fen,
                f"go nodes {config.opening_nodes}",
                config.timeout,
            )
            if not ranked:
                rejected = True
                break
            if opening_score_is_extreme(ranked[0].info, config.opening_max_score):
                rejected = True
                break

            legal_set = set(legal) if legal is not None else None
            candidates: list[str] = []
            weights: list[float] = []
            for rank, result in enumerate(ranked[:len(config.opening_weights)]):
                move = result.bestmove
                if (
                    move is None
                    or (legal_set is not None and move not in legal_set)
                    or move in candidates
                ):
                    continue
                candidates.append(move)
                weights.append(config.opening_weights[rank])
            if not candidates:
                raise EngineError(
                    f"{opening_engine.label} returned no legal MultiPV opening move"
                )
            moves.append(rng.choices(candidates, weights=weights, k=1)[0])

        if (
            not rejected
            and config.opening_nodes > 0
            and config.opening_max_score > 0
            and len(moves) == config.opening_plies
        ):
            final_ranked = opening_engine.search_multipv(
                moves,
                fen,
                f"go nodes {config.opening_nodes}",
                config.timeout,
            )
            rejected = not final_ranked or opening_score_is_extreme(
                final_ranked[0].info, config.opening_max_score
            )

        if not rejected:
            return StartPosition(fen, moves, "fen" if fen else "startpos")

    raise EngineError(
        "Could not generate a balanced opening after "
        f"{attempts} attempts; increase --opening-max-score, provide more FENs, "
        "or increase --opening-attempts"
    )

# Builds either a fixed-limit search or a four-color clock search. Every color
# retains its own time even though teammates share an engine process.
def go_command(config: MatchConfig, clocks: dict[str, int]) -> str:
    if config.limit_kind == "clock":
        times = " ".join(
            f"{UCI_CLOCK_PREFIX[color]}time {max(0, clocks[color])}"
            for color in TURN_ORDER
        )
        increments = " ".join(
            f"{UCI_CLOCK_PREFIX[color]}inc {config.increment_ms}"
            for color in TURN_ORDER
        )
        return f"go {times} {increments}"
    return f"go {config.limit_kind} {config.limit_value}"

# Gives clock searches the active color's remaining time plus the configured
# response grace. Fixed-limit searches use the response timeout directly.
def move_timeout(config: MatchConfig, clocks: dict[str, int], color: str) -> float:
    if config.limit_kind != "clock":
        return config.timeout
    remaining = max(0, clocks[color]) + max(0, config.margin_ms)
    return remaining / 1000 + config.timeout

# Returns the opposing team result after a time, protocol, or move failure.
def winner_for_failure(team: str) -> str:
    return "bg_win" if team == "ry" else "ry_win"

# Converts a team result to a score from Engine 1's assignment in this game.
# Unknown terminal values are rejected rather than silently counted as draws.
def engine1_score(result: str, engine1_team: str) -> float:
    if result == "draw":
        return 0.5
    if result == "ry_win":
        return 1.0 if engine1_team == "ry" else 0.0
    if result == "bg_win":
        return 1.0 if engine1_team == "bg" else 0.0
    raise ValueError(f"Invalid finished game result: {result}")

# Interprets a mate score from the searching team's side-to-move perspective.
def mate_score_result(info: dict[str, Any], team: str) -> str | None:
    score = info.get("score")
    if not isinstance(score, dict) or score.get("type") != "mate":
        return None
    try:
        value = int(score["value"])
    except (KeyError, TypeError, ValueError):
        return None
    return winner_for_failure(team) if value <= 0 else f"{team}_win"


# A move searched as mate in one is authoritative when the resulting position
# returns no move but an older engine cannot report the terminal game result.
def previous_mate_result(searches: Any) -> str | None:
    if not isinstance(searches, list) or not searches:
        return None
    previous = searches[-1]
    if not isinstance(previous, dict):
        return None
    team = previous.get("team")
    score = previous.get("score")
    if team not in {"ry", "bg"} or not isinstance(score, dict):
        return None
    try:
        mate = int(score["value"])
    except (KeyError, TypeError, ValueError):
        return None
    return f"{team}_win" if score.get("type") == "mate" and mate == 1 else None


# Corrects legacy records written before mate-in-one no-move endings were
# distinguished from genuine stalemates. New records already store this result.
def effective_game_result(record: dict[str, Any]) -> str:
    result = str(record.get("result", ""))
    if result == "draw" and record.get("termination") == "no_legal_moves":
        return previous_mate_result(record.get("searches")) or result
    return result

# Finalizes the mutable game record in one place so every termination includes
# the same moves, score, and four-color clock snapshot.
def finish_game(
    record: dict[str, Any],
    result: str,
    termination: str,
    moves: list[str],
    clocks: dict[str, int],
    engine1_team: str,
    **details: Any,
) -> dict[str, Any]:
    record.update(
        moves=list(moves),
        result=result,
        engine1_score=engine1_score(result, engine1_team),
        termination=termination,
        final_clocks_ms=clocks,
        **details,
    )
    return record

# Asks the other engine for a shallow confirmation when the moving engine
# returns no move. Explicit result messages take precedence over mate scores.
def confirm_no_legal_moves(
    engine: UciEngine,
    moves: list[str],
    fen: str | None,
    search: SearchResult,
    team: str,
    timeout: float,
) -> str | None:
    try:
        confirmation = engine.search(moves, fen, "go depth 1", min(timeout, 30.0))
    except TimeoutError:
        confirmation = SearchResult(None, {}, 0)
    if confirmation.result in {"ry_win", "bg_win", "draw"}:
        return confirmation.result
    return mate_score_result(search.info, team) or mate_score_result(
        confirmation.info, team
    )

# Borrows a worker's persistent engines when available, or creates a scoped set
# for direct callers of play_game().
@contextlib.contextmanager
def game_engines(
    config: MatchConfig,
    active_engines: ActiveEngines,
    reusable: WorkerEngines | None,
) -> Iterator[WorkerEngines]:
    if reusable is not None:
        yield reusable
        return
    with WorkerEngines(config, active_engines) as owned:
        yield owned

# Plays one complete scheduled game. Engine processes are assigned to teams for
# this task, synchronized with ucinewgame, and driven one color at a time. The
# returned record contains enough search data for summaries, PGN4, and optional
# NNUE export.
def play_game(
    config: MatchConfig,
    task: GameTask,
    stop_event: threading.Event,
    active_engines: ActiveEngines,
    reusable_engines: WorkerEngines | None = None,
) -> dict[str, Any]:
    if stop_event.is_set():
        raise MatchInterrupted("Match interrupted")
    engine1_is_ry = task.engine1_team == "ry"
    ry_number = 1 if engine1_is_ry else 2
    bg_number = 2 if engine1_is_ry else 1
    moves = list(task.start.opening_moves)
    searches: list[dict[str, Any]] = []
    clocks = {color: config.base_time_ms for color in TURN_ORDER}
    start_turn = fen_turn_index(task.start.fen)
    record: dict[str, Any] = {
        "game_id": task.game_id,
        "pair": task.pair_index if task.paired else None,
        "game": task.pair_index,
        "paired": task.paired,
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

    with game_engines(config, active_engines, reusable_engines) as worker:
        engine1 = worker.engine1
        engine2 = worker.engine2
        arbiter = worker.arbiter
        ry_engine = engine1 if engine1_is_ry else engine2
        bg_engine = engine2 if engine1_is_ry else engine1
        ry_engine.label = f"{task.game_id}:RY"
        bg_engine.label = f"{task.game_id}:BG"
        if arbiter is not None:
            arbiter.label = f"{task.game_id}:arbiter"
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
            # A dedicated arbiter is authoritative for both terminal state and
            # legality. Contradictory responses indicate a protocol failure.
            if arbiter is not None:
                result = arbiter.game_result(moves, task.start.fen)
                if result in FINAL_RESULTS:
                    return finish_game(
                        record, result, "game_result", moves, clocks, task.engine1_team
                    )
                if result != "ongoing":
                    raise EngineError(
                        f"{arbiter.label} returned an invalid game result: {result}"
                    )
            legal = (
                arbiter.legal_moves(moves, task.start.fen)
                if arbiter is not None
                else None
            )
            if legal == []:
                raise EngineError(
                    f"{arbiter.label} reports an ongoing game with no legal moves"
                )
            engine_number = ry_number if team == "ry" else bg_number
            if legal is not None and len(legal) == 1:
                search = SearchResult(legal[0], {}, 0)
            else:
                try:
                    search = engine.search(
                        moves,
                        task.start.fen,
                        go_command(config, clocks),
                        move_timeout(config, clocks, color),
                    )
                except TimeoutError:
                    result = winner_for_failure(team)
                    return finish_game(
                        record, result, f"engine{engine_number}_timeout",
                        moves, clocks, task.engine1_team
                    )
            if search.position_error is not None:
                if not moves:
                    raise EngineError(
                        f"{engine.label} rejected the starting position: "
                        f"{search.position_error}"
                    )
                # The next engine discovers a rejected move only after receiving
                # the updated position, so responsibility belongs to the
                # previous team.
                previous_team = "bg" if team == "ry" else "ry"
                previous_number = bg_number if previous_team == "bg" else ry_number
                result = winner_for_failure(previous_team)
                return finish_game(
                    record, result, f"engine{previous_number}_illegal_move",
                    moves, clocks, task.engine1_team, illegal_move=moves[-1],
                    reported_by=engine.name, position_error=search.position_error
                )
            if search.result in {"ry_win", "bg_win", "draw"}:
                return finish_game(
                    record, search.result, "engine_reported_result",
                    moves, clocks, task.engine1_team
                )
            if search.bestmove is None:
                other_engine = bg_engine if engine is ry_engine else ry_engine
                result = confirm_no_legal_moves(
                    other_engine, moves, task.start.fen, search, team, config.timeout
                )
                result = result or previous_mate_result(searches)
                return finish_game(
                    record, result or "draw",
                    "no_legal_moves_result" if result else "no_legal_moves",
                    moves, clocks, task.engine1_team
                )
            # Clock time belongs to the moving color, not to its two-color team.
            if config.limit_kind == "clock":
                clocks[color] -= search.elapsed_ms
                if clocks[color] < -config.margin_ms:
                    result = winner_for_failure(team)
                    return finish_game(
                        record, result, f"engine{engine_number}_time_loss",
                        moves, clocks, task.engine1_team
                    )
                clocks[color] += config.increment_ms
            if legal is not None and search.bestmove not in legal:
                result = winner_for_failure(team)
                return finish_game(
                    record, result, f"engine{engine_number}_illegal_move",
                    moves, clocks, task.engine1_team, illegal_move=search.bestmove
                )
            # The root FEN must be captured before bestmove is appended; its
            # side-to-move team defines both CP and result perspective.
            position_fen = None
            score = search.info.get("score")
            if (
                config.nnue_output
                and isinstance(score, dict)
                and score.get("type") == "cp"
            ):
                position_fen = engine.current_fen(config.timeout)
            moves.append(search.bestmove)
            search_record = {
                "ply": len(moves),
                "color": color,
                "team": team,
                "engine": engine_number,
                "move": search.bestmove,
                "elapsed_ms": search.elapsed_ms,
                "clock_ms": clocks[color] if config.limit_kind == "clock" else None,
                **search.info,
            }
            if position_fen is not None:
                search_record["fen"] = position_fen
            searches.append(search_record)
            if config.show_moves:
                print_move(task.game_id, search_record)
    return finish_game(
        record, "draw", "max_plies", moves, clocks, task.engine1_team
    )

# Formats either a centipawn or mate score for the live move display.
def score_text(info: dict[str, Any]) -> str:
    score = info.get("score")
    if not isinstance(score, dict):
        return "--"
    if score.get("type") == "mate":
        return f"M{score.get('value')}"
    return f"{int(score.get('value', 0)):+d}"

# Prints one compact, flush-safe search line for interactive monitoring.
def print_move(game_id: str, info: dict[str, Any]) -> None:
    print(
        f"{game_id} ply {info['ply']:03d} {info['color']:<6} "
        f"E{info['engine']} {info['move']:<6} "
        f"score {score_text(info):>6} depth {info.get('depth', '--'):>3} "
        f"time {info['elapsed_ms']:>6}ms nodes {info.get('nodes', '--')}",
        flush=True,
    )

# Replays records through the streaming accumulator so batch and live summaries
# use exactly the same statistics implementation.
def summarize(
    records: list[dict[str, Any]],
    games_expected: int,
    *,
    paired: bool = True,
    sprt: SprtConfig | None = None,
    final: bool = False,
) -> dict[str, Any]:
    accumulator = SummaryAccumulator(games_expected, paired=paired, sprt=sprt)
    for record in records:
        accumulator.add(record)
    return accumulator.summary(final=final)

# Renders finite and saturated Elo values with an explicit sign.
def elo_text(elo: float) -> str:
    return (
        "+inf" if elo == math.inf
        else "-inf" if elo == -math.inf
        else f"{elo:+.1f}"
    )

# Describes exceptional terminations appended to a live summary line.
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

# Prints cumulative WDL, score, and Elo after one completed game.
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


def sprt_control_text(config: MatchConfig) -> str:
    if config.limit_kind == "clock":
        base_seconds = config.base_time_ms / 1000
        base = f"{base_seconds:.3f}".rstrip("0").rstrip(".")
        control = f"{base}+{config.increment_ms / 1000:.2f}"
    elif config.limit_kind == "nodes":
        control = f"Nodes={config.limit_value}"
    elif config.limit_kind == "depth":
        control = f"Depth={config.limit_value}"
    else:
        control = f"Move={config.limit_value}ms"
    return (
        f"{control} Th={config.engine1.threads} "
        f"Hash={config.engine1.hash_mb}MB Conc={config.workers}"
    )


def sprt_info(summary: dict[str, Any]) -> str | None:
    parts: list[str] = []
    paired_counts = (
        ("Timeouts", "engine1_timeouts", "engine2_timeouts"),
        ("Time losses", "engine1_time_losses", "engine2_time_losses"),
        ("Illegal moves", "engine1_illegal_moves", "engine2_illegal_moves"),
    )
    for label, first_key, second_key in paired_counts:
        first = int(summary[first_key])
        second = int(summary[second_key])
        if first or second:
            parts.append(f"[{label}: {first} / {second}]")
    terminations = summary["terminations"]
    scalar_counts = (("Runner errors", "runner_error"),)
    for label, key in scalar_counts:
        count = int(terminations.get(key, 0))
        if count:
            parts.append(f"[{label}: {count}]")
    return " ".join(parts) if parts else None


def format_sprt_report(summary: dict[str, Any], config: MatchConfig) -> str:
    sprt = summary["sprt"]
    if not sprt["pairs_completed"]:
        raise ValueError("Cannot format an SPRT report without a complete pair")
    lines = [
        f"Elo   | {sprt['elo']:.2f} +- {sprt['elo_95']:.2f} (95%)",
        (
            f"nElo  | {sprt['normalized_elo']:.2f} +- "
            f"{sprt['normalized_elo_95']:.2f} (95%)"
        ),
        f"SPRT  | {sprt_control_text(config)}",
        (
            f"LLR   | {sprt['llr']:.2f} "
            f"({sprt['lower_bound']:.2f}, {sprt['upper_bound']:.2f}) "
            f"[{sprt['elo0']:.2f}, {sprt['elo1']:.2f} normalized]"
        ),
        (
            f"Games | N: {summary['games_completed']} "
            f"W: {summary['engine1_wins']} L: {summary['engine2_wins']} "
            f"D: {summary['draws']}"
        ),
        "Penta | " + " ".join(str(value) for value in sprt["pentanomial"]),
    ]
    info = sprt_info(summary)
    if info is not None:
        lines.append(f"Info  | {info}")
    lines.append(SPRT_REPORT_SEPARATOR)
    return "\n".join(lines)


def print_sprt_report(summary: dict[str, Any], config: MatchConfig) -> None:
    print(format_sprt_report(summary, config), flush=True)

# Prints the final Engine 1-oriented result and any exceptional termination
# counts that merit attention.
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

# Recovers advertised engine names from the newest record, falling back to
# executable filenames before any game has completed.
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

# Performs a short UCI handshake solely to obtain the advertised engine name.
def probe_engine_name(config: EngineConfig, label: str) -> str:
    with UciEngine(config, label=label) as engine:
        return engine.name

# Verifies the nonstandard legalmoves and optional gameresult commands before a
# long run starts, producing an immediate configuration error when unsupported.
def validate_rules_commands(
    config: EngineConfig,
    label: str,
    *,
    require_game_result: bool,
) -> None:
    with UciEngine(config, label=label) as engine:
        engine.new_game()
        engine.legal_moves([], None)
        if require_game_result:
            result = engine.game_result([], None)
            if result not in {"ongoing", "ry_win", "bg_win", "draw"}:
                raise EngineError(
                    f"{label} returned an invalid gameresult response: {result}"
                )

# Produces the human-readable search limit used in the start banner.
def control_text(config: MatchConfig) -> str:
    if config.limit_kind == "clock":
        return f"tc {config.base_time_ms}ms + {config.increment_ms}ms"
    if config.limit_kind == "movetime":
        return f"movetime {config.limit_value}ms"
    return f"{config.limit_kind} {config.limit_value}"

# Summarizes the match, opening policy, and scoring perspective before workers
# begin producing interleaved output.
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
    if config.opening_plies:
        if config.opening_nodes > 0:
            weights = ",".join(f"{weight:g}" for weight in config.opening_weights)
            balance = (
                f" | max score {config.opening_max_score} cp"
                if config.opening_max_score > 0
                else ""
            )
            print(
                f"Guided openings: {config.opening_plies} plies | "
                f"{config.opening_nodes} nodes | weights {weights}{balance}",
                flush=True,
            )
        else:
            print(
                f"Uniform-random openings: {config.opening_plies} plies",
                flush=True,
            )
    print("Results, percentage, and Elo are from Engine 1's perspective.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

# Appends one self-contained game record and closes the file immediately so
# completed games survive interruption.
def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

# Removes display and diagnostic fields while retaining everything needed to
# resume a match or derive position-level training samples.
def compact_training_record(record: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: record[key]
        for key in (
            "game_id",
            "pair",
            "game",
            "paired",
            "engine1_team",
            "fen",
            "opening",
            "moves",
            "result",
            "engine1_score",
            "termination",
            "illegal_move",
            "error",
        )
        if key in record
    }
    compact["searches"] = [
        {
            key: search[key]
            for key in ("ply", "team", "move", "depth", "nodes", "score", "fen")
            if key in search
        }
        for search in record.get("searches", [])
    ]
    return compact

# Renders position samples from the side-to-move team's perspective. Technical
# endings are excluded because they are unreliable game labels.
def nnue_lines(record: dict[str, Any]) -> list[str]:
    if record.get("termination") not in NNUE_GAME_TERMINATIONS:
        return []
    game_result = effective_game_result(record)
    lines: list[str] = []
    for search in record.get("searches", []):
        if not isinstance(search, dict):
            continue
        fen = search.get("fen")
        team = search.get("team")
        score = search.get("score")
        if (
            not isinstance(fen, str)
            or not fen
            or team not in {"ry", "bg"}
            or not isinstance(score, dict)
            or score.get("type") != "cp"
        ):
            continue
        try:
            centipawn = int(score["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if game_result == "draw":
            result = "0.5"
        elif game_result == f"{team}_win":
            result = "1"
        elif game_result in {"ry_win", "bg_win"}:
            result = "0"
        else:
            continue
        lines.append(f"| {fen} | {centipawn} | {result} |\n")
    return lines

# Appends every eligible position from one completed game.
def append_nnue(path: Path, record: dict[str, Any]) -> None:
    lines = nnue_lines(record)
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.writelines(lines)

# Recovers a schedule index from current fields or a legacy game identifier.
def record_index(record: dict[str, Any]) -> int:
    raw = record.get("game", record.get("pair"))
    if raw is not None:
        return int(raw)
    match = re.match(r"^(?:game|pair)(\d+)-", str(record.get("game_id", "")))
    if match is None:
        raise ValueError(f"Cannot determine game index: {record.get('game_id')}")
    return int(match.group(1))

# Replaces a JSON document through a sibling temporary file so readers never
# observe a partially written schedule, summary, or metadata file.
def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)

# Escapes a quoted PGN4 tag value.
def pgn4_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')

# Converts normalized coordinate moves to Chess.com-style PGN4 coordinates.
def pgn4_move(move: str) -> str:
    if len(move) == 1 or "-" in move:
        return move
    castle = move.upper()
    if castle in {"OO", "OOO", "O-O", "O-O-O"}:
        return "O-O-O" if castle.replace("-", "") == "OOO" else "O-O"
    match = re.fullmatch(
        r"([a-z]+\d+)([a-z]+\d+)(?:=?([nbrqk]))?",
        move.lower(),
    )
    if match is None:
        return move
    source, target, promotion = match.groups()
    suffix = f"={promotion.upper()}" if promotion else ""
    return f"{source}-{target}{suffix}"

# Encodes clock and fixed-movetime limits in the PGN4 TimeControl tag.
def pgn4_time_control(config: MatchConfig) -> str:
    if config.limit_kind == "clock":
        return f"{config.base_time_ms // 60000}+{config.increment_ms // 1000}"
    if config.limit_kind == "movetime":
        return f"0+{config.limit_value // 1000}"
    return "0+0"

# Maps team results to the two-team PGN result field.
def pgn4_result(result: str) -> str:
    return {
        "ry_win": "1-0",
        "bg_win": "0-1",
        "draw": "1/2-1/2",
    }.get(result, "*")

# Produces an explanatory Termination tag only for exceptional endings.
def pgn4_termination(record: dict[str, Any]) -> str | None:
    termination = str(record.get("termination", ""))
    if termination == "no_legal_moves" and effective_game_result(record) != "draw":
        return None
    labels = {
        "max_plies": "Draw due to max moves reached",
        "no_legal_moves": "Draw due to no legal moves",
        "runner_error": "Runner error",
    }
    if termination in labels:
        return labels[termination]
    if termination.endswith("_illegal_move"):
        return "Illegal move"
    if termination.endswith("_time_loss"):
        return "Loss on time"
    if termination.endswith("_timeout"):
        return "Engine timeout"
    return None

# Selects the Chess.com move-list marker for draws, time losses, and ordinary
# decisive endings. Technical errors are not mislabeled as checkmate.
def pgn4_terminal_marker(record: dict[str, Any]) -> str | None:
    result = effective_game_result(record)
    termination = str(record.get("termination", ""))
    if result == "draw":
        return "D" if termination == "max_plies" else "S"
    if result in {"ry_win", "bg_win"}:
        if termination.endswith("_time_loss") or termination.endswith("_timeout"):
            return "T"
        if termination.endswith("_illegal_move") or termination == "runner_error":
            return None
        return "#"
    return None

# Returns Red/Yellow and Blue/Green names after applying this game's team swap.
def pgn4_engine_names(
    record: dict[str, Any],
    engine1_name: str,
    engine2_name: str,
) -> tuple[str, str]:
    names = record.get("engine_names")
    if isinstance(names, dict):
        engine1_name = str(names.get("engine1", engine1_name))
        engine2_name = str(names.get("engine2", engine2_name))
    if record.get("engine1_team") == "ry":
        return engine1_name, engine2_name
    return engine2_name, engine1_name

# Serializes one game as standard multiline PGN4 or as a single transport line.
def serialize_pgn4_game(
    record: dict[str, Any],
    config: MatchConfig,
    engine1_name: str,
    engine2_name: str,
) -> str:
    red_name, blue_name = pgn4_engine_names(record, engine1_name, engine2_name)
    result = pgn4_result(effective_game_result(record))
    lines = [
        '[Variant "Teams"]',
        '[RuleVariants "EnPassant"]',
        f'[TimeControl "{pgn4_time_control(config)}"]',
        f'[Red "{pgn4_escape(red_name)}"]',
        f'[Blue "{pgn4_escape(blue_name)}"]',
        f'[Result "{result}"]',
    ]
    termination = pgn4_termination(record)
    if termination is not None:
        lines.append(f'[Termination "{pgn4_escape(termination)}"]')
    fen = record.get("fen")
    if fen:
        lines.append(f'[StartFen4 "{pgn4_escape(str(fen))}"]')

    moves = [str(move) for move in record.get("moves", [])]
    marker = pgn4_terminal_marker(record)
    if marker is not None:
        moves.append(marker)
    first_player = fen_turn_index(str(fen) if fen else None)
    move_tokens: list[str] = []
    for index, move in enumerate(moves):
        shifted = index + first_player
        formatted = pgn4_move(move)
        if shifted % 4 == 0 or index == 0:
            move_tokens.append(f"{shifted // 4 + 1}.")
        move_tokens.append(formatted)
    if config.pgn4_single_line:
        return " ".join(lines + move_tokens) + "\n"

    move_lines: list[str] = []
    for index, move in enumerate(moves):
        shifted = index + first_player
        formatted = pgn4_move(move)
        if shifted % 4 == 0 or index == 0:
            move_lines.append(f"{shifted // 4 + 1}. {formatted}")
        else:
            move_lines[-1] += f" .. {formatted}"
    return "\n".join(lines + [""] + move_lines) + "\n\n"

# Appends one completed game while keeping the output usable during long runs.
def append_pgn4(
    path: Path,
    record: dict[str, Any],
    config: MatchConfig,
    engine1_name: str,
    engine2_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(
            serialize_pgn4_game(record, config, engine1_name, engine2_name)
        )

# Adds a descriptive suffix without replacing the primary file's extension.
def sidecar(path: Path, suffix: str) -> Path:
    return path.with_name(path.name + suffix)

# Rejects aliases among user outputs, sidecars, and rebuild temporaries before
# any of them are created or truncated.
def validate_output_paths(paths: dict[str, Path | None]) -> None:
    seen: dict[Path, str] = {}
    for name, path in paths.items():
        if path is None:
            continue
        resolved = path.resolve()
        previous = seen.get(resolved)
        if previous is not None:
            raise ValueError(
                f"Output paths overlap: {previous} and {name} both use {resolved}"
            )
        seen[resolved] = name

# Opens the smallest engine set needed for opening generation. Guided openings
# alternate search providers; a dedicated arbiter supplies rules when present.
def open_opening_engines(
    config: MatchConfig,
    stack: contextlib.ExitStack,
) -> tuple[tuple[UciEngine, ...], UciEngine | None]:
    if config.opening_nodes > 0:
        multipv = len(config.opening_weights)
        providers = (
            stack.enter_context(UciEngine(
                opening_search_config(config.engine1, multipv),
                label="opening-search-engine1",
            )),
            stack.enter_context(UciEngine(
                opening_search_config(config.engine2, multipv),
                label="opening-search-engine2",
            )),
        )
        rules = (
            stack.enter_context(UciEngine(config.arbiter, label="opening-rules"))
            if config.arbiter is not None
            else None
        )
        return providers, rules

    if config.arbiter is not None:
        rules = stack.enter_context(UciEngine(config.arbiter, label="opening-rules"))
        return (rules,), rules

    return (
        stack.enter_context(UciEngine(
            config.engine1,
            label="opening-rules-engine1",
        )),
        stack.enter_context(UciEngine(
            config.engine2,
            label="opening-rules-engine2",
        )),
    ), None

# Materializes the in-memory schedule used by programmatic callers. A persisted
# schedule is reused only when it contains exactly the requested entry count.
def create_schedule(
    config: MatchConfig,
    entries: int,
    seed: int,
    path: Path | None,
    resume: bool,
) -> list[StartPosition]:
    if resume and path is not None and path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or len(raw) != entries:
            raise ValueError(f"Opening schedule does not contain {entries} entries")
        return [
            StartPosition(item.get("fen"), list(item["opening_moves"]), item["source"])
            for item in raw
        ]
    rng = random.Random(seed)
    fens = shuffled_fens(config.fens, entries, seed)
    if config.opening_plies:
        starts = []
        with contextlib.ExitStack() as stack:
            providers, rules = open_opening_engines(config, stack)
            first_provider = rng.randrange(len(providers))
            for entry_index, fen in enumerate(fens):
                provider_index = (first_provider + entry_index) % len(providers)
                opening_engine = providers[provider_index]
                starts.append(
                    generate_start(config, rng, fen, opening_engine, rules)
                )
    else:
        starts = [generate_start(config, rng, fen) for fen in fens]
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

# Streams a deterministic schedule to avoid retaining every opening in memory.
# Existing entries are validated and replayed before new entries are appended.
def iter_persistent_schedule(
    config: MatchConfig,
    entries: int,
    seed: int,
    path: Path | None,
    *,
    resume: bool,
) -> Iterator[tuple[int, StartPosition]]:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        path.open(encoding="utf-8")
        if path is not None and resume and path.is_file()
        else None
    )
    output = path.open("a", encoding="utf-8") if path is not None else None
    stack = contextlib.ExitStack()
    providers: tuple[UciEngine, ...] | None = None
    rules: UciEngine | None = None
    existing_exhausted = existing is None
    try:
        scheduled_fens = shuffled_fens(config.fens, entries, seed)
        for index, fen in enumerate(scheduled_fens, 1):
            raw = (
                existing.readline()
                if existing is not None and not existing_exhausted
                else ""
            )
            if not raw:
                existing_exhausted = True
            if raw:
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid opening schedule line {index}: {path}"
                    ) from exc
                if int(item.get("index", index)) != index:
                    raise ValueError(
                        f"Opening schedule index mismatch at line {index}: {path}"
                    )
                start = StartPosition(
                    item.get("fen"),
                    list(item["opening_moves"]),
                    str(item["source"]),
                )
            else:
                rng = random.Random(f"{seed}:{index}")
                opening_engine = None
                if config.opening_plies:
                    if providers is None:
                        providers, rules = open_opening_engines(config, stack)
                    opening_engine = providers[(index - 1) % len(providers)]
                start = generate_start(config, rng, fen, opening_engine, rules)
                if output is not None:
                    output.write(json.dumps({
                        "index": index,
                        "fen": start.fen,
                        "opening_moves": start.opening_moves,
                        "source": start.source,
                    }, separators=(",", ":")) + "\n")
                    output.flush()
            yield index, start
        if (
            existing is not None
            and not existing_exhausted
            and existing.readline()
        ):
            raise ValueError(
                f"Opening schedule contains more than {entries} entries: {path}"
            )
    finally:
        stack.close()
        if output is not None:
            output.close()
        if existing is not None:
            existing.close()

# Expands each scheduled opening into one unpaired task or two color-swapped
# paired tasks, skipping slots already present in resumed output.
def iter_tasks(
    starts: Iterable[tuple[int, StartPosition]],
    completed: CompletionTracker,
    *,
    paired: bool,
) -> Iterator[GameTask]:
    for index, start in starts:
        if paired:
            team_order = ("ry", "bg") if index % 2 else ("bg", "ry")
        else:
            team_order = ("ry",) if index % 2 else ("bg",)
        for team in team_order:
            if not completed.contains(index, team):
                yield GameTask(index, team, start, paired)

# Checks executable availability before spawning worker threads.
def validate_engine(config: EngineConfig, label: str) -> None:
    if not config.path.is_file():
        raise ValueError(f"{label} does not exist: {config.path}")
    if not os.access(config.path, os.X_OK):
        raise ValueError(f"{label} is not executable: {config.path}")


def validate_sprt_config(
    config: MatchConfig,
    *,
    continue_on_error: bool,
) -> None:
    sprt = config.sprt
    if sprt is None:
        return
    if not math.isfinite(sprt.elo0) or not math.isfinite(sprt.elo1):
        raise ValueError("SPRT Elo bounds must be finite")
    if sprt.elo0 >= sprt.elo1:
        raise ValueError("SPRT ELO0 must be less than ELO1")
    if not 0 < sprt.alpha < 1 or not 0 < sprt.beta < 1:
        raise ValueError("SPRT alpha and beta must be between 0 and 1")
    if sprt.alpha + sprt.beta >= 1:
        raise ValueError("SPRT alpha plus beta must be less than 1")
    if config.engine1.threads != config.engine2.threads:
        raise ValueError("SPRT requires equal engine thread counts")
    if config.engine1.hash_mb != config.engine2.hash_mb:
        raise ValueError("SPRT requires equal engine hash sizes")
    if continue_on_error:
        raise ValueError("SPRT cannot use continue-on-error")

# Builds the list form of iter_tasks() for in-process matches and older callers.
def build_tasks(
    starts: list[StartPosition],
    records_by_id: dict[str, dict[str, Any]] | None = None,
    *,
    paired: bool = True,
) -> list[GameTask]:
    completed = records_by_id or {}
    tasks: list[GameTask] = []
    for index, start in enumerate(starts, 1):
        if paired:
            team_order = ("ry", "bg") if index % 2 else ("bg", "ry")
        else:
            team_order = ("ry",) if index % 2 else ("bg",)
        for team in team_order:
            task = GameTask(index, team, start, paired)
            if task.game_id not in completed:
                tasks.append(task)
    return tasks

# Runs tasks concurrently with bounded lookahead, retries transient engine
# failures, and emits each finished record exactly once. Each executor thread
# retains its engine set until shutdown or an unhealthy result retires it.
def execute_tasks(
    config: MatchConfig,
    tasks: Iterable[GameTask],
    records_by_id: dict[str, dict[str, Any]] | None,
    *,
    continue_on_error: bool,
    retries: int = 2,
    on_record: Any = None,
    ordered: bool = False,
    should_stop: Any = None,
) -> bool:
    stop_event = threading.Event()
    active_engines = ActiveEngines()
    worker_registry = WorkerEngineRegistry()
    worker_state = threading.local()

    def run_task(task: GameTask) -> dict[str, Any]:
        engines = getattr(worker_state, "engines", None)
        if engines is None:
            engines = WorkerEngines(config, active_engines)
            worker_registry.add(engines)
            worker_state.engines = engines

        # A failed protocol exchange may leave unread output in the queue. Such
        # a set is closed rather than reused by the next game.
        def retire_engines() -> None:
            worker_registry.discard(engines)
            engines.close()
            if getattr(worker_state, "engines", None) is engines:
                del worker_state.engines

        try:
            record = play_game(
                config,
                task,
                stop_event,
                active_engines,
                reusable_engines=engines,
            )
        except BaseException:
            retire_engines()
            raise
        if str(record.get("termination", "")).endswith("_timeout"):
            retire_engines()
        return record

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.workers)
    futures: dict[
        concurrent.futures.Future[dict[str, Any]], tuple[GameTask, int, int]
    ] = {}
    ready: dict[
        int,
        tuple[GameTask, int, concurrent.futures.Future[dict[str, Any]]],
    ] = {}
    task_iterator = iter(tasks)
    next_order = 0
    next_commit = 0

    def submit_task(task: GameTask, attempt: int, order: int) -> None:
        futures[executor.submit(run_task, task)] = (task, attempt, order)

    def submit_next() -> bool:
        nonlocal next_order
        try:
            task = next(task_iterator)
        except StopIteration:
            return False
        submit_task(task, 0, next_order)
        next_order += 1
        return True

    def is_retryable_error(exc: BaseException) -> bool:
        return isinstance(exc, (EngineError, TimeoutError, OSError)) and not isinstance(
            exc, EngineOptionError
        )

    def save_record(
        task: GameTask,
        attempt: int,
        order: int,
        future: concurrent.futures.Future[dict[str, Any]],
    ) -> bool:
        try:
            record = future.result()
        except Exception as exc:
            if is_retryable_error(exc) and attempt < retries and not stop_event.is_set():
                print(
                    f"{task.game_id}: retrying after {type(exc).__name__}: {exc} "
                    f"({attempt + 1}/{retries})",
                    file=sys.stderr,
                    flush=True,
                )
                submit_task(task, attempt + 1, order)
                return False
            if not continue_on_error:
                raise
            record = {
                "game_id": task.game_id,
                "pair": task.pair_index if task.paired else None,
                "game": task.pair_index,
                "paired": task.paired,
                "engine1_team": task.engine1_team,
                "fen": task.start.fen,
                "opening": task.start.opening_moves,
                "moves": task.start.opening_moves,
                "result": "draw",
                "engine1_score": 0.5,
                "termination": "runner_error",
                "error": f"{type(exc).__name__}: {exc}",
                "attempts": attempt + 1,
            }
        if records_by_id is not None:
            records_by_id[task.game_id] = record
        if on_record is not None:
            on_record(record, records_by_id)
        return True

    # Keep at most two tasks queued per worker. This supplies work promptly
    # without materializing or submitting a very large schedule at once.
    try:
        for _ in range(max(config.workers, config.workers * 2)):
            if not submit_next():
                break
        statistical_stop = False
        while futures and not statistical_stop:
            done, _ = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if ordered:
                for future in done:
                    task, attempt, order = futures.pop(future)
                    ready[order] = (task, attempt, future)
                while next_commit in ready:
                    task, attempt, future = ready.pop(next_commit)
                    completed = save_record(task, attempt, next_commit, future)
                    if not completed:
                        break
                    next_commit += 1
                    if should_stop is not None and should_stop():
                        statistical_stop = True
                        break
                    submit_next()
            else:
                for future in done:
                    task, attempt, order = futures.pop(future)
                    completed = save_record(task, attempt, order, future)
                    if completed:
                        submit_next()
        if statistical_stop:
            stop_event.set()
            for future in futures:
                future.cancel()
            active_engines.close_all()
            executor.shutdown(wait=True, cancel_futures=True)
            worker_registry.close_all()
            return False
    except KeyboardInterrupt:
        stop_event.set()
        for future in futures:
            future.cancel()
        active_engines.close_all()
        executor.shutdown(wait=True, cancel_futures=True)
        worker_registry.close_all()
        return True
    except BaseException:
        stop_event.set()
        for future in futures:
            future.cancel()
        active_engines.close_all()
        executor.shutdown(wait=True, cancel_futures=True)
        worker_registry.close_all()
        raise
    else:
        executor.shutdown(wait=True)
        worker_registry.close_all()
        return False

# Runs an in-memory paired match for callers such as tuning tools. It validates
# executables, builds deterministic openings, and returns all records plus the
# Engine 1-oriented aggregate result.
def play_match(
    config: MatchConfig,
    *,
    pairs: int,
    seed: int,
    continue_on_error: bool = False,
    retries: int = 2,
    on_record: Any = None,
) -> MatchResult:
    if pairs < 1:
        raise ValueError("pairs must be at least 1")
    validate_sprt_config(config, continue_on_error=continue_on_error)
    for engine, label in (
        (config.engine1, "Engine 1"),
        (config.engine2, "Engine 2"),
    ):
        validate_engine(engine, label)
    if config.arbiter is not None:
        validate_engine(config.arbiter, "Arbiter")

    starts = create_schedule(config, pairs, seed, None, False)
    records_by_id: dict[str, dict[str, Any]] = {}
    stats = SummaryAccumulator(pairs * 2, paired=True, sprt=config.sprt)

    def collect_record(
        record: dict[str, Any],
        current_by_id: dict[str, dict[str, Any]] | None,
    ) -> None:
        stats.add(record)
        if on_record is not None:
            on_record(record, current_by_id)

    interrupted = execute_tasks(
        config,
        build_tasks(starts),
        records_by_id,
        continue_on_error=continue_on_error,
        retries=retries,
        on_record=collect_record,
        ordered=config.sprt is not None,
        should_stop=(
            (lambda: stats.sprt is not None and stats.sprt.terminal)
            if config.sprt is not None
            else None
        ),
    )
    records = list(records_by_id.values())
    name1, name2 = engine_names(records, config.engine1, config.engine2)
    summary = stats.summary(final=not interrupted)
    summary["engine1_name"] = name1
    summary["engine2_name"] = name2
    return MatchResult(records, summary, name1, name2, interrupted)

# Orchestrates the persistent command-line workflow: validates engines and
# outputs, verifies resume metadata, rebuilds derived files, streams unfinished
# tasks, and checkpoints each completed record.
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
    pgn4 = Path(args.pgn4).resolve() if args.pgn4 else None
    nnue_output = (
        default_nnue_output_path(engine1.path, engine2.path)
        if args.nnue_output == AUTO_NNUE_OUTPUT
        else Path(args.nnue_output).resolve()
        if args.nnue_output
        else None
    )
    config = MatchConfig(engine1, engine2, arbiter, args.limit_kind, args.limit_value, args.tc,
                        args.inc, args.timeout, args.margin, args.max_plies, args.opening_plies,
                        load_fens(args.fens), args.workers, args.moves,
                        opening_nodes=args.opening_nodes,
                        opening_weights=args.opening_weights,
                        opening_max_score=args.opening_max_score,
                        opening_attempts=args.opening_attempts,
                        pgn4_single_line=args.pgn4_single_line,
                        nnue_output=nnue_output is not None,
                        sprt=args.sprt_config,
    )
    validate_sprt_config(config, continue_on_error=args.continue_on_error)
    if arbiter is not None:
        validate_rules_commands(arbiter, "Arbiter", require_game_result=True)

    metadata_path = sidecar(out, ".meta.json") if out is not None else None
    schedule_path = sidecar(out, ".schedule.jsonl") if out is not None else None
    summary_path = sidecar(out, ".summary.json") if out is not None else None
    validate_output_paths({
        "--out": out,
        "metadata sidecar": metadata_path,
        "schedule sidecar": schedule_path,
        "summary sidecar": summary_path,
        "--pgn4": pgn4,
        "PGN4 temporary file": (
            pgn4.with_suffix(pgn4.suffix + ".tmp") if pgn4 is not None else None
        ),
        "--nnue-output": nnue_output,
        "NNUE temporary file": (
            nnue_output.with_suffix(nnue_output.suffix + ".tmp")
            if nnue_output is not None
            else None
        ),
    })
    payload = {
        "format": 2,
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
        "opening_nodes": args.opening_nodes,
        "opening_weights": args.opening_weights,
        "opening_max_score": args.opening_max_score,
        "opening_attempts": args.opening_attempts,
        "pgn4_single_line": args.pgn4_single_line,
        "fens": config.fens,
        "training_output": args.training_output,
        "nnue_output": nnue_output is not None,
    }
    if not args.paired:
        payload["mode"] = "unpaired"
        payload["games"] = args.game_count
    if config.sprt is not None:
        payload["sprt"] = {
            "model": "normalized",
            "elo0": config.sprt.elo0,
            "elo1": config.sprt.elo1,
            "alpha": config.sprt.alpha,
            "beta": config.sprt.beta,
        }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    # Resume is allowed only when the saved configuration describes the same
    # engines, schedule, limits, and output schema.
    if args.fresh:
        for path in (
            out, metadata_path, schedule_path, summary_path, pgn4, nnue_output
        ):
            if path is not None:
                path.unlink(missing_ok=True)
    if out is None and nnue_output is not None and nnue_output.exists():
        raise ValueError(
            f"NNUE output exists; use --fresh or another path: {nnue_output}"
        )
    if out is not None and out.exists() and not args.resume:
        raise ValueError(f"Output exists; use --resume or --fresh: {out}")
    if metadata_path is not None and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or not metadata_matches_payload(
            metadata, payload, fingerprint
        ):
            raise ValueError(
                f"Existing match configuration differs: {metadata_path}. "
                "Use the original settings, --fresh, or another --out."
            )
        if metadata.get("fingerprint") != fingerprint:
            atomic_json(
                metadata_path,
                {"fingerprint": fingerprint, "config": payload},
            )
    elif metadata_path is not None:
        atomic_json(metadata_path, {"fingerprint": fingerprint, "config": payload})

    name1 = probe_engine_name(engine1, "engine1-probe")
    name2 = probe_engine_name(engine2, "engine2-probe")
    completed = CompletionTracker(args.schedule_count, paired=args.paired)
    stats = SummaryAccumulator(
        args.game_count,
        paired=args.paired,
        sprt=config.sprt,
    )
    resume_existing = (
        out is not None and out.is_file() and args.resume and not args.fresh
    )
    resume_schedule = (
        schedule_path is not None
        and schedule_path.is_file()
        and args.resume
        and not args.fresh
    )
    # PGN4 and NNUE are derived from game JSONL. Rebuilding them through
    # temporaries prevents duplicate samples when a run resumes.
    pgn_temp = (
        pgn4.with_suffix(pgn4.suffix + ".tmp")
        if pgn4 is not None and resume_existing
        else None
    )
    pgn_rebuild = None
    if pgn_temp is not None:
        pgn_temp.parent.mkdir(parents=True, exist_ok=True)
        pgn_rebuild = pgn_temp.open("w", encoding="utf-8")
    nnue_temp = (
        nnue_output.with_suffix(nnue_output.suffix + ".tmp")
        if nnue_output is not None
        else None
    )
    nnue_rebuild = None
    if nnue_temp is not None:
        nnue_temp.parent.mkdir(parents=True, exist_ok=True)
        nnue_rebuild = nnue_temp.open("w", encoding="utf-8")
    try:
        if resume_existing and out is not None:
            with out.open(encoding="utf-8") as rows:
                for line_number, raw in enumerate(rows, 1):
                    if not raw.strip():
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {out} line {line_number}"
                        ) from exc
                    index = record_index(record)
                    team = str(record.get("engine1_team", ""))
                    if not completed.add(index, team):
                        continue
                    stats.add(record)
                    if pgn_rebuild is not None:
                        pgn_rebuild.write(
                            serialize_pgn4_game(
                                record, config, name1, name2
                            )
                        )
                    if nnue_rebuild is not None:
                        nnue_rebuild.writelines(nnue_lines(record))
    finally:
        if pgn_rebuild is not None:
            pgn_rebuild.close()
        if nnue_rebuild is not None:
            nnue_rebuild.close()
    if pgn_temp is not None and pgn4 is not None:
        pgn_temp.replace(pgn4)
    elif pgn4 is not None:
        pgn4.parent.mkdir(parents=True, exist_ok=True)
        pgn4.write_text("", encoding="utf-8")
    if nnue_temp is not None and nnue_output is not None:
        nnue_temp.replace(nnue_output)

    print_start_banner(config, args.game_count, name1, name2)
    if nnue_output is not None:
        print(f"NNUE output: {nnue_output}", flush=True)
    if stats.games_completed:
        print(
            f"Resuming with {stats.games_completed} completed games.",
            flush=True,
        )
    starts = iter_persistent_schedule(
        config,
        args.schedule_count,
        args.seed,
        schedule_path,
        resume=resume_schedule,
    )
    tasks = iter_tasks(starts, completed, paired=args.paired)
    last_sprt_report: tuple[int, int, str] | None = None

    # The completion tracker is the single gate for persistence and statistics,
    # so retries or duplicate resume records cannot be emitted twice.
    def on_record(
        record: dict[str, Any],
        _current_by_id: dict[str, dict[str, Any]] | None,
    ) -> None:
        nonlocal last_sprt_report
        index = record_index(record)
        team = str(record.get("engine1_team", ""))
        if not completed.add(index, team):
            return
        persisted = (
            compact_training_record(record)
            if args.training_output
            else record
        )
        if out is not None:
            append_jsonl(out, persisted)
        if nnue_output is not None:
            append_nnue(nnue_output, record)
        pairs_before = stats.pairs_completed
        stats.add(record)
        completed_pair = stats.pairs_completed > pairs_before
        reached_cap = stats.games_completed >= stats.games_expected
        summary = stats.summary(final=reached_cap)
        summary["engine1_name"] = name1
        summary["engine2_name"] = name2
        if (
            summary_path is not None
            and stats.games_completed % args.summary_interval == 0
        ):
            atomic_json(summary_path, summary)
        if pgn4 is not None:
            append_pgn4(pgn4, record, config, name1, name2)
        if config.sprt is not None and completed_pair and not args.quiet:
            print_sprt_report(summary, config)
            last_sprt_report = (
                summary["games_completed"],
                summary["pairs_completed"],
                summary["sprt"]["state"],
            )
        elif config.sprt is None and not args.quiet:
            print_summary(summary, name1, name2, record)

    already_decided = stats.sprt is not None and stats.sprt.terminal
    if already_decided:
        interrupted = False
    else:
        interrupted = execute_tasks(
            config,
            tasks,
            None,
            continue_on_error=args.continue_on_error,
            retries=args.retries,
            on_record=on_record,
            ordered=config.sprt is not None,
            should_stop=(
                (lambda: stats.sprt is not None and stats.sprt.terminal)
                if config.sprt is not None
                else None
            ),
        )
    summary = stats.summary(final=not interrupted)
    summary["engine1_name"] = name1
    summary["engine2_name"] = name2
    if summary_path is not None:
        atomic_json(summary_path, summary)
    if config.sprt is not None and summary["pairs_completed"]:
        current_report = (
            summary["games_completed"],
            summary["pairs_completed"],
            summary["sprt"]["state"],
        )
        if current_report != last_sprt_report:
            print_sprt_report(summary, config)
            last_sprt_report = current_report
    if interrupted:
        if config.sprt is None:
            print_final_summary(summary, name1, name2, interrupted=True)
        message = (
            "Completed games are saved and can be resumed."
            if out is not None
            else ""
        )
        if message:
            print(message, file=sys.stderr)
        return 130

    if config.sprt is None:
        print_final_summary(summary, name1, name2, interrupted=False)
    return 0


# Runs a resumable NNUE corpus for every requested seed. The compact JSONL is
# retained as the authoritative checkpoint so the derived text file can be
# rebuilt without duplicate positions after an interruption.
def run_nnue_data_seeds(args: argparse.Namespace) -> int:
    seeds = list(args.nnue_data_seeds)
    for position, seed in enumerate(seeds, 1):
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        seed_args.out = f"engine_seed_{seed}.jsonl"
        seed_args.nnue_output = f"engine_seed_{seed}.txt"
        print(
            f"\nNNUE seed {seed} ({position}/{len(seeds)}): "
            f"{seed_args.game_count:,} games -> {seed_args.nnue_output}",
            flush=True,
        )
        result = run_match(seed_args)
        if result:
            return result
    return 0


# Returns whether an option was explicitly supplied, including --name=value.
def cli_option_present(argv: list[str], *names: str) -> bool:
    return any(
        token == name or token.startswith(name + "=")
        for token in argv
        for name in names
    )


# Applies the NNUE generation profile while retaining explicit search/opening
# overrides. Per-seed output paths are intentionally automatic and unambiguous.
def apply_nnue_data_profile(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> None:
    if not args.nnue_data_seeds:
        return
    if len(set(args.nnue_data_seeds)) != len(args.nnue_data_seeds):
        parser.error("--nnue-data-seeds cannot contain duplicate seeds")
    if args.pairs is not None:
        parser.error("--nnue-data-seeds cannot be combined with --pairs")
    if args.sprt:
        parser.error("--nnue-data-seeds cannot be combined with --sprt")
    if args.out or args.nnue_output or args.pgn4:
        parser.error(
            "--nnue-data-seeds creates its own per-seed --out and "
            "--nnue-output files"
        )
    if cli_option_present(argv, "--seed"):
        parser.error("use --nnue-data-seeds instead of --seed in NNUE data mode")

    if args.games is None:
        args.games = NNUE_DATA_DEFAULT_GAMES
    args.unpaired = True
    args.training_output = True
    if not cli_option_present(argv, "--opening-plies"):
        args.opening_plies = NNUE_DATA_DEFAULT_OPENING_PLIES
    if not cli_option_present(argv, "--opening-nodes"):
        args.opening_nodes = NNUE_DATA_DEFAULT_OPENING_NODES


# Defines the complete public command-line surface without performing I/O.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run matches between two 4PC UCI engines"
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
    games.add_argument("--games", type=int, help="total games")
    parser.add_argument(
        "--sprt",
        action="store_true",
        help="enable normalized-Elo SPRT; requires an explicit --pairs maximum",
    )
    parser.add_argument(
        "--sprt-elo0",
        type=float,
        default=None,
        help="SPRT rejection hypothesis in normalized Elo (default: 0)",
    )
    parser.add_argument(
        "--sprt-elo1",
        type=float,
        default=None,
        help="SPRT acceptance hypothesis in normalized Elo (default: 5)",
    )
    parser.add_argument(
        "--sprt-alpha",
        type=float,
        default=None,
        help="SPRT false-positive limit (default: 0.05)",
    )
    parser.add_argument(
        "--sprt-beta",
        type=float,
        default=None,
        help="SPRT false-negative limit (default: 0.05)",
    )
    parser.add_argument(
        "--unpaired",
        action="store_true",
        help="run independent games; requires --games and allows odd counts",
    )
    parser.add_argument("--workers", type=int, default=1)
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--nodes", type=int)
    limit.add_argument("--depth", type=int)
    limit.add_argument("--movetime", "--fixed", type=int)
    limit.add_argument("--tc", type=int, help="base time in milliseconds")
    parser.add_argument("--inc", type=int, default=0, help="increment in milliseconds")
    parser.add_argument("--margin", type=int, default=50, help="clock-loss margin in milliseconds")
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help=(
            "engine response timeout in seconds; with --tc, this is grace time "
            "after the active color's remaining clock"
        ),
    )
    parser.add_argument("--max-plies", "--maxmoves", type=int, default=1000)
    parser.add_argument(
        "--opening-plies",
        type=int,
        default=0,
        help="number of randomized opening plies shared by paired games",
    )
    parser.add_argument(
        "--opening-nodes",
        type=int,
        default=0,
        help="nodes per guided opening move; 0 keeps uniform legal-move randomization",
    )
    parser.add_argument(
        "--opening-weights",
        type=parse_opening_weights,
        default=(60.0, 30.0, 10.0),
        metavar="W1,W2,...",
        help="relative probabilities for ranked MultiPV opening moves",
    )
    parser.add_argument(
        "--opening-max-score",
        type=int,
        default=1000,
        metavar="CP",
        help="reject guided openings exceeding this absolute score; 0 disables",
    )
    parser.add_argument(
        "--opening-attempts",
        type=int,
        default=100,
        help="maximum attempts to generate a balanced guided opening",
    )
    parser.add_argument("--fens", default="", help="opening FEN file")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--nnue-data-seeds",
        type=int,
        nargs="+",
        metavar="N",
        default=None,
        help=(
            "generate one resumable NNUE corpus per seed; defaults to 50,000 "
            "unpaired games, 10,000 nodes, 12 opening plies, 5,000 opening "
            "nodes, and opening weights 60,30,10"
        ),
    )
    parser.add_argument(
        "--out",
        default="",
        help="optional JSONL output path; enables persistence and resume",
    )
    parser.add_argument(
        "--compact-output",
        dest="training_output",
        action="store_true",
        help="write compact JSONL records",
    )
    parser.add_argument(
        "--nnue-output",
        nargs="?",
        const=AUTO_NNUE_OUTPUT,
        metavar="PATH",
        default="",
        help=(
            "write side-to-move '| FEN | CENTIPAWN | RESULT |' samples; "
            "without PATH, use ENGINE-vs-ENGINE-YYYY-MM-DD.txt"
        ),
    )
    parser.add_argument(
        "--summary-interval",
        type=int,
        default=100,
        help="update the summary file every N completed games",
    )
    parser.add_argument(
        "--pgn4",
        nargs="?",
        const="games.pgn4",
        default="",
        metavar="PATH",
        help="save Chess.com-style PGN4; default path: games.pgn4",
    )
    parser.add_argument(
        "--pgn4-single-line",
        action="store_true",
        help="write each PGN4 game on one line instead of standard multiline format",
    )
    parser.add_argument("--moves", "--pmoves", action="store_true", help="show move search data")
    parser.add_argument("--quiet", action="store_true", help="only print the final result")
    parser.add_argument("--continue-on-error", "--continue", action="store_true")
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="retry runner/engine failures before aborting or recording an error",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fresh", action="store_true")
    return parser

# Normalizes paired/unpaired counts and search limits, validates cross-option
# constraints, then translates runner failures into stable process exit codes.
def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    apply_nnue_data_profile(args, parser, sys.argv[1:])
    if args.sprt and args.pairs is None:
        parser.error("--sprt requires an explicit --pairs maximum")
    if not args.sprt and (
        args.sprt_elo0 is not None
        or args.sprt_elo1 is not None
        or args.sprt_alpha is not None
        or args.sprt_beta is not None
    ):
        parser.error("--sprt settings require --sprt")
    if args.games is not None:
        if args.games < 1:
            parser.error("--games must be at least 1")
        if args.unpaired:
            args.paired = False
            args.pairs = None
            args.game_count = args.games
            args.schedule_count = args.games
        else:
            if args.games < 2 or args.games % 2:
                parser.error("--games must be an even number of at least 2 unless --unpaired is used")
            args.paired = True
            args.pairs = args.games // 2
            args.game_count = args.games
            args.schedule_count = args.pairs
    elif args.unpaired:
        parser.error("--unpaired requires --games")
    elif args.pairs is None:
        args.paired = True
        args.pairs = 1
        args.game_count = 2
        args.schedule_count = 1
    else:
        args.paired = True
        args.game_count = args.pairs * 2
        args.schedule_count = args.pairs
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
    if args.paired and args.pairs < 1:
        parser.error("--pairs must be at least 1")
    if args.workers < 1 or args.threads1 < 1 or args.threads2 < 1:
        parser.error("worker and engine thread counts must be at least 1")
    if args.hash1 < 1 or args.hash2 < 1:
        parser.error("engine hash sizes must be at least 1 MB")
    if args.arbiter and args.arbiter_hash < 1:
        parser.error("--arbiter-hash must be at least 1 MB")
    if args.limit_kind != "clock" and args.limit_value <= 0:
        parser.error(f"--{args.limit_kind} must be greater than 0")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    if args.max_plies < 1:
        parser.error("--max-plies must be at least 1")
    if args.inc < 0:
        parser.error("--inc must be at least 0")
    if args.margin < 0:
        parser.error("--margin must be at least 0")
    if args.retries < 0:
        parser.error("--retries must be at least 0")
    if args.summary_interval < 1:
        parser.error("--summary-interval must be at least 1")
    if args.opening_plies < 0:
        parser.error("--opening-plies must be at least 0")
    if args.opening_nodes < 0:
        parser.error("--opening-nodes must be at least 0")
    if args.opening_nodes and not args.opening_plies:
        parser.error("--opening-nodes requires --opening-plies")
    if args.opening_max_score < 0:
        parser.error("--opening-max-score must be at least 0")
    if args.opening_attempts < 1:
        parser.error("--opening-attempts must be at least 1")
    if args.training_output and not (args.out or args.nnue_data_seeds):
        parser.error("--compact-output requires --out")
    if args.pgn4_single_line and not args.pgn4:
        parser.error("--pgn4-single-line requires --pgn4")
    if args.limit_kind == "clock" and args.tc <= 0:
        parser.error("--tc must be greater than 0")
    if args.limit_kind != "clock" and args.inc:
        parser.error("--inc requires --tc")
    if not args.arbiter and (args.arbiter_options or args.arbiter_option):
        parser.error("--arbiter-options and --arbiter-option require --arbiter")
    if args.sprt:
        elo0 = 0.0 if args.sprt_elo0 is None else args.sprt_elo0
        elo1 = 5.0 if args.sprt_elo1 is None else args.sprt_elo1
        alpha = 0.05 if args.sprt_alpha is None else args.sprt_alpha
        beta = 0.05 if args.sprt_beta is None else args.sprt_beta
        if not math.isfinite(elo0) or not math.isfinite(elo1):
            parser.error("--sprt bounds must be finite")
        if elo0 >= elo1:
            parser.error("--sprt ELO0 must be less than ELO1")
        if not 0 < alpha < 1 or not 0 < beta < 1:
            parser.error("--sprt-alpha and --sprt-beta must be between 0 and 1")
        if alpha + beta >= 1:
            parser.error("--sprt-alpha plus --sprt-beta must be less than 1")
        if args.continue_on_error:
            parser.error("--sprt cannot be combined with --continue-on-error")
        if args.threads1 != args.threads2:
            parser.error("--sprt requires equal --threads1 and --threads2")
        if args.hash1 != args.hash2:
            parser.error("--sprt requires equal --hash1 and --hash2")
        args.sprt_config = SprtConfig(elo0, elo1, alpha, beta)
    else:
        args.sprt_config = None
    if args.fresh and not (
        args.out or args.pgn4 or args.nnue_output or args.nnue_data_seeds
    ):
        parser.error("--fresh requires an output option")
    if not args.resume and not (args.out or args.nnue_data_seeds):
        parser.error("--no-resume requires --out")
    try:
        return (
            run_nnue_data_seeds(args)
            if args.nnue_data_seeds
            else run_match(args)
        )
    except KeyboardInterrupt:
        message = (
            "Interrupted; completed games are saved and can be resumed."
            if args.out or args.nnue_data_seeds
            else "Interrupted."
        )
        print(message, file=sys.stderr)
        return 130
    except (EngineError, TimeoutError, OSError, ValueError) as exc:
        print(f"match: error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
