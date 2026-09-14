#!/usr/bin/env python3
"""Stream timestamp logs and summarize inclusive timings per genetic run."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Callable, Dict, Iterable, Iterator, List, Optional


RUN_EVENT = "DTM_GENETIC_RUN"


@dataclass
class EventTotals:
    """Accumulated microseconds and number of completed pairs for one name."""

    duration_us: int = 0
    calls: int = 0


@dataclass
class RunResult:
    """One completed run and the inclusive totals of events inside it."""

    number: int
    start_us: int
    end_us: int
    events: Dict[str, EventTotals] = field(default_factory=dict)

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us


def analyze(
    lines: Iterable[str], warn: Optional[Callable[[str], None]] = None
) -> Iterator[RunResult]:
    """Yield each completed run without retaining the log or previous runs.

    Pair repeated/nested event names using a stack per name. Events outside a
    run are ignored. Unmatched pairs are reported and excluded; an unfinished
    run is discarded at EOF or at the next run's beginning. The logger has a
    single producer, so timestamps inside a run must be nondecreasing.
    """
    run_number = 0
    run_start = None
    last_timestamp = 0
    starts: Dict[str, List[int]] = {}
    totals: Dict[str, EventTotals] = {}

    def report(message: str) -> None:
        if warn is not None:
            warn(message)

    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if not fields or fields[0] != "[TIME_STAMP]:":
            continue
        if (
            len(fields) != 4
            or fields[1] not in ("EVENT_BEGIN", "EVENT_END")
            or not fields[3].isascii()
            or not fields[3].isdigit()
        ):
            raise ValueError(f"line {line_number}: malformed timestamp record")

        _, event_type, name, value = fields
        timestamp = int(value)
        if name == RUN_EVENT and event_type == "EVENT_BEGIN":
            if run_start is not None:
                report(
                    f"line {line_number}: discarding incomplete run {run_number} "
                    "because another DTM_GENETIC_RUN began"
                )
            run_number += 1
            run_start = timestamp
            last_timestamp = timestamp
            starts = {}
            totals = {}
            continue

        if run_start is None:
            continue
        if timestamp < last_timestamp:
            raise ValueError(
                f"line {line_number}: timestamp decreased inside run {run_number}"
            )
        last_timestamp = timestamp

        if name == RUN_EVENT:  # EVENT_END: the current run is complete.
            incomplete = sum(len(stack) for stack in starts.values())
            if incomplete:
                report(
                    f"run {run_number}: excluding {incomplete} event begin(s) "
                    "without matching ends"
                )
            yield RunResult(run_number, run_start, timestamp, totals)
            run_start = None
            starts = {}
            totals = {}
        elif event_type == "EVENT_BEGIN":
            starts.setdefault(name, []).append(timestamp)
        else:
            stack = starts.get(name)
            if not stack:
                report(
                    f"line {line_number}: ignoring unmatched EVENT_END for "
                    f"{name} in run {run_number}"
                )
                continue
            duration = timestamp - stack.pop()
            event = totals.setdefault(name, EventTotals())
            event.duration_us += duration
            event.calls += 1

    if run_start is not None:
        report(f"end of file: discarding incomplete run {run_number}")


def format_run(run: RunResult) -> str:
    """Format totals in descending duration order, including zero-time pairs."""
    rows = sorted(run.events.items(), key=lambda item: (-item[1].duration_us, item[0]))
    width = max([len("Event")] + [len(name) for name in run.events])
    output = [
        f"Run {run.number}: {RUN_EVENT}",
        f"  Start: {run.start_us:,} us | End: {run.end_us:,} us | "
        f"Total: {run.duration_us:,} us ({run.duration_us / 1_000_000:.6f} s)",
        f"  {'Event':<{width}}  {'Calls':>10}  {'Total (us)':>16}  {'% of run':>10}",
        f"  {'-' * width}  {'-' * 10}  {'-' * 16}  {'-' * 10}",
    ]
    for name, event in rows:
        percentage = (
            f"{100 * event.duration_us / run.duration_us:.2f}%"
            if run.duration_us else "N/A"
        )
        output.append(
            f"  {name:<{width}}  {event.calls:>10,}  "
            f"{event.duration_us:>16,}  {percentage:>10}"
        )
    if not rows:
        output.append("  No completed inner events.")
    return "\n".join(output)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path, help="path to timeStampLog.log")
    args = parser.parse_args(argv)

    def warn(message: str) -> None:
        print(f"Warning: {message}", file=sys.stderr)

    completed = 0
    try:
        with args.log_file.open(encoding="utf-8") as log:
            for run in analyze(log, warn):
                if not completed:
                    print("Inclusive timings: nested events overlap; percentages need not sum to 100%.\n")
                print(format_run(run), end="\n\n")
                completed += 1
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if not completed:
        print("No complete DTM_GENETIC_RUN pairs found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
