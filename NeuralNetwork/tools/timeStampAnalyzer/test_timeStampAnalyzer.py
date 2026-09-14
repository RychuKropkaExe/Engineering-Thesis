"""Regression tests using synthetic logs and the repository's example log."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import unittest

from timeStampAnalyzer import analyze, format_run, main


def record(kind, name, timestamp):
    return f"[TIME_STAMP]: EVENT_{kind} {name} {timestamp}\n"


class TimeStampAnalyzerTest(unittest.TestCase):
    def test_multiple_runs_repeated_events_and_outside_events(self):
        lines = [
            record("BEGIN", "OUTSIDE", 0),
            record("BEGIN", "DTM_GENETIC_RUN", 10),
            record("BEGIN", "WORK", 12),
            record("END", "WORK", 16),
            record("BEGIN", "WORK", 20),
            record("END", "WORK", 26),
            record("END", "DTM_GENETIC_RUN", 30),
            record("END", "OUTSIDE", 31),
            "[INFO]: ordinary log\n",
            record("BEGIN", "DTM_GENETIC_RUN", 40),
            record("BEGIN", "WORK", 41),
            record("END", "WORK", 43),
            record("END", "DTM_GENETIC_RUN", 50),
        ]
        runs = list(analyze(iter(lines)))
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0].duration_us, 20)
        self.assertEqual(runs[0].events["WORK"].duration_us, 10)
        self.assertEqual(runs[0].events["WORK"].calls, 2)
        self.assertEqual(runs[1].events["WORK"].duration_us, 2)
        self.assertNotIn("OUTSIDE", runs[0].events)
        self.assertIn("50.00%", format_run(runs[0]))
        self.assertIn("20.00%", format_run(runs[1]))

    def test_nested_same_name_uses_last_begin(self):
        run = list(analyze([
            record("BEGIN", "DTM_GENETIC_RUN", 0),
            record("BEGIN", "WORK", 0),
            record("BEGIN", "WORK", 2),
            record("END", "WORK", 4),
            record("END", "WORK", 10),
            record("END", "DTM_GENETIC_RUN", 10),
        ]))[0]
        self.assertEqual(run.events["WORK"].duration_us, 12)
        self.assertEqual(run.events["WORK"].calls, 2)
        self.assertIn("120.00%", format_run(run))

    def test_zero_duration(self):
        run = list(analyze([
            record("BEGIN", "DTM_GENETIC_RUN", 100),
            record("BEGIN", "FAST", 100),
            record("END", "FAST", 100),
            record("END", "DTM_GENETIC_RUN", 100),
        ]))[0]
        self.assertEqual(run.events["FAST"].calls, 1)
        self.assertIn("N/A", format_run(run))

    def test_incomplete_runs_and_pairs_are_not_matched_across_runs(self):
        warnings = []
        runs = list(analyze([
            record("BEGIN", "DTM_GENETIC_RUN", 0),
            record("BEGIN", "WORK", 1),
            record("BEGIN", "DTM_GENETIC_RUN", 2),
            record("END", "WORK", 3),
            record("BEGIN", "UNFINISHED", 4),
            record("END", "DTM_GENETIC_RUN", 5),
            record("BEGIN", "DTM_GENETIC_RUN", 6),
        ], warnings.append))
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].number, 2)
        self.assertEqual(runs[0].events, {})
        self.assertEqual(len(warnings), 4)

    def test_invalid_records_and_decreasing_time(self):
        invalid = [
            "[TIME_STAMP]: EVENT_BEGIN BROKEN\n",
            record("OTHER", "BROKEN", 1),
            record("BEGIN", "BROKEN", -1),
            record("BEGIN", "BROKEN", "not-a-number"),
            record("END", "DTM_GENETIC_RUN", 9),
        ]
        for line in invalid:
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, "line 2"):
                list(analyze([record("BEGIN", "DTM_GENETIC_RUN", 10), line]))

    def test_example_log_and_cli(self):
        example = Path(__file__).resolve().parents[3] / "exampleTimeStampLog.log"
        with example.open() as log:
            runs = list(analyze(log))
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].duration_us, 21)
        expected = {
            "DTM_POPULATION_INITIALIZATION": 4,
            "DTM_MODEL_ADD_OUT_SYNAPSE": 1,
            "DTM_SPECIATION": 1,
            "DTM_EVALUATE_FITNESS": 1,
            "DTM_TOURNAMENT_SELECTION": 2,
            "DTM_CROSSOVER": 10,
            "DTM_MUTATE_POPULATION": 3,
            "DTM_MODEL_ADD_NEURON": 3,
        }
        self.assertEqual(
            {name: event.duration_us for name, event in runs[0].events.items()}, expected
        )
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main([str(example)]), 0)
        self.assertIn("47.62%", output.getvalue())

    def test_missing_file_and_no_runs(self):
        with redirect_stderr(StringIO()) as errors:
            self.assertEqual(main([str(Path(__file__) / "missing.log")]), 1)
            self.assertIn("Error:", errors.getvalue())
        self.assertEqual(list(analyze(["ordinary log\n"])), [])


if __name__ == "__main__":
    unittest.main()
