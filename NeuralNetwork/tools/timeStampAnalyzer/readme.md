# Timestamp analyzer

Note: Tool written with heavy usage of AI

Summarize each completed `DTM_GENETIC_RUN` in a timestamp log using Python 3.8+ and the standard library only.

From the repository root:

```bash
python3 NeuralNetwork/tools/timeStampAnalyzer/timeStampAnalyzer.py exampleTimeStampLog.log
python3 NeuralNetwork/tools/timeStampAnalyzer/timeStampAnalyzer.py NeuralNetwork/build/timeStampLog.log
```

The report lists event names, completed call counts, total microseconds, and percentage of the enclosing run's duration, ordered by total time. Each run is reported separately. `DTM_GENETIC_RUN` itself supplies the total duration in the report header.

Each duration is `EVENT_END - EVENT_BEGIN`. Timings are **inclusive**: a parent event includes time spent in nested events, so percentages across rows need not sum to 100%. Repeated nested occurrences of the same name are paired last-in, first-out. Percentages are `N/A` for a zero-microsecond run.

The tool reads line by line and retains only the current run's totals and open event pairs. Ordinary log lines and events outside runs are ignored. Unmatched pairs produce warnings on stderr and are excluded. An unfinished run is discarded at the end of the file or when a new run begins; genetic runs are assumed not to nest. Malformed timestamp records and decreasing timestamps within a run are errors. Exit status is `1` on error or if no complete runs are found, otherwise `0`.

Run regression tests from the repository root:

```bash
python3 -m unittest discover -s NeuralNetwork/tools/timeStampAnalyzer -v
```
