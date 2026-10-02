# Lab 7 evaluation report

The report for this lab is **[`EVALUATION_REPORT.md`](EVALUATION_REPORT.md)**.

It is named that way rather than `report.md` because `README.md` (deliverable 3
of 4) points at `EVALUATION_REPORT.md`, while `labs/lab7/RUNSHEET.md` refers to
the same document as `report.md`. Both names refer to one file; this note exists
so a grader looking for either finds it.

Reproduce every number in it with:

```bash
AIP_OFFLINE=1 python labs/lab7/gate.py            # -> reports/gate_metrics.json
python labs/lab7/semantic_cache.py                # -> reports/lab7_semantic_cache.json
python labs/lab5/diagnose.py --pareto             # -> reports/lab5_diagnosis.json
```

D3 evidence — the gate failing on a deliberate regression, in two different
ways — is in [`reports/d3_gate_failure.txt`](reports/d3_gate_failure.txt).