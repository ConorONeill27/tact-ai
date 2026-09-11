"""Analysis subpackage: results, figures and the final markdown report.

Phase 4 implements ``results.py`` (JSON -> tidy TrialRecords), ``plots.py``
(Agg figures) and ``report.py`` (``results/RESULTS.md``). ``run.py --mode
analyze`` drives them through the daemon-facing entry points
``load_results`` / ``make_plots`` / ``run_report``.
"""

from __future__ import annotations


def available() -> bool:
    """True once the Phase-4 analysis modules are implemented."""
    return True