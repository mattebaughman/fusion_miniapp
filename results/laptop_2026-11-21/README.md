# Laptop ladder, 2026-11-21 (the IGNOS outer loop)

The run records only. The per-campaign files are left out because of their size. These replace the
2026-11-19 ladder in `../laptop_2026-11-19_final/` as the reference; that one predates the outer loop.
Recorded on the development laptop (Apple silicon, 12 cores, 24 GB, macOS, torch 2.15.0 CPU) with the
stub model. All physics output is synthetic. These are functional checks, not performance claims.

- `rung_a/`: 1,000 campaigns x P 4 x G 2 = 8,000 real stand-in evaluations on the CPU, 1 agent process and
  4 workers (`rung_a.yaml`, `commands.txt`).
  - Both layers pass, with 8,000 of 8,000 touch checks and the same `results_sha256`.
  - LangGraph: 77.7 s, 487 MiB. Academy: 75.4 s, 348 MiB.
- `rung_b/`: 10,000 campaigns x P 16 x G 2 = 320,000 fixture evaluations, 1 agent process and 10 threads.
  - Both layers pass, with all 10,000 campaigns live at once and the same `results_sha256`.
  - LangGraph: 462 s, 6,142 MiB. Academy: 172 s, 3,553 MiB.
  - The LangGraph run overlapped a Docker image build, and the Academy run did not, so their times are not
    a clean comparison.
- The 100,000-campaign rung was not re-run here: at P 16 it needs about 60 GiB (LangGraph) or 35 GiB
  (Academy) of host memory, more than this laptop has.

Before publishing, the laptop's host name in the run records was replaced by `laptop`, and the
home-directory prefix of `constraints_path` in the campaign configs was removed. Nothing else was changed.
