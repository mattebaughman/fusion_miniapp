# The reference problem: `ignos_shape`

```sh
python -m fusion_miniapp.run --preset ignos_shape --sim real --knobs results/ignos_shape_reference/cpu.yaml --out DIR
```

4 campaigns (seeds 0 to 3), each the IGNOS outer loop at population 16 for 20 generations, with the real
stand-in evaluate on the CPU and the stub model: 1,280 simulations and 80 model calls. Recorded on
2026-11-21 on the development laptop (Apple silicon, macOS, torch 2.15.0 CPU). All physics output is
synthetic.

- `run_record.json` (LangGraph layer) and `run_record_academy.json` (Academy layer): both `status: pass`,
  1,280 of 1,280 touch checks passed, and the same `results_sha256` (`2a9e4fb71c35ad57...`).
- `agents.jsonl`: the four campaigns' results.
- `campaigns/a0..a3/`: the parent's `config.json`, `evolution_log.csv`, `evolution_log.jsonl` and
  `best_result.json`.
  - Between the two layers, `evolution_log.csv` and `best_result.json` are byte-identical.
  - `evolution_log.jsonl` differs only in its timestamps, and `config.json` only in its git provenance.
- `cpu.yaml`: the one knob that selects the CPU.

To check another machine, run the command above and compare. The comparison falls back to the campaigns
in `agents.jsonl` within a float tolerance when the last digits differ:

```sh
python -m fusion_miniapp.compare_records results/ignos_shape_reference/run_record.json DIR/run_record.json
```

The timings are this laptop's and are only reported, never compared. The LangGraph run (859 s) overlapped
other work; the Academy run (570 s) did not.

Before publishing, the laptop's host name in the run records was replaced by `laptop`, and the
home-directory prefix of `constraints_path` in the campaign configs was removed. Nothing else was changed.
