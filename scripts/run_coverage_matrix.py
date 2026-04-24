"""Phase E: aggregate `bench/_coverage/*.json` sidecars into COVERAGE_MATRIX.md.

Each sidecar is written by one of the three real-task runners:
  * tier1_mlp.json       <- scripts/train_real_world_classifier.py
  * tier2_one_block.json <- scripts/train_real_world_tt_block_classifier.py
  * tier3_two_block.json <- scripts/train_real_world_tt_block_depth2_blend.py

Run those three first (they each emit their MD report + sidecar JSON), then
run this script to produce a single coherent matrix:

    python scripts/run_coverage_matrix.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COV_DIR = ROOT / "bench" / "_coverage"
TIERS = ("tier1_mlp", "tier2_one_block", "tier3_two_block")


def _load() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tier in TIERS:
        p = COV_DIR / f"{tier}.json"
        if not p.exists():
            print(f"WARNING: missing sidecar {p.relative_to(ROOT)} — run the corresponding script first.")
            continue
        out[tier] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.4f}"


def _fmt_num(x: float | None, fmt: str = ".2f") -> str:
    if x is None:
        return "—"
    return format(x, fmt)


def main() -> None:
    sidecars = _load()
    if not sidecars:
        print("No sidecars found — run the three Tier scripts first.")
        return

    devices = sorted({s["device"] for s in sidecars.values()})
    out = ROOT / "bench" / "COVERAGE_MATRIX.md"

    lines: list[str] = []
    lines.append("# DMRG-Transformer — Architecture Coverage Matrix (Phase E)")
    lines.append("")
    lines.append(f"**Device(s):** {', '.join(f'`{d}`' for d in devices)}  ")
    lines.append("**Source sidecars:** `bench/_coverage/{tier1_mlp,tier2_one_block,tier3_two_block}.json`")
    lines.append("")
    lines.append(
        "Single-row-per-architecture comparison of DMRG vs Adam (MSE-on-one-hot, "
        "matched loss) across three architectural tiers, all on `sklearn.datasets.load_digits`. "
        "All runs use the same seed (42), float64, and CUDA device. Adam columns report "
        "**iso-time test accuracy** (Adam's test acc at the wall-clock the matched DMRG "
        "run took to finish) so the comparison is honest at the algorithm level."
    )
    lines.append("")
    lines.append("## Test accuracy")
    lines.append("")
    lines.append("| Tier | Architecture | DMRG test acc | Adam-MSE test acc | Adam-MSE iso-time | Adam-CE test acc | Adam-CE iso-time |")
    lines.append("| :--- | :----------- | ------------: | ----------------: | ----------------: | ---------------: | ---------------: |")
    for tier in TIERS:
        s = sidecars.get(tier)
        if s is None:
            lines.append(f"| {tier} | (missing) | — | — | — | — | — |")
            continue
        tt = s["tt"]; am = s["adam_mse"]; ac = s["adam_ce"]
        lines.append(
            f"| {tier} | {s['label']} | **{_fmt_pct(tt['test_acc'])}** | "
            f"{_fmt_pct(am['test_acc'])} | {_fmt_pct(am.get('iso_time_test_acc'))} | "
            f"{_fmt_pct(ac['test_acc'])} | {_fmt_pct(ac.get('iso_time_test_acc'))} |"
        )
    lines.append("")
    lines.append("## Wall-clock and GPU memory")
    lines.append("")
    lines.append("| Tier | DMRG wall (s) | Adam-MSE wall (s) | DMRG peak GPU (MiB) | Adam-MSE peak GPU (MiB) | Adam-CE peak GPU (MiB) |")
    lines.append("| :--- | ------------: | ----------------: | ------------------: | ----------------------: | ---------------------: |")
    for tier in TIERS:
        s = sidecars.get(tier)
        if s is None:
            continue
        tt = s["tt"]; am = s["adam_mse"]; ac = s["adam_ce"]
        lines.append(
            f"| {tier} | {_fmt_num(tt['wall_s'])} | {_fmt_num(am['wall_s'])} | "
            f"{_fmt_num(tt.get('peak_mem_mib'), '.1f')} | "
            f"{_fmt_num(am.get('peak_mem_mib'), '.1f')} | "
            f"{_fmt_num(ac.get('peak_mem_mib'), '.1f')} |"
        )
    lines.append("")
    lines.append("## Inference latency (median ms)")
    lines.append("")
    lines.append("| Tier | DMRG full-batch | DMRG batch=1 | Adam-MSE full-batch | Adam-CE full-batch |")
    lines.append("| :--- | --------------: | -----------: | ------------------: | -----------------: |")
    for tier in TIERS:
        s = sidecars.get(tier)
        if s is None:
            continue
        tt = s["tt"]; am = s["adam_mse"]; ac = s["adam_ce"]
        lines.append(
            f"| {tier} | {_fmt_num(tt.get('inference_full_ms'), '.3f')} | "
            f"{_fmt_num(tt.get('inference_b1_ms'), '.3f')} | "
            f"{_fmt_num(am.get('inference_full_ms'), '.3f')} | "
            f"{_fmt_num(ac.get('inference_full_ms'), '.3f')} |"
        )
    lines.append("")
    lines.append("## Acceptance rates (DMRG-only diagnostics)")
    lines.append("")
    lines.append("| Tier | input_proj accept rate | attn accept rate | Notes |")
    lines.append("| :--- | ---------------------: | ---------------: | :---- |")
    for tier in TIERS:
        s = sidecars.get(tier)
        if s is None:
            continue
        tt = s["tt"]
        ipa = tt.get("input_proj_accept_rate")
        ata = tt.get("attn_accept_rate")
        notes = []
        if "blend" in tt:
            notes.append(f"target_blend={tt['blend']:.2f}")
        if "attn_reject_rate_final" in tt:
            notes.append(f"final attn reject={tt['attn_reject_rate_final']:.2f}")
        if tier == "tier1_mlp":
            notes.append("no attention; pure TT-MLP")
        lines.append(
            f"| {tier} | {_fmt_pct(ipa) if ipa is not None else '—'} | "
            f"{_fmt_pct(ata) if ata is not None else '—'} | {'; '.join(notes) or '—'} |"
        )
    lines.append("")
    lines.append("## Parameter counts")
    lines.append("")
    lines.append("| Tier | DMRG TT params | Adam dense params | Compression |")
    lines.append("| :--- | -------------: | ----------------: | ----------: |")
    for tier in TIERS:
        s = sidecars.get(tier)
        if s is None:
            continue
        tt_p = s["params"]["tt"]; d_p = s["params"]["dense"]
        comp = d_p / tt_p if tt_p > 0 else float("nan")
        lines.append(f"| {tier} | {tt_p:,} | {d_p:,} | {comp:.2f}x |")
    lines.append("")
    lines.append("## How to reproduce")
    lines.append("")
    lines.append("```")
    lines.append("python scripts/train_real_world_classifier.py            # Tier 1: pure TT-MLP")
    lines.append("python scripts/train_real_world_tt_block_classifier.py   # Tier 2: 1x TTBlock")
    lines.append("python scripts/train_real_world_tt_block_depth2_blend.py # Tier 3: 2x TTBlock stack")
    lines.append("python scripts/run_coverage_matrix.py                    # aggregate to this file")
    lines.append("```")
    lines.append("")
    lines.append("## Honest reading")
    lines.append("")
    lines.append(
        "- **DMRG vs Adam-final** is a single-shot comparison; the scientifically "
        "honest column is **Adam iso-time**, which gives Adam exactly the wall-clock "
        "the DMRG run consumed."
    )
    lines.append(
        "- The compression column is small at this scale because dimensions are tiny "
        "(EMBED_DIM=16, HIDDEN=32). See [bench/HEADLINE.md](HEADLINE.md) for "
        "1024×1024 scaling and [bench/HEADLINE_ISO_RANK.md](HEADLINE_ISO_RANK.md) "
        "for iso-rank fairness on the synthetic regression."
    )
    lines.append(
        "- Tier 1 has no attention, so input_proj/attn acceptance columns are `—`. "
        "Tier 3 reports the better-of-two-blends row; the per-blend table lives in "
        "[bench/REAL_WORLD_TT_BLOCK_DEPTH2_BLEND.md](REAL_WORLD_TT_BLOCK_DEPTH2_BLEND.md)."
    )
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}")
    print(f"Tiers covered: {sorted(sidecars.keys())}")


if __name__ == "__main__":
    main()
