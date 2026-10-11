"""Combine disjoint scan timings without claiming pooled multi-GPU throughput."""

import json
from collections import Counter
from pathlib import Path

from rev_ct.benchmark import summarise

ROOT = Path(__file__).resolve().parents[1]


def assert_cohort(samples, names):
    assert len(samples) == 64
    assert Counter(s["volume"] for s in samples) == Counter(names)


def main():
    reports = [json.loads((ROOT / f"reports/benchmark_64_worker_{i}.json").read_text()) for i in range(4)]
    fast = [json.loads((ROOT / f"reports/fast_preprocessing_64_worker_{i}.json").read_text()) for i in range(4)]
    cohort = [r for report in reports for r in report["cohort"]]
    names = [r["name"] for r in cohort]
    assert len(set(names)) == len({n.split("_")[1] for n in names}) == 64
    assert all(r["hardware"] == reports[0]["hardware"] for r in reports)
    assert all(r["feature_parity"] == reports[0]["feature_parity"] for r in reports)
    combined = {
        "hardware": {**reports[0]["hardware"], "allocated_vcpu": 12, "flavour": "a100-large"},
        "settings": {k: v for k, v in reports[0]["settings"].items() if k not in ["worker_index", "num_workers"]},
        "aggregation": "64 disjoint patients in four 16-scan jobs; latencies concatenated, per-GPU stream throughput is count divided by summed job wall times",
        "cohort": cohort,
        "strata": dict(Counter(r["stratum"] for r in cohort)),
        "feature_parity": reports[0]["feature_parity"],
        "results": {},
    }
    for key in ["rev", "colipri_zero_shot"]:
        samples = [s for r in reports for s in r["results"][key]["samples"]]
        assert_cohort(samples, names)
        combined["results"][key] = {"summary": summarise(samples), "samples": samples, "peak_vram_gib": max(r["results"][key]["peak_vram_gib"] for r in reports)}
    nv = {}
    for key in ["per_label", "single_list"]:
        samples = [s for r in reports for s in r["results"]["nv_reason_ct"]["protocols"][key]["samples"]]
        assert_cohort(samples, names)
        nv[key] = {"summary": summarise(samples), "samples": samples}
    binary = [o for s in nv["per_label"]["samples"] for o in s["outputs"]]
    nv["valid_yes_no_responses"] = sum(o.strip() in ["Yes", "No"] for o in binary)
    nv["total_binary_responses"] = len(binary)
    labels = json.loads((ROOT / "artifacts/config.json").read_text())["labels"]
    assert len(labels) == 18
    valid_json = 0
    for s in nv["single_list"]["samples"]:
        try:
            value = json.loads(s["outputs"][0])
            valid_json += isinstance(value, dict) and set(value) == set(labels) and all(type(v) is bool for v in value.values())
        except json.JSONDecodeError:
            pass
    nv["complete_boolean_json_responses"] = valid_json
    nv["budget_exhaustions"] = sum(s["budget_exhaustions"] for p in ["per_label", "single_list"] for s in nv[p]["samples"])
    nv["peak_vram_gib"] = max(r["results"]["nv_reason_ct"]["peak_vram_gib"] for r in reports)
    combined["results"]["nv_reason_ct"] = nv
    samples = [s for r in fast for s in r["samples"]]
    assert_cohort(samples, names)
    assert all(s["changed_voxels"] == 0 and s["max_pixel_error"] == 0 for s in samples)
    combined["results"]["rev_fast_preprocessing"] = {"summary": summarise(samples), "samples": samples, "changed_voxels": 0, "max_pixel_error": 0}
    for key, rows in [("reference", [r["results"]["rev"]["parallel_preprocessing"] for r in reports]), ("fast", [r["parallel"] for r in fast])]:
        assert all(r["decision_disagreements"] == 0 and r["max_score_error"] == 0 for r in rows)
        wall = sum(r["wall_s"] for r in rows)
        combined.setdefault("parallel_preprocessing", {})[key] = {"workers": 4, "batch_size": 1, "includes_startup_per_16_scans": True, "sum_stream_wall_s": wall, "per_gpu_volumes_per_second": 64 / wall, "decision_disagreements": 0, "max_score_error": 0}
    volume_rows = [p for r in reports for p in r["results"]["rev"]["volume_parity"]]
    combined["volume_parity"] = {"n": len(volume_rows), "decision_disagreements": sum(p["decision_disagreements"] for p in volume_rows), "max_feature_error": max(p["max_feature_error"] for p in volume_rows)}
    reference = combined["results"]["rev"]["summary"]["mean_s"]
    fast_mean = combined["results"]["rev_fast_preprocessing"]["summary"]["mean_s"]
    nv_mean = nv["per_label"]["summary"]["mean_s"]
    combined["ratios"] = {"fast_vs_reference_serial": reference / fast_mean, "reference_vs_nv_18_serial": nv_mean / reference, "fast_vs_nv_18_serial": nv_mean / fast_mean}
    (ROOT / "reports/benchmark_64.json").write_text(json.dumps(combined, indent=2) + "\n")
    pillar_files = [ROOT / f"reports/pillar_volumes_64_worker_{i}.json" for i in range(4)]
    if all(p.exists() for p in pillar_files):
        rows = [json.loads(p.read_text()) for p in pillar_files]
        pillar = {k: v for k, v in rows[0].items() if k not in ["worker_index", "cohort", "results", "load_s"]}
        pillar["cohort"] = [c for r in rows for c in r["cohort"]]
        pillar["results"] = {}
        for mode in ["fp32", "amp_bf16"]:
            samples = [s for r in rows for s in r["results"][mode]["samples"]]
            assert_cohort(samples, names)
            values = {"summary": summarise(samples), "samples": samples, "peak_vram_gib": max(r["results"][mode]["peak_vram_gib"] for r in rows)}
            if mode == "amp_bf16":
                values["max_feature_error"] = max(s["max_feature_error"] for s in samples)
                values["minimum_feature_cosine_similarity"] = min(s["feature_cosine_similarity"] for s in samples)
            pillar["results"][mode] = values
        (ROOT / "reports/pillar_volumes_64.json").write_text(json.dumps(pillar, indent=2) + "\n")
    for k, v in combined["results"].items():
        print(k, v.get("summary") or {p: v[p]["summary"] for p in ["per_label", "single_list"]})
    print("parallel", combined["parallel_preprocessing"])
    print("parity", combined["volume_parity"], "ratios", combined["ratios"])


if __name__ == "__main__":
    main()
