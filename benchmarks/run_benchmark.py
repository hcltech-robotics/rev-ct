"""Same-volume Rev/COLIPRI/NV comparison, executable on STC or HF Jobs."""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import importlib.metadata
import json
import os
import platform
import random
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

from rev_ct.benchmark import summarise
from rev_ct.export import SOURCE_REPO, export_artifact
from rev_ct.head import RevHead
from rev_ct.inference import RevInference

SOURCE_REVISION = "e93a9ecd27e4153bab56258bdd97f7c4668ff9e2"
NV_REVISION = "386b93e034983f6c1fc841a43833a1b6a0cd9c13"
DATA_REVISION = "deeca4d89e9f978d4d1bccd88a55071ddbb146bb"


def sync():
    torch.cuda.synchronize()


def checkpoint(output, result):
    temp = Path(str(output) + ".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False))
    temp.replace(output)
    print(
        "REV_RESULT_JSON " + json.dumps(result, separators=(",", ":"), allow_nan=False), flush=True
    )


def release_memory():
    gc.collect()
    torch.cuda.empty_cache()


def load_data(root, count, seed, worker_index=0, num_workers=1):
    repo = "ibrahimhamamci/CT-RATE"

    def fetch(file):
        return hf_hub_download(repo, file, repo_type="dataset", revision=DATA_REVISION)

    no_chest = {
        Path(x.strip()).name
        for x in Path(fetch("dataset/metadata/no_chest_valid.txt")).read_text().splitlines()
    }
    metadata = list(csv.DictReader(open(fetch("dataset/metadata/validation_metadata.csv"))))
    rows = [r for r in metadata if r["VolumeName"] not in no_chest]
    random.Random(seed).shuffle(rows)

    def group(r):
        spacing = ast.literal_eval(r["XYSpacing"])[0]
        if int(r["Rows"]) * float(spacing) < 300:
            return "small_fov"
        if float(r["ZSpacing"]) > 1.5:
            return "thick_slice"
        return "matrix_1024" if int(r["Rows"]) == 1024 else "matrix_512"

    buckets = {
        k: [r for r in rows if group(r) == k]
        for k in ["matrix_512", "matrix_1024", "small_fov", "thick_slice"]
    }
    selected, used = [], set()
    while len(selected) < count and any(buckets.values()):
        for key, bucket in buckets.items():
            while bucket:
                row = bucket.pop()
                study = row["VolumeName"].split("_")[1]
                if study in used:
                    continue
                used.add(study)
                selected.append(
                    {
                        "name": row["VolumeName"],
                        "stratum": key,
                        "matrix": int(row["Rows"]),
                        "z_spacing_mm": float(row["ZSpacing"]),
                    }
                )
                break
            if len(selected) == count:
                break
    if not 0 <= worker_index < num_workers:
        raise ValueError("worker_index must be in [0, num_workers)")
    start = len(selected) * worker_index // num_workers
    stop = len(selected) * (worker_index + 1) // num_workers
    selected = selected[start:stop]
    root.mkdir(parents=True, exist_ok=True)
    for row in selected:
        parts = row["name"].split("_")
        file = f"dataset/valid_fixed/valid_{parts[1]}/valid_{parts[1]}_{parts[2]}/{row['name']}"
        row["path"] = hf_hub_download(
            repo, file, repo_type="dataset", revision=DATA_REVISION, local_dir=root
        )
    return selected, no_chest, fetch("dataset/multi_abnormality_labels/valid_predicted_labels.csv")


def validate_features(head, source, no_chest, labels_csv):
    from sklearn.metrics import f1_score, roc_auc_score

    repo = "chrisvoncsefalvay/ct-rate-colipri-features"
    api = HfApi()
    revision = api.dataset_info(repo).sha
    xs, names = [], []
    for file in api.list_repo_files(repo, repo_type="dataset", revision=revision):
        if file.startswith("features/valid/") and file.endswith(".npz"):
            with np.load(
                hf_hub_download(repo, file, repo_type="dataset", revision=revision)
            ) as shard:
                xs.append(shard["feats"])
                names.extend(shard["names"].astype(str).tolist())
    x = np.concatenate(xs).astype(np.float32)
    z = source["scaler"].transform(x)
    reference_p = np.column_stack([clf.predict_proba(z)[:, 1] for _, clf in source["clfs"]])
    reference_scores = np.column_stack(
        [iso.transform(reference_p[:, j]) for j, iso in enumerate(source["isotonic"])]
    )
    reference_decisions = (
        np.log(np.clip(reference_p, 1e-7, 1 - 1e-7) / (1 - np.clip(reference_p, 1e-7, 1 - 1e-7)))
        >= source["biases"]
    )
    outputs = {mode: head.predict(x, mode=mode) for mode in ["reference", "fused"]}
    oracle = {
        r["VolumeName"]: [int(r[label]) for label in head.labels]
        for r in csv.DictReader(open(labels_csv))
    }
    y = np.array([oracle[name] for name in names])
    corrected = np.array([name not in no_chest for name in names])
    report = {
        "n": len(names),
        "n_unique": len(set(names)),
        "feature_revision": revision,
        "modes": {},
    }
    for mode, out in outputs.items():
        report["modes"][mode] = {
            "decision_disagreements": int(np.sum(out.decisions != reference_decisions)),
            "max_probability_error": float(np.max(abs(out.raw_probabilities - reference_p))),
            "max_score_error": float(np.max(abs(out.scores - reference_scores))),
            "metrics": {},
        }
        for name, mask in [
            ("full_3039", np.ones(len(x), dtype=bool)),
            ("corrected_3002", corrected),
        ]:
            report["modes"][mode]["metrics"][name] = {
                "n": int(mask.sum()),
                "macro_f1": float(
                    f1_score(y[mask], out.decisions[mask], average="macro", zero_division=0)
                ),
                "macro_auroc": float(
                    roc_auc_score(y[mask], out.raw_probabilities[mask], average="macro")
                ),
            }
    lookup = {name: x[i] for i, name in enumerate(names)}
    return report, lookup


def rev_benchmark(head, cohort, repeats, lookup):
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = RevInference(head)
    load_s = time.perf_counter() - started
    cold_start = time.perf_counter()
    model.encode(model.preprocess([cohort[0]["path"]]))
    sync()
    cold_s = time.perf_counter() - cold_start
    samples, feature_parity = [], []
    features_cache = {}
    for repeat in range(repeats):
        for row in cohort:
            sync()
            t0 = time.perf_counter()
            pixels = model.preprocess([row["path"]])
            t1 = time.perf_counter()
            inputs = pixels.to("cuda", dtype=model.dtype)
            sync()
            t2 = time.perf_counter()
            with torch.inference_mode():
                pooled = model.model.encode_image(inputs, pool=True, project=True, normalize=False)
            sync()
            t3 = time.perf_counter()
            features = pooled.cpu().to(torch.float16).numpy().astype(np.float32)
            sync()
            t4 = time.perf_counter()
            records = head.records(features, [row["name"]])
            t5 = time.perf_counter()
            print(
                f"REV_VOLUME {repeat} {row['name']} preprocess={t1 - t0:.3f}s encoder={t3 - t2:.3f}s",
                flush=True,
            )
            samples.append(
                {
                    "volume": row["name"],
                    "repeat": repeat,
                    "preprocess_s": t1 - t0,
                    "h2d_s": t2 - t1,
                    "encoder_s": t3 - t2,
                    "d2h_s": t4 - t3,
                    "head_json_s": t5 - t4,
                    "total_s": t5 - t0,
                }
            )
            features_cache[row["name"]] = features
            if repeat == 0:
                historical = lookup[row["name"]][None, :]
                feature_parity.append(
                    {
                        "volume": row["name"],
                        "max_feature_error": float(np.max(abs(features - historical))),
                        "mean_feature_error": float(np.mean(abs(features - historical))),
                        "decision_disagreements": int(
                            np.sum(
                                head.predict(features).decisions
                                != head.predict(historical).decisions
                            )
                        ),
                        "records": records,
                    }
                )
            del inputs, pooled, pixels
    report = {
        "dtype": str(model.dtype),
        "load_s": load_s,
        "first_inference_s": cold_s,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "summary": summarise(samples),
        "samples": samples,
        "volume_parity": feature_parity,
    }
    report["head_microbenchmark"] = {}
    x = np.concatenate(list(features_cache.values()))
    for mode in ["reference", "fused"]:
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            head.records(x, [str(i) for i in range(len(x))], mode=mode)
            times.append(time.perf_counter() - t0)
        report["head_microbenchmark"][mode] = {
            "batch_size": len(x),
            "p50_batch_s": float(np.median(times)),
        }
    t0 = time.perf_counter()
    parallel_records = list(model.predict([r["path"] for r in cohort], workers=4, batch_size=1))
    sync()
    wall = time.perf_counter() - t0
    report["parallel_preprocessing"] = {
        "workers": 4,
        "batch_size": 1,
        "includes_worker_startup": True,
        "wall_s": wall,
        "volumes_per_second": len(cohort) / wall,
        "decision_disagreements": sum(
            a["decision"] != b["decision"]
            for i, r in enumerate(parallel_records)
            for a, b in zip(
                r["decisions"], feature_parity[i]["records"][0]["decisions"], strict=True
            )
        ),
    }
    parallel_scores = np.array([[d["score"] for d in r["decisions"]] for r in parallel_records])
    eager_scores = np.array(
        [[d["score"] for d in r["records"][0]["decisions"]] for r in feature_parity]
    )
    report["parallel_preprocessing"]["max_score_error"] = float(
        np.max(abs(parallel_scores - eager_scores))
    )
    # torch.compile includes substantial startup; measure it separately from warm forwards.
    try:
        pixels = model.preprocess([cohort[0]["path"]]).to("cuda", dtype=model.dtype)
        eager = features_cache[cohort[0]["name"]]
        compiled = torch.compile(model.model.image_encoder, mode="reduce-overhead")
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = compiled(pixels, pool=True, project=True, normalize=False)
        sync()
        compile_s = time.perf_counter() - t0
        candidate = output.cpu().to(torch.float16).numpy().astype(np.float32)
        timings = []
        for _ in range(6):
            sync()
            t0 = time.perf_counter()
            with torch.inference_mode():
                compiled(pixels, pool=True, project=True, normalize=False)
            sync()
            timings.append(time.perf_counter() - t0)
        report["compile"] = {
            "mode": "reduce-overhead",
            "compile_first_forward_s": compile_s,
            "warm_p50_encoder_s": float(np.median(timings)),
            "max_feature_error": float(np.max(abs(eager - candidate))),
            "decision_disagreements": int(
                np.sum(head.predict(eager).decisions != head.predict(candidate).decisions)
            ),
        }
        compile_parity = []
        for row in cohort:
            pixels = model.preprocess([row["path"]]).to("cuda", dtype=model.dtype)
            with torch.inference_mode():
                compiled_features = compiled(pixels, pool=True, project=True, normalize=False)
            candidate = compiled_features.cpu().to(torch.float16).numpy().astype(np.float32)
            eager_output = head.predict(features_cache[row["name"]])
            compiled_output = head.predict(candidate)
            compile_parity.append(
                {
                    "volume": row["name"],
                    "max_score_error": float(
                        np.max(abs(eager_output.scores - compiled_output.scores))
                    ),
                    "max_feature_error": float(
                        np.max(abs(features_cache[row["name"]] - candidate))
                    ),
                    "decision_disagreements": int(
                        np.sum(eager_output.decisions != compiled_output.decisions)
                    ),
                }
            )
        report["compile"]["cohort_parity"] = compile_parity
        del compiled, output, pixels
    except Exception as error:
        report["compile"] = {"error": type(error).__name__ + ": " + str(error)[:500]}
    del model
    release_memory()
    return report


def colipri_zero_shot(head, cohort, repeats):
    from colipri import get_model, get_processor

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    checkpoint_path = hf_hub_download(
        "microsoft/colipri", "model.safetensors", revision=head.config["encoder"]["revision"]
    )
    model = get_model(checkpoint_path).eval().cuda().to(torch.bfloat16)
    processor = get_processor()
    prompts = [
        text
        for finding in head.labels
        for text in [f"No {finding.lower()}.", f"{finding} is present."]
    ]
    ids, mask = processor.process_text(prompts)
    with torch.inference_mode():
        text = model.encode_text(ids.cuda(), mask.cuda(), pool=True, normalize=True)
    sync()
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    warm_pixels = (
        processor.to_images_batch(processor.process_images(cohort[0]["path"]))
        .cuda()
        .to(torch.bfloat16)
    )
    with torch.inference_mode():
        model.encode_image(warm_pixels, pool=True, project=True, normalize=True)
    sync()
    first_inference_s = time.perf_counter() - t0
    del warm_pixels
    samples = []
    for repeat in range(repeats):
        for row in cohort:
            sync()
            t0 = time.perf_counter()
            pixels = processor.to_images_batch(processor.process_images(row["path"]))
            t1 = time.perf_counter()
            pixels = pixels.cuda().to(torch.bfloat16)
            sync()
            t2 = time.perf_counter()
            with torch.inference_mode():
                features = model.encode_image(pixels, pool=True, project=True, normalize=True)
                similarities = model.compute_similarities(features, text).reshape(1, 18, 2)
                scores = (similarities / model.softmax_temperature).softmax(-1)[0, :, 1]
            sync()
            t3 = time.perf_counter()
            scores = scores.float().cpu().tolist()
            t4 = time.perf_counter()
            samples.append(
                {
                    "volume": row["name"],
                    "repeat": repeat,
                    "preprocess_s": t1 - t0,
                    "h2d_s": t2 - t1,
                    "encoder_classify_s": t3 - t2,
                    "postprocess_s": t4 - t3,
                    "total_s": t4 - t0,
                    "scores": scores,
                }
            )
    report = {
        "load_and_text_cache_s": load_s,
        "first_image_inference_s": first_inference_s,
        "prompts": prompts,
        "protocol": "18 independent negative/positive prompt pairs; 0.5 threshold; upstream temperature",
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "summary": summarise(samples),
        "samples": samples,
    }
    del model, text, processor
    release_memory()
    return report


def nv_benchmark(head, cohort, repeats):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = (
        AutoModelForImageTextToText.from_pretrained(
            "nvidia/NV-Reason-CT",
            revision=NV_REVISION,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .eval()
        .cuda()
    )
    processor = AutoProcessor.from_pretrained(
        "nvidia/NV-Reason-CT", revision=NV_REVISION, trust_remote_code=True
    )
    sync()
    load_s = time.perf_counter() - t0

    def prompt(text, proc):
        return proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    binary_prompts = [
        prompt(f"Is {finding.lower()} present in this CT? Answer only Yes or No.", processor)
        for finding in head.labels
    ]
    list_text = (
        "Identify the abnormalities present in this chest CT. Choose only from this list: "
        + ", ".join(head.labels)
        + ". Return a JSON object mapping every listed abnormality to true or false."
    )
    list_prompt = prompt(list_text, processor)
    cold = {}
    for name, text, limit in [("binary", binary_prompts[0], 8), ("single_list", list_prompt, 512)]:
        t0 = time.perf_counter()
        inputs = processor(
            text=text, images3d=[cohort[0]["path"]], anatomy_region="chest", return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=limit, do_sample=False, use_cache=True
            )
        sync()
        cold[name] = time.perf_counter() - t0
        del inputs, generated

    class CachedLoader:
        def __init__(self, original, pixels):
            self.merge_size = original.merge_size
            self.pixels = pixels

        def __call__(self, **kwargs):
            return self.pixels

    original_loader = processor.image_processor_3d
    samples = {"per_label": [], "single_list": []}
    for repeat in range(repeats):
        for row in cohort:
            sync()
            t0 = time.perf_counter()
            image = original_loader(images=[row["path"]], anatomy_region="chest")
            prep_s = time.perf_counter() - t0
            processor.image_processor_3d = CachedLoader(original_loader, image)
            for protocol, prompts, limit in [
                ("per_label", binary_prompts, 8),
                ("single_list", [list_prompt], 512),
            ]:
                t0 = time.perf_counter()
                tokenise_s = 0.0
                transfer_s = 0.0
                generate_s = 0.0
                outputs = []
                tokens = 0
                truncated = 0
                for text in prompts:
                    t1 = time.perf_counter()
                    inputs = processor(
                        text=text,
                        images3d=[row["path"]],
                        anatomy_region="chest",
                        return_tensors="pt",
                    )
                    t2 = time.perf_counter()
                    inputs = inputs.to("cuda")
                    sync()
                    t3 = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs, max_new_tokens=limit, do_sample=False, use_cache=True
                        )
                    sync()
                    t4 = time.perf_counter()
                    new = generated[:, inputs.input_ids.shape[1] :]
                    tokens += new.shape[1]
                    truncated += int(new.shape[1] >= limit)
                    outputs.append(
                        processor.batch_decode(
                            new, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                    )
                    tokenise_s += t2 - t1
                    transfer_s += t3 - t2
                    generate_s += t4 - t3
                    del generated, inputs, new
                inference_s = time.perf_counter() - t0
                samples[protocol].append(
                    {
                        "volume": row["name"],
                        "repeat": repeat,
                        "preprocess_s": prep_s,
                        "tokenise_s": tokenise_s,
                        "h2d_s": transfer_s,
                        "generate_s": generate_s,
                        "total_s": prep_s + inference_s,
                        "n_prefills": len(prompts),
                        "generated_tokens": tokens,
                        "budget_exhaustions": truncated,
                        "outputs": outputs,
                    }
                )
                print(
                    "NV_SAMPLE_JSON "
                    + json.dumps(
                        {"protocol": protocol, **samples[protocol][-1]}, separators=(",", ":")
                    ),
                    flush=True,
                )
            processor.image_processor_3d = original_loader
            del image
    report = {
        "load_s": load_s,
        "first_protocol_inference_s": cold,
        "revision": NV_REVISION,
        "attn_implementation": "sdpa",
        "dtype": "bfloat16",
        "thinking": False,
        "preprocessing": "one official chest crop per volume, reused for 18 prompts; no vision/LLM prefill caching",
        "binary_prompts": binary_prompts,
        "single_list_prompt": list_text,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "protocols": {
            name: {
                "max_new_tokens": 8 if name == "per_label" else 512,
                "summary": summarise(rows),
                "samples": rows,
            }
            for name, rows in samples.items()
        },
    }
    del model, processor
    release_memory()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmark.json")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "packages": {
                p: importlib.metadata.version(p)
                for p in ["colipri", "transformers", "numpy", "scikit-learn", "safetensors"]
            },
        },
        "settings": {
            "count": args.count,
            "repeats": args.repeats,
            "seed": args.seed,
            "batch_size": 1,
            "torch_cpu_threads": 1,
            "tf32": False,
            "data_variant": "valid_fixed",
            "rev_precision": "fp32",
            "worker_index": args.worker_index,
            "num_workers": args.num_workers,
        },
        "results": {},
    }
    import joblib

    source = joblib.load(
        hf_hub_download(SOURCE_REPO, "typed_calibration.joblib", revision=SOURCE_REVISION)
    )
    export_artifact(source, "/tmp/rev-head", source_revision=SOURCE_REVISION)
    head = RevHead.from_pretrained("/tmp/rev-head")
    cohort, no_chest, labels_csv = load_data(
        Path("/tmp/rev-benchmark-data"), args.count, args.seed, args.worker_index, args.num_workers
    )
    result["cohort"] = [{k: v for k, v in row.items() if k != "path"} for row in cohort]
    result["data_revision"] = DATA_REVISION
    result["feature_parity"], lookup = validate_features(head, source, no_chest, labels_csv)
    checkpoint(args.output, result)
    for name, fn in [
        ("rev", lambda: rev_benchmark(head, cohort, args.repeats, lookup)),
        ("colipri_zero_shot", lambda: colipri_zero_shot(head, cohort, args.repeats)),
        ("nv_reason_ct", lambda: nv_benchmark(head, cohort, args.repeats)),
    ]:
        try:
            print("PHASE " + name, flush=True)
            result["results"][name] = fn()
        except Exception as error:
            result["results"][name] = {
                "error": type(error).__name__ + ": " + str(error)[:1500],
                "traceback": traceback.format_exc(),
            }
            traceback.print_exc()
            release_memory()
        checkpoint(args.output, result)


if __name__ == "__main__":
    main()
