"""Clef-shaped loading and typed requests for Rev's fixed finding vocabulary."""

from __future__ import annotations

import torch


def load_release_model(path_or_repo="hcltech-robotics/rev-ct", *, device="cuda", revision=None):
    from transformers import AutoModel, AutoProcessor

    model = (
        AutoModel.from_pretrained(path_or_repo, revision=revision, trust_remote_code=False)
        .eval()
        .to(device)
    )
    processor = AutoProcessor.from_pretrained(
        path_or_repo, revision=revision, trust_remote_code=False
    )
    return model, processor


def encode_record(processor, record):
    state = record.get("state", {})
    if isinstance(state, dict) and "features" in state:
        return {"features": torch.as_tensor(state["features"], dtype=torch.float32).reshape(1, -1)}
    volume = state.get("volume") if isinstance(state, dict) else None
    images = record.get("images3d") or ([volume] if volume else None)
    if not images or processor is None:
        raise ValueError("state.volume or images3d must provide a CT volume")
    if len(images) != 1:
        raise ValueError("One volume per typed request is supported")
    batch = dict(processor(images3d=images, return_tensors="pt"))
    batch["volume_names"] = [str(images[0])]
    return batch


def collate_records(records):
    if not records:
        raise ValueError("Cannot collate an empty batch")
    key = "features" if "features" in records[0] else "pixel_values"
    if any(key not in record for record in records):
        raise ValueError("Batch records must all contain features or all contain volumes")
    batch = {key: torch.cat([record[key] for record in records])}
    if all("volume_names" in record for record in records):
        batch["volume_names"] = [name for record in records for name in record["volume_names"]]
    return batch


def systemone(model, processor, request):
    labels = model.config.labels
    questions = request.get("questions") or {
        finding: {"type": "noul", "finding": finding} for finding in labels
    }
    for question_id, question in questions.items():
        if question.get("type") != "noul":
            raise ValueError(f"{question_id}: Rev supports noul questions about trained findings")
        if question.get("finding", question_id) not in labels:
            raise ValueError(f"{question_id}: finding must be one of the 18 trained findings")
    batch = encode_record(processor, request)
    with torch.inference_mode():
        output = model(**batch)
    answers = {}
    for question_id, question in questions.items():
        j = labels.index(question.get("finding", question_id))
        score = float(output.probabilities[0, j])
        present = bool(output.decisions[0, j])
        answers[question_id] = {
            "type": "noul",
            "noul": round(score, 4),
            "decision": "present" if present else "absent",
            "confidence": round(score if present else 1 - score, 4),
            "probabilities": {"true": round(score, 4), "false": round(1 - score, 4)},
        }
    return {
        "model": request.get("model", "rev-ct"),
        "answers": answers,
        "usage": {"input_volumes": 1, "output_tokens": 0},
    }
