import json
from pathlib import Path

from rev_ct.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_features_cli_writes_valid_records(tmp_path):
    out = tmp_path / "out.jsonl"
    main(
        [
            "features",
            str(ROOT / "artifacts/validation_sample.npz"),
            "--model",
            str(ROOT / "artifacts"),
            "--output",
            str(out),
        ]
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 32
    assert records[0]["volume"].endswith(".nii.gz")
    assert len(records[0]["decisions"]) == 18
