from __future__ import annotations

import asyncio
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("gradio")

from demo.backend import DemoBackend, validate_volume  # noqa: E402
from demo.ui import render_findings  # noqa: E402


def test_bundled_example_survives_cache_cleanup(tmp_path, monkeypatch):
    import gradio as gr
    from fastapi.testclient import TestClient
    from gradio.processing_utils import move_files_to_cache
    from gradio.route_utils import delete_files_created_by_app
    from gradio.state_holder import SessionState

    from demo.ui import build_demo

    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "cache"))
    app = build_demo(DemoBackend())
    session = SessionState(app)
    functions = {fn.api_name: fn for fn in app.fns.values()}
    volume = functions["analyze_ct"].inputs[0]
    temporary = tmp_path / "temporary.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4)), temporary)
    temporary_data = move_files_to_cache(volume.postprocess(str(temporary)), volume)

    async def exercise():
        before = await app.process_api(functions["load_example"], [0], state=session)
        original = before["data"][0]["value"]
        # Expire every eligible file immediately using Gradio's real cleanup.
        delete_files_created_by_app(app, age=-1)
        assert not Path(temporary_data["path"]).exists()
        after = await app.process_api(functions["load_example"], [0], state=session)
        example = after["data"][0]["value"]
        assert example == original
        with TestClient(gr.routes.App.create_app(app)) as client:
            response = client.get(example["url"])
            assert response.status_code == 200
            assert response.content == Path(DemoBackend().available_examples()[0][0]).read_bytes()
            loaded = await app.process_api(functions["load_volume"], [example, None], state=session)
            assert loaded["data"][0] is not None
            assert loaded["data"][2]["interactive"] is True
            result = await app.process_api(
                functions["analyze_ct"], [example, "All findings", None, None], state=session,
            )
            assert "precomputed" in result["data"][0]
            assert len(result["data"][1]["data"]) == 18
            record = result["data"][2].model_dump()
            assert len(record["decisions"]) == 18
            download = client.get(result["data"][3]["value"]["url"])
            assert download.status_code == 200
            assert download.json() == record
            delete_files_created_by_app(app, age=-1)
            assert client.get(result["data"][3]["value"]["url"]).status_code == 403
            assert client.get(example["url"]).status_code == 200

    asyncio.run(exercise())


def test_preview_preserves_physical_aspect_and_rejects_non_volume(tmp_path):
    path = tmp_path / "chest.nii.gz"
    data = np.full((20, 30, 10), -1000, dtype=np.int16)
    data[:, 20:, :] = 0
    nib.save(nib.Nifti1Image(data, np.diag([1, 2, 3, 1])), path)
    backend = DemoBackend()
    image = backend.preview(str(path), "Axial", 0.5, "Lung")
    assert image.shape[0] / image.shape[1] == pytest.approx(3, abs=0.02)
    assert image[0].mean() > image[-1].mean()  # anterior at the top
    assert image.dtype == np.uint8
    invalid = tmp_path / "series.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4, 2)), np.eye(4)), invalid)
    with pytest.raises(ValueError, match="three-dimensional"):
        validate_volume(str(invalid))


def test_invalid_payload_cannot_reach_inference(tmp_path):
    path = tmp_path / "invalid.nii.gz"
    path.write_bytes(b"not a CT")
    with pytest.raises(ValueError, match="could not be read"):
        DemoBackend().infer(str(path))
    nan_path = tmp_path / "nan.nii.gz"
    data = np.zeros((4, 4, 4))
    data[2, 2, 2] = np.nan
    nib.save(nib.Nifti1Image(data, np.eye(4)), nan_path)
    with pytest.raises(ValueError, match="invalid intensity"):
        DemoBackend().infer(str(nan_path))


def test_only_exact_example_uses_the_cached_record(tmp_path):
    backend = DemoBackend()
    original = Path(backend.available_examples()[0][0])
    renamed = tmp_path / "uploaded.nii.gz"
    renamed.write_bytes(original.read_bytes())
    record, elapsed = backend.infer(str(renamed))
    assert elapsed is None
    assert backend._model is None
    assert len(record["decisions"]) == 18
    assert record["volume"] == renamed.name
    tampered = bytearray(renamed.read_bytes())
    tampered[-1] ^= 1
    renamed.write_bytes(tampered)
    assert backend._cached_example(renamed) is None
    alternative = DemoBackend(revision="another-checkpoint")
    assert alternative._cached_example(original) is None
    metadata = json.loads(original.with_name("lung1.provenance.json").read_text())
    assert metadata["revision"] == backend.revision


def test_display_uses_trained_decision_separately_from_presence_score():
    record = {"decisions": [
        {"finding": "Pleural effusion", "decision": "present", "score": 0.3, "confidence": 0.3},
        {"finding": "Lung nodule", "decision": "absent", "score": 0.1, "confidence": 0.9},
    ]}
    rendered = render_findings(record, "Present only")
    assert rendered == [["Pleural effusion", "Present", "30.0%", "30.0%"]]
    assert render_findings(record)[1] == ["Lung nodule", "Absent", "10.0%", "90.0%"]
