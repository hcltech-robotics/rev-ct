"""Gradio interface for the RevCT chest-CT research demo."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

MODEL_URL = "https://huggingface.co/hcltech-robotics/rev-ct"
CODE_URL = "https://github.com/hcltech-robotics/rev-ct"


def _percentage(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("A finding score is outside its expected range.")
    return f"{number:.1%}"


def render_findings(record: dict | None, selection: str = "All findings") -> list[list[str]]:
    """Keep the trained decision separate from the calibrated presence score."""
    if record is None:
        return []
    ordered = sorted(
        record["decisions"],
        key=lambda item: (item["decision"] != "present", -float(item["score"])),
    )
    rows = []
    for item in ordered:
        decision = item["decision"]
        if decision not in {"present", "absent"}:
            raise ValueError("A finding decision is outside the expected vocabulary.")
        if selection == "Present only" and decision != "present":
            continue
        rows.append([
            str(item["finding"]), decision.capitalize(),
            _percentage(item["score"]), _percentage(item["confidence"]),
        ])
    return rows


def render_summary(record: dict | None, elapsed: float | None = None) -> str:
    if record is None:
        return ""
    count = sum(item["decision"] == "present" for item in record["decisions"])
    timing = f"{elapsed:.2f} s" if elapsed is not None else "Example result (precomputed)"
    return f"**{count} present** · {timing}"


def _delete_download(path: str | None) -> None:
    if path:
        Path(path).unlink(missing_ok=True)


def demo_launch_kwargs() -> dict[str, Any]:
    """Gradio 6 receives theme and CSS at launch time."""
    return {
        "theme": gr.themes.Default(
            primary_hue="gray", secondary_hue="gray", neutral_hue="gray",
        ),
        "css": Path(__file__).with_name("style.css").read_text(),
        "footer_links": [],
    }


def build_demo(backend: Any) -> gr.Blocks:
    """Build the interface around a preview/inference backend."""
    with gr.Blocks(
        title="RevCT · Chest CT findings",
        analytics_enabled=False,
        delete_cache=(600, 1800),
    ) as demo:
        current_record = gr.State(None)
        download_path = gr.State(None, time_to_live=1800, delete_callback=_delete_download)
        session_generation = gr.State({"value": 0})
        gr.Markdown(
            "# RevCT\n\n"
            "A typed decision model for computed tomography (CT).\n\n"
            f"[Model]({MODEL_URL}) · [Code]({CODE_URL}) · "
            "[Citation](https://doi.org/10.57967/hf/10861)",
            elem_id="rev-header",
        )

        with gr.Row(elem_id="rev-workspace", equal_height=False):
            with gr.Column(scale=1, min_width=300, elem_id="rev-input-column"):
                volume = gr.File(
                    label="Chest CT (.nii / .nii.gz)",
                    file_count="single",
                    file_types=[".nii", ".nii.gz"],
                    type="filepath",
                    height=112,
                    elem_id="rev-upload",
                )
                gr.Markdown("Input: Hounsfield units and physical voxel spacing.")
                examples = backend.available_examples()
                if examples:
                    gr.Examples(examples=examples, inputs=[volume], label="Examples")
                    gr.Markdown("Example output is precomputed; uploaded scans run live.")
                with gr.Row():
                    analyze = gr.Button("Analyze CT", variant="primary", interactive=False)
                    clear = gr.Button("Clear")
                image = gr.Image(
                    label="CT slice",
                    type="numpy",
                    image_mode="L",
                    format="png",
                    height=360,
                    interactive=False,
                    buttons=["fullscreen"],
                    elem_id="rev-slice-viewer",
                )
                geometry = gr.Markdown("")
                with gr.Row():
                    plane = gr.Radio(
                        ["Axial", "Coronal", "Sagittal"], value="Axial", label="Plane",
                    )
                    window = gr.Radio(
                        ["Lung", "Mediastinum", "Bone"], value="Lung", label="Window",
                    )
                position = gr.Slider(
                    0, 1, value=0.5, step=0.001, label="Slice position",
                    elem_id="rev-position",
                )

            with gr.Column(scale=1, min_width=300, elem_id="rev-output-column"):
                findings = gr.Dataframe(
                    value=[],
                    headers=["Finding", "Decision", "Presence score", "Confidence"],
                    row_count=0,
                    column_count=4,
                    datatype="str",
                    type="array",
                    interactive=False,
                    wrap=True,
                    column_widths=["220px", "90px", "115px", "115px"],
                    max_height=620,
                    label="Findings",
                    elem_id="rev-findings",
                )
                selection = gr.Radio(
                    ["All findings", "Present only"], value="All findings", label="Show",
                )
                summary = gr.Markdown("")
                with gr.Accordion("JSON", open=False, elem_id="rev-record"):
                    record_json = gr.JSON(show_label=False)
                download = gr.DownloadButton(
                    "Download JSON", interactive=False, elem_id="rev-download",
                )
        gr.Markdown(
            "Research use only. Not for diagnosis or treatment. "
            "Upload only de-identified scans. "
            f"[Run locally]({CODE_URL}/tree/main/demo) for private data.",
            elem_id="rev-footer",
        )

        def preview(file_path: str | None, selected_plane: str, offset: float, preset: str):
            if not file_path:
                return None
            try:
                return backend.preview(file_path, selected_plane, offset, preset)
            except Exception as error:
                raise gr.Error(str(error)) from error

        def reset_volume(old_download: str | None, generation: dict[str, int]):
            generation["value"] += 1
            _delete_download(old_download)
            return (
                None,
                "",
                gr.update(interactive=False),
                render_summary(None),
                [],
                None,
                gr.update(value=None, interactive=False),
                None,
                None,
                0.5,
                "Axial",
                "Lung",
                "All findings",
            )

        def load_volume(file_path: str | None, generation: dict[str, int]):
            started_generation = generation["value"]
            if not file_path:
                return None, "", gr.update(interactive=False)
            try:
                description = backend.describe(file_path)
                initial_image = backend.preview(file_path, "Axial", 0.5, "Lung")
            except Exception as error:
                if generation["value"] != started_generation:
                    return (gr.skip(),) * 3
                raise gr.Error(str(error)) from error
            if generation["value"] != started_generation:
                return (gr.skip(),) * 3
            return (
                initial_image,
                description,
                gr.update(interactive=True),
            )

        inference_progress = gr.Progress()

        def infer(
            file_path: str | None,
            selected: str,
            old_download: str | None,
            generation: dict[str, int],
            progress=inference_progress,
        ):
            if not file_path:
                raise gr.Error("Load a chest CT before starting analysis.")
            started_generation = generation["value"]
            path = None
            try:
                progress(0, desc="Analyzing chest CT…")
                record, elapsed = backend.infer(file_path)
                if generation["value"] != started_generation:
                    return (gr.skip(),) * 6
                progress(0.95, desc="Preparing typed decisions…")
                table = render_findings(record, selected)
                heading = render_summary(record, elapsed)
                with tempfile.NamedTemporaryFile(
                    mode="w", prefix="revct-", suffix=".json", delete=False, encoding="utf-8"
                ) as output:
                    json.dump(record, output, ensure_ascii=False, indent=2, allow_nan=False)
                    output.write("\n")
                    path = output.name
                _delete_download(old_download)
            except Exception as error:
                _delete_download(path)
                if generation["value"] != started_generation:
                    return (gr.skip(),) * 6
                raise gr.Error(str(error)) from error
            if generation["value"] != started_generation:
                _delete_download(path)
                return (gr.skip(),) * 6
            return heading, table, record, gr.update(value=path, interactive=True), record, path

        inference_event = analyze.click(
            infer,
            inputs=[volume, selection, download_path, session_generation],
            outputs=[summary, findings, record_json, download, current_record, download_path],
            concurrency_limit=1,
            concurrency_id="revct-inference",
            show_progress="minimal",
            show_progress_on=[findings],
            api_name="analyze_ct",
            api_description="Analyze a HU-valued chest-CT NIfTI volume into 18 typed decisions.",
        )
        inference_event.then(
            render_findings,
            inputs=[current_record, selection],
            outputs=[findings],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        reset_outputs = [
            image,
            geometry,
            analyze,
            summary,
            findings,
            record_json,
            download,
            current_record,
            download_path,
            position,
            plane,
            window,
            selection,
        ]
        volume.change(
            reset_volume,
            inputs=[download_path, session_generation],
            outputs=reset_outputs,
            cancels=[inference_event],
            queue=False,
            show_progress="minimal",
            api_visibility="private",
        ).then(
            load_volume,
            inputs=[volume, session_generation],
            outputs=[image, geometry, analyze],
            queue=False,
            show_progress="minimal",
            show_progress_on=[image],
            api_visibility="private",
        )
        def clear_volume(old_download: str | None, generation: dict[str, int]):
            return (None, *reset_volume(old_download, generation))

        clear.click(
            clear_volume,
            inputs=[download_path, session_generation],
            outputs=[volume, *reset_outputs],
            cancels=[inference_event],
            queue=False,
            api_visibility="private",
        )
        for control in [plane, window]:
            control.input(
                preview,
                inputs=[volume, plane, position, window],
                outputs=[image],
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )
        position.release(
            preview,
            inputs=[volume, plane, position, window],
            outputs=[image],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        selection.input(
            render_findings,
            inputs=[current_record, selection],
            outputs=[findings],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
    return demo.queue(default_concurrency_limit=1, max_size=12)
