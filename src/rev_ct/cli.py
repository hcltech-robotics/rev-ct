from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .head import RevHead


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rev 18-label chest CT decision inference")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("features", "predict"):
        p = commands.add_parser(command)
        p.add_argument("inputs", nargs="+" if command == "predict" else None)
        p.add_argument("--model", default="hcltech-robotics/rev-ct")
        p.add_argument("--revision")
        p.add_argument("--output", default="-")
        p.add_argument("--head-mode", choices=["reference", "fused"], default="reference")
        if command == "predict":
            p.add_argument("--device", default="cuda")
            p.add_argument("--precision", choices=["bf16", "fp32"], default="fp32")
            p.add_argument("--workers", type=int, default=0)
            p.add_argument("--batch-size", type=int, default=1)
            p.add_argument("--preprocessing", choices=["reference", "fast"], default="reference")
            p.add_argument("--compile", choices=["default", "reduce-overhead", "max-autotune"])
    p = commands.add_parser("export", help="Convert a trusted source training artifact")
    p.add_argument("input")
    p.add_argument("destination")
    p.add_argument("--source-revision", required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        import joblib

        from .export import export_artifact

        export_artifact(
            joblib.load(args.input), args.destination, source_revision=args.source_revision
        )
        return
    head = RevHead.from_pretrained(args.model, revision=args.revision)
    if args.command == "features":
        with np.load(args.inputs, allow_pickle=False) as data:
            features = data["features"] if "features" in data else data["feats"]
            names = data["names"].astype(str)
            records = head.records(features, names, mode=args.head_mode)
    else:
        from .inference import RevInference

        model = RevInference(
            head,
            device=args.device,
            precision=args.precision,
            compile_mode=args.compile,
            preprocessing=args.preprocessing,
        )
        records = model.predict(
            args.inputs, workers=args.workers, batch_size=args.batch_size, head_mode=args.head_mode
        )
    context = nullcontext(sys.stdout) if args.output == "-" else Path(args.output).open("w")
    with context as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
