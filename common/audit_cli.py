"""CLI: ``python -m common.audit_cli …`` — layer audit + gen compare (report-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import audit, gen_compare
from .errors import ModelError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m common.audit_cli",
        description="Dequant layer audit and light gen/forward compare (report-only, never fails CI)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("layer", help="Stratified layer RMSE audit")
    pl.add_argument("--bundle", required=True, help="bundle directory (config.json + weight.bin)")
    pl.add_argument("--model", default=None, help="HF repo id for reference weights")
    pl.add_argument(
        "--ref",
        choices=("hf", "tiny"),
        default="hf",
        help="hf=stream from --model; tiny=synthetic state dict (offline)",
    )
    pl.add_argument("--sample", type=int, default=8, help="number of tensors to sample")
    pl.add_argument("--seed", type=int, default=0, help="sampling RNG seed")
    pl.add_argument("--family", default=None, help="optional family slug for kind inference")
    pl.add_argument(
        "--report",
        default=None,
        help="JSON report path (default: <bundle>/audit_layer.json)",
    )

    pg = sub.add_parser("gen", help="Text generation or VLA forward compare")
    pg.add_argument("--bundle", required=True)
    pg.add_argument("--model", required=True, help="HF repo id")
    pg.add_argument(
        "--kind",
        choices=(audit.TEXT_KIND, audit.VLA_KIND),
        default=None,
        help="override auto-detect (text vs vla)",
    )
    pg.add_argument("--family", default=None)
    pg.add_argument("--max-new-tokens", type=int, default=32)
    pg.add_argument("--device", default="cpu")
    pg.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="text prompt (repeatable); text kind only",
    )
    pg.add_argument(
        "--report",
        default=None,
        help="JSON report path (default: <bundle>/audit_gen.json)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "layer":
            report = audit.run_layer_audit(
                args.bundle,
                model=args.model,
                sample=args.sample,
                seed=args.seed,
                ref_tiny=(args.ref == "tiny"),
                family=args.family,
            )
            out = Path(args.report or Path(args.bundle) / "audit_layer.json")
        else:
            report = gen_compare.run_gen_compare(
                args.bundle,
                args.model,
                kind=args.kind,
                family=args.family,
                prompts=args.prompt,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
            out = Path(args.report or Path(args.bundle) / "audit_gen.json")
        audit.write_report(report, out)
        print(json.dumps({"report": str(out), "summary": _summary(report)}, indent=2))
        # Spec: never fail CI on threshold breaches.
        return 0
    except ModelError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _summary(report: dict) -> dict:
    if report.get("mode") == "layer":
        return {
            "kind": report.get("kind"),
            "sample": report.get("sample"),
            "fail_count": report.get("fail_count"),
            "ci_fail": report.get("ci_fail"),
        }
    return {
        "kind": report.get("kind"),
        "status": report.get("status"),
        "ci_fail": report.get("ci_fail"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
