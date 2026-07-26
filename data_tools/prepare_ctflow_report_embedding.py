#!/usr/bin/env python
import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.io import load_json


DATA_ROOT = Path(os.environ.get("LUNG_NODULE_DATA_ROOT", "data"))


def sha256_file(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(value, default="unknown"):
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _float_text(value, suffix="", default="unknown"):
    try:
        val = float(value)
        if not np.isfinite(val):
            return default
        return f"{val:.1f}{suffix}"
    except Exception:
        return default


def report_from_metadata(meta):
    subtype = _clean(meta.get("target_subtype") or meta.get("subtype"), "pulmonary nodule")
    diameter = _float_text(meta.get("diameter_mm"), " mm")
    volume = _float_text(meta.get("volume_mm3"), " cubic mm")
    lobe = _clean(meta.get("lobe"), "lung")
    location = _clean(meta.get("location_type"), "intraparenchymal")
    pleural = _float_text(meta.get("pleural_distance_mm"), " mm")
    hu_hint = ""
    hist = meta.get("target_histogram") or meta.get("actual_histogram")
    if isinstance(hist, list) and hist:
        arr = np.asarray(hist, dtype=np.float64)
        if arr.sum() > 0:
            bins = np.linspace(-1000.0, 400.0, arr.size + 1)
            centers = (bins[:-1] + bins[1:]) * 0.5
            mean = float((arr / arr.sum() * centers).sum())
            hu_hint = f" The lesion attenuation distribution is centered around approximately {mean:.1f} HU."

    return (
        "Non-contrast chest CT. "
        f"There is a {subtype} lung nodule in the {lobe}, {location} location. "
        f"The nodule diameter is approximately {diameter}; estimated volume is {volume}. "
        f"The nearest pleural distance is approximately {pleural}."
        f"{hu_hint} "
        "No task-specific generator fine-tuning is requested; this report is used only as the frozen CTFlow global condition."
    )


def read_metadata(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"empty CSV: {path}")
        return rows[0]
    return load_json(path)


def load_embedding(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        value = torch.from_numpy(np.load(path))
    else:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        for key in ["embedding", "text_embedding", "prompt_embedding", "last_hidden_state"]:
            if key in value:
                value = value[key]
                break
    if not hasattr(value, "shape"):
        raise ValueError(f"embedding input {path} did not contain a tensor-like value")
    value = torch.as_tensor(value).detach().float().cpu()
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.shape[-1] != 768:
        raise ValueError(f"expected CTFlow/CT-CLIP embedding last dimension 768, got shape {list(value.shape)}")
    value = value.reshape(-1, 768)
    value = value[0].contiguous()
    value = value / (value.norm(p=2) + 1.0e-6)
    return value


def embedding_stats(embedding):
    value = torch.as_tensor(embedding).detach().float().cpu()
    return {
        "shape": [int(v) for v in value.shape],
        "l2_norm": float(value.norm(p=2).item()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "finite": bool(torch.isfinite(value).all().item()),
    }


def is_non_formal_path(path):
    lower = str(path).lower()
    return "dummy" in lower or "not_for_formal" in lower or "placeholder" in lower


def update_method_config(config_path, embedding_path):
    cfg = load_config(config_path)
    cfg["ctflow_embedding"] = str(embedding_path)
    text = json.dumps(cfg, indent=2)
    try:
        import yaml

        text = yaml.safe_dump(cfg, sort_keys=False)
    except Exception:
        pass
    Path(config_path).write_text(text, encoding="utf-8")


def default_ctclip_model_path(cfg):
    configured = cfg.get("ctclip_model_path")
    if configured:
        return Path(str(configured))
    model_file = cfg.get("ctclip_model_file", "models/CT-CLIP-Related/CT-CLIP_v2.pt")
    return DATA_ROOT / "checkpoints" / "ctclip" / str(model_file)


def format_encoder_command(command, text_input, embedding_output, ctclip_model, ctclip_text_model):
    values = {
        "text_input": str(text_input),
        "embedding_output": str(embedding_output),
        "ctclip_model": str(ctclip_model),
        "ctclip_text_model": str(ctclip_text_model),
    }
    try:
        return command.format(**{key: shlex.quote(value) for key, value in values.items()})
    except KeyError as exc:
        raise ValueError(f"unknown encoder-command placeholder {exc}") from exc


def run_encoder_command(command, text_input, embedding_output, ctclip_model, ctclip_text_model):
    if not command:
        raise ValueError("empty encoder command")
    if not ctclip_model.exists():
        raise FileNotFoundError(
            f"CT-CLIP model checkpoint not found: {ctclip_model}. "
            "Download the authorized CT-RATE CT-CLIP_v2.pt asset first."
        )
    rendered = format_encoder_command(command, text_input, embedding_output, ctclip_model, ctclip_text_model)
    env = dict(os.environ)
    env["CTCLIP_MODEL"] = str(ctclip_model)
    env["CTCLIP_TEXT_MODEL"] = str(ctclip_text_model)
    env["CTFLOW_REPORT_TEXT"] = str(text_input)
    env["CTFLOW_EMBEDDING_OUTPUT"] = str(embedding_output)
    subprocess.run(shlex.split(rendered), check=True, env=env)


def official_encoder_command():
    encoder = ROOT / "data_tools" / "encode_ctflow_report_with_ctclip.py"
    return (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(encoder))} "
        "--text-input {text_input} --checkpoint {ctclip_model} --output {embedding_output} "
        "--text-model {ctclip_text_model}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CTFlow report text and, when a real CT-CLIP tensor is supplied, "
            "a normalized 768-d report embedding."
        )
    )
    parser.add_argument("--metadata", required=True, help="Target metadata JSON/CSV row used to build report text.")
    parser.add_argument("--text-output", required=True)
    parser.add_argument("--embedding-input", default="", help="Existing real CT-CLIP/report embedding .pt or .npy.")
    parser.add_argument("--embedding-output", default="")
    parser.add_argument(
        "--ctclip-model",
        default="",
        help="Authorized CT-CLIP_v2.pt checkpoint used by --encoder-command. Defaults to config ctclip_model_path or the downloader cache path.",
    )
    parser.add_argument(
        "--ctclip-text-model",
        default="",
        help=(
            "CXR-BERT config/tokenizer directory or Hugging Face id. Defaults to ctclip_text_model in the "
            "method config. The official CT-CLIP checkpoint supplies all trained model weights."
        ),
    )
    parser.add_argument(
        "--encoder-command",
        default=os.environ.get("CTFLOW_CTCLIP_ENCODER_COMMAND", ""),
        help=(
            "Official/external CT-CLIP text-encoder command. It must write a 768-d tensor to {embedding_output}. "
            "Placeholders: {text_input}, {embedding_output}, {ctclip_model}. No heuristic embedding is generated."
        ),
    )
    parser.add_argument(
        "--official-ctclip-encoder",
        action="store_true",
        help=(
            "Use data_tools/encode_ctflow_report_with_ctclip.py to extract the trained text_transformer "
            "from the authorized CT-CLIP_v2 checkpoint and encode the report CLS token."
        ),
    )
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    if args.official_ctclip_encoder and args.encoder_command:
        raise SystemExit("choose either --official-ctclip-encoder or --encoder-command, not both")
    encoder_command = official_encoder_command() if args.official_ctclip_encoder else args.encoder_command

    cfg = load_config(args.method_config)
    ctclip_text_model = args.ctclip_text_model or cfg.get(
        "ctclip_text_model", "microsoft/BiomedVLP-CXR-BERT-specialized"
    )
    meta = read_metadata(args.metadata)
    report = report_from_metadata(meta)
    text_output = Path(args.text_output)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.write_text(report + "\n", encoding="utf-8")

    result = {
        "metadata": str(args.metadata),
        "text_output": str(text_output),
        "report_text": report,
        "embedding_created": False,
        "embedding_output": "",
        "embedding_shape": "",
        "embedding_source": "",
        "ctclip_model": "",
        "encoder_command_used": False,
        "strict_note": (
            "No embedding is created unless --embedding-input supplies a real 768-d CT-CLIP/report tensor, "
            "or --encoder-command runs an authorized CT-CLIP text encoder and writes a real 768-d tensor."
        ),
    }

    if args.embedding_input:
        if not args.embedding_output:
            raise SystemExit("--embedding-output is required with --embedding-input")
        embedding = load_embedding(args.embedding_input)
        embedding_output = Path(args.embedding_output)
        embedding_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding, embedding_output)
        result.update(
            {
                "embedding_created": True,
                "embedding_output": str(embedding_output),
                "embedding_shape": list(embedding.shape),
                "embedding_source": str(args.embedding_input),
                "embedding_source_sha256": sha256_file(args.embedding_input),
                "embedding_output_sha256": sha256_file(embedding_output),
                "embedding_stats": embedding_stats(embedding),
                "embedding_provenance_status": "formal_candidate"
                if not is_non_formal_path(args.embedding_input)
                else "non_formal_source_path",
            }
        )
        if args.write_config:
            update_method_config(args.method_config, embedding_output)
    elif encoder_command:
        if not args.embedding_output:
            raise SystemExit("--embedding-output is required with --encoder-command")
        embedding_output = Path(args.embedding_output)
        embedding_output.parent.mkdir(parents=True, exist_ok=True)
        ctclip_model = Path(args.ctclip_model) if args.ctclip_model else default_ctclip_model_path(cfg)
        run_encoder_command(encoder_command, text_output, embedding_output, ctclip_model, ctclip_text_model)
        embedding = load_embedding(embedding_output)
        torch.save(embedding, embedding_output)
        result.update(
            {
                "embedding_created": True,
                "embedding_output": str(embedding_output),
                "embedding_shape": list(embedding.shape),
                "embedding_source": "official_ctclip_text_transformer"
                if args.official_ctclip_encoder
                else "encoder_command",
                "ctclip_model": str(ctclip_model),
                "ctclip_text_model": str(ctclip_text_model),
                "ctclip_model_sha256": sha256_file(ctclip_model),
                "embedding_output_sha256": sha256_file(embedding_output),
                "embedding_stats": embedding_stats(embedding),
                "embedding_provenance_status": "formal_candidate"
                if not is_non_formal_path(embedding_output) and not is_non_formal_path(ctclip_model)
                else "non_formal_source_path",
                "encoder_command_used": True,
                "encoder_implementation": "official_ctclip_text_transformer_cls"
                if args.official_ctclip_encoder
                else "external_encoder_command",
            }
        )
        if args.write_config:
            update_method_config(args.method_config, embedding_output)
    elif not args.text_only:
        raise SystemExit(
            "CTFlow formal validation requires a real 768-d CT/report embedding. "
            "Provide --embedding-input, use --official-ctclip-encoder with an authorized CT-CLIP_v2 checkpoint, "
            "or provide --encoder-command with an authorized CT-CLIP text encoder, "
            "or pass --text-only to write only the report prompt."
        )

    sidecar = text_output.with_suffix(text_output.suffix + ".json")
    sidecar.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result.get("embedding_created") and result.get("embedding_output"):
        embedding_sidecar = Path(result["embedding_output"]).with_suffix(Path(result["embedding_output"]).suffix + ".json")
        embedding_sidecar.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
