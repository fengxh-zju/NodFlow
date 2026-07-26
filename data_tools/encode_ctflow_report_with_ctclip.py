#!/usr/bin/env python
"""Encode one CTFlow report with the authorized CT-CLIP text encoder.

CTFlow's official dataset loader selects token zero from a saved CT-CLIP
sequence embedding and L2-normalizes it. This tool reproduces that contract
without instantiating CT-CLIP's large image encoder: it extracts the trained
``text_transformer`` state directly from the official CT-CLIP checkpoint,
loads it into the matching CXR-BERT architecture, and saves the CLS vector.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch


DEFAULT_TEXT_MODEL = "microsoft/BiomedVLP-CXR-BERT-specialized"
_NON_PARAMETER_TEXT_BUFFERS = {
    "embeddings.position_ids",
    "embeddings.token_type_ids",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unwrap_state_dict(checkpoint):
    value = checkpoint
    for key in ("state_dict", "model_state_dict", "model", "module", "clip"):
        if isinstance(value, dict) and key in value and isinstance(value[key], dict):
            value = value[key]
            break
    if not isinstance(value, dict):
        raise ValueError(f"CT-CLIP checkpoint must contain a state dict, got {type(value).__name__}")
    tensors = {str(key): tensor for key, tensor in value.items() if torch.is_tensor(tensor)}
    if not tensors:
        raise ValueError("CT-CLIP checkpoint contains no tensor state")
    return tensors


def strip_wrappers(key):
    prefixes = ("module.", "model.", "clip.", "_orig_mod.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def extract_text_transformer_state(checkpoint):
    state = unwrap_state_dict(checkpoint)
    extracted = {}
    for key, value in state.items():
        key = strip_wrappers(key)
        marker = "text_transformer."
        if key.startswith(marker):
            extracted[key[len(marker) :]] = value
        elif f".{marker}" in key:
            extracted[key.split(f".{marker}", 1)[1]] = value
    required = {
        "embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight",
        "encoder.layer.0.attention.self.query.weight",
    }
    missing = sorted(required - set(extracted))
    if missing:
        raise ValueError(
            "checkpoint does not expose the official CT-CLIP text_transformer state; "
            f"missing representative keys: {missing}"
        )
    return extracted


def incompatible_parameter_keys(incompatible):
    missing = list(incompatible.missing_keys)
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if key not in _NON_PARAMETER_TEXT_BUFFERS
    ]
    return missing, unexpected


def load_authorized_text_encoder(checkpoint_path, model_id, device):
    from transformers import BertConfig, BertModel, BertTokenizer

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"authorized CT-CLIP checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    text_state = extract_text_transformer_state(checkpoint)
    config = BertConfig.from_pretrained(model_id)
    model = BertModel(config)
    incompatible = model.load_state_dict(text_state, strict=False)
    missing, unexpected = incompatible_parameter_keys(incompatible)
    if missing or unexpected:
        raise ValueError(
            "official CT-CLIP text state is incompatible with the configured CXR-BERT architecture: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    model = model.to(device).eval()
    tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=True)
    return model, tokenizer, len(text_state)


def encode_report(model, tokenizer, report, device):
    tokens = tokenizer(
        report,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.inference_mode():
        hidden = model(**tokens).last_hidden_state
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != 768:
        raise ValueError(f"unexpected CT-CLIP text hidden state shape: {list(hidden.shape)}")
    embedding = hidden[0, 0].detach().float().cpu().contiguous()
    if embedding.shape != (768,) or not torch.isfinite(embedding).all():
        raise ValueError("CT-CLIP CLS embedding is invalid or non-finite")
    embedding = embedding / (embedding.norm(p=2) + 1.0e-6)
    return embedding, int(tokens["attention_mask"].sum().item())


def main():
    parser = argparse.ArgumentParser(description="Create a formal CTFlow condition using official CT-CLIP_v2 weights.")
    parser.add_argument("--text-input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-checkpoint-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()

    text_path = Path(args.text_input)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    if not text_path.exists():
        raise FileNotFoundError(f"report text does not exist: {text_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"authorized CT-CLIP checkpoint does not exist: {checkpoint_path}")
    if checkpoint_path.stat().st_size < int(args.min_checkpoint_bytes):
        raise ValueError(
            f"CT-CLIP checkpoint is unexpectedly small: {checkpoint_path.stat().st_size} "
            f"< {int(args.min_checkpoint_bytes)} bytes"
        )
    report = text_path.read_text(encoding="utf-8").strip()
    if not report:
        raise ValueError(f"report text is empty: {text_path}")

    model, tokenizer, num_text_state_tensors = load_authorized_text_encoder(
        checkpoint_path, args.text_model, torch.device(args.device)
    )
    embedding, token_count = encode_report(model, tokenizer, report, torch.device(args.device))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding, output_path)
    result = {
        "status": "ok",
        "implementation": "official_ctclip_text_transformer_cls",
        "text_input": str(text_path),
        "text_sha256": sha256_file(text_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "text_model_architecture": args.text_model,
        "num_text_state_tensors": num_text_state_tensors,
        "token_count": token_count,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "shape": list(embedding.shape),
        "l2_norm": float(embedding.norm(p=2).item()),
        "finite": bool(torch.isfinite(embedding).all().item()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
