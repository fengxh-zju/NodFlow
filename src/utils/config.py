import os
import re
from pathlib import Path


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment(text, path):
    missing = set()

    def replace(match):
        name = match.group(1)
        value = os.environ.get(name)
        if value is None:
            missing.add(name)
            return match.group(0)
        return value

    expanded = _ENV_PATTERN.sub(replace, text)
    if missing:
        names = ", ".join(sorted(missing))
        raise KeyError(f"missing environment variables in {path}: {names}")
    return expanded


def _deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path, _seen=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    path = path.resolve()
    seen = set(_seen or ())
    if path in seen:
        chain = " -> ".join(str(item) for item in [*seen, path])
        raise ValueError(f"cyclic config extends chain: {chain}")
    seen.add(path)
    text = _expand_environment(path.read_text(encoding="utf-8"), path)
    try:
        import yaml

        config = yaml.safe_load(text) or {}
    except Exception:
        config = _tiny_yaml(text)
    if not isinstance(config, dict):
        raise TypeError(f"config root must be a mapping: {path}")
    parent = config.pop("extends", None)
    if not parent:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(load_config(parent_path, _seen=seen), config)


def load_method_config(path):
    path = Path(path).resolve()
    config = load_config(path)
    hyperparameters_file = config.get("hyperparameters_file")
    if not hyperparameters_file:
        raise ValueError(f"method config must declare hyperparameters_file: {path}")
    hyperparameters_path = Path(hyperparameters_file)
    if not hyperparameters_path.is_absolute():
        hyperparameters_path = path.parent / hyperparameters_path
    hyperparameters = load_config(hyperparameters_path)
    if not hyperparameters:
        raise ValueError(f"hyperparameter file is empty: {hyperparameters_path}")
    merged = _deep_merge(config, hyperparameters)
    merged["_hyperparameters_file"] = str(hyperparameters_path.resolve())
    return merged


def _parse_scalar(value):
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value == "{}":
        return {}
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x) for x in inner.split(",")]
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def _tiny_yaml(text):
    data = {}
    stack = [(0, data)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, value = raw.strip().partition(":")
        while stack and indent < stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = _parse_scalar(value)
        else:
            parent[key] = {}
            stack.append((indent + 2, parent[key]))
    return data
