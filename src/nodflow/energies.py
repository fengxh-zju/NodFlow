import math
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class PosteriorEnergyContext:
    """Tensors and metadata exposed to a user-defined posterior energy."""

    decoded: Any
    lesion_mask: Any
    source_image: Any
    background_mask: Any
    target_histogram: Any
    prior: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PosteriorEnergy:
    name: str
    weight: float
    function: Callable[..., Any]
    kwargs: Mapping[str, Any]
    source: str


def _resolve_callable(value):
    if callable(value):
        return value, f"{value.__module__}:{value.__qualname__}"
    if not isinstance(value, str) or not value.strip():
        raise TypeError("posterior energy callable must be a callable or import path")
    path = value.strip()
    if ":" in path:
        module_name, attribute = path.split(":", 1)
    else:
        module_name, separator, attribute = path.rpartition(".")
        if not separator:
            raise ValueError(
                f"posterior energy callable must use 'package.module:function': {path!r}"
            )
    function = getattr(import_module(module_name), attribute)
    if not callable(function):
        raise TypeError(f"posterior energy target is not callable: {path!r}")
    return function, path


def load_posterior_energies(config, prior):
    """Load enabled energy specifications that apply to the selected prior."""

    raw_specs = config.get("posterior_energies", [])
    if raw_specs is None:
        return ()
    if not isinstance(raw_specs, list):
        raise TypeError("posterior_energies must be a list of mappings")

    energies = []
    names = set()
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, Mapping):
            raise TypeError(f"posterior_energies[{index}] must be a mapping")
        if not bool(raw.get("enabled", True)):
            continue
        priors = raw.get("priors")
        if priors is not None:
            priors = [priors] if isinstance(priors, str) else list(priors)
            if prior not in priors:
                continue

        function, source = _resolve_callable(raw.get("callable"))
        name = str(raw.get("name") or getattr(function, "__name__", f"energy_{index}"))
        if not name or name in names:
            raise ValueError(f"posterior energy names must be non-empty and unique: {name!r}")
        names.add(name)

        weight = float(raw.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"posterior energy {name!r} has invalid weight: {weight}")
        kwargs = raw.get("kwargs") or {}
        if not isinstance(kwargs, Mapping):
            raise TypeError(f"posterior energy {name!r} kwargs must be a mapping")
        energies.append(
            PosteriorEnergy(
                name=name,
                weight=weight,
                function=function,
                kwargs=dict(kwargs),
                source=source,
            )
        )
    return tuple(energies)


def evaluate_posterior_energies(energies, context):
    """Return the weighted scalar sum and weighted terms for logging."""

    import torch

    total = context.decoded.sum() * 0.0
    terms = {}
    for energy in energies:
        try:
            value = energy.function(context, **energy.kwargs)
        except Exception as exc:
            raise RuntimeError(f"posterior energy {energy.name!r} failed") from exc
        if not torch.is_tensor(value):
            raise TypeError(
                f"posterior energy {energy.name!r} must return a torch.Tensor, "
                f"got {type(value).__name__}"
            )
        if value.numel() != 1:
            raise ValueError(
                f"posterior energy {energy.name!r} must return one scalar, "
                f"got shape {list(value.shape)}"
            )
        value = value.reshape(())
        if not bool(torch.isfinite(value).detach().cpu()):
            raise ValueError(f"posterior energy {energy.name!r} returned a non-finite value")
        if torch.is_grad_enabled() and energy.weight > 0 and not value.requires_grad:
            raise ValueError(
                f"posterior energy {energy.name!r} is detached from the decoded image"
            )
        weighted = energy.weight * value
        total = total + weighted
        terms[energy.name] = weighted
    return total, terms


def describe_posterior_energies(energies):
    return [
        {"name": energy.name, "weight": energy.weight, "callable": energy.source}
        for energy in energies
    ]
