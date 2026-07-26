# Runtime configuration

The repository separates portable method descriptors from user-supplied
experiment hyperparameters.

## Data configuration

Start from `configs/templates/data.yaml`. It defines processed ROI storage,
split files, a train-only target library, output roots, CT geometry, and
histogram preprocessing. Environment variables are expanded before YAML is
parsed, so list and numeric values may be supplied as YAML strings such as
`"[1.0, 1.0, 1.0]"`.

## Method descriptor

Choose the MAISI or CTFlow descriptor in `configs/templates/`. It identifies
the frozen prior and its local weight locations. Real-data descriptors must
include `hyperparameters_file`.

## Hyperparameters

The referenced hyperparameter file is a flat YAML mapping consumed by the
selected implementation. Optional `posterior_energies` entries register
differentiable user constraints using the interface documented in the README.

Validate before launching:

```bash
python data_tools/validate_config.py /path/to/data.yaml --kind data
python data_tools/validate_config.py /path/to/method.yaml --kind method
```

Missing files, unresolved environment variables, and empty hyperparameter
files fail before GPU initialization.
