# NodFlow

Official implementation of **NodFlow**, a training-free lesion-constrained flow
matching method for 3D lung nodule synthesis. NodFlow uses a frozen volumetric
CT prior (MAISI or CTFlow) and optimizes lesion-aware constraints during
sampling; the generator itself is not fine-tuned.

> **Research use only.** This software is not a medical device and must not be
> used for diagnosis or clinical decision-making.

## Method

NodFlow consists of four stages:

1. load a normal 3D CT region and a target nodule condition;
2. initialize generation with a frozen MAISI or CTFlow prior;
3. apply lesion-aware background, phenotype, proximity, and spatial-texture
   constraints during flow sampling;
4. decode the optimized latent and preserve non-target anatomy.

Users provide data paths, prior paths, and experiment settings through local
YAML files and environment variables.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/fengxh-zju/NodFlow.git
cd NodFlow

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[generative]"
```

Install a CUDA-enabled PyTorch build that matches the local driver. CTFlow may
also require the dependencies declared by its upstream repository.

## Data

Medical data is not distributed with this repository. Download LIDC-IDRI,
LUNA16, or LNDb independently and keep raw and processed files outside Git.
The required processed layout is defined in
[`docs/data_contract.md`](docs/data_contract.md).

Preparation entry points:

```bash
# LIDC-IDRI scans paired with LUNA16 annotations
python data_tools/prepare_lidc_idc_luna.py --help

# LUNA16
python data_tools/prepare_luna16.py --help

# LNDb
python data_tools/prepare_lndb.py --help
```

Then create group-aware splits, the training-only target library, and method
manifests:

```bash
python data_tools/build_splits.py --help
python data_tools/build_target_library.py --help
python data_tools/build_method_manifests.py --help
```

### Public-data representation

The preparation tools turn each public annotation into a fixed-size 3D CT ROI
and the following inputs:

- `m` is the binary nodule support mask. The LUNA16 and
  LIDC-IDRI-with-LUNA16 paths rasterize a sphere at the published center using
  the published diameter. The LNDb path uses the same geometric construction
  with its equivalent-diameter field. These masks localize the edit but are
  approximations, not radiologist-drawn nodule boundaries.
- The nodule condition `c` contains `m`, diameter, a coarse
  `ground_glass`/`part_solid`/`solid` subtype, and a target HU histogram
  computed inside a training nodule mask. LNDb subtype uses its texture rating
  when available; the LUNA16-based paths use diameter bands as a reproducible
  proxy and should not be interpreted as clinical subtype ground truth.
  Available public scalar ratings are preserved as optional condition metadata.
- The pathological ROI is the CT crop around the annotation. The paired source
  ROI used by this pipeline replaces voxels inside `m` with the surrounding
  median HU value; the original implementation records both volumes and their
  provenance.

Richer public information can be incorporated, but it is not equivalent to a
spatial annotation. For example, original LIDC-IDRI reader contours can be
rasterized and aggregated to replace the spherical mask, while LIDC-IDRI or
LNDb semantic ratings can extend `c` with margin, lobulation, spiculation,
texture, or malignancy attributes. A scalar rating does not identify where a
spicule, vessel attachment, or texture pattern occurs. Spatial control of
those structures requires reader contours, manually curated structure labels,
or predictions from a separately validated vessel or morphology model.

The default representation is therefore a deliberately limited common
denominator supported by all three preparation paths. The same mask
construction, condition fields, split rules, and training-only target library
must be used for every compared method. This makes comparisons controlled and
reproducible, while leaving fine-grained morphology control as an explicitly
separate extension rather than claiming labels that the selected public tables
do not provide.

Validate the prepared data before generation:

```bash
python data_tools/validate_real_data_contracts.py \
  --processed_root /path/to/processed \
  --target_library /path/to/processed/target_library \
  --splits_dir /path/to/processed/splits \
  --manifests_dir /path/to/processed/manifests \
  --output /path/to/validation.json
```

## Configuration

Copy [`configs/templates/data.yaml`](configs/templates/data.yaml) and either
[`configs/templates/nodflow_maisi.yaml`](configs/templates/nodflow_maisi.yaml)
or [`configs/templates/nodflow_ctflow.yaml`](configs/templates/nodflow_ctflow.yaml)
to a user-controlled location.

The method descriptor points to a user-owned hyperparameter file:

```yaml
method_id: nodflow_maisi
implementation: training_free_lesion_constrained_flow_matching
prior: maisi_rflow
maisi_bundle: ${MAISI_BUNDLE_ROOT}
maisi_weight_root: ${MAISI_WEIGHT_ROOT}
hyperparameters_file: /path/to/nodflow_hparams.yaml
```

The hyperparameter file is a flat YAML mapping consumed by the implementation.
See [`docs/configuration.md`](docs/configuration.md).

### Custom posterior energies

NodFlow can add user-defined differentiable constraints to MAISI or CTFlow
latent refinement without changing the frozen prior. Each function receives a
[`PosteriorEnergyContext`](src/nodflow/energies.py) containing the decoded
volume, lesion and background masks, source image, target histogram, selected
prior, and auxiliary metadata. The current target entry, including subtype,
diameter, and available public semantic ratings, is exposed as
`context.metadata["condition"]`. Image tensors are shaped `[B, C, D, H, W]`
and normalized to `[0, 1]`.

Define an importable function that returns one scalar PyTorch tensor:

```python
def morphology_energy(context, target, **options):
    measured = differentiable_morphology_measure(
        context.decoded, context.lesion_mask, **options
    )
    return (measured - target).square()
```

Register it in the user hyperparameter YAML:

```yaml
posterior_energies:
  - name: morphology
    callable: my_project.energies:morphology_energy
    weight: ${MORPHOLOGY_ENERGY_WEIGHT}
    priors: [maisi_rflow, ctflow]
    kwargs:
      target: ${MORPHOLOGY_TARGET}
```

Entries may be restricted to one prior or disabled with `enabled: false`.
Texture, vessel-connectivity, and spiculation energies can use the same
interface, provided their callable remains differentiable with respect to
`context.decoded`. No entry means the standard NodFlow objective is unchanged.
Custom energies are active whenever the selected prior's latent-refinement
steps are enabled, and their weighted values are recorded in refinement
metadata.

Validate both descriptors before using a GPU:

```bash
python data_tools/validate_config.py /path/to/data.yaml --kind data
python data_tools/validate_config.py /path/to/nodflow_method.yaml --kind method
```

## Run NodFlow

The preparation command validates required assets and writes the run manifest.
It does not train or fine-tune the frozen generator.

```bash
python -m src.nodflow.prepare \
  --data_config /path/to/data.yaml \
  --method_config /path/to/nodflow_method.yaml

python -m src.nodflow.generate \
  --data_config /path/to/data.yaml \
  --method_config /path/to/nodflow_method.yaml \
  --num_samples 100
```

Generated volumes, masks, source inputs, and metadata are written below the
`output_root` specified by the data configuration.

## Repository Layout

```text
configs/       public data and method templates
data_tools/    data preparation and contract validation
docs/          data and configuration contracts
src/nodflow/   NodFlow implementation and frozen-prior adapters
```

## Reproducibility and Privacy

- Raw data, processed data, outputs, checkpoints, weights, and local configs
  must remain outside Git.
- Splits are group-aware and target libraries must contain training cases only.
- Generated metadata records source, target condition, split, and asset
  provenance.
- Never publish patient identifiers or protected health information.

## Acknowledgements

NodFlow integrates frozen priors from
[MONAI/MAISI](https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi)
and [CTFlow](https://github.com/WongJiayi/CTFlow). Please follow their licenses
and cite the corresponding papers when using those priors.

## License

Repository code is released under the [Apache License 2.0](LICENSE). Datasets,
third-party repositories, and model weights remain subject to their own terms.
