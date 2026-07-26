# Processed data contract

Each processed dataset uses the following layout:

```text
processed-root/
|-- rois/
|   |-- normal/
|   |   |-- images/
|   |   |-- masks/
|   |   `-- metadata/
|   `-- pathological/
|       |-- images/
|       |-- masks/
|       `-- metadata/
|-- splits/
|   |-- train_cases.txt
|   |-- val_cases.txt
|   `-- test_cases.txt
|-- target_library/
|   |-- histograms/
|   |-- masks/
|   `-- metadata/
`-- manifests/
```

Volumes use NIfTI or NumPy-compatible storage supported by `src.utils.io`.
Metadata must preserve a de-identified case/group ID, source geometry, nodule
subtype, image path, and mask path. Split assignment is group-aware, and the
target library must be built from training cases only.

Run the contract validator after preparation:

```bash
python data_tools/validate_real_data_contracts.py \
  --processed_root /path/to/processed-root \
  --target_library /path/to/processed-root/target_library \
  --splits_dir /path/to/processed-root/splits \
  --manifests_dir /path/to/processed-root/manifests \
  --output /path/to/validation.json
```
