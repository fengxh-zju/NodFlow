import json
import math
from pathlib import Path

import numpy as np

from src.nodflow.base import NodFlowRuntime
from src.nodflow.energies import (
    PosteriorEnergyContext,
    describe_posterior_energies,
    evaluate_posterior_energies,
    load_posterior_energies,
)
from src.datasets.roi_dataset import list_roi_image_paths
from src.metrics.histogram import compute_histogram, js_divergence
from src.utils.geometry import boundary_band
from src.utils.io import ensure_dir, load_json, load_volume, save_json, save_volume, write_jsonl
from src.utils.seeds import seed_everything


HU_MIN = -1000.0
HU_MAX = 400.0


def _install_scipy_stats_import_stub():
    """Avoid importing broken scipy.stats in the NV-Generate-CTMR helper path."""
    import sys
    import types

    import scipy

    stub = types.ModuleType("scipy.stats")
    stub.__all__ = []
    stub.__doc__ = (
        "Compatibility stub installed by NodFlow because "
        "the MAISI rflow helper imports scipy.stats but does not use it."
    )
    stub.__file__ = "<nodflow scipy.stats import stub>"
    sys.modules["scipy.stats"] = stub
    setattr(scipy, "stats", stub)
    return "scipy.stats_stub_for_nv_generate_ctmr"


def _patch_monai_rflow_scheduler_input_size():
    import numpy as np
    import torch
    from monai.networks.schedulers.rectified_flow import RFlowScheduler, timestep_transform

    if getattr(RFlowScheduler.set_timesteps, "_lung_nodule_cf_patched", False):
        return "monai_rflow_input_size_int_patch"

    def set_timesteps(self, num_inference_steps, device=None, input_img_size_numel=None):
        if num_inference_steps > self.num_train_timesteps or num_inference_steps < 1:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} should be at least 1, "
                "and cannot be larger than `self.num_train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )
        if input_img_size_numel is not None:
            try:
                input_img_size_numel = int(input_img_size_numel.item())
            except AttributeError:
                input_img_size_numel = int(input_img_size_numel)
        self.num_inference_steps = num_inference_steps
        timesteps = [(1.0 - i / self.num_inference_steps) * self.num_train_timesteps for i in range(self.num_inference_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [
                timestep_transform(
                    t,
                    input_img_size_numel=input_img_size_numel,
                    base_img_size_numel=self.base_img_size_numel,
                    num_train_timesteps=self.num_train_timesteps,
                    spatial_dim=self.spatial_dim,
                )
                for t in timesteps
            ]
        timesteps_np = np.array(timesteps).astype(np.float16)
        if self.use_discrete_timesteps:
            timesteps_np = timesteps_np.astype(np.int64)
        timesteps_tensor = torch.tensor(timesteps_np)
        self.timesteps = timesteps_tensor.to(device) if device is not None else timesteps_tensor
        self.timesteps += self.steps_offset

    set_timesteps._lung_nodule_cf_patched = True
    RFlowScheduler.set_timesteps = set_timesteps
    return "monai_rflow_input_size_int_patch"


def _patch_ctflow_pos_embed_from_numpy():
    """Patch CTFlow STDiT position embeddings for current NumPy/PyTorch ABI."""
    import torch
    import echosyn.common.models as models

    stdit = getattr(models, "STDiT", None)
    if stdit is None:
        return "ctflow_pos_embed_patch_unavailable"
    if getattr(stdit, "_lung_nodule_cf_pos_embed_patched", False):
        return "ctflow_pos_embed_tensor_patch"

    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = self.input_size[1:]
        pos_embed = models.get_2d_sincos_pos_embed(
            self.hidden_size,
            (grid_size[0] // self.patch_size[1], grid_size[1] // self.patch_size[2]),
            scale=self.space_scale,
        )
        return torch.tensor(pos_embed, dtype=torch.float32).unsqueeze(0).requires_grad_(False)

    def get_temporal_pos_embed(self):
        pos_embed = models.get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size[0],
            scale=self.time_scale,
        )
        return torch.tensor(pos_embed, dtype=torch.float32).unsqueeze(0).requires_grad_(False)

    stdit.get_spatial_pos_embed = get_spatial_pos_embed
    stdit.get_temporal_pos_embed = get_temporal_pos_embed
    stdit._lung_nodule_cf_pos_embed_patched = True
    return "ctflow_pos_embed_tensor_patch"


def _normal_case_paths(data_config, split):
    return list_roi_image_paths(data_config, kind="normal", split=split)


def _smooth(mask, sigma):
    mask = np.asarray(mask, dtype=np.float32)
    if sigma <= 0:
        return mask
    try:
        from scipy.ndimage import gaussian_filter

        out = gaussian_filter(mask, sigma=float(sigma))
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    except Exception:
        return mask


def _multiscale_3d_texture_energy(
    decoded_unit,
    target_unit,
    masks,
    scales=(1, 2, 4),
    band_weights=(0.55, 0.30, 0.15),
    autocorr_lags=(1, 2),
    core_erosion=1,
    min_voxels=8,
    channel_weights=None,
    local_variance_weight=0.0,
    gram_weight=0.0,
    patch_swd_weight=0.0,
    patch_swd_sizes=(3,),
    patch_swd_projections=12,
    patch_swd_min_support=0.25,
    patch_swd_max_samples=2048,
    patch_swd_seed=1729,
    patch_swd_quantiles=257,
    patch_target_unit=None,
    patch_target_masks=None,
):
    """Match masked multiscale statistics and higher-order 3D patch distributions."""
    import torch
    import torch.nn.functional as F

    if decoded_unit.ndim != 5 or target_unit.ndim != 5 or masks.ndim != 5:
        raise ValueError("texture energy expects [batch, channel, depth, height, width] tensors")
    if decoded_unit.shape[1] != 1 or target_unit.shape[1] != 1:
        raise ValueError("texture energy currently expects one CT intensity channel")
    if tuple(decoded_unit.shape[-3:]) != tuple(target_unit.shape[-3:]):
        target_unit = F.interpolate(
            target_unit,
            size=decoded_unit.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
    if tuple(masks.shape[-3:]) != tuple(decoded_unit.shape[-3:]):
        masks = F.interpolate(masks.float(), size=decoded_unit.shape[-3:], mode="nearest")
    if masks.shape[0] == 1 and decoded_unit.shape[0] > 1:
        masks = masks.expand(decoded_unit.shape[0], -1, -1, -1, -1)
    if masks.shape[0] != decoded_unit.shape[0]:
        raise ValueError("texture masks and decoded CT must have the same batch size")

    scales = tuple(int(value) for value in scales)
    weights = tuple(float(value) for value in band_weights)
    lags = tuple(int(value) for value in autocorr_lags)
    if not scales or any(value <= 0 for value in scales):
        raise ValueError(f"texture scales must be positive, got {scales}")
    if len(scales) != len(weights) or any(value < 0 for value in weights):
        raise ValueError("texture band_weights must be non-negative and match scales")
    if not any(weights):
        raise ValueError("texture band_weights must contain positive mass")
    if any(value <= 0 for value in lags):
        raise ValueError(f"texture autocorrelation lags must be positive, got {lags}")
    local_variance_weight = float(local_variance_weight)
    gram_weight = float(gram_weight)
    patch_swd_weight = float(patch_swd_weight)
    patch_swd_sizes = tuple(int(value) for value in patch_swd_sizes)
    patch_swd_projections = int(patch_swd_projections)
    patch_swd_min_support = float(patch_swd_min_support)
    patch_swd_max_samples = int(patch_swd_max_samples)
    patch_swd_seed = int(patch_swd_seed)
    patch_swd_quantiles = int(patch_swd_quantiles)
    if local_variance_weight < 0 or gram_weight < 0 or patch_swd_weight < 0:
        raise ValueError(
            "texture local-variance, Gram, and patch-SWD weights must be non-negative"
        )
    if patch_swd_weight > 0:
        if not patch_swd_sizes or any(size <= 0 or size % 2 == 0 for size in patch_swd_sizes):
            raise ValueError(
                f"texture patch-SWD sizes must be positive odd integers, got {patch_swd_sizes}"
            )
        if (
            patch_swd_projections <= 0
            or patch_swd_max_samples <= 0
            or patch_swd_quantiles < 2
        ):
            raise ValueError(
                "texture patch-SWD projections/max samples must be positive and "
                "quantiles must be at least 2"
            )
        if not 0.0 <= patch_swd_min_support <= 1.0:
            raise ValueError("texture patch-SWD min support must be in [0, 1]")

    hard_masks = (masks > 0.5).to(decoded_unit.dtype)
    erosion = max(0, int(core_erosion))
    if erosion:
        kernel = 2 * erosion + 1
        eroded = 1.0 - F.max_pool3d(
            1.0 - hard_masks,
            kernel_size=kernel,
            stride=1,
            padding=erosion,
        )
        enough = eroded.sum(dim=(2, 3, 4), keepdim=True) >= float(min_voxels)
        core_masks = torch.where(enough, eroded, hard_masks)
    else:
        core_masks = hard_masks

    channels = int(core_masks.shape[1])
    if channel_weights is None:
        channel_weights_t = torch.ones(channels, device=decoded_unit.device, dtype=decoded_unit.dtype)
    else:
        channel_weights_t = torch.as_tensor(
            channel_weights,
            device=decoded_unit.device,
            dtype=decoded_unit.dtype,
        ).reshape(-1)
        if channel_weights_t.numel() != channels:
            raise ValueError(
                f"texture channel_weights has {channel_weights_t.numel()} values for {channels} masks"
            )
        if bool((channel_weights_t < 0).any()):
            raise ValueError("texture channel_weights must be non-negative")
    if float(channel_weights_t.sum().detach().cpu()) <= 0:
        raise ValueError("texture channel_weights must contain positive mass")
    channel_weights_t = channel_weights_t / channel_weights_t.sum()

    if patch_target_unit is None:
        patch_target_tensor = target_unit.detach()
        patch_target_core_masks = core_masks
    else:
        patch_target_tensor = patch_target_unit.detach().to(
            device=decoded_unit.device, dtype=decoded_unit.dtype
        )
        if patch_target_tensor.ndim != 5 or patch_target_tensor.shape[1] != 1:
            raise ValueError("texture patch target must be [batch, 1, depth, height, width]")
        if patch_target_masks is None:
            raise ValueError("texture patch target requires its own masks")
        patch_target_masks = patch_target_masks.to(
            device=decoded_unit.device, dtype=decoded_unit.dtype
        )
        if patch_target_masks.ndim != 5:
            raise ValueError("texture patch target masks must be 5D")
        if tuple(patch_target_masks.shape[-3:]) != tuple(patch_target_tensor.shape[-3:]):
            patch_target_masks = F.interpolate(
                patch_target_masks,
                size=patch_target_tensor.shape[-3:],
                mode="nearest",
            )
        if patch_target_tensor.shape[0] == 1 and decoded_unit.shape[0] > 1:
            patch_target_tensor = patch_target_tensor.expand(
                decoded_unit.shape[0], -1, -1, -1, -1
            )
        if patch_target_masks.shape[0] == 1 and decoded_unit.shape[0] > 1:
            patch_target_masks = patch_target_masks.expand(
                decoded_unit.shape[0], -1, -1, -1, -1
            )
        if patch_target_tensor.shape[0] != decoded_unit.shape[0]:
            raise ValueError("texture patch target and decoded CT batch sizes differ")
        if patch_target_masks.shape[:2] != (decoded_unit.shape[0], channels):
            raise ValueError(
                "texture patch target masks must match decoded batch and mask channels"
            )
        patch_target_hard_masks = (patch_target_masks > 0.5).to(decoded_unit.dtype)
        if erosion:
            kernel = 2 * erosion + 1
            patch_target_eroded = 1.0 - F.max_pool3d(
                1.0 - patch_target_hard_masks,
                kernel_size=kernel,
                stride=1,
                padding=erosion,
            )
            patch_target_enough = patch_target_eroded.sum(
                dim=(2, 3, 4), keepdim=True
            ) >= float(min_voxels)
            patch_target_core_masks = torch.where(
                patch_target_enough,
                patch_target_eroded,
                patch_target_hard_masks,
            )
        else:
            patch_target_core_masks = patch_target_hard_masks

    def blur(value, radius):
        padding = (radius, radius, radius, radius, radius, radius)
        padded = F.pad(value, padding, mode="replicate")
        return F.avg_pool3d(padded, kernel_size=2 * radius + 1, stride=1)

    decoded_bands = []
    target_bands = []
    decoded_low = decoded_unit
    target_low = target_unit.detach()
    for radius in scales:
        next_decoded = blur(decoded_unit, radius)
        next_target = blur(target_unit.detach(), radius)
        decoded_bands.append(decoded_low - next_decoded)
        target_bands.append(target_low - next_target)
        decoded_low = next_decoded
        target_low = next_target

    def masked_center(value, mask):
        expanded = value.expand(-1, channels, -1, -1, -1)
        mass = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
        mean = (expanded * mask).sum(dim=(2, 3, 4)) / mass
        centered = (expanded - mean[:, :, None, None, None]) * mask
        variance = centered.square().sum(dim=(2, 3, 4)) / mass
        return centered, variance, mass

    def autocorrelation(centered, mask, variance, lag, axis):
        if lag >= centered.shape[axis]:
            return torch.zeros_like(variance)
        left = [slice(None)] * 5
        right = [slice(None)] * 5
        left[axis] = slice(lag, None)
        right[axis] = slice(None, -lag)
        pair_mask = mask[tuple(left)] * mask[tuple(right)]
        pair_mass = pair_mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
        covariance = (
            centered[tuple(left)] * centered[tuple(right)] * pair_mask
        ).sum(dim=(2, 3, 4)) / pair_mass
        return covariance / variance.clamp_min(1.0e-5)

    def masked_moments(value, mask):
        mass = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
        mean = (value * mask).sum(dim=(2, 3, 4)) / mass
        variance = (
            (value - mean[:, :, None, None, None]).square() * mask
        ).sum(dim=(2, 3, 4)) / mass
        return mean, variance

    total = decoded_unit.sum() * 0.0
    weight_mass = 0.0
    for decoded_band, target_band, band_weight in zip(decoded_bands, target_bands, weights):
        if band_weight <= 0:
            continue
        decoded_centered, decoded_var, _ = masked_center(decoded_band, core_masks)
        target_centered, target_var, _ = masked_center(target_band, core_masks)
        amplitude = torch.abs(torch.sqrt(decoded_var + 1.0e-6) - torch.sqrt(target_var + 1.0e-6))
        amplitude = amplitude / (torch.sqrt(target_var + 1.0e-6) + 0.01)
        channel_loss = amplitude
        corr_count = 0
        for lag in lags:
            for axis in (2, 3, 4):
                decoded_corr = autocorrelation(decoded_centered, core_masks, decoded_var, lag, axis)
                target_corr = autocorrelation(target_centered, core_masks, target_var, lag, axis)
                channel_loss = channel_loss + torch.abs(decoded_corr - target_corr)
                corr_count += 1
        channel_loss = channel_loss / float(1 + corr_count)
        if local_variance_weight > 0:
            decoded_local_rms = torch.sqrt(
                F.avg_pool3d(decoded_centered.square(), kernel_size=3, stride=1, padding=1)
                + 1.0e-6
            )
            target_local_rms = torch.sqrt(
                F.avg_pool3d(target_centered.square(), kernel_size=3, stride=1, padding=1)
                + 1.0e-6
            )
            decoded_local_mean, decoded_local_var = masked_moments(
                decoded_local_rms, core_masks
            )
            target_local_mean, target_local_var = masked_moments(
                target_local_rms, core_masks
            )
            local_mean_gap = torch.abs(decoded_local_mean - target_local_mean) / (
                target_local_mean.abs() + 0.01
            )
            local_std_gap = torch.abs(
                torch.sqrt(decoded_local_var + 1.0e-6)
                - torch.sqrt(target_local_var + 1.0e-6)
            ) / (torch.sqrt(target_local_var + 1.0e-6) + 0.01)
            channel_loss = channel_loss + local_variance_weight * 0.5 * (
                local_mean_gap + local_std_gap
            )
        total = total + float(band_weight) * (
            channel_loss.mean(dim=0) * channel_weights_t
        ).sum()
        weight_mass += float(band_weight)
    total = total / max(weight_mass, 1.0e-8)

    if gram_weight > 0:
        def forward_difference(value, axis):
            first = value.narrow(axis, 0, 1)
            return torch.diff(value, dim=axis, prepend=first)

        decoded_features = decoded_bands + [
            forward_difference(decoded_unit, axis) for axis in (2, 3, 4)
        ]
        target_features = target_bands + [
            forward_difference(target_unit.detach(), axis) for axis in (2, 3, 4)
        ]
        decoded_features = torch.cat(decoded_features, dim=1)
        target_features = torch.cat(target_features, dim=1)
        gram_loss = decoded_unit.sum() * 0.0
        for channel in range(channels):
            mask = core_masks[:, channel : channel + 1]
            mass = mask.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)

            def normalized_features(features):
                mean = (features * mask).sum(dim=(2, 3, 4), keepdim=True) / mass
                centered = (features - mean) * mask
                variance = centered.square().sum(
                    dim=(2, 3, 4), keepdim=True
                ) / mass
                return centered / torch.sqrt(variance + 1.0e-5)

            decoded_normalized = normalized_features(decoded_features).flatten(2)
            target_normalized = normalized_features(target_features).flatten(2)
            voxel_mass = mass.flatten(1).clamp_min(1.0)
            decoded_gram = torch.bmm(
                decoded_normalized, decoded_normalized.transpose(1, 2)
            ) / voxel_mass[:, :, None]
            target_gram = torch.bmm(
                target_normalized, target_normalized.transpose(1, 2)
            ) / voxel_mass[:, :, None]
            gram_loss = gram_loss + channel_weights_t[channel] * torch.abs(
                decoded_gram - target_gram
            ).mean()
        total = total + gram_weight * gram_loss

    if patch_swd_weight > 0:
        patch_loss = decoded_unit.sum() * 0.0
        patch_weight_mass = 0.0
        for patch_size in patch_swd_sizes:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(patch_swd_seed + 104729 * patch_size)
            filters = torch.randn(
                patch_swd_projections,
                1,
                patch_size,
                patch_size,
                patch_size,
                generator=generator,
                dtype=torch.float32,
            )
            filters = filters - filters.mean(dim=(2, 3, 4), keepdim=True)
            filters = filters / torch.sqrt(
                filters.square().sum(dim=(2, 3, 4), keepdim=True) + 1.0e-8
            )
            filters = filters.to(device=decoded_unit.device, dtype=decoded_unit.dtype)
            padding = patch_size // 2
            decoded_responses = F.conv3d(decoded_unit, filters, padding=padding)
            target_responses = F.conv3d(
                patch_target_tensor, filters, padding=padding
            )
            for channel in range(channels):
                mask = core_masks[:, channel : channel + 1]
                target_mask = patch_target_core_masks[:, channel : channel + 1]
                support = F.avg_pool3d(
                    mask,
                    kernel_size=patch_size,
                    stride=1,
                    padding=padding,
                )
                target_support = F.avg_pool3d(
                    target_mask,
                    kernel_size=patch_size,
                    stride=1,
                    padding=padding,
                )
                for batch in range(decoded_unit.shape[0]):
                    valid = support[batch, 0] >= patch_swd_min_support
                    if int(valid.sum().detach().cpu()) < int(min_voxels):
                        valid = mask[batch, 0] > 0.5
                    target_valid = target_support[batch, 0] >= patch_swd_min_support
                    if int(target_valid.sum().detach().cpu()) < int(min_voxels):
                        target_valid = target_mask[batch, 0] > 0.5
                    decoded_count = int(valid.sum().detach().cpu())
                    target_count = int(target_valid.sum().detach().cpu())
                    if decoded_count < 2 or target_count < 2:
                        continue
                    decoded_values = decoded_responses[batch, :, valid]
                    target_values = target_responses[batch, :, target_valid]
                    if decoded_count > patch_swd_max_samples:
                        stride = int(
                            math.ceil(decoded_count / float(patch_swd_max_samples))
                        )
                        decoded_values = decoded_values[:, ::stride]
                    if target_count > patch_swd_max_samples:
                        stride = int(
                            math.ceil(target_count / float(patch_swd_max_samples))
                        )
                        target_values = target_values[:, ::stride]
                    target_mean = target_values.mean(dim=1, keepdim=True)
                    target_scale = target_values.std(dim=1, keepdim=True).clamp_min(0.01)
                    decoded_values = (decoded_values - target_mean) / target_scale
                    target_values = (target_values - target_mean) / target_scale
                    decoded_sorted = torch.sort(decoded_values, dim=1).values
                    target_sorted = torch.sort(target_values, dim=1).values

                    def fixed_quantiles(sorted_values):
                        positions = torch.linspace(
                            0.0,
                            float(sorted_values.shape[1] - 1),
                            steps=patch_swd_quantiles,
                            device=sorted_values.device,
                            dtype=torch.float32,
                        )
                        lower = positions.floor().long()
                        upper = positions.ceil().long()
                        fraction = (positions - lower.to(positions.dtype))[None]
                        return (
                            sorted_values[:, lower] * (1.0 - fraction)
                            + sorted_values[:, upper] * fraction
                        )

                    decoded_sorted = fixed_quantiles(decoded_sorted)
                    target_sorted = fixed_quantiles(target_sorted)
                    patch_loss = patch_loss + channel_weights_t[channel] * F.smooth_l1_loss(
                        decoded_sorted,
                        target_sorted,
                    )
                    patch_weight_mass += float(channel_weights_t[channel].detach().cpu())
        if patch_weight_mass > 0:
            total = total + patch_swd_weight * patch_loss / patch_weight_mass
    return total


def _late_texture_schedule_scale(config, prefix, progress):
    start = float(config.get(f"{prefix}_latent_texture_start_fraction", 0.0))
    power = float(config.get(f"{prefix}_latent_texture_ramp_power", 1.0))
    if not 0.0 <= start < 1.0:
        raise ValueError(f"{prefix}_latent_texture_start_fraction must be in [0, 1), got {start}")
    if power <= 0:
        raise ValueError(f"{prefix}_latent_texture_ramp_power must be positive, got {power}")
    progress = min(max(float(progress), 0.0), 1.0)
    if progress <= start:
        return 0.0
    return float(((progress - start) / (1.0 - start)) ** power)


def _refine_image_spatial_texture(
    image,
    texture_target,
    texture_masks,
    config,
    channel_weights=None,
    patch_texture_target=None,
    patch_texture_masks=None,
):
    """Optimize full-resolution texture guidance, then preserve HU values by core rank projection."""
    import torch
    import torch.nn.functional as F

    steps = int(config.get("image_texture_refinement_steps", 0))
    if steps <= 0:
        return np.asarray(image, dtype=np.float32), {
            "status": "disabled",
            "steps": 0,
        }
    if texture_target is None:
        raise ValueError("image_texture_refinement_steps requires verified train texture guidance")
    image_np = np.asarray(image, dtype=np.float32)
    target_np = np.asarray(texture_target, dtype=np.float32)
    masks_np = np.asarray(texture_masks, dtype=np.float32)
    if masks_np.ndim == image_np.ndim:
        masks_np = masks_np[None]
    if target_np.shape != image_np.shape or tuple(masks_np.shape[-3:]) != image_np.shape:
        raise ValueError(
            "image texture target and masks must match the generated image shape: "
            f"image={image_np.shape}, target={target_np.shape}, masks={masks_np.shape}"
        )
    requested_device = str(config.get("image_texture_refinement_device", "cuda"))
    device = torch.device(
        requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    x0 = torch.as_tensor(
        (image_np - HU_MIN) / (HU_MAX - HU_MIN), dtype=torch.float32, device=device
    )[None, None].clamp(0.0, 1.0)
    target = torch.as_tensor(
        (target_np - HU_MIN) / (HU_MAX - HU_MIN), dtype=torch.float32, device=device
    )[None, None].clamp(0.0, 1.0)
    masks = torch.as_tensor(masks_np, dtype=torch.float32, device=device)[None]
    patch_target = None
    patch_masks = None
    if patch_texture_target is not None:
        patch_target_np = np.asarray(patch_texture_target, dtype=np.float32)
        patch_masks_np = np.asarray(patch_texture_masks, dtype=np.float32)
        if patch_masks_np.ndim == patch_target_np.ndim:
            patch_masks_np = patch_masks_np[None]
        if tuple(patch_masks_np.shape[-3:]) != patch_target_np.shape:
            raise ValueError("raw patch texture target and masks must have matching shapes")
        patch_target = torch.as_tensor(
            (patch_target_np - HU_MIN) / (HU_MAX - HU_MIN),
            dtype=torch.float32,
            device=device,
        )[None, None].clamp(0.0, 1.0)
        patch_masks = torch.as_tensor(
            patch_masks_np, dtype=torch.float32, device=device
        )[None]
    lesion = (masks[:, :1] > 0.5).float()
    erosion = max(0, int(config.get("image_texture_core_erosion", 1)))
    if erosion:
        kernel = 2 * erosion + 1
        core = 1.0 - F.max_pool3d(
            1.0 - lesion,
            kernel_size=kernel,
            stride=1,
            padding=erosion,
        )
        if int(core.sum().detach().cpu()) < int(config.get("image_texture_min_voxels", 8)):
            core = lesion
    else:
        core = lesion
    core_voxels = int(core.sum().detach().cpu())
    variable = x0.clone().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [variable], lr=float(config.get("image_texture_refinement_lr", 0.03))
    )
    texture_weight = float(config.get("image_texture_lambda", 1.0))
    anchor_weight = float(config.get("image_texture_anchor_lambda", 0.05))
    rank_blend = float(config.get("image_texture_rank_blend", 1.0))
    optimized_value_blend = float(
        config.get("image_texture_optimized_value_blend", 0.0)
    )
    if texture_weight < 0 or anchor_weight < 0:
        raise ValueError("image texture and anchor weights must be non-negative")
    if not 0.0 <= rank_blend <= 1.0:
        raise ValueError(
            f"image_texture_rank_blend must be in [0, 1], got {rank_blend}"
        )
    if not 0.0 <= optimized_value_blend <= 1.0:
        raise ValueError(
            "image_texture_optimized_value_blend must be in [0, 1], got "
            f"{optimized_value_blend}"
        )

    def terms(current):
        guided = x0 + (current.clamp(0.0, 1.0) - x0) * core
        texture = _multiscale_3d_texture_energy(
            guided,
            target,
            masks,
            scales=config.get("image_texture_scales", [1, 2, 4]),
            band_weights=config.get("image_texture_band_weights", [0.55, 0.30, 0.15]),
            autocorr_lags=config.get("image_texture_autocorr_lags", [1, 2]),
            core_erosion=int(config.get("image_texture_energy_core_erosion", 1)),
            min_voxels=int(config.get("image_texture_min_voxels", 8)),
            channel_weights=channel_weights,
            local_variance_weight=float(
                config.get("image_texture_local_variance_weight", 0.0)
            ),
            gram_weight=float(config.get("image_texture_gram_weight", 0.0)),
            patch_swd_weight=float(
                config.get("image_texture_patch_swd_weight", 0.0)
            ),
            patch_swd_sizes=config.get("image_texture_patch_swd_sizes", [3]),
            patch_swd_projections=int(
                config.get("image_texture_patch_swd_projections", 12)
            ),
            patch_swd_min_support=float(
                config.get("image_texture_patch_swd_min_support", 0.25)
            ),
            patch_swd_max_samples=int(
                config.get("image_texture_patch_swd_max_samples", 2048)
            ),
            patch_swd_seed=int(config.get("image_texture_patch_swd_seed", 1729)),
            patch_swd_quantiles=int(
                config.get("image_texture_patch_swd_quantiles", 257)
            ),
            patch_target_unit=patch_target,
            patch_target_masks=patch_masks,
        )
        anchor = (((guided - x0) ** 2) * core).sum() / (core.sum() + 1.0)
        return texture_weight * texture + anchor_weight * anchor, texture, anchor, guided

    with torch.no_grad():
        initial_loss, initial_texture, initial_anchor, _ = terms(variable)
    trace = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, texture, anchor, _ = terms(variable)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [variable], float(config.get("image_texture_grad_clip", 1.0))
        )
        optimizer.step()
        trace.append(
            {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "texture": float(texture.detach().cpu()),
                "anchor": float(anchor.detach().cpu()),
            }
        )
    with torch.no_grad():
        final_loss, final_texture, final_anchor, guided = terms(variable)
    guidance = guided[0, 0].detach().cpu().numpy()
    blended_guidance = (
        (1.0 - rank_blend) * x0[0, 0].detach().cpu().numpy()
        + rank_blend * guidance
    )
    core_np = core[0, 0].detach().cpu().numpy() > 0.5
    output = image_np.copy()
    original_values = output[core_np].copy()
    ranked = original_values.copy()
    ranked[np.argsort(blended_guidance[core_np], kind="stable")] = np.sort(
        original_values
    )
    optimized_values = np.asarray(
        (
            guided[0, 0][core[0, 0] > 0.5]
            .detach()
            .cpu()
            .reshape(-1)
            .tolist()
        ),
        dtype=np.float32,
    )
    optimized_values = optimized_values * (HU_MAX - HU_MIN) + HU_MIN
    output[core_np] = np.clip(
        (1.0 - optimized_value_blend) * ranked
        + optimized_value_blend * optimized_values,
        HU_MIN,
        HU_MAX,
    )
    return output, {
        "status": "pass",
        "steps": int(steps),
        "device": str(device),
        "core_voxels": core_voxels,
        "texture_weight": float(texture_weight),
        "anchor_weight": float(anchor_weight),
        "rank_blend": float(rank_blend),
        "optimized_value_blend": float(optimized_value_blend),
        "optimized_value_delta_rms_hu": float(
            np.sqrt(
                np.mean(
                    np.square(optimized_values - original_values, dtype=np.float64)
                )
            )
        )
        if original_values.size
        else 0.0,
        "local_variance_weight": float(
            config.get("image_texture_local_variance_weight", 0.0)
        ),
        "gram_weight": float(config.get("image_texture_gram_weight", 0.0)),
        "patch_swd_weight": float(
            config.get("image_texture_patch_swd_weight", 0.0)
        ),
        "patch_swd_sizes": [
            int(value) for value in config.get("image_texture_patch_swd_sizes", [3])
        ],
        "patch_swd_projections": int(
            config.get("image_texture_patch_swd_projections", 12)
        ),
        "patch_swd_quantiles": int(
            config.get("image_texture_patch_swd_quantiles", 257)
        ),
        "patch_target_mode": (
            "raw_train_donor" if patch_target is not None else "transported_target"
        ),
        "initial_loss": float(initial_loss.detach().cpu()),
        "final_loss": float(final_loss.detach().cpu()),
        "initial_texture": float(initial_texture.detach().cpu()),
        "final_texture": float(final_texture.detach().cpu()),
        "initial_anchor": float(initial_anchor.detach().cpu()),
        "final_anchor": float(final_anchor.detach().cpu()),
        "trace": trace,
        "histogram_preservation": (
            "exact_core_value_multiset_rank_projection"
            if optimized_value_blend == 0.0
            else "soft_rank_projection_plus_optimized_texture_values"
        ),
    }


def _dilate(mask, iterations):
    mask = np.asarray(mask) > 0
    if iterations <= 0:
        return mask
    try:
        from scipy.ndimage import binary_dilation

        return binary_dilation(mask, iterations=int(iterations))
    except Exception:
        out = mask.copy()
        for _ in range(int(iterations)):
            grown = out.copy()
            for axis in range(3):
                grown |= np.roll(out, 1, axis=axis)
                grown |= np.roll(out, -1, axis=axis)
            out = grown
        return out


def _inner_boundary(mask, iterations=1):
    mask = np.asarray(mask) > 0
    if iterations <= 0 or not mask.any():
        return np.zeros_like(mask, dtype=bool)
    try:
        from scipy.ndimage import binary_erosion

        eroded = binary_erosion(mask, iterations=int(iterations))
        return mask & ~eroded
    except Exception:
        return boundary_band(mask) & mask


def _smooth_hist_preserving(image, mask, sigma):
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask) > 0
    if sigma <= 0 or not mask.any():
        return image
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return image
    smoothed = gaussian_filter(image, sigma=float(sigma)).astype(np.float32)
    out = image.copy()
    original = image[mask]
    smooth_values = smoothed[mask]
    order = np.argsort(smooth_values)
    remapped = smooth_values.copy()
    remapped[order] = np.sort(original)
    out[mask] = remapped
    return out.astype(np.float32)


def _generation_sample_seed(config, sample_index):
    return int(config.get("seed", 42)) + int(sample_index)


def _apply_inner_boundary_reference_blend(image, reference, mask, blend, width=1):
    try:
        blend = float(blend)
    except Exception:
        blend = 0.0
    if blend <= 0:
        return np.asarray(image, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32).copy()
    reference = np.asarray(reference, dtype=np.float32)
    mask = np.asarray(mask) > 0
    if image.shape != reference.shape or not mask.any():
        return image
    band = _inner_boundary(mask, int(width))
    if not band.any():
        return image
    blend = min(max(blend, 0.0), 1.0)
    image[band] = np.clip((1.0 - blend) * image[band] + blend * reference[band], HU_MIN, HU_MAX)
    return image


def _apply_inner_distance_reference_blend(
    image, reference, mask, blend, width=1, power=1.0
):
    """Blend toward source anatomy inside the lesion with a distance-decaying ramp."""
    try:
        blend = float(blend)
        width = int(width)
        power = float(power)
    except Exception:
        return np.asarray(image, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32).copy()
    reference = np.asarray(reference, dtype=np.float32)
    mask = np.asarray(mask) > 0
    if (
        blend <= 0
        or width <= 0
        or image.shape != reference.shape
        or not mask.any()
    ):
        return image

    blend = min(max(blend, 0.0), 1.0)
    power = max(power, 1.0e-6)
    try:
        from scipy.ndimage import distance_transform_edt

        distance = distance_transform_edt(mask).astype(np.float32)
        ramp = np.clip((float(width) + 1.0 - distance) / float(width), 0.0, 1.0)
        ramp = np.power(ramp, power, dtype=np.float32) * mask
    except Exception:
        ramp = np.zeros(mask.shape, dtype=np.float32)
        remaining = mask.copy()
        for layer in range(1, width + 1):
            band = _inner_boundary(remaining, 1)
            layer_weight = ((float(width) + 1.0 - float(layer)) / float(width)) ** power
            ramp[band] = layer_weight
            remaining[band] = False
            if not remaining.any():
                break

    alpha = np.clip(blend * ramp, 0.0, 1.0)
    image[mask] = np.clip(
        (1.0 - alpha[mask]) * image[mask] + alpha[mask] * reference[mask],
        HU_MIN,
        HU_MAX,
    )
    return image


def _boundary_reference_blend_for_condition(config, equivalent_diameter_mm, subtype):
    blend = float(config.get("boundary_reference_blend", 0.0))
    size_key = (
        "small"
        if float(equivalent_diameter_mm) < 10.0
        else "medium"
        if float(equivalent_diameter_mm) < 20.0
        else "large"
    )
    by_size = config.get("boundary_reference_blend_by_size", {})
    if by_size:
        blend = float(by_size.get(size_key, by_size.get("default", blend)))
    subtype_key = str(subtype or "default").lower().replace("-", "_")
    multipliers = config.get("boundary_reference_blend_subtype_multiplier", {})
    multiplier = float(multipliers.get(subtype_key, multipliers.get("default", 1.0)))
    return min(max(blend * multiplier, 0.0), 1.0), size_key, multiplier


def _apportion_histogram_counts(histogram, total):
    histogram = np.maximum(np.asarray(histogram, dtype=np.float64).reshape(-1), 0.0)
    total = max(int(total), 0)
    if total == 0:
        return np.zeros(histogram.size, dtype=np.int64)
    if histogram.size == 0 or float(histogram.sum()) <= 0:
        raise ValueError("histogram compensation requires positive target mass")
    probabilities = histogram / float(histogram.sum())
    raw = probabilities * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def _compensate_inner_core_histogram(
    image,
    mask,
    target_hist,
    boundary_width,
    strength=1.0,
    hu_range=(HU_MIN, HU_MAX),
    rank_guidance=None,
):
    """Restore target HU-bin mass in the lesion core without changing its boundary ring."""
    image = np.asarray(image, dtype=np.float32).copy()
    mask = np.asarray(mask) > 0
    target_hist = np.asarray(target_hist, dtype=np.float64).reshape(-1)
    boundary_width = max(int(boundary_width), 0)
    strength = min(max(float(strength), 0.0), 1.0)
    if not mask.any() or target_hist.size == 0 or boundary_width <= 0 or strength <= 0:
        return image
    boundary = _inner_boundary(mask, boundary_width)
    core = mask & ~boundary
    if not core.any():
        return image

    edges = np.linspace(
        float(hu_range[0]), float(hu_range[1]), target_hist.size + 1, dtype=np.float32
    )
    desired_counts = _apportion_histogram_counts(target_hist, int(mask.sum()))
    boundary_counts, _ = np.histogram(image[boundary], bins=edges)
    residual = np.maximum(desired_counts - boundary_counts.astype(np.int64), 0)
    core_counts = _apportion_histogram_counts(
        residual if int(residual.sum()) > 0 else target_hist,
        int(core.sum()),
    )

    target_values = []
    for index, count in enumerate(core_counts):
        count = int(count)
        if count <= 0:
            continue
        fractions = (np.arange(count, dtype=np.float32) + 0.5) / float(count)
        target_values.append(edges[index] + fractions * (edges[index + 1] - edges[index]))
    corrected = np.concatenate(target_values).astype(np.float32)
    current = image[core]
    if corrected.size != current.size:
        raise RuntimeError(
            f"core histogram compensation size mismatch: {corrected.size} != {current.size}"
        )
    ordering = current
    if rank_guidance is not None:
        rank_guidance = np.asarray(rank_guidance, dtype=np.float32)
        if rank_guidance.shape != image.shape:
            raise ValueError(
                "core histogram rank guidance shape mismatch: "
                f"{rank_guidance.shape} != {image.shape}"
            )
        ordering = rank_guidance[core]
    ranked = np.empty_like(current)
    ranked[np.argsort(ordering, kind="stable")] = np.sort(corrected)
    image[core] = np.clip(
        (1.0 - strength) * current + strength * ranked,
        HU_MIN,
        HU_MAX,
    )
    return image


def _histogram_bin_guided_projection(
    image, guidance, mask, bins=40, hu_range=(-1000.0, 400.0)
):
    """Project to guidance while preserving fixed-bin lesion HU counts."""
    image = np.asarray(image, dtype=np.float32)
    guidance = np.asarray(guidance, dtype=np.float32)
    mask = np.asarray(mask) > 0
    out = image.copy()
    if image.shape != guidance.shape or not mask.any():
        return out
    values = image[mask]
    guided = guidance[mask]
    edges = np.linspace(hu_range[0], hu_range[1], int(bins) + 1, dtype=np.float32)
    groups = np.searchsorted(edges[1:-1], values, side="right").astype(np.int32)
    groups[values < edges[0]] = -1
    groups[values > edges[-1]] = int(bins)
    order = np.argsort(guided, kind="stable")
    assigned = np.empty_like(groups)
    assigned[order] = np.sort(groups)
    projected = guided.copy()
    for group in range(int(bins)):
        selected = assigned == group
        if not selected.any():
            continue
        upper = edges[group + 1]
        if group < int(bins) - 1:
            upper = np.nextafter(upper, edges[group], dtype=np.float32)
        projected[selected] = np.clip(projected[selected], edges[group], upper)
    underflow = assigned < 0
    if underflow.any():
        projected[underflow] = np.minimum(
            projected[underflow],
            np.nextafter(edges[0], np.float32(-np.inf), dtype=np.float32),
        )
    overflow = assigned >= int(bins)
    if overflow.any():
        projected[overflow] = np.maximum(
            projected[overflow],
            np.nextafter(edges[-1], np.float32(np.inf), dtype=np.float32),
        )
    out[mask] = projected
    return out


def _apply_histogram_bin_boundary_projection(
    image,
    reference,
    mask,
    blend,
    width=2,
    bins=40,
    hu_range=(-1000.0, 400.0),
):
    guidance = _apply_inner_boundary_reference_blend(
        image, reference, mask, blend, width=width
    )
    return _histogram_bin_guided_projection(
        image, guidance, mask, bins=bins, hu_range=hu_range
    )


def _apply_local_contrast_match(image, mask, target_contrast, strength=1.0, shell_width=2, max_shift=120.0):
    try:
        target_contrast = float(target_contrast)
        strength = float(strength)
    except Exception:
        return np.asarray(image, dtype=np.float32)
    if target_contrast <= 0 or strength <= 0:
        return np.asarray(image, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32).copy()
    mask = np.asarray(mask) > 0
    if not mask.any():
        return image
    shell = _dilate(mask, int(shell_width)) & ~mask
    if not shell.any():
        return image
    inside_mean = float(np.mean(image[mask]))
    shell_mean = float(np.mean(image[shell]))
    desired_inside_mean = shell_mean + target_contrast
    shift = (desired_inside_mean - inside_mean) * min(max(strength, 0.0), 1.0)
    max_shift = abs(float(max_shift))
    if max_shift > 0:
        shift = min(max(shift, -max_shift), max_shift)
    image[mask] = np.clip(image[mask] + shift, HU_MIN, HU_MAX)
    return image


def _apply_perilesion_shell_match(image, mask, target_mean, strength=1.0, shell_width=1, max_shift=120.0):
    try:
        target_mean = float(target_mean)
        strength = float(strength)
    except Exception:
        return np.asarray(image, dtype=np.float32)
    if strength <= 0:
        return np.asarray(image, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32).copy()
    mask = np.asarray(mask) > 0
    if not mask.any():
        return image
    shell = _dilate(mask, int(shell_width)) & ~mask
    if not shell.any():
        return image
    shell_mean = float(np.mean(image[shell]))
    shift = (target_mean - shell_mean) * min(max(strength, 0.0), 1.0)
    max_shift = abs(float(max_shift))
    if max_shift > 0:
        shift = min(max(shift, -max_shift), max_shift)
    image[shell] = np.clip(image[shell] + shift, HU_MIN, HU_MAX)
    return image


def _resize_array(array, shape, order=1):
    array = np.asarray(array)
    shape = tuple(int(v) for v in shape)
    if tuple(array.shape) == shape:
        return array.astype(array.dtype, copy=False)
    try:
        from scipy.ndimage import zoom

        factors = [n / o for n, o in zip(shape, array.shape)]
        return zoom(array, factors, order=order)
    except Exception:
        coords = [np.linspace(0, old - 1, new).round().astype(np.int64) for old, new in zip(array.shape, shape)]
        return array[np.ix_(coords[0], coords[1], coords[2])]


def _sample_histogram(hist, count, rng, hu_range=(HU_MIN, HU_MAX)):
    hist = np.asarray(hist, dtype=np.float64)
    if hist.size == 0:
        return rng.normal(-350.0, 120.0, size=count).astype(np.float32)
    hist = np.maximum(hist, 0.0)
    hist = hist / (hist.sum() + 1.0e-12)
    edges = np.linspace(float(hu_range[0]), float(hu_range[1]), hist.size + 1)
    bins = rng.choice(hist.size, size=count, p=hist)
    values = rng.uniform(edges[bins], edges[bins + 1]).astype(np.float32)
    return np.clip(values, hu_range[0], hu_range[1]).astype(np.float32)


def _apply_final_target_histogram_reprojection(
    image,
    mask,
    target_hist,
    strength=1.0,
    hu_range=(HU_MIN, HU_MAX),
):
    """Match target bin counts after texture refinement while preserving spatial ranks."""
    image = np.asarray(image, dtype=np.float32).copy()
    mask = np.asarray(mask) > 0
    strength = max(0.0, min(float(strength), 1.0))
    count = int(mask.sum())
    hist = np.maximum(np.asarray(target_hist, dtype=np.float64).reshape(-1), 0.0)
    if strength <= 0 or count == 0 or hist.size == 0 or float(hist.sum()) <= 0:
        return image

    probabilities = hist / hist.sum()
    expected = probabilities * count
    bin_counts = np.floor(expected).astype(np.int64)
    remaining = count - int(bin_counts.sum())
    if remaining > 0:
        fractional = expected - bin_counts
        order = np.argsort(-fractional, kind="stable")
        bin_counts[order[:remaining]] += 1

    edges = np.linspace(float(hu_range[0]), float(hu_range[1]), hist.size + 1)
    target_values = []
    for index, bin_count in enumerate(bin_counts):
        if bin_count <= 0:
            continue
        fractions = (np.arange(bin_count, dtype=np.float64) + 0.5) / bin_count
        target_values.append(edges[index] + fractions * (edges[index + 1] - edges[index]))
    target_values = np.concatenate(target_values).astype(np.float32)

    current = image[mask].copy()
    corrected = current.copy()
    corrected[np.argsort(current, kind="stable")] = target_values
    image[mask] = (1.0 - strength) * current + strength * corrected
    return np.clip(image, hu_range[0], hu_range[1]).astype(np.float32)


def _hist_quantile(hist, q, hu_range=(HU_MIN, HU_MAX)):
    hist = np.asarray(hist, dtype=np.float64)
    if hist.size == 0 or hist.sum() <= 0:
        return float(hu_range[0])
    hist = np.maximum(hist, 0.0)
    cdf = np.cumsum(hist) / (hist.sum() + 1.0e-12)
    edges = np.linspace(float(hu_range[0]), float(hu_range[1]), hist.size + 1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    idx = int(np.searchsorted(cdf, float(q), side="left"))
    idx = max(0, min(idx, centers.size - 1))
    return float(centers[idx])


def _unit_from_hu_torch(values, a_min=-1000.0, a_max=1000.0):
    return ((values - float(a_min)) / (float(a_max) - float(a_min))).clamp(0.0, 1.0)


def _hist_moments(hist, hu_range=(HU_MIN, HU_MAX)):
    hist = np.asarray(hist, dtype=np.float64)
    if hist.size == 0 or hist.sum() <= 0:
        return 0.0, 0.0
    hist = np.maximum(hist, 0.0)
    prob = hist / (hist.sum() + 1.0e-12)
    edges = np.linspace(float(hu_range[0]), float(hu_range[1]), hist.size + 1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    mean = float((prob * centers).sum())
    std = float(np.sqrt((prob * (centers - mean) ** 2).sum()))
    return mean, std


def _target_tail_score(target_root, meta_path):
    meta = load_json(meta_path)
    hist = np.load(Path(target_root) / meta["hist_path"]).astype(np.float32)
    subtype = str(meta.get("subtype", ""))
    subtype_bonus = 0.0
    if subtype in {"solid", "part_solid"}:
        subtype_bonus = 25.0 if subtype == "part_solid" else 50.0
    return _hist_quantile(hist, 0.75) + 0.5 * _hist_quantile(hist, 0.95) + subtype_bonus


def _target_feature_vector(target_root, meta_path):
    meta = load_json(meta_path)
    root = Path(target_root)
    hist = np.load(root / meta["hist_path"]).astype(np.float32)
    mean, std = _hist_moments(hist)
    quantiles = [_hist_quantile(hist, q) for q in (0.05, 0.25, 0.50, 0.75, 0.95)]
    try:
        mask = load_volume(root / meta["mask_path"]) > 0
        volume_fraction = float(mask.mean())
    except Exception:
        volume_fraction = 0.0
    try:
        diameter = float(meta.get("diameter_mm") or 0.0)
    except Exception:
        diameter = 0.0
    subtype = str(meta.get("subtype", "unknown")).lower()
    subtype_one_hot = [
        1.0 if subtype in {"ground_glass", "ggn", "groundglass"} else 0.0,
        1.0 if subtype in {"part_solid", "partsolid"} else 0.0,
        1.0 if subtype == "solid" else 0.0,
    ]
    return np.asarray(
        [mean, std, *quantiles, volume_fraction * 20000.0, diameter * 25.0, *subtype_one_hot],
        dtype=np.float64,
    )


def _farthest_first_order(target_metas, target_root, seed, start="tail"):
    if not target_metas:
        return []
    rng = np.random.default_rng(int(seed))
    features = np.vstack([_target_feature_vector(target_root, p) for p in target_metas])
    center = np.nanmedian(features, axis=0)
    scale = np.nanpercentile(features, 75, axis=0) - np.nanpercentile(features, 25, axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    norm = (features - center) / scale
    if start == "tail":
        first = int(np.argmax([_target_tail_score(target_root, p) for p in target_metas]))
    elif start == "random":
        first = int(rng.integers(0, len(target_metas)))
    else:
        first = int(np.argmin(np.linalg.norm(norm, axis=1)))
    chosen = [first]
    remaining = set(range(len(target_metas))) - {first}
    min_dist = np.sqrt(((norm - norm[first]) ** 2).sum(axis=1))
    while remaining:
        best = max(remaining, key=lambda idx: (min_dist[idx], rng.random()))
        chosen.append(best)
        remaining.remove(best)
        dist = np.sqrt(((norm - norm[best]) ** 2).sum(axis=1))
        min_dist = np.minimum(min_dist, dist)
    return [target_metas[i] for i in chosen]


def _stratified_tail_diverse_order(target_metas, target_root, seed):
    if not target_metas:
        return []
    rows = []
    for meta_path in target_metas:
        meta = load_json(meta_path)
        hist = np.load(Path(target_root) / meta["hist_path"]).astype(np.float32)
        subtype = str(meta.get("subtype", "unknown")).lower()
        try:
            diameter = float(meta.get("diameter_mm") or 0.0)
        except Exception:
            diameter = 0.0
        q95 = _hist_quantile(hist, 0.95)
        size_bin = "small" if diameter <= 6.0 else "medium" if diameter <= 12.0 else "large"
        hu_bin = "low" if q95 <= -50.0 else "mid" if q95 <= 150.0 else "high"
        rows.append((meta_path, subtype, size_bin, hu_bin, _target_tail_score(target_root, meta_path)))

    buckets = {}
    for row in rows:
        buckets.setdefault(row[1:4], []).append(row)
    for key, bucket in buckets.items():
        ordered_paths = [row[0] for row in sorted(bucket, key=lambda item: item[4], reverse=True)]
        buckets[key] = ordered_paths

    def bucket_priority(key):
        subtype, size_bin, hu_bin = key
        subtype_weight = {"part_solid": 0, "solid": 1, "ground_glass": 2, "ggn": 2}.get(subtype, 3)
        size_weight = {"small": 0, "large": 1, "medium": 2}.get(size_bin, 3)
        hu_weight = {"high": 0, "low": 1, "mid": 2}.get(hu_bin, 3)
        return subtype_weight, size_weight, hu_weight, str(key)

    ordered = []
    active = sorted(buckets, key=bucket_priority)
    while active:
        next_active = []
        for key in active:
            bucket = buckets[key]
            if bucket:
                ordered.append(bucket.pop(0))
            if bucket:
                next_active.append(key)
        active = next_active
    return ordered


def _target_equivalent_diameter_mm(target_root, meta_path):
    meta = load_json(meta_path)
    try:
        diameter = float(meta.get("diameter_mm") or 0.0)
    except Exception:
        diameter = 0.0
    if diameter > 0:
        return diameter
    mask = load_volume(Path(target_root) / meta["mask_path"]) > 0
    volume_mm3 = float(mask.sum())
    return float((6.0 * volume_mm3 / np.pi) ** (1.0 / 3.0)) if volume_mm3 > 0 else 0.0


def _size_subtype_grid_order(target_metas, target_root, seed):
    rng = np.random.default_rng(int(seed))
    bins = (("6-10", 6.0, 10.0, 8.0), ("10-20", 10.0, 20.0, 15.0), ("20-30", 20.0, 30.0, 25.0))
    subtypes = ("ground_glass", "part_solid", "solid")
    buckets = {(name, subtype): [] for name, *_ in bins for subtype in subtypes}
    remainder = []
    for meta_path in target_metas:
        meta = load_json(meta_path)
        subtype = str(meta.get("subtype", "unknown")).lower().replace("-", "_")
        diameter = _target_equivalent_diameter_mm(target_root, meta_path)
        matched = False
        for name, lower, upper, center in bins:
            if lower <= diameter < upper and subtype in subtypes:
                buckets[(name, subtype)].append((abs(diameter - center), rng.random(), meta_path))
                matched = True
                break
        if not matched:
            remainder.append(meta_path)
    ordered = []
    for name, *_ in bins:
        for subtype in subtypes:
            bucket = sorted(buckets[(name, subtype)])
            if not bucket:
                raise ValueError(f"size_subtype_grid has no target for size={name}, subtype={subtype}")
            ordered.append(bucket[0][2])
            remainder.extend(item[2] for item in bucket[1:])
    rng.shuffle(remainder)
    return ordered + remainder


def _order_target_metas(target_metas, target_root, strategy, seed):
    strategy = str(strategy or "sequential")
    target_metas = list(target_metas)
    if strategy == "sequential":
        return target_metas
    rng = np.random.default_rng(int(seed))
    if strategy == "shuffle":
        out = target_metas[:]
        rng.shuffle(out)
        return out
    if strategy == "subtype_round_robin":
        buckets = {}
        for meta_path in target_metas:
            subtype = str(load_json(meta_path).get("subtype", "unknown"))
            buckets.setdefault(subtype, []).append(meta_path)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        preferred = ["ground_glass", "part_solid", "solid"]
        keys = [key for key in preferred if key in buckets] + sorted(set(buckets) - set(preferred))
        ordered = []
        while any(buckets[key] for key in keys):
            for key in keys:
                if buckets[key]:
                    ordered.append(buckets[key].pop())
        return ordered
    if strategy == "size_subtype_grid":
        return _size_subtype_grid_order(target_metas, target_root, seed)
    if strategy == "high_hu_tail":
        return sorted(target_metas, key=lambda p: _target_tail_score(target_root, p), reverse=True)
    if strategy == "feature_diverse":
        return _farthest_first_order(target_metas, target_root, seed, start="tail")
    if strategy == "feature_diverse_center":
        return _farthest_first_order(target_metas, target_root, seed, start="center")
    if strategy == "tail_diverse":
        return _stratified_tail_diverse_order(target_metas, target_root, seed)
    raise ValueError(f"unknown target_sampling_strategy={strategy}")


def _target_metas_for_config(target_metas, target_root, config, prefix="target"):
    strategy_key = f"{prefix}_sampling_strategy" if prefix != "target" else "target_sampling_strategy"
    seed_key = f"{prefix}_sampling_seed" if prefix != "target" else "target_sampling_seed"
    strategy = config.get(strategy_key, config.get("target_sampling_strategy", "sequential"))
    seed = config.get(seed_key, config.get("target_sampling_seed", config.get("seed", 42)))
    if str(strategy) == "explicit_target_ids":
        ids_key = f"{prefix}_sampling_ids" if prefix != "target" else "target_sampling_ids"
        requested = [str(value) for value in config.get(ids_key, []) or []]
        if not requested:
            raise ValueError(f"{ids_key} is required for explicit_target_ids")
        if len(requested) != len(set(requested)):
            raise ValueError(f"{ids_key} contains duplicate target IDs")
        indexed = {str(load_json(path).get("target_id")): path for path in target_metas}
        missing = [target_id for target_id in requested if target_id not in indexed]
        if missing:
            raise ValueError(f"explicit target IDs are absent from the target library: {missing}")
        selected = [indexed[target_id] for target_id in requested]
        explicit_only_key = (
            f"{prefix}_sampling_explicit_only"
            if prefix != "target"
            else "target_sampling_explicit_only"
        )
        if bool(config.get(explicit_only_key, False)):
            return selected
        selected_set = set(selected)
        return selected + [path for path in target_metas if path not in selected_set]
    return _order_target_metas(target_metas, target_root, strategy, seed)


def _target_injection_lookup(config):
    lookup = {}
    for item in config.get("target_sampling_injections", []) or []:
        try:
            sample_index = int(item.get("sample_index"))
        except Exception:
            continue
        lookup[sample_index] = dict(item)
    return lookup


def _target_order_key_for_injection(injection, config):
    strategy = injection.get(
        "target_sampling_strategy",
        config.get("injection_target_sampling_strategy", config.get("target_sampling_strategy", "sequential")),
    )
    seed = injection.get(
        "target_sampling_seed",
        config.get("injection_target_sampling_seed", config.get("target_sampling_seed", config.get("seed", 42))),
    )
    return str(strategy or "sequential"), int(seed)


def _subtype_hu_shift(config, subtype):
    shifts = config.get("subtype_hu_shift", {}) or {}
    try:
        return float(shifts.get(str(subtype), shifts.get("default", 0.0)))
    except Exception:
        return 0.0


def _apply_masked_hu_shift(image, mask, shift):
    image = np.asarray(image, dtype=np.float32).copy()
    mask = np.asarray(mask) > 0
    if abs(float(shift)) > 1.0e-6 and mask.any():
        image[mask] = np.clip(image[mask] + float(shift), HU_MIN, HU_MAX)
    return image


def _apply_background_weight(alpha, mask, lambda_bg):
    alpha = np.asarray(alpha, dtype=np.float32).copy()
    mask = np.asarray(mask) > 0
    try:
        weight = float(lambda_bg)
    except Exception:
        weight = 1.0
    if abs(weight - 1.0) <= 1.0e-6:
        return alpha
    outside = ~mask
    if weight > 1.0:
        alpha[outside] *= 1.0 / weight
    elif weight > 0:
        alpha[outside] = np.clip(alpha[outside] / weight, 0.0, 1.0)
    return alpha


def _normal_lung_suitability_from_values(inside, around):
    inside = np.asarray(inside, dtype=np.float32)
    around = np.asarray(around, dtype=np.float32)
    def fraction(values, lower, upper):
        return float(np.mean((values >= lower) & (values <= upper))) if values.size else 0.0

    lung_inside = fraction(inside, -950.0, -450.0)
    lung_around = fraction(around, -950.0, -450.0)
    tissue_inside = float(np.mean(inside > -300.0))
    tissue_around = float(np.mean(around > -300.0))
    air_inside = float(np.mean(inside < -985.0))
    air_around = float(np.mean(around < -985.0))
    texture_std = float(np.std(around))
    texture_penalty = abs(texture_std - 100.0) / 400.0
    return float(
        3.0 * lung_inside
        + 2.0 * lung_around
        - 5.0 * tissue_inside
        - 3.0 * tissue_around
        - 3.0 * air_inside
        - 2.0 * air_around
        - texture_penalty
    )


def _normal_mask_lung_suitability(normal, mask):
    normal = np.asarray(normal, dtype=np.float32)
    mask = np.asarray(mask) > 0
    if normal.shape != mask.shape or not mask.any():
        return -1.0e9
    shell = _dilate(mask, 3) & ~mask
    inside = normal[mask]
    around = normal[shell] if shell.any() else inside
    return _normal_lung_suitability_from_values(inside, around)


def _transport_texture_rank_guidance(
    source_image,
    source_mask,
    destination_mask,
    interpolation_order=1,
):
    source_image = np.asarray(source_image, dtype=np.float32)
    source_mask = np.asarray(source_mask) > 0
    destination_mask = np.asarray(destination_mask) > 0
    if source_image.shape != source_mask.shape or not source_mask.any():
        raise ValueError("spatial-rank source image/mask is invalid")
    if not destination_mask.any():
        raise ValueError("spatial-rank destination mask is empty")
    if source_mask.shape == destination_mask.shape and np.array_equal(source_mask, destination_mask):
        return source_image.copy()

    def bbox(mask):
        coordinates = np.argwhere(mask)
        start = coordinates.min(axis=0)
        stop = coordinates.max(axis=0) + 1
        return tuple(slice(int(start[axis]), int(stop[axis])) for axis in range(3))

    source_slices = bbox(source_mask)
    destination_slices = bbox(destination_mask)
    source_crop = source_image[source_slices]
    source_mask_crop = source_mask[source_slices]
    try:
        from scipy.ndimage import distance_transform_edt

        _distance, nearest = distance_transform_edt(
            ~source_mask_crop, return_distances=True, return_indices=True
        )
        source_crop = source_crop[tuple(nearest)]
    except Exception:
        source_crop = np.where(source_mask_crop, source_crop, float(np.mean(source_crop[source_mask_crop])))
    destination_shape = tuple(
        destination_slices[axis].stop - destination_slices[axis].start for axis in range(3)
    )
    interpolation_order = int(interpolation_order)
    if interpolation_order not in {0, 1, 2, 3}:
        raise ValueError(
            "texture transport interpolation order must be in {0, 1, 2, 3}, "
            f"got {interpolation_order}"
        )
    transported = _resize_array(
        source_crop,
        destination_shape,
        order=interpolation_order,
    )
    guidance = np.zeros(destination_mask.shape, dtype=np.float32)
    guidance[destination_slices] = transported
    return guidance


def _multiscale_texture_rank_guidance(
    guidance,
    mask,
    sigmas=(0.6, 1.2, 2.4),
    band_weights=(0.55, 0.30, 0.15),
    detail_gain=1.0,
    lowpass_weight=1.0,
):
    """Build a zero-parameter donor texture score from a Laplacian pyramid."""
    guidance = np.asarray(guidance, dtype=np.float32)
    mask = np.asarray(mask) > 0
    if guidance.shape != mask.shape or not mask.any():
        raise ValueError("multiscale texture guidance image/mask is invalid")
    sigmas = tuple(float(value) for value in sigmas if float(value) > 0)
    if not sigmas:
        return guidance.copy()
    weights = np.asarray(band_weights, dtype=np.float32).reshape(-1)
    if weights.size != len(sigmas):
        raise ValueError(
            "multiscale texture band weights must match sigmas: "
            f"{weights.size} != {len(sigmas)}"
        )
    if float(np.abs(weights).sum()) <= 1.0e-8:
        raise ValueError("multiscale texture band weights have zero mass")
    weights = weights / float(np.abs(weights).sum())

    try:
        from scipy.ndimage import gaussian_filter

        smooth = lambda array, sigma: gaussian_filter(
            array, sigma=float(sigma), mode="nearest"
        )
    except Exception:
        smooth = lambda array, sigma: _smooth(array, float(sigma))

    previous = guidance
    bands = []
    for sigma in sorted(sigmas):
        low = np.asarray(smooth(guidance, sigma), dtype=np.float32)
        bands.append(previous - low)
        previous = low
    detail = sum(float(weight) * band for weight, band in zip(weights, bands))

    def standardized(values):
        values = np.asarray(values, dtype=np.float32)
        return (values - float(values.mean())) / (float(values.std()) + 1.0e-6)

    score = np.zeros_like(guidance, dtype=np.float32)
    score[mask] = (
        float(lowpass_weight) * standardized(previous[mask])
        + float(detail_gain) * standardized(detail[mask])
    )
    return score


def _apply_multiscale_donor_residual_transfer(
    image,
    donor,
    texture_masks,
    strength,
    sigmas=(0.6, 1.2, 2.4),
    band_weights=(1.0, 0.75, 0.5),
    channel_weights=None,
    boundary_taper_width=2.0,
    boundary_taper_power=1.0,
    max_delta_hu=180.0,
    mode="match",
):
    """Transfer train-exemplar 3D detail bands without changing non-lesion voxels."""
    image = np.asarray(image, dtype=np.float32)
    donor = np.asarray(donor, dtype=np.float32)
    masks = np.asarray(texture_masks, dtype=np.float32)
    if masks.ndim == image.ndim:
        masks = masks[None]
    if donor.shape != image.shape or tuple(masks.shape[-3:]) != image.shape:
        raise ValueError(
            "residual texture image, donor, and masks must share a spatial shape: "
            f"image={image.shape}, donor={donor.shape}, masks={masks.shape}"
        )
    lesion = masks[0] > 0.5
    strength = float(strength)
    if strength < 0:
        raise ValueError(f"image_texture_residual_transfer_strength must be non-negative, got {strength}")
    if strength == 0 or not lesion.any():
        return image.copy(), {
            "status": "disabled" if strength == 0 else "skipped_empty_mask",
            "strength": strength,
        }

    sigmas = tuple(sorted(float(value) for value in sigmas if float(value) > 0))
    weights = np.asarray(band_weights, dtype=np.float32).reshape(-1)
    if not sigmas or weights.size != len(sigmas) or bool(np.any(weights < 0)):
        raise ValueError(
            "residual texture sigmas must be positive and have matching non-negative band weights"
        )
    if float(weights.sum()) <= 0:
        raise ValueError("residual texture band weights must contain positive mass")
    boundary_taper_width = float(boundary_taper_width)
    boundary_taper_power = float(boundary_taper_power)
    max_delta_hu = float(max_delta_hu)
    mode = str(mode).strip().lower()
    if boundary_taper_width < 0 or boundary_taper_power <= 0 or max_delta_hu <= 0:
        raise ValueError(
            "residual texture taper width must be non-negative, taper power and max delta positive"
        )
    if mode not in {"match", "add"}:
        raise ValueError(
            f"image_texture_residual_mode must be match or add, got {mode!r}"
        )

    channels = int(masks.shape[0])
    if channel_weights is None:
        weights_by_channel = np.ones(channels, dtype=np.float32)
    else:
        weights_by_channel = np.asarray(channel_weights, dtype=np.float32).reshape(-1)
        if weights_by_channel.size != channels or bool(np.any(weights_by_channel < 0)):
            raise ValueError(
                "residual texture channel weights must be non-negative and match mask channels"
            )
    if float(weights_by_channel.sum()) <= 0:
        raise ValueError("residual texture channel weights must contain positive mass")

    try:
        from scipy.ndimage import distance_transform_edt, gaussian_filter
    except Exception as exc:
        raise RuntimeError("multiscale donor residual transfer requires scipy.ndimage") from exc

    def nearest_fill(array):
        _distance, nearest = distance_transform_edt(
            ~lesion, return_distances=True, return_indices=True
        )
        return np.asarray(array, dtype=np.float32)[tuple(nearest)]

    def bands(array):
        filled = nearest_fill(array)
        previous = filled
        result = []
        for sigma in sigmas:
            low = gaussian_filter(filled, sigma=sigma, mode="nearest").astype(np.float32)
            result.append(previous - low)
            previous = low
        return result

    image_bands = bands(image)
    donor_bands = bands(donor)
    raw_delta = np.zeros_like(image, dtype=np.float32)
    band_report = []
    for sigma, weight, current_band, donor_band in zip(
        sigmas, weights, image_bands, donor_bands
    ):
        current_values = current_band[lesion]
        donor_values = donor_band[lesion]
        current_centered = current_band - float(current_values.mean())
        donor_centered = donor_band - float(donor_values.mean())
        residual = (
            donor_centered - current_centered
            if mode == "match"
            else donor_centered
        )
        raw_delta += float(weight) * residual
        band_report.append(
            {
                "sigma": float(sigma),
                "weight": float(weight),
                "current_rms_hu": float(
                    np.sqrt(np.mean(np.square(current_values, dtype=np.float64)))
                ),
                "donor_rms_hu": float(
                    np.sqrt(np.mean(np.square(donor_values, dtype=np.float64)))
                ),
            }
        )

    channel_envelope = np.zeros_like(image, dtype=np.float32)
    for channel, weight in zip(masks, weights_by_channel):
        channel_envelope += float(weight) * np.clip(channel, 0.0, 1.0)
    channel_envelope *= lesion.astype(np.float32)
    envelope_peak = float(channel_envelope.max())
    if envelope_peak > 0:
        channel_envelope /= envelope_peak

    if boundary_taper_width > 0:
        distance = distance_transform_edt(lesion).astype(np.float32)
        taper = np.clip(distance / boundary_taper_width, 0.0, 1.0)
        taper = np.power(taper, boundary_taper_power).astype(np.float32)
    else:
        taper = lesion.astype(np.float32)
    envelope = channel_envelope * taper
    envelope_mass = float(envelope[lesion].sum())
    if envelope_mass <= 1.0e-8:
        return image.copy(), {
            "status": "skipped_empty_envelope",
            "strength": strength,
        }

    # Remove the weighted DC component so this term changes texture, not phenotype HU.
    weighted_mean = float((raw_delta[lesion] * envelope[lesion]).sum() / envelope_mass)
    applied_delta = strength * (raw_delta - weighted_mean) * envelope
    applied_delta = np.clip(applied_delta, -max_delta_hu, max_delta_hu).astype(np.float32)
    output = image.copy()
    output[lesion] = np.clip(
        image[lesion] + applied_delta[lesion], HU_MIN, HU_MAX
    )
    changed = output[lesion] - image[lesion]
    return output, {
        "status": "pass",
        "mode": "train_exemplar_multiscale_3d_laplacian_residual",
        "transfer_mode": mode,
        "strength": strength,
        "sigmas": [float(value) for value in sigmas],
        "band_weights": [float(value) for value in weights],
        "channel_weights": [float(value) for value in weights_by_channel],
        "boundary_taper_width": boundary_taper_width,
        "boundary_taper_power": boundary_taper_power,
        "max_delta_hu": max_delta_hu,
        "mean_delta_hu": float(changed.mean()),
        "rms_delta_hu": float(
            np.sqrt(np.mean(np.square(changed, dtype=np.float64)))
        ),
        "max_abs_delta_hu": float(np.max(np.abs(changed))),
        "outside_mask_mae_hu": float(np.mean(np.abs(output[~lesion] - image[~lesion]))),
        "bands": band_report,
    }


def _apply_structured_donor_patch_transport(
    image,
    donor,
    texture_masks,
    strength,
    channel_weights=None,
    donor_std_blend=1.0,
    boundary_taper_width=1.5,
    boundary_taper_power=1.0,
    max_delta_hu=240.0,
):
    """Transport train-donor 3D intensity phase into the lesion core."""
    image = np.asarray(image, dtype=np.float32)
    donor = np.asarray(donor, dtype=np.float32)
    masks = np.asarray(texture_masks, dtype=np.float32)
    if masks.ndim == image.ndim:
        masks = masks[None]
    if donor.shape != image.shape or tuple(masks.shape[-3:]) != image.shape:
        raise ValueError(
            "structured donor image, target image, and masks must share a spatial shape: "
            f"image={image.shape}, donor={donor.shape}, masks={masks.shape}"
        )
    lesion = masks[0] > 0.5
    strength = float(strength)
    donor_std_blend = float(donor_std_blend)
    boundary_taper_width = float(boundary_taper_width)
    boundary_taper_power = float(boundary_taper_power)
    max_delta_hu = float(max_delta_hu)
    if strength < 0 or strength > 1:
        raise ValueError(
            "image_texture_phase_transfer_strength must be in [0, 1], "
            f"got {strength}"
        )
    if donor_std_blend < 0 or donor_std_blend > 1:
        raise ValueError(
            "image_texture_phase_std_blend must be in [0, 1], "
            f"got {donor_std_blend}"
        )
    if boundary_taper_width < 0 or boundary_taper_power <= 0 or max_delta_hu <= 0:
        raise ValueError(
            "structured donor taper width must be non-negative, and taper power and max delta positive"
        )
    if strength == 0 or not lesion.any():
        return image.copy(), {
            "status": "disabled" if strength == 0 else "skipped_empty_mask",
            "strength": strength,
        }

    channels = int(masks.shape[0])
    if channel_weights is None:
        weights_by_channel = np.ones(channels, dtype=np.float32)
    else:
        weights_by_channel = np.asarray(channel_weights, dtype=np.float32).reshape(-1)
        if weights_by_channel.size != channels or bool(np.any(weights_by_channel < 0)):
            raise ValueError(
                "structured donor channel weights must be non-negative and match mask channels"
            )
    if float(weights_by_channel.sum()) <= 0:
        raise ValueError("structured donor channel weights must contain positive mass")

    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as exc:
        raise RuntimeError("structured donor patch transport requires scipy.ndimage") from exc

    channel_envelope = np.zeros_like(image, dtype=np.float32)
    for channel, weight in zip(masks, weights_by_channel):
        channel_envelope += float(weight) * np.clip(channel, 0.0, 1.0)
    channel_envelope *= lesion.astype(np.float32)
    peak = float(channel_envelope.max())
    if peak > 0:
        channel_envelope /= peak
    if boundary_taper_width > 0:
        distance = distance_transform_edt(lesion).astype(np.float32)
        taper = np.power(
            np.clip(distance / boundary_taper_width, 0.0, 1.0),
            boundary_taper_power,
        ).astype(np.float32)
    else:
        taper = lesion.astype(np.float32)
    envelope = channel_envelope * taper
    if float(envelope[lesion].sum()) <= 1.0e-8:
        return image.copy(), {
            "status": "skipped_empty_envelope",
            "strength": strength,
        }

    current_values = image[lesion].astype(np.float64)
    donor_values = donor[lesion].astype(np.float64)
    current_mean = float(current_values.mean())
    donor_mean = float(donor_values.mean())
    current_std = float(current_values.std())
    donor_std = float(donor_values.std())
    target_std = (1.0 - donor_std_blend) * current_std + donor_std_blend * donor_std
    if donor_std > 1.0e-6:
        aligned_donor = (donor - donor_mean) * (target_std / donor_std) + current_mean
    else:
        aligned_donor = np.full_like(donor, current_mean, dtype=np.float32)
    raw_delta = (np.asarray(aligned_donor, dtype=np.float32) - image) * envelope
    applied_delta = np.clip(strength * raw_delta, -max_delta_hu, max_delta_hu).astype(np.float32)
    output = image.copy()
    output[lesion] = np.clip(
        image[lesion] + applied_delta[lesion], HU_MIN, HU_MAX
    )
    changed = output[lesion] - image[lesion]
    return output, {
        "status": "pass",
        "mode": "train_exemplar_shape_aligned_3d_phase_transport",
        "strength": strength,
        "donor_std_blend": donor_std_blend,
        "current_mean_hu": current_mean,
        "donor_mean_hu": donor_mean,
        "current_std_hu": current_std,
        "donor_std_hu": donor_std,
        "transport_target_std_hu": target_std,
        "channel_weights": [float(value) for value in weights_by_channel],
        "boundary_taper_width": boundary_taper_width,
        "boundary_taper_power": boundary_taper_power,
        "max_delta_hu": max_delta_hu,
        "mean_delta_hu": float(changed.mean()),
        "rms_delta_hu": float(
            np.sqrt(np.mean(np.square(changed, dtype=np.float64)))
        ),
        "max_abs_delta_hu": float(np.max(np.abs(changed))),
        "outside_mask_mae_hu": float(np.mean(np.abs(output[~lesion] - image[~lesion]))),
    }


def _apply_gradient_domain_donor_transport(
    image,
    donor,
    lesion_mask,
    strength,
    donor_std_blend=1.0,
    screening=0.05,
    histogram_blend=0.50,
    max_delta_hu=240.0,
    max_iterations=160,
    tolerance_hu=0.01,
    relaxation=0.80,
):
    """Reconstruct lesion detail from train-donor gradients with fixed CT boundary values."""
    image = np.asarray(image, dtype=np.float32)
    donor = np.asarray(donor, dtype=np.float32)
    lesion = np.asarray(lesion_mask) > 0
    strength = float(strength)
    donor_std_blend = float(donor_std_blend)
    screening = float(screening)
    histogram_blend = float(histogram_blend)
    max_delta_hu = float(max_delta_hu)
    max_iterations = int(max_iterations)
    tolerance_hu = float(tolerance_hu)
    relaxation = float(relaxation)
    if image.shape != donor.shape or image.shape != lesion.shape:
        raise ValueError(
            "gradient-domain image, donor, and lesion mask must share a shape: "
            f"image={image.shape}, donor={donor.shape}, mask={lesion.shape}"
        )
    if not 0.0 <= strength <= 1.0:
        raise ValueError(
            "image_texture_gradient_transport_strength must be in [0, 1], "
            f"got {strength}"
        )
    if not 0.0 <= donor_std_blend <= 1.0:
        raise ValueError(
            "image_texture_gradient_std_blend must be in [0, 1], "
            f"got {donor_std_blend}"
        )
    if (
        screening < 0
        or not 0.0 <= histogram_blend <= 1.0
        or max_delta_hu <= 0
        or max_iterations <= 0
        or tolerance_hu <= 0
        or not 0.0 < relaxation <= 1.0
    ):
        raise ValueError(
            "gradient-domain screening must be non-negative, histogram blend in [0, 1], "
            "max delta/iterations/tolerance positive, and relaxation in (0, 1]"
        )
    if strength == 0 or not lesion.any():
        return image.copy(), {
            "status": "disabled" if strength == 0 else "skipped_empty_mask",
            "strength": strength,
        }

    current_values = image[lesion].astype(np.float64)
    donor_values = donor[lesion].astype(np.float64)
    current_mean = float(current_values.mean())
    current_std = float(current_values.std())
    donor_mean = float(donor_values.mean())
    donor_std = float(donor_values.std())
    target_std = (1.0 - donor_std_blend) * current_std + donor_std_blend * donor_std
    if donor_std > 1.0e-6:
        aligned_donor = (donor.astype(np.float64) - donor_mean) * (
            target_std / donor_std
        ) + current_mean
    else:
        aligned_donor = np.full_like(donor, current_mean, dtype=np.float64)

    def laplacian(volume):
        volume = np.asarray(volume, dtype=np.float64)
        result = np.zeros_like(volume, dtype=np.float64)
        for axis in range(volume.ndim):
            left = [slice(None)] * volume.ndim
            right = [slice(None)] * volume.ndim
            left[axis] = slice(0, -1)
            right[axis] = slice(1, None)
            delta = volume[tuple(right)] - volume[tuple(left)]
            result[tuple(left)] += delta
            result[tuple(right)] -= delta
        return result

    target_laplacian = (
        (1.0 - strength) * laplacian(image)
        + strength * laplacian(aligned_donor)
    )
    coordinates = np.argwhere(lesion)
    starts = np.maximum(coordinates.min(axis=0) - 1, 0)
    stops = np.minimum(coordinates.max(axis=0) + 2, np.asarray(lesion.shape))
    crop_slices = tuple(
        slice(int(starts[axis]), int(stops[axis])) for axis in range(lesion.ndim)
    )
    local_image = image[crop_slices].astype(np.float64)
    local_mask = lesion[crop_slices]
    local_laplacian = target_laplacian[crop_slices]
    solution = local_image.copy()
    degree = np.zeros(local_image.shape, dtype=np.float64)
    for axis in range(local_image.ndim):
        left = [slice(None)] * local_image.ndim
        right = [slice(None)] * local_image.ndim
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        degree[tuple(left)] += 1.0
        degree[tuple(right)] += 1.0
    denominator = degree + screening
    converged = False
    final_update_hu = float("inf")
    iterations_run = 0
    for iteration in range(max_iterations):
        neighbor_sum = np.zeros_like(solution)
        for axis in range(solution.ndim):
            left = [slice(None)] * solution.ndim
            right = [slice(None)] * solution.ndim
            left[axis] = slice(0, -1)
            right[axis] = slice(1, None)
            neighbor_sum[tuple(left)] += solution[tuple(right)]
            neighbor_sum[tuple(right)] += solution[tuple(left)]
        candidate = (
            neighbor_sum - local_laplacian + screening * local_image
        ) / denominator
        updated = solution[local_mask] + relaxation * (
            candidate[local_mask] - solution[local_mask]
        )
        final_update_hu = float(np.max(np.abs(updated - solution[local_mask])))
        solution[local_mask] = updated
        iterations_run = iteration + 1
        if final_update_hu <= tolerance_hu:
            converged = True
            break
    solved = np.asarray(solution[local_mask], dtype=np.float64)
    if not np.all(np.isfinite(solved)):
        raise RuntimeError("gradient-domain donor transport produced non-finite values")

    if histogram_blend > 0:
        ranked = np.empty_like(solved)
        ranked[np.argsort(solved, kind="stable")] = np.sort(current_values)
        solved = (1.0 - histogram_blend) * solved + histogram_blend * ranked
    solved = np.clip(
        solved,
        current_values - max_delta_hu,
        current_values + max_delta_hu,
    )
    solved = np.clip(solved, HU_MIN, HU_MAX)
    output = image.copy()
    output[lesion] = solved.astype(np.float32)

    def gradient_rms(volume):
        samples = []
        for axis in range(volume.ndim):
            left = [slice(None)] * volume.ndim
            right = [slice(None)] * volume.ndim
            left[axis] = slice(0, -1)
            right[axis] = slice(1, None)
            valid = lesion[tuple(left)] & lesion[tuple(right)]
            if valid.any():
                delta = volume[tuple(right)] - volume[tuple(left)]
                samples.append(np.asarray(delta[valid], dtype=np.float64))
        if not samples:
            return 0.0
        concatenated = np.concatenate(samples)
        return float(np.sqrt(np.mean(np.square(concatenated))))

    changed = output[lesion] - image[lesion]
    return output, {
        "status": "pass",
        "mode": "screened_poisson_train_donor_gradient_transport",
        "strength": strength,
        "donor_std_blend": donor_std_blend,
        "screening": screening,
        "histogram_blend": histogram_blend,
        "max_delta_hu": max_delta_hu,
        "solver": "cropped_weighted_jacobi",
        "solver_iterations": int(iterations_run),
        "solver_max_iterations": int(max_iterations),
        "solver_tolerance_hu": tolerance_hu,
        "solver_relaxation": relaxation,
        "solver_final_update_hu": final_update_hu,
        "solver_converged": bool(converged),
        "current_gradient_rms_hu": gradient_rms(image),
        "donor_gradient_rms_hu": gradient_rms(aligned_donor),
        "output_gradient_rms_hu": gradient_rms(output),
        "mean_delta_hu": float(changed.mean()),
        "rms_delta_hu": float(np.sqrt(np.mean(np.square(changed, dtype=np.float64)))),
        "max_abs_delta_hu": float(np.max(np.abs(changed))),
        "outside_mask_mae_hu": float(np.mean(np.abs(output[~lesion] - image[~lesion]))),
    }


def _spectral_surrogate_texture_rank_guidance(
    candidate,
    destination_mask,
    source_image,
    source_mask,
    rng,
    sigmas=(0.6, 1.2, 2.4),
    detail_gain=1.0,
    lowpass_weight=1.0,
    candidate_phase_blend=0.75,
):
    """Transfer train-lesion band energy without copying its spatial phase."""
    candidate = np.asarray(candidate, dtype=np.float32)
    destination_mask = np.asarray(destination_mask) > 0
    source_image = np.asarray(source_image, dtype=np.float32)
    source_mask = np.asarray(source_mask) > 0
    if candidate.shape != destination_mask.shape or not destination_mask.any():
        raise ValueError("spectral surrogate destination image/mask is invalid")
    if source_image.shape != source_mask.shape or not source_mask.any():
        raise ValueError("spectral surrogate train source image/mask is invalid")
    sigmas = tuple(sorted(float(value) for value in sigmas if float(value) > 0))
    if not sigmas:
        return candidate.copy()
    candidate_phase_blend = min(max(float(candidate_phase_blend), 0.0), 1.0)

    try:
        from scipy.ndimage import distance_transform_edt, gaussian_filter

        _distance, nearest = distance_transform_edt(
            ~source_mask, return_distances=True, return_indices=True
        )
        source_filled = source_image[tuple(nearest)]
        smooth = lambda array, sigma: gaussian_filter(
            array, sigma=float(sigma), mode="nearest"
        )
    except Exception:
        source_filled = np.where(
            source_mask, source_image, float(np.mean(source_image[source_mask]))
        )
        smooth = lambda array, sigma: _smooth(array, float(sigma))

    source_previous = source_filled
    donor_energy = []
    for sigma in sigmas:
        source_low = np.asarray(smooth(source_filled, sigma), dtype=np.float32)
        source_band = source_previous - source_low
        donor_energy.append(
            float(
                np.sqrt(
                    np.mean(
                        np.square(source_band[source_mask], dtype=np.float64)
                    )
                )
            )
        )
        source_previous = source_low
    donor_energy = np.asarray(donor_energy, dtype=np.float32)
    if float(donor_energy.sum()) <= 1.0e-6:
        donor_energy = np.ones(len(sigmas), dtype=np.float32)
    donor_energy /= float(donor_energy.sum())

    noise = rng.normal(size=candidate.shape).astype(np.float32)
    candidate_previous = candidate
    noise_previous = noise
    detail = np.zeros(candidate.shape, dtype=np.float32)

    def standardized(array):
        values = np.asarray(array, dtype=np.float32)[destination_mask]
        out = np.zeros(candidate.shape, dtype=np.float32)
        out[destination_mask] = (values - float(values.mean())) / (
            float(values.std()) + 1.0e-6
        )
        return out

    for weight, sigma in zip(donor_energy, sigmas):
        candidate_low = np.asarray(smooth(candidate, sigma), dtype=np.float32)
        noise_low = np.asarray(smooth(noise, sigma), dtype=np.float32)
        candidate_band = standardized(candidate_previous - candidate_low)
        noise_band = standardized(noise_previous - noise_low)
        phase = (
            candidate_phase_blend * candidate_band
            + (1.0 - candidate_phase_blend) * noise_band
        )
        detail += float(weight) * phase
        candidate_previous = candidate_low
        noise_previous = noise_low

    score = np.zeros(candidate.shape, dtype=np.float32)
    score[destination_mask] = (
        float(lowpass_weight) * standardized(candidate_previous)[destination_mask]
        + float(detail_gain) * detail[destination_mask]
    )
    return score


def _apply_structure_guided_rank_projection(
    image,
    reference,
    mask,
    strength,
    sigma=1.4,
    tissue_threshold=-650.0,
):
    """Place existing lesion HU ranks along source-context structures."""
    image = np.asarray(image, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    mask = np.asarray(mask) > 0
    strength = min(max(float(strength), 0.0), 1.0)
    if image.shape != reference.shape or not mask.any() or strength <= 0:
        return image.copy()
    try:
        from scipy.ndimage import gaussian_filter

        smooth = gaussian_filter(reference, sigma=max(float(sigma), 1.0e-3), mode="nearest")
    except Exception:
        smooth = _smooth(reference, max(float(sigma), 1.0e-3))
    positive_detail = np.maximum(reference - smooth, 0.0)
    tissue = np.maximum(reference - float(tissue_threshold), 0.0)

    def standardized(values):
        values = np.asarray(values, dtype=np.float32)
        return (values - float(values.mean())) / (float(values.std()) + 1.0e-6)

    current = image[mask]
    structure = 0.7 * standardized(positive_detail[mask]) + 0.3 * standardized(
        tissue[mask]
    )
    guidance = (1.0 - strength) * standardized(current) + strength * structure
    projected = current.copy()
    projected[np.argsort(guidance, kind="stable")] = np.sort(current)
    out = image.copy()
    out[mask] = projected
    return out


def _load_training_texture_source(target_meta, data_root, source_sample_id=None):
    source_id = str(
        source_sample_id
        or target_meta.get("spatial_rank_source_sample_id")
        or target_meta.get("source_sample_id")
        or ""
    )
    if not source_id:
        raise ValueError("training texture-rank guidance requires source_sample_id")
    data_root = Path(target_meta.get("source_data_root") or data_root)
    manifest_path = data_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"training texture-rank manifest is missing: {manifest_path}")
    source_row = None
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("sample_id") or "") == source_id:
                source_row = row
                break
    if source_row is None:
        raise ValueError(f"texture-rank source is absent from the data manifest: {source_id}")
    if str(source_row.get("split") or "") != "train":
        raise ValueError(
            "texture-rank guidance is restricted to train exemplars; "
            f"source={source_id}, split={source_row.get('split')!r}"
        )
    source_image_path = data_root / "rois" / "pathological" / "images" / f"{source_id}.nii.gz"
    source_mask_path = data_root / "rois" / "pathological" / "masks" / f"{source_id}.nii.gz"
    if not source_image_path.exists():
        raise FileNotFoundError(f"training texture-rank image is missing: {source_image_path}")
    if not source_mask_path.exists():
        raise FileNotFoundError(f"training texture-rank mask is missing: {source_mask_path}")
    return (
        load_volume(source_image_path).astype(np.float32),
        load_volume(source_mask_path) > 0,
    )


def _load_training_texture_rank_guidance(
    target_meta,
    data_root,
    destination_mask=None,
    source_sample_id=None,
):
    source_image, source_mask = _load_training_texture_source(
        target_meta, data_root, source_sample_id=source_sample_id
    )
    if destination_mask is None:
        return source_image
    return _transport_texture_rank_guidance(source_image, source_mask, destination_mask)


def _load_texture_mask_channels(
    target_meta,
    target_root,
    lesion_mask,
    derived_channels=None,
):
    """Load or derive phenotype masks while always retaining the whole lesion."""
    lesion_mask = np.asarray(lesion_mask) > 0
    channels = [lesion_mask]
    names = ["lesion"]
    weights = [1.0]
    derived_channels = derived_channels or {}
    if not isinstance(derived_channels, dict):
        raise ValueError("texture_derived_mask_channels must be a mapping")
    for name, value in derived_channels.items():
        if str(name) == "lesion":
            continue
        record = value if isinstance(value, dict) else {"operation": value}
        operation = str(record.get("operation", name)).lower()
        width = max(1, int(record.get("width", 1)))
        weight = float(record.get("weight", 1.0))
        boundary = _inner_boundary(lesion_mask, width)
        if operation in {"core", "erode", "eroded_core"}:
            channel = lesion_mask & ~boundary
            min_voxels = max(1, int(record.get("min_voxels", 8)))
            if int(channel.sum()) < min_voxels:
                channel = lesion_mask.copy()
        elif operation in {"shell", "boundary", "inner_boundary"}:
            channel = boundary
        else:
            raise ValueError(
                f"unsupported derived texture mask operation {operation!r} for {name!r}"
            )
        if not channel.any():
            raise ValueError(f"derived texture mask channel {name!r} is empty")
        if weight < 0:
            raise ValueError(f"derived texture mask channel {name!r} has a negative weight")
        channels.append(channel)
        names.append(str(name))
        weights.append(weight)
    specification = target_meta.get("texture_mask_paths") or target_meta.get("spatial_mask_paths") or {}
    if specification and not isinstance(specification, dict):
        raise ValueError("texture_mask_paths must map channel names to paths or path/weight records")
    for name, value in specification.items():
        if str(name) in names:
            continue
        if isinstance(value, dict):
            path_value = value.get("path")
            weight = float(value.get("weight", 1.0))
        else:
            path_value = value
            weight = 1.0
        if not path_value:
            raise ValueError(f"texture mask channel {name!r} has no path")
        path = Path(path_value)
        if not path.is_absolute():
            path = Path(target_root) / path
        if not path.exists():
            raise FileNotFoundError(f"texture mask channel {name!r} is missing: {path}")
        channel = np.asarray(load_volume(path)) > 0
        if channel.shape != lesion_mask.shape:
            channel = _resize_array(channel.astype(np.float32), lesion_mask.shape, order=0) > 0.5
        channel &= lesion_mask
        if not channel.any():
            raise ValueError(f"texture mask channel {name!r} is empty inside the lesion mask")
        if weight < 0:
            raise ValueError(f"texture mask channel {name!r} has a negative weight")
        channels.append(channel)
        names.append(str(name))
        weights.append(weight)
    return np.stack(channels, axis=0).astype(np.float32), names, weights


def _apply_target_texture_blend(image, mask, target_meta, data_root, blend):
    blend = float(blend)
    if blend <= 0:
        return np.asarray(image, dtype=np.float32)
    source_id = target_meta.get("source_sample_id")
    if not source_id:
        return np.asarray(image, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32).copy()
    mask = np.asarray(mask) > 0
    if not mask.any():
        return image
    source_data_root = Path(target_meta.get("source_data_root") or data_root)
    source_image_path = source_data_root / "rois" / "pathological" / "images" / f"{source_id}.nii.gz"
    source_mask_path = source_data_root / "rois" / "pathological" / "masks" / f"{source_id}.nii.gz"
    if not source_image_path.exists() or not source_mask_path.exists():
        return image
    source_image = load_volume(source_image_path).astype(np.float32)
    source_mask = load_volume(source_mask_path) > 0
    current = image[mask]
    target = source_image[source_mask]
    if current.size == 0 or target.size == 0:
        return image
    target_sorted = np.sort(target.astype(np.float32))
    if target_sorted.size != current.size:
        xp = np.linspace(0.0, 1.0, target_sorted.size)
        xq = np.linspace(0.0, 1.0, current.size)
        target_sorted = np.interp(xq, xp, target_sorted).astype(np.float32)
    order = np.argsort(current)
    ranked = current.copy()
    ranked[order] = target_sorted
    image[mask] = np.clip((1.0 - blend) * current + blend * ranked, HU_MIN, HU_MAX)
    return image


def _validate_posthoc_target_transfer_role(generation_split, texture_blend, subtype_hu_shift):
    """Keep real-target voxel transfer out of validation and formal inference."""
    texture_blend = float(texture_blend)
    shifts = subtype_hu_shift if isinstance(subtype_hu_shift, dict) else {}
    nonzero_shifts = {
        str(key): float(value)
        for key, value in shifts.items()
        if abs(float(value)) > 1.0e-8
    }
    if str(generation_split) != "train" and (
        abs(texture_blend) > 1.0e-8 or nonzero_shifts
    ):
        raise ValueError(
            "Post-hoc real-target texture transfer and subtype HU shifts are train-augmentation-only; "
            f"got split={generation_split!r}, target_texture_blend={texture_blend}, "
            f"nonzero_subtype_hu_shift={nonzero_shifts}"
        )
    return {
        "generation_split": str(generation_split),
        "target_texture_blend": texture_blend,
        "nonzero_subtype_hu_shift": nonzero_shifts,
        "role": "train_augmentation" if texture_blend or nonzero_shifts else "disabled",
    }


def _asset_incomplete(path, min_bytes):
    path = Path(path)
    if path.is_dir():
        return False, ""
    aria2_state = Path(str(path) + ".aria2")
    if aria2_state.exists():
        return True, f"{path} has unfinished aria2 state file {aria2_state}"
    if min_bytes is not None and path.stat().st_size < int(min_bytes):
        return True, f"{path} is incomplete: {path.stat().st_size} < {int(min_bytes)} bytes"
    return False, ""


class NodFlow(NodFlowRuntime):
    name = "nodflow"

    def _posterior_energies(self, prior):
        cache = getattr(self, "_posterior_energy_cache", None)
        if cache is None:
            cache = {}
            self._posterior_energy_cache = cache
        if prior not in cache:
            cache[prior] = load_posterior_energies(self.method_config, prior)
        return cache[prior]

    def _select_suitable_normal(self, normal_images, mask, sample_index):
        cache = getattr(self, "_normal_suitability_volume_cache", None)
        if cache is None:
            cache = {}
            self._normal_suitability_volume_cache = cache
        mask = np.asarray(mask) > 0
        inside_indices = np.flatnonzero(mask)
        shell = _dilate(mask, 3) & ~mask
        around_indices = np.flatnonzero(shell) if shell.any() else inside_indices
        scored = []
        for path in normal_images:
            key = str(path)
            if key not in cache:
                cache[key] = self._preprocess_source(load_volume(path))
            flat = cache[key].reshape(-1)
            score = _normal_lung_suitability_from_values(
                flat[inside_indices], flat[around_indices]
            )
            scored.append((score, key, path))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        top_k = max(1, int(self.method_config.get("normal_suitability_top_k", 5)))
        chosen = scored[int(sample_index) % min(top_k, len(scored))]
        return chosen[2], cache[str(chosen[2])], float(chosen[0])

    def _method_id(self):
        return str(self.method_config.get("method_id", self.name))

    def _prior_name(self):
        return str(self.method_config.get("prior", "maisi_rflow"))

    def _uses_maisi_stepwise_solver(self):
        return self._prior_name() == "maisi_rflow" and bool(
            self.method_config.get("maisi_stepwise_flow_solver", False)
        )

    def _uses_ctflow_stepwise_solver(self):
        return self._prior_name() == "ctflow" and bool(
            self.method_config.get("ctflow_stepwise_flow_solver", False)
        )

    def _uses_latent_stepwise_solver(self):
        return self._uses_maisi_stepwise_solver() or self._uses_ctflow_stepwise_solver()

    def _solver_fidelity(self):
        if self._uses_latent_stepwise_solver():
            return "latent_eq18_22_stepwise_plus_image_projection"
        configured = str(self.method_config.get("solver_fidelity", "image_space_projection_approximation"))
        if configured.startswith("latent_eq18_22"):
            return "image_space_projection_approximation"
        return configured

    def _trajectory_reprojection_status(self):
        if self._uses_maisi_stepwise_solver():
            return "latent_eq22_each_rflow_timestep"
        if self._uses_ctflow_stepwise_solver():
            return "latent_eq22_each_ctflow_euler_timestep"
        return "approximation_not_full_latent_eq18_21"

    def _prior_root(self):
        if self._prior_name() == "ctflow":
            key = "ctflow_root"
            env = "CTFLOW_ROOT"
        else:
            key = "maisi_bundle"
            env = "MAISI_BUNDLE_ROOT"
        value = self.method_config.get(key)
        if not value:
            import os

            value = os.environ.get(env)
        return Path(value) if value else None

    def _maisi_weight_root(self):
        value = self.method_config.get("maisi_weight_root")
        if not value:
            import os

            value = os.environ.get("MAISI_RFLOW_CT_WEIGHT_ROOT")
        return Path(value) if value else None

    def _ctflow_code_root(self):
        value = self.method_config.get("ctflow_code_root")
        if not value:
            import os

            value = os.environ.get("CTFLOW_CODE_ROOT")
        return Path(value) if value else None

    def _validate_prior_assets(self):
        root = self._prior_root()
        prior = self._prior_name()
        if root is None or not root.exists():
            raise FileNotFoundError(f"{self._method_id()} requires {prior} prior root; configure it before generation")
        min_bytes = self.method_config.get("asset_min_bytes", {})
        if prior == "ctflow":
            required = self.method_config.get("ctflow_required_files", ["checkpoint-680000/denoiser_ema"])
            missing = []
            for rel in required:
                path = root / rel
                if not path.exists():
                    missing.append(str(path))
                else:
                    incomplete, reason = _asset_incomplete(path, min_bytes.get(rel))
                    if incomplete:
                        missing.append(reason)
            code_root = self._ctflow_code_root()
            for rel in self.method_config.get("ctflow_code_required_files", []):
                if code_root is None or not (code_root / rel).exists():
                    missing.append(str((code_root / rel) if code_root is not None else f"CTFLOW_CODE_ROOT/{rel}"))
        else:
            weight_root = self._maisi_weight_root()
            required = self.method_config.get(
                "maisi_required_files",
                [
                    "scripts/sample.py",
                    "scripts/infer_image_from_mask.py",
                    "scripts/utils_infer.py",
                    "configs/config_infer.json",
                    "configs/environment_rflow-ct.json",
                    "configs/config_network_rflow.json",
                    "models/autoencoder_v1.pt",
                    "models/diff_unet_3d_rflow-ct.pt",
                    "models/controlnet_3d_rflow-ct.pt",
                ],
            )
            missing = []
            for rel in required:
                path = root / rel
                weight_path = weight_root / rel if weight_root is not None else None
                if path.exists():
                    incomplete, reason = _asset_incomplete(path, min_bytes.get(rel))
                    if incomplete:
                        missing.append(reason)
                    continue
                if weight_path is not None and weight_path.exists():
                    incomplete, reason = _asset_incomplete(weight_path, min_bytes.get(rel))
                    if incomplete:
                        missing.append(reason)
                    continue
                missing.append(str(path if weight_root is None else f"{path} or {weight_path}"))
        if missing:
            raise FileNotFoundError(f"{self._method_id()} missing frozen prior assets: {missing}")
        return root

    def _resolve_prior_asset(self, rel):
        rel = Path(rel)
        root = self._prior_root()
        weight_root = self._maisi_weight_root()
        if weight_root is not None and (weight_root / rel).exists():
            return weight_root / rel
        if root is not None and (root / rel).exists():
            return root / rel
        raise FileNotFoundError(f"missing prior asset {rel}")

    def _load_maisi_rflow_backend(self):
        if hasattr(self, "_maisi_rflow_backend"):
            return self._maisi_rflow_backend
        root = self._validate_prior_assets()
        weight_root = self._maisi_weight_root() or root
        import sys

        import torch

        root = Path(root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        runtime_compat = [
            _install_scipy_stats_import_stub(),
            _patch_monai_rflow_scheduler_input_size(),
        ]
        from scripts.diff_model_setting import load_config
        from scripts.infer_image_from_mask import ldm_conditional_sample_one_image
        from scripts.utils_infer import build_conditioning_tensors, load_image_models

        env_file = root / "configs" / "environment_rflow-ct.json"
        infer_file = root / str(self.method_config.get("maisi_inference_config", "configs/config_infer_16g_256x256x128.json"))
        net_file = root / "configs" / "config_network_rflow.json"
        cfg = load_config(str(env_file), str(infer_file), str(net_file))
        for key in ("trained_autoencoder_path", "trained_diffusion_path", "trained_controlnet_path"):
            path = Path(getattr(cfg, key))
            if not path.is_absolute():
                candidate = weight_root / path
                setattr(cfg, key, str(candidate if candidate.exists() else root / path))
        cfg.output_size = list(self.method_config.get("maisi_output_size", cfg.output_size))
        cfg.spacing = list(self.method_config.get("maisi_spacing", cfg.spacing))
        cfg.num_inference_steps = int(
            self.method_config.get(
                "maisi_num_inference_steps",
                self.method_config.get("sampling_steps", cfg.num_inference_steps),
            )
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autoencoder, diffusion_unet, controlnet, scale_factor, noise_scheduler = load_image_models(cfg, device)
        self._maisi_rflow_backend = {
            "cfg": cfg,
            "device": device,
            "autoencoder": autoencoder,
            "diffusion_unet": diffusion_unet,
            "controlnet": controlnet,
            "scale_factor": scale_factor,
            "noise_scheduler": noise_scheduler,
            "build_conditioning_tensors": build_conditioning_tensors,
            "sample_image": ldm_conditional_sample_one_image,
            "runtime_compat": ",".join(runtime_compat),
        }
        return self._maisi_rflow_backend

    def _load_ctflow_embedding(self):
        import torch

        path = self._ctflow_embedding_path()
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            for key in ("embedding", "text_embedding", "prompt_embedding", "last_hidden_state"):
                if key in value:
                    value = value[key]
                    break
        value = torch.as_tensor(value).detach().float()
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if int(value.shape[-1]) != 768:
            raise ValueError(f"CTFlow embedding must have last dimension 768, got {list(value.shape)}")
        value = value.reshape(-1, 768)[:1]
        value = value / (value.norm(p=2, dim=-1, keepdim=True) + 1.0e-6)
        return value.unsqueeze(0)

    def _load_ctflow_backend(self):
        if hasattr(self, "_ctflow_backend"):
            return self._ctflow_backend
        root = self._validate_prior_assets()
        code_root = self._ctflow_code_root()
        if code_root is None:
            raise FileNotFoundError("ctflow_code_root is not configured")
        vae_path = self.method_config.get("ctflow_flux_vae")
        if not vae_path:
            raise FileNotFoundError("ctflow_flux_vae is not configured")
        vae_path = Path(vae_path)
        if not vae_path.exists():
            raise FileNotFoundError(f"CTFlow FLUX VAE path does not exist: {vae_path}")

        import sys

        import torch
        from omegaconf import OmegaConf

        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))
        from auto_regressive_generate import LatentAutoregressiveGenerator
        from echosyn.common import get_vae_scaler, instantiate, instantiate_class_from_config

        ctflow_runtime_compat = _patch_ctflow_pos_embed_from_numpy()
        config_path = Path(self.method_config.get("ctflow_config", code_root / "lvfm/configs/jiayi_lvfm_STDiT-L2_16f8_all.yaml"))
        config = OmegaConf.load(str(config_path))
        config.vae.pretrained = str(vae_path)
        ckpt_dir = root / self.method_config.get("official_model_variant", "checkpoint-680000/denoiser_ema")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        denoiser = instantiate_class_from_config(config.denoiser)
        denoiser = denoiser.from_pretrained(str(ckpt_dir)).to(device).eval()
        vae = instantiate(config.vae).eval().to(device)
        vae_scaling = get_vae_scaler(config, device)
        generator = LatentAutoregressiveGenerator(
            denoiser=denoiser,
            vae=vae,
            device=device,
            vae_scaling=vae_scaling,
            config=config,
            block_size=int(self.method_config.get("ctflow_block_size", 16)),
            overlap=int(self.method_config.get("ctflow_overlap", 8)),
        )
        self._ctflow_backend = {
            "root": root,
            "code_root": code_root,
            "config": config,
            "device": device,
            "denoiser": denoiser,
            "vae": vae,
            "vae_scaling": vae_scaling,
            "generator": generator,
            "runtime_compat": ctflow_runtime_compat,
        }
        return self._ctflow_backend

    def train(self, output_dir=None):
        seed_everything(42)
        output_dir = Path(output_dir) if output_dir else Path("checkpoints") / self._method_id()
        ensure_dir(output_dir)
        prior_root = self._validate_prior_assets()
        metadata = {
            "method_name": self.name,
            "method_id": self._method_id(),
            "implementation": "training_free_lesion_constrained_flow_matching",
            "generator_finetuning": False,
            "trainable_generator_parameters": 0,
            "registration_mode": "frozen_pretrained_assets_no_training",
            "selection_stage": "official",
            "selection_split": "not_applicable",
            "frozen_prior": self._prior_name(),
            "frozen_prior_root": str(prior_root),
            "frozen_prior_weight_root": str(self._maisi_weight_root()) if self._prior_name() != "ctflow" and self._maisi_weight_root() else "",
            "num_train_samples_seen": 0,
            "network_parts": [
                "frozen_ct_flow_prior",
                "frozen_encoder_decoder",
                "image_space_background_energy",
                "image_space_hu_histogram_energy",
                "official_prior_candidate_endpoint",
                "image_space_output_projection",
            ]
            + (
                [
                    "decoder_backpropagated_latent_endpoint_refinement",
                    "per_timestep_flow_trajectory_reprojection",
                ]
                if self._uses_latent_stepwise_solver()
                else []
            ),
            "solver_fidelity": self._solver_fidelity(),
            "latent_endpoint_solver": self._uses_latent_stepwise_solver(),
            "trajectory_reprojection_status": self._trajectory_reprojection_status(),
            "hyperparameters": {
                "lambda_bg": float(self.method_config.get("lambda_bg", 1.0)),
                "lambda_hu": float(self.method_config.get("lambda_hu", 1.0)),
                "sampling_steps": int(self.method_config.get("sampling_steps", 16)),
                "refinement_steps": int(self.method_config.get("refinement_steps", 4)),
                "boundary_dilation": int(self.method_config.get("boundary_dilation", 2)),
                "boundary_feather_sigma": float(self.method_config.get("boundary_feather_sigma", 1.0)),
                "lesion_smoothing_sigma": float(self.method_config.get("lesion_smoothing_sigma", 0.0)),
                "boundary_reference_blend": float(self.method_config.get("boundary_reference_blend", 0.0)),
                "boundary_reference_width": int(self.method_config.get("boundary_reference_width", 1)),
                "final_lesion_smoothing_sigma": float(
                    self.method_config.get("final_lesion_smoothing_sigma", 0.0)
                ),
                "maisi_latent_refinement_accept_if_improved": bool(
                    self.method_config.get("maisi_latent_refinement_accept_if_improved", True)
                ),
                "maisi_latent_refinement_min_delta": float(
                    self.method_config.get("maisi_latent_refinement_min_delta", 0.0)
                ),
                "maisi_latent_prox_eta": float(self.method_config.get("maisi_latent_prox_eta", 10.0)),
                "maisi_stepwise_flow_solver": bool(self.method_config.get("maisi_stepwise_flow_solver", False)),
                "maisi_stepwise_refinement_steps": int(
                    self.method_config.get(
                        "maisi_stepwise_refinement_steps",
                        self.method_config.get("maisi_latent_refinement_steps", 0),
                    )
                ),
                "maisi_stepwise_refinement_every": int(
                    self.method_config.get("maisi_stepwise_refinement_every", 1)
                ),
                "maisi_trajectory_alpha_search": bool(self.method_config.get("maisi_trajectory_alpha_search", False)),
                "maisi_trajectory_alpha_search_alphas": list(
                    self.method_config.get("maisi_trajectory_alpha_search_alphas", [0.75, 0.875, 1.0])
                ),
                "maisi_trajectory_alpha_bg_mae_scale": float(
                    self.method_config.get("maisi_trajectory_alpha_bg_mae_scale", 400.0)
                ),
                "local_contrast_target": float(self.method_config.get("local_contrast_target", 0.0)),
                "local_contrast_strength": float(self.method_config.get("local_contrast_strength", 0.0)),
                "local_contrast_shell_width": int(self.method_config.get("local_contrast_shell_width", 2)),
                "local_contrast_max_shift": float(self.method_config.get("local_contrast_max_shift", 120.0)),
                "perilesion_shell_target_mean": float(self.method_config.get("perilesion_shell_target_mean", -700.0)),
                "perilesion_shell_strength": float(self.method_config.get("perilesion_shell_strength", 0.0)),
                "perilesion_shell_width": int(self.method_config.get("perilesion_shell_width", 1)),
                "perilesion_shell_max_shift": float(self.method_config.get("perilesion_shell_max_shift", 120.0)),
                "projection_training_texture_rank_mode": str(
                    self.method_config.get(
                        "projection_training_texture_rank_mode", "single_scale"
                    )
                ),
                "projection_training_texture_sigmas": list(
                    self.method_config.get(
                        "projection_training_texture_sigmas", [0.6, 1.2, 2.4]
                    )
                ),
                "projection_training_texture_band_weights": list(
                    self.method_config.get(
                        "projection_training_texture_band_weights",
                        [0.55, 0.30, 0.15],
                    )
                ),
                "projection_training_texture_detail_gain": float(
                    self.method_config.get(
                        "projection_training_texture_detail_gain", 1.0
                    )
                ),
                "texture_transport_interpolation_order": int(
                    self.method_config.get("texture_transport_interpolation_order", 1)
                ),
                "image_texture_gradient_transport_strength": float(
                    self.method_config.get(
                        "image_texture_gradient_transport_strength", 0.0
                    )
                ),
            },
        }
        if (
            metadata["hyperparameters"]["projection_training_texture_rank_mode"]
            == "multiscale_laplacian"
        ):
            metadata["network_parts"].append(
                "train_exemplar_multiscale_texture_phenotype_energy"
            )
        if float(
            self.method_config.get("image_texture_residual_transfer_strength", 0.0)
        ) > 0:
            metadata["network_parts"].append(
                "train_exemplar_multiscale_3d_laplacian_residual_transfer"
            )
        if float(
            self.method_config.get("image_texture_phase_transfer_strength", 0.0)
        ) > 0:
            metadata["network_parts"].append(
                "train_exemplar_shape_aligned_3d_phase_transport"
            )
        if float(
            self.method_config.get("image_texture_gradient_transport_strength", 0.0)
        ) > 0:
            metadata["network_parts"].append(
                "screened_poisson_train_donor_gradient_transport"
            )
        save_json(metadata, output_dir / "best.ckpt")
        print(f"[{self._method_id()}] wrote training-free metadata to {output_dir / 'best.ckpt'}")

    register_pretrained = train

    def _refine_destination(self, normal, mask, target_hist, rng):
        mask = np.asarray(mask) > 0
        edit_mask = _dilate(mask, int(self.method_config.get("boundary_dilation", 2)))
        alpha = _smooth(edit_mask.astype(np.float32), float(self.method_config.get("boundary_feather_sigma", 1.0)))
        alpha = np.maximum(alpha, mask.astype(np.float32))
        alpha = _apply_background_weight(alpha, mask, self.method_config.get("lambda_bg", 1.0))
        image = np.asarray(normal, dtype=np.float32).copy()
        values = _sample_histogram(target_hist, int(mask.sum()), rng)
        texture = rng.normal(0.0, float(self.method_config.get("texture_noise_std", 18.0)), size=int(mask.sum())).astype(np.float32)
        lesion = image.copy()
        lesion_values = np.clip(values + texture, HU_MIN, HU_MAX)
        lesion[mask] = lesion_values
        steps = max(1, int(self.method_config.get("refinement_steps", 4)))
        lambda_hu = float(self.method_config.get("lambda_hu", 1.0))
        for _ in range(steps):
            actual = lesion[mask]
            target = _sample_histogram(target_hist, int(mask.sum()), rng)
            order = np.argsort(actual)
            corrected = actual.copy()
            corrected[order] = np.sort(target)
            lesion[mask] = (1.0 - min(lambda_hu, 1.0)) * actual + min(lambda_hu, 1.0) * corrected
        out = image * (1.0 - alpha) + lesion * alpha
        out[~edit_mask] = image[~edit_mask]
        band = boundary_band(mask)
        out[band & ~mask] = 0.75 * image[band & ~mask] + 0.25 * out[band & ~mask]
        out = _smooth_hist_preserving(out, mask, float(self.method_config.get("lesion_smoothing_sigma", 0.0)))
        return out.astype(np.float32), edit_mask

    def _maisi_roi_mapping(self, roi_shape, out_shape):
        source_spacing = tuple(
            float(v) for v in (self.data_config.get("preprocess") or {}).get("spacing", [1.0, 1.0, 1.0])
        )
        if len(source_spacing) != 3:
            source_spacing = (1.0, 1.0, 1.0)
        maisi_spacing = tuple(float(v) for v in self.method_config.get("maisi_spacing", [1.5, 1.5, 4.0]))
        embedded_shape = tuple(
            max(2, min(int(out_shape[axis]), int(round(roi_shape[axis] * source_spacing[axis] / maisi_spacing[axis]))))
            for axis in range(3)
        )
        center_fraction = tuple(
            float(v) for v in self.method_config.get("maisi_roi_center_fraction", [0.66, 0.55, 0.50])
        )
        center = tuple(int(round((out_shape[axis] - 1) * center_fraction[axis])) for axis in range(3))
        starts = []
        stops = []
        for axis in range(3):
            start = center[axis] - embedded_shape[axis] // 2
            start = max(0, min(int(out_shape[axis]) - embedded_shape[axis], start))
            starts.append(start)
            stops.append(start + embedded_shape[axis])
        return {
            "source_shape": tuple(int(v) for v in roi_shape),
            "output_shape": tuple(int(v) for v in out_shape),
            "embedded_shape": embedded_shape,
            "slices": tuple(slice(starts[axis], stops[axis]) for axis in range(3)),
            "source_spacing": source_spacing,
            "maisi_spacing": maisi_spacing,
            "center_fraction": center_fraction,
        }

    @staticmethod
    def _maisi_embed_roi(array, mapping, fill_value, order=1):
        canvas = np.full(mapping["output_shape"], fill_value, dtype=np.float32)
        resized = _resize_array(np.asarray(array, dtype=np.float32), mapping["embedded_shape"], order=order)
        canvas[mapping["slices"]] = resized
        return canvas

    @staticmethod
    def _maisi_embed_mask(mask, mapping):
        source = np.asarray(mask) > 0
        patch = np.zeros(mapping["embedded_shape"], dtype=bool)
        coords = np.argwhere(source)
        if coords.size:
            source_shape = np.asarray(source.shape, dtype=np.float64)
            embedded_shape = np.asarray(mapping["embedded_shape"], dtype=np.int64)
            mapped = np.floor((coords.astype(np.float64) + 0.5) * embedded_shape / source_shape).astype(np.int64)
            mapped = np.clip(mapped, 0, embedded_shape - 1)
            patch[tuple(mapped.T)] = True
        canvas = np.zeros(mapping["output_shape"], dtype=bool)
        canvas[mapping["slices"]] = patch
        return canvas

    def _maisi_extract_roi(self, array, output_shape):
        mapping = getattr(self, "_last_maisi_roi_mapping", None)
        if mapping is None:
            return _resize_array(np.asarray(array, dtype=np.float32), output_shape, order=1)
        crop = np.asarray(array, dtype=np.float32)[mapping["slices"]]
        return _resize_array(crop, output_shape, order=1)

    @staticmethod
    def _maisi_mapping_metadata(mapping):
        if mapping is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "source_shape": list(mapping["source_shape"]),
            "output_shape": list(mapping["output_shape"]),
            "embedded_shape": list(mapping["embedded_shape"]),
            "embedded_slices": [
                [int(axis_slice.start), int(axis_slice.stop)] for axis_slice in mapping["slices"]
            ],
            "source_spacing_mm": list(mapping["source_spacing"]),
            "maisi_spacing_mm": list(mapping["maisi_spacing"]),
            "center_fraction": list(mapping["center_fraction"]),
        }

    def _maisi_label_condition(self, mask, normal=None):
        out_shape = tuple(int(v) for v in self.method_config.get("maisi_output_size", [256, 256, 128]))
        label_lung = int(self.method_config.get("maisi_label_lung", 28))
        label_body = int(self.method_config.get("maisi_label_background", 200))
        label_tumor = int(self.method_config.get("maisi_label_lung_tumor", 23))
        if bool(self.method_config.get("maisi_physical_roi_embedding", False)):
            mapping = self._maisi_roi_mapping(np.asarray(mask).shape, out_shape)
            self._last_maisi_roi_mapping = mapping
            mask_rs = self._maisi_embed_mask(mask, mapping)
            x, y, z = np.ogrid[-1.0:1.0:complex(out_shape[0]), -1.0:1.0:complex(out_shape[1]), -1.0:1.0:complex(out_shape[2])]
            lung_left = ((x + 0.32) / 0.24) ** 2 + ((y - 0.10) / 0.34) ** 2 + (z / 0.44) ** 2 <= 1.0
            lung_right = ((x - 0.32) / 0.24) ** 2 + ((y - 0.10) / 0.34) ** 2 + (z / 0.44) ** 2 <= 1.0
            lung_mask = lung_left | lung_right | _dilate(mask_rs, 2)
            label = np.full(out_shape, label_body, dtype=np.int64)
            label[lung_mask] = label_lung
            label[mask_rs] = label_tumor
            return label
        self._last_maisi_roi_mapping = None
        mask_rs = _resize_array((np.asarray(mask) > 0).astype(np.float32), out_shape, order=0) > 0
        use_source_anatomy = bool(self.method_config.get("maisi_source_anatomy_condition", False))
        label = np.full(out_shape, label_body, dtype=np.int64)
        lung_mask = None
        if use_source_anatomy and normal is not None:
            normal_rs = _resize_array(np.asarray(normal, dtype=np.float32), out_shape, order=1)
            lower = float(self.method_config.get("maisi_lung_hu_min", -990.0))
            upper = float(self.method_config.get("maisi_lung_hu_max", -250.0))
            lung_mask = (normal_rs >= lower) & (normal_rs <= upper)
            try:
                from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes

                lung_mask = binary_closing(lung_mask, iterations=2)
                lung_mask = binary_fill_holes(lung_mask)
                lung_mask = binary_dilation(lung_mask, iterations=2)
            except Exception:
                lung_mask = _dilate(lung_mask, 2)
            lesion_context = _dilate(mask_rs, int(self.method_config.get("maisi_lesion_lung_margin", 6)))
            lung_mask |= lesion_context
            lung_fraction = float(np.mean(lung_mask))
            min_fraction = float(self.method_config.get("maisi_lung_fraction_min", 0.02))
            max_fraction = float(self.method_config.get("maisi_lung_fraction_max", 0.995))
            if not (min_fraction <= lung_fraction <= max_fraction):
                lung_mask = None
        if lung_mask is None:
            lung_mask = np.ones(out_shape, dtype=bool)
            pad = max(2, min(out_shape) // 32)
            lung_mask[:pad, :, :] = False
            lung_mask[-pad:, :, :] = False
            lung_mask[:, :pad, :] = False
            lung_mask[:, -pad:, :] = False
            lung_mask[:, :, :pad] = False
            lung_mask[:, :, -pad:] = False
        label[lung_mask] = label_lung
        label[mask_rs] = label_tumor
        return label

    def _maisi_soft_hist_loss_decoded(self, decoded_unit, lesion_mask, target_hist):
        import torch

        values = decoded_unit[lesion_mask > 0.25]
        if values.numel() == 0 or target_hist is None:
            return decoded_unit.sum() * 0.0
        hu = (values * 2000.0 - 1000.0).clamp(HU_MIN, HU_MAX)
        bins = int(self.method_config.get("histogram_bins", 40))
        centers = torch.linspace(HU_MIN, HU_MAX, bins, device=hu.device, dtype=hu.dtype)
        width = (HU_MAX - HU_MIN) / float(bins)
        soft = torch.relu(1.0 - torch.abs(hu[:, None] - centers[None, :]) / width)
        pred = soft.sum(dim=0)
        pred = pred / (pred.sum() + 1.0e-8)
        target = torch.as_tensor(target_hist, dtype=pred.dtype, device=pred.device)
        target = target / (target.sum() + 1.0e-8)
        mid = 0.5 * (pred + target)
        return 0.5 * (pred * ((pred + 1.0e-8) / (mid + 1.0e-8)).log()).sum() + 0.5 * (
            target * ((target + 1.0e-8) / (mid + 1.0e-8)).log()
        ).sum()

    def _encode_maisi_source_latent(self, normal, output_size):
        import torch
        from monai.inferers.inferer import SlidingWindowInferer

        from scripts.utils import dynamic_infer

        backend = self._load_maisi_rflow_backend()
        autoencoder = backend["autoencoder"]
        scale_factor = backend["scale_factor"]
        cfg = backend["cfg"]
        device = backend["device"]
        normal_rs = _resize_array(np.asarray(normal, dtype=np.float32), tuple(int(v) for v in output_size), order=1)
        normal_t = torch.tensor(normal_rs, dtype=torch.float32, device=device)[None, None]
        normal_unit = _unit_from_hu_torch(normal_t, -1000.0, 1000.0)
        inferer = SlidingWindowInferer(
            roi_size=list(cfg.autoencoder_sliding_window_infer_size),
            sw_batch_size=1,
            progress=False,
            mode="gaussian",
            overlap=float(cfg.autoencoder_sliding_window_infer_overlap),
            sw_device=device,
            device=device,
        )
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            z = dynamic_infer(inferer, autoencoder.encode_stage_2_inputs, normal_unit).detach().float()
        return z * scale_factor

    def _maisi_trajectory_alpha_search(self, source_latent, endpoint, normal, label, target_hist):
        import torch
        import torch.nn.functional as F
        from monai.inferers.inferer import SlidingWindowInferer

        from scripts.utils import dynamic_infer
        from scripts.utils_infer import ReconModel

        if not bool(self.method_config.get("maisi_trajectory_alpha_search", False)):
            return endpoint, {"status": "skipped", "enabled": False}
        backend = self._load_maisi_rflow_backend()
        autoencoder = backend["autoencoder"]
        scale_factor = backend["scale_factor"]
        cfg = backend["cfg"]
        device = backend["device"]
        z0 = source_latent.detach().float()
        z1 = endpoint.detach().float()
        if z0.shape != z1.shape:
            return endpoint, {
                "status": "shape_mismatch",
                "enabled": True,
                "source_shape": [int(v) for v in z0.shape],
                "endpoint_shape": [int(v) for v in z1.shape],
            }
        label_tensor = label.as_tensor() if hasattr(label, "as_tensor") else label
        lesion_label = int(self.method_config.get("maisi_label_lung_tumor", 23))
        output_size = tuple(int(v) for v in label_tensor.shape[-3:])
        lesion_mask = label_tensor.detach().cpu().squeeze().long() == lesion_label
        normal_rs = np.asarray(_resize_array(np.asarray(normal, dtype=np.float32), output_size, order=1), dtype=np.float32)
        normal_t = torch.tensor(np.ascontiguousarray(normal_rs, dtype=np.float32), dtype=torch.float32)
        alpha_values = self.method_config.get("maisi_trajectory_alpha_search_alphas", [0.75, 0.875, 1.0])
        min_alpha = float(self.method_config.get("maisi_trajectory_alpha_search_min_alpha", 0.75))
        alphas = sorted({float(value) for value in alpha_values if float(value) >= min_alpha})
        if 1.0 not in alphas:
            alphas.append(1.0)
        lambda_hu = float(self.method_config.get("maisi_trajectory_alpha_lambda_hu", self.method_config.get("lambda_hu", 1.0)))
        lambda_bg = float(self.method_config.get("maisi_trajectory_alpha_lambda_bg", self.method_config.get("lambda_bg", 1.0)))
        lambda_prox = float(self.method_config.get("maisi_trajectory_alpha_lambda_prox", 0.02))
        bg_scale = max(1.0, float(self.method_config.get("maisi_trajectory_alpha_bg_mae_scale", 400.0)))
        min_delta = float(self.method_config.get("maisi_trajectory_alpha_min_delta", 0.0))
        require_hist_improvement = bool(self.method_config.get("maisi_trajectory_alpha_require_hist_improvement", True))
        hist_min_delta = float(self.method_config.get("maisi_trajectory_alpha_hist_min_delta", 0.0))
        recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
        inferer = SlidingWindowInferer(
            roi_size=list(cfg.autoencoder_sliding_window_infer_size),
            sw_batch_size=1,
            progress=False,
            mode="gaussian",
            overlap=float(cfg.autoencoder_sliding_window_infer_overlap),
            sw_device=device,
            device=torch.device("cpu"),
        )
        rows = []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            for alpha in alphas:
                z = z0 + float(alpha) * (z1 - z0)
                decoded_unit = dynamic_infer(inferer, recon_model, z).float().cpu()
                decoded_hu = decoded_unit.squeeze().float() * 2000.0 - 1000.0
                if tuple(decoded_hu.shape) != tuple(normal_t.shape):
                    decoded_hu = F.interpolate(
                        decoded_hu[None, None],
                        size=tuple(normal_t.shape),
                        mode="trilinear",
                        align_corners=False,
                    ).squeeze()
                lesion_values = decoded_hu[lesion_mask] if bool(lesion_mask.any().item()) else decoded_hu.reshape(-1)
                bg = ~lesion_mask
                hist_jsd = 0.0
                if target_hist is not None and int(lesion_values.numel()) > 0:
                    bins = int(self.method_config.get("histogram_bins", 40))
                    pred = torch.histc(lesion_values.float(), bins=bins, min=HU_MIN, max=HU_MAX)
                    pred = pred / (pred.sum() + 1.0e-8)
                    target = torch.tensor(np.asarray(target_hist, dtype=np.float32), dtype=pred.dtype)
                    target = target / (target.sum() + 1.0e-8)
                    mid = 0.5 * (pred + target)
                    hist_jsd = float(
                        (
                            0.5 * (pred * ((pred + 1.0e-8) / (mid + 1.0e-8)).log()).sum()
                            + 0.5 * (target * ((target + 1.0e-8) / (mid + 1.0e-8)).log()).sum()
                        ).item()
                    )
                bg_mae = float((decoded_hu[bg] - normal_t[bg]).abs().mean().item()) if bool(bg.any().item()) else 0.0
                objective = lambda_hu * hist_jsd + lambda_bg * (bg_mae / bg_scale) + lambda_prox * ((1.0 - float(alpha)) ** 2)
                rows.append(
                    {
                        "alpha": float(alpha),
                        "objective": float(objective),
                        "target_hist_jsd": float(hist_jsd),
                        "bg_mae_hu": float(bg_mae),
                        "lesion_mean_hu": float(lesion_values.mean().item()) if int(lesion_values.numel()) > 0 else "",
                        "lesion_std_hu": float(lesion_values.std(unbiased=False).item()) if int(lesion_values.numel()) > 0 else "",
                    }
                )
        if not rows:
            return endpoint, {"status": "no_candidates", "enabled": True}
        endpoint_row = next((row for row in rows if abs(float(row["alpha"]) - 1.0) < 1.0e-6), rows[-1])
        best = min(rows, key=lambda row: (float(row["objective"]), -float(row["alpha"])))
        delta_objective = float(best["objective"]) - float(endpoint_row["objective"])
        delta_hist_jsd = float(best["target_hist_jsd"]) - float(endpoint_row["target_hist_jsd"])
        hist_ok = (not require_hist_improvement) or (delta_hist_jsd <= -hist_min_delta)
        accepted = delta_objective <= -min_delta and hist_ok and float(best["alpha"]) < 1.0
        selected_alpha = float(best["alpha"]) if accepted else 1.0
        selected = z0 + selected_alpha * (z1 - z0)
        return selected.to(endpoint.dtype), {
            "status": "pass",
            "enabled": True,
            "accepted": bool(accepted),
            "acceptance_status": "accepted_lower_objective" if accepted else "kept_endpoint",
            "selected_alpha": float(selected_alpha),
            "endpoint_objective": float(endpoint_row["objective"]),
            "best_objective": float(best["objective"]),
            "delta_objective": float(delta_objective),
            "endpoint_target_hist_jsd": float(endpoint_row["target_hist_jsd"]),
            "best_target_hist_jsd": float(best["target_hist_jsd"]),
            "delta_target_hist_jsd": float(delta_hist_jsd),
            "hist_improvement_required": bool(require_hist_improvement),
            "hist_improvement_ok": bool(hist_ok),
            "hist_min_delta": float(hist_min_delta),
            "min_delta": float(min_delta),
            "lambda_hu": float(lambda_hu),
            "lambda_bg": float(lambda_bg),
            "lambda_prox": float(lambda_prox),
            "bg_mae_scale": float(bg_scale),
            "rows": rows,
            "note": "Partial straight-line latent alpha search; not a full Eq.18-21 trajectory re-projection solver.",
        }

    def _prepare_maisi_latent_refinement(
        self,
        endpoint,
        normal,
        label,
        texture_target=None,
        texture_masks=None,
        texture_channel_names=None,
        texture_channel_weights=None,
    ):
        import torch
        import torch.nn.functional as F

        backend = self._load_maisi_rflow_backend()
        autoencoder = backend["autoencoder"]
        scale_factor = backend["scale_factor"]
        device = backend["device"]
        for param in autoencoder.parameters():
            param.requires_grad_(False)
        label_tensor = label.as_tensor() if hasattr(label, "as_tensor") else label
        lesion_label = int(self.method_config.get("maisi_label_lung_tumor", 23))
        lesion_mask = (label_tensor.to(device).float() == float(lesion_label)).float()
        output_size = tuple(int(v) for v in label_tensor.shape[-3:])
        mapping = getattr(self, "_last_maisi_roi_mapping", None)
        if bool(self.method_config.get("maisi_physical_roi_embedding", False)) and mapping is not None:
            normal_rs = self._maisi_embed_roi(normal, mapping, HU_MIN, order=1)
            valid_np = np.zeros(output_size, dtype=np.float32)
            valid_np[mapping["slices"]] = 1.0
            texture_target_rs = (
                self._maisi_embed_roi(texture_target, mapping, HU_MIN, order=1)
                if texture_target is not None
                else None
            )
            texture_masks_rs = (
                np.stack(
                    [self._maisi_embed_mask(channel, mapping) for channel in np.asarray(texture_masks)],
                    axis=0,
                ).astype(np.float32)
                if texture_masks is not None
                else None
            )
        else:
            normal_rs = _resize_array(np.asarray(normal, dtype=np.float32), output_size, order=1)
            valid_np = np.ones(output_size, dtype=np.float32)
            texture_target_rs = (
                _resize_array(np.asarray(texture_target, dtype=np.float32), output_size, order=1)
                if texture_target is not None
                else None
            )
            texture_masks_rs = (
                np.stack(
                    [
                        _resize_array(np.asarray(channel, dtype=np.float32), output_size, order=0)
                        for channel in np.asarray(texture_masks)
                    ],
                    axis=0,
                ).astype(np.float32)
                if texture_masks is not None
                else None
            )
        normal_t = torch.tensor(normal_rs, dtype=torch.float32, device=device)[None, None]
        normal_unit = _unit_from_hu_torch(normal_t, -1000.0, 1000.0)
        valid_mask = torch.tensor(valid_np, dtype=torch.float32, device=device)[None, None]
        texture_target_unit = None
        if texture_target_rs is not None:
            texture_target_t = torch.tensor(
                texture_target_rs, dtype=torch.float32, device=device
            )[None, None]
            texture_target_unit = _unit_from_hu_torch(texture_target_t, -1000.0, 1000.0)
        texture_mask_tensor = (
            torch.tensor(texture_masks_rs, dtype=torch.float32, device=device)[None]
            if texture_masks_rs is not None
            else lesion_mask
        )
        z_full = endpoint.detach().float()
        spatial = tuple(int(v) for v in z_full.shape[-3:])
        crop_mode = bool(self.method_config.get("maisi_latent_refinement_crop", True))
        refinement_mask_resolved = bool(lesion_mask.sum() > 0)
        crop_slices = (slice(None), slice(None), slice(None))
        decoded_crop_slices = (slice(None), slice(None), slice(None))
        if crop_mode and lesion_mask.sum() > 0:
            latent_mask = F.interpolate(lesion_mask, size=spatial, mode="trilinear", align_corners=False) > 0.05
            coords = latent_mask[0, 0].nonzero(as_tuple=False)
            if coords.numel() > 0:
                pad = int(self.method_config.get("maisi_latent_refinement_crop_padding", 4))
                min_size = int(self.method_config.get("maisi_latent_refinement_min_crop", 16))
                starts = []
                stops = []
                for axis in range(3):
                    lo = int(coords[:, axis].min().item()) - pad
                    hi = int(coords[:, axis].max().item()) + pad + 1
                    if hi - lo < min_size:
                        extra = min_size - (hi - lo)
                        lo -= extra // 2
                        hi += extra - extra // 2
                    lo = max(0, lo)
                    hi = min(spatial[axis], hi)
                    if hi - lo < min_size:
                        lo = max(0, hi - min_size)
                        hi = min(spatial[axis], lo + min_size)
                    starts.append(lo)
                    stops.append(hi)
                crop_slices = tuple(slice(starts[axis], stops[axis]) for axis in range(3))
                scale = [output_size[axis] / float(spatial[axis]) for axis in range(3)]
                decoded_crop_slices = tuple(
                    slice(int(round(starts[axis] * scale[axis])), int(round(stops[axis] * scale[axis])))
                    for axis in range(3)
                )
            else:
                refinement_mask_resolved = False
        return {
            "autoencoder": autoencoder,
            "scale_factor": scale_factor,
            "device": device,
            "lesion_mask": lesion_mask,
            "normal_unit": normal_unit,
            "valid_mask": valid_mask,
            "texture_target_unit": texture_target_unit,
            "texture_masks": texture_mask_tensor,
            "texture_channel_names": list(texture_channel_names or ["lesion"]),
            "texture_channel_weights": list(texture_channel_weights or [1.0]),
            "output_size": output_size,
            "spatial": spatial,
            "crop_mode": crop_mode,
            "refinement_mask_resolved": refinement_mask_resolved,
            "crop_slices": crop_slices,
            "decoded_crop_slices": decoded_crop_slices,
        }

    def _refine_maisi_latent_endpoint(
        self,
        endpoint,
        normal,
        label,
        target_hist,
        prepared=None,
        steps_override=None,
        texture_target=None,
        texture_masks=None,
        texture_channel_names=None,
        texture_channel_weights=None,
        texture_weight_scale=1.0,
        condition_metadata=None,
    ):
        import torch
        import torch.nn.functional as F

        steps = int(
            self.method_config.get("maisi_latent_refinement_steps", 0)
            if steps_override is None
            else steps_override
        )
        if steps <= 0:
            return endpoint, {"status": "skipped", "steps": 0, "losses": []}
        prepared = prepared or self._prepare_maisi_latent_refinement(
            endpoint,
            normal,
            label,
            texture_target=texture_target,
            texture_masks=texture_masks,
            texture_channel_names=texture_channel_names,
            texture_channel_weights=texture_channel_weights,
        )
        if not bool(prepared.get("refinement_mask_resolved", True)):
            return endpoint, {
                "status": "skipped_unresolved_lesion_grid",
                "steps": 0,
                "accepted": False,
                "losses": [],
                "note": "The physical lesion is smaller than the frozen MAISI latent grid; image-space energies remain active.",
            }
        autoencoder = prepared["autoencoder"]
        scale_factor = prepared["scale_factor"]
        device = prepared["device"]
        lesion_mask = prepared["lesion_mask"]
        normal_unit = prepared["normal_unit"]
        valid_mask = prepared["valid_mask"]
        texture_target_unit = prepared.get("texture_target_unit")
        texture_mask_tensor = prepared.get("texture_masks")
        texture_channel_names = list(prepared.get("texture_channel_names") or ["lesion"])
        texture_channel_weights = list(prepared.get("texture_channel_weights") or [1.0])
        posterior_energies = self._posterior_energies("maisi_rflow")
        condition_metadata = dict(condition_metadata or {})
        output_size = prepared["output_size"]
        spatial = prepared["spatial"]
        crop_mode = prepared["crop_mode"]
        crop_slices = prepared["crop_slices"]
        decoded_crop_slices = prepared["decoded_crop_slices"]
        if tuple(int(v) for v in endpoint.shape[-3:]) != tuple(int(v) for v in spatial):
            raise ValueError(
                "MAISI refinement context shape mismatch: "
                f"endpoint={list(endpoint.shape[-3:])}, prepared={list(spatial)}"
            )
        for param in autoencoder.parameters():
            param.requires_grad_(False)
        z_full = endpoint.detach().float()
        z0 = z_full[(slice(None), slice(None)) + crop_slices].contiguous()
        z = z0.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=float(self.method_config.get("maisi_latent_refinement_lr", 1.0e-3)))
        lambda_hu = float(self.method_config.get("maisi_latent_lambda_hu", self.method_config.get("lambda_hu", 1.0)))
        lambda_bg = float(self.method_config.get("maisi_latent_lambda_bg", self.method_config.get("lambda_bg", 1.0)))
        lambda_boundary = float(self.method_config.get("maisi_latent_lambda_boundary", 0.0))
        lambda_texture_base = float(self.method_config.get("maisi_latent_lambda_texture", 0.0))
        texture_weight_scale = float(texture_weight_scale)
        if lambda_texture_base < 0 or not 0.0 <= texture_weight_scale <= 1.0:
            raise ValueError(
                "MAISI texture weight and schedule scale must be non-negative, with scale <= 1; "
                f"got weight={lambda_texture_base}, scale={texture_weight_scale}"
            )
        lambda_texture = lambda_texture_base * texture_weight_scale
        if lambda_texture > 0 and texture_target_unit is None:
            raise ValueError("maisi_latent_lambda_texture requires verified train texture guidance")
        prox_eta_value = self.method_config.get("maisi_latent_prox_eta")
        if prox_eta_value is not None:
            prox_eta = float(prox_eta_value)
            if prox_eta <= 0:
                raise ValueError(f"maisi_latent_prox_eta must be positive, got {prox_eta}")
            lambda_prox = 1.0 / (2.0 * prox_eta)
        else:
            lambda_prox = float(self.method_config.get("maisi_latent_lambda_prox", 0.05))
            prox_eta = 1.0 / (2.0 * lambda_prox) if lambda_prox > 0 else float("inf")
        losses = []
        autocast_enabled = device.type == "cuda"
        lesion_mask_crop = lesion_mask[(slice(None), slice(None)) + decoded_crop_slices]
        normal_unit_crop = normal_unit[(slice(None), slice(None)) + decoded_crop_slices]
        valid_mask_crop = valid_mask[(slice(None), slice(None)) + decoded_crop_slices]
        texture_target_crop = (
            texture_target_unit[(slice(None), slice(None)) + decoded_crop_slices]
            if texture_target_unit is not None
            else None
        )
        texture_masks_crop = (
            texture_mask_tensor[(slice(None), slice(None)) + decoded_crop_slices]
            if texture_mask_tensor is not None
            else lesion_mask_crop
        )

        def _loss_terms(current_z):
            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                decoded = autoencoder.decode_stage_2_outputs(current_z / scale_factor).float()
                decoded = decoded.clamp(0.0, 1.0)
                lesion_mask_loss = F.interpolate(lesion_mask_crop, size=decoded.shape[-3:], mode="trilinear", align_corners=False)
                normal_unit_loss = F.interpolate(normal_unit_crop, size=decoded.shape[-3:], mode="trilinear", align_corners=False)
                valid_mask_loss = F.interpolate(valid_mask_crop, size=decoded.shape[-3:], mode="nearest")
                hist_loss = self._maisi_soft_hist_loss_decoded(decoded, lesion_mask_loss, target_hist)
                bg_mask = ((1.0 - lesion_mask_loss).clamp(0.0, 1.0) * valid_mask_loss).clamp(0.0, 1.0)
                bg_loss = (((decoded - normal_unit_loss) ** 2) * bg_mask).sum() / (bg_mask.sum() + 1.0)
                hard_mask = (lesion_mask_loss > 0.5).float()
                eroded = 1.0 - F.max_pool3d(1.0 - hard_mask, kernel_size=3, stride=1, padding=1)
                inner_boundary = (hard_mask - eroded).clamp(0.0, 1.0)
                boundary_loss = (((decoded - normal_unit_loss) ** 2) * inner_boundary).sum() / (
                    inner_boundary.sum() + 1.0
                )
                if lambda_texture > 0:
                    texture_loss = _multiscale_3d_texture_energy(
                        decoded,
                        texture_target_crop,
                        texture_masks_crop,
                        scales=self.method_config.get("maisi_latent_texture_scales", [1, 2, 4]),
                        band_weights=self.method_config.get(
                            "maisi_latent_texture_band_weights", [0.55, 0.30, 0.15]
                        ),
                        autocorr_lags=self.method_config.get(
                            "maisi_latent_texture_autocorr_lags", [1, 2]
                        ),
                        core_erosion=int(
                            self.method_config.get("maisi_latent_texture_core_erosion", 1)
                        ),
                        min_voxels=int(
                            self.method_config.get("maisi_latent_texture_min_voxels", 8)
                        ),
                        channel_weights=texture_channel_weights,
                        local_variance_weight=float(
                            self.method_config.get(
                                "maisi_latent_texture_local_variance_weight", 0.0
                            )
                        ),
                        gram_weight=float(
                            self.method_config.get("maisi_latent_texture_gram_weight", 0.0)
                        ),
                        patch_swd_weight=float(
                            self.method_config.get(
                                "maisi_latent_texture_patch_swd_weight", 0.0
                            )
                        ),
                        patch_swd_sizes=self.method_config.get(
                            "maisi_latent_texture_patch_swd_sizes", [3]
                        ),
                        patch_swd_projections=int(
                            self.method_config.get(
                                "maisi_latent_texture_patch_swd_projections", 12
                            )
                        ),
                        patch_swd_min_support=float(
                            self.method_config.get(
                                "maisi_latent_texture_patch_swd_min_support", 0.25
                            )
                        ),
                        patch_swd_max_samples=int(
                            self.method_config.get(
                                "maisi_latent_texture_patch_swd_max_samples", 2048
                            )
                        ),
                        patch_swd_seed=int(
                            self.method_config.get(
                                "maisi_latent_texture_patch_swd_seed", 1729
                            )
                        ),
                    )
                else:
                    texture_loss = decoded.sum() * 0.0
                custom_loss, custom_terms = evaluate_posterior_energies(
                    posterior_energies,
                    PosteriorEnergyContext(
                        decoded=decoded,
                        lesion_mask=lesion_mask_loss,
                        source_image=normal_unit_loss,
                        background_mask=bg_mask,
                        target_histogram=torch.as_tensor(
                            target_hist, dtype=decoded.dtype, device=decoded.device
                        ),
                        prior="maisi_rflow",
                        metadata={
                            "condition": condition_metadata,
                            "valid_mask": valid_mask_loss,
                            "texture_target": texture_target_crop,
                            "texture_masks": texture_masks_crop,
                        },
                    ),
                )
                prox_loss = ((current_z - z0) ** 2).mean()
                loss = (
                    lambda_hu * hist_loss
                    + lambda_bg * bg_loss
                    + lambda_boundary * boundary_loss
                    + lambda_texture * texture_loss
                    + lambda_prox * prox_loss
                    + custom_loss
                )
            return (
                loss,
                hist_loss,
                bg_loss,
                boundary_loss,
                texture_loss,
                prox_loss,
                custom_terms,
            )

        def _loss_dict(current_z):
            with torch.no_grad():
                (
                    loss,
                    hist_loss,
                    bg_loss,
                    boundary_loss,
                    texture_loss,
                    prox_loss,
                    custom_terms,
                ) = _loss_terms(current_z)
            return {
                "loss": float(loss.detach().cpu()),
                "hist": float(hist_loss.detach().cpu()),
                "background": float(bg_loss.detach().cpu()),
                "boundary": float(boundary_loss.detach().cpu()),
                "texture": float(texture_loss.detach().cpu()),
                "prox": float(prox_loss.detach().cpu()),
                "custom": {
                    name: float(value.detach().cpu())
                    for name, value in custom_terms.items()
                },
            }

        initial_eval = _loss_dict(z0.detach())
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            (
                loss,
                hist_loss,
                bg_loss,
                boundary_loss,
                texture_loss,
                prox_loss,
                custom_terms,
            ) = _loss_terms(z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], float(self.method_config.get("maisi_latent_grad_clip", 1.0)))
            optimizer.step()
            losses.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "hist": float(hist_loss.detach().cpu()),
                    "background": float(bg_loss.detach().cpu()),
                    "boundary": float(boundary_loss.detach().cpu()),
                    "texture": float(texture_loss.detach().cpu()),
                    "prox": float(prox_loss.detach().cpu()),
                    "custom": {
                        name: float(value.detach().cpu())
                        for name, value in custom_terms.items()
                    },
                }
            )
        final_eval = _loss_dict(z.detach())
        delta_objective = final_eval["loss"] - initial_eval["loss"]
        min_delta = float(self.method_config.get("maisi_latent_refinement_min_delta", 0.0))
        accept_if_improved = bool(self.method_config.get("maisi_latent_refinement_accept_if_improved", True))
        accepted = (not accept_if_improved) or (delta_objective <= -min_delta)
        accepted_z = z.detach() if accepted else z0.detach()
        refined = z_full.clone()
        refined[(slice(None), slice(None)) + crop_slices] = accepted_z
        def _slice_bounds(slc, limit):
            start = 0 if slc.start is None else int(slc.start)
            stop = int(limit) if slc.stop is None else int(slc.stop)
            return [start, stop]

        return refined.to(endpoint.dtype), {
            "status": "pass",
            "steps": steps,
            "losses": losses,
            "initial_eval": initial_eval,
            "final_eval": final_eval,
            "delta_objective": float(delta_objective),
            "accepted": bool(accepted),
            "acceptance_status": "accepted_improved" if accepted else "rejected_no_improvement",
            "accept_if_improved": bool(accept_if_improved),
            "min_delta": float(min_delta),
            "prox_eta": float(prox_eta),
            "prox_weight": float(lambda_prox),
            "texture_weight": float(lambda_texture),
            "texture_weight_base": float(lambda_texture_base),
            "texture_weight_scale": float(texture_weight_scale),
            "texture_channels": texture_channel_names,
            "posterior_energies": describe_posterior_energies(posterior_energies),
            "boundary_weight": float(lambda_boundary),
            "crop_mode": crop_mode,
            "latent_crop": [_slice_bounds(crop_slices[axis], spatial[axis]) for axis in range(3)],
            "decoded_crop": [_slice_bounds(decoded_crop_slices[axis], output_size[axis]) for axis in range(3)],
        }

    @staticmethod
    def _maisi_rflow_endpoint_reprojection(sample, velocity, timestep, next_timestep, num_train_timesteps):
        """Apply paper Eq. (18) and Eq. (22) in MONAI's reversed RFlow time convention."""
        import torch

        total = float(num_train_timesteps)
        if total <= 0:
            raise ValueError(f"num_train_timesteps must be positive, got {total}")
        scheduler_t = float(torch.as_tensor(timestep).detach().cpu()) / total
        scheduler_next_t = float(torch.as_tensor(next_timestep).detach().cpu()) / total
        forward_t = 1.0 - scheduler_t
        forward_next_t = 1.0 - scheduler_next_t
        sample_f = sample.detach().float()
        velocity_f = velocity.detach().float()
        endpoint_hat = sample_f + scheduler_t * velocity_f
        source_hat = sample_f - forward_t * velocity_f
        prior_next = sample_f + (scheduler_t - scheduler_next_t) * velocity_f
        identity_next = scheduler_next_t * source_hat + forward_next_t * endpoint_hat
        identity_error = float(torch.max(torch.abs(identity_next - prior_next)).detach().cpu())
        return {
            "source_hat": source_hat,
            "endpoint_hat": endpoint_hat,
            "prior_next": prior_next,
            "identity_next": identity_next,
            "identity_max_abs": identity_error,
            "scheduler_t": scheduler_t,
            "scheduler_next_t": scheduler_next_t,
            "forward_t": forward_t,
            "forward_next_t": forward_next_t,
        }

    def _maisi_rflow_candidate_with_endpoint(
        self,
        normal,
        label,
        latent_shape,
        tensors,
        target_hist=None,
        texture_target=None,
        texture_masks=None,
        texture_channel_names=None,
        texture_channel_weights=None,
        condition_metadata=None,
    ):
        import gc

        import torch
        from monai.inferers.inferer import SlidingWindowInferer
        from monai.networks.schedulers import DDPMScheduler, RFlowScheduler

        from scripts.augmentation import remove_tumors
        from scripts.infer_image_from_mask import crop_img_body_mask
        from scripts.utils import binarize_labels, dynamic_infer
        from scripts.utils_infer import ReconModel

        backend = self._load_maisi_rflow_backend()
        cfg = backend["cfg"]
        device = backend["device"]
        spacing_tensor, top_region_index_tensor, bottom_region_index_tensor, modality_tensor = tensors
        output_size = tuple(int(v) for v in label.shape[-3:])
        noise_scheduler = backend["noise_scheduler"]
        diffusion_unet = backend["diffusion_unet"]
        controlnet = backend["controlnet"]
        autoencoder = backend["autoencoder"]
        scale_factor = backend["scale_factor"]
        cfg_guidance_scale = float(cfg.cfg_guidance_scale)
        include_body_region = diffusion_unet.include_top_region_index_input
        include_modality = diffusion_unet.num_class_embeds is not None

        if modality_tensor is not None and int(modality_tensor.flatten()[0]) <= 7:
            a_min, a_max, crop_min = -1000, 1000, -1000
        else:
            a_min, a_max, crop_min = 0, 1000, 0
        b_min, b_max = 0.0, 1.0

        controlnet_cond_tensor = binarize_labels(label.as_tensor().long()).half()
        controlnet_uncond_tensor = None
        if cfg_guidance_scale > 0:
            label_no_tumor = torch.nn.functional.interpolate(
                remove_tumors(label.squeeze(0)).unsqueeze(0).float(),
                size=output_size,
                mode="nearest",
            ).to(label.dtype)
            controlnet_uncond_tensor = binarize_labels(label_no_tumor.as_tensor().long()).half()
            del label_no_tumor

        latents = (
            torch.randn([1] + list(latent_shape), device=device)
            .half()
            .mul(float(self.method_config.get("maisi_noise_factor", 1.0)))
        )
        initial_norm = float(latents.float().norm().detach().cpu())
        if isinstance(noise_scheduler, RFlowScheduler):
            noise_scheduler.set_timesteps(
                num_inference_steps=int(cfg.num_inference_steps),
                input_img_size_numel=torch.prod(torch.tensor(latents.shape[-3:])),
            )
        else:
            noise_scheduler.set_timesteps(num_inference_steps=int(cfg.num_inference_steps))
        if isinstance(noise_scheduler, DDPMScheduler) and int(cfg.num_inference_steps) < noise_scheduler.num_train_timesteps:
            raise ValueError("MAISI endpoint capture refuses shortened DDPM inference; use full scheduler steps or rflow-ct.")

        timesteps = noise_scheduler.timesteps
        next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype)))
        trace_norms = []
        keep_trace = bool(self.method_config.get("maisi_trace_latent_norms", False))
        autocast_enabled = device.type == "cuda"
        stepwise_solver_enabled = bool(self.method_config.get("maisi_stepwise_flow_solver", False))
        if stepwise_solver_enabled and not isinstance(noise_scheduler, RFlowScheduler):
            raise ValueError("maisi_stepwise_flow_solver requires MONAI RFlowScheduler")
        stepwise_refinement_steps = int(
            self.method_config.get(
                "maisi_stepwise_refinement_steps",
                self.method_config.get("maisi_latent_refinement_steps", 0),
            )
        )
        stepwise_refinement_every = max(1, int(self.method_config.get("maisi_stepwise_refinement_every", 1)))
        stepwise_allow_fallback = bool(self.method_config.get("maisi_stepwise_allow_fallback", False))
        stepwise_trace = []
        stepwise_prepared = None
        if stepwise_solver_enabled and stepwise_refinement_steps > 0:
            stepwise_prepared = self._prepare_maisi_latent_refinement(
                latents,
                normal,
                label,
                texture_target=texture_target,
                texture_masks=texture_masks,
                texture_channel_names=texture_channel_names,
                texture_channel_weights=texture_channel_weights,
            )
        for step_index, (t, next_t) in enumerate(zip(timesteps, next_timesteps)):
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=autocast_enabled):
                controlnet_inputs = {
                    "x": latents,
                    "timesteps": torch.Tensor((t,)).to(device),
                    "controlnet_cond": controlnet_cond_tensor,
                }
                if include_modality:
                    controlnet_inputs["class_labels"] = modality_tensor
                if cfg_guidance_scale > 0:
                    for key in list(controlnet_inputs.keys()):
                        if key == "class_labels":
                            controlnet_inputs[key] = torch.cat([modality_tensor, torch.zeros_like(modality_tensor)])
                        elif key == "controlnet_cond":
                            controlnet_inputs[key] = torch.cat([controlnet_cond_tensor, controlnet_uncond_tensor])
                        else:
                            controlnet_inputs[key] = torch.cat([controlnet_inputs[key]] * 2)
                down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

                unet_inputs = {
                    "x": latents,
                    "timesteps": torch.Tensor((t,)).to(device),
                    "spacing_tensor": spacing_tensor,
                    "down_block_additional_residuals": down_block_res_samples,
                    "mid_block_additional_residual": mid_block_res_sample,
                }
                if include_body_region:
                    unet_inputs.update(
                        {
                            "top_region_index_tensor": top_region_index_tensor,
                            "bottom_region_index_tensor": bottom_region_index_tensor,
                        }
                    )
                if include_modality:
                    unet_inputs["class_labels"] = modality_tensor
                if cfg_guidance_scale > 0:
                    for key in list(unet_inputs.keys()):
                        if key in ("down_block_additional_residuals", "mid_block_additional_residual"):
                            continue
                        if key != "class_labels":
                            unet_inputs[key] = torch.cat([unet_inputs[key]] * 2)
                        else:
                            unet_inputs[key] = torch.cat([unet_inputs[key], torch.zeros_like(modality_tensor)])
                if cfg_guidance_scale == 0:
                    model_output = diffusion_unet(**unet_inputs)
                else:
                    model_t, model_uncond = diffusion_unet(**unet_inputs).chunk(2)
                    model_output = model_uncond + cfg_guidance_scale * (model_t - model_uncond)
            del down_block_res_samples, mid_block_res_sample, controlnet_inputs, unet_inputs
            if stepwise_solver_enabled:
                estimates = self._maisi_rflow_endpoint_reprojection(
                    latents,
                    model_output,
                    t,
                    next_t,
                    noise_scheduler.num_train_timesteps,
                )
                endpoint_hat = estimates["endpoint_hat"]
                refined_endpoint = endpoint_hat
                should_refine = stepwise_refinement_steps > 0 and step_index % stepwise_refinement_every == 0
                texture_weight_scale = _late_texture_schedule_scale(
                    self.method_config,
                    "maisi",
                    estimates["forward_t"],
                )
                if should_refine:
                    try:
                        with torch.enable_grad():
                            refined_endpoint, refinement = self._refine_maisi_latent_endpoint(
                                endpoint_hat,
                                normal,
                                label,
                                target_hist,
                                prepared=stepwise_prepared,
                                steps_override=stepwise_refinement_steps,
                                texture_target=texture_target,
                                texture_masks=texture_masks,
                                texture_channel_names=texture_channel_names,
                                texture_channel_weights=texture_channel_weights,
                                texture_weight_scale=texture_weight_scale,
                                condition_metadata=condition_metadata,
                            )
                    except Exception as exc:
                        if not stepwise_allow_fallback:
                            raise
                        refined_endpoint = endpoint_hat
                        refinement = {
                            "status": "fallback_after_error",
                            "accepted": False,
                            "error": str(exc),
                            "steps": stepwise_refinement_steps,
                        }
                else:
                    refinement = {
                        "status": "skipped_cadence",
                        "accepted": False,
                        "steps": 0,
                    }
                with torch.no_grad():
                    latents = (
                        estimates["scheduler_next_t"] * estimates["source_hat"]
                        + estimates["forward_next_t"] * refined_endpoint.float()
                    ).to(dtype=model_output.dtype)
                    refinement_delta_norm = float(
                        (refined_endpoint.float() - endpoint_hat.float()).norm().detach().cpu()
                    )
                stepwise_trace.append(
                    {
                        "step": int(step_index),
                        "scheduler_t": float(estimates["scheduler_t"]),
                        "scheduler_next_t": float(estimates["scheduler_next_t"]),
                        "forward_t": float(estimates["forward_t"]),
                        "forward_next_t": float(estimates["forward_next_t"]),
                        "identity_max_abs": float(estimates["identity_max_abs"]),
                        "endpoint_norm": float(endpoint_hat.norm().detach().cpu()),
                        "source_norm": float(estimates["source_hat"].norm().detach().cpu()),
                        "refinement_status": refinement.get("status", ""),
                        "refinement_accepted": bool(refinement.get("accepted", False)),
                        "refinement_delta_objective": refinement.get("delta_objective", ""),
                        "refinement_initial_terms": refinement.get("initial_eval", {}),
                        "refinement_final_terms": refinement.get("final_eval", {}),
                        "texture_weight_scale": float(texture_weight_scale),
                        "refinement_delta_norm": refinement_delta_norm,
                        "latent_next_norm": float(latents.float().norm().detach().cpu()),
                    }
                )
            elif not isinstance(noise_scheduler, RFlowScheduler):
                with torch.no_grad():
                    latents, _ = noise_scheduler.step(model_output, t, latents)  # type: ignore
            else:
                with torch.no_grad():
                    latents, _ = noise_scheduler.step(model_output, t, latents, next_t)  # type: ignore
            if keep_trace:
                trace_norms.append(float(latents.float().norm().detach().cpu()))
            del model_output

        endpoint = latents.detach()
        endpoint_info = {
            "captured": True,
            "domain": "maisi_rflow_latent",
            "shape": [int(v) for v in endpoint.shape],
            "dtype": str(endpoint.dtype).replace("torch.", ""),
            "num_inference_steps": int(len(timesteps)),
            "initial_norm": initial_norm,
            "endpoint_norm": float(endpoint.float().norm().detach().cpu()),
            "scheduler": type(noise_scheduler).__name__,
            "trace_norms": trace_norms[:64],
            "stepwise_flow_solver": {
                "status": "pass" if stepwise_solver_enabled else "disabled",
                "enabled": bool(stepwise_solver_enabled),
                "equations": [18, 19, 20, 21, 22] if stepwise_solver_enabled else [],
                "num_sampling_steps": int(len(timesteps)),
                "refinement_steps_per_selected_timestep": int(stepwise_refinement_steps),
                "refinement_every": int(stepwise_refinement_every),
                "num_refinement_attempts": int(
                    sum(row.get("refinement_status") != "skipped_cadence" for row in stepwise_trace)
                ),
                "num_refinement_accepted": int(sum(bool(row.get("refinement_accepted")) for row in stepwise_trace)),
                "max_reprojection_identity_error": float(
                    max((row.get("identity_max_abs", 0.0) for row in stepwise_trace), default=0.0)
                ),
                "trace": stepwise_trace,
                "note": (
                    "MONAI scheduler time runs from noise=1000 to data=0; forward_t=1-scheduler_t/T. "
                    "Each row estimates the endpoint with Eq.18, applies decoder-backed Eq.19-21 refinement, "
                    "and advances with Eq.22."
                ),
            },
        }
        source_latent = None
        if bool(self.method_config.get("maisi_trajectory_alpha_search", False)):
            try:
                source_latent = self._encode_maisi_source_latent(normal, output_size)
            except Exception as exc:
                if not bool(self.method_config.get("maisi_trajectory_alpha_search_allow_fallback", True)):
                    raise
                endpoint_info["trajectory_alpha_search"] = {
                    "status": "fallback_after_error",
                    "enabled": True,
                    "error": str(exc),
                }
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if stepwise_solver_enabled:
            refinement_info = {
                "status": "integrated_stepwise",
                "steps": int(stepwise_refinement_steps),
                "losses": [],
                "note": "Endpoint refinement was applied inside every selected RFlow timestep before Eq.22 re-projection.",
            }
        else:
            try:
                endpoint, refinement_info = self._refine_maisi_latent_endpoint(
                    endpoint,
                    normal,
                    label,
                    target_hist,
                    texture_target=texture_target,
                    texture_masks=texture_masks,
                    texture_channel_names=texture_channel_names,
                    texture_channel_weights=texture_channel_weights,
                    condition_metadata=condition_metadata,
                )
            except Exception as exc:
                if not bool(self.method_config.get("maisi_latent_refinement_allow_fallback", True)):
                    raise
                refinement_info = {
                    "status": "fallback_after_error",
                    "steps": int(self.method_config.get("maisi_latent_refinement_steps", 0)),
                    "error": str(exc),
                    "losses": [],
                }
        endpoint_info["latent_refinement"] = refinement_info
        endpoint_info["endpoint_norm_after_refinement"] = float(endpoint.float().norm().detach().cpu())
        if source_latent is not None:
            try:
                endpoint, alpha_search_info = self._maisi_trajectory_alpha_search(
                    source_latent,
                    endpoint,
                    normal,
                    label,
                    target_hist,
                )
            except Exception as alpha_exc:
                if not bool(self.method_config.get("maisi_trajectory_alpha_search_allow_fallback", True)):
                    raise
                alpha_search_info = {
                    "status": "fallback_after_error",
                    "enabled": bool(self.method_config.get("maisi_trajectory_alpha_search", False)),
                    "error": str(alpha_exc),
                }
            endpoint_info["trajectory_alpha_search"] = alpha_search_info
            endpoint_info["endpoint_norm_after_trajectory_alpha_search"] = float(endpoint.float().norm().detach().cpu())
            del source_latent

        recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
        inferer = SlidingWindowInferer(
            roi_size=list(cfg.autoencoder_sliding_window_infer_size),
            sw_batch_size=1,
            progress=False,
            mode="gaussian",
            overlap=float(cfg.autoencoder_sliding_window_infer_overlap),
            sw_device=device,
            device=torch.device("cpu"),
        )
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=autocast_enabled):
            synthetic_image = dynamic_infer(inferer, recon_model, endpoint)
            if modality_tensor is not None and int(modality_tensor.flatten()[0]) <= 7:
                synthetic_image = torch.clip(synthetic_image, b_min, b_max).cpu()
            else:
                synthetic_image = torch.clip(synthetic_image, b_min, None).cpu()
            synthetic_image = (synthetic_image - b_min) / (b_max - b_min)
            synthetic_image = synthetic_image * (a_max - a_min) + a_min
            synthetic_image = crop_img_body_mask(synthetic_image, label, a_min=crop_min)
        candidate = synthetic_image.squeeze().float().cpu().numpy().astype(np.float32)
        candidate = self._maisi_extract_roi(candidate, np.asarray(normal).shape)
        return np.clip(candidate, HU_MIN, HU_MAX).astype(np.float32), endpoint_info

    def _maisi_rflow_candidate(
        self,
        normal,
        mask,
        target_hist=None,
        texture_target=None,
        texture_masks=None,
        texture_channel_names=None,
        texture_channel_weights=None,
        condition_metadata=None,
    ):
        import torch
        from monai.data import MetaTensor

        backend = self._load_maisi_rflow_backend()
        cfg = backend["cfg"]
        device = backend["device"]
        label_np = self._maisi_label_condition(mask, normal=normal)
        spacing = tuple(float(v) for v in cfg.spacing)
        affine = torch.diag(torch.tensor([spacing[0], spacing[1], spacing[2], 1.0], dtype=torch.float32))
        label_tensor = torch.tensor(label_np, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        label = MetaTensor(label_tensor, affine=affine).to(device)
        include_body_region = backend["diffusion_unet"].include_top_region_index_input
        spacing_tensor, top_region_index_tensor, bottom_region_index_tensor, modality_tensor = backend["build_conditioning_tensors"](
            label,
            spacing,
            int(cfg.modality),
            include_body_region,
            device,
        )
        latent_shape = (int(cfg.latent_channels), label_np.shape[0] // 4, label_np.shape[1] // 4, label_np.shape[2] // 4)
        if bool(self.method_config.get("maisi_capture_latent_endpoint", False)):
            candidate, endpoint_info = self._maisi_rflow_candidate_with_endpoint(
                normal,
                label,
                latent_shape,
                (spacing_tensor, top_region_index_tensor, bottom_region_index_tensor, modality_tensor),
                target_hist=target_hist,
                texture_target=texture_target,
                texture_masks=texture_masks,
                texture_channel_names=texture_channel_names,
                texture_channel_weights=texture_channel_weights,
                condition_metadata=condition_metadata,
            )
            endpoint_info["physical_roi_mapping"] = self._maisi_mapping_metadata(
                getattr(self, "_last_maisi_roi_mapping", None)
            )
            self._last_maisi_endpoint_info = endpoint_info
            return candidate
        synthetic_image, _returned_label = backend["sample_image"](
            autoencoder=backend["autoencoder"],
            diffusion_unet=backend["diffusion_unet"],
            controlnet=backend["controlnet"],
            noise_scheduler=backend["noise_scheduler"],
            scale_factor=backend["scale_factor"],
            device=device,
            combine_label_or=label,
            spacing_tensor=spacing_tensor,
            latent_shape=latent_shape,
            output_size=label_np.shape,
            noise_factor=1.0,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            modality_tensor=modality_tensor,
            num_inference_steps=int(cfg.num_inference_steps),
            autoencoder_sliding_window_infer_size=cfg.autoencoder_sliding_window_infer_size,
            autoencoder_sliding_window_infer_overlap=float(cfg.autoencoder_sliding_window_infer_overlap),
            cfg_guidance_scale=float(cfg.cfg_guidance_scale),
        )
        candidate = synthetic_image.squeeze().float().cpu().numpy().astype(np.float32)
        candidate = self._maisi_extract_roi(candidate, np.asarray(normal).shape)
        self._last_maisi_endpoint_info = {
            "captured": False,
            "domain": "maisi_rflow_latent",
            "physical_roi_mapping": self._maisi_mapping_metadata(
                getattr(self, "_last_maisi_roi_mapping", None)
            ),
        }
        return np.clip(candidate, HU_MIN, HU_MAX).astype(np.float32)

    def _ctflow_encode_source_latent(self, normal):
        import torch

        backend = self._load_ctflow_backend()
        vae = backend["vae"]
        device = backend["device"]
        image_size = int(self.method_config.get("ctflow_image_size", 256))
        num_slices = int(self.method_config.get("ctflow_num_slices", 32))
        batch_size = int(self.method_config.get("ctflow_vae_batch_size", 8))
        volume = _resize_array(np.asarray(normal, dtype=np.float32), (normal.shape[0], normal.shape[1], num_slices), order=1)
        scaled = (np.clip(volume, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
        slices = torch.tensor(scaled.astype(np.float32), dtype=torch.float32).permute(2, 0, 1).unsqueeze(1).repeat(1, 3, 1, 1)
        if tuple(slices.shape[-2:]) != (image_size, image_size):
            import torch.nn.functional as F

            slices = F.interpolate(slices, size=(image_size, image_size), mode="bilinear", align_corners=False)
        latents = []
        with torch.no_grad():
            for chunk in slices.split(batch_size, dim=0):
                encoded = vae.encode(chunk.to(device)).latent_dist.sample().detach().float().cpu()
                latents.append(encoded)
        latent = torch.cat(latents, dim=0).permute(1, 0, 2, 3).contiguous()
        return latent

    @staticmethod
    def _ctflow_flow_endpoint_reprojection(sample, velocity, timestep, next_timestep):
        """Apply paper Eq. (18) and Eq. (22) in CTFlow's data=0, noise=1 time."""
        import torch

        t = float(torch.as_tensor(timestep).detach().cpu())
        next_t = float(torch.as_tensor(next_timestep).detach().cpu())
        if not (0.0 <= next_t <= t <= 1.0):
            raise ValueError(f"CTFlow timesteps must satisfy 0 <= next_t <= t <= 1, got {next_t}, {t}")
        sample_f = sample.detach().float()
        velocity_f = velocity.detach().float()
        endpoint_hat = sample_f - t * velocity_f
        source_hat = sample_f + (1.0 - t) * velocity_f
        prior_next = sample_f + (next_t - t) * velocity_f
        identity_next = next_t * source_hat + (1.0 - next_t) * endpoint_hat
        identity_error = float(torch.max(torch.abs(identity_next - prior_next)).detach().cpu())
        return {
            "source_hat": source_hat,
            "endpoint_hat": endpoint_hat,
            "prior_next": prior_next,
            "identity_next": identity_next,
            "identity_max_abs": identity_error,
            "ctflow_t": t,
            "ctflow_next_t": next_t,
            "forward_t": 1.0 - t,
            "forward_next_t": 1.0 - next_t,
        }

    def _decode_ctflow_latent_unit(self, latent, output_shape):
        import torch
        import torch.nn.functional as F

        from echosyn.common import unscale_latents

        backend = self._load_ctflow_backend()
        vae = backend["vae"]
        for parameter in vae.parameters():
            parameter.requires_grad_(False)
        latent = unscale_latents(latent, backend["vae_scaling"])
        batch, channels, frames, height, width = latent.shape
        flat = latent.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        decoded_parts = []
        batch_size = max(1, int(self.method_config.get("ctflow_refinement_decode_batch_size", 4)))
        gradient_checkpointing = bool(
            self.method_config.get("ctflow_refinement_gradient_checkpointing", True)
        )

        def decode_chunk(value):
            return vae.decode(value).sample.float()

        for chunk in flat.split(batch_size, dim=0):
            chunk = chunk.float()
            if gradient_checkpointing and torch.is_grad_enabled() and chunk.requires_grad:
                from torch.utils.checkpoint import checkpoint

                decoded_parts.append(checkpoint(decode_chunk, chunk, use_reentrant=False))
            else:
                decoded_parts.append(decode_chunk(chunk))
        decoded = torch.cat(decoded_parts, dim=0)
        decoded_unit = (decoded * 306.0).clamp(0.0, 255.0).mean(dim=1, keepdim=True) / 255.0
        decoded_unit = decoded_unit.reshape(batch, frames, 1, decoded.shape[-2], decoded.shape[-1])
        decoded_unit = decoded_unit.permute(0, 2, 1, 3, 4).contiguous()
        return F.interpolate(
            decoded_unit,
            size=tuple(int(value) for value in output_shape),
            mode="trilinear",
            align_corners=False,
        )

    def _ctflow_soft_hist_loss(self, decoded_unit, lesion_mask, target_hist):
        import torch

        values = decoded_unit[lesion_mask > 0.25]
        if values.numel() == 0 or target_hist is None:
            return decoded_unit.sum() * 0.0
        hu = (values * (HU_MAX - HU_MIN) + HU_MIN).clamp(HU_MIN, HU_MAX)
        bins = int(self.method_config.get("histogram_bins", 40))
        centers = torch.linspace(HU_MIN, HU_MAX, bins, device=hu.device, dtype=hu.dtype)
        width = (HU_MAX - HU_MIN) / float(bins)
        soft = torch.relu(1.0 - torch.abs(hu[:, None] - centers[None, :]) / width)
        pred = soft.sum(dim=0)
        pred = pred / (pred.sum() + 1.0e-8)
        target = torch.as_tensor(target_hist, dtype=pred.dtype, device=pred.device).reshape(-1)
        if target.numel() != bins:
            raise ValueError(
                "CTFlow target histogram must match histogram_bins: "
                f"got {target.numel()} values for {bins} bins"
            )
        if not torch.isfinite(target).all() or float(target.sum().detach().cpu()) <= 0.0:
            raise ValueError("CTFlow target histogram must contain finite positive mass")
        target = target / (target.sum() + 1.0e-8)
        mid = 0.5 * (pred + target)
        return 0.5 * (pred * ((pred + 1.0e-8) / (mid + 1.0e-8)).log()).sum() + 0.5 * (
            target * ((target + 1.0e-8) / (mid + 1.0e-8)).log()
        ).sum()

    def _refine_ctflow_latent_endpoint(
        self,
        endpoint,
        prefix_latent,
        normal,
        mask,
        target_hist,
        steps_override=None,
        texture_target=None,
        texture_masks=None,
        texture_channel_names=None,
        texture_channel_weights=None,
        texture_weight_scale=1.0,
        condition_metadata=None,
    ):
        import torch

        steps = int(
            self.method_config.get("ctflow_latent_refinement_steps", 0)
            if steps_override is None
            else steps_override
        )
        if steps <= 0:
            return endpoint, {"status": "skipped", "steps": 0, "losses": []}
        backend = self._load_ctflow_backend()
        device = backend["device"]
        normal_t = torch.as_tensor(np.asarray(normal), dtype=torch.float32, device=device)[None, None]
        normal_unit = ((normal_t - HU_MIN) / (HU_MAX - HU_MIN)).clamp(0.0, 1.0)
        lesion_mask = torch.as_tensor(np.asarray(mask) > 0, dtype=torch.float32, device=device)[None, None]
        texture_target_unit = None
        if texture_target is not None:
            texture_target_t = torch.as_tensor(
                np.asarray(texture_target), dtype=torch.float32, device=device
            )[None, None]
            texture_target_unit = ((texture_target_t - HU_MIN) / (HU_MAX - HU_MIN)).clamp(0.0, 1.0)
        texture_mask_tensor = (
            torch.as_tensor(np.asarray(texture_masks), dtype=torch.float32, device=device)[None]
            if texture_masks is not None
            else lesion_mask
        )
        texture_channel_names = list(texture_channel_names or ["lesion"])
        texture_channel_weights = list(texture_channel_weights or [1.0])
        posterior_energies = self._posterior_energies("ctflow")
        condition_metadata = dict(condition_metadata or {})
        z0 = endpoint.detach().float()
        z = z0.clone().requires_grad_(True)
        optimizer = torch.optim.Adam(
            [z], lr=float(self.method_config.get("ctflow_latent_refinement_lr", 1.0e-3))
        )
        lambda_hu = float(self.method_config.get("ctflow_latent_lambda_hu", self.method_config.get("lambda_hu", 1.0)))
        lambda_bg = float(self.method_config.get("ctflow_latent_lambda_bg", self.method_config.get("lambda_bg", 1.0)))
        lambda_texture_base = float(self.method_config.get("ctflow_latent_lambda_texture", 0.0))
        texture_weight_scale = float(texture_weight_scale)
        if lambda_texture_base < 0 or not 0.0 <= texture_weight_scale <= 1.0:
            raise ValueError(
                "CTFlow texture weight and schedule scale must be non-negative, with scale <= 1; "
                f"got weight={lambda_texture_base}, scale={texture_weight_scale}"
            )
        lambda_texture = lambda_texture_base * texture_weight_scale
        if lambda_texture > 0 and texture_target_unit is None:
            raise ValueError("ctflow_latent_lambda_texture requires verified train texture guidance")
        prox_eta = float(self.method_config.get("ctflow_latent_prox_eta", 10.0))
        if prox_eta <= 0:
            raise ValueError(f"ctflow_latent_prox_eta must be positive, got {prox_eta}")
        lambda_prox = 1.0 / (2.0 * prox_eta)
        prefix = prefix_latent.detach().float()

        def loss_terms(current):
            full_latent = torch.cat([prefix, current], dim=2)
            decoded = self._decode_ctflow_latent_unit(full_latent, normal_t.shape[-3:])
            hist_loss = self._ctflow_soft_hist_loss(decoded, lesion_mask, target_hist)
            background_mask = (1.0 - lesion_mask).clamp(0.0, 1.0)
            background_loss = (((decoded - normal_unit) ** 2) * background_mask).sum() / (
                background_mask.sum() + 1.0
            )
            if lambda_texture > 0:
                texture_loss = _multiscale_3d_texture_energy(
                    decoded,
                    texture_target_unit,
                    texture_mask_tensor,
                    scales=self.method_config.get("ctflow_latent_texture_scales", [1, 2, 4]),
                    band_weights=self.method_config.get(
                        "ctflow_latent_texture_band_weights", [0.55, 0.30, 0.15]
                    ),
                    autocorr_lags=self.method_config.get(
                        "ctflow_latent_texture_autocorr_lags", [1, 2]
                    ),
                    core_erosion=int(
                        self.method_config.get("ctflow_latent_texture_core_erosion", 1)
                    ),
                    min_voxels=int(
                        self.method_config.get("ctflow_latent_texture_min_voxels", 8)
                    ),
                    channel_weights=texture_channel_weights,
                    local_variance_weight=float(
                        self.method_config.get(
                            "ctflow_latent_texture_local_variance_weight", 0.0
                        )
                    ),
                    gram_weight=float(
                        self.method_config.get("ctflow_latent_texture_gram_weight", 0.0)
                    ),
                    patch_swd_weight=float(
                        self.method_config.get(
                            "ctflow_latent_texture_patch_swd_weight", 0.0
                        )
                    ),
                    patch_swd_sizes=self.method_config.get(
                        "ctflow_latent_texture_patch_swd_sizes", [3]
                    ),
                    patch_swd_projections=int(
                        self.method_config.get(
                            "ctflow_latent_texture_patch_swd_projections", 12
                        )
                    ),
                    patch_swd_min_support=float(
                        self.method_config.get(
                            "ctflow_latent_texture_patch_swd_min_support", 0.25
                        )
                    ),
                    patch_swd_max_samples=int(
                        self.method_config.get(
                            "ctflow_latent_texture_patch_swd_max_samples", 2048
                        )
                    ),
                    patch_swd_seed=int(
                        self.method_config.get(
                            "ctflow_latent_texture_patch_swd_seed", 1729
                        )
                    ),
                )
            else:
                texture_loss = decoded.sum() * 0.0
            custom_loss, custom_terms = evaluate_posterior_energies(
                posterior_energies,
                PosteriorEnergyContext(
                    decoded=decoded,
                    lesion_mask=lesion_mask,
                    source_image=normal_unit,
                    background_mask=background_mask,
                    target_histogram=torch.as_tensor(
                        target_hist, dtype=decoded.dtype, device=decoded.device
                    ),
                    prior="ctflow",
                    metadata={
                        "condition": condition_metadata,
                        "texture_target": texture_target_unit,
                        "texture_masks": texture_mask_tensor,
                    },
                ),
            )
            prox_loss = ((current - z0) ** 2).mean()
            loss = (
                lambda_hu * hist_loss
                + lambda_bg * background_loss
                + lambda_texture * texture_loss
                + lambda_prox * prox_loss
                + custom_loss
            )
            return loss, hist_loss, background_loss, texture_loss, prox_loss, custom_terms

        def evaluate(current):
            with torch.no_grad():
                (
                    loss,
                    hist_loss,
                    background_loss,
                    texture_loss,
                    prox_loss,
                    custom_terms,
                ) = loss_terms(current)
            return {
                "loss": float(loss.detach().cpu()),
                "hist": float(hist_loss.detach().cpu()),
                "background": float(background_loss.detach().cpu()),
                "texture": float(texture_loss.detach().cpu()),
                "prox": float(prox_loss.detach().cpu()),
                "custom": {
                    name: float(value.detach().cpu())
                    for name, value in custom_terms.items()
                },
            }

        initial_eval = evaluate(z0)
        losses = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            (
                loss,
                hist_loss,
                background_loss,
                texture_loss,
                prox_loss,
                custom_terms,
            ) = loss_terms(z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], float(self.method_config.get("ctflow_latent_grad_clip", 1.0)))
            optimizer.step()
            losses.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "hist": float(hist_loss.detach().cpu()),
                    "background": float(background_loss.detach().cpu()),
                    "texture": float(texture_loss.detach().cpu()),
                    "prox": float(prox_loss.detach().cpu()),
                    "custom": {
                        name: float(value.detach().cpu())
                        for name, value in custom_terms.items()
                    },
                }
            )
        final_eval = evaluate(z.detach())
        delta = final_eval["loss"] - initial_eval["loss"]
        min_delta = float(self.method_config.get("ctflow_latent_refinement_min_delta", 0.0))
        accept_if_improved = bool(self.method_config.get("ctflow_latent_refinement_accept_if_improved", True))
        accepted = (not accept_if_improved) or delta <= -min_delta
        return (z.detach() if accepted else z0).to(endpoint.dtype), {
            "status": "pass",
            "steps": int(steps),
            "losses": losses,
            "initial_eval": initial_eval,
            "final_eval": final_eval,
            "delta_objective": float(delta),
            "accepted": bool(accepted),
            "acceptance_status": "accepted_improved" if accepted else "rejected_no_improvement",
            "accept_if_improved": bool(accept_if_improved),
            "min_delta": float(min_delta),
            "prox_eta": float(prox_eta),
            "prox_weight": float(lambda_prox),
            "texture_weight": float(lambda_texture),
            "texture_weight_base": float(lambda_texture_base),
            "texture_weight_scale": float(texture_weight_scale),
            "texture_channels": texture_channel_names,
            "posterior_energies": describe_posterior_energies(posterior_energies),
        }

    def _ctflow_stepwise_block(
        self,
        prompt_embedding,
        prefix_latent,
        normal,
        mask,
        target_hist,
        texture_target=None,
        texture_masks=None,
        texture_channel_names=None,
        texture_channel_weights=None,
        condition_metadata=None,
    ):
        import torch

        from echosyn.common import sample_latents

        backend = self._load_ctflow_backend()
        generator = backend["generator"]
        device = backend["device"]
        denoiser = backend["denoiser"]
        batch = int(prefix_latent.shape[0])
        channels = int(generator.config.globals.latent_channels)
        height = width = int(generator.config.globals.latent_res)
        frames = int(generator.block_size)
        steps = int(self.method_config.get("ctflow_sampling_steps", 200))
        if steps <= 0:
            raise ValueError(f"ctflow_sampling_steps must be positive, got {steps}")
        refinement_steps = int(self.method_config.get("ctflow_stepwise_refinement_steps", 0))
        refinement_every = max(1, int(self.method_config.get("ctflow_stepwise_refinement_every", 1)))
        allow_fallback = bool(self.method_config.get("ctflow_stepwise_allow_fallback", False))
        condition = sample_latents(generator.config, prefix_latent)
        current = torch.randn(
            (batch, channels, frames, height, width),
            device=device,
            dtype=generator.dtype,
        )
        timesteps = torch.linspace(1.0, 0.0, steps=steps + 1, device=device, dtype=generator.dtype)
        trace = []
        for step_index, (timestep, next_timestep) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            with torch.no_grad():
                velocity = denoiser(
                    current,
                    timestep,
                    encoder_hidden_states=prompt_embedding,
                    cond_image=condition,
                ).sample
            estimates = self._ctflow_flow_endpoint_reprojection(
                current, velocity, timestep, next_timestep
            )
            endpoint = estimates["endpoint_hat"]
            should_refine = refinement_steps > 0 and step_index % refinement_every == 0
            texture_weight_scale = _late_texture_schedule_scale(
                self.method_config,
                "ctflow",
                estimates["forward_t"],
            )
            if should_refine:
                try:
                    with torch.enable_grad():
                        endpoint, refinement = self._refine_ctflow_latent_endpoint(
                            endpoint,
                            prefix_latent,
                            normal,
                            mask,
                            target_hist,
                            steps_override=refinement_steps,
                            texture_target=texture_target,
                            texture_masks=texture_masks,
                            texture_channel_names=texture_channel_names,
                            texture_channel_weights=texture_channel_weights,
                            texture_weight_scale=texture_weight_scale,
                            condition_metadata=condition_metadata,
                        )
                except Exception as exc:
                    if not allow_fallback:
                        raise
                    endpoint = estimates["endpoint_hat"]
                    refinement = {
                        "status": "fallback_after_error",
                        "accepted": False,
                        "error": str(exc),
                    }
            else:
                refinement = {"status": "skipped_cadence", "accepted": False}
            current = (
                estimates["ctflow_next_t"] * estimates["source_hat"]
                + estimates["forward_next_t"] * endpoint.float()
            ).to(velocity.dtype)
            trace.append(
                {
                    "step_index": int(step_index),
                    "ctflow_t": estimates["ctflow_t"],
                    "ctflow_next_t": estimates["ctflow_next_t"],
                    "identity_max_abs": estimates["identity_max_abs"],
                    "refinement_status": refinement.get("status", ""),
                    "refinement_accepted": bool(refinement.get("accepted", False)),
                    "refinement_delta_objective": refinement.get("delta_objective", ""),
                    "refinement_initial_terms": refinement.get("initial_eval", {}),
                    "refinement_final_terms": refinement.get("final_eval", {}),
                    "texture_weight_scale": float(texture_weight_scale),
                }
            )
        self._last_ctflow_endpoint_info = {
            "captured": True,
            "domain": "ctflow_stdit_latent",
            "stepwise_flow_solver": {
                "status": "pass",
                "enabled": True,
                "equations": [18, 19, 20, 21, 22],
                "num_sampling_steps": int(steps),
                "refinement_steps_per_selected_timestep": int(refinement_steps),
                "refinement_every": int(refinement_every),
                "num_refinement_attempts": int(
                    sum(row["refinement_status"] != "skipped_cadence" for row in trace)
                ),
                "num_refinement_accepted": int(sum(row["refinement_accepted"] for row in trace)),
                "max_reprojection_identity_error": float(
                    max((row["identity_max_abs"] for row in trace), default=0.0)
                ),
                "trace": trace,
                "note": (
                    "CTFlow time runs from noise=1 to data=0. Each Euler row estimates the data endpoint "
                    "with Eq.18, applies FLUX-VAE-backed Eq.19-21 refinement, and advances with Eq.22."
                ),
            },
        }
        return current

    @staticmethod
    def _ctflow_gt_head_source_block(gt_latent, block_size):
        """Select the second encoded block used by CTFlow's official gt-head path."""
        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError(f"CTFlow block_size must be positive, got {block_size}")
        if gt_latent.ndim != 5:
            raise ValueError(f"CTFlow gt latent must be 5D, got {list(gt_latent.shape)}")
        if gt_latent.shape[2] < block_size * 2:
            raise ValueError(
                f"CTFlow gt-head requires at least {block_size * 2} latent slices, "
                f"got {list(gt_latent.shape)}"
            )
        return gt_latent[:, :, block_size : 2 * block_size, :, :]

    def _ctflow_prior_candidate(
        self,
        normal,
        mask=None,
        target_hist=None,
        texture_target=None,
        texture_masks=None,
        texture_channel_names=None,
        texture_channel_weights=None,
        condition_metadata=None,
    ):
        import torch

        prompt_embedding = self._load_ctflow_embedding()
        backend = self._load_ctflow_backend()
        generator = backend["generator"]
        device = backend["device"]
        prompt_embedding = prompt_embedding.to(device)
        gt_latent = self._ctflow_encode_source_latent(normal).unsqueeze(0).to(device)
        source_latent_mode = "per_sample_source_roi_encoding"
        if int(gt_latent.shape[1]) != 16:
            raise ValueError(f"CTFlow gt latent must have 16 channels, got {list(gt_latent.shape)}")
        self._last_ctflow_conditioning_info = {
            "source_latent_mode": source_latent_mode,
            "source_latent_shape": [int(value) for value in gt_latent.shape],
            "formal_same_input_conditioning": True,
        }
        mode = str(self.method_config.get("ctflow_inference_type", "gt-head"))
        max_blocks = int(self.method_config.get("ctflow_max_blocks", 1))
        stepwise_solver = self._uses_ctflow_stepwise_solver()
        with torch.no_grad():
            if mode == "gt-head":
                block_size = int(generator.block_size)
                gt_first_block = self._ctflow_gt_head_source_block(gt_latent, block_size)
                self._last_ctflow_conditioning_info.update(
                    {
                        "official_gt_head_block_selection": "second_encoded_block",
                        "gt_head_source_latent_frame_range": [block_size, 2 * block_size],
                    }
                )
                from echosyn.common import scale_latents

                gt_first_block = scale_latents(gt_first_block, backend["vae_scaling"])
                if stepwise_solver:
                    if max_blocks != 1:
                        raise ValueError(
                            "CTFlow Eq.18-22 integration currently requires ctflow_max_blocks=1 so "
                            "the source-conditioned prefix and endpoint energy remain exactly paired."
                        )
                    if mask is None or target_hist is None:
                        raise ValueError("CTFlow stepwise solver requires the paired target mask and HU histogram")
                    with torch.enable_grad():
                        generated_block = self._ctflow_stepwise_block(
                            prompt_embedding,
                            gt_first_block,
                            normal,
                            mask,
                            target_hist,
                            texture_target=texture_target,
                            texture_masks=texture_masks,
                            texture_channel_names=texture_channel_names,
                            texture_channel_weights=texture_channel_weights,
                            condition_metadata=condition_metadata,
                        )
                    result_latent = torch.cat([gt_first_block, generated_block], dim=2)
                else:
                    result_latent = generator.generate(
                        prompt_embeds=prompt_embedding,
                        max_blocks=max_blocks,
                        gt_first_block=gt_first_block,
                    )
            elif mode == "full-body":
                if stepwise_solver:
                    raise ValueError(
                        "CTFlow Eq.18-22 integration is validated only for source-conditioned gt-head mode"
                    )
                result_latent = generator.generate(prompt_embeds=prompt_embedding, max_blocks=max_blocks)
            else:
                raise ValueError(f"unsupported ctflow_inference_type={mode}")
            decoded = generator.decode_latent(result_latent, max_batch_size=int(self.method_config.get("ctflow_decode_batch_size", 16)))
        if not stepwise_solver:
            self._last_ctflow_endpoint_info = {
                "captured": False,
                "domain": "ctflow_stdit_latent",
                "stepwise_flow_solver": {"status": "disabled", "enabled": False, "equations": []},
            }
        candidate = decoded[0].float().mean(dim=0).cpu().numpy().astype(np.float32)
        candidate = (candidate / 255.0) * (HU_MAX - HU_MIN) + HU_MIN
        candidate = _resize_array(candidate, np.asarray(normal).shape, order=1)
        return np.clip(candidate, HU_MIN, HU_MAX).astype(np.float32)

    def _project_prior_candidate(
        self,
        normal,
        mask,
        target_hist,
        candidate,
        rng,
        subtype=None,
        texture_rank_guidance=None,
    ):
        mask = np.asarray(mask) > 0
        image = np.asarray(normal, dtype=np.float32)
        subtype_name = str(subtype or "unknown").lower().replace("-", "_")
        projection_modes = self.method_config.get("projection_mode_by_subtype", {})
        projection_mode = str(
            projection_modes.get(
                subtype_name,
                projection_modes.get(
                    "default",
                    self.method_config.get("projection_mode", "rank_histogram"),
                ),
            )
        )
        edit_mask = _dilate(mask, int(self.method_config.get("boundary_dilation", 2)))
        alpha = _smooth(edit_mask.astype(np.float32), float(self.method_config.get("boundary_feather_sigma", 1.0)))
        alpha = np.maximum(alpha, mask.astype(np.float32))
        alpha = _apply_background_weight(alpha, mask, self.method_config.get("lambda_bg", 1.0))
        lesion = image.copy()
        prior_values = np.asarray(candidate, dtype=np.float32)[mask]
        if prior_values.size:
            target = _sample_histogram(target_hist, int(mask.sum()), rng)
            order = np.argsort(prior_values)
            corrected = prior_values.copy()
            corrected[order] = np.sort(target)
            projection_strength = float(self.method_config.get("projection_histogram_strength", 1.0))
            lam = float(self.method_config.get("lambda_hu", 1.0)) * projection_strength
            lam = max(0.0, min(lam, 1.0))
            if projection_mode == "anatomical_texture_rank":
                try:
                    from scipy.ndimage import distance_transform_edt, gaussian_filter

                    texture_sigma = float(self.method_config.get("projection_texture_sigma", 1.2))
                    normal_structure = image.copy()
                    candidate_array = np.asarray(candidate, dtype=np.float32)
                    candidate_structure = gaussian_filter(candidate_array, sigma=texture_sigma)
                    correlated_noise = gaussian_filter(
                        rng.normal(size=image.shape).astype(np.float32), sigma=texture_sigma
                    )
                    distance = distance_transform_edt(mask).astype(np.float32)
                except Exception:
                    normal_structure = _smooth(image, 1.0)
                    candidate_array = np.asarray(candidate, dtype=np.float32)
                    candidate_structure = _smooth(candidate_array, 1.0)
                    correlated_noise = rng.normal(size=image.shape).astype(np.float32)
                    distance = mask.astype(np.float32)

                def _standardize(values):
                    values = np.asarray(values, dtype=np.float32)
                    return (values - float(np.mean(values))) / (float(np.std(values)) + 1.0e-6)

                normal_score = _standardize(normal_structure[mask])
                candidate_score = _standardize(candidate_structure[mask])
                noise_score = _standardize(correlated_noise[mask])
                radial_score = _standardize(distance[mask])
                if subtype_name in {"solid"}:
                    weights = (0.10, 0.10, 0.75, 0.05)
                elif subtype_name in {"part_solid", "partsolid"}:
                    weights = (0.25, 0.10, 0.60, 0.05)
                else:
                    weights = (0.75, 0.10, 0.05, 0.10)
                guidance = (
                    weights[0] * normal_score
                    + weights[1] * candidate_score
                    + weights[2] * radial_score
                    + weights[3] * noise_score
                )
                texture_rank_by_subtype = self.method_config.get(
                    "projection_training_texture_rank_strength_by_subtype", {}
                )
                texture_rank_strength = float(
                    texture_rank_by_subtype.get(
                        subtype_name,
                        texture_rank_by_subtype.get(
                            "default",
                            self.method_config.get(
                                "projection_training_texture_rank_strength", 0.0
                            ),
                        ),
                    )
                )
                texture_rank_strength = max(0.0, min(texture_rank_strength, 1.0))
                if texture_rank_strength > 0:
                    if texture_rank_guidance is None:
                        raise ValueError(
                            "projection_training_texture_rank_strength requires verified train guidance"
                        )
                    texture_rank_guidance = np.asarray(texture_rank_guidance, dtype=np.float32)
                    if texture_rank_guidance.shape != image.shape:
                        raise ValueError(
                            "training texture-rank guidance shape mismatch: "
                            f"{texture_rank_guidance.shape} != {image.shape}"
                        )
                    texture_score = _standardize(texture_rank_guidance[mask])
                    guidance = (
                        (1.0 - texture_rank_strength) * guidance
                        + texture_rank_strength * texture_score
                    )
                guided = prior_values.copy()
                guided[np.argsort(guidance, kind="stable")] = np.sort(target)
                lesion_values = (1.0 - lam) * prior_values + lam * guided
            elif projection_mode == "affine_quantile_texture":
                prior_mean = float(np.mean(prior_values))
                prior_std = float(np.std(prior_values))
                target_mean = float(np.mean(target))
                target_std = float(np.std(target))
                if prior_std > 1.0e-5:
                    affine = (prior_values - prior_mean) * (target_std / prior_std) + target_mean
                else:
                    affine = np.full_like(prior_values, target_mean)
                quantile_blend = float(self.method_config.get("projection_quantile_blend", 0.15))
                quantile_blend = max(0.0, min(quantile_blend, 1.0))
                texture_preserving = (1.0 - quantile_blend) * affine + quantile_blend * corrected
                lesion_values = (1.0 - lam) * prior_values + lam * texture_preserving
            elif projection_mode == "rank_histogram":
                lesion_values = (1.0 - lam) * prior_values + lam * corrected
            else:
                raise ValueError(f"unsupported projection_mode={projection_mode}")
            lesion[mask] = np.clip(lesion_values, HU_MIN, HU_MAX)
        out = image * (1.0 - alpha) + lesion * alpha
        out[~edit_mask] = image[~edit_mask]
        inner_width = int(self.method_config.get("projection_inner_boundary_width", 0))
        subtype_key = str(subtype or "default").lower().replace("-", "_")
        boundary_by_subtype = self.method_config.get("projection_inner_boundary_blend_by_subtype", {})
        inner_blend = float(
            boundary_by_subtype.get(
                subtype_key,
                boundary_by_subtype.get(
                    "default", self.method_config.get("projection_inner_boundary_blend", 0.0)
                ),
            )
        )
        if inner_width > 0 and inner_blend > 0:
            band = _inner_boundary(mask, inner_width)
            strength = max(0.0, min(inner_blend, 1.0))
            out[band] = (1.0 - strength) * out[band] + strength * image[band]
        smoothing_by_subtype = self.method_config.get("lesion_smoothing_sigma_by_subtype", {})
        smoothing_sigma = float(
            smoothing_by_subtype.get(
                subtype_key,
                smoothing_by_subtype.get("default", self.method_config.get("lesion_smoothing_sigma", 0.0)),
            )
        )
        out = _smooth_hist_preserving(out, mask, smoothing_sigma)
        final_hist_strength = float(self.method_config.get("projection_final_histogram_strength", 0.0))
        final_hist_strength = max(0.0, min(final_hist_strength, 1.0))
        if final_hist_strength > 0 and mask.any():
            final_target = _sample_histogram(target_hist, int(mask.sum()), rng)
            current = out[mask].copy()
            corrected = current.copy()
            corrected[np.argsort(current, kind="stable")] = np.sort(final_target)
            out[mask] = (1.0 - final_hist_strength) * current + final_hist_strength * corrected
        return out.astype(np.float32), edit_mask

    def generate(self, checkpoint=None, target_library=None, output_dir=None, num_samples=None):
        seed_everything(42)
        prior_root = self._validate_prior_assets()
        data_root = Path(self.data_config["root"])
        out_base = Path(output_dir or Path(self.data_config.get("output_root", "outputs")) / self._method_id())
        gen_dir = out_base / "generation"
        save_prior_candidate = bool(self.method_config.get("save_prior_candidate", False))
        output_parts = ["images", "masks", "inputs", "metadata"]
        if save_prior_candidate:
            output_parts.append("prior_candidates")
        for rel in output_parts:
            ensure_dir(gen_dir / rel)
        target_root = Path(target_library or self.data_config.get("target_library", data_root / "target_library"))
        texture_donor_map_path_value = self.method_config.get(
            "projection_training_texture_donor_map"
        )
        texture_donor_map_path = (
            Path(texture_donor_map_path_value)
            if texture_donor_map_path_value
            else None
        )
        texture_donor_map = {}
        texture_donor_map_report = {}
        if texture_donor_map_path is not None:
            texture_donor_map_report = load_json(texture_donor_map_path)
            if texture_donor_map_report.get("status") != "pass":
                raise ValueError(
                    f"texture donor map validation failed: {texture_donor_map_path}"
                )
            if texture_donor_map_report.get("selection_reads_generated_pixels") is not False:
                raise ValueError(
                    "texture donor map must be selected without generated-image pixels"
                )
            if texture_donor_map_report.get("all_donors_train") is not True:
                raise ValueError("texture donor map contains a non-training donor")
            texture_donor_map = dict(texture_donor_map_report.get("mapping") or {})
        generation_split = self.method_config.get("generation_split", self.data_config.get("generation_split", "train"))
        transfer_role = _validate_posthoc_target_transfer_role(
            generation_split,
            self.method_config.get("target_texture_blend", 0.0),
            self.method_config.get("subtype_hu_shift", {}),
        )
        normal_images = _normal_case_paths(self.data_config, generation_split)
        excluded_normal_ids = {
            str(value)
            for value in self.method_config.get("normal_exclude_case_ids", [])
        }
        if excluded_normal_ids:
            normal_images = [
                path
                for path in normal_images
                if path.name.replace(".nii.gz", "") not in excluded_normal_ids
            ]
        all_target_metas = sorted((target_root / "metadata").glob("*.json"))
        target_metas = _target_metas_for_config(all_target_metas, target_root, self.method_config)
        target_meta_by_id = {}
        target_index_by_id = {}
        for target_index, target_meta_path in enumerate(all_target_metas):
            target_meta_item = load_json(target_meta_path)
            target_meta_id = str(target_meta_item.get("target_id") or target_meta_path.stem)
            target_meta_by_id[target_meta_id] = target_meta_item
            target_index_by_id[target_meta_id] = target_index
        reference_generation_value = self.method_config.get("reference_generation_dir")
        reference_generation_dir = Path(reference_generation_value) if reference_generation_value else None
        reuse_reference_prior = bool(self.method_config.get("reuse_reference_prior_candidate", False))
        if reference_generation_dir is not None:
            if reference_generation_dir.name != "generation":
                reference_generation_dir = reference_generation_dir / "generation"
            for rel in ("inputs", "metadata"):
                if not (reference_generation_dir / rel).is_dir():
                    raise FileNotFoundError(
                        f"reference generation is missing {rel}: {reference_generation_dir / rel}"
                    )
            if reuse_reference_prior and not (reference_generation_dir / "prior_candidates").is_dir():
                raise FileNotFoundError(
                    "reference generation is missing reusable prior candidates: "
                    f"{reference_generation_dir / 'prior_candidates'}"
                )
        injection_lookup = _target_injection_lookup(self.method_config)
        injection_order_cache = {}
        if not normal_images or not target_metas:
            raise FileNotFoundError("normal ROI samples or target library are missing")
        n = int(num_samples or min(500, len(normal_images) * len(target_metas)))
        rows = []
        resume_existing = bool(self.method_config.get("resume_existing_generation", True))
        generation_sample_offset = int(
            self.method_config.get("generation_sample_offset", 0)
        )
        for i in range(n):
            logical_index = i + generation_sample_offset
            sid = f"synth_{i:06d}"
            image_path = gen_dir / "images" / f"{sid}.nii.gz"
            mask_path = gen_dir / "masks" / f"{sid}.nii.gz"
            input_path = gen_dir / "inputs" / f"{sid}.nii.gz"
            meta_path = gen_dir / "metadata" / f"{sid}.json"
            if resume_existing and image_path.exists() and mask_path.exists() and input_path.exists() and meta_path.exists():
                rows.append(load_json(meta_path))
                continue
            sample_seed = _generation_sample_seed(self.method_config, logical_index)
            seed_everything(sample_seed)
            rng = np.random.default_rng(sample_seed)
            normal_path = normal_images[logical_index % len(normal_images)]
            target_plan_source = "primary"
            target_plan_index = logical_index % len(target_metas)
            target_plan_strategy = str(self.method_config.get("target_sampling_strategy", "sequential"))
            target_plan_seed = int(self.method_config.get("target_sampling_seed", self.method_config.get("seed", 42)))
            texture_blend = float(self.method_config.get("target_texture_blend", 0.0))
            reference_meta = {}
            reference_input_path = None
            if reference_generation_dir is not None:
                reference_meta_path = reference_generation_dir / "metadata" / f"{sid}.json"
                reference_input_path = reference_generation_dir / "inputs" / f"{sid}.nii.gz"
                if not reference_meta_path.exists() or not reference_input_path.exists():
                    raise FileNotFoundError(
                        f"reference source plan is incomplete for {sid}: "
                        f"metadata={reference_meta_path.exists()}, input={reference_input_path.exists()}"
                    )
                reference_meta = load_json(reference_meta_path)
                reference_target_id = str(reference_meta.get("target_id") or "")
                if not reference_target_id or reference_target_id not in target_meta_by_id:
                    raise ValueError(
                        f"reference target_id for {sid} is unavailable in current target library: "
                        f"{reference_target_id or 'missing'}"
                    )
                normal_path = reference_input_path
                target_meta = dict(target_meta_by_id[reference_target_id])
                target_plan_source = "reference_generation"
                target_plan_index = int(target_index_by_id[reference_target_id])
                target_plan_strategy = "reference_generation"
            else:
                injection = injection_lookup.get(i)
                if injection:
                    target_plan_strategy, target_plan_seed = _target_order_key_for_injection(injection, self.method_config)
                    cache_key = (target_plan_strategy, target_plan_seed)
                    if cache_key not in injection_order_cache:
                        injection_order_cache[cache_key] = _order_target_metas(
                            all_target_metas, target_root, target_plan_strategy, target_plan_seed
                        )
                    injection_target_metas = injection_order_cache[cache_key]
                    try:
                        target_plan_index = int(injection.get("target_index", target_plan_index)) % len(injection_target_metas)
                    except Exception:
                        target_plan_index = i % len(injection_target_metas)
                    target_meta = load_json(injection_target_metas[target_plan_index])
                    target_plan_source = "injection"
                    if "target_texture_blend" in injection:
                        texture_blend = float(injection.get("target_texture_blend"))
                else:
                    target_meta = load_json(target_metas[target_plan_index])
            target_id = target_meta["target_id"]
            sample_transfer_role = _validate_posthoc_target_transfer_role(
                generation_split,
                texture_blend,
                self.method_config.get("subtype_hu_shift", {}),
            )
            mask = load_volume(target_root / target_meta["mask_path"])
            hist = np.load(target_root / target_meta["hist_path"]).astype(np.float32)
            normal_suitability_score = ""
            normal_sampling_strategy = str(
                self.method_config.get("normal_sampling_strategy", "data_config_order")
            )
            if (
                normal_sampling_strategy == "target_mask_lung_suitability"
                and reference_generation_dir is None
            ):
                normal_path, normal, normal_suitability_score = self._select_suitable_normal(
                    normal_images, mask, logical_index
                )
            else:
                normal = self._preprocess_source(load_volume(normal_path))
            used_official_prior = False
            prior_error = ""
            maisi_runtime_compat = ""
            ctflow_runtime_compat = ""
            prior_candidate_hist_jsd = ""
            prior_candidate_lesion_mean_hu = ""
            prior_candidate_lesion_std_hu = ""
            prior_candidate_path = ""
            texture_rank_by_subtype = self.method_config.get(
                "projection_training_texture_rank_strength_by_subtype", {}
            )
            subtype_key = str(target_meta.get("subtype") or "default").lower().replace("-", "_")
            texture_rank_strength = float(
                texture_rank_by_subtype.get(
                    subtype_key,
                    texture_rank_by_subtype.get(
                        "default",
                        self.method_config.get("projection_training_texture_rank_strength", 0.0),
                    ),
                )
            )
            prior_name = self._prior_name()
            latent_texture_weight = float(
                self.method_config.get(
                    "maisi_latent_lambda_texture"
                    if prior_name == "maisi_rflow"
                    else "ctflow_latent_lambda_texture",
                    0.0,
                )
            )
            image_texture_config = dict(self.method_config)
            for texture_key in (
                "image_texture_refinement_steps",
                "image_texture_refinement_lr",
                "image_texture_lambda",
                "image_texture_anchor_lambda",
                "image_texture_rank_blend",
                "image_texture_optimized_value_blend",
                "image_texture_core_erosion",
                "image_texture_energy_core_erosion",
                "image_texture_local_variance_weight",
                "image_texture_gram_weight",
                "image_texture_patch_swd_weight",
                "image_texture_patch_swd_sizes",
                "image_texture_patch_swd_projections",
                "image_texture_patch_swd_min_support",
                "image_texture_patch_swd_max_samples",
                "image_texture_residual_transfer_strength",
                "image_texture_residual_boundary_taper_width",
                "image_texture_residual_boundary_taper_power",
                "image_texture_residual_max_delta_hu",
                "image_texture_residual_mode",
                "image_texture_phase_transfer_strength",
                "image_texture_phase_std_blend",
                "image_texture_phase_boundary_taper_width",
                "image_texture_phase_boundary_taper_power",
                "image_texture_phase_max_delta_hu",
                "image_texture_gradient_transport_strength",
                "image_texture_gradient_std_blend",
                "image_texture_gradient_screening",
                "image_texture_gradient_histogram_blend",
                "image_texture_gradient_max_delta_hu",
                "image_texture_gradient_max_iterations",
                "image_texture_gradient_tolerance_hu",
                "image_texture_gradient_relaxation",
            ):
                values_by_subtype = self.method_config.get(
                    f"{texture_key}_by_subtype", {}
                )
                if values_by_subtype:
                    image_texture_config[texture_key] = values_by_subtype.get(
                        subtype_key,
                        values_by_subtype.get(
                            "default", self.method_config.get(texture_key)
                        ),
                    )
            image_texture_steps = int(
                image_texture_config.get("image_texture_refinement_steps", 0)
            )
            image_texture_residual_strength = float(
                image_texture_config.get(
                    "image_texture_residual_transfer_strength", 0.0
                )
            )
            image_texture_phase_strength = float(
                image_texture_config.get(
                    "image_texture_phase_transfer_strength", 0.0
                )
            )
            image_texture_gradient_strength = float(
                image_texture_config.get(
                    "image_texture_gradient_transport_strength", 0.0
                )
            )
            if image_texture_steps < 0:
                raise ValueError(
                    f"image_texture_refinement_steps must be non-negative, got {image_texture_steps}"
                )
            if latent_texture_weight < 0:
                raise ValueError(f"latent texture weight must be non-negative, got {latent_texture_weight}")
            if image_texture_residual_strength < 0:
                raise ValueError(
                    "image_texture_residual_transfer_strength must be non-negative, got "
                    f"{image_texture_residual_strength}"
                )
            if not 0.0 <= image_texture_phase_strength <= 1.0:
                raise ValueError(
                    "image_texture_phase_transfer_strength must be in [0, 1], got "
                    f"{image_texture_phase_strength}"
                )
            if not 0.0 <= image_texture_gradient_strength <= 1.0:
                raise ValueError(
                    "image_texture_gradient_transport_strength must be in [0, 1], got "
                    f"{image_texture_gradient_strength}"
                )
            if latent_texture_weight > 0 and reuse_reference_prior:
                raise ValueError(
                    "latent texture energy cannot reuse a prior candidate because the decoder-backed "
                    "endpoint refinement would be skipped; set reuse_reference_prior_candidate=false"
                )
            texture_rank_guidance = None
            texture_rank_source = None
            latent_texture_target = None
            patch_texture_target = None
            patch_texture_masks = None
            texture_masks, texture_channel_names, texture_channel_weights = _load_texture_mask_channels(
                target_meta,
                target_root,
                mask,
                derived_channels=self.method_config.get(
                    "texture_derived_mask_channels", {}
                ),
            )
            texture_rank_source_sample_id = str(
                target_meta.get("spatial_rank_source_sample_id")
                or target_meta.get("source_sample_id")
                or ""
            )
            texture_donor_record = texture_donor_map.get(str(target_id))
            if texture_donor_record is not None:
                if not isinstance(texture_donor_record, dict):
                    raise ValueError(
                        f"invalid texture donor record for target {target_id}"
                    )
                texture_rank_source_sample_id = str(
                    texture_donor_record.get("source_sample_id") or ""
                )
            elif bool(
                self.method_config.get(
                    "projection_training_texture_donor_map_required", False
                )
            ):
                raise ValueError(
                    f"required texture donor map has no entry for target {target_id}"
                )
            texture_rank_mode = str(
                self.method_config.get(
                    "projection_training_texture_rank_mode", "single_scale"
                )
            )
            texture_detail_by_subtype = self.method_config.get(
                "projection_training_texture_detail_gain_by_subtype", {}
            )
            texture_detail_gain = float(
                texture_detail_by_subtype.get(
                    subtype_key,
                    texture_detail_by_subtype.get(
                        "default",
                        self.method_config.get(
                            "projection_training_texture_detail_gain", 1.0
                        ),
                    ),
                )
            )
            if (
                texture_rank_strength > 0
                or latent_texture_weight > 0
                or image_texture_steps > 0
                or image_texture_residual_strength > 0
                or image_texture_phase_strength > 0
                or image_texture_gradient_strength > 0
            ):
                texture_source = _load_training_texture_source(
                    target_meta,
                    data_root,
                    source_sample_id=texture_rank_source_sample_id,
                )
                latent_texture_target = _transport_texture_rank_guidance(
                    texture_source[0],
                    texture_source[1],
                    mask,
                    interpolation_order=int(
                        self.method_config.get(
                            "texture_transport_interpolation_order", 1
                        )
                    ),
                )
                patch_target_mode = str(
                    self.method_config.get(
                        "image_texture_patch_target_mode", "transported"
                    )
                )
                if patch_target_mode == "raw_train_donor":
                    patch_texture_target = texture_source[0]
                    (
                        patch_texture_masks,
                        patch_texture_channel_names,
                        patch_texture_channel_weights,
                    ) = _load_texture_mask_channels(
                        {},
                        data_root,
                        texture_source[1],
                        derived_channels=self.method_config.get(
                            "texture_derived_mask_channels", {}
                        ),
                    )
                    if patch_texture_channel_names != texture_channel_names:
                        raise ValueError(
                            "raw donor patch masks do not match destination texture channels: "
                            f"{patch_texture_channel_names} != {texture_channel_names}"
                        )
                    if not np.allclose(
                        patch_texture_channel_weights,
                        texture_channel_weights,
                        atol=1.0e-8,
                    ):
                        raise ValueError(
                            "raw donor patch mask weights do not match destination weights"
                        )
                elif patch_target_mode != "transported":
                    raise ValueError(
                        "image_texture_patch_target_mode must be transported or "
                        f"raw_train_donor, got {patch_target_mode!r}"
                    )
                if texture_rank_strength > 0:
                    if texture_rank_mode == "spectral_surrogate":
                        texture_rank_source = texture_source
                    else:
                        texture_rank_guidance = latent_texture_target.copy()
            if texture_rank_strength > 0:
                if texture_rank_mode == "multiscale_laplacian":
                    texture_rank_guidance = _multiscale_texture_rank_guidance(
                        texture_rank_guidance,
                        mask,
                        sigmas=self.method_config.get(
                            "projection_training_texture_sigmas", [0.6, 1.2, 2.4]
                        ),
                        band_weights=self.method_config.get(
                            "projection_training_texture_band_weights",
                            [0.55, 0.30, 0.15],
                        ),
                        detail_gain=texture_detail_gain,
                        lowpass_weight=float(
                            self.method_config.get(
                                "projection_training_texture_lowpass_weight", 1.0
                            )
                        ),
                    )
                elif texture_rank_mode not in {"single_scale", "spectral_surrogate"}:
                    raise ValueError(
                        "unsupported projection_training_texture_rank_mode="
                        f"{texture_rank_mode!r}"
                    )
            if prior_name == "maisi_rflow" and bool(self.method_config.get("official_prior_inference", True)):
                try:
                    reference_prior_path = (
                        reference_generation_dir / "prior_candidates" / f"{sid}.nii.gz"
                        if reference_generation_dir is not None and reuse_reference_prior
                        else None
                    )
                    if reference_prior_path is not None:
                        if not reference_prior_path.exists():
                            raise FileNotFoundError(
                                f"reusable reference prior candidate is missing: {reference_prior_path}"
                            )
                        candidate = load_volume(reference_prior_path).astype(np.float32)
                        self._last_maisi_endpoint_info = reference_meta.get(
                            "maisi_latent_endpoint",
                            {"captured": True, "domain": "maisi_rflow_latent"},
                        )
                    else:
                        candidate = self._maisi_rflow_candidate(
                            normal,
                            mask,
                            target_hist=hist,
                            texture_target=latent_texture_target,
                            texture_masks=texture_masks,
                            texture_channel_names=texture_channel_names,
                            texture_channel_weights=texture_channel_weights,
                            condition_metadata=target_meta,
                        )
                    if texture_rank_source is not None:
                        texture_rank_guidance = _spectral_surrogate_texture_rank_guidance(
                            candidate,
                            mask,
                            texture_rank_source[0],
                            texture_rank_source[1],
                            np.random.default_rng(sample_seed + 7919),
                            sigmas=self.method_config.get(
                                "projection_training_texture_sigmas", [0.6, 1.2, 2.4]
                            ),
                            detail_gain=texture_detail_gain,
                            lowpass_weight=float(
                                self.method_config.get(
                                    "projection_training_texture_lowpass_weight", 1.0
                                )
                            ),
                            candidate_phase_blend=float(
                                self.method_config.get(
                                    "projection_training_texture_candidate_phase_blend", 0.75
                                )
                            ),
                        )
                    candidate_values = np.asarray(candidate, dtype=np.float32)[mask > 0]
                    if candidate_values.size:
                        candidate_hist = compute_histogram(candidate_values)
                        prior_candidate_hist_jsd = js_divergence(candidate_hist, hist)
                        prior_candidate_lesion_mean_hu = float(np.mean(candidate_values))
                        prior_candidate_lesion_std_hu = float(np.std(candidate_values))
                    if save_prior_candidate:
                        candidate_path = gen_dir / "prior_candidates" / f"{sid}.nii.gz"
                        save_volume(candidate, candidate_path)
                        prior_candidate_path = str(candidate_path)
                    image, edit_mask = self._project_prior_candidate(
                        normal,
                        mask,
                        hist,
                        candidate,
                        rng,
                        subtype=target_meta.get("subtype"),
                        texture_rank_guidance=texture_rank_guidance,
                    )
                    used_official_prior = True
                    maisi_runtime_compat = getattr(self, "_maisi_rflow_backend", {}).get("runtime_compat", "")
                except Exception as exc:
                    prior_error = str(exc)
                    if not bool(self.method_config.get("allow_heuristic_fallback", False)):
                        raise
                    image, edit_mask = self._refine_destination(normal, mask, hist, rng)
            elif prior_name == "ctflow" and bool(self.method_config.get("official_prior_inference", True)):
                try:
                    candidate = self._ctflow_prior_candidate(
                        normal,
                        mask=mask,
                        target_hist=hist,
                        texture_target=latent_texture_target,
                        texture_masks=texture_masks,
                        texture_channel_names=texture_channel_names,
                        texture_channel_weights=texture_channel_weights,
                        condition_metadata=target_meta,
                    )
                    if texture_rank_source is not None:
                        texture_rank_guidance = _spectral_surrogate_texture_rank_guidance(
                            candidate,
                            mask,
                            texture_rank_source[0],
                            texture_rank_source[1],
                            np.random.default_rng(sample_seed + 7919),
                            sigmas=self.method_config.get(
                                "projection_training_texture_sigmas", [0.6, 1.2, 2.4]
                            ),
                            detail_gain=texture_detail_gain,
                            lowpass_weight=float(
                                self.method_config.get(
                                    "projection_training_texture_lowpass_weight", 1.0
                                )
                            ),
                            candidate_phase_blend=float(
                                self.method_config.get(
                                    "projection_training_texture_candidate_phase_blend", 0.75
                                )
                            ),
                        )
                    candidate_values = np.asarray(candidate, dtype=np.float32)[mask > 0]
                    if candidate_values.size:
                        candidate_hist = compute_histogram(candidate_values)
                        prior_candidate_hist_jsd = js_divergence(candidate_hist, hist)
                        prior_candidate_lesion_mean_hu = float(np.mean(candidate_values))
                        prior_candidate_lesion_std_hu = float(np.std(candidate_values))
                    if save_prior_candidate:
                        candidate_path = gen_dir / "prior_candidates" / f"{sid}.nii.gz"
                        save_volume(candidate, candidate_path)
                        prior_candidate_path = str(candidate_path)
                    image, edit_mask = self._project_prior_candidate(
                        normal,
                        mask,
                        hist,
                        candidate,
                        rng,
                        subtype=target_meta.get("subtype"),
                        texture_rank_guidance=texture_rank_guidance,
                    )
                    used_official_prior = True
                    ctflow_runtime_compat = getattr(self, "_ctflow_backend", {}).get("runtime_compat", "")
                except Exception as exc:
                    prior_error = str(exc)
                    if not bool(self.method_config.get("allow_heuristic_fallback", False)):
                        raise
                    image, edit_mask = self._refine_destination(normal, mask, hist, rng)
            else:
                image, edit_mask = self._refine_destination(normal, mask, hist, rng)
            image = _apply_target_texture_blend(image, mask, target_meta, data_root, texture_blend)
            subtype_shift = _subtype_hu_shift(self.method_config, target_meta.get("subtype", "unknown"))
            image = _apply_masked_hu_shift(image, mask, subtype_shift)
            local_contrast_target = float(self.method_config.get("local_contrast_target", 0.0))
            local_contrast_strength = float(self.method_config.get("local_contrast_strength", 0.0))
            local_contrast_shell_width = int(self.method_config.get("local_contrast_shell_width", 2))
            local_contrast_max_shift = float(self.method_config.get("local_contrast_max_shift", 120.0))
            image = _apply_local_contrast_match(
                image,
                mask,
                local_contrast_target,
                strength=local_contrast_strength,
                shell_width=local_contrast_shell_width,
                max_shift=local_contrast_max_shift,
            )
            perilesion_shell_target_mean = float(self.method_config.get("perilesion_shell_target_mean", -700.0))
            perilesion_shell_strength = float(self.method_config.get("perilesion_shell_strength", 0.0))
            perilesion_shell_width = int(self.method_config.get("perilesion_shell_width", 1))
            perilesion_shell_max_shift = float(self.method_config.get("perilesion_shell_max_shift", 120.0))
            image = _apply_perilesion_shell_match(
                image,
                mask,
                perilesion_shell_target_mean,
                strength=perilesion_shell_strength,
                shell_width=perilesion_shell_width,
                max_shift=perilesion_shell_max_shift,
            )
            equivalent_diameter_mm = float(
                (6.0 * float(np.count_nonzero(mask)) / np.pi) ** (1.0 / 3.0)
            )
            boundary_reference_blend, size_key, boundary_subtype_multiplier = (
                _boundary_reference_blend_for_condition(
                    image_texture_config,
                    equivalent_diameter_mm,
                    target_meta.get("subtype", "default"),
                )
            )
            boundary_reference_width = int(self.method_config.get("boundary_reference_width", 1))
            boundary_width_by_size = self.method_config.get(
                "boundary_reference_width_by_size", {}
            )
            if boundary_width_by_size:
                boundary_reference_width = int(
                    boundary_width_by_size.get(
                        size_key,
                        boundary_width_by_size.get("default", boundary_reference_width),
                    )
                )
            boundary_reference_profile = str(
                self.method_config.get("boundary_reference_profile", "flat")
            )
            boundary_reference_power = float(
                self.method_config.get("boundary_reference_power", 1.0)
            )
            if boundary_reference_profile == "distance_ramp":
                image = _apply_inner_distance_reference_blend(
                    image,
                    normal,
                    mask,
                    boundary_reference_blend,
                    width=boundary_reference_width,
                    power=boundary_reference_power,
                )
            elif boundary_reference_profile == "flat":
                image = _apply_inner_boundary_reference_blend(
                    image,
                    normal,
                    mask,
                    boundary_reference_blend,
                    width=boundary_reference_width,
                )
            else:
                raise ValueError(
                    "unsupported boundary_reference_profile="
                    f"{boundary_reference_profile!r}"
                )
            boundary_core_histogram_compensation = bool(
                self.method_config.get("boundary_core_histogram_compensation", False)
            )
            boundary_core_histogram_strength = float(
                self.method_config.get("boundary_core_histogram_strength", 1.0)
            )
            boundary_core_histogram_rank_guidance = str(
                self.method_config.get(
                    "boundary_core_histogram_rank_guidance", "current_image"
                )
            )
            if boundary_core_histogram_compensation:
                core_rank_guidance = None
                if boundary_core_histogram_rank_guidance == "training_texture":
                    if texture_rank_guidance is None:
                        raise ValueError(
                            "training_texture core histogram guidance requires a "
                            "verified training texture donor"
                        )
                    core_rank_guidance = texture_rank_guidance
                elif boundary_core_histogram_rank_guidance != "current_image":
                    raise ValueError(
                        "unsupported boundary_core_histogram_rank_guidance="
                        f"{boundary_core_histogram_rank_guidance!r}"
                    )
                image = _compensate_inner_core_histogram(
                    image,
                    mask,
                    hist,
                    boundary_reference_width,
                    strength=boundary_core_histogram_strength,
                    rank_guidance=core_rank_guidance,
                )
            histogram_bin_boundary_projection = bool(
                self.method_config.get("histogram_bin_boundary_projection", False)
            )
            histogram_bin_boundary_blend = float(
                self.method_config.get("histogram_bin_boundary_blend", 0.0)
            )
            histogram_bin_boundary_width = int(
                self.method_config.get("histogram_bin_boundary_width", 2)
            )
            histogram_bin_boundary_bins = int(
                self.method_config.get(
                    "histogram_bin_boundary_bins",
                    self.method_config.get("histogram_bins", 40),
                )
            )
            histogram_bin_boundary_hu_range = tuple(
                float(value)
                for value in self.method_config.get(
                    "histogram_bin_boundary_hu_range", [-1000.0, 400.0]
                )
            )
            if histogram_bin_boundary_projection:
                image = _apply_histogram_bin_boundary_projection(
                    image,
                    normal,
                    mask,
                    histogram_bin_boundary_blend,
                    width=histogram_bin_boundary_width,
                    bins=histogram_bin_boundary_bins,
                    hu_range=histogram_bin_boundary_hu_range,
                )
            structure_strength_by_subtype = self.method_config.get(
                "structure_guided_rank_strength_by_subtype", {}
            )
            structure_guided_rank_strength = float(
                structure_strength_by_subtype.get(
                    subtype_key,
                    structure_strength_by_subtype.get(
                        "default",
                        self.method_config.get("structure_guided_rank_strength", 0.0),
                    ),
                )
            )
            structure_guided_sigma = float(
                self.method_config.get("structure_guided_sigma", 1.4)
            )
            structure_guided_tissue_threshold = float(
                self.method_config.get("structure_guided_tissue_threshold", -650.0)
            )
            image = _apply_structure_guided_rank_projection(
                image,
                normal,
                mask,
                structure_guided_rank_strength,
                sigma=structure_guided_sigma,
                tissue_threshold=structure_guided_tissue_threshold,
            )
            final_smoothing_sigma = float(self.method_config.get("final_lesion_smoothing_sigma", 0.0))
            image = _smooth_hist_preserving(image, mask, final_smoothing_sigma)
            image_before_texture_transport = np.asarray(image, dtype=np.float32).copy()
            image_texture_info = {"status": "disabled", "steps": 0}
            if image_texture_steps > 0:
                image, image_texture_info = _refine_image_spatial_texture(
                    image,
                    latent_texture_target,
                    texture_masks,
                    image_texture_config,
                    channel_weights=texture_channel_weights,
                    patch_texture_target=patch_texture_target,
                    patch_texture_masks=patch_texture_masks,
                )
            image_texture_residual_info = {"status": "disabled", "strength": 0.0}
            if image_texture_residual_strength > 0:
                image, image_texture_residual_info = _apply_multiscale_donor_residual_transfer(
                    image,
                    latent_texture_target,
                    texture_masks,
                    image_texture_residual_strength,
                    sigmas=image_texture_config.get(
                        "image_texture_residual_sigmas", [0.6, 1.2, 2.4]
                    ),
                    band_weights=image_texture_config.get(
                        "image_texture_residual_band_weights", [1.0, 0.75, 0.5]
                    ),
                    channel_weights=texture_channel_weights,
                    boundary_taper_width=float(
                        image_texture_config.get(
                            "image_texture_residual_boundary_taper_width", 2.0
                        )
                    ),
                    boundary_taper_power=float(
                        image_texture_config.get(
                            "image_texture_residual_boundary_taper_power", 1.0
                        )
                    ),
                    max_delta_hu=float(
                        image_texture_config.get(
                            "image_texture_residual_max_delta_hu", 180.0
                        )
                    ),
                    mode=str(
                        image_texture_config.get(
                            "image_texture_residual_mode", "match"
                        )
                    ),
                )
            image_texture_phase_info = {"status": "disabled", "strength": 0.0}
            if image_texture_phase_strength > 0:
                image, image_texture_phase_info = _apply_structured_donor_patch_transport(
                    image,
                    latent_texture_target,
                    texture_masks,
                    image_texture_phase_strength,
                    channel_weights=texture_channel_weights,
                    donor_std_blend=float(
                        image_texture_config.get("image_texture_phase_std_blend", 1.0)
                    ),
                    boundary_taper_width=float(
                        image_texture_config.get(
                            "image_texture_phase_boundary_taper_width", 1.5
                        )
                    ),
                    boundary_taper_power=float(
                        image_texture_config.get(
                            "image_texture_phase_boundary_taper_power", 1.0
                        )
                    ),
                    max_delta_hu=float(
                        image_texture_config.get(
                            "image_texture_phase_max_delta_hu", 240.0
                        )
                    ),
                )
            image_texture_gradient_info = {"status": "disabled", "strength": 0.0}
            if image_texture_gradient_strength > 0:
                image, image_texture_gradient_info = _apply_gradient_domain_donor_transport(
                    image,
                    latent_texture_target,
                    texture_masks[0],
                    image_texture_gradient_strength,
                    donor_std_blend=float(
                        image_texture_config.get(
                            "image_texture_gradient_std_blend", 1.0
                        )
                    ),
                    screening=float(
                        image_texture_config.get(
                            "image_texture_gradient_screening", 0.05
                        )
                    ),
                    histogram_blend=float(
                        image_texture_config.get(
                            "image_texture_gradient_histogram_blend", 0.50
                        )
                    ),
                    max_delta_hu=float(
                        image_texture_config.get(
                            "image_texture_gradient_max_delta_hu", 240.0
                        )
                    ),
                    max_iterations=int(
                        image_texture_config.get(
                            "image_texture_gradient_max_iterations", 160
                        )
                    ),
                    tolerance_hu=float(
                        image_texture_config.get(
                            "image_texture_gradient_tolerance_hu", 0.01
                        )
                    ),
                    relaxation=float(
                        image_texture_config.get(
                            "image_texture_gradient_relaxation", 0.80
                        )
                    ),
                )
            post_boundary_restore_info = {
                "status": "disabled",
                "width": 0,
                "strength": 0.0,
            }
            post_boundary_restore_width = int(
                image_texture_config.get(
                    "image_texture_post_boundary_restore_width", 0
                )
            )
            post_boundary_restore_strength = float(
                image_texture_config.get(
                    "image_texture_post_boundary_restore_strength", 0.0
                )
            )
            if post_boundary_restore_width < 0:
                raise ValueError("image_texture_post_boundary_restore_width must be non-negative")
            if not 0.0 <= post_boundary_restore_strength <= 1.0:
                raise ValueError(
                    "image_texture_post_boundary_restore_strength must be in [0, 1]"
                )
            if post_boundary_restore_width > 0 and post_boundary_restore_strength > 0:
                restore_band = _inner_boundary(mask, post_boundary_restore_width)
                image[restore_band] = (
                    (1.0 - post_boundary_restore_strength) * image[restore_band]
                    + post_boundary_restore_strength
                    * image_before_texture_transport[restore_band]
                )
                post_boundary_restore_info = {
                    "status": "applied",
                    "width": post_boundary_restore_width,
                    "strength": post_boundary_restore_strength,
                    "num_voxels": int(np.count_nonzero(restore_band)),
                }
            final_target_histogram_reprojection = bool(
                self.method_config.get("final_target_histogram_reprojection", False)
            )
            final_target_histogram_reprojection_strength = float(
                self.method_config.get("final_target_histogram_reprojection_strength", 1.0)
            )
            if final_target_histogram_reprojection:
                image = _apply_final_target_histogram_reprojection(
                    image,
                    mask,
                    hist,
                    strength=final_target_histogram_reprojection_strength,
                )
            save_volume(image, image_path)
            save_volume(mask, mask_path)
            save_volume(normal, input_path)
            actual = compute_histogram(image[(mask > 0)])
            meta = {
                "sample_id": sid,
                "method_name": self.name,
                "method": "training_free_lesion_constrained_flow_matching",
                "method_id": self._method_id(),
                "implementation": "training_free_lesion_constrained_flow_matching",
                "solver_fidelity": self._solver_fidelity(),
                "refinement_domain": "latent_endpoint_plus_image_projection"
                if self._uses_latent_stepwise_solver()
                else "image_space",
                "latent_endpoint_solver": self._uses_latent_stepwise_solver(),
                "trajectory_reprojection_status": self._trajectory_reprojection_status(),
                "official_prior_inference": used_official_prior,
                "official_prior_error": prior_error,
                "maisi_runtime_compat": maisi_runtime_compat,
                "ctflow_runtime_compat": ctflow_runtime_compat,
                "ctflow_source_conditioning": getattr(
                    self,
                    "_last_ctflow_conditioning_info",
                    {
                        "source_latent_mode": "not_run",
                        "formal_same_input_conditioning": False,
                    },
                ),
                "ctflow_latent_endpoint": getattr(
                    self,
                    "_last_ctflow_endpoint_info",
                    {
                        "captured": False,
                        "domain": "ctflow_stdit_latent",
                    },
                ),
                "maisi_latent_endpoint": getattr(
                    self,
                    "_last_maisi_endpoint_info",
                    {"captured": False, "domain": "maisi_rflow_latent"},
                ),
                "frozen_prior": self._prior_name(),
                "frozen_prior_root": str(prior_root),
                "frozen_prior_weight_root": str(self._maisi_weight_root()) if self._prior_name() != "ctflow" and self._maisi_weight_root() else "",
                "generator_finetuning": False,
                "trainable_generator_parameters": 0,
                "input_split": reference_meta.get("input_split", generation_split),
                "input_case_id": reference_meta.get("input_case_id")
                or normal_path.name.replace(".nii.gz", ""),
                "reference_generation_dir": str(reference_generation_dir) if reference_generation_dir else "",
                "reference_sample_id": reference_meta.get("sample_id", sid) if reference_meta else "",
                "source_plan_status": "matched_reference_generation" if reference_meta else "data_config_order",
                "normal_sampling_strategy": normal_sampling_strategy,
                "normal_suitability_score": normal_suitability_score,
                "target_id": target_id,
                "subtype": target_meta.get("subtype", "unknown"),
                "target_subtype": target_meta.get("subtype", "unknown"),
                "diameter_mm": target_meta.get("diameter_mm", ""),
                "volume_mm3": target_meta.get("volume_mm3", ""),
                "lobe": target_meta.get("lobe", ""),
                "location_type": target_meta.get("location_type", ""),
                "pleural_distance_mm": target_meta.get("pleural_distance_mm", ""),
                "hist_path": str(target_root / target_meta["hist_path"]),
                "hu_hist_path": str(target_root / target_meta["hist_path"]),
                "mask_path": str(target_root / target_meta["mask_path"]),
                "image_path": str(image_path),
                "input_path": str(input_path),
                "target_histogram": hist.tolist(),
                "actual_histogram": actual.tolist(),
                "target_actual_hist_jsd": js_divergence(actual, hist),
                "prior_candidate_target_hist_jsd": prior_candidate_hist_jsd,
                "prior_candidate_lesion_mean_hu": prior_candidate_lesion_mean_hu,
                "prior_candidate_lesion_std_hu": prior_candidate_lesion_std_hu,
                "prior_candidate_path": prior_candidate_path,
                "prior_candidate_reused_from_reference": bool(
                    reuse_reference_prior and reference_generation_dir is not None
                ),
                "projection_training_texture_rank_strength": texture_rank_strength,
                "projection_training_texture_rank_source": texture_rank_source_sample_id
                if texture_rank_strength > 0
                else "",
                "latent_texture_energy_weight": latent_texture_weight,
                "latent_texture_energy_source": texture_rank_source_sample_id
                if latent_texture_weight > 0
                else "",
                "latent_texture_energy_channels": texture_channel_names
                if latent_texture_weight > 0
                else [],
                "latent_texture_energy_channel_weights": texture_channel_weights
                if latent_texture_weight > 0
                else [],
                "latent_texture_energy_role": (
                    "verified_train_exemplar_multiscale_3d_band_autocorrelation"
                    if latent_texture_weight > 0
                    else "disabled"
                ),
                "image_texture_refinement": image_texture_info,
                "image_texture_residual_transfer": image_texture_residual_info,
                "image_texture_phase_transport": image_texture_phase_info,
                "image_texture_gradient_transport": image_texture_gradient_info,
                "image_texture_post_boundary_restore": post_boundary_restore_info,
                "texture_transport_interpolation_order": int(
                    self.method_config.get("texture_transport_interpolation_order", 1)
                ),
                "final_target_histogram_reprojection": final_target_histogram_reprojection,
                "final_target_histogram_reprojection_strength": (
                    final_target_histogram_reprojection_strength
                ),
                "projection_training_texture_rank_mode": texture_rank_mode,
                "projection_training_texture_detail_gain": texture_detail_gain,
                "projection_training_texture_donor_map": str(texture_donor_map_path)
                if texture_donor_map_path is not None
                else "",
                "projection_training_texture_donor_spacing_mm": (
                    texture_donor_record.get("source_spacing_mm", [])
                    if texture_donor_record
                    else []
                ),
                "projection_training_texture_rank_role": (
                    "verified_train_exemplar_spectral_energy_no_spatial_phase"
                    if texture_rank_strength > 0
                    and texture_rank_mode == "spectral_surrogate"
                    else
                    "verified_train_exemplar_multiscale_texture_energy"
                    if texture_rank_strength > 0
                    and texture_rank_mode == "multiscale_laplacian"
                    else "verified_train_exemplar_spatial_rank_transport"
                    if texture_rank_strength > 0
                    and (
                        texture_donor_record
                        or target_meta.get("spatial_rank_source_sample_id")
                    )
                    else "verified_train_exemplar_spatial_rank_only"
                    if texture_rank_strength > 0
                    else "disabled"
                ),
                "lambda_bg": float(self.method_config.get("lambda_bg", 1.0)),
                "lambda_hu": float(self.method_config.get("lambda_hu", 1.0)),
                "projection_histogram_strength": float(self.method_config.get("projection_histogram_strength", 1.0)),
                "projection_mode": str(self.method_config.get("projection_mode", "rank_histogram")),
                "projection_final_histogram_strength": float(
                    self.method_config.get("projection_final_histogram_strength", 0.0)
                ),
                "sampling_steps": int(self.method_config.get("sampling_steps", 16)),
                "refinement_steps": int(self.method_config.get("refinement_steps", 4)),
                "boundary_dilation": int(self.method_config.get("boundary_dilation", 2)),
                "lesion_smoothing_sigma": float(self.method_config.get("lesion_smoothing_sigma", 0.0)),
                "boundary_reference_blend": boundary_reference_blend,
                "boundary_reference_subtype_multiplier": boundary_subtype_multiplier,
                "boundary_reference_size_bin": size_key,
                "equivalent_diameter_mm": equivalent_diameter_mm,
                "boundary_reference_width": boundary_reference_width,
                "histogram_bin_boundary_projection": histogram_bin_boundary_projection,
                "histogram_bin_boundary_blend": histogram_bin_boundary_blend,
                "histogram_bin_boundary_width": histogram_bin_boundary_width,
                "histogram_bin_boundary_bins": histogram_bin_boundary_bins,
                "histogram_bin_boundary_hu_range": list(
                    histogram_bin_boundary_hu_range
                ),
                "boundary_reference_profile": boundary_reference_profile,
                "boundary_reference_power": boundary_reference_power,
                "boundary_core_histogram_compensation": boundary_core_histogram_compensation,
                "boundary_core_histogram_strength": boundary_core_histogram_strength,
                "boundary_core_histogram_rank_guidance": boundary_core_histogram_rank_guidance,
                "final_lesion_smoothing_sigma": final_smoothing_sigma,
                "structure_guided_rank_strength": structure_guided_rank_strength,
                "structure_guided_sigma": structure_guided_sigma,
                "structure_guided_tissue_threshold": structure_guided_tissue_threshold,
                "local_contrast_target": local_contrast_target,
                "local_contrast_strength": local_contrast_strength,
                "local_contrast_shell_width": local_contrast_shell_width,
                "local_contrast_max_shift": local_contrast_max_shift,
                "perilesion_shell_target_mean": perilesion_shell_target_mean,
                "perilesion_shell_strength": perilesion_shell_strength,
                "perilesion_shell_width": perilesion_shell_width,
                "perilesion_shell_max_shift": perilesion_shell_max_shift,
                "target_sampling_strategy": str(self.method_config.get("target_sampling_strategy", "sequential")),
                "target_plan_source": target_plan_source,
                "target_plan_index": target_plan_index,
                "target_plan_strategy": target_plan_strategy,
                "target_plan_seed": target_plan_seed,
                "generation_sample_seed": sample_seed,
                "generation_sample_offset": generation_sample_offset,
                "generation_logical_index": logical_index,
                "generation_rng_policy": "independent_base_seed_plus_sample_index",
                "target_texture_blend": texture_blend,
                "subtype_hu_shift": subtype_shift,
                "posthoc_target_transfer_role": sample_transfer_role["role"],
                "posthoc_target_transfer_policy": (
                    "real_target_voxel_transfer_allowed_only_for_train_augmentation"
                ),
                "edit_mask_voxels": int(edit_mask.sum()),
            }
            save_json(meta, meta_path)
            rows.append(meta)
            if bool(self.method_config.get("clear_cuda_cache_between_samples", False)):
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        write_jsonl(rows, gen_dir / "metrics_ready.jsonl")
        print(f"[{self._method_id()}] generated {len(rows)} training-free samples under {gen_dir}")
