"""Sample vessel-normal CT planes in physical space, identically across cohorts."""

import hashlib

import numpy as np
from scipy.ndimage import map_coordinates

from scripts.centerline_paths import parallel_transport_frames, resample_path


def physical_to_indices(points_lps, origin, spacing, direction):
    points = np.asarray(points_lps, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    if points.shape[-1] != 3 or spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("Expected xyz physical points and positive xyz spacing")
    return ((points - np.asarray(origin)) @ np.linalg.inv(matrix).T) / spacing


def sample_planes(volume_zyx, centers, u, v, origin, spacing, direction,
                  fov_mm=16.0, resolution=32, chunk_size=128):
    """Pixel centers cover a fixed physical field; linear interpolation in HU.

    A 16 mm field with 32 pixels has centers -7.75,...,+7.75 mm. Image direction
    is applied before indexing the SimpleITK z,y,x array. Outside CT is -300 HU.
    """
    if resolution < 2 or fov_mm <= 0 or chunk_size < 1:
        raise ValueError("Invalid plane sampling dimensions")
    volume = np.asarray(volume_zyx)
    centers, u, v = [np.asarray(a, dtype=np.float64) for a in (centers, u, v)]
    if volume.ndim != 3 or centers.shape != u.shape or u.shape != v.shape:
        raise ValueError("Expected 3D CT and matching center/frame arrays")
    offset = (np.arange(resolution) + 0.5 - resolution / 2) * (fov_mm / resolution)
    xx, yy = np.meshgrid(offset, offset, indexing="xy")
    planes = np.empty((len(centers), resolution, resolution), dtype=np.float32)
    outside, total = 0, 0
    size_xyz = np.asarray(volume.shape[::-1])
    for start in range(0, len(centers), chunk_size):
        stop = min(start + chunk_size, len(centers))
        coordinates = (centers[start:stop, None, None, :] +
                       xx[None, :, :, None] * u[start:stop, None, None, :] +
                       yy[None, :, :, None] * v[start:stop, None, None, :])
        indices = physical_to_indices(coordinates, origin, spacing, direction)
        outside += int(((indices < 0) | (indices > size_xyz - 1)).any(axis=-1).sum())
        total += int(np.prod(indices.shape[:-1]))
        planes[start:stop] = map_coordinates(
            volume, np.moveaxis(indices[..., ::-1], -1, 0), order=1,
            mode="constant", cval=-300.0, prefilter=False)
    return planes, outside / total


def window_starts(count, depth=64, stride=32):
    if count < 1 or depth < 1 or not 1 <= stride <= depth:
        raise ValueError("Invalid sequence/window dimensions")
    if count <= depth:
        return [0]
    starts = list(range(0, count - depth + 1, stride))
    if starts[-1] != count - depth:
        starts.append(count - depth)
    return starts


def make_patches(volume_zyx, paths_lps, origin, spacing, direction, protocol):
    """Root-to-tip windows, exact geometric duplicate windows removed per scan.

    Short paths repeat their final plane. This is recorded as valid_depth, rather
    than extrapolating anatomy. Proximal tissue can still occur in different
    bifurcation windows; only identical full window geometry is deduplicated.
    """
    patches, rows, geometry, offsets = [], [], [], [0]
    seen, duplicate_windows, short_paths = set(), 0, 0
    outside_weighted, plane_count, steps = 0.0, 0, []
    depth, stride = protocol["depth"], protocol["stride"]
    frame_arrays = {"tangent_lps": [], "u_lps": [], "v_lps": []}
    path_rows = []
    for path_index, path in enumerate(paths_lps):
        sampled = resample_path(path, spacing_mm=protocol["longitudinal_spacing_mm"])
        if len(sampled) < 2:
            short_paths += 1
            continue
        tangent, u, v = parallel_transport_frames(sampled)
        planes, outside = sample_planes(
            volume_zyx, sampled, u, v, origin, spacing, direction,
            protocol["cross_section_fov_mm"], protocol["cross_section_pixels"])
        outside_weighted += outside * len(planes)
        plane_count += len(planes)
        steps.extend(np.linalg.norm(np.diff(sampled, axis=0), axis=1).tolist())
        geometry.append(sampled)
        for key, array in zip(frame_arrays, (tangent, u, v)):
            frame_arrays[key].append(array)
        path_rows.append({"path_index": path_index, "geometry_start": offsets[-1],
                          "point_count": len(sampled),
                          "source_arc_length_mm": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                          "sampled_arc_extent_mm": (len(sampled) - 1) * protocol["longitudinal_spacing_mm"]})
        offsets.append(offsets[-1] + len(sampled))
        for start in window_starts(len(planes), depth, stride):
            stop = min(start + depth, len(planes))
            valid_depth = stop - start
            # Frame is included: equal centers with different rotations are not
            # interchangeable cross sections. Quantization is 0.00001 mm/unit.
            signature = np.concatenate([sampled[start:stop], u[start:stop], v[start:stop]], axis=1)
            digest = hashlib.sha256(np.round(signature, 5).astype("<f8").tobytes()).hexdigest()
            if digest in seen:
                duplicate_windows += 1
                continue
            seen.add(digest)
            patch = planes[start:stop]
            if valid_depth < depth:
                patch = np.pad(patch, ((0, depth - valid_depth), (0, 0), (0, 0)), mode="edge")
            rows.append({"patch_index": len(patches), "path_index": path_index,
                         "start_sample": start, "valid_depth": valid_depth,
                         "start_arc_mm": start * protocol["longitudinal_spacing_mm"],
                         "geometry_sha256": digest})
            patches.append(patch)
    if not patches:
        raise ValueError("No usable centerline path / patch")
    array = np.stack(patches).astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite sampled HU")
    geometry_out = {"points_lps_mm": np.concatenate(geometry),
                    "path_offsets": np.asarray(offsets, dtype=np.int64),
                    **{key: np.concatenate(values) for key, values in frame_arrays.items()}}
    qc = {"patch_count": len(array), "path_count": len(path_rows),
          "padded_patch_count": sum(row["valid_depth"] < depth for row in rows),
          "duplicate_windows_removed": duplicate_windows, "unusable_subspacing_paths": short_paths,
          "plane_outside_ct_fraction": outside_weighted / plane_count,
          "resampled_adjacent_chord_median_mm": float(np.median(steps)),
          "resampled_adjacent_chord_max_mm": float(np.max(steps)),
          "hu_min": float(array.min()), "hu_max": float(array.max())}
    return array, rows, path_rows, geometry_out, qc
