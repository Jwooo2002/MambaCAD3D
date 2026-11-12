# Independent centerline preprocessing audit

Date: 2026-11-07. Scope: source geometry, line connectivity, path extraction, physical resampling and moving frames. Original data and historical preprocessing code were read only. No disease metric was used to select parameters, paths, or patients.

## Decision and physical sampling contract

Use a separately versioned corrected preprocessing pipeline. The final fixed physical sampling is 32 × 32 pixels at **0.25 mm in-plane pitch (8 × 8 mm pixel-support FOV)**, 64 planes at 0.5 mm source arc-length spacing, and stride 32 planes (16 mm). The first-to-last plane-center span is **31.5 mm**, not 32 mm. The even 32-pixel grid uses `(arange(32) - 15.5) * 0.25`, i.e. centers −3.875…3.875 mm. This interpolation does not increase native CT resolution. An initial 16 mm FOV proposal prioritized isotropic sampling but had no evidence of superiority and was replaced before full preprocessing or corrected-model scoring: 8 mm stays closer to the legacy approximately 7 mm field in Normal_10 and avoids an unnecessary doubling of the context width. Four exploratory 16 mm geometry samples are archived separately and are not model inputs.

This is a methodological change, not a claim to reproduce legacy inputs. Legacy `size=8` in `plane_v5.py` means ±8 **voxel coordinates** with an inclusive grid; physical FOV depends on spacing and plane orientation. Legacy code also rounds physical centers to integer voxel indices before estimating tangents and planes. Corrected paths and frames must remain in LPS millimetres and sampling should transform the final plane grid to continuous CT indices using the full origin/direction/spacing. This avoids anisotropic-voxel plane distortion and center quantization.

Fix FOV, length, spacing, intensity processing, short-path handling, window aggregation and topology-repair policy before external outcome analysis. If paths shorter than 64 samples are padded, retain valid depth and label the repeated terminal samples as padding; they are not another 31.5 mm of anatomy. The helper never pads or extrapolates.

## Direct proof of the historical ordering error

Source: `/home/jiwoong/PycharmProjects/journal_preprocess/ASOCADataAccess`.

* `Normal/Centerlines/Normal_10.vtp` contains **14,837 points and 10 polyline cells**. The first cell contains 989 IDs and starts **[14818, 986, 985, 984, 983]**, ending **[3, 2, 1, 0, 14817]**. Thus `GetPoints()` storage order is not the vessel path. Correct cell traversal gives a first-cell median adjacent step of 0.0729525 mm and maximum 1.745178 mm. Across its 10 cells medians are approximately 0.0622–0.0730 mm.
* The 10 source cells have two identical coordinate starts repeated in 6 and 4 cells. They encode duplicated proximal prefixes on full root-to-tip trajectories. Most overlapping samples have different VTK point IDs. A graph based only on raw IDs would retain 10 separate components; blindly merging all equal coordinates could create false joins at crossings. The implemented ordered-prefix trie merges only shared proximal history.
* `sort_cnt_2.py` lines 51–53 convert branches to coordinate sets and intersect them; line 66 also converts a trimmed branch to a set. Line 73 subtracts `seen_points`; line **78** creates `new_points_list = [list(p) for p in new_unique_points]`. That is the exact order-loss operation. It retains membership, not the source trajectory. A set may iterate reproducibly on one runtime and still has no anatomical ordering guarantee.
* The first five centers in `vessel2_preprocessed/centerlines_corrected_Normal_10_processed_cumulative.pkl` have consecutive distances **31.655826, 16.947241, 7.988608, 19.362424 mm**. The earlier full first-patch reproduction recorded in `ASOCA_PREPROCESSING_FINDING.md` obtained MSE = 0 with the historical sampler and `size=8,resolution=32`. That reproduction is prior recorded evidence; this audit independently inspected the source cell IDs and remeasured the legacy coordinate distances.
* The alternative `ASOCA_preprocess_linvol_v1/.../preprocess/2_remove_overlap.py` preserves original order with list filtering (lines 89–91), but its upstream `1_align.py` estimates a translation from centerline/segmentation bounding-box centers. Preserving order alone therefore does not establish a correct physical coordinate convention. Furthermore, filtering arbitrary previously seen internal points can bridge gaps unless each remaining contiguous run is split explicitly.

Source fingerprints (SHA-256):

| Artifact | SHA-256 |
|---|---|
| Normal_10 source VTP | `b7b9b3f55252f710e8884601b32b708194974de5463289956511b17b14ceb0ac` |
| Diseased_6 source VTP | `4a122e89a8b445d844298f508a73c785c083e0a7f4ef13fc81651da006fbd8ae` |
| Normal_10 processed cumulative PKL | `11960c6885c28389bd3c1c80a0d0f462edf7f481ada4cd491ce9129eb9501c97` |
| sort_cnt_2.py | `1ffbc603eef091f3c5bd01464387da8dc6873f63d7c1c1d8790811ae30548e83` |

## ASOCA source coordinates are not directly CT LPS

For Normal_10, the raw VTP bounds are x 148.043…274.617, y 163.122…278.182, z −247.008…−147.245 mm; the CT origin is (−90.5, −130.100006, −260), spacing (0.435547, 0.435547, 0.625) and identity direction. Neither raw coordinates nor a plain RAS→LPS x/y sign flip fall inside the CT. The empirically verified source-specific transform is:

```
lps_x = source_x + 2 * ct_origin_x
lps_y = source_y + 2 * ct_origin_y
lps_z = source_z
```

This is a correction for the observed ASOCA VTP origin convention, **not a general RAS/LPS conversion**. `asoca_lps_affine` implements the translation and rejects nonidentity directions, for which this source convention has not been verified. It must not be applied to already aligned/converted VTPs or ImageCAS-X graphs. The evidence establishes the mapping on these files; it does not establish which upstream export program introduced the convention.

Read-only verification across all 40 ASOCA subjects found:

* CT and annotation size/origin/spacing/direction match in every subject.
* Corrected raw VTP points are **100% in bounds** for every subject.
* Nearest-voxel positive-lumen fractions are **98.9427489%–100%**. No bbox fit or disease information was used.
* All source polylines survive exact-prefix construction with no loops, repeated nonconsecutive coordinates, or graph cycles. Median unique-edge spacing ranges **0.0605823–0.1193183 mm** across subjects.
* 39 subjects have two distinct source roots. **Diseased_6 has three**, with source-start frequencies 1, 4, 5. Its singleton-start component begins at raw (188.273407, 223.658340, −143.317078) and is at least 14.6074 mm from the other source paths. Preserve this component and flag it; the annotation alone does not prove this point is an aortic ostium. The implementation promises source-start→terminal order, not independently verified ostial localization.

The positive-lumen check validates coordinate placement and does not demonstrate perfect centerline accuracy, plaque localization, or automatic centerline extraction.

## ImageCAS-X topology audit

All **800 patients × two sides = 1,600 graphs** were traversed using `prepared/graphs/<id>.npz` LPS coordinates, unique undirected edges and `start_points` attributes. **1,599 graphs** pass strict tree traversal. Scan **455, right** contains one genuine graph cycle. This is not a duplicate-edge artifact: it has 712 unique edges, one connected component, root ID 701 and 712 nodes.

The optional deterministic shortest-path tree removes edge [362, 411], length **0.712952159 mm**. Its alternative route is `[411,648,39,191,334,170,166,412,173,261,362]`, length **4.734218940 mm**. The total loop length is 5.447171100 mm. Both cut-edge endpoints have segment_label 10, branch_points 0, end_points 0, start_points 0. Cutting creates artificial tree terminals and must not be described as finding annotated anatomical tips.

Default behavior is to raise for review. To retain this patient in a frozen external cohort, a caller may explicitly choose `cycle_policy='shortest_path_tree'` and persist the returned audit's `cycle_edges_removed`. This is a documented topology repair, not an automatic exclusion or an inference of true anatomy. No patient is silently dropped. No other graph required cycle repair in this audit.

## Implemented API and safeguards

`scripts/centerline_paths.py` provides:

```python
from centerline_paths import (
    read_asoca_graph, read_imagecas_graph, extract_paths,
    resample_path, parallel_transport_frames,
)

graph = read_asoca_graph(source_vtp, ct.GetOrigin(), ct.GetDirection())
# Or: graph = read_imagecas_graph(prepared_npz, side='left')
audit = {}
paths = extract_paths(graph, mode='segments', audit=audit)
for path_lps in paths:
    sampled_lps, source_arc_mm = resample_path(
        path_lps, spacing_mm=0.5, return_distances=True)
    if len(sampled_lps) >= 2:
        tangent, u, v = parallel_transport_frames(sampled_lps)
```

* `CenterlineGraph` stores points, edges, roots, original polylines/attributes and provenance. Raw VTP cell IDs are preserved in ASOCA metadata. Source paths are never sorted by x/y/z, nearest neighbor, or point storage order.
* `graph_paths(...)` returns ordered ID arrays; `extract_paths(...)` returns coordinate arrays. `mode='segments'` partitions a tree into maximal root/branch-to-branch/tip paths, covering each edge exactly once. Branch endpoint coordinates appear in adjacent segments, but no edge is duplicated.
* `mode='root_to_leaf'` preserves complete source-start-to-terminal paths and deliberately repeats common proximal prefixes. If a caller chooses this mode, deduplicating identical full window coordinates removes exact repeats but not all shared proximal tissue across divergent windows. Report the policy and multiplicity; do not claim edge-disjoint sampling. Segments can be shorter and require a predeclared short-path policy.
* Consecutive identical samples are removed in source polylines. Duplicate graph edges and self-ID edges are removed. Distinct connected IDs at zero distance raise rather than inventing orientation. Nonconsecutive coordinate revisits in a polyline raise for review. Geometric crossings after divergence retain distinct IDs and are not connected. A curve crossing itself between samples is not inherently a graph connection and remains a continuous source trajectory.
* Every nontrivial graph component must have exactly one supplied/annotated root. Unknown direction, disconnected unrooted components and multiple roots are errors rather than guessed proximity order. Isolated unused points are reported.
* Cycles raise by default. `cycle_policy='shortest_path_tree'` is an explicit repair with removed-edge IDs in the audit. It retains all connected nodes, but removed edges and introduced tips require interpretation.
* `resample_path` linearly interpolates cumulative 3D source arc length. Default output has exactly 0.5 mm arc increments and omits the final residual tail (<0.5 mm). Euclidean chords may be shorter around bends; this is expected. `include_endpoint=True` is available for display but can append a shorter interval and should not be treated as a uniformly sampled model volume.
* Frames use minimal-rotation parallel transport with tangents in physical space, not a 30° tangent clamp. They are orthonormal, right-handed and continuous on smooth curves. Exact reversals and undefined tangents fail. This stabilizes roll without changing the requested plane normal. No disease-guided smoothing was introduced.

## Validation

Command:

```bash
/home/jiwoong/anaconda3/envs/journal_preprocess/bin/python -m unittest discover -s tests -p 'test_centerline_paths.py' -v
```

**13 tests passed**, including scrambled VTK-style IDs, branch edge coverage, proximal duplicate merging, crossing without a false junction, duplicate samples/polylines, loop rejection and explicit repair, root ambiguity, invalid coordinates/indices, true arc-length resampling, an orthonormal continuous helix frame, reversal rejection, ASOCA transform constraints, real Normal_10 cell IDs and lumen overlap, and real ImageCAS-X scan 68 bilateral graph/frame checks. Real-source tests skip explicitly if source data or VTK/SimpleITK are unavailable.

The full read-only population audit additionally traversed 40 ASOCA graphs and all 1,600 ImageCAS-X graphs. Per-ASOCA metrics below preserve the complete population result; the only ImageCAS-X topology exception is detailed above. Temporary machine-readable audit outputs at execution time were `/tmp/asoca_audit_results.json` and `/tmp/imagecas_path_audit_results.json`; durable evidence is recorded here.

## Per-subject ASOCA verification

All rows have in-bounds fraction 1.0, no cycle repair, and zero unused points. Points are shown before/after shared-prefix merging; segment counts are edge-disjoint maximal paths.

| Subject | Source points | Unique-prefix points | Roots | Lumen fraction | Median edge mm | Segments |
|---|---:|---:|---:|---:|---:|---:|
| Normal_1 | 14607 | 8034 | 2 | 0.999383857 | 0.064326554 | 18 |
| Normal_10 | 14837 | 8819 | 2 | 0.999865202 | 0.064944970 | 18 |
| Normal_11 | 14500 | 9473 | 2 | 0.999448276 | 0.062295075 | 22 |
| Normal_12 | 7445 | 5076 | 2 | 0.999059772 | 0.064103524 | 10 |
| Normal_13 | 13399 | 9619 | 2 | 0.998955146 | 0.062835414 | 16 |
| Normal_14 | 12424 | 9434 | 2 | 0.999839021 | 0.064166651 | 14 |
| Normal_15 | 7642 | 5778 | 2 | 0.999476577 | 0.064411207 | 10 |
| Normal_16 | 8556 | 6296 | 2 | 0.999766246 | 0.060582256 | 12 |
| Normal_17 | 8634 | 6701 | 2 | 0.999305073 | 0.063013313 | 14 |
| Normal_18 | 15339 | 9413 | 2 | 0.998956907 | 0.064130576 | 24 |
| Normal_19 | 8691 | 5872 | 2 | 0.999079508 | 0.064887907 | 12 |
| Normal_2 | 7970 | 6474 | 2 | 0.999623588 | 0.064913946 | 14 |
| Normal_20 | 12073 | 7046 | 2 | 0.999751512 | 0.064735248 | 18 |
| Normal_3 | 15091 | 9070 | 2 | 0.998939765 | 0.062306232 | 22 |
| Normal_4 | 9018 | 6635 | 2 | 0.999445553 | 0.065376185 | 10 |
| Normal_5 | 9224 | 7222 | 2 | 0.999132697 | 0.064269369 | 14 |
| Normal_6 | 7865 | 5366 | 2 | 0.999745709 | 0.061495095 | 8 |
| Normal_7 | 11636 | 8112 | 2 | 0.999054658 | 0.064210161 | 16 |
| Normal_8 | 15360 | 8251 | 2 | 0.998958333 | 0.065673165 | 20 |
| Normal_9 | 10100 | 6911 | 2 | 0.999504950 | 0.063315615 | 14 |
| Diseased_1 | 4994 | 4387 | 2 | 0.999199039 | 0.063700349 | 6 |
| Diseased_10 | 6081 | 5125 | 2 | 0.999177767 | 0.061725563 | 8 |
| Diseased_11 | 5013 | 3873 | 2 | 0.989427489 | 0.119318290 | 16 |
| Diseased_12 | 7275 | 5029 | 2 | 0.999862543 | 0.065622508 | 10 |
| Diseased_13 | 9537 | 7448 | 2 | 0.999580581 | 0.063939027 | 14 |
| Diseased_14 | 8201 | 4905 | 2 | 0.992805755 | 0.115459365 | 20 |
| Diseased_15 | 5983 | 4287 | 2 | 0.996490055 | 0.117784304 | 12 |
| Diseased_16 | 5744 | 4590 | 2 | 0.996344011 | 0.102066716 | 10 |
| Diseased_17 | 11417 | 7103 | 2 | 0.996671630 | 0.108444374 | 32 |
| Diseased_18 | 8629 | 5283 | 2 | 0.994669139 | 0.112572436 | 20 |
| Diseased_19 | 11643 | 7605 | 2 | 0.999570557 | 0.064199356 | 16 |
| Diseased_2 | 7321 | 5632 | 2 | 0.998770660 | 0.063710287 | 10 |
| Diseased_20 | 8083 | 5693 | 2 | 0.999505134 | 0.097110066 | 18 |
| Diseased_3 | 7532 | 4699 | 2 | 0.998672331 | 0.113324420 | 14 |
| Diseased_4 | 10373 | 7233 | 2 | 0.998939555 | 0.064202970 | 14 |
| Diseased_5 | 12878 | 7531 | 2 | 0.998214008 | 0.068219787 | 20 |
| Diseased_6 | 12096 | 6191 | 3 | 1.000000000 | 0.075283275 | 17 |
| Diseased_7 | 11409 | 6641 | 2 | 0.999474099 | 0.065348461 | 16 |
| Diseased_8 | 12698 | 8222 | 2 | 0.999291227 | 0.064478214 | 18 |
| Diseased_9 | 12404 | 7923 | 2 | 0.999274428 | 0.064609695 | 14 |

## Raw source-ID inspection example

```python
from centerline_paths import read_vtk_graph, asoca_lps_affine
raw = read_vtk_graph(source_vtp)  # identity coordinates: raw source units retained
first_source_ids = raw.polylines[0]
raw_ordered_coordinates = raw.points[first_source_ids]
physical = read_vtk_graph(source_vtp, asoca_lps_affine(ct.GetOrigin()))
lps_ordered_coordinates = physical.points[first_source_ids]
```

Here the identity-affine `raw.points` is used only to inspect source coordinates; it must not be treated as CT LPS merely because the generic graph metadata says LPS. For model sampling always supply the verified source affine or use `read_asoca_graph`. The generic reader intentionally cannot identify coordinate conventions from VTK data alone; passing identity is the caller's responsibility. `read_asoca_graph(...).metadata['source_point_ids_per_cell']` preserves the original source cell IDs even though the resulting trie has new node IDs. The example deliberately distinguishes these two ID spaces.

## Independent model-code observations communicated to integration

Two separate issues were confirmed by reading `main_files/model.py` before the integration agent changed it. Grouped `x_proj` stores channels as `[group0 dt/B/C, group1 dt/B/C]`; splitting the entire channel axis into `[K*rank,K*state,K*state]` mixes the directions' parameter fields. Correct extraction reshapes to `[batch,K,rank+2*state,length]`, splits within the third axis and flattens only dt before its grouped projection. Existing checkpoints must remain associated with their original implementation.

The historical memory layer applies `hardtanh(attention,0,1)` to softmax weights, which are already in [0,1]; it therefore performs no sparsification and never uses the numerical threshold. The intended thresholded weighting can use `relu(w-lambda)*w/(abs(w-lambda)+eps)` followed by L1 renormalization. All-zero rows require an explicit policy and monitoring. With the revision's stated memory size 300 and threshold 0.0025, the threshold lies below uniform attention 1/300, unlike the historical constructor's 2000-slot case. This changes the risk of all-zero rows but does not eliminate the need to measure them. These fixes are implemented and tested by the integration agent, outside this audit's file ownership; they must be identified as a new model version alongside the preprocessing changes.

Final regression addition: ImageCAS-X **455 right** now has an optional real-source test verifying strict rejection, exact recorded removed edge [362,411], retention of every node, N−1 edge coverage and valid resampled moving frames after explicit repair. Final helper test count: **14 passed**.
