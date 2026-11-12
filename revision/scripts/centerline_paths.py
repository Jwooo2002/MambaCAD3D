"""Connectivity-preserving centerlines in LPS millimetres.

No nearest-neighbour sorting or spatial merging is performed. Use source IDs for
VTK graphs, and a prefix trie for ASOCA's duplicated root-to-tip polylines. The
latter merges only identical ordered proximal prefixes, never geometric crossings.
"""
from dataclasses import dataclass, field
import heapq
import json
from pathlib import Path
import numpy as np


@dataclass
class CenterlineGraph:
    points: np.ndarray
    edges: np.ndarray
    roots: np.ndarray
    polylines: list = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def _points(points):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('Points must be finite [N,3] physical coordinates')
    return p


def read_vtk_graph(path, affine_lps=None, *, source_polyline_roots=False):
    """Read actual line-cell IDs, never the order in vtkPoints.

    A missing affine means the caller has verified source coordinates are LPS mm.
    Root direction must come from start_points or explicit source-polyline order.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    reader = vtk.vtkXMLPolyDataReader() if path.suffix.lower() == '.vtp' else vtk.vtkPolyDataReader()
    reader.SetFileName(str(path)); reader.Update()
    poly = reader.GetOutput()
    if poly.GetNumberOfPoints() == 0:
        raise ValueError(f'Empty polydata: {path}')
    points = _points(vtk_to_numpy(poly.GetPoints().GetData()))
    if affine_lps is not None:
        affine = np.asarray(affine_lps, dtype=float)
        if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1]):
            raise ValueError('affine_lps must be a finite affine 4x4 matrix')
        points = points @ affine[:3, :3].T + affine[:3, 3]
    lines, ids = [], vtk.vtkIdList()
    cells = poly.GetLines(); cells.InitTraversal()
    while cells.GetNextCell(ids):
        line = np.array([ids.GetId(j) for j in range(ids.GetNumberOfIds())], dtype=np.int64)
        if len(line) >= 2:
            lines.append(line)
    if not lines:
        raise ValueError('No line connectivity in source')
    edges = np.concatenate([np.column_stack((p[:-1], p[1:])) for p in lines])
    pd = poly.GetPointData()
    attributes = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)).copy()
                  for i in range(pd.GetNumberOfArrays()) if pd.GetArray(i) is not None}
    roots = np.flatnonzero(np.asarray(attributes.get('start_points', np.zeros(len(points)))).reshape(-1) > 0)
    if source_polyline_roots:
        roots = np.unique([p[0] for p in lines])
    return CenterlineGraph(points, edges, roots, lines, attributes,
                           {'source': str(path.resolve()), 'coordinate_space': 'LPS_mm',
                            'affine_lps': None if affine_lps is None else np.asarray(affine_lps).tolist()})


def read_imagecas_graph(path, side):
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    with np.load(path, allow_pickle=False) as data:
        attributes = {v['name']: data[f'{side}_point_attribute_{v["index"]}'].copy()
                      for v in json.loads(str(data[f'{side}_point_attributes_json']))}
        roots = np.flatnonzero(np.asarray(attributes['start_points']).reshape(-1) > 0)
        return CenterlineGraph(_points(data[f'{side}_points_lps_mm']).copy(),
                               data[f'{side}_edges'].copy(), roots, attributes=attributes,
                               metadata={'source': str(Path(path).resolve()), 'side': side,
                                         'coordinate_space': 'LPS_mm', 'root_source': 'start_points'})


def asoca_lps_affine(origin_lps, direction_lps=None):
    """ASOCA raw VTP correction, validated against all 40 source lumen masks.

    This is a source-specific origin-sign convention, NOT a generic RAS transform.
    It is only verified for identity-direction ASOCA images; fail on other geometry.
    """
    origin = np.asarray(origin_lps, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('origin_lps must be finite xyz')
    if direction_lps is not None and not np.allclose(np.asarray(direction_lps).reshape(3, 3), np.eye(3), atol=1e-6):
        raise ValueError('ASOCA source convention is only validated for identity image direction')
    affine = np.eye(4)
    affine[:2, 3] = 2 * origin[:2]
    return affine


def graph_from_polylines(points, polylines):
    """Merge exact shared prefixes of already oriented source root-to-tip lines.

    Coordinates at crossings AFTER divergence stay distinct. Consecutive identical
    samples are skipped; revisiting a coordinate within a path is an ambiguous
    loop/self-touch and raises. Near points are deliberately not merged.
    """
    points = _points(points)
    new_points, edges, roots, mapping, source_paths = [], [], [], {}, []
    duplicate_samples = 0
    for line in polylines:
        line = np.asarray(line, dtype=np.int64)
        if line.ndim != 1 or np.any(line < 0) or np.any(line >= len(points)):
            raise ValueError('Invalid polyline point IDs')
        parent, previous, visited, output = -1, None, set(), []
        for source_id in line:
            coordinate = tuple(points[source_id])
            if coordinate == previous:
                duplicate_samples += 1
                continue
            if coordinate in visited:
                raise ValueError('Polyline revisits a coordinate: loop or self-touch requires review')
            visited.add(coordinate)
            key = (parent, coordinate)
            if key not in mapping:
                node = len(new_points)
                mapping[key] = node
                new_points.append(points[source_id])
                if parent == -1:
                    roots.append(node)
                else:
                    edges.append((parent, node))
            parent = mapping[key]
            output.append(parent)
            previous = coordinate
        if len(output) >= 2:
            source_paths.append(np.asarray(output, dtype=np.int64))
    if not edges:
        raise ValueError('No nonzero-length polyline edges')
    return CenterlineGraph(np.asarray(new_points), np.asarray(edges, dtype=np.int64),
                           np.asarray(roots, dtype=np.int64), source_paths,
                           metadata={'merge_policy': 'exact_ordered_prefix_only',
                                     'consecutive_duplicate_samples_removed': duplicate_samples,
                                     'source_polyline_count': len(polylines)})


def read_asoca_graph(path, origin_lps, direction_lps=None):
    source = read_vtk_graph(path, asoca_lps_affine(origin_lps, direction_lps))
    graph = graph_from_polylines(source.points, source.polylines)
    graph.metadata.update(source.metadata)
    graph.metadata['root_source'] = 'ASOCA_source_cell_first_point'
    graph.metadata['source_point_count'] = len(source.points)
    graph.metadata['source_point_ids_per_cell'] = [p.tolist() for p in source.polylines]
    return graph


def graph_paths(points, edges, roots, *, mode='segments', cycle_policy='error', audit=None):
    """Return ordered point-ID arrays, oriented from annotated roots outward.

    segments: maximal root/branch-to-branch/tip paths, each edge exactly once.
    root_to_leaf: complete paths, intentionally sharing their proximal prefixes.
    Every nontrivial component requires exactly one supplied root. Cycles fail by
    default. Explicit shortest_path_tree removes non-tree edges with a QC record;
    this is a repair policy, never an automatic anatomical inference.
    """
    p = _points(points)
    raw = np.asarray(edges)
    if raw.ndim != 2 or raw.shape[1] != 2 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError('edges must be integer [E,2] point IDs')
    if np.any(raw < 0) or np.any(raw >= len(p)):
        raise ValueError('Edge point IDs outside points array')
    if mode not in ('segments', 'root_to_leaf') or cycle_policy not in ('error', 'shortest_path_tree'):
        raise ValueError('Unsupported path mode or cycle policy')
    r = np.asarray(roots)
    if r.ndim != 1 or not np.issubdtype(r.dtype, np.integer) or np.any(r < 0) or np.any(r >= len(p)):
        raise ValueError('roots must be valid integer point IDs')
    r = np.unique(r)
    # Repeated cell edges and self-IDs do not describe additional anatomy.
    e = np.unique(np.sort(raw, axis=1), axis=0)
    self_edges = int(np.sum(e[:, 0] == e[:, 1]))
    e = e[e[:, 0] != e[:, 1]]
    length = np.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=1)
    if np.any(length <= 1e-10):
        raise ValueError('Distinct connected IDs have coincident coordinates; collapse in source polylines or review')
    adjacency = [[] for _ in p]
    for (a, b), distance in zip(e, length):
        adjacency[a].append((int(b), float(distance)))
        adjacency[b].append((int(a), float(distance)))
    active = {i for i, neighbours in enumerate(adjacency) if neighbours}
    if any(int(root) not in active for root in r):
        raise ValueError('Annotated root is isolated')
    parent, children, all_roots, removed = {}, {i: [] for i in active}, [], []
    remaining = set(active)
    while remaining:
        seed = min(remaining); component = {seed}; stack = [seed]
        while stack:
            u = stack.pop()
            for v, _ in adjacency[u]:
                if v not in component:
                    component.add(v); stack.append(v)
        remaining -= component
        component_roots = sorted(component.intersection(map(int, r)))
        if len(component_roots) != 1:
            raise ValueError(f'Component has {len(component_roots)} annotated roots; expected exactly one')
        root = component_roots[0]; all_roots.append(root)
        cycle_count = sum(len(adjacency[v]) for v in component) // 2 - len(component) + 1
        if cycle_count and cycle_policy == 'error':
            raise ValueError(f'Graph has {cycle_count} independent cycles; explicit review required')
        distances = {root: 0.}; parent[root] = -1; queue = [(0., root)]
        while queue:
            distance, u = heapq.heappop(queue)
            if distance != distances[u]:
                continue
            for v, weight in sorted(adjacency[u]):
                candidate = distance + weight
                if candidate < distances.get(v, float('inf')):
                    distances[v] = candidate; parent[v] = u
                    heapq.heappush(queue, (candidate, v))
        for node in sorted(component):
            if parent[node] != -1:
                children[parent[node]].append(node)
        if cycle_count:
            for a in sorted(component):
                for b, _ in adjacency[a]:
                    if a < b and parent.get(a) != b and parent.get(b) != a:
                        removed.append([a, b])
    paths = []
    if mode == 'root_to_leaf':
        for leaf in sorted(v for v in active if not children[v]):
            sequence = [leaf]
            while parent[sequence[-1]] != -1:
                sequence.append(parent[sequence[-1]])
            paths.append(np.asarray(sequence[::-1], dtype=np.int64))
    else:
        for start in sorted(v for v in active if parent[v] == -1 or len(children[v]) != 1):
            for child in children[start]:
                sequence = [start, child]
                while len(children[sequence[-1]]) == 1:
                    sequence.append(children[sequence[-1]][0])
                paths.append(np.asarray(sequence, dtype=np.int64))
    if audit is not None:
        audit.update({'input_edges': len(raw), 'unique_nonself_edges': len(e),
                      'self_edges_removed': self_edges, 'duplicate_edges_removed': len(raw) - len(e) - self_edges,
                      'isolated_points': len(p) - len(active), 'component_count': len(all_roots),
                      'cycle_edges_removed': removed, 'mode': mode, 'path_count': len(paths)})
    return paths


def extract_paths(graph, **kwargs):
    """Convenience wrapper returning LPS-mm coordinate paths."""
    return [graph.points[ids] for ids in graph_paths(graph.points, graph.edges, graph.roots, **kwargs)]


def resample_path(points, spacing_mm=0.5, *, include_endpoint=False, return_distances=False):
    """Linear interpolation on cumulative 3D arc length, without extrapolation.

    Default samples have exactly spacing_mm arc increments and omit the residual
    tail. include_endpoint appends a shorter last interval; do not treat it as a
    regular model sample. Returned distances are source arc lengths, not chords.
    """
    p = _points(points)
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError('spacing_mm must be finite and positive')
    if len(p) < 2:
        raise ValueError('At least two distinct centerline points are required')
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-10]]
    if len(p) < 2:
        raise ValueError('Zero-length centerline')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    count = int(np.floor(arc[-1] / spacing_mm + 1e-10))
    distances = np.arange(count + 1, dtype=float) * spacing_mm
    distances = distances[distances <= arc[-1] + 1e-9]
    if include_endpoint and arc[-1] - distances[-1] > 1e-9:
        distances = np.r_[distances, arc[-1]]
    result = np.column_stack([np.interp(distances, arc, p[:, axis]) for axis in range(3)])
    return (result, distances) if return_distances else result


def parallel_transport_frames(points, initial_u=None):
    """Return tangent, u, v [N,3] orthonormal frames in physical coordinates.

    Minimal-rotation transport avoids arbitrary axis switching and preserves the
    true tangent instead of clamping it. Exact tangent reversal is ambiguous and
    raises for source review. End tangents are one-sided; interior tangents bisect
    adjacent unit segment directions.
    """
    p = _points(points)
    if len(p) < 2:
        raise ValueError('At least two points required for frames')
    segments = np.diff(p, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths < 1e-10):
        raise ValueError('Consecutive duplicate samples have undefined tangents')
    segments /= lengths[:, None]
    tangent = np.vstack((segments[0], segments[:-1] + segments[1:], segments[-1]))
    norm = np.linalg.norm(tangent, axis=1)
    if np.any(norm < 1e-8):
        raise ValueError('180-degree centerline reversal has undefined frame')
    tangent /= norm[:, None]
    u, v = np.empty_like(tangent), np.empty_like(tangent)
    seed = np.eye(3)[np.argmin(np.abs(tangent[0]))] if initial_u is None else np.asarray(initial_u, dtype=float)
    if seed.shape != (3,) or not np.isfinite(seed).all():
        raise ValueError('initial_u must be a finite xyz vector')
    u[0] = seed - np.dot(seed, tangent[0]) * tangent[0]
    if np.linalg.norm(u[0]) < 1e-8:
        raise ValueError('initial_u is parallel to tangent')
    u[0] /= np.linalg.norm(u[0])
    for i in range(1, len(p)):
        a, b = tangent[i - 1], tangent[i]
        axis = np.cross(a, b); cosine = float(np.clip(np.dot(a, b), -1, 1))
        if cosine < -1 + 1e-8:
            raise ValueError('180-degree tangent reversal requires review')
        # Rodrigues rotation with non-unit cross product, stable near zero angle.
        rotated = u[i - 1] + np.cross(axis, u[i - 1]) + np.cross(axis, np.cross(axis, u[i - 1])) / (1 + cosine)
        u[i] = rotated - np.dot(rotated, b) * b
        u[i] /= np.linalg.norm(u[i])
    v[:] = np.cross(tangent, u)
    return tangent, u, v
