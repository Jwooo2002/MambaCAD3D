"""Synthetic invariants and optional read-only real-source integration checks."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from centerline_paths import (asoca_lps_affine, extract_paths, graph_from_polylines,
                             graph_paths, parallel_transport_frames, read_asoca_graph,
                             read_imagecas_graph, read_vtk_graph, resample_path)


class TopologyTests(unittest.TestCase):
    def test_scrambled_ids_and_branch_edges(self):
        p = np.array([[2, 1, 0], [0, 0, 0], [2, -1, 0], [1, 0, 0.]])
        e = np.array([[3, 0], [1, 3], [3, 2], [3, 1], [1, 1]])
        audit = {}
        paths = graph_paths(p, e, np.array([1]), audit=audit)
        observed = [tuple(edge) for path in paths for edge in zip(path[:-1], path[1:])]
        self.assertCountEqual(observed, [(1, 3), (3, 0), (3, 2)])
        full = graph_paths(p, e, np.array([1]), mode='root_to_leaf')
        self.assertEqual([q.tolist() for q in full], [[1, 3, 0], [1, 3, 2]])
        self.assertEqual(audit['duplicate_edges_removed'], 1)
        self.assertEqual(audit['self_edges_removed'], 1)

    def test_shared_prefix_without_false_crossing_join(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 0, 0],
                      [0, 0, 0], [1, 0, 0], [2, -1, 0], [3, 0, 0.]])
        g = graph_from_polylines(p, [np.arange(4), np.arange(4, 8)])
        self.assertEqual(len(g.points), 6)
        self.assertEqual(len(g.roots), 1)
        self.assertEqual(sum(np.all(g.points == [3, 0, 0], axis=1)), 2)
        paths = graph_paths(g.points, g.edges, g.roots)
        self.assertEqual(sum(len(q) - 1 for q in paths), len(g.edges))
        self.assertEqual(len(extract_paths(g, mode='root_to_leaf')), 2)

    def test_consecutive_repeat_and_duplicate_whole_polyline(self):
        p = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0.]])
        g = graph_from_polylines(p, [np.arange(3), np.arange(3)])
        self.assertEqual(g.edges.tolist(), [[0, 1]])
        self.assertEqual(g.metadata['consecutive_duplicate_samples_removed'], 2)

    def test_loop_requires_explicit_policy_and_reports_removed_edge(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.]])
        e = np.array([[0, 1], [1, 2], [2, 0]])
        with self.assertRaisesRegex(ValueError, 'cycles'):
            graph_paths(p, e, np.array([0]))
        audit = {}
        paths = graph_paths(p, e, np.array([0]), cycle_policy='shortest_path_tree', audit=audit)
        self.assertEqual(sum(len(q) - 1 for q in paths), 2)
        self.assertEqual(len(audit['cycle_edges_removed']), 1)
        with self.assertRaisesRegex(ValueError, 'revisits'):
            graph_from_polylines(p, [np.array([0, 1, 2, 0])])

    def test_root_required_per_component(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.]])
        e = np.array([[0, 1], [2, 3]])
        for roots in (np.array([0]), np.array([0, 1, 2])):
            with self.assertRaisesRegex(ValueError, 'annotated roots'):
                graph_paths(p, e, roots)
        self.assertEqual(len(graph_paths(p, e, np.array([0, 2]))), 2)

    def test_nonfinite_and_invalid_ids_rejected(self):
        with self.assertRaises(ValueError):
            graph_paths([[0, 0, 0], [np.nan, 1, 2]], np.array([[0, 1]]), np.array([0]))
        with self.assertRaises(ValueError):
            graph_paths([[0, 0, 0]], np.array([[0, 2]]), np.array([0]))


class GeometryTests(unittest.TestCase):
    def test_uniform_source_arclength_not_chord(self):
        p = [[0, 0, 0], [0, 0, 0], [1.2, 0, 0], [1.2, 1.1, 0]]
        q, arc = resample_path(p, .5, return_distances=True)
        np.testing.assert_allclose(arc, [0, .5, 1, 1.5, 2])
        np.testing.assert_allclose(q[3], [1.2, .3, 0])
        self.assertLess(np.linalg.norm(q[3] - q[2]), .5)
        _, endpoint_arc = resample_path(p, .5, include_endpoint=True, return_distances=True)
        self.assertAlmostEqual(endpoint_arc[-1], 2.3)

    def test_zero_length_and_bad_spacing_fail(self):
        for points, spacing in (([[1, 1, 1], [1, 1, 1]], .5), ([[0, 0, 0], [1, 1, 1]], 0)):
            with self.assertRaises(ValueError):
                resample_path(points, spacing)

    def test_frames_are_orthonormal_continuous_on_helix(self):
        angle = np.linspace(0, 4 * np.pi, 500)
        p = np.column_stack((np.cos(angle), np.sin(angle), .2 * angle))
        t, u, v = parallel_transport_frames(p)
        frame = np.stack((t, u, v), axis=2)
        np.testing.assert_allclose(np.einsum('nji,njk->nik', frame, frame), np.broadcast_to(np.eye(3), (len(p), 3, 3)), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(frame), 1., atol=1e-12)
        self.assertGreater(np.min(np.sum(u[:-1] * u[1:], axis=1)), .99)

    def test_frame_reversal_is_not_silently_clamped(self):
        with self.assertRaisesRegex(ValueError, 'reversal'):
            parallel_transport_frames([[0, 0, 0], [1, 0, 0], [0, 0, 0]])

    def test_asoca_transform_not_ras_flip(self):
        affine = asoca_lps_affine([-90.5, -130.10000610351565, -260])
        np.testing.assert_allclose(affine[:3, 3], [-181, -260.2000122070313, 0])
        np.testing.assert_allclose(affine[:3, :3], np.eye(3))
        with self.assertRaises(ValueError):
            asoca_lps_affine([0, 0, 0], np.diag([-1, -1, 1]))


class RealSourceTests(unittest.TestCase):
    def test_imagecas_455_cycle_requires_recorded_repair(self):
        path = ROOT / 'external_data/imagecas_x/prepared/graphs/455.npz'
        if not path.exists():
            self.skipTest('ImageCAS-X graph not mounted')
        g = read_imagecas_graph(path, 'right')
        with self.assertRaisesRegex(ValueError, 'cycles'):
            graph_paths(g.points, g.edges, g.roots)
        audit = {}
        paths = graph_paths(g.points, g.edges, g.roots,
                            cycle_policy='shortest_path_tree', audit=audit)
        self.assertEqual(audit['cycle_edges_removed'], [[362, 411]])
        self.assertEqual(len(set(np.concatenate(paths).tolist())), len(g.points))
        self.assertEqual(sum(len(q) - 1 for q in paths), len(g.edges) - 1)
        for p in paths:
            sampled = resample_path(g.points[p])
            if len(sampled) >= 2:
                parallel_transport_frames(sampled)

    @unittest.skipUnless(importlib.util.find_spec('vtk') and importlib.util.find_spec('SimpleITK'), 'VTK and SimpleITK not installed')
    def test_asoca_normal10_connectivity_and_lumen(self):
        import SimpleITK as sitk
        base = Path('/home/jiwoong/PycharmProjects/journal_preprocess/ASOCADataAccess/Normal')
        if not (base / 'Centerlines/Normal_10.vtp').exists():
            self.skipTest('ASOCA source not mounted')
        mask = sitk.ReadImage(str(base / 'Annotations/Normal_10.nrrd'))
        source = read_vtk_graph(base / 'Centerlines/Normal_10.vtp')
        self.assertEqual(source.polylines[0][:5].tolist(), [14818, 986, 985, 984, 983])
        g = read_asoca_graph(base / 'Centerlines/Normal_10.vtp', mask.GetOrigin(), mask.GetDirection())
        self.assertEqual(len(g.roots), 2)
        q = np.rint((g.points - mask.GetOrigin()) / mask.GetSpacing()).astype(int)
        a = sitk.GetArrayFromImage(mask)
        self.assertGreater(np.mean(a[q[:, 2], q[:, 1], q[:, 0]] > 0), .99)
        paths = graph_paths(g.points, g.edges, g.roots)
        self.assertEqual(sum(len(q) - 1 for q in paths), len(g.edges))
        for p in extract_paths(g):
            sampled, arc = resample_path(p, return_distances=True)
            np.testing.assert_allclose(np.diff(arc), .5)
            if len(sampled) >= 2:
                parallel_transport_frames(sampled)

    def test_imagecas_real_graph_roots_and_edge_coverage(self):
        path = ROOT / 'external_data/imagecas_x/prepared/graphs/68.npz'
        if not path.exists():
            self.skipTest('ImageCAS-X graph not mounted')
        for side in ('left', 'right'):
            g = read_imagecas_graph(path, side)
            self.assertEqual(len(g.roots), 1)
            paths = graph_paths(g.points, g.edges, g.roots)
            self.assertEqual(sum(len(q) - 1 for q in paths), len(g.edges))
            for p in extract_paths(g):
                sampled = resample_path(p)
                if len(sampled) >= 2:
                    parallel_transport_frames(sampled)


if __name__ == '__main__':
    unittest.main()
