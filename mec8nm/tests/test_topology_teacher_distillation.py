from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from dbmim.datasets import EMVolumeDataset
from train_finetune import topology_routed_teacher_loss, topology_simple_teacher_route


class TopologyStatsDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "volume.h5"
        shape = (6, 12, 12)
        raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        labels = np.arange(1, np.prod(shape) + 1, dtype=np.int64).reshape(shape)
        with h5py.File(self.path, "w") as handle:
            handle.create_dataset("raw", data=raw)
            handle.create_dataset("labels", data=labels)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_stats_are_measured_before_boundary_widening(self) -> None:
        crop_shape = (4, 8, 8)
        dataset = EMVolumeDataset(
            paths=[self.path],
            label_paths=[self.path],
            keys=["raw"],
            label_keys=["labels"],
            volume_size=crop_shape,
            augment=False,
            widen_border=True,
            widen_border_radius=1,
            return_topology_stats=True,
            seed=79,
        )
        item = dataset[0]
        self.assertEqual(float(item["topology_foreground_fraction"]), 1.0)
        self.assertEqual(float(item["topology_boundary_fraction"]), 1.0)
        self.assertEqual(int(item["topology_unique_nonzero"]), int(np.prod(crop_shape)))
        self.assertEqual(int(torch.count_nonzero(item["label"])), 0)


class TopologyTeacherDistillationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {
            "channels": 3,
            "foreground_threshold": 0.95,
            "foreground_width": 0.02,
            "boundary_threshold": 0.05,
            "boundary_width": 0.02,
            "unique_threshold": 5.5,
            "unique_width": 2.0,
            "confidence_floor": 0.0,
        }

    def test_route_prefers_simple_teacher_only_for_simple_topology(self) -> None:
        route = topology_simple_teacher_route(
            torch.tensor([0.99, 0.70]),
            torch.tensor([0.01, 0.30]),
            torch.tensor([2.0, 30.0]),
            self.cfg,
        )
        self.assertGreater(float(route[0]), 0.6)
        self.assertLess(float(route[1]), 1e-6)

    def test_route_rejects_nonpositive_width(self) -> None:
        with self.assertRaises(ValueError):
            topology_simple_teacher_route(
                torch.tensor([0.99]),
                torch.tensor([0.01]),
                torch.tensor([2.0]),
                {**self.cfg, "unique_width": 0.0},
            )

    def test_routes_select_the_expected_teacher(self) -> None:
        simple = torch.full((2, 3, 2, 4, 4), 1.25)
        complex_teacher = torch.full_like(simple, -0.75)
        student = torch.stack((simple[0], complex_teacher[1]))
        loss = topology_routed_teacher_loss(
            student,
            simple,
            complex_teacher,
            torch.tensor([1.0, 0.0]),
            self.cfg,
        )
        self.assertEqual(float(loss), 0.0)

    def test_soft_route_blends_teacher_probabilities(self) -> None:
        simple = torch.full((1, 3, 2, 4, 4), 2.0)
        complex_teacher = torch.full_like(simple, -1.0)
        route = torch.tensor([0.25])
        blended = route.item() * torch.sigmoid(simple) + (1.0 - route.item()) * torch.sigmoid(
            complex_teacher
        )
        student = torch.logit(blended)
        loss = topology_routed_teacher_loss(
            student,
            simple,
            complex_teacher,
            route,
            self.cfg,
        )
        self.assertLess(float(loss), 1e-12)

    def test_gradients_only_enter_student(self) -> None:
        student = torch.randn(2, 3, 2, 4, 4, requires_grad=True)
        simple = torch.randn_like(student, requires_grad=True)
        complex_teacher = torch.randn_like(student, requires_grad=True)
        route = torch.tensor([0.2, 0.8], requires_grad=True)
        loss = topology_routed_teacher_loss(
            student,
            simple,
            complex_teacher,
            route,
            self.cfg,
        )
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0.0)
        self.assertIsNone(simple.grad)
        self.assertIsNone(complex_teacher.grad)
        self.assertIsNone(route.grad)


if __name__ == "__main__":
    unittest.main()
