import unittest
import numpy as np
import gunpowder as gp
from skimage import img_as_float
from autoseg.gp_filters import RandomNoiseAugment


class TestRandomNoiseAugment(unittest.TestCase):

    def setUp(self):
        # Set up any required data or configuration for your tests
        self.array_key = gp.ArrayKey("array")
        self.spec = gp.ArraySpec(roi=gp.Roi((0, 0, 0), (64, 64, 64)), voxel_size=(1, 1, 1))
        self.batch = gp.Batch()
        self.batch[self.array_key] = gp.Array(data=np.zeros((64, 64, 64), dtype=np.float32), spec=self.spec)

    def test_process_no_noise(self):
        # Test that the process method does not modify the array when mode is "none"
        random_noise_augment = RandomNoiseAugment(self.array_key, seed=42)
        request = gp.BatchRequest()
        request[self.array_key] = self.spec

        batch_out = gp.Batch()
        batch_out[self.array_key] = gp.Array(data=np.zeros((64, 64, 64), dtype=np.float32), spec=self.spec)

        random_noise_augment.prepare(request)
        random_noise_augment.process(batch_out, request)

        np.testing.assert_array_equal(batch_out[self.array_key].data, self.batch[self.array_key].data)

    def test_process_with_noise(self):
        # Test that the process method adds noise to the array when mode is not "none"
        random_noise_augment = RandomNoiseAugment(self.array_key, seed=42)
        request = gp.BatchRequest()
        request[self.array_key] = self.spec

        batch_out = gp.Batch()
        batch_out[self.array_key] = gp.Array(data=np.zeros((64, 64, 64), dtype=np.float32), spec=self.spec)

        random_noise_augment.prepare(request)
        random_noise_augment.process(batch_out, request)

        self.assertFalse(np.array_equal(batch_out[self.array_key].data, self.batch[self.array_key].data))

    def test_process_clip_values(self):
        # Test that the process method clips values when clip is True
        random_noise_augment = RandomNoiseAugment(self.array_key, seed=42, clip=True)
        request = gp.BatchRequest()
        request[self.array_key] = self.spec

        self.batch[self.array_key].data = img_as_float(np.random.rand(64, 64, 64))

        batch_out = gp.Batch()
        batch_out[self.array_key] = gp.Array(data=np.zeros((64, 64, 64), dtype=np.float32), spec=self.spec)

        random_noise_augment.prepare(request)
        random_noise_augment.process(batch_out, request)

        self.assertTrue(np.all(np.logical_and(batch_out[self.array_key].data >= -1, batch_out[self.array_key].data <= 1)))
