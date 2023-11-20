import numpy as np
import gunpowder as gp
import random
from scipy.ndimage import gaussian_filter


class SmoothArray(gp.BatchFilter):
    def __init__(self, array, blur_range):
        self.array = array
        self.range = blur_range

    def process(self, batch, request):

        array = batch[self.array].data

        assert len(array.shape) == 3

        # different numbers will simulate noisier or cleaner array
        sigma = random.uniform(self.range[0], self.range[1])

        for z in range(array.shape[0]):
            array_sec = array[z]

            array[z] = np.array(
                    gaussian_filter(array_sec, sigma=sigma)
            ).astype(array_sec.dtype)

        batch[self.array].data = array
