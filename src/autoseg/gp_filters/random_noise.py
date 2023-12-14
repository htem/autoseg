import numpy as np
import gunpowder as gp
import random
from skimage.util import random_noise

class RandomNoiseAugment(gp.BatchFilter):
    """
    Random Noise Augmentation for Gunpowder.

    This class applies random noise augmentation to the specified array in a Gunpowder batch.

    Args:
        array (str): 
            The name of the array in the batch to which noise should be applied.
        seed (int, optional): 
            Seed for the random number generator. Default is None.
        clip (bool, optional): 
            Whether to clip the values after applying noise. Default is True.
        **kwargs: 
            Additional keyword arguments to be passed to the `random_noise` function.

    Attributes:
        array (str): 
            The name of the array in the batch to which noise is applied.
        seed (int): 
            Seed for the random number generator.
        clip (bool): 
            Whether to clip the values after applying noise.
        kwargs (dict): 
            Additional keyword arguments passed to the `random_noise` function.
    """

    def __init__(self, array, seed=None, clip=True, **kwargs):
        self.array = array
        self.seed = seed
        self.clip = clip
        self.kwargs = kwargs

    def setup(self):
        """
        Set up the filter by enabling autoskip and defining array updates.
        """
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        """
        Prepare the dependencies for processing based on the requested batch.

        Args:
            request (BatchRequest): 
                The requested batch.

        Returns:
            BatchRequest: 
                The dependencies for processing.
        """
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):
        """
        Apply random noise augmentation to the specified array in the batch.

        Args:
            batch (Batch): 
                The input batch.
            request (BatchRequest): 
                The requested batch.
        """
        raw = batch.arrays[self.array]

        mode = random.choice(["gaussian", "poisson", "none", "none"])

        if mode != "none":
            assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, (
                "Noise augmentation requires float types for the raw array (not "
                + str(raw.data.dtype)
                + "). Consider using Normalize before."
            )
            if self.clip:
                assert (
                    raw.data.min() >= -1 and raw.data.max() <= 1
                ), "Noise augmentation expects raw values in [-1,1] or [0,1]. Consider using Normalize before."
            raw.data = random_noise(
                raw.data, mode=mode, seed=self.seed, clip=self.clip, **self.kwargs
            ).astype(raw.data.dtype)
