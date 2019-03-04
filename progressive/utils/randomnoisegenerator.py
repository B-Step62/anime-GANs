# -*- coding: utf-8 -*-
import numpy as np 

class RandomNoiseGenerator():
    def __init__(self, size, noise_type='gaussian', clip=None):
        self.size = size
        self.noise_type = noise_type.lower()
        assert self.noise_type in ['gaussian', 'uniform']
        self.generator_map = {'gaussian': np.random.randn, 'uniform': np.random.uniform}
        if self.noise_type == 'gaussian':
            if clip is None:
                self.generator = lambda s: np.random.randn(*s)
            else:
                self.generator = lambda s: self.min_max_normalize(np.random.randn(*s), clip[0], clip[1])
        elif self.noise_type == 'uniform':
            if clip is None:
                self.generator = lambda s: np.random.uniform(-1, 1, size=s)
            else:
                self.generator = lambda s: np.random.uniform(clip[0], clip[1], size=s)

    def min_max_normalize(self, array, _min, _max):
        return ((array - np.min(array)) / (np.max(array) - np.min(array))) * (_max - _min) + _min

    def __call__(self, batch_size):
        return self.generator([batch_size, self.size]).astype(np.float32)
