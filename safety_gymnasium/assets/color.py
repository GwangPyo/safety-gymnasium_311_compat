# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Color."""
import numpy as np


class RGBAHashableArray(np.ndarray):
    def __new__(cls, input_array, dtype=np.float32):
        obj = np.asarray(input_array, dtype=dtype).view(cls)
        return obj

    def is_integer(self):
        return np.issubdtype(self.dtype, np.integer)

    def __str__(self):
        if self.is_integer():
            ar = self.astype(np.uint8)
        else:
            ar = (255 * self).astype(np.uint8)
        return ' '.join([str(x) for x in ar])

    def __hash__(self):
        return str(self)

    def __eq__(self, other):
        return str(other) == str(self)


COLOR = {
    # Distinct colors for different types of objects.
    # For now this is mostly used for visualization.
    # This also affects the vision observation, so if training from pixels.
    'push_box': RGBAHashableArray([1, 1, 0, 1], dtype=np.float32),
    'button': RGBAHashableArray([1, 0.5, 0, 1], dtype=np.float32),
    'goal': RGBAHashableArray([0, 1, 0, 1], dtype=np.float32),
    'vase': RGBAHashableArray([0, 1, 1, 1], dtype=np.float32),
    'hazard': RGBAHashableArray([0, 0, 1, 1], dtype=np.float32),
    'pillar': RGBAHashableArray([0.5, 0.5, 1, 1], dtype=np.float32),
    'wall': RGBAHashableArray([0.5, 0.5, 0.5, 1], dtype=np.float32),
    'gremlin': RGBAHashableArray([0.5, 0, 1, 1], dtype=np.float32),
    'circle': RGBAHashableArray([0, 1, 0, 1], dtype=np.float32),
    'red': RGBAHashableArray([1, 0, 0, 1], dtype=np.float32),
    'apple': RGBAHashableArray([0.835, 0.169, 0.169, 1], dtype=np.float32),
    'orange': RGBAHashableArray([1, 0.6, 0, 1], dtype=np.float32),
    'sigwall': RGBAHashableArray([1, 1, 0, 1], dtype=np.float32),
}
