from typing import Tuple, List, Dict

from math import floor

import os
import torchvision
from torchvision.datasets.folder import default_loader


class FundusDataset(torchvision.datasets.ImageFolder):
    """
    """
    def __init__(self,
                 root_dir: str,
                 fn_set: set,
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 is_valid_file=None,
                 class_map=None):
        self.class_map = class_map
        if self.class_map is None:
            self.class_map = {i:i for i in range(11)}
        super(FundusDataset, self).__init__(root_dir,
                                            transform=transform,
                                            target_transform=target_transform,
                                            loader=loader,
                                            is_valid_file=is_valid_file)
        self.root_dir = root_dir
        self.fn_set = fn_set

        self._convert_sample()
    
    def _get_valid_classes(self, class_map):
        """return only (old_class, new_class) pairs where new_class >= 0. new_class=-1 means that we will not use this class"""
        valid_classes = {key: val for key, val in self.class_map.items() if val >= 0}
        return valid_classes

    def _convert_sample(self):
        print(len(self.samples))
        self.samples = [(x, y) for x, y in self.samples if (x.replace(self.root_dir, '') in self.fn_set or x.replace(self.root_dir + "/", "") in self.fn_set)]
        self.targets = [s[1] for s in self.samples]
        print("Final Data", len(self.targets))


    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        valid_class_map = self._get_valid_classes(self.class_map)
        classes = list(set(valid_class_map.values()))
        classes_to_idx = {str(i): valid_class_map[i] for i in sorted(valid_class_map.keys())}
        print(classes, classes_to_idx)
        return classes, classes_to_idx

