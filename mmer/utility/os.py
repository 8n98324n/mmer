        
import numpy as np
import os
import pandas as pd

"""
This module contains classes and methods for transforming a BVP signal in a BPM signal.
"""

class Path:
    """
    Manage (multi-channel, row-wise) BVP signals, and transforms them in BPMs.
    """
    def __init__(self):
        pass

    def check_path_or_create(self, path):
        # Check if the path exists
        if not os.path.exists(path):
            # If it does not exist, create the directory
            os.makedirs(path)
        return path