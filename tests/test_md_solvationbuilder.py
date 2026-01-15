import pytest
import sys
import numpy as np
from mofbuilder.md.solvationbuilder import SolvationBuilder

if __name__ == "__main__":
    import time
    packer = SolvationBuilder()
    packer.box_size = np.array([100, 100, 100])  # Å
    packer.preferred_region_box = np.array([[30, 70], [30, 70], [30, 70]])  # Å
    packer.buffer = 1.8  # Å
    packer.max_fill_rounds = 20
    start_time = time.time()
    packer.solute_file = "water.xyz"
    packer.solvents_files = ["water.xyz", "dmso.xyz"]
    packer.solvents_quantities = [330, 10]
    best_solvents_dict, best_keep_masks = packer.solvate()
    packer._update_datalines()
    packer.write_output(output_file="solvated_system",format=['xyz','gro'])
    print("Total time (s):", time.time() - start_time)


