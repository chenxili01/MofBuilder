import pytest
import sys
import numpy as np
from mofbuilder.md.solvationbuilder import SolvationBuilder

if __name__ == "__main__":
    import time
    packer = SolvationBuilder()
    packer.box_size = np.array([100, 100, 100])  # Å
    packer.buffer = 1.8  # Å
    packer.max_fill_rounds = 400
    start_time = time.time()
    best_solvents_dict, best_keep_masks = packer.solvate(
        #solute_file="output/UiO-66_mofbuilder_output.xyz",
        solute_file="water.xyz",
        solvents_files=["water.xyz", "dmso.xyz"],
        target_solvents_numbers=[33000, 0000],
        box_buffer=2)
    print("Total time (s):", time.time() - start_time)


