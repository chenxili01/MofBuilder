import numpy as np
from pathlib import Path
from typing import Optional, Any, Union
from .basic import nn
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
import mpi4py.MPI as MPI
import sys

"""
GRO file reader for Gromacs coordinate files.

Expected atom line format:
atom_type, atom_label, atom_number, residue_name, residue_number, x, y, z, spin, charge, note
"""


class GroReader:
    """Reader for GROMACS .gro coordinate files.

    Handles loading of GRO files with optional MPI-aware parallel output support
    and recentering of structure to center of mass.

    Attributes:
        comm (Any): MPI communicator.
        rank (int): MPI process rank.
        nodes (int): Number of MPI processes.
        ostream (OutputStream): Output stream for printing/logging.
        filepath (Optional[str]): Path to the .gro coordinate file.
        data (Optional[np.ndarray]): Loaded atom data after parsing.
        _debug (bool): Enables debug output if True.

    Methods:
        read_gro(filepath, recenter, com_type): Read, parse, (optionally recenter) GRO file.
    """

    def __init__(
        self,
        comm: Optional[Any] = None,
        ostream: Optional[OutputStream] = None,
        filepath: Optional[Union[str, Path]] = None
    ):
        """Initializes the GroReader.

        Args:
            comm (Optional[Any]): MPI communicator. Defaults to MPI.COMM_WORLD.
            ostream (Optional[OutputStream]): Output stream for info/warning. If not provided, set based on MPI master rank.
            filepath (Optional[str or Path]): Path to the .gro file to load.

        """
        if comm is None:
            comm = MPI.COMM_WORLD

        if ostream is None:
            if comm.Get_rank() == mpi_master():
                ostream = OutputStream(sys.stdout)
            else:
                ostream = OutputStream(None)

        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream
        self.filepath: Optional[Union[str, Path]] = filepath
        self.data: Optional[np.ndarray] = None
        self._debug = False

    def read_gro(
        self,
        filepath: Optional[Union[str, Path]] = None,
        recenter: bool = False,
        com_type: Optional[str] = None
    ) -> None:
        """Reads a GRO file and parses atomic coordinates.

        Optionally recenters the structure so the center of mass (or
        specified atom type) is at the origin.

        Args:
            filepath (Optional[str or Path]): Path to the .gro file. If provided, overrides the current filepath.
            recenter (bool): Whether to recenter coordinates at the center of mass (default False).
            com_type (Optional[str]): If recentering, type of atom to use for COM; if None, use all atoms.

        Raises:
            AssertionError: If the file does not exist.

        """
        if filepath is not None:
            self.filepath = filepath
        assert_msg_critical(
            Path(self.filepath).exists(),
            f"gro file {self.filepath} not found"
        )
        if self._debug:
            self.ostream.print_info(f"Reading gro file {self.filepath}")

        inputfile = str(self.filepath)
        with open(inputfile, "r") as fp:
            lines = fp.readlines()

        data = []
        count = 1

        for line in lines:
            line = line.strip()
            if len(line.strip()) < 4:
                continue
            # GRO format: See GROMACS gro documentation for line slices.
            residue_number = int(line[0:5].strip())
            residue_name = line[5:10].strip()
            atom_label = line[10:15].strip()
            atom_number = count
            x = float(line[20:28]) * 10  # nm to Å
            y = float(line[28:36]) * 10
            z = float(line[36:44]) * 10
            atom_type = line[44:46].strip()
            charge = 0.0  # Default charge
            spin = 1.00   # Default spin
            note = nn(atom_type)  # Note field; e.g. element symbol
            count += 1
            data.append([
                atom_type, atom_label, atom_number, residue_name,
                residue_number, x, y, z, spin, charge, note
            ])

        self.data = np.vstack(data)

        def type_data(arr: np.ndarray) -> np.ndarray:
            """Helper to enforce correct data types per column."""
            arr[:, 2] = arr[:, 2].astype(int)
            arr[:, 4] = arr[:, 4].astype(int)
            arr[:, 5:8] = arr[:, 5:8].astype(float)
            arr[:, 8] = arr[:, 8].astype(float)
            arr[:, 9] = arr[:, 9].astype(float)
            return arr

        self.data = type_data(self.data)

        if recenter:
            # If no com_type, use all atoms for center-of-mass.
            # Else, use only specified type.
            if com_type is None:
                com_type_coords = self.data[:, 5:8].astype(float)
            else:
                if com_type not in self.data[:, -1]:
                    self.ostream.print_warning(
                        f"com_type {com_type} not in the pdb file, use all atoms to calculate com"
                    )
                    com_type_coords = self.data[:, 5:8].astype(float)
                else:
                    com_type_coords = self.data[
                        self.data[:, -1] == com_type][:, 5:8].astype(float)
            com = np.mean(com_type_coords, axis=0)
            if self._debug:
                self.ostream.print_info(
                    f"Center of mass type {com_type} at {com}"
                )
            self.data[:, 5:8] = self.data[:, 5:8].astype(float) - com
