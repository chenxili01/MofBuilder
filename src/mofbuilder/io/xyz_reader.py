import numpy as np
from pathlib import Path
from typing import Optional, Any, Union
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
import mpi4py.MPI as MPI
import sys
import re

"""
XYZ file reader for simple ASCII molecular coordinate files.

Expected atom line format in array:
atom_type, atom_label, atom_number, residue_name, residue_number, x, y, z, spin, charge, note
"""


class XyzReader:
    """Reader for XYZ molecular coordinate files.

    Handles loading of XYZ files with MPI-aware output and optional structure recentering.

    Attributes:
        comm (Any): MPI communicator.
        rank (int): MPI process rank.
        nodes (int): Number of MPI processes.
        ostream (OutputStream): Output stream for info/logging.
        filepath (Optional[str]): Path to the input .xyz file.
        data (Optional[np.ndarray]): Atom information as parsed and processed.
        _debug (bool): Enables debug output if True.

    Methods:
        read_xyz: Read, parse, and (optionally recenter) XYZ file into array format.
    """

    def __init__(
        self,
        comm: Optional[Any] = None,
        ostream: Optional[OutputStream] = None,
        filepath: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initializes the XyzReader.

        Args:
            comm (Any, optional): MPI communicator. Defaults to MPI.COMM_WORLD.
            ostream (Optional[OutputStream]): Output stream for info/debug. Defaults to output on master rank.
            filepath (Optional[str or Path]): Path to the .xyz file to load.
        """
        if comm is None:
            comm = MPI.COMM_WORLD

        if ostream is None:
            if comm.Get_rank() == mpi_master():
                ostream = OutputStream(sys.stdout)
            else:
                ostream = OutputStream(None)

        # mpi information
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()

        # output stream
        self.ostream = ostream

        self.filepath = filepath
        self.data: Optional[np.ndarray] = None

        # debug
        self._debug: bool = False

    def read_xyz(
        self,
        filepath: Optional[Union[str, Path]] = None,
        recenter: bool = False,
        com_type: Optional[str] = None,
        residue_name: str = 'MOL',
        residue_number: int = 1,
    ) -> None:
        """Reads atom and coordinate information from an XYZ file, storing structure in a NumPy array.

        Optionally recenters structure coordinates to a specified atom type or whole molecule center of mass.

        Args:
            filepath (Optional[str or Path]): Path to the .xyz file. If not supplied, uses instance filepath.
            recenter (bool): If True, recenter coordinates to center-of-mass. Defaults to False.
            com_type (Optional[str]): If specified, use only atoms of this type for COM calculation.
            residue_name (str): Residue name assignment for all atoms. Defaults to 'MOL'.
            residue_number (int): Residue number for all atoms. Defaults to 1.

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified file path does not exist.

        Note:
            Assigns atom_type, atom_label (e.g., "C1", "O2"), atom_number (1-based),
            residue_name, residue_number, x, y, z, spin (1.0), charge (from file or 0.0),
            and note (from file or empty) for each atom.

            Array output shape is (num_atoms, 11).
        """
        if filepath is not None:
            self.filepath = filepath

        if self.filepath is None or not Path(self.filepath).exists():
            raise FileNotFoundError(f"XYZ file {self.filepath} not found")

        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            raise ValueError(f"XYZ file {self.filepath} has insufficient lines")

        comment = lines[1].strip()
        # Extract the atom coordinates from the subsequent lines
        data = []
        for line in lines[2:]:
            if len(line.strip().split()) < 4:
                continue
            parts = line.split()
            atom_type = parts[0]
            atom_type = re.sub(r'\d', '', atom_type)  # remove digits
            atom_number = len(data) + 1
            atom_label = atom_type + str(atom_number)
            res_name = residue_name
            res_number = residue_number
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            if len(parts) > 4:
                charge = float(parts[4])
            else:
                charge = 0.0
            if len(parts) > 5:
                note = parts[5]
            else:
                note = ''
            spin = 1.00
            data.append((
                atom_type, atom_label, atom_number, res_name,
                res_number, x, y, z, spin, charge, note
            ))

        def type_data(arr: np.ndarray) -> np.ndarray:
            """Ensures appropriate dtype conversion for numeric columns.

            Args:
                arr (np.ndarray): Array of atom records (shape: N, 11).

            Returns:
                np.ndarray: Array with selected columns cast to appropriate types.
            """
            arr[:, 2] = arr[:, 2].astype(int)     # atom_number
            arr[:, 4] = arr[:, 4].astype(int)     # residue_number
            arr[:, 5:8] = arr[:, 5:8].astype(float)  # x, y, z
            arr[:, 8] = arr[:, 8].astype(float)   # spin
            arr[:, 9] = arr[:, 9].astype(float)   # charge
            return arr

        self.data = type_data(np.array(data, dtype=object))

        if recenter:
            # If not define com_type, use all atoms to calculate com
            if com_type is None:
                com_type_ccoords = self.data[:, 5:8].astype(float)
            else:
                # Try to get atom_type match for center of mass
                matched_types = self.data[:, 0] == com_type
                if not np.any(matched_types):
                    self.ostream.print_warning(
                        f"com_type {com_type} not in the xyz file, using all atoms to calculate com"
                    )
                    com_type_ccoords = self.data[:, 5:8].astype(float)
                else:
                    com_type_ccoords = self.data[matched_types][:, 5:8].astype(float)
            com = np.mean(com_type_ccoords, axis=0)
            if self._debug:
                self.ostream.print_info(
                    f"Center of mass type {com_type} at {com}"
                )
            self.data[:, 5:8] = self.data[:, 5:8].astype(float) - com
