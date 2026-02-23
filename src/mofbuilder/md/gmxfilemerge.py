from operator import index
import re
import sys
from pathlib import Path
from typing import Optional, Any, List, Dict, Tuple, Sequence
from veloxchem.outputstream import OutputStream
from veloxchem.molecule import Molecule
from veloxchem.scfrestdriver import ScfRestrictedDriver
from veloxchem.molecularbasis import MolecularBasis
from veloxchem.optimizationdriver import OptimizationDriver
from veloxchem.mmforcefieldgenerator import MMForceFieldGenerator
from veloxchem.veloxchemlib import mpi_master, hartree_in_kcalpermol, hartree_in_kjpermol
from veloxchem.errorhandler import assert_msg_critical
import mpi4py.MPI as MPI


class GromacsForcefieldMerger:
    """
    Main class responsible for assembling and merging GROMACS forcefield files for MOF simulations.

    Attributes:
        comm (mpi4py.MPI.Comm): MPI communicator for parallel execution.
        rank (int): Rank of the current MPI process.
        nodes (int): Total number of MPI processes.
        ostream (OutputStream): Output stream for logging.
        database_dir (Optional[str]): Directory containing reference forcefield database.
        target_dir (Optional[str]): Output directory for all generated files.
        node_metal_type (Optional[str]): Type of metal node in the MOF.
        dummy_atom_node (bool): Whether to use dummy atoms for the node (default: False).
        termination_name (Optional[str]): Name of termination group for the structure.
        linker_itp_dir (str): Directory holding linker .itp files.
        linker_name (Optional[str]): Name of the linker molecule.
        residues_info (Optional[Dict[str, int]]): Dictionary mapping residue names to copy number.
        mof_name (Optional[str]): Name for the MOF/model (used for the top file).
        other_residues (List[str]): List of additional residues that can be included.
        solvents_name (Optional[Sequence[str]]): List or tuple of solvent names.
        solvents_dict (Optional[Dict[str, Dict[str, Any]]]): Maps solvent names to their info, including Molecule object.
        _debug (bool): If True, print extensive debug information.
        top_path (Optional[Path]): The path to the generated .top file (written upon generation).
    """

    def __init__(self, comm: Optional[Any] = None, ostream: Optional[OutputStream] = None) -> None:
        """
        Initializes a GromacsForcefieldMerger instance.

        Args:
            comm (Optional[Any]): Optional MPI communicator.
            ostream (Optional[OutputStream]): Optional custom output stream for logging.
        """
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank == mpi_master() else None)

        # Attributes related to input and output directories, model setup, and optional features.
        self.database_dir: Optional[str] = None
        self.target_dir: Optional[str] = None
        self.node_metal_type: Optional[str] = None
        self.dummy_atom_node: bool = False
        self.termination_name: Optional[str] = None
        self.linker_itp_dir: str = ''
        self.linker_name: Optional[str] = None
        self.residues_info: Optional[Dict[str, int]] = None
        self.mof_name: Optional[str] = None
        self.other_residues: List[str] = ['O', 'HO', 'HHO']

        # Solvent related attributes
        self.solvents_name: Optional[Sequence[str]] = None
        self.solvents_dict: Optional[Dict[str, Dict[str, Any]]] = None

        self._debug: bool = False
        self.top_path: Optional[Path] = None

    def _copy_file(self, old_path: str, new_path: str) -> None:
        """
        Copy a file from `old_path` to `new_path` if it doesn't already exist at `new_path`.

        Args:
            old_path (str): Source file path.
            new_path (str): Destination file path.
        """
        src = Path(old_path)
        dest = Path(new_path)
        if not dest.is_file():
            if not dest.parent.is_dir():
                dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(src.read_text())
        if self._debug:
            self.ostream.print_info(f"File copied from {old_path} to {new_path}")
            self.ostream.flush()

    def _backup_and_rename(self, target_path: str) -> None:
        """
        If the directory at target_path exists and is not empty, rename it by prefixing with '#i_' where i is the next available integer.

        Args:
            target_path (str): Folder to potentially back up/rename.
        """
        p = Path(target_path)
        if p.exists() and any(p.iterdir()):  # non-empty
            i = 1
            new_path = Path(p.parent, f"#{i}_{p.name}")
            while new_path.exists():
                i += 1
                new_path = Path(p.parent, f"#{i}_{p.name}")
            self.ostream.print_info(f"{p} existed and not empty, renaming {p} --> {new_path}")
            self.ostream.flush()
            p.rename(new_path)
            # Recreate the original folder
            Path(target_path).mkdir(parents=True, exist_ok=True)

    def _get_itps_from_database(self, data_path: Optional[str] = None) -> None:
        """
        Collect .itp files for nodes, linkers, terminations, and solvents/ions/gas. 
        If some are missing (such as new solvents), they will be generated and added to the database.

        Args:
            data_path (Optional[str]): Path to forcefield database. Defaults to self.database_dir.
        """
        if data_path is None:
            data_path = self.database_dir
        target_itp_path = Path(self.target_dir, 'MD_run/itps')
        self._backup_and_rename(str(target_itp_path))
        target_itp_path.mkdir(parents=True, exist_ok=True)

        # Copy node ITPs
        node_itp_name = f"{self.node_metal_type}_dummy" if self.dummy_atom_node else f"{self.node_metal_type}"
        if self._debug:
            self.ostream.print_info(f"looking for {node_itp_name}.itp for node")

        for i in Path(data_path, 'nodes_itps').rglob("*.itp"):
            if (i.stem == node_itp_name) or (i.stem in self.other_residues):
                dest_p = Path(target_itp_path, i.name)
                self._copy_file(str(i), str(dest_p))

        # Copy linker itp(s)
        if self.linker_itp_dir not in [None, '']:
            for j in Path(self.linker_itp_dir).rglob('*.itp'):
                itp_name = j.stem
                dest_p = Path(target_itp_path, j.name)
                if itp_name == Path(str(self.linker_name)).stem:
                    self._copy_file(str(j), str(dest_p))

        # Copy termination itp
        for k in Path(data_path, 'terminations_itps').rglob('*.itp'):
            dest_p = Path(target_itp_path, k.name)
            if k.stem == Path(str(self.termination_name)).stem:
                self._copy_file(str(k), str(dest_p))
                self.ostream.print_info(f"term.  {k} to {dest_p}")
                self.ostream.flush()

        # Copy/generate solvent itp files
        if self.solvents_name:
            for sol in self.solvents_name:
                src_p = Path(data_path, 'solvents_database', f'{sol}.itp')
                if not src_p.is_file():
                    # generate solvent itp if not found
                    self.ostream.print_info(
                        f"solvent itp file {src_p} not found in database... will generate {sol} forcefield and add it to database!"
                    )
                    self.ostream.flush()
                    sol_molecule = self.solvents_dict[sol]['molecule']
                    src_p = Path(
                        self._generate_solvent_itp(
                            sol, sol_molecule, str(Path(data_path, 'solvents_database')))
                    )
                dest_p = target_itp_path / f'{sol}.itp'
                self.ostream.print_info(f"copying solvent itp file {src_p} to {dest_p}")
                self.ostream.flush()
                self._copy_file(str(src_p), str(dest_p))

        # Print files in itp directory if debug
        final_itp_files = [
            str(i) for i in Path(target_itp_path).rglob("*.itp")
        ]
        str_itps = ",".join(final_itp_files)
        if self._debug:
            self.ostream.print_info(
                f"{str(target_itp_path)} directory have {len(final_itp_files)} files"
            )
            self.ostream.print_info(f"include {str_itps}")
            self.ostream.flush()

    def _generate_solvent_itp(self, solvent_name: str, molecule: Molecule, target_path: str) -> str:
        """
        Generates a GROMACS .itp file for a given solvent molecule. Also performs structure optimization.

        Args:
            solvent_name (str): Name of the solvent.
            molecule (Molecule): Molecule object for the solvent.
            target_path (str): Directory where the .itp and intermediate files are saved.

        Returns:
            str: Path to the generated .itp file.
        """
        mol_scf_drv = ScfRestrictedDriver()
        mol_basis = MolecularBasis.read(molecule, "def2-svp")
        mol_scf_drv.conv_thresh = 1e-3
        mol_scf_drv.file_name = f"{solvent_name}_opt_scf"
        mol_scf_drv.xcfun = "b3lyp"
        mol_scf_drv.ostream.mute()
        mol_scf_results = mol_scf_drv.compute(molecule, mol_basis)
        mol_opt_drv = OptimizationDriver(mol_scf_drv)
        mol_opt_drv.conv_energy = 1e-04
        mol_opt_drv.conv_drms = 1e-02
        mol_opt_drv.conv_dmax = 2e-02
        mol_opt_drv.conv_grms = 4e-03
        mol_opt_drv.conv_gmax = 8e-03
        # mol_opt_drv.constraints = ... # Constraints could be added here if needed.
        mol_opt_drv.tmax = 0.02
        mol_opt_drv.filename = mol_scf_drv.file_name
        mol_opt_drv.ostream.mute()
        opt_results = mol_opt_drv.compute(molecule, mol_basis, mol_scf_results)
        opt_mol = Molecule.read_xyz_string(opt_results["final_geometry"])
        ffgen = MMForceFieldGenerator()
        ffgen.create_topology(opt_mol)
        ff_name = str(Path(target_path, f"{solvent_name}"))
        ffgen.write_gromacs_files(filename=f"{ff_name}", mol_name=solvent_name)
        # Remove .gro and .top files generated
        gro_file = ff_name + ".gro"
        top_file = ff_name + ".top"
        Path(gro_file).unlink(missing_ok=True)
        Path(top_file).unlink(missing_ok=True)
        return ff_name + ".itp"

    def _itp_extract(self, itp_file: str) -> List[str]:
        """
        Extracts the atomtypes section (typically lines between [ atomtypes ] and [ moleculetype ]) from an .itp file.
        Also removes this section from the .itp file in place.

        Args:
            itp_file (str): Path to the .itp file.

        Returns:
            List[str]: The lines of the atomtypes section (with whitespace trimmed), or empty if not found.
        """
        with open(itp_file, "r") as f:
            lines = f.readlines()
        keyword1 = "atomtypes"
        keyword2 = "moleculetype"
        start = None
        end = None
        for eachline in lines:  # search for keywords and get linenumber
            if re.search(keyword1, eachline):
                start = lines.index(eachline) + 2
            elif re.search(keyword2, eachline):
                end = lines.index(eachline) - 1
        if start is None or end is None:
            return []
        target_lines = [line for line in lines[start:end] if line.strip()]
        newstart = end + 1
        with open(itp_file, "w") as fp:
            fp.writelines(lines[newstart:])
        return target_lines

    def _extract_atomstypes(self, itp_path: str) -> List[str]:
        """
        Walks through all .itp files in a given directory and collects all atomtype lines,
        except those from "posre.itp".

        Args:
            itp_path (str): Folder to search for .itp files.

        Returns:
            List[str]: A list of atomtype lines collected from all .itp files.
        """
        all_secs: List[str] = []
        for f in Path(itp_path).rglob("*itp"):
            if str(Path(f).parent) not in ["posre.itp"]:
                if self._debug:
                    self.ostream.print_info(f"found file: {f}")
                    self.ostream.flush()
                sec_atomtypes = self._itp_extract(str(f))
                all_secs += sec_atomtypes
        return all_secs

    def _get_unique_atomtypes(self, all_secs: List[str]) -> List[str]:
        """
        Remove duplicated atomtype lines, keeping only the first occurrence for each atomtype.

        Args:
            all_secs (List[str]): List of all atomtype lines.

        Returns:
            List[str]: List of unique atomtype lines.
        """
        types = [str(line.split()[0]) for line in all_secs]
        overlap_lines = []
        for ty in set(types):
            search = [ind for ind, value in enumerate(types) if value == ty]
            if len(search) > 1:
                overlap_lines += search[1:]
        unique_atomtypes = [
            all_secs[i] for i in range(len(all_secs)) if i not in overlap_lines
        ]
        return unique_atomtypes

    def _parsetop(self, inputfile: str) -> Tuple[List[List[str]], List[str]]:
        """
        Parses a GROMACS .top template file by splitting it into sections by the occurrence of ']'.

        Args:
            inputfile (str): Path to the template .top file.

        Returns:
            Tuple[List[List[str]], List[str]]: 
                - middlelines: a list where each element is a section (list of lines) from the .top file
                - sectorname: list of the name/header line of each section
        """
        with open(inputfile, "r") as fp:
            original_lines = fp.readlines()
        lines = [line for line in original_lines if line.strip()]
        number = []
        lineNumber = 1
        keyword1 = "]"
        for eachline in lines:  # search for keywords and get linenumber
            m = re.search(keyword1, eachline)
            if m is not None:
                number.append(lineNumber - 1)  # split by linenumber
            lineNumber += 1
        number.append(len(lines))
        number = list(set(number))
        number.sort()
        size = int(len(number))
        middlelines = []
        sectorname = []
        for i in range(size - 1):
            start = number[i]
            end = number[i + 1]
            middlelines.append(lines[start:end])
            sectorname.append(lines[start])
        return middlelines, sectorname

    def _generate_top_file(
        self,
        itp_path: str,
        data_path: Optional[str] = None,
        res_info: Optional[Dict[str, int]] = None,
        model_name: Optional[str] = None,
    ) -> Path:
        """
        Assembles the final GROMACS .top file for the system from supplied .itp files and molecule counts.

        Args:
            itp_path (str): Directory of .itp files to include.
            data_path (Optional[str]): Path to forcefield database. Defaults to self.database_dir if None.
            res_info (Optional[Dict[str, int]]): Dictionary with residue names and counts.
            model_name (Optional[str]): Name for the system/model.

        Returns:
            Path: The path to the written .top file.
        """
        all_secs = self._extract_atomstypes(itp_path)
        unique_atomtypes = self._get_unique_atomtypes(all_secs)
        middlelines, sectorname = self._parsetop(
            str(Path(data_path or self.database_dir, "nodes_itps/template.top"))
        )
        top_res_lines = []
        res_info = res_info or {}
        for resname in list(res_info):
            if resname[0] == ';':
                continue
            if res_info[resname] <= 0:
                continue
            line = "%-5s%16d" % (resname[:3], res_info[resname])
            top_res_lines.append(line)
            top_res_lines.append("\n")

        top_itp_lines = []
        for i in Path(itp_path).rglob("*itp"):
            if str(Path(i).name) not in ["posre.itp"]:
                if self._debug:
                    self.ostream.print_info(f"found file: {i} in path {itp_path}")
                    self.ostream.flush()
                line = '#include "itps/' + i.name + '"\n'
                top_itp_lines.append(line)
                if self._debug:
                    self.ostream.print_info(f"line{line}")
                    self.ostream.flush()

        if model_name is None:
            model_name = "MOF"
        newtop = (
            middlelines[0] + ["\n", "\n"] +
            middlelines[1] +
            unique_atomtypes + ["\n", "\n"] +
            top_itp_lines + ["\n", "\n"] +
            middlelines[2] +
            ["MOF", "\n", "\n"] +
            middlelines[3] +
            ["\n"] + top_res_lines
        )
        topname = model_name + ".top"
        top_path = Path(self.target_dir, "MD_run", topname)
        top_path.parent.mkdir(parents=True, exist_ok=True)

        with open(top_path, "w") as f:
            f.writelines(newtop)
        self.ostream.print_info(f" {topname} is generated")
        self.ostream.flush()
        return top_path

    def _copy_mdps(self, data_path: Optional[str] = None) -> Path:
        """
        Copies all .mdp files (GROMACS MD parameter files) from the database to the target run directory.

        Args:
            data_path (Optional[str]): Path to forcefield database. Uses self.database_dir if not provided.

        Returns:
            Path: Path to the destination mdps directory.
        """
        if data_path is None:
            data_path = self.database_dir
        dest_mdp_path = Path(self.target_dir, "MD_run", "mdps")
        dest_mdp_path.mkdir(parents=True, exist_ok=True)
        src_mdp_path = Path(data_path, "mdps")
        for i in src_mdp_path.rglob("*.mdp"):
            self._copy_file(str(i), str(Path(dest_mdp_path, Path(i).name)))
        return dest_mdp_path

    def generate_MOF_gromacsfile(self) -> None:
        """
        Public entry point: Generates all GROMACS forcefield files for the MOF system.
        This includes .itp, .top, and .mdp files, copying or generating as needed.
        """
        database_path = self.database_dir
        itps_path = Path(self.target_dir, 'MD_run/itps')
        res_info = self.residues_info
        model_name = self.mof_name

        self._get_itps_from_database()
        self.top_path = self._generate_top_file(str(itps_path), database_path, res_info, model_name)
        self._copy_mdps()


if __name__ == "__main__":
    # Example usage for testing. Adjust paths and parameters as needed.
    gmx_ff = GromacsForcefieldMerger()
    gmx_ff.database_dir = 'tests/database'
    gmx_ff.target_dir = 'tests/out'
    gmx_ff.node_metal_type = 'Zr'
    gmx_ff.dummy_atom_node = True
    gmx_ff.termination_name = 'acetate'
    gmx_ff.linker_itp_dir = ''
    gmx_ff.linker_name = 'Linker'
    gmx_ff.residues_info = {"METAL": 5}
    gmx_ff.mof_name = "testmof"
    gmx_ff.generate_MOF_gromacsfile()
