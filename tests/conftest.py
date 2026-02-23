import sys
import types
from pathlib import Path


# Ensure local package imports work in CI without editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_mpi_stub() -> None:
    if "mpi4py" in sys.modules:
        return

    class _FakeComm:

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

    mpi4py_module = types.ModuleType("mpi4py")
    mpi_submodule = types.ModuleType("mpi4py.MPI")
    mpi_submodule.COMM_WORLD = _FakeComm()
    mpi4py_module.MPI = mpi_submodule
    sys.modules["mpi4py"] = mpi4py_module
    sys.modules["mpi4py.MPI"] = mpi_submodule


def _install_veloxchem_stub() -> None:
    if "veloxchem" in sys.modules:
        return

    veloxchem_module = types.ModuleType("veloxchem")

    outputstream_module = types.ModuleType("veloxchem.outputstream")

    class OutputStream:

        def __init__(self, _stream):
            self.messages = []

        def print_info(self, msg):
            self.messages.append(("info", str(msg)))

        def print_warning(self, msg):
            self.messages.append(("warning", str(msg)))

        def print_title(self, msg):
            self.messages.append(("title", str(msg)))

        def print_separator(self):
            self.messages.append(("separator", ""))

        def flush(self):
            return None

    outputstream_module.OutputStream = OutputStream

    veloxchemlib_module = types.ModuleType("veloxchem.veloxchemlib")
    veloxchemlib_module.mpi_master = lambda: 0

    errorhandler_module = types.ModuleType("veloxchem.errorhandler")

    def assert_msg_critical(cond, msg):
        if not cond:
            raise AssertionError(msg)

    errorhandler_module.assert_msg_critical = assert_msg_critical

    molecule_module = types.ModuleType("veloxchem.molecule")

    class Molecule:
        """Minimal test stub for veloxchem.molecule.Molecule."""

        def __init__(self, labels, coords):
            import numpy as _np

            self._labels = list(labels)
            self._coords = _np.asarray(coords, dtype=float)

        @classmethod
        def _from_xyz_lines(cls, lines):
            labels = []
            coords = []
            for line in lines:
                parts = line.split()
                if len(parts) < 4:
                    continue
                labels.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            return cls(labels, coords)

        @classmethod
        def read_xyz_file(cls, filepath):
            text = Path(filepath).read_text(encoding="utf-8").splitlines()
            data_lines = text[2:] if len(text) >= 2 else text
            return cls._from_xyz_lines(data_lines)

        @classmethod
        def read_xyz_string(cls, xyz_string):
            lines = xyz_string.splitlines()
            data_lines = lines[2:] if len(lines) >= 2 else lines
            return cls._from_xyz_lines(data_lines)

        @classmethod
        def read_smiles(cls, _smiles):
            # Minimal 2-atom placeholder for tests that only require object presence.
            return cls(["C", "C"], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])

        def get_labels(self):
            return list(self._labels)

        def get_coordinates_in_angstrom(self):
            import numpy as _np

            return _np.array(self._coords, dtype=float)

        def get_distance_matrix_in_angstrom(self):
            import numpy as _np

            c = self.get_coordinates_in_angstrom()
            diff = c[:, None, :] - c[None, :, :]
            return _np.linalg.norm(diff, axis=2)

        def get_connectivity_matrix(self):
            import numpy as _np

            d = self.get_distance_matrix_in_angstrom()
            # Generic covalent-like threshold for testing graph logic.
            conn = ((d > 1e-8) & (d < 1.95)).astype(int)
            _np.fill_diagonal(conn, 0)
            return conn

        def center_of_mass_in_bohr(self):
            # Simple geometric center is sufficient for unit tests.
            center_ang = self.get_coordinates_in_angstrom().mean(axis=0)
            return center_ang / 0.529177

    molecule_module.Molecule = Molecule

    # Additional veloxchem submodules imported by md/* and core/*.
    def _dummy_class(name):
        return type(name, (), {})

    mmff_module = types.ModuleType("veloxchem.mmforcefieldgenerator")
    mmff_module.MMForceFieldGenerator = _dummy_class("MMForceFieldGenerator")

    xtb_module = types.ModuleType("veloxchem.xtbdriver")
    xtb_module.XtbDriver = _dummy_class("XtbDriver")

    opt_module = types.ModuleType("veloxchem.optimizationdriver")
    opt_module.OptimizationDriver = _dummy_class("OptimizationDriver")

    mb_module = types.ModuleType("veloxchem.molecularbasis")
    mb_module.MolecularBasis = _dummy_class("MolecularBasis")

    scfr_module = types.ModuleType("veloxchem.scfrestdriver")
    scfr_module.ScfRestrictedDriver = _dummy_class("ScfRestrictedDriver")

    scfu_module = types.ModuleType("veloxchem.scfunrestdriver")
    scfu_module.ScfUnrestrictedDriver = _dummy_class("ScfUnrestrictedDriver")

    env_module = types.ModuleType("veloxchem.environment")
    env_module.get_data_path = lambda: Path.cwd() / "database"

    veloxchemlib_module.hartree_in_kcalpermol = 627.509
    veloxchemlib_module.hartree_in_kjpermol = 2625.5

    sys.modules["veloxchem"] = veloxchem_module
    sys.modules["veloxchem.outputstream"] = outputstream_module
    sys.modules["veloxchem.veloxchemlib"] = veloxchemlib_module
    sys.modules["veloxchem.errorhandler"] = errorhandler_module
    sys.modules["veloxchem.molecule"] = molecule_module
    sys.modules["veloxchem.mmforcefieldgenerator"] = mmff_module
    sys.modules["veloxchem.xtbdriver"] = xtb_module
    sys.modules["veloxchem.optimizationdriver"] = opt_module
    sys.modules["veloxchem.molecularbasis"] = mb_module
    sys.modules["veloxchem.scfrestdriver"] = scfr_module
    sys.modules["veloxchem.scfunrestdriver"] = scfu_module
    sys.modules["veloxchem.environment"] = env_module


def _install_openmm_stub() -> None:
    if "openmm" in sys.modules:
        return

    openmm_module = types.ModuleType("openmm")
    openmm_app_module = types.ModuleType("openmm.app")
    openmm_unit_module = types.ModuleType("openmm.unit")

    def _dummy_class(name):
        return type(name, (), {"__init__": lambda self, *a, **k: None})

    # app namespace classes/constants imported by md.setup
    for cls_name in [
            "GromacsGroFile",
            "GromacsTopFile",
            "PDBFile",
            "PDBReporter",
            "Simulation",
            "StateDataReporter",
    ]:
        setattr(openmm_app_module, cls_name, _dummy_class(cls_name))
    openmm_app_module.NoCutoff = object()
    openmm_app_module.PME = object()
    openmm_app_module.HBonds = object()

    openmm_module.LangevinIntegrator = _dummy_class("LangevinIntegrator")
    openmm_module.MonteCarloBarostat = _dummy_class("MonteCarloBarostat")

    # unit names used by md.setup
    for unit_name in [
            "atmosphere",
            "picosecond",
            "picoseconds",
            "kelvin",
            "femtoseconds",
            "nanometer",
    ]:
        setattr(openmm_unit_module, unit_name, 1.0)

    sys.modules["openmm"] = openmm_module
    sys.modules["openmm.app"] = openmm_app_module
    sys.modules["openmm.unit"] = openmm_unit_module


def _install_scipy_stub() -> None:
    if "scipy" in sys.modules:
        return

    scipy_module = types.ModuleType("scipy")
    scipy_spatial = types.ModuleType("scipy.spatial")
    scipy_optimize = types.ModuleType("scipy.optimize")
    scipy_transform = types.ModuleType("scipy.spatial.transform")

    class cKDTree:

        def __init__(self, data):
            self.data = data

        def query(self, points, k=1):
            import numpy as _np

            pts = _np.atleast_2d(points)
            d = _np.zeros((len(pts),))
            idx = _np.zeros((len(pts),), dtype=int)
            return d if k == 1 else _np.zeros((len(pts), k)), idx if k == 1 else _np.zeros((len(pts), k), dtype=int)

    class Rotation:
        @staticmethod
        def random():
            return Rotation()

        def as_matrix(self):
            import numpy as _np

            return _np.eye(3)

    scipy_spatial.cKDTree = cKDTree
    scipy_transform.Rotation = Rotation
    scipy_optimize.linear_sum_assignment = lambda cost: (
        __import__("numpy").arange(min(cost.shape)),
        __import__("numpy").arange(min(cost.shape)),
    )
    scipy_optimize.minimize = lambda fun, x0, **kwargs: types.SimpleNamespace(
        x=x0, success=True, fun=float(fun(x0)), message="stub minimize"
    )

    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.spatial"] = scipy_spatial
    sys.modules["scipy.spatial.transform"] = scipy_transform
    sys.modules["scipy.optimize"] = scipy_optimize


def _install_rdkit_stub() -> None:
    if "rdkit" in sys.modules:
        return

    rdkit_module = types.ModuleType("rdkit")
    chem_module = types.ModuleType("rdkit.Chem")
    rdkit_module.Chem = chem_module
    sys.modules["rdkit"] = rdkit_module
    sys.modules["rdkit.Chem"] = chem_module


def _install_networkx_stub() -> None:
    if "networkx" in sys.modules:
        return

    nx_module = types.ModuleType("networkx")
    nx_algorithms = types.ModuleType("networkx.algorithms")
    nx_iso = types.ModuleType("networkx.algorithms.isomorphism")

    class Graph:

        def __init__(self):
            self.nodes = {}
            self.edges = set()

        def add_node(self, node, **attrs):
            self.nodes[node] = dict(attrs)

        def add_edge(self, i, j):
            self.edges.add(tuple(sorted((i, j))))

    class GraphMatcher:

        def __init__(self, g1, g2, node_match=None):
            self.g1 = g1
            self.g2 = g2
            self.node_match = node_match or (lambda *_: True)

        def is_isomorphic(self):
            if len(self.g1.nodes) != len(self.g2.nodes):
                return False
            if len(self.g1.edges) != len(self.g2.edges):
                return False
            keys1 = sorted(self.g1.nodes)
            keys2 = sorted(self.g2.nodes)
            return all(
                self.node_match(self.g1.nodes[k1], self.g2.nodes[k2])
                for k1, k2 in zip(keys1, keys2)
            )

        def isomorphisms_iter(self):
            if not self.is_isomorphic():
                return iter(())
            keys1 = sorted(self.g1.nodes)
            keys2 = sorted(self.g2.nodes)
            return iter(({k1: k2 for k1, k2 in zip(keys1, keys2)},))

    nx_module.Graph = Graph
    nx_iso.GraphMatcher = GraphMatcher
    nx_module.algorithms = nx_algorithms
    nx_algorithms.isomorphism = nx_iso

    sys.modules["networkx"] = nx_module
    sys.modules["networkx.algorithms"] = nx_algorithms
    sys.modules["networkx.algorithms.isomorphism"] = nx_iso


_install_mpi_stub()
_install_veloxchem_stub()
_install_openmm_stub()
_install_scipy_stub()
_install_rdkit_stub()
_install_networkx_stub()
