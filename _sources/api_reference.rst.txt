API Reference
=============

The API reference below intentionally focuses on public interfaces used in
typical workflows. Internal helper functions and imported implementation details
are omitted to keep the reference concise.

Builder Workflow
----------------

.. autosummary::
   :toctree: api_generated
   :nosignatures:

   mofbuilder.MetalOrganicFrameworkBuilder
   mofbuilder.core.framework.Framework

Core Components
---------------

.. autosummary::
   :toctree: api_generated
   :nosignatures:

   mofbuilder.core.FrameNet
   mofbuilder.core.FrameNode
   mofbuilder.core.FrameLinker
   mofbuilder.core.FrameTermination
   mofbuilder.core.MofTopLibrary
   mofbuilder.core.NetOptimizer
   mofbuilder.core.SupercellBuilder
   mofbuilder.core.EdgeGraphBuilder
   mofbuilder.core.TerminationDefectGenerator
   mofbuilder.core.MofWriter

Input and Output
----------------

.. autosummary::
   :toctree: api_generated
   :nosignatures:

   mofbuilder.io.CifReader
   mofbuilder.io.CifWriter
   mofbuilder.io.PdbReader
   mofbuilder.io.PdbWriter
   mofbuilder.io.GroReader
   mofbuilder.io.GroWriter
   mofbuilder.io.XyzReader
   mofbuilder.io.XyzWriter

Modelling and Simulation
------------------------

.. autosummary::
   :toctree: api_generated
   :nosignatures:

   mofbuilder.md.OpenmmSetup
   mofbuilder.md.SolvationBuilder
   mofbuilder.md.LinkerForceFieldGenerator
   mofbuilder.md.GromacsForcefieldMerger
   mofbuilder.md.ForceFieldMapper

Visualization
-------------

.. autosummary::
   :toctree: api_generated
   :nosignatures:

   mofbuilder.visualization.Viewer

Utilities
---------

.. autosummary::
   :toctree: api_generated
   :nosignatures:

   mofbuilder.utils.fetch_pdbfile
   mofbuilder.utils.unit_cell_to_cartesian_matrix
   mofbuilder.utils.fractional_to_cartesian
   mofbuilder.utils.cartesian_to_fractional
