MOFBuilder Manual
=================

MOFBuilder is a Python toolkit for constructing, inspecting, and preparing
Metal-Organic Framework (MOF) structures for downstream simulation workflows.
The documentation below is organized as a practical scientific software manual:
you can start from setup and examples, then move into API details.

.. image:: _static/images/welcomepage.png
   :alt: MOFBuilder welcome page image.
   :align: center
   :width: 80%

.. image:: _static/images/mofbuilder_workflow.svg
   :alt: MOFBuilder workflow from topology and linker inputs to framework and simulation preparation.
   :align: center
   :width: 95%

Highlights
----------

* Topology-guided MOF construction with configurable linker and node inputs.
* Framework export to common structure formats (`cif`, `pdb`, `gro`, `xyz`).
* Hooks for visualization and MD preparation workflows.
* Documentation focused on user-facing usage rather than internal helpers.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   manual/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   api_reference
