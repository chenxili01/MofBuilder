# Installation

This guide installs MofBuilder from source and configures optional stacks for
development and documentation.

## 1. Clone the repository

```bash
git clone https://github.com/chenxili01/MofBuilder.git
cd MofBuilder
```

## 2. Create and activate an environment

MofBuilder targets Python `>=3.8`. A clean environment is strongly recommended.

```bash
conda create -n mofbuilder python=3.10
conda activate mofbuilder
```

## 3. Install the package

```bash
pip install -e .
```

## 4. Optional dependency groups

Install only the extras you need:

```bash
# Development tooling (pytest, formatting, linting)
pip install -e ".[dev]"

# Documentation toolchain (Sphinx + MyST + theme)
pip install -e ".[docs]"

# Additional stacks
pip install -e ".[core,md,visualization]"
```

## 5. Build the documentation locally

```bash
cd docs
make html
```

The built HTML site is written to `docs/_build/html`.

## 6. Smoke check

After installation, verify the package import works:

```bash
python -c "import mofbuilder; print(mofbuilder.__version__)"
```

For command-line checks:

```bash
mofbuilder --version
mofbuilder list-families
```
