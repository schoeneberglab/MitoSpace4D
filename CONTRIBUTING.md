# Contributing to MitoSpace

Thanks for your interest in MitoSpace. This document describes how to set up a
development environment, the conventions the codebase follows, and how to
propose changes.

## License and scope

This repository is currently under a [Review License](./LICENSE): materials are
provided for evaluation of the associated *Cell* submission, and uses outside
those terms are not permitted until re-release after publication. During this
period, external contributions may be limited. If you are unsure whether a
change is in scope, [open an issue](https://github.com/schoeneberglab/MitoSpace4D/issues)
to discuss it before investing significant effort.

## Reporting bugs and requesting features

Please [open an issue](https://github.com/schoeneberglab/MitoSpace4D/issues).
A useful bug report includes:

- What you ran (command, config file, and relevant `config.yaml` fields).
- What you expected versus what happened, with the full traceback.
- Environment details: OS, Python version, PyTorch/CUDA versions, and GPU.

## Development setup

Follow the [Installation](./README.md#installation) steps, then install in
editable mode:

```bash
git clone https://github.com/schoeneberglab/MitoSpace4D.git
cd MitoSpace4D
conda create -n mitospace python=3.11
conda activate mitospace
pip install -e .
```

Python 3.11 or later is required (see `pyproject.toml`). Training and inference
with the released module require a CUDA GPU.

## Code style

The codebase is formatted with **[black](https://black.readthedocs.io/)** and
imports are sorted with **[isort](https://pycqa.github.io/isort/)**, both with
default settings. Run them before committing:

```bash
pip install black isort
isort .
black .
```

Please keep changes surgical: touch only what the change requires, and match the
surrounding style rather than reformatting unrelated code.

## Making changes

1. Create a branch off `main` using a descriptive prefix, consistent with the
   project's existing branches:
   - `fix/...` for bug fixes
   - `feat/...` for new functionality
   - `style/...` for formatting-only changes

   ```bash
   git checkout -b fix/short-description
   ```

2. Make your change and run `isort` and `black`.

3. Write clear, imperative commit messages (e.g. `Fix latent encoding path`).

4. Push your branch and open a pull request against `main`. In the description,
   explain the motivation, summarize the change, and note how you verified it.

Changes are reviewed and merged via pull request. Keep PRs focused on a single
concern where possible to make review straightforward.

## Questions

For questions not suited to an issue, contact the
[Schoeneberg Lab](https://github.com/schoeneberglab) on GitHub.
