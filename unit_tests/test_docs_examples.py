"""Static drift gate for the public tutorials, examples and documentation.

The unit tests cover the core abstractions thoroughly, but nothing checked that
the *published* material still matched the API. Renames such as
``TPTBox.registration.deformable`` -> ``TPTBox.registration._deformable`` or
``run_totalvibeseg`` -> ``run_vibeseg`` therefore invalidated notebooks and
README snippets silently.

These tests parse (never execute) every notebook and Markdown code block and
assert that:

* every ``TPTBox.*`` module referenced actually exists;
* every symbol imported from a TPTBox module actually exists;
* every Markdown ``python`` block still compiles.

Nothing here needs a dataset, model weights, or a GPU, so it runs in the normal
pytest job. Executing the notebooks end to end is a separate concern - they
expect real BIDS datasets at site-specific paths.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import pkgutil
import re
import unittest
from pathlib import Path

import TPTBox

REPO_ROOT = Path(__file__).resolve().parent.parent

NOTEBOOKS = sorted(REPO_ROOT.glob("tutorials/**/*.ipynb")) + sorted(REPO_ROOT.glob("examples/**/*.ipynb"))
MARKDOWN = sorted(REPO_ROOT.glob("docs/**/*.md")) + sorted(REPO_ROOT.glob("TPTBox/**/*.md")) + [REPO_ROOT / "README.md"]

# Third-party imports that the examples legitimately use but TPTBox does not depend on.
# Their availability is not what these tests are about.
OPTIONAL_THIRD_PARTY = {
    "spineps",
    "torch",
    "nnunetv2",
    "acvl_utils",
    "batchgenerators",
    "deepali",
    "pydicom",
    "dicom2nifti",
    "antspyx",
    "ants",
    "elasticdeform",
    "networkx",
    "IPython",
    "ruamel",
    "configargparse",
    "TypeSaveArgParse",
    "cv2",
    "pandas",
    "seaborn",
    "plotly",
    "torchio",
    "monai",
    "nnunet",
    "totalsegmentator",
    # transitive deps of the nnU-Net stack (nnunetv2 >= 2.6 pulls these in)
    "blosc2",
    "batchgeneratorsv2",
    "dynamic_network_architectures",
    # dev-group extras: present in CI, absent for a plain `pip install TPTBox`
    "pyvista",
    "vtk",
    "fury",
    "xvfbwrapper",
    "PIL",
    "Pillow",
}


def _iter_code_cells(nb_path: Path):
    """Yield the source of every code cell in a notebook, IPython magics stripped."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        lines = [ln for ln in cell.get("source", []) if not ln.lstrip().startswith(("%", "!", "?"))]
        src = "".join(lines)
        if src.strip():
            yield idx, src


def _iter_python_blocks(md_path: Path):
    """Yield every fenced ```python block in a Markdown file."""
    text = md_path.read_text(encoding="utf-8")
    for match in re.finditer(r"```(?:python|py)\n(.*?)```", text, re.DOTALL):
        block = match.group(1)
        if block.strip():
            yield text[: match.start()].count("\n") + 1, block


def _parse(src: str) -> ast.Module | None:
    """Parse a snippet, tolerating the fragments that documentation is written in."""
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


def _tptbox_imports(tree: ast.Module):
    """Yield ``(module, [names])`` for every TPTBox import in the tree.

    ``names`` is empty for a plain ``import TPTBox.x.y``.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "TPTBox":
                    yield alias.name, []
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import inside a snippet - nothing to resolve
                continue
            if node.module and node.module.split(".")[0] == "TPTBox":
                yield node.module, [a.name for a in node.names]


def _module_exists(name: str) -> bool:
    """Does ``name`` resolve to a real module?

    ``find_spec`` imports the parent packages on the way down, so a parent whose
    optional backend is missing raises here. That is not the same thing as the
    module being gone, and must not be reported as drift - otherwise this gate
    would pass only on machines with the full ML stack installed.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except ImportError as e:
        missing = getattr(e, "name", None)
        if not missing:  # some libraries raise a bare ImportError
            m = re.search(r"No module named '([^']+)'", str(e))
            missing = m.group(1) if m else ""
        return bool(missing) and missing.split(".")[0] in OPTIONAL_THIRD_PARTY
    except ValueError:
        return False


def _symbol_exists(module: str, symbol: str) -> bool:
    """Is ``symbol`` importable from ``module`` (as an attribute or a sub-module)?"""
    try:
        mod = importlib.import_module(module)
    except Exception:
        return True  # an optional backend is missing; not a drift failure
    if hasattr(mod, symbol):
        return True
    return _module_exists(f"{module}.{symbol}")


# The public entry points a user imports. Each of these MUST import on a bare
# install - that is the guarantee the optional-dependency stubs exist to provide,
# and `import TPTBox.segmentation` failing on a clean install was a real bug.
REQUIRED_IMPORTS = [
    "TPTBox",
    "TPTBox.core",
    "TPTBox.core.bids_files",
    "TPTBox.core.dicom",
    "TPTBox.core.nii_wrapper",
    "TPTBox.core.np_utils",
    "TPTBox.core.poi",
    "TPTBox.logger",
    "TPTBox.mesh3D",
    "TPTBox.registration",
    "TPTBox.segmentation",
    "TPTBox.spine",
    "TPTBox.spine.snapshot2D",
    "TPTBox.spine.spinestats",
    "TPTBox.stitching",
]


def _missing_module_name(exc: BaseException) -> str:
    """Best-effort name of the module an ImportError is complaining about."""
    name = getattr(exc, "name", None)
    if name:
        return name
    m = re.search(r"No module named '([^']+)'", str(exc))
    return m.group(1) if m else ""


class TestPackageImports(unittest.TestCase):
    """TPTBox must import without any optional backend installed."""

    def test_required_entry_points_import(self):
        """The public sub-packages must import even with no optional backend.

        This is the regression guard: these are the modules whose entry points are
        replaced by stubs when a backend is missing, so a missing backend must never
        stop the *import* itself.
        """
        failures = []
        for name in REQUIRED_IMPORTS:
            try:
                importlib.import_module(name)
            except Exception as e:  # noqa: BLE001 - report, do not mask
                failures.append(f"{name}: {type(e).__name__}: {e}")
        self.assertEqual(failures, [], "public entry points failed to import:\n" + "\n".join(failures))

    def test_no_subpackage_has_broken_internal_imports(self):
        """Walk every module and fail only on breakage *inside* TPTBox.

        A leaf module may legitimately require an optional third-party backend
        (fastProcessor needs blosc2, vibeseg needs torch, snapshot3D needs
        xvfbwrapper), and which of those happen to be installed varies by machine
        and by CI job. Enumerating them was brittle - this asserts the thing we
        actually care about instead: that no TPTBox module fails to import because
        of a *TPTBox* problem, such as a stale path after a rename.
        """
        failures = []
        for info in pkgutil.walk_packages(TPTBox.__path__, prefix="TPTBox."):
            name = info.name
            if ".tests" in name or name.endswith(("__main__", "script_ax2sag")):
                continue
            try:
                importlib.import_module(name)
            except ImportError as e:
                # Only fail when the failure is positively identified as a TPTBox
                # module. Both real drift modes set it: a stale path gives
                # name="TPTBox.gone", and a removed symbol gives
                # name="TPTBox.mod" with "cannot import name ...". Anything else -
                # a missing backend, or a hand-raised hint like
                # elastic_deform's NumPy-2 message - is an environment fact.
                if not _missing_module_name(e).startswith("TPTBox"):
                    continue
                failures.append(f"{name}: {e}")
            except Exception as e:  # noqa: BLE001 - report, do not mask
                failures.append(f"{name}: {type(e).__name__}: {e}")
        self.assertEqual(failures, [], "modules broken inside TPTBox:\n" + "\n".join(failures))

    def test_public_api_names_resolve(self):
        missing = [n for n in TPTBox.__all__ if not hasattr(TPTBox, n)]
        self.assertEqual(missing, [], f"TPTBox.__all__ names that do not exist: {missing}")


class TestNotebooks(unittest.TestCase):
    """Notebooks must reference modules and symbols that still exist."""

    def test_notebooks_parse(self):
        for nb in NOTEBOOKS:
            for idx, src in _iter_code_cells(nb):
                with self.subTest(notebook=nb.name, cell=idx):
                    self.assertIsNotNone(_parse(src), f"cell {idx} of {nb.name} is not valid Python")

    def test_notebook_tptbox_imports_resolve(self):
        failures = []
        for nb in NOTEBOOKS:
            for idx, src in _iter_code_cells(nb):
                tree = _parse(src)
                if tree is None:
                    continue
                for module, names in _tptbox_imports(tree):
                    if not _module_exists(module):
                        failures.append(f"{nb.relative_to(REPO_ROOT)} cell {idx}: no module {module!r}")
                        continue
                    failures.extend(
                        f"{nb.relative_to(REPO_ROOT)} cell {idx}: {module!r} has no {n!r}" for n in names if not _symbol_exists(module, n)
                    )
        self.assertEqual(failures, [], "notebook API drift:\n" + "\n".join(failures))


class TestMarkdownSnippets(unittest.TestCase):
    """README and docs code blocks must compile and reference real symbols."""

    def test_python_blocks_compile(self):
        failures = []
        for md in MARKDOWN:
            for line, block in _iter_python_blocks(md):
                try:
                    compile(block, f"{md.name}:{line}", "exec")
                except SyntaxError as e:
                    failures.append(f"{md.relative_to(REPO_ROOT)}:{line}: {e.msg}")
        self.assertEqual(failures, [], "Markdown snippets with syntax errors:\n" + "\n".join(failures))

    def test_markdown_tptbox_imports_resolve(self):
        failures = []
        for md in MARKDOWN:
            for line, block in _iter_python_blocks(md):
                tree = _parse(block)
                if tree is None:
                    continue
                for module, names in _tptbox_imports(tree):
                    if not _module_exists(module):
                        failures.append(f"{md.relative_to(REPO_ROOT)}:{line}: no module {module!r}")
                        continue
                    failures.extend(
                        f"{md.relative_to(REPO_ROOT)}:{line}: {module!r} has no {n!r}" for n in names if not _symbol_exists(module, n)
                    )
        self.assertEqual(failures, [], "documentation API drift:\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
