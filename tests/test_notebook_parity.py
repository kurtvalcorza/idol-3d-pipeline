"""NOTEBOOK_SPEC parity (PAR1–PAR3, SRC13) and the kernel boundary: the notebook carries the committed files unchanged,
installs nothing into the kernel, and never prints or stores the token."""

from __future__ import annotations

# ruff: noqa: E501  -- messages and records are kept on one line each
import ast
import json

import pytest

from conftest import ROOT, load_tool

build = load_tool("build_notebook")
TEMPLATE = load_tool("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _carrier(notebook):
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code" and c.get("metadata", {}).get("dimer", {}).get("embedded_sources")]
    assert len(cells) == 1
    values = {}
    for node in ast.parse(_source(cells[0])).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("CARRIED_FILES", "CARRIED_HASHES"):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    return values["CARRIED_FILES"], values["CARRIED_HASHES"]


def test_committed_notebook_equals_the_generator_output():
    assert build.main(["--check"]) == 0


def test_carried_files_equal_repository_sources(notebook):
    files, hashes = _carrier(notebook)
    assert list(files) == [*TEMPLATE["carried"], build.SOURCE_RECORD]
    for dest, source in TEMPLATE["carried"].items():
        assert files[dest] == (ROOT / source).read_text(encoding="utf-8"), f"carried {dest} drifted from {source}; regenerate the notebook"
        assert hashes[dest] == build.sha256_text(files[dest])


def test_the_whole_vendored_sdk_is_carried(notebook):
    files, _ = _carrier(notebook)
    vendored = {str(p.relative_to(ROOT)) for p in (ROOT / "src" / "idol").rglob("*.py")}
    assert vendored <= set(files)


def test_kernel_installs_nothing_and_every_code_cell_parses(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = _source(cell)
        ast.parse(source)
        assert "pip install" not in source and "!pip" not in source and "%pip" not in source


def test_token_is_read_once_passed_to_stages_and_never_printed(notebook):
    sources = [_source(c) for c in notebook["cells"] if c["cell_type"] == "code"]
    token_cells = [s for s in sources if "userdata.get('HF_TOKEN')" in s]
    assert len(token_cells) == 1
    assert "print(HF_TOKEN_VALUE" not in token_cells[0] and "len(HF_TOKEN_VALUE)" in token_cells[0]
    install = next(s for s in sources if "def run_stage" in s)
    assert "ENV['HF_TOKEN'] = HF_TOKEN_VALUE" in install
    assert all("write_text(HF_TOKEN_VALUE" not in s for s in sources)


def test_metadata_declares_profile_spec_and_candidate_boundary(notebook):
    meta = notebook["metadata"]["dimer"]
    assert meta["notebook_profile"] == "TASK-INFERENCE" and meta["notebook_mode"] == "GUIDED"
    assert meta["notebook_spec"] == build.NOTEBOOK_SPEC == "3.1"
    header = _source(notebook["cells"][0])
    assert "Credential" not in header or "RUN13" in _source(next(c for c in notebook["cells"] if _source(c).startswith("## Prerequisites")))


def test_lock_pins_every_runtime_dependency_with_hashes():
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    build.check_lock(build._pins(ROOT), lock)
    assert "flash-attn" not in lock and "flash_attn" not in lock
