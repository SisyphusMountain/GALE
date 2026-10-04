import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "gpurec"


def _environment_reads(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(), filename=str(path))
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}:
            if isinstance(node.value, ast.Name) and node.value.id == "os":
                lines.append(node.lineno)
    return lines


def test_package_reads_no_environment_variables():
    offenders = {
        str(path.relative_to(PACKAGE)): lines
        for path in PACKAGE.rglob("*.py")
        if (lines := _environment_reads(path))
    }
    assert offenders == {}


def test_warm_adjoint_is_a_solver_setting():
    from gpurec.config import GpurecConfig, load_config

    assert GpurecConfig().solver.warm_adjoint is False
    assert load_config(None).solver.warm_adjoint is False
