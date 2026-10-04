import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "gpurec"
# The command line writes usage errors to stderr itself; everything else uses logging.
ALLOWED = set()


def _print_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
    ]


def test_package_uses_logging_not_print():
    offenders = {
        str(path.relative_to(PACKAGE)): lines
        for path in PACKAGE.rglob("*.py")
        if str(path.relative_to(PACKAGE)) not in ALLOWED and (lines := _print_calls(path))
    }
    assert offenders == {}
