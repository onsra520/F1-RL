import ast
import importlib.util
from pathlib import Path


def imports_project_layers(path: Path, layers: tuple[str, ...]) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    parts = path.parts
    package = ".".join(parts[parts.index("src"):-1])
    absolute = tuple("src." + layer for layer in layers)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = importlib.util.resolve_name("." * node.level + module, package)
            names = [module] + [(module + "." if module else "") + alias.name for alias in node.names]
        else:
            continue
        if any(name == prefix or name.startswith(prefix + ".") for name in names for prefix in absolute):
            return True
    return False
