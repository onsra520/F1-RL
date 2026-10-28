from pathlib import Path

import yaml


def load_yaml_document(path: str | Path, *, resolve: bool) -> tuple[Path, dict]:
    if not isinstance(path, (str, Path)):
        raise ValueError("path must be a string or Path")
    if isinstance(path, str) and not path.strip():
        raise ValueError("path must not be blank")
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError("configuration path must exist as a file")
    if resolve:
        config_path = config_path.resolve()
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("configuration file could not be loaded") from error
    if not isinstance(document, dict):
        raise ValueError("configuration root must be a mapping")
    return config_path, document
