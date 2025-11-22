from pathlib import Path
from typing import Dict, List
import yaml


class CategoryTaxonomy:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.raw = self._load()
        self.flat: Dict[str, str] = {}
        self._flatten()

    def _load(self) -> dict:
        with self.path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _flatten(self) -> None:
        def recurse(node_list: List[dict]):
            for node in node_list:
                cid = node["id"]
                label = node["label"]
                self.flat[cid] = label
                children = node.get("children") or []
                recurse(children)

        recurse(self.raw.get("categories", []))

    def get_all_labels(self) -> Dict[str, str]:
        return self.flat

    def reload(self) -> None:
        self.raw = self._load()
        self.flat = {}
        self._flatten()
