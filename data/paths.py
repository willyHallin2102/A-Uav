"""
    data / paths.py
    ---------------
    Module for managing the path resolution for the loader. All
    the supplied paths are resolved relative to a single root
    directory and verified to stay within it. This should prevent
    accidental reads / writes outside the dataset tree.
"""
from __future__ import annotations

import glob

from pathlib import Path
from collections.abc import Sequence



class DatasetRoot:
    """
    Resolves and expands paths inside a root directory
    """
    def __init__(self):
        """
            Initialize Dataset-Root Instance
        """
        self.root = Path(__file__).parent / "datasets"
        self.root = self.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
    

    def resolve(self, filepath: str | Path, *, must_exist: bool = True) -> Path:
        """
        Resolves the `filepath` under root, creates a stronger retainment 
        within the intended space of datasets, this should keep all files 
        read from to be in this directory as all which is being written 
        to also be kept within the directory.
        """
        path = (self.root / filepath).resolve()
        try:
            path.relative_to(self.root)
        
        except ValueError as e:
            raise ValueError(f"Path {filepath!r} not in root: {self.root}") from e
        
        if must_exist and not path.is_file():
            raise FileNotFoundError(path)
        
        return path
    

    def expand(self, patterns: str | Sequence[str]) -> list[Path]:
        """
        Expand one or more glob patterns (relative to the root) into
        a sorted list of files. Supports `**` for recursion.
        """
        patterns = [patterns] if isinstance(patterns, str) else list(patterns)
        found: list[Path] = []

        for pattern in patterns:
            pattern = str(self.root / pattern)
            found.extend(Path(p) for p in glob.glob(pattern, recursive=True))
        
        seen: set[Path] = set()
        result: list[Path] = []

        for path in sorted(found):
            relative_path = path.resolve()
            try:
                relative_path.relative_to(self.root)
            
            except ValueError:
                continue

            if relative_path.is_file() and relative_path not in seen:
                seen.add(path)
                result.append(relative_path)
        
        return result

        



































# from data import Loader

# ld = Loader(root="./data/datasets")

# # ---- single file ----
# df = ld.load("train.csv", columns=["a", "b"], kind="pandas")

# # ---- lazy plan, nothing read yet ----
# lf = ld.lazy("train.parquet", columns=["a", "b"])
# out = lf.filter(pl.col("a") > 0).group_by("b").agg(pl.len()).collect()

# # ---- multi-file, lazy concat, shuffled file order ----
# lf = ld.lazy_many(
#     ["shards/**/*.parquet", "extra/*.parquet"],
#     shuffle_paths=True, seed=42,
# )
# df = ld.load_many(
#     "shards/**/*.parquet",
#     sample=0.1, seed=42, kind="polars",
# )

# # ---- chunked iteration ----
# for chunk in ld.load_iter("big.csv", size=100_000, kind="polars"):
#     process(chunk)

# # ---- saving ----
# ld.save(df, "out/clean.parquet")               # atomic, zstd
# ld.save(df, "out/clean.json")                  # compression dropped automatically
# ld.save_many(chunks, "out/shards", basename="part")  # part-0000.parquet, ...

# # ---- utilities ----
# ld.exists("out/clean.parquet")
# ld.list("**/*.parquet")