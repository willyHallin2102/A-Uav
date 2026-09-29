"""
    data / loader.py
    ----------------
    Dataset loading and persistence utilities using `polars` and `pandas`.
    This module provides a single high-level `loader` interface for 
    reading, writing, concatenating, and iterating tabular datasets.
    This loader managing the UAV datasets primarily I/O purposes.
    ------------------------------------------------------------
    loader = Loader()
    df = loader.load("uav_beijing/train.csv")
    lazy = loader.lazy_many("uav_london/*.parquet")

    loader.save(df, "test/processed.parquet")
    ------------------------------------------------------------
"""
from __future__ import annotations

import os
import random
import pandas as pd
import polars as pl

from data.paths import DatasetRoot
from data.formats import (
    Format, fmt_from_suffix, reader_for, writer_for, filter_kwargs
)
import data.iterators as it

from pathlib import Path
from collections.abc import Iterator, Sequence
from typing import Any, Literal
from logs.logger import Logger, Level, Profile



ReturnKind = Literal["polars", "pandas"]

class Loader:
    """
    High-level interface for loading the UAV - datasets I / O and 
    dataframe conversions. `Loader` provides a API for reading and 
    also writing datasets with `root=Path(data/datasets)` without
    concern for the file-format. It supports lazy Polars execution,
    eager loading as Polars/Pandas, multi-file loading, sampling,
    batching, and atomic persistence. All datasets are resolved 
    as relative to the configured `DatasetRoot`.
    """
    def __init__(self, *,
        level: Level = Level.INFO, 
        profile: Profile = "runtime", 
        logger: Logger | None = None,
    ):
        """
            Initialize Loader Instance
        """
        self.logger = logger or Logger("Loader", level=level, profile=profile)
        self.root = DatasetRoot()
        self.logger.info("Loader initialized (root=%s)", self.root.root)



    # ======================================================================
    # Reading
    # ======================================================================
    def lazy(self, filepath: str | Path, *, columns: list[str] | None = None,
        dtypes: dict[str, pl.DataType] | None = None, **kwargs: Any,
    ) -> pl.LazyFrame:
        """
        Creates a lazy polars frame from a dataset. Resolves the path, 
        selects optional schema overrides and column projection. The 
        dataset is not materialized until the returned `LazyFrame` has
        been collected.

        -----
        Args:
        filepath: Dataset path relative to the dataset-root \
            `"data/datasets"`
        columns: Optional list of columns to read/select
        dtypes: Optional mapping of column names to Polars \
            data-types
        **kwargs: Additional arguments passed to the format-specific \
            reader.
        --------
        Returns:
            A Polars LazyFrame representing the dataset.
        --------
        """
        path = self.root.resolve(filepath)
        reader = reader_for(path.suffix)
        schema = dict(dtypes or {})
        lf = reader(path, schema_overrides=schema or None, **kwargs)
        if columns:
            lf = lf.select(columns)

        return lf


    def lazy_many(self,
        patterns: str | Sequence[str], *, columns: list[str] | None = None,
        dtypes: dict[str, pl.DataType] | None = None,
        how: Literal["vertical", "horizontal", "diagonal"] = "vertical",
        shuffle: bool = False, seed: int | None = None, **kwargs: Any,
    ) -> pl.LazyFrame:
        """
        Lazily loads and combines multiple datasets. Expands one or more 
        path patterns, creates a `LazyFrame` of these using a concatenation
        with the assigned strategy `how`.

        -----
        Args:
        patterns: Path patterns or `Sequence` of patterns to find the datasets.
        columns: Optional list of columns to select from each dataset.
        dtypes: Optional mapping of column names to Polars data types.
        how: Concatenation strategy passed to the lazy concatenation helper.
        shuffle: Whether to shuffle / randomize the order of matched paths.
        seed: Optional random seed used when shuffling paths.
        **kwargs: Additional arguments passed to the dataset reader.
        --------
        Returns:
            A combined Polars LazyFrame
        -------
        Raises:
        FileNotFoundError: If not datasets match the patterns provided.
        -------
        """
        paths = self.root.expand(patterns)
        if not paths:
            raise FileNotFoundError(f"No matched {patterns!r}: {self.root.root}")
        
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(paths)

        lfs = [
            self.lazy(p, columns=columns, dtypes=dtypes, **kwargs) for p in paths
        ]
        return it.concatenate_lazy(lfs, how=how)


    def load(self,
        filepath: str | Path, *, columns: list[str] | None = None,
        dtypes: dict[str, pl.DataType] | None = None, n_rows: int | None = None,
        sample: float | None = None, seed: int | None = None,
        kind: ReturnKind = "pandas", streaming: bool = True, **kwargs: Any,
    ) -> pl.DataFrame | pd.DataFrame:
        """
        Eager load a dataset into a Polars or Pandas DataFrame. The 
        dataset is read lazily and then materialized. Optionally row
        limits and random sampling may be applied after collection.

        -----
        Args:
        filepath: Dataset path relative to the dataset root.
        columns: Optional list of columns to load
        dtypes: Optionally mapping od column names to Polars data types.
        n_rows: Optional maximum number of rows to return
        sample: Optional fraction of rows to sample, in the range [0,1]
        seed: Optional random seed used in sampling.
        kind: Output dataframe type, either `polars` or `pandas`
        streaming: Whether to use Polars streaming or execution when \
            collecting.
        **kwargs: Additional arguments passed to the dataset reader.
        --------
        Returns:
            A Polars or Pandas DataFrame according to ``kind``
        -------
        Raises:
        ValueError: If `sample` is not in the range `[0,1]`
        -------
        """
        lf = self.lazy(filepath, columns=columns, dtypes=dtypes, **kwargs)

        if sample is not None:
            
            if not 0 < sample <= 1:
                raise ValueError("sample must be in (0, 1]")

            # Sampling requires materialization; streaming is not possible
            # for random sampling in Polars.
            df = lf.collect().sample(fraction=sample, seed=seed, shuffle=True)
            streaming = False

        else:
            df = lf.collect(streaming=streaming)

        if n_rows is not None:
            df = df.head(n_rows)

        return df if kind == "polars" else it.to_pandas(df)


    def load_many(self,
        patterns: str | Sequence[str], *, columns: list[str] | None = None,
        dtypes: dict[str, pl.DataType] | None = None,
        how: Literal["vertical", "horizontal", "diagonal"] = "vertical",
        n_rows: int | None = None, sample: float | None = None,
        seed: int | None = None, shuffle: bool = False,
        kind: ReturnKind = "pandas", streaming: bool = True, **kwargs: Any,
    ) -> pl.DataFrame | pd.DataFrame:
        """
        Eagerly loads and combine multiple datasets. Expands the supplied
        path patterns, lazily loads the matching datasets, combines them,
        and materializes the result as either Polars or Pandas.

        -----
        Args:
        patterns: Path pattern or sequence of patterns used to locate datasets.
        columns: Optional list of columns to load.
        dtypes: Optional mapping of column names to Polars data types.
        how: Concatenation strategy used to combine the datasets.
        n_rows: Optional maximum number of rows to return.
        sample: Optional fraction of rows to sample, in the range (0, 1].
        seed: Optional random seed used for sampling.
        shuffle: Whether to randomize the input path order.
        kind: Output dataframe type, either "polars" or "pandas".
        streaming: Whether to use Polars streaming execution when collecting.
        **kwargs: Additional arguments passed to the dataset readers.
        --------
        Returns:
            A combined Polars or Pandas DataFrame.
        -------
        Raises:
        FileNotFoundError: If no datasets match the supplied patterns.
        ValueError: If `sample` is not within the range (0,1]
        -------
        """
        lf = self.lazy_many(
            patterns, columns=columns, dtypes=dtypes, how=how,
            shuffle=shuffle, seed=seed, **kwargs,
        )

        if sample is not None:
            if not 0 < sample <= 1:
                raise ValueError("sample must be in (0, 1]")
            
            df = lf.collect().sample(fraction=sample, seed=seed, shuffle=True)

        else:
            df = lf.collect(streaming=streaming)

        if n_rows is not None:
            df = df.head(n_rows)

        return df if kind == "polars" else it.to_pandas(df)


    def load_iter(self,
        filepath: str | Path, *, size: int,
        columns: list[str] | None = None, 
        dtypes: dict[str, pl.DataType] | None = None,
        kind: ReturnKind = "pandas", **kwargs: Any,
    ) -> Iterator[pl.DataFrame | pd.DataFrame]:
        """
        Load a dataset and yield it as a sequence of row-batches. The
        dataset is materialized once and then divided into slices of the 
        requested size. Each yielded batch can be returned as either a 
        Polars or a Pandas DataFrame.

        -----
        Args:
        filepath: Dataset path relative to the dataset root.
        size: Maximum number of rows in each yielded batch.
        columns: Optional list of columns to load.
        dtypes: Optional mapping of column names to Polars data types.
        kind: Output dataframe type for each batch.
        **kwargs: Additional arguments passed to the dataset reader.
        -------
        Yields:
            DataFrame batches containing at most `size` rows.
        """
        df = self.load(
            filepath, columns=columns, dtypes=dtypes,
            kind="polars", streaming=False, **kwargs,
        )

        assert isinstance(df, pl.DataFrame)
        yield from it.iter_slices(df, size, as_polars=(kind == "polars"))

    # ======================================================================
    # Writing
    # ======================================================================
    def save(self,
        data: pl.DataFrame | pd.DataFrame, filepath: str | Path, *,
        fmt: Format | None = None, compression: str | None = "zstd",
        atomic: bool = True, overwrite: bool = True, **kwargs: Any,
    ) -> Path:
        """
        Write a dataframe to a dataset file. Converts the input to Polars,
        determines the output format, and delegates the serialization to
        the appropriate format - specific writer. Writers can be performed 
        atomically to avoid leaving partially written output files.

        -----
        Args:
        data: Polars or Pandas DataFrame to write.
        filepath: Destination path relative to the dataset root.
        fmt: Output format. If omitted, inferred from the file suffix.
        compression: Compression codec to use when supported by the format.
        atomic: Whether to write to a temporary file before replacing the \
            destination.
        overwrite: Whether an existing file may be replaced.
        **kwargs: Additional arguments passed to the format-specific writer.
        --------
        Returns:
            The resolved path of the written dataset.
        -------
        Raises:
        FileExistsError: If the destination exists and `overwrite` is False.
        """
        path = self.root.resolve(filepath, must_exist=False)

        if path.exists() and not overwrite:
            raise FileExistsError(path)

        fmt = fmt or fmt_from_suffix(path.suffix)
        df = it.to_polars(data)

        kwargs: dict[str, Any] = dict(writer_kwargs)
        if compression is not None:
            kwargs.setdefault("compression", compression)
        
        kwargs = filter_kwargs(fmt, kwargs)

        target = path.with_name(path.name + ".tmp") if atomic else path
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            writer_for(fmt)(df, target, **kwargs)
            if atomic:
                os.replace(target, path)
        
        finally:
            # Clean up the tmp file if something went wrong before rename.
            if atomic and target.exists() and target != path:
                try:
                    target.unlink()
        
                except OSError:
                    pass

        self.logger.info("Saved %s -> %s", fmt, path)
        return path


    def save_many(self,
        frames: Sequence[pl.DataFrame | pd.DataFrame], directory: str | Path,
        *, basename: str = "part", fmt: Format = "parquet", 
        compression: str | None = "zstd", atomic: bool = True,
        overwrite: bool = True, **kwargs: Any,
    ) -> list[Path]:
        """
        Write multiple dataframes as numbered dataset parts. Each frame
        is written as a separated file using the naming convention 
        `{basename}-{index:04d}.{extension}`.

        -----
        Args:
        frames: Sequence of Polars or Pandas DataFrames to write.
        directory: Destination directory relative to the dataset root.
        basename: Base name used for generated part files.
        fmt: Output format for all parts.
        compression: Compression codec to use when supported.
        atomic: Whether each file should be written atomically.
        overwrite: Whether existing part files may be replaced.
        **kwargs: Additional arguments passed to the format-specific \
            writer.
        --------
        Returns:
            A list of paths corresponding to the written part files.
        --------
        """
        outdir = self.root.resolve_dir(directory)
        ext = {
            "csv": ".csv", "parquet": ".parquet", "ipc": ".arrow",
            "json": ".json", "ndjson": ".ndjson"
        }[fmt]

        written: list[Path] = []
        for i, frame in enumerate(frames):
            name = f"{basename}-{i:04d}{ext}"
            written.append(self.save(
                frame, outdir.relative_to(self.root.root) / name,
                fmt=fmt, compression=compression, atomic=atomic,
                overwrite=overwrite, **kwargs,
            ))

        return written

    # ======================================================================
    # Convenience
    # ======================================================================
    def exists(self, filepath: str | Path) -> bool:
        """
        Check whether a dataset file exists.

        -----
        Args:
        filepath: Dataset path relative to the dataset root.
        --------
        Returns:
            True if the resolved path refers to an existing file, \
                otherwise False.
        --------
        """
        try:
            self.root.resolve(filepath, must_exist=False)
        
        except ValueError:
            return False
        
        return (self.root.root / filepath).is_file()

    def list(self, patterns: str | Sequence[str] = "**/*") -> list[Path]:
        """
        List dataset paths matching the supplied patterns.

        -----
        Args:
            patterns: Glob pattern or sequence of patterns used to find \
                datasets. Defaults to all paths below the dataset root.
        --------
        Returns:
            A list of matching paths.
        --------
        """
        return self.root.expand(patterns)
