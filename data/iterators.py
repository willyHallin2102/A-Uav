"""
    data / iterators.py
    -------------------
    DataFrame iteration and concatenation utilities. This module 
    provides pure helper functions for converting between Polars
    and Pandas DataFrames, splitting completed DataFrames into 
    row batches, and concatenating eager or lazy Polar frames.
    The helper exclusively use in-memory DataFrames and 
    LazyFrames.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from collections.abc import Iterable, Iterator
PolarsOrPandas = pl.DataFrame | pl.DataFrame


def to_polars(data: PolarsOrPandas) -> pd.DataFrame:
    """
    Convert a Polars or Pandas DataFrame to a Polars DataFrame. Polars
    inputs are returned unchanged. Pandas inputs are converted without
    preserving the Pandas index as a dataframe column.

    -----
    Args:
    data: Polars or Pandas DataFrame to convert.
    --------
    Returns:
        A Polars DataFrame.
    -------
    Raises:
    TypeError: If `data` is neither a Polars nor Pandas DataFrame.
    -------
    """
    if isinstance(data, pl.DataFrame):
        return data
    
    if isinstance(data, pd.DataFrame):
        return pl.from_pandas(data, include_index=False)
    
    raise TypeError(f"Expected polars / pandas DataFrame, got {type(data)!r}")


def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """
    Convert a Polars DataFrame to a Pandas DataFrame.

    -----
    Args:
    df: Polars DataFrame to convert.
    --------
    Returns:
        A Pandas DataFrame containing the same tabular data.
    """
    return df.to_pandas()



def iter_slices(
    df: pl.DataFrame, size: int, *, as_polars: bool = False
) -> Iterator[PolarsOrPandas]:
    """
    Yield a DataFrame as consecutive row-sized batches. The input 
    DataFrame is divided into slices of at most ``size`` rows. Each 
    slice can be returned as either a Polars or Pandas DataFrame.

    -----
    Args:
    df: Polars DataFrame to divide into batches.
    size: Maximum number of rows per batch.
    as_polars: If True, yield Polars DataFrames; otherwise yield \
        Pandas DataFrames.
    -------
    Yields:
        Consecutive DataFrame slices containing at most ``size`` rows.
    -------
    Raises:
    ValueError: If ``size`` is not positive.
    -------
    """
    if size <= 0:
        raise ValueError("size must be positive")

    for start in range(0, df.height, size):
        chunk = df.slice(start, size)
        yield chunk if as_polars else chunk.to_pandas()


def concatenate(
    frames: Iterable[PolarsOrPandas], *, how: str = "vertical",
    as_polars: bool = False, rechunk: bool = False,
) -> PolarsOrPandas:
    """
    Concatenate an iterable of Polars or Pandas DataFrames. All
    inputs are converted to Polars before concatenation. The 
    concatenation semantics are delegated to Polars.

    -----
    Args:
    frames: Iterable of Polars or Pandas DataFrames to concatenate.
    how: Polars concatenation strategy. Common options include \
        `"vertical"`, `"horizontal"`, and `"diagonal"`.
    as_polars: If True, return a Polars DataFrame; otherwise return \
        a Pandas DataFrame.
    rechunk: Whether Polars should rechunk the resulting DataFrame.
    --------
    Returns:
        The concatenated DataFrame in the requested dataframe type.
    -------
    Raises:
    ValueError: If `frames` is empty.
    TypeError: If an input is neither a Polars nor Pandas DataFrame.
    -------
    """
    pls = [to_polars(f) for f in frames]
    if not pls:
        raise ValueError("concat() received no frames")

    # Polars requires at least one frame; short-circuit single.
    output = pls[0] if len(pls) == 1 else pl.concat(pls, how=how, rechunk=rechunk)
    return output if as_polars else to_pandas(out)


def concatenate_lazy(
    lfs: Iterable[pl.LazyFrame], *, how: str = "vertical", rechunk: bool = False
) -> pl.LazyFrame:
    """
    Lazily concatenate multiple Polars LazyFrames. The LazyFrames are 
    combined into a single lazy query without collecting or completing 
    the underlying data.

    -----
    Args:
    lfs: Iterable of Polars LazyFrames to concatenate.
    how: Polars concatenation strategy. Common options include \
        `"vertical"`, `"horizontal"`, and `"diagonal"`.
    rechunk: Whether Polars should rechunk the resulting DataFrame \
        when the lazy query is collected.
    --------
    Returns:
        A combined Polars LazyFrame.
    -------
    Raises:
    ValueError: If `lfs` is empty.
    -------
    """
    items = list(lfs)
    if not items:
        raise ValueError("concat_lazy() received no frames")

    if len(items) == 1:
        return items[0]
    
    return pl.concat(items, how=how, rechunk=rechunk)
