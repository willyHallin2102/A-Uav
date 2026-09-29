"""
    data / formats.py
    -----------------
    Format dispatch utilities collection for tabular dataset I/O. This
    module defines the canonical dataset formats supported by the data
    loading layer and provides a mapping needed to direct between file
    suffixes, Polars readers, and Polars writers. The module icontains
    no schema, filesystem, or dataset-level logic. Its purpose is only
    to provide a small, centralized format registry for high-level can
    determine the appropriate reader or writer without individual formats.

    Supported canonical formats are

        - ``csv``
        - ``parquet``
        - ``ipc``
        - ``json``
        - ``ndjson``
    
    Multiple file extensions may map tp the same canonical format. For 
    example, `.pq`, maps to `parquet` and `.arrow`/`.feather` map to `ipc`.
"""
from __future__ import annotations
from typing import Any, Callable, Literal

import polars as pl


Format = Literal["csv", "parquet", "ipc", "json", "ndjson"]

EXT_TO_FMT: dict[str, Format] = {
    ".csv"      : "csv",
    ".parquet"  : "parquet",
    ".pq"       : "parquet",
    ".ipc"      : "ipc",
    ".arrow"    : "ipc",
    ".feather"  : "ipc",
    ".json"     : "json",
    ".ndjson"   : "ndjson",
    ".jsonl"    : "ndjson",
}

# Canonical format names -> lazy reader factory (path -> LazyFrame)
READERS: dict[Format, Callable[..., pl.LazyFrame]] = {
    "csv"       : pl.scan_csv,
    "parquet"   : pl.scan_parquet,
    "ipc"       : pl.scan_ipc,
    "json"      : pl.scan_ndjson,   # json lines only; differ in loader
    "ndjson"    : pl.scan_ndjson,
}

# Canonical format names -> writers (DataFrame -> path)
WRITERS: dict[Format, Callable[..., None]] = {
    "csv"       : lambda df, p, **kw: df.write_csv(p, **kw),
    "parquet"   : lambda df, p, **kw: df.write_parquet(p, **kw),
    "ipc"       : lambda df, p, **kw: df.write_ipc(p, **kw),
    "json"      : lambda df, p, **kw: df.write_json(p, **kw),
    "ndjson"    : lambda df, p, **kw: df.write_ndjson(p, **kw),
}


# Formats that do not accept a `compression = ` kwargs in Polars.
NO_COMPRESSION: set[Format] = {"json", "ndjson"}

def fmt_from_suffix(suffix: str) -> Format:
    """
    Resolve a file suffix to its canonical dataset format. Suffix matching 
    is case-insensitive and supports all aliases registered in ``EXT_TO_FMT``.
    
    -----
    Args:
    suffix: File suffix, including the leading dot, such as ``".parquet"``.
    --------
    Returns:
        The canonical format name associated with the suffix.
    -------
    Raises:
    ValueError: If the suffix is not recognized.
    """
    suffix = suffix.lower()
    try:
        return EXT_TO_FMT[suffix]
    
    except KeyError as ke:
        raise ValueError(f"Cannot infer format from suffix: {suffix!r}") from ke


def reader_for(suffix: str) -> Callable[..., pl.LazyFrame]:
    """
    Return the Polars lazy reader associated with a file suffix. The 
    suffix is first resolved to its canonical format using 
    ``fmt_from_suffix()``, then mapped to the corresponding reader in 
    ``READERS``.

    -----
    Args:
    suffix: File suffix, including the leading dot.
    --------
    Returns:
        A Polars lazy reader callable, such as ``pl.scan_parquet``.
    -------
    Raises:
    ValueError: If the suffix does not correspond to a supported format.
    """
    return READERS[fmt_from_suffix(suffix)]


def writer_for(suffix: str) -> Callable[..., None]:
    """
    Return the Polars writer associated with a canonical format.
    
    -----
    Args:
    fmt: Canonical dataset format.
    --------
    Returns:
        A writer callable for the requested format.
    -------
    Raises:
    KeyError: If ``fmt`` is not registered in ``WRITERS``.
    """
    return WRITERS[fmt_from_suffix(suffix)]


def filter_kwargs(fmt: Format, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Remove writer arguments that are unsupported by a dataset format. The
    input dictionary is copied before filtering, so the caller's dictionary 
    is not modified.

    -----
    Args:
    fmt: Canonical dataset format.
    kwargs: Writer keyword arguments to filter.
    --------
    Returns:
        A filtered copy of ``kwargs`` containing only arguments that are \
            applicable to the specified format.
    """
    output = dict(kwargs)
    if fmt in NO_COMPRESSION:
        output.pop("compression", None)
    
    return output
