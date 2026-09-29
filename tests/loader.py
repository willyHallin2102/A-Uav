"""
    test / loader.py
    ----------------
    ···
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import polars as pl
import pandas as pd
import data.iterators as it

from argparse import Namespace
from data.loader import Loader
from data.formats import fmt_from_suffix, reader_for, writer_for, filter_kwargs
from tools.timer import Timer

from logs.logger import Logger, Level, Profile


CITIES = ["beijing","boston","london","moscow","tokyo"]

# ==================================================
#       Additional Helper Functions
# ==================================================

TEST_DIRNAME = "test"

def get_dataset(city: str, fmt: str = "csv") -> str:
    return f"uav_{city}/train.{fmt}"


def describe(df: pl.DataFrame | pd.DataFrame, label: str = "frame") -> None:
    """
    Print a compact description of a Polars or Pandas DataFrame so the 
    loaded data can be inspected during tests.
    """
    print(f"\n---------- {label} ----------")
    if isinstance(df, pl.DataFrame):
        print(f"type       : polars.DataFrame")
        print(f"shape      : {df.shape}")
        print(f"columns    : {df.columns}")
        print(f"dtypes     : {dict(df.schema)}")
        print(df.head(3))
    elif isinstance(df, pd.DataFrame):
        print(f"type       : pandas.DataFrame")
        print(f"shape      : {df.shape}")
        print(f"columns    : {list(df.columns)}")
        print(f"dtypes     : {dict(df.dtypes)}")
        print(df.head(3))
    else:
        print(f"unknown type: {type(df)!r}")


# ==================================================
#       Test Functions for Loader Capabilities
# ==================================================

def test_loading_single_performance(args: Namespace):
    """
    """
    path = get_dataset(args.load)
    loader = Loader()

    with Timer(f"Loading {args.load}", show=True) as t:
        data = loader.load(path)


def test_loading_multiple_performance(args: Namespace):
    """
    """
    loader = Loader()
    patterns = [get_dataset(city,"csv") for city in CITIES]
    print(patterns)

    with Timer(f"load_many([{patterns}]): ", show=True):
        df = loader.load_many(patterns=patterns, kind="polars")
    
    describe(df, label="load_many")

    for city in CITIES:
        n = len(loader.list(patterns))
        print(f"[INFO] uav_{city} file(s)")


def test_loading_inspect_pandas(args: Namespace):
    """
    Load a single dataset as pandas and inspect its shape, columns,
    and dtypes. This also confirms the loader's default `kind="pandas"`
    behavior.
    """
    path = get_dataset(args.load)
    loader = Loader()

    df = loader.load(path, kind="pandas")

    assert isinstance(df, pd.DataFrame)
    describe(df, label=f"pandas :: {path}")

    expected_columns = {
        "dvec", "rx_type", "link_state",
        "los_pl", "los_ang", "los_dly",
        "nlos_pl", "nlos_ang", "nlos_dly",
    }

    assert expected_columns.issubset(set(df.columns)), (
        f"Missing expected columns: {expected_columns - set(df.columns)}"
    )
    assert len(df) > 0


def test_loading_inspect_polars(args: Namespace):
    """
    Load a single dataset as polars and inspect its shape, columns,
    and dtypes. This also confirms the loader's default `kind="polars"`
    behavior.
    """
    path = get_dataset(args.load)
    loader = Loader()

    df = loader.load(path, kind="polars")

    assert isinstance(df, pl.DataFrame)
    describe(df, label=f"polars :: {path}")

    assert "dvec" in df.columns
    assert "link_state" in df.columns
    assert df.height > 0


def test_loading_column_projection(args: Namespace):
    """
    Load a subset of columns and verify projection is  conserved.
    """
    path = get_dataset(args.load)
    loader = Loader()

    columns = ["rx_type", "link_state", "los_pl"]
    df = loader.load(path, columns=columns, kind="polars")

    assert isinstance(df, pl.DataFrame)
    assert df.columns == columns, f"Expected {columns}, got {df.columns}"

    describe(df, label="projection")


def test_loading_n_rows_and_sample(args: Namespace):
    """
    Exercise the `n_rows` and `sample` knobs on `Loader.load`.
    """
    path = get_dataset(args.load)
    loader = Loader()

    head = loader.load(path, n_rows=5, kind="polars")
    assert head.height <= 5

    sampled = loader.load(path, sample=0.5, seed=42, kind="polars")
    full = loader.load(path, kind="polars")
    
    assert sampled.height <= full.height

    with pytest.raises(ValueError):
        loader.load(path, sample=0.0)
    
    with pytest.raises(ValueError):
        loader.load(path, sample=1.5)


def test_lazy_plan_is_not_completed(args: Namespace):
    """
    A LazyFrame should be returned without touching the data, and only
    complete once `.collect()` is called.
    """
    path = get_dataset(args.load)
    loader = Loader()

    lf = loader.lazy(path, columns=["rx_type", "link_state"])
    assert isinstance(lf, pl.LazyFrame)

    out = lf.collect()

    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["rx_type", "link_state"]


def test_load_iter_batching(args: Namespace):
    """
    Confirm `load_iter` yields consecutive batches of at most `size`
    rows, in both Polars and Pandas forms.
    """
    path = get_dataset(args.load)
    loader = Loader()

    batches_pl = list(loader.load_iter(path, size=4, kind="polars"))
    assert all(isinstance(b, pl.DataFrame) for b in batches_pl)
    assert all(b.height <= 4 for b in batches_pl)

    total = sum(b.height for b in batches_pl)
    full = loader.load(path, kind="polars")
    assert total == full.height

    batches_pd = list(loader.load_iter(path, size=4, kind="pandas"))
    assert all(isinstance(b, pd.DataFrame) for b in batches_pd)


def test_iterators_directly(args: Namespace):
    """
    Exercise `data.iterators` helpers on a loaded frame.
    """
    path = get_dataset(args.load)
    loader = Loader()
    df = loader.load(path, kind="polars")
    assert isinstance(df, pl.DataFrame)

    pdf = it.to_pandas(df)
    assert isinstance(pdf, pd.DataFrame)
    
    back = it.to_polars(pdf)
    assert isinstance(back, pl.DataFrame)
    assert back.columns == df.columns
    assert back.height == df.height

    sizes = [c.height for c in it.iter_slices(df, size=3, as_polars=True)]
    assert sum(sizes) == df.height
    assert all(s <= 3 for s in sizes)

    chunks = list(it.iter_slices(df, size=3, as_polars=True))
    rebuilt = it.concatenate(chunks, as_polars=True)
    assert rebuilt.height == df.height

    lazy_rebuilt = it.concatenate_lazy([c.lazy() for c in chunks])
    
    assert isinstance(lazy_rebuilt, pl.LazyFrame)
    assert lazy_rebuilt.collect().height == df.height

    with pytest.raises(ValueError):
        it.concatenate([])
    
    with pytest.raises(ValueError):
        it.concatenate_lazy([])



def test_format_dispatch(args: Namespace):
    """
    Verify suffix -> format -> reader/writer resolution and the
    compression filtering helper.
    """
    assert fmt_from_suffix(".CSV") == "csv"
    assert fmt_from_suffix(".pq") == "parquet"
    assert fmt_from_suffix(".arrow") == "ipc"
    assert fmt_from_suffix(".jsonl") == "ndjson"

    assert reader_for(".csv") is pl.scan_csv
    assert reader_for(".parquet") is pl.scan_parquet
    assert reader_for(".arrow") is pl.scan_ipc

    assert callable(writer_for(".csv"))
    assert callable(writer_for(".parquet"))

    filtered = filter_kwargs("json", {"compression": "zstd", "foo": 1})
    assert "compression" not in filtered
    assert filtered["foo"] == 1

    filtered = filter_kwargs("parquet", {"compression": "zstd"})
    assert filtered["compression"] == "zstd"

    with pytest.raises(ValueError):
        fmt_from_suffix(".nope")


def test_save_round_trip(tmp_path: Path, args: Namespace):
    """
    Save a loaded frame to parquet and csv, then load it back and verify
    shape and column parity. Uses a unique subdirectory under the
    dataset root so it doesn't clobber user data.
    """
    loader = Loader()
    path = get_dataset(args.load)

    df = loader.load(path, kind="polars")
    assert isinstance(df, pl.DataFrame)

    subdir = f"_test_tmp/{args.load}"
    written_pq = loader.save(df, f"{subdir}/roundtrip.parquet")
    assert written_pq.exists()

    written_csv = loader.save(
        df, f"{subdir}/roundtrip.csv", compression=None,
    )
    assert written_csv.exists()

    reloaded_pq = loader.load(f"{subdir}/roundtrip.parquet", kind="polars")
    reloaded_csv = loader.load(f"{subdir}/roundtrip.csv", kind="polars")

    assert reloaded_pq.height == df.height
    assert set(reloaded_pq.columns) == set(df.columns)

    assert reloaded_csv.height == df.height
    assert set(reloaded_csv.columns) == set(df.columns)



def test_save_many(tmp_path: Path, args: Namespace):
    """
    Split a loaded frame into parts and verify `save_many` produces
    numbered files that can be re-loaded and concatenated.
    """
    loader = Loader()
    path = get_dataset(args.load)
    df = loader.load(path, kind="polars")

    chunks = list(it.iter_slices(df, size=3, as_polars=True))
    subdir = f"_test_tmp/{args.load}_parts"

    written = loader.save_many(chunks, subdir, basename="part", fmt="parquet")
    assert len(written) == len(chunks)
    assert all(p.exists() for p in written)
    assert all(p.suffix == ".parquet" for p in written)

    pattern = f"{subdir}/part-*.parquet"
    lf = loader.lazy_many(pattern)
    combined = lf.collect()
    assert combined.height == df.height


def test_exists_and_list(args: Namespace):
    """
    Confirm `exists` and `list` behave for known/unknown paths and
    glob patterns.
    """
    loader = Loader()
    path = get_dataset(args.load)

    assert loader.exists(path) is True
    assert loader.exists("does/not/exist.csv") is False

    matched = loader.list(f"uav_{args.load}/*")
    assert any(p.name.startswith("train") for p in matched)


# ==================================================
#       Main Runner
# ==================================================

from tests._utils import runner, CommandSpec, build_cli

COMMON = [
    {"flags": ["--verbose", "-v"], "kwargs": {"action": "store_true"}},
    {"flags": ["--load", "-l"], "kwargs": {
        "type": str, "choices": ["beijing", "boston", "london", "moscow", "tokyo"],
        "default": "beijing"
    }},
]


@runner
def main():
    parser = build_cli([
        CommandSpec(
            "time_single", "Test level enum values and ordering",
            test_loading_single_performance, [*COMMON]
        ),
        CommandSpec(
            "time_multiple", "Test level enum values and ordering",
            test_loading_multiple_performance, [*COMMON]
        ),
        CommandSpec(
            "inspect_pandas", "Load as Pandas and inspect schema",
            test_loading_inspect_pandas, [*COMMON]
        ),
        CommandSpec(
            "inspect_polars", "Load as Polars and inspect schema",
            test_loading_inspect_polars, [*COMMON]
        ),
        CommandSpec(
            "projection", "Load a column subset",
            test_loading_column_projection, [*COMMON]
        ),
        CommandSpec(
            "sample", "Exercise n_rows and sample controls",
            test_loading_n_rows_and_sample, [*COMMON]
        ),
        CommandSpec(
            "lazy", "Confirm lazy frames are not materialized",
            test_lazy_plan_is_not_completed, [*COMMON]
        ),
        CommandSpec(
            "iterate", "Exercise load_iter batching",
            test_load_iter_batching, [*COMMON]
        ),
        CommandSpec(
            "iterators", "Exercise data.iterators helpers",
            test_iterators_directly, [*COMMON]
        ),
        CommandSpec(
            "formats", "Exercise format dispatch utilities",
            test_format_dispatch, [*COMMON]
        ),
        CommandSpec(
            "save", "Round-trip save/load for parquet and csv",
            test_save_round_trip, [*COMMON]
        ),
        CommandSpec(
            "save_many", "Split and persist via save_many",
            test_save_many, [*COMMON]
        ),
        CommandSpec(
            "exists", "Check exists / list helpers",
            test_exists_and_list, [*COMMON]
        ),
    ])

    args = parser.parse_args()
    args._handler(args)


if __name__ == "__main__":
    main()
