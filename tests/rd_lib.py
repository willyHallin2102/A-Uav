#!/usr/bin/env python3

import sys
import os
import time
import tempfile
import traceback


# ============================================================
# Helpers
# ============================================================

PASSED, FAILED, SKIPPED = 0, 0, 0

def section(name):
    print(f"\n{name}")

def test(name, f):
    global PASSED, FAILED, SKIPPED
    print(f"\n[TEST] {name}")

    try:
        result = f()
        if result is False:
            SKIPPED += 1
            print(f"[SKIP] {name}")
        
        else:
            PASSED += 1
            print(f"[PASS] {name}")

    except Exception as exc:
        FAILED += 1
        print(f"[FAIL] {name}")
        print(f"       {type(exc).__name__}: {exc}")

        traceback.print_exc()


# ============================================================
# Environment
# ============================================================

section("ENVIRONMENT")

print(f"Python : {sys.version}")
print(f"Executable : {sys.executable}")
print(f"Platform : {sys.platform}")
print(f"Working directory : {os.getcwd()}")

# ============================================================
# Package imports
# ============================================================

section("PACKAGE IMPORTS")

def import_packages():
    import numpy
    import scipy
    import pyarrow
    import polars
    import numba
    import matplotlib
    import seaborn
    import orjson
    import einops
    import h5py
    import tensorboard
    import tensorflow
    import sionna

    packages = {
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "polars": polars.__version__,
        "numba": numba.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__,
        "orjson": orjson.__version__,
        "einops": einops.__version__,
        "h5py": h5py.__version__,
        "tensorflow": tensorflow.__version__,
    }

    for name, version in packages.items():
        print(f"  {name:15} {version}")

    print(f"  {'sionna':15} imported successfully")

test("Import all packages", import_packages)


# ============================================================
# NumPy
# ============================================================

section("NUMPY")

def test_numpy():
    import numpy as np

    a = np.arange(1_000_000, dtype=np.float32)
    b = np.sqrt(a)

    assert b.shape == a.shape
    assert np.isfinite(b).all()

    print(f"Array shape : {a.shape}")
    print(f"Dtype       : {a.dtype}")
    print(f"Checksum    : {b.sum():.4f}")

test("NumPy computation", test_numpy)

# ============================================================
# SciPy
# ============================================================

section("SCIPY")

def test_scipy():
    import numpy as np
    from scipy import linalg

    matrix = np.array([
        [4.0, 1.0],
        [1.0, 3.0],
    ])

    eigenvalues = linalg.eigvalsh(matrix)

    assert len(eigenvalues) == 2
    assert np.all(eigenvalues > 0)

    print(f"Eigenvalues : {eigenvalues}")

test("SciPy linear algebra", test_scipy)

# ============================================================
# Polars
# ============================================================

section("POLARS")

def test_polars():
    import polars as pl

    df = pl.DataFrame({
        "id": range(1_000_000),
        "value": [float(x % 100) for x in range(1_000_000)],
    })

    result = (
        df
        .filter(pl.col("value") > 50)
        .group_by((pl.col("id") % 10).alias("group"))
        .agg(
            pl.col("value").mean().alias("mean"),
            pl.len().alias("count"),
        )
        .sort("group")
    )

    assert result.height == 10

    print(result)

test("Polars DataFrame operations", test_polars)

# ============================================================
# PyArrow
# ============================================================

section("PYARROW")

def test_pyarrow():
    import pyarrow as pa

    array = pa.array(range(100_000))
    table = pa.table({"numbers": array,})

    assert table.num_rows == 100_000

    print(f"Rows : {table.num_rows}")
    print(f"Schema : {table.schema}")


test("PyArrow table creation", test_pyarrow)

# ============================================================
# Numba
# ============================================================

section("NUMBA")

def test_numba():
    import numpy as np
    from numba import njit

    @njit
    def calculate(values):
        total = 0.0

        for value in values:
            total += value * value

        return total

    values = np.arange(1_000_000, dtype=np.float64)
    result = calculate(values)
    expected = np.sum(values * values)

    assert np.isclose(result, expected)

    print(f"Result : {result:.4e}")

test("Numba JIT compilation and execution", test_numba)

# ============================================================
# orjson
# ============================================================

section("ORJSON")

def test_orjson():
    import orjson
    data = {
        "name": "GPU test",
        "values": list(range(10_000)),
        "enabled": True,
    }

    encoded = orjson.dumps(data)
    decoded = orjson.loads(encoded)

    assert decoded["name"] == "GPU test"
    assert len(decoded["values"]) == 10_000

    print(f"Serialized size : {len(encoded):,} bytes")

test("JSON serialization", test_orjson)

# ============================================================
# einops
# ============================================================

section("EINOPS")

def test_einops():
    import numpy as np
    from einops import rearrange

    x = np.random.rand(2, 3, 4, 5)
    y = rearrange(
        x, "batch channels height width -> batch height width channels",
    )

    assert y.shape == (2, 4, 5, 3)

    print(f"Original : {x.shape}")
    print(f"Rearranged : {y.shape}")

test("Einops tensor rearrangement", test_einops)

# ============================================================
# h5py
# ============================================================

section("H5PY")

def test_h5py():
    import numpy as np
    import h5py

    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        data = np.arange(100_000, dtype=np.float32)

        with h5py.File(f.name, "w") as h5:
            h5.create_dataset("data", data=data, compression="gzip",)

        with h5py.File(f.name, "r") as h5:
            loaded = h5["data"][:]

        assert np.array_equal(data, loaded)

        print(f"Dataset shape : {loaded.shape}")
        print(f"Dataset dtype : {loaded.dtype}")

test("HDF5 write/read", test_h5py)

# ============================================================
# Matplotlib
# ============================================================

section("MATPLOTLIB")

def test_matplotlib():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    figure = plt.figure()
    axes = figure.add_subplot(111)

    axes.plot([1, 2, 3], [1, 4, 9])
    axes.set_title("Environment Test")

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        figure.savefig(f.name)

        assert os.path.getsize(f.name) > 0

    plt.close(figure)

test("Matplotlib rendering", test_matplotlib)

# ============================================================
# Seaborn
# ============================================================

section("SEABORN")

def test_seaborn():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import seaborn as sns

    figure = plt.figure()
    sns.lineplot(x=[1, 2, 3, 4], y=[1, 4, 9, 16],)

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        figure.savefig(f.name)

        assert os.path.getsize(f.name) > 0

    plt.close(figure)

test("Seaborn plotting", test_seaborn)

# ============================================================
# TensorFlow
# ============================================================

section("TENSORFLOW")

def test_tensorflow():
    import tensorflow as tf

    print(f"TensorFlow version : {tf.__version__}")
    print("\nBuild information:")

    print(tf.sysconfig.get_build_info())
    print("\nPhysical devices:")

    for device in tf.config.list_physical_devices():
        print(f"  {device}")

    print("\nGPU devices:")

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        print(f"  {gpu}")

    if not gpus:
        raise RuntimeError("TensorFlow cannot see an NVIDIA GPU.")

test("TensorFlow installation and GPU detection", test_tensorflow)

# ============================================================
# TensorFlow CPU computation
# ============================================================

section("TENSORFLOW CPU")


def test_tensorflow_cpu():
    import tensorflow as tf

    with tf.device("/CPU:0"):
        a = tf.random.normal((1000, 1000))
        b = tf.random.normal((1000, 1000))
        c = tf.matmul(a, b)

    result = float(tf.reduce_mean(c).numpy())

    assert tf.math.is_finite(result)

    print(f"Result : {result}")

test("TensorFlow CPU computation", test_tensorflow_cpu)

# ============================================================
# TensorFlow GPU computation
# ============================================================

section("TENSORFLOW GPU")

def test_tensorflow_gpu():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        raise RuntimeError("No GPU detected by TensorFlow.")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True,)

        except RuntimeError:
            pass

    with tf.device("/GPU:0"):
        a = tf.random.normal((2048, 2048), dtype=tf.float32,)
        b = tf.random.normal((2048, 2048), dtype=tf.float32,)

        start = time.perf_counter()
        c = tf.matmul(a, b)

        result = float(tf.reduce_mean(c).numpy())
        elapsed = time.perf_counter() - start

    assert tf.math.is_finite(result)

    print(f"GPU : {gpus[0].name}")
    print(f"Matrix : 2048 x 2048")
    print(f"Result : {result:.6f}")
    print(f"Time : {elapsed:.4f} seconds")

test("Actual TensorFlow GPU matrix multiplication", test_tensorflow_gpu)

# ============================================================
# TensorFlow neural network
# ============================================================

section("TENSORFLOW NEURAL NETWORK")

def test_tensorflow_model():
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10),
    ])

    model.compile(
        optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
    )

    x = tf.random.normal((512, 32))
    y = tf.random.uniform((512,), minval=0, maxval=10, dtype=tf.int32,)

    with tf.device("/GPU:0"):
        history = model.fit(x, y, epochs=2, batch_size=32, verbose=0,)

    loss = history.history["loss"][-1]

    assert os.path.exists("/dev/null")
    assert loss > 0

    print(f"Final loss : {loss:.6f}")

test("TensorFlow GPU neural network training", test_tensorflow_model)

# ============================================================
# TensorFlow tf.data
# ============================================================

section("TENSORFLOW DATA PIPELINE")

def test_tfdata():
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices((
        tf.random.normal((1000, 32)),
        tf.random.uniform((1000,), minval=0, maxval=10, dtype=tf.int32,),
    ))

    dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    batches = 0

    for x, y in dataset:
        assert x.shape[1] == 32
        assert y.shape[0] == x.shape[0]
        batches += 1

    assert batches > 0

    print(f"Batches processed : {batches}")

test("TensorFlow tf.data pipeline", test_tfdata)

# ============================================================
# Sionna
# ============================================================

section("SIONNA")

def test_sionna():
    import sionna

    print(f"Sionna version : {getattr(sionna, '__version__', 'unknown')}")

test("Sionna import", test_sionna)

# ============================================================
# Sionna + TensorFlow
# ============================================================

section("SIONNA + TENSORFLOW")

def test_sionna_tensorflow():
    import tensorflow as tf
    import sionna

    x = tf.constant([[1.0, 2.0, 3.0, 4.0]], dtype=tf.float32,)

    with tf.device("/GPU:0"):
        y = tf.math.square(x)

    result = y.numpy()
    expected = [[1.0, 4.0, 9.0, 16.0]]

    assert (result == expected).all()

    print("Sionna imported successfully.")
    print("TensorFlow GPU tensor operation successful.")

test("Sionna/TensorFlow compatibility", test_sionna_tensorflow)

# ============================================================
# Summary
# ============================================================

section("TEST SUMMARY")
section("------------")

print(f"Passed : {PASSED}")
print(f"Failed : {FAILED}")
print(f"skipped: {SKIPPED}\n")

if FAILED:
    print("RESULT: Failed")
    sys.exit(1)

print("RESULT: All tests have passed")
