"""spark/tensorfold_yue2 (YuE2's decode step on TensorFold's CUDA engine) without a GPU: the C structs the seam
mirrors against the library's own layout test, the engine config read from YuE2's config.json, the logits rows a
phase can draw, the ABI checks that refuse a mismatched library, the PyTorch fallback when no engine is usable, the
GraphAR seam's install and uninstall, and spark/build-yue2-engine.sh's decisions with stand-ins on PATH."""
from __future__ import annotations

import ctypes
import hashlib
import importlib
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spark"))
tf = importlib.import_module("tensorfold_yue2")
SCRIPT = ROOT / "spark" / "build-yue2-engine.sh"
FAMILY = ROOT / "spark" / "tensorfold_yue2" / "tensorfold"

# YuE2-3B's config.json (m-a-p/YuE2-3B), the fields the engine reads
CONFIG = {"hidden_size": 2048, "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
          "head_dim": 128, "intermediate_size": 6144, "vocab_size": 184704, "rms_norm_eps": 1e-06,
          "rope_theta": 1000000}


def test_the_structs_match_the_library_layout():
    # the numbers zig/src/families/yue2/cuda_lib.zig's test pins
    assert tf.SIZES == (44, 80)
    assert tf.Step.branches.offset == 48 and tf.Step.hidden_out.offset == 72 and tf.Config.eps.offset == 32
    lib = (FAMILY / "zig/src/families/yue2/cuda_lib.zig").read_text()
    assert "[2]u32{ 44, 80 }" in lib and '@offsetOf(engine.Step, "hidden_out")' in lib
    engine = (FAMILY / "zig/src/families/yue2/engine.zig").read_text()
    # the field order of both structs, as the Zig declares them
    for struct in (tf.Config, tf.Step):
        body = engine.split(f"pub const {struct.__name__} = extern struct {{", 1)[1].split("};", 1)[0]
        names = [line.split(":")[0].strip() for line in body.splitlines() if ":" in line and not line.strip().startswith("//")]
        assert names == [name for name, _ in struct._fields_]


def test_the_weight_names_match_the_engine():
    engine = (FAMILY / "zig/src/families/yue2/engine.zig").read_text()
    which = engine.split("pub const Which = struct {", 1)[1].split("};", 1)[0]
    numbers = {line.split("=")[0].split()[-1]: int(line.split("=")[1].strip(" ;")) for line in which.splitlines() if "pub const" in line}
    assert numbers["count"] == 15
    mine = {**{k.split(".")[-1]: v for k, v in tf.LAYER_WEIGHTS.items()}, **tf.MODEL_WEIGHTS}
    names = {"input_layernorm": "input_norm", "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
             "post_attention_layernorm": "post_norm", "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
    for name, value in mine.items():
        assert numbers[names.get(name, name)] == value, name


def test_config_from_yue2_config_json():
    c = tf.config_for(CONFIG, "int8-groups")
    assert (c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim, c.mlp, c.vocab) == (2048, 28, 16, 8, 128, 6144, 184704)
    assert c.weights == 2 and c.splits == tf.DEFAULT_SPLITS and c.graphs == 1 and abs(c.eps - 1e-6) < 1e-12
    assert tf.config_for(CONFIG, "bf16", graphs=False).graphs == 0
    with pytest.raises(ValueError):
        tf.config_for(CONFIG, "fp8")
    with pytest.raises(ValueError):
        tf.config_for({**CONFIG, "head_dim": 64})
    with pytest.raises(ValueError):
        tf.config_for({**CONFIG, "num_key_value_heads": 3})
    with pytest.raises(ValueError):
        tf.config_for(CONFIG, splits=0)
    # an object with the same attributes (a YuE2Config) reads the same
    assert tf.config_for(types.SimpleNamespace(**CONFIG)).vocab == 184704


def test_the_rows_a_phase_can_draw():
    vocab = CONFIG["vocab_size"]
    # semantic: MUSIC_END then the 32768 codec rows, contiguous
    assert tf.rows_for([151643, 5, tf.MUSIC_START], vocab) == (tf.MUSIC_END, tf.CODEC_OFFSET + tf.CODEC_SIZE)
    assert tf.MUSIC_END + 1 == tf.CODEC_OFFSET
    # the score: every text row below EOD and ABC_END
    row0, row1 = tf.rows_for([151643, 5, tf.ABC_START], vocab)
    assert row0 == 0 and row1 == tf.ABC_END + 1 and tf.EOD < row1
    # anything else: all of them
    assert tf.rows_for([151643, 5, 6], vocab) == (0, vocab) and tf.rows_for([], vocab) == (0, vocab)
    # the package's own numbers, when it is installed here
    protocol = pytest.importorskip("yue2.protocol")
    assert (protocol.EOD, protocol.ABC_START, protocol.ABC_END, protocol.MUSIC_START, protocol.MUSIC_END,
            protocol.CODEC_OFFSET, protocol.CODEC_SIZE) == (tf.EOD, tf.ABC_START, tf.ABC_END, tf.MUSIC_START,
                                                           tf.MUSIC_END, tf.CODEC_OFFSET, tf.CODEC_SIZE)


class FakeFn:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


class FakeLib:
    def __init__(self, version=1, sizes=(44, 80)):
        def write_sizes(ref):
            ref._obj[0], ref._obj[1] = sizes
        self.tf_yue2_abi_version = FakeFn(lambda: version)
        self.tf_yue2_struct_sizes = FakeFn(write_sizes)
        for name in ("tf_yue2_open", "tf_yue2_close", "tf_yue2_set_weight", "tf_yue2_ready", "tf_yue2_info", "tf_yue2_step"):
            setattr(self, name, FakeFn(lambda *a: 0))


def test_a_library_of_another_abi_or_layout_is_refused():
    assert tf.bind(FakeLib()).tf_yue2_open.argtypes == [ctypes.POINTER(tf.Config)]
    with pytest.raises(RuntimeError, match="ABI 2"):
        tf.bind(FakeLib(version=2))
    with pytest.raises(RuntimeError, match="structs"):
        tf.bind(FakeLib(sizes=(44, 72)))


def test_no_engine_means_pytorch(tmp_path, monkeypatch):
    monkeypatch.setenv("TF_YUE2_LIB", str(tmp_path / "missing.so"))
    monkeypatch.setattr(tf, "_library", None)
    assert tf.library_path() == tmp_path / "missing.so"
    with pytest.raises(FileNotFoundError, match="build-yue2-engine.sh"):
        tf.library()
    reason = tf.available()
    assert isinstance(reason, str) and reason
    # the installed engine under PREFIX is the default
    monkeypatch.delenv("TF_YUE2_LIB")
    monkeypatch.setenv("PREFIX", str(tmp_path))
    (tmp_path / "tf-yue2").mkdir()
    (tmp_path / "tf-yue2" / "libtf_yue2.so").write_bytes(b"")
    assert tf.library_path() == tmp_path / "tf-yue2" / "libtf_yue2.so"


def test_enable_stays_pytorch_without_an_engine(monkeypatch):
    monkeypatch.setattr(tf, "available", lambda: "no CUDA device")
    pipe = types.SimpleNamespace(backend="torch", quantization="none", _load_model=lambda: pytest.fail("loaded"))
    assert tf.enable(pipe) is None


def test_the_servers_blocks_seam_falls_back_to_pytorch(monkeypatch):
    """spark/yue2_server.py --backend blocks calls prepare(model, quant=...): without an engine it says why and the
    token loop stays PyTorch's; its quant names map to the engine's weights (int8 a scale a group of 128)."""
    monkeypatch.setattr(tf, "available", lambda: "no CUDA device")
    assert tf.prepare(object(), quant="int8") == {"decode": "pytorch", "decode_reason": "no CUDA device"}
    with pytest.raises(ValueError):
        tf.prepare(object(), quant="fp8")
    assert tf.QUANT == {"bf16": "bf16", "int8": "int8-groups"} and set(tf.QUANT.values()) <= set(tf.WEIGHTS)


def test_install_swaps_the_package_graph_and_uninstall_restores_it(monkeypatch):
    class GraphAR:
        def __init__(self, *a, **k):
            pass
    package = types.ModuleType("yue2")
    module = types.ModuleType("yue2.cuda_graph")
    module.GraphAR = GraphAR
    package.cuda_graph = module
    monkeypatch.setitem(sys.modules, "yue2", package)
    monkeypatch.setitem(sys.modules, "yue2.cuda_graph", module)
    engine = types.SimpleNamespace(torch=types.SimpleNamespace(inference_mode=lambda: (lambda f: f)), weights="int8")
    monkeypatch.setattr(tf, "_installed", None)
    tf.install(engine)
    assert module.GraphAR is not GraphAR and issubclass(module.GraphAR, GraphAR) and tf.installed() is engine
    # a second install builds on the package's own class, not on the first install's
    tf.install(engine)
    assert module.GraphAR.__mro__[1] is GraphAR
    tf.uninstall()
    assert module.GraphAR is GraphAR and tf.installed() is None
    tf.uninstall()   # twice is harmless
    assert module.GraphAR is GraphAR


# ---- spark/build-yue2-engine.sh ----

def stub(folder: Path, name: str, body: str) -> None:
    path = folder / name
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


@pytest.fixture
def box(tmp_path):
    """Stand-ins: a GB10, an aarch64 machine, an nvcc that says 13.0, no network; a TensorFold checkout to lay the
    family over (TF_YUE2_SOURCE) and a Zig that writes the library instead of building it."""
    bin_dir, prefix, src = tmp_path / "bin", tmp_path / "prefix", tmp_path / "tensorfold"
    bin_dir.mkdir()
    stub(bin_dir, "nvidia-smi", 'case "$*" in *compute_cap*) echo "${FAKE_CC:-12.1}";; *) echo "${FAKE_GPU:-NVIDIA GB10}";; esac\n')
    stub(bin_dir, "uname", '[ "$1" = -m ] && { echo aarch64; exit 0; }; exec /usr/bin/uname "$@"\n')
    stub(bin_dir, "nvcc", 'echo "Cuda compilation tools, release 13.0, V13.0.88"\n')
    real = shutil.which("sha256sum")
    stub(bin_dir, "sha256sum", f'exec "{real}" "$@"\n' if real else 'exec shasum -a 256 "$@"\n')
    stub(bin_dir, "git", "exit 1\n")
    stub(bin_dir, "curl", "exit 1\n")
    zig = prefix / "zig-0.17.0"
    zig.mkdir(parents=True)
    stub(zig, "zig", f'echo "$*" >> "{tmp_path}/zig.log"\ncase "$*" in *tf-yue2*) mkdir -p zig-out/lib; echo built > zig-out/lib/libtf_yue2.so;; esac\n')
    (src / "zig/build").mkdir(parents=True)
    (src / "zig/build/cuda.zig").write_text("pub fn targets() void {\n"
                                            '    @import("h3_host.zig").library(b, target, optimize, h3_image);\n'
                                            "    const cuda = 1;\n}\n")
    env = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin", "PREFIX": str(prefix), "HOME": str(tmp_path),
           "NVCC": str(bin_dir / "nvcc")}
    for name in ("TF_YUE2_SOURCE", "TF_SM", "YUE2_ANY_GPU", "STUDIO_GPU"):
        env.pop(name, None)
    return {"tmp": tmp_path, "prefix": prefix, "src": src, "env": env}


def run(box, **extra) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT)], env={**box["env"], **extra}, capture_output=True, text=True, timeout=60)


def test_it_refuses_another_gpu(box):
    done = run(box, FAKE_GPU="NVIDIA H100 80GB HBM3", TF_YUE2_SOURCE=str(box["src"]))
    assert done.returncode == 1 and "measured on a GB10" in done.stderr and "YUE2_ANY_GPU=1" in done.stderr
    assert not (box["prefix"] / "tf-yue2").exists()


def test_it_lays_the_family_over_the_checkout_and_hooks_the_step_once(box):
    done = run(box, TF_YUE2_SOURCE=str(box["src"]))
    assert done.returncode == 0, done.stderr
    src = box["src"]
    for f in FAMILY.rglob("*"):
        if f.is_file():
            assert (src / f.relative_to(FAMILY)).read_bytes() == f.read_bytes()
    cuda = (src / "zig/build/cuda.zig").read_text()
    hook = '    @import("yue2_lib.zig").library(b, target, optimize, nvcc, sms);'
    lines = cuda.splitlines()
    assert lines.count(hook) == 1 and lines[lines.index(hook) - 1].strip().startswith('@import("h3_host.zig").library')
    log = (box["tmp"] / "zig.log").read_text()
    assert "build tf-yue2" in log and "-Dsm=121" in log and "test-yue2" in log
    out = box["prefix"] / "tf-yue2"
    assert (out / "libtf_yue2.so").read_text().strip() == "built"
    assert (out / "stamp").read_text().split()[0] == "5505dbfaf38efce55e564062886ad74bae6f21e5"
    # again: the hook is not added twice
    assert run(box, TF_YUE2_SOURCE=str(box["src"])).returncode == 0
    assert (src / "zig/build/cuda.zig").read_text().splitlines().count(hook) == 1


def test_a_build_at_the_stamp_is_kept_and_an_edit_rebuilds(box, tmp_path):
    out = box["prefix"] / "tf-yue2"
    out.mkdir(parents=True)
    (out / "libtf_yue2.so").write_text("old")
    files = sorted(p for p in FAMILY.rglob("*") if p.is_file())
    listing = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(FAMILY)}\n"
                      for p in sorted(files, key=lambda p: str(p.relative_to(FAMILY)).encode()))
    stamp = "5505dbfaf38efce55e564062886ad74bae6f21e5 " + hashlib.sha256(listing.encode()).hexdigest()[:16]
    (out / "stamp").write_text(stamp + "\n")
    done = run(box)
    assert done.returncode == 0 and "already built at 5505dbfa" in done.stdout
    assert (out / "libtf_yue2.so").read_text() == "old"
    (out / "stamp").write_text("5505dbfaf38efce55e564062886ad74bae6f21e5 0000000000000000\n")
    # a stale stamp goes on to fetch the source, which the stand-in git refuses here
    assert "already built" not in run(box).stdout


def test_a_checkout_without_the_anchor_is_refused(box):
    (box["src"] / "zig/build/cuda.zig").write_text("pub fn targets() void {}\n")
    done = run(box, TF_YUE2_SOURCE=str(box["src"]))
    assert done.returncode == 1 and "no H3 library line" in done.stderr
    assert not (box["prefix"] / "tf-yue2" / "libtf_yue2.so").exists()


def test_the_kernels_use_music3s_gemvs_and_pytorchs_rounding():
    cu = (FAMILY / "zig/kernels/cuda/yue2.cu").read_text()
    assert '#include "music3_ar.cuh"' in cu
    # RMS norm as PyTorch's bf16 RMSNorm: rs rounded, then x * rs rounded, then * w
    assert "rb(rsqrtf(ss / (float)dim + eps))" in cu and "f2b(rb(b2f(xr[i]) * rs) * b2f(w[i]))" in cu
    # rope's cos / sin rounded to bf16 before the products, each product rounded
    assert "rb(cosf(ang)), s = rb(sinf(ang))" in cu
