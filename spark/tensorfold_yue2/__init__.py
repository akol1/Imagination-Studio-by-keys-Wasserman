"""YuE2's autoregressive decode step on TensorFold's CUDA engine (libtf_yue2.so, built by spark/build-yue2-engine.sh).

The YuE2 server keeps everything else in PyTorch: the prompt pass (it fills the KV cache exactly as the reference
does), sampling (the package's own `distribution`, top-k / top-p, window penalty and seeded multinomial), the acoustic
flow matching and the VAE. Each generated token's pass through the 28 layers, the final norm and the logits is one
call into the engine, which holds its own copy of the AR weights (int8 weight-only by default, quantized on the GPU
from the model's bf16 tensors at open) and replays a CUDA graph per cache binding on PyTorch's stream.

The seam is yue2.cuda_graph.GraphAR: `install(engine)` puts a subclass in its place for yue2.sampling.generate_tokens,
`uninstall()` puts the original back. Only the rows the phase can draw are computed (the score's text rows, or the
codec rows and the end token); the rest of the logits row stays 0, which the package's allowed-token mask removes.

Environment: TF_YUE2_LIB (the library; default $PREFIX/tf-yue2/libtf_yue2.so, else beside this file).
"""

import ctypes
import os
from pathlib import Path

ABI = 1
WEIGHTS = {"bf16": 0, "int8": 1, "int8-groups": 2}
DEFAULT_WEIGHTS = "int8"
DEFAULT_SPLITS = 32

# the package's protocol ids (yue2/protocol.py): the rows each phase can draw
EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768

# tf_yue2_set_weight's names: a layer's, then the model's (layer -1)
LAYER_WEIGHTS = {"input_layernorm": 0, "self_attn.q_proj": 1, "self_attn.k_proj": 2, "self_attn.v_proj": 3,
                 "self_attn.o_proj": 4, "self_attn.q_norm": 5, "self_attn.k_norm": 6,
                 "post_attention_layernorm": 7, "mlp.gate_proj": 8, "mlp.up_proj": 9, "mlp.down_proj": 10}
MODEL_WEIGHTS = {"embed": 11, "norm": 12, "lm_head": 13, "inv_freq": 14}


class Config(ctypes.Structure):
    _fields_ = [("hidden", ctypes.c_uint32), ("layers", ctypes.c_uint32), ("heads", ctypes.c_uint32),
                ("kv_heads", ctypes.c_uint32), ("head_dim", ctypes.c_uint32), ("mlp", ctypes.c_uint32),
                ("vocab", ctypes.c_uint32), ("weights", ctypes.c_uint32), ("eps", ctypes.c_float),
                ("splits", ctypes.c_uint32), ("graphs", ctypes.c_uint32)]


class Step(ctypes.Structure):
    _fields_ = [("tokens", ctypes.c_uint64), ("positions", ctypes.c_uint64), ("keys", ctypes.c_uint64),
                ("values", ctypes.c_uint64), ("logits", ctypes.c_uint64), ("stream", ctypes.c_uint64),
                ("branches", ctypes.c_uint32), ("capacity", ctypes.c_uint32), ("row0", ctypes.c_uint32),
                ("row1", ctypes.c_uint32), ("logits_stride", ctypes.c_uint32), ("layers", ctypes.c_uint32),
                ("hidden_out", ctypes.c_uint64)]


SIZES = (ctypes.sizeof(Config), ctypes.sizeof(Step))   # (44, 80), as the library's own test pins them

_library = None


def library_path():
    explicit = os.environ.get("TF_YUE2_LIB")
    if explicit:
        return Path(explicit)
    prefix = Path(os.environ.get("PREFIX", Path.home() / ".local/opt/tensorfold-studio"))
    installed = prefix / "tf-yue2" / "libtf_yue2.so"
    return installed if installed.exists() else Path(__file__).resolve().parent / "libtf_yue2.so"


def bind(lib):
    """Declares the C functions on a loaded library and checks its ABI; returns the library."""
    lib.tf_yue2_abi_version.restype = ctypes.c_uint32
    lib.tf_yue2_abi_version.argtypes = []
    version = lib.tf_yue2_abi_version()
    if version != ABI:
        raise RuntimeError(f"TensorFold YuE2: the library speaks ABI {version}, this seam {ABI}; rebuild it")
    lib.tf_yue2_struct_sizes.restype = None
    lib.tf_yue2_struct_sizes.argtypes = [ctypes.POINTER(ctypes.c_uint32 * 2)]
    sizes = (ctypes.c_uint32 * 2)()
    lib.tf_yue2_struct_sizes(ctypes.byref(sizes))
    if tuple(sizes) != SIZES:
        raise RuntimeError(f"TensorFold YuE2: the library's structs are {tuple(sizes)} bytes, this seam's {SIZES}")
    lib.tf_yue2_open.restype = ctypes.c_void_p
    lib.tf_yue2_open.argtypes = [ctypes.POINTER(Config)]
    lib.tf_yue2_close.restype = None
    lib.tf_yue2_close.argtypes = [ctypes.c_void_p]
    lib.tf_yue2_set_weight.restype = ctypes.c_int
    lib.tf_yue2_set_weight.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                       ctypes.c_uint32, ctypes.c_uint32]
    lib.tf_yue2_ready.restype = ctypes.c_int
    lib.tf_yue2_ready.argtypes = [ctypes.c_void_p]
    lib.tf_yue2_info.restype = None
    lib.tf_yue2_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64 * 4)]
    lib.tf_yue2_step.restype = ctypes.c_int
    lib.tf_yue2_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(Step)]
    return lib


def library():
    global _library
    if _library is None:
        path = library_path()
        if not path.exists():
            raise FileNotFoundError(f"TensorFold YuE2: no engine at {path} (spark/build-yue2-engine.sh builds it)")
        _library = bind(ctypes.CDLL(str(path)))
    return _library


def available():
    """None when the engine can be used here, else the reason it cannot (the server then decodes in PyTorch)."""
    try:
        import torch
    except ImportError:
        return "PyTorch is not installed"
    if not torch.cuda.is_available():
        return "no CUDA device"
    try:
        library()
    except (OSError, RuntimeError) as error:
        return str(error)
    return None


def config_for(config, weights=DEFAULT_WEIGHTS, splits=DEFAULT_SPLITS, graphs=True):
    """The engine's Config from a YuE2Config (or its config.json as a dict)."""
    get = config.get if isinstance(config, dict) else (lambda key, default=None: getattr(config, key, default))
    if weights not in WEIGHTS:
        raise ValueError(f"weights must be one of {sorted(WEIGHTS)}")
    if int(get("head_dim")) != 128:
        raise ValueError("the engine is built for head_dim 128")
    if int(get("num_attention_heads")) % int(get("num_key_value_heads")):
        raise ValueError("query heads must be a multiple of key/value heads")
    if not 1 <= int(splits) <= 256:
        raise ValueError("splits must be 1..256")
    return Config(hidden=int(get("hidden_size")), layers=int(get("num_hidden_layers")),
                  heads=int(get("num_attention_heads")), kv_heads=int(get("num_key_value_heads")),
                  head_dim=int(get("head_dim")), mlp=int(get("intermediate_size")), vocab=int(get("vocab_size")),
                  weights=WEIGHTS[weights], eps=float(get("rms_norm_eps")), splits=int(splits),
                  graphs=1 if graphs else 0)


def rows_for(prefix, vocab):
    """The logits rows a phase can draw, from its prefix's last token: [row0, row1)."""
    last = int(prefix[-1]) if len(prefix) else -1
    if last == MUSIC_START:                       # semantic: the end token then the codec rows, contiguous
        return MUSIC_END, min(vocab, CODEC_OFFSET + CODEC_SIZE)
    if last == ABC_START:                         # the score: text rows below EOD and ABC_END
        return 0, min(vocab, ABC_END + 1)
    return 0, vocab


class Engine:
    """The AR body's weights on the device in the engine's format, copied from a loaded YuE2ForCausalLM."""

    def __init__(self, model, weights=DEFAULT_WEIGHTS, splits=DEFAULT_SPLITS, graphs=True, lib=None):
        import torch
        self.torch = torch
        self.lib = lib or library()
        self.weights = weights
        self.config = model.config
        cfg = config_for(model.config, weights, splits, graphs)
        self.cfg = cfg
        embed = model.model.embed_tokens.weight
        if embed.device.type != "cuda" or embed.dtype != torch.bfloat16:
            raise ValueError("the engine copies a bf16 model already on the CUDA device")
        torch.cuda.synchronize(embed.device)
        handle = self.lib.tf_yue2_open(ctypes.byref(cfg))
        if not handle:
            raise RuntimeError("TensorFold YuE2: the engine did not open (its reason is on stderr)")
        self.handle = handle
        try:
            with torch.no_grad():
                self._set(-1, MODEL_WEIGHTS["embed"], embed)
                self._set(-1, MODEL_WEIGHTS["norm"], model.model.norm.weight)
                self._set(-1, MODEL_WEIGHTS["lm_head"], model.lm_head.weight)
                rotary = model.model.rotary_emb
                inv_freq = 1.0 / (rotary.base ** (torch.arange(0, rotary.head_dim, 2, dtype=torch.float32,
                                                               device=embed.device) / rotary.head_dim))
                self._set(-1, MODEL_WEIGHTS["inv_freq"], inv_freq)
                for index, layer in enumerate(model.model.layers):
                    for name, which in LAYER_WEIGHTS.items():
                        module = layer
                        for part in name.split("."):
                            module = getattr(module, part)
                        self._set(index, which, module.weight)
            if not self.lib.tf_yue2_ready(self.handle):
                raise RuntimeError("TensorFold YuE2: the engine is missing weights after loading")
        except BaseException:
            self.close()
            raise

    def _set(self, layer, which, tensor):
        torch = self.torch
        t = tensor.detach()
        if which != MODEL_WEIGHTS["inv_freq"] and t.dtype != torch.bfloat16:
            raise ValueError(f"weight {layer}/{which} is {t.dtype}, not bf16")
        t = t.contiguous()
        rows, cols = (1, t.shape[0]) if t.ndim == 1 else (t.shape[0], t.shape[1])
        torch.cuda.synchronize(t.device)
        if self.lib.tf_yue2_set_weight(self.handle, layer, which, t.data_ptr(), rows, cols):
            raise RuntimeError(f"TensorFold YuE2: the engine refused weight {layer}/{which} {tuple(t.shape)}")

    def info(self):
        out = (ctypes.c_uint64 * 4)()
        self.lib.tf_yue2_info(self.handle, ctypes.byref(out))
        return {"weight_bytes": int(out[0]), "steps": int(out[1]), "graphs": int(out[2]),
                "weights": {v: k for k, v in WEIGHTS.items()}[int(out[3])]}

    def step(self, step):
        if not self.handle:
            raise RuntimeError("TensorFold YuE2: the engine is closed")
        if self.lib.tf_yue2_step(self.handle, ctypes.byref(step)):
            raise RuntimeError("TensorFold YuE2: the decode step failed (its reason is on stderr)")

    def close(self):
        if getattr(self, "handle", None):
            self.torch.cuda.synchronize()
            self.lib.tf_yue2_close(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def graph_ar_class(engine, base=None):
    """A yue2.cuda_graph.GraphAR whose decode steps run on `engine`; the prompt pass stays the package's own."""
    if base is None:
        from yue2.cuda_graph import GraphAR as base
    torch = engine.torch

    class TensorFoldAR(base):
        def __init__(self, model, prefixes, max_tokens, *, capture=True, attention_backend="auto",
                     fuse_projections=False):
            super().__init__(model, prefixes, max_tokens, capture=capture, attention_backend=attention_backend,
                             fuse_projections=False)
            config = model.config
            if config is not engine.config and config.to_dict() != engine.config.to_dict():
                raise ValueError("TensorFold YuE2: this model is not the one the engine was loaded from")
            self.attention_backend = "tensorfold-" + engine.weights
            self.vocab = config.vocab_size
            self.row0, self.row1 = rows_for(self.prefixes[0], self.vocab)
            for prefix in self.prefixes[1:]:
                if rows_for(prefix, self.vocab) != (self.row0, self.row1):
                    self.row0, self.row1 = 0, self.vocab
            # rows outside the phase's are never written: 0 keeps a CFG mix of them finite for the mask to remove
            self.logits = torch.zeros((self.branches, self.vocab), dtype=self.dtype, device=self.device)
            layers = len(self.keys)
            self._key_table = (ctypes.c_uint64 * layers)(*[t.data_ptr() for t in self.keys])
            self._value_table = (ctypes.c_uint64 * layers)(*[t.data_ptr() for t in self.values])
            self.kv_bytes = 2 * layers * self.keys[0].numel() * self.keys[0].element_size()

        def step_inputs(self, layers=0, hidden_out=0):
            return Step(tokens=self.tokens.data_ptr(), positions=self.positions.data_ptr(),
                        keys=ctypes.addressof(self._key_table), values=ctypes.addressof(self._value_table),
                        logits=self.logits.data_ptr(), stream=torch.cuda.current_stream(self.device).cuda_stream,
                        branches=self.branches, capacity=self.capacity, row0=self.row0, row1=self.row1,
                        logits_stride=self.vocab, layers=layers, hidden_out=hidden_out)

        def _capture(self):
            # the engine captures its own graph on first use of this cache binding
            self._inputs = self.step_inputs()

        @torch.inference_mode()
        def step(self, token):
            from numbers import Integral
            if self.closed or not self.ready:
                raise RuntimeError("Call prefill before step and do not use a closed GraphAR")
            if self.steps >= self.max_tokens - 1:
                raise ValueError("Requested generation budget is exhausted")
            if isinstance(token, Integral) and not isinstance(token, bool):
                if not 0 <= int(token) < self.vocab:
                    raise ValueError("Token is outside the model vocabulary")
                self.tokens.fill_(int(token))
            elif isinstance(token, torch.Tensor) and token.numel() == 1 and token.dtype in {torch.int32, torch.int64}:
                if token.device.type == "cpu" and not 0 <= token.item() < self.vocab:
                    raise ValueError("Token is outside the model vocabulary")
                self.tokens.copy_(token.reshape(1, 1).expand(self.branches, 1))
            else:
                raise ValueError("step needs one shared integer token, including for CFG")
            inputs = getattr(self, "_inputs", None) or self.step_inputs()
            engine.step(inputs)
            self.steps += 1
            return self.logits

        def close(self):
            self._key_table = self._value_table = None
            self._inputs = None
            self.logits = None
            super().close()

    return TensorFoldAR


_installed = None


def install(engine):
    """yue2.sampling.generate_tokens decodes on `engine` from now on (its GraphAR is looked up at each call)."""
    global _installed
    import yue2.cuda_graph as cuda_graph
    original = _installed[0] if _installed else cuda_graph.GraphAR
    cuda_graph.GraphAR = graph_ar_class(engine, original)
    _installed = (original, engine)
    return engine


def uninstall():
    """Puts the package's own GraphAR back."""
    global _installed
    if _installed:
        import yue2.cuda_graph as cuda_graph
        cuda_graph.GraphAR = _installed[0]
        _installed = None


def installed():
    return _installed[1] if _installed else None


def enable(pipeline, weights=DEFAULT_WEIGHTS, splits=DEFAULT_SPLITS):
    """For a YuE2Pipeline (backend "torch"): load its model, open an engine from it and install it. Returns the
    engine, or None with the reason logged when the engine is not usable here (the pipeline then stays PyTorch)."""
    import logging
    reason = available()
    if reason:
        logging.info(f"TensorFold YuE2: decoding in PyTorch ({reason})")
        return None
    if getattr(pipeline, "backend", "torch") != "torch" or getattr(pipeline, "quantization", "none") != "none":
        logging.info("TensorFold YuE2: decoding in PyTorch (the engine replaces the torch backend's bf16 decode only)")
        return None
    engine = Engine(pipeline._load_model(), weights=weights, splits=splits)
    install(engine)
    info = engine.info()
    logging.info(f"TensorFold YuE2: decoding on the CUDA engine ({engine.weights}, {info['weight_bytes'] / 2**30:.2f} GiB)")
    return engine


QUANT = {"bf16": "bf16", "int8": "int8-groups"}   # the YuE2 server's --quant names: int8 a scale a group of 128


def prepare(model, quant="bf16", splits=DEFAULT_SPLITS):
    """spark/yue2_server.py's `--backend blocks` seam: open an engine from the loaded model and install it, so every
    score and song token runs on it. Returns what the server reports; when the engine cannot be used here the token
    loop stays the package's PyTorch one and the reason is said."""
    if quant not in QUANT:
        raise ValueError(f"quant must be one of {sorted(QUANT)}")
    reason = available()
    if reason:
        uninstall()
        return {"decode": "pytorch", "decode_reason": reason}
    previous = installed()
    if previous is not None:
        uninstall()
        previous.close()
    engine = install(Engine(model, weights=QUANT[quant], splits=splits))
    return {"decode": "tensorfold-cuda", "decode_weights": engine.weights,
            "decode_engine_gb": round(engine.info()["weight_bytes"] / 1e9, 2)}
