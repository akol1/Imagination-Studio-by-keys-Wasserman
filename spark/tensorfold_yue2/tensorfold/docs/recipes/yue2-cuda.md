# YuE2-3B's decode step on one CUDA GPU (DGX Spark GB10)

`libtf_yue2.so` (`zig build tf-yue2 -Dnvcc=...`) runs YuE2-3B's autoregressive decode step on CUDA for a host that
shares its CUDA context: the PyTorch YuE2 pipeline (`yue2-infer`, m-a-p/YuE2-3B) keeps the prompt pass, the KV cache,
sampling, the acoustic flow matching and the VAE, and calls the library once a generated token. It is a family laid
over TensorFold 1.1.5 + music3 (`music3-cuda-115`, 5505dbfa): the CUDA runtime, and music3's warp-a-row decode GEMVs
and int8 quantizers (`music3_ar.cuh`), with kernels of its own for YuE2's shape (GQA 16 / 8, head 128, per-head q / k
RMS norms, rotate-half rope at theta 1e6, split-key decode attention over the host's cache).

YuE2 weights are CC BY-NC 4.0 (m-a-p); none are part of this code.

## What runs where, and why the decode step

YuE2 is an AR-NAR mixture of transformers: one Qwen3-1.7B-shaped backbone (28 layers, width 2048, MLP 6144, vocab
184,704) with two sets of projections. A song is

1. the score (ABC): AR decode, temperature 0.7, top-k 30, up to 4,096 tokens;
2. the song's semantic tokens: AR decode at 25 a second of audio, up to 9,000;
3. the acoustic latents: 32 midpoint steps (64 full-sequence passes) of the NAR projections over the song's frames,
   attending to the AR prefix's keys (computed once);
4. the VAE (Oobleck, fp32) to 48 kHz stereo.

Measured in PyTorch on an idle GB10 (bf16, the package's own CUDA-graph decode with FlashAttention; table below),
the two AR phases are about half of a song's time and run at ~37 tokens a second, a quarter of the read ceiling: each
token reads 1.4 G params of body + the 378 M-row lm_head, 3.6 GB in bf16. That is the lever this family takes. The
NAR passes are prefill-shaped (2,700 rows x 28 layers a pass for a 108 s song) and the VAE is convolutions; both stay
in PyTorch here (see "Not done").

## The engine

| Part | How |
| --- | --- |
| weights | copied from the host's bf16 tensors at open (`tf_yue2_set_weight`): bf16 as they are, or int8 weight-only quantized on the GPU (a scale a row, or a scale a group of 128 inputs); q / k / v fused into one matrix, gate / up into one; embeddings and norms bf16 |
| a step | embed the branch's token, 28 x (RMS norm, qkv GEMV, q / k norm + rope + cache write, split-key attention, o GEMV + residual, RMS norm, gate / up GEMV writing SwiGLU, down GEMV + residual), final norm, lm_head GEMV over only the rows the phase can draw (the score's text rows, or MUSIC_END + the 32,768 codec rows), positions + 1 |
| batching | one branch, or a CFG pair in the same GEMVs (M = 2: the weights are read once for both) |
| cache | the host's own (`yue2.cuda_graph.GraphAR`'s per-layer [branches, capacity, 8, 128] bf16 tensors): the prompt pass is the package's, unchanged |
| graphs | one CUDA graph captured a cache binding (pointers, capacity, rows) on the engine's stream, replayed on the host's stream; positions are read on the device, so a replay needs no update |
| rounding | PyTorch's: RMS norm as x * rs.to(bf16) * w, rope cos / sin rounded to bf16 before each product, every product and sum rounded to bf16, GEMVs accumulate in fp32 and round once |

C ABI (`zig/src/families/yue2/cuda_lib.zig`, ABI 1): `tf_yue2_open(Config*)`, `tf_yue2_set_weight(h, layer, which,
ptr, rows, cols)`, `tf_yue2_ready`, `tf_yue2_step(h, Step*)` (queues one step on the host's stream), `tf_yue2_info`,
`tf_yue2_close`, `tf_yue2_struct_sizes` (44, 80). `Step.layers` / `Step.hidden_out` run the first N layers and copy
the residual stream out, for checks.

## Parity (teacher forcing, 256 decoded tokens a case)

The reference's own CUDA-graph decode draws every token with the package's sampling; the engine gets the same tokens
from the same prompt pass. "Floor" is the package's two PyTorch decodes against each other (its eager SDPA decode
against its CUDA-graph FlashAttention one): the noise of bf16 itself. Logits are compared on the rows the phase can
draw. TV is the total-variation distance of the phase's sampling distributions (after top-k / top-p / penalty).

| Case | Check | Floor (torch vs torch) | bf16 | int8, a scale a group of 128 | int8, a scale a row |
| --- | --- | --- | --- | --- | --- |
| score (ABC) | logits cos mean / min | 0.999982 / 0.99892 | 0.999981 / 0.99846 | 0.999949 / 0.99743 | 0.999593 / 0.98343 |
| | greedy agreement, top-50 overlap | 1.0, 0.9949 | 1.0, 0.9959 | 1.0, 0.9940 | 1.0, 0.9927 |
| | TV of the sampling distributions | 0.0031 | 0.0030 | 0.0037 | 0.0040 |
| | keys / values written, relative error (worst layer) | 0.0162 | 0.0162 | 0.0218 | 0.0249 |
| song (semantic) | logits cos mean / min | 0.999961 / 0.99979 | 0.999962 / 0.99989 | 0.999940 / 0.99972 | 0.999923 / 0.99976 |
| | greedy agreement, top-50 overlap | 0.980, 0.9885 | 0.977, 0.9867 | 0.977, 0.9863 | 0.969, 0.9845 |
| | TV of the sampling distributions | 0.0183 | 0.0180 | 0.0200 | 0.0213 |
| | keys / values, relative error | 0.0148 | 0.0148 | 0.0197 | 0.0225 |
| song, CFG pair | logits cos mean / min | 0.999957 / 0.99977 | 0.999959 / 0.99979 | 0.999940 / 0.99963 | 0.999926 / 0.99975 |
| | TV of the guided distributions | 0.0329 | 0.0323 | 0.0334 | 0.0351 |

bf16 is inside the floor on every check: it is as close to the reference as the reference is to itself. int8 with a
scale a group of 128 is a little outside it (the sampling distributions move 10% more than bf16 noise moves them);
int8 a scale a row is further, with one score step at cos 0.983. Group scales are the int8 default.

The residual stream after each layer for one token (engine `layers = k` against PyTorch eager) is 0.4% after layer 1
and 3% after layer 28 in bf16 (a single token: indicative only; the 256-step key / value errors above are the robust
per-layer measure).

Over a whole song (the reference's own 2,703 semantic tokens of a 108 s song forced into both decodes from a
1,103-token prompt, positions 1,103 to 3,806, `check.py longparity`), nothing drifts with context. In each window of
256 positions:

| | logits cos | greedy agreement | TV of the sampling distributions | mean NLL of the forced tokens, reference decode / engine |
| --- | --- | --- | --- | --- |
| bf16 | 0.99995 to 0.99996 | 0.96 to 0.99 | 0.0176 to 0.0197 | within 0.007 of the reference in every window (e.g. 3.4474 / 3.4457, 2.8054 / 2.8073) |
| int8, groups of 128 | 0.99992 to 0.99994 | 0.96 to 0.99 | 0.0200 to 0.0219 | within 0.01 (3.4474 / 3.4433, 2.8054 / 2.8149) |

Songs drawn on each decode, scored by the reference model (`check.py nll`, one song each, seed 7): score NLL 0.153
(PyTorch) / 0.145 (bf16 engine) / 0.142 (int8 engine); semantic NLL 2.79 / 3.16 / 3.22. The semantic gap is between
different songs (one draw each: how repetitive a song turns out moves its NLL by more than this), not the decode:
forced onto the same tokens the two decodes score them within 0.01, above.

## Speed and memory

GB10, PyTorch 2.14.1 + CUDA 13, timings taken only while no other job held the GPU (the check scripts record the
machine's other reservations and CUDA processes before and after each run; runs beside another job are not quoted).

Decode steps alone (one branch, no sampling, 200 steps):

| Context | PyTorch bf16 (GraphAR, FlashAttention) | engine int8 (groups of 128) | speedup |
| --- | --- | --- | --- |
| 1,000 | 24.5 ms (40.9 / s) | 7.7 ms (130 / s) | 3.2x |
| 4,000 | 26.0 ms (38.5 / s) | 9.5 ms (105 / s) | 2.7x |
| 12,000 | 29.9 ms (33.4 / s) | 13.9 ms (72 / s) | 2.15x |

A step at 1,000 positions reads ~1.6 GB (int8 body 1.41 GB + group scales, the phase's lm_head rows, the cache): 7.7
ms is ~210 GB/s, against a ~230 GB/s read ceiling. Past a few thousand positions the cache read grows (115 KB a
position) and the attention kernel is the next lever.

Whole generations with the package's sampling (same prompt, same seed):

| | PyTorch bf16 | engine int8 (groups) |
| --- | --- | --- |
| 1,500 semantic tokens from a 575-token prompt | 38.0 s (39.5 tokens/s) | 12.6 s (119.5 tokens/s), 3.0x |
| a 108 s / 104 s song: score (929 / 874 tokens) | 25.1 s (36.9 / s) | 8.5 s (103 / s) |
| the same song: semantic (2,704 / 2,600 tokens) | 72.1 s (37.5 / s) | 23.5 s (110.5 / s) |
| the same song: flow matching (unchanged code) | 16.3 s | 15.2 s |
| the same song: AR stages together | 97.3 s | 32.0 s, 3.0x |

The songs differ after the first tokens (int8 logits are not the reference's, so the seeded draws part ways: the
first 66 semantic tokens were the same in the 1,500-token run); the engine itself is deterministic (two runs of the
same request drew the same 874 and 2,600 tokens). Memory: the engine holds 2.42 GiB (int8 body and lm_head with group
scales, bf16 embeddings) beside PyTorch's 7.9 GiB peak; bf16 weights hold 4.03 GiB.

### Through the Studio's YuE2 server

`spark/yue2_server.py` end to end, the model loaded, the same English indie-pop request (style and lyrics) at
seeds 7, 11 and 21, the GPU to itself. "PyTorch int8" is the server's shipped token loop (int8 weight-only copies of
the projections, a Triton matrix-vector kernel, in YuE2's CUDA-graph GraphAR); "engine" is `--backend blocks` on this
library (int8, groups of 128). Times are the server's own stage stamps; the score's length is the model's draw.

| Song | Seed | PyTorch int8: score / song / flow / VAE / total | engine: score / song / flow / VAE / total |
| --- | --- | --- | --- |
| 60 s | 7 | 17.5 / 18.6 / 8.2 / 1.4 / 45.7 s (1,452 score tokens) | 11.9 / 13.5 / 8.0 / 1.4 / 35.0 s (1,236) |
| 60 s | 11 | 19.0 / 18.9 / 8.2 / 1.2 / 47.3 s (1,618) | 15.0 / 13.9 / 8.3 / 1.1 / 38.4 s (1,611) |
| 60 s | 21 | 19.5 / 18.7 / 8.2 / 1.2 / 47.6 s (1,653) | 16.8 / 14.0 / 8.3 / 1.1 / 40.4 s (1,788) |
| 120 s | 7 | 16.8 / 38.1 / 18.7 / 2.3 / 76.0 s | 11.4 / 28.3 / 18.4 / 2.3 / 60.4 s |
| 120 s | 11 | 19.1 / 38.4 / 18.6 / 2.3 / 78.4 s | 15.1 / 28.9 / 18.5 / 2.2 / 64.8 s |
| 120 s | 21 | 19.4 / 38.4 / 18.6 / 2.3 / 78.8 s | 16.8 / 29.2 / 18.6 / 2.3 / 66.9 s |

Tokens a second: score 83 to 86 (PyTorch int8) against 104 to 108 (engine), song 78 to 81 against 103 to 111. A
60 s song is 19% faster end to end, a 120 s song 18%; the server is ready in 6.0 s instead of 21.1 s (the engine
quantizes on the GPU at load; the Triton loop tunes its kernels). Whisper (faster-whisper large-v3-turbo, the
Studio's lyric check) heard, per song in the order above: PyTorch int8 68, 97, 97 / 79, 87, 99 %; engine 97, 94,
97 / 97, 97, 97 %. The flow matching and the VAE are the same code in both (8.2 s and 1.1 to 1.4 s for 60 s).

## Not done

- The NAR flow matching (64 passes of ~2,700 rows) and the VAE stay PyTorch. Profiled: a NAR pass is 400 ms, of which
  the bf16 GEMMs run at ~48 TFLOP/s (sm_80 CUTLASS and nvjet kernels, half the GB10's fp16/bf16 rate), FlashAttention
  19%, and elementwise glue (fp32 norms, rope's cat, the per-layer cat of the AR keys) ~37%. A native pass (cuBLASLt
  GEMMs as music3's DiT, a GQA head-128 flash kernel, fused norms / rope) would be ~2.5x on this stage.
- GPUs other than GB10 (sm_121): the build refuses unless YUE2_ANY_GPU=1.
- Programmatic dependent launch for the GEMVs (`TF_YUE2_PDL=1`, as music3 does) is built in but off: it was built,
  not measured, so the numbers above are without it.
- Several seeds and listening tests: the parity above is per token; songs were checked by the reference's likelihood,
  one song per decode.
- vLLM's backend and the package's fp8 path are not touched; the engine replaces the torch backend's bf16 decode.
