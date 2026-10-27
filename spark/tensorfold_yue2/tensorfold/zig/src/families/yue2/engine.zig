//! YuE2-3B's autoregressive decode step on CUDA for a host that shares its CUDA context (the Studio's PyTorch server).
//! The host keeps the prompt pass, the KV cache, sampling, the acoustic flow matching and the VAE; this engine holds
//! its own copy of the AR body's weights (bf16, or int8 weight-only a scale a row / a group of 128 inputs, quantized
//! on the GPU from the host's bf16 tensors) and runs one token a branch through the 28 layers into the host's cache
//! and logits, captured once as a CUDA graph a cache binding and replayed on the host's stream.
const std = @import("std");
const cuda = @import("cuda");

const image align(16) = @embedFile("fatbin_yue2").*;

pub const head_dim = 128;
pub const max_branches = 2;
pub const max_layers = 64;
const graph_slots = 4;

pub const Format = enum(u32) { bf16 = 0, int8_rows = 1, int8_groups = 2 };

/// The model's shape and how its weights are held; the host fills it from config.json.
pub const Config = extern struct {
    hidden: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    mlp: u32,
    vocab: u32,
    /// Format: 0 bf16, 1 int8 a scale a row, 2 int8 a scale a group of 128 inputs
    weights: u32,
    eps: f32,
    /// key splits of the decode attention (1..256)
    splits: u32,
    /// 0: run every step eagerly (no CUDA graphs)
    graphs: u32,
};

/// One decode step for `branches` rows (one request, or a CFG pair) sharing the token.
pub const Step = extern struct {
    /// int64 [branches] device: each branch's token
    tokens: u64,
    /// int64 [branches] device: each branch's position (its cache slot); advanced by one after the step
    positions: u64,
    /// host address of u64 [layers]: each layer's key cache, bf16 [branches, capacity, kv_heads, head_dim] (device)
    keys: u64,
    /// host address of u64 [layers]: the value caches, the same layout
    values: u64,
    /// bf16 [branches, logits_stride] device: rows row0..row1 are written, the others are left as they are
    logits: u64,
    /// the host's CUDA stream (0: the legacy default stream)
    stream: u64,
    branches: u32,
    capacity: u32,
    row0: u32,
    row1: u32,
    logits_stride: u32,
    /// layers to run from the first (0: all); fewer than all skips the final norm and the logits (a check)
    layers: u32,
    /// when not 0: bf16 [branches, hidden] device that receives the residual stream after the run layers (a check)
    hidden_out: u64,
};

/// Weight names for set_weight: a layer's (0..10) and the model's (11..14, layer -1).
pub const Which = struct {
    pub const input_norm = 0;
    pub const q = 1;
    pub const k = 2;
    pub const v = 3;
    pub const o = 4;
    pub const q_norm = 5;
    pub const k_norm = 6;
    pub const post_norm = 7;
    pub const gate = 8;
    pub const up = 9;
    pub const down = 10;
    pub const embed = 11;
    pub const norm = 12;
    pub const lm_head = 13;
    pub const inv_freq = 14;
    pub const count = 15;
};

const kernel_names = [_][:0]const u8{
    "y2_embed",      "y2_rmsnorm",     "y2_qk_rope",     "y2_attn_part",   "y2_attn_merge",  "y2_advance",
    "m3_quant_rows", "m3_quant_groups", "y2_gemv_bf16_1", "y2_gemv_bf16_2", "y2_gemv_i8_1",   "y2_gemv_i8_2",
    "y2_gemv_g8_1",  "y2_gemv_g8_2",   "y2_glu_bf16_1",  "y2_glu_bf16_2",  "y2_glu_i8_1",    "y2_glu_i8_2",
    "y2_glu_g8_1",   "y2_glu_g8_2",
};

fn kernelIndex(comptime name: []const u8) usize {
    inline for (kernel_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return i;
    @compileError("no kernel " ++ name);
}

/// A projection [n, k]: bf16 rows, or int8 rows with fp32 scales (a row, or a group of 128 inputs).
pub const Mat = struct {
    w: cuda.DeviceBuffer,
    sc: ?cuda.DeviceBuffer = null,
    n: usize,
    k: usize,

    fn free(m: *Mat) void {
        m.w.free();
        if (m.sc) |*s| s.free();
    }
};

pub const Layer = struct {
    norms: cuda.DeviceBuffer, // input, post (hidden each), q, k (head_dim each), bf16
    qkv: Mat, // q | k | v rows
    o: Mat,
    gate_up: Mat, // gate | up rows
    down: Mat,
    set: u16 = 0, // a bit a weight name that was set
};

const Bound = struct {
    step: Step,
    keys: [max_layers]u64,
    values: [max_layers]u64,
    exec: cuda.graph.Exec,
    used: u64,
};

pub fn seconds() f64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return 0;
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    module: cuda.Module,
    functions: [kernel_names.len]cuda.Function,
    capture: cuda.Stream, // the engine's own stream: graphs are captured here, then launched on the host's
    cfg: Config,
    format: Format,
    layers: []Layer,
    embed: cuda.DeviceBuffer,
    norm: cuda.DeviceBuffer,
    lm_head: Mat,
    inv_freq: cuda.DeviceBuffer,
    model_set: u16 = 0,
    // a step's scratch, sized for max_branches
    x: cuda.DeviceBuffer,
    xn: cuda.DeviceBuffer,
    qkv: cuda.DeviceBuffer,
    q: cuda.DeviceBuffer,
    att: cuda.DeviceBuffer,
    gu: cuda.DeviceBuffer,
    ml: cuda.DeviceBuffer,
    acc: cuda.DeviceBuffer,
    graphs: [graph_slots]?Bound = @splat(null),
    clock: u64 = 0,
    weight_bytes: usize = 0,
    steps: u64 = 0,
    captures: u64 = 0,
    /// GEMVs with programmatic dependent launch (sm_90 up, TF_YUE2_PDL=1): their first weight loads overlap the
    /// kernel before. Off by default: the measured parity and speed are without it.
    pdl: bool = false,

    pub fn qkvRows(self: *const Model) usize {
        return (self.cfg.heads + 2 * self.cfg.kv_heads) * head_dim;
    }

    fn validConfig(c: *const Config) bool {
        if (c.head_dim != head_dim) return false;
        if (c.hidden == 0 or c.hidden % 128 != 0 or c.mlp == 0 or c.mlp % 128 != 0) return false;
        if (c.layers == 0 or c.layers > max_layers or c.heads == 0 or c.kv_heads == 0) return false;
        if (c.heads % c.kv_heads != 0 or c.heads / c.kv_heads > 32) return false;
        if ((c.heads * head_dim) % 128 != 0 or c.vocab == 0) return false;
        if (c.weights > 2 or c.splits == 0 or c.splits > 256) return false;
        return std.math.isFinite(c.eps) and c.eps > 0;
    }

    pub fn open(gpa: std.mem.Allocator, c: *const Config) !*Model {
        if (!validConfig(c)) return error.BadConfig;
        const self = try gpa.create(Model);
        errdefer gpa.destroy(self);
        self.* = undefined;
        self.gpa = gpa;
        self.cfg = c.*;
        self.format = @enumFromInt(c.weights);
        self.model_set = 0;
        self.graphs = @splat(null);
        self.clock = 0;
        self.weight_bytes = 0;
        self.steps = 0;
        self.captures = 0;
        self.driver = try cuda.Driver.open();
        errdefer self.driver.close();
        self.ctx = try cuda.Context.shared(&self.driver);
        errdefer self.ctx.deinit();
        const d = &self.driver;
        self.module = try cuda.Module.load(d, &image);
        errdefer self.module.unload();
        inline for (kernel_names, 0..) |name, i| self.functions[i] = try self.module.function(name);
        self.capture = try cuda.Stream.init(d, true);
        errdefer self.capture.deinit();
        self.pdl = (try self.ctx.capability()) >= 90 and std.c.getenv("TF_YUE2_PDL") != null;
        var made: std.ArrayList(*cuda.DeviceBuffer) = .empty;
        defer made.deinit(gpa);
        errdefer for (made.items) |b| b.free();
        const H: usize = c.hidden;
        const qkv_rows = (c.heads + 2 * c.kv_heads) * head_dim;
        const inner: usize = c.heads * head_dim;
        self.layers = try gpa.alloc(Layer, c.layers);
        errdefer gpa.free(self.layers);
        var mats: std.ArrayList(*Mat) = .empty;
        defer mats.deinit(gpa);
        errdefer for (mats.items) |m| m.free();
        for (self.layers) |*l| {
            l.set = 0;
            l.norms = try cuda.DeviceBuffer.alloc(d, (2 * H + 2 * head_dim) * 2);
            try made.append(gpa, &l.norms);
            l.qkv = try self.mat(qkv_rows, H);
            try mats.append(gpa, &l.qkv);
            l.o = try self.mat(H, inner);
            try mats.append(gpa, &l.o);
            l.gate_up = try self.mat(2 * @as(usize, c.mlp), H);
            try mats.append(gpa, &l.gate_up);
            l.down = try self.mat(H, c.mlp);
            try mats.append(gpa, &l.down);
        }
        self.embed = try cuda.DeviceBuffer.alloc(d, @as(usize, c.vocab) * H * 2);
        try made.append(gpa, &self.embed);
        self.norm = try cuda.DeviceBuffer.alloc(d, H * 2);
        try made.append(gpa, &self.norm);
        self.inv_freq = try cuda.DeviceBuffer.alloc(d, head_dim / 2 * 4);
        try made.append(gpa, &self.inv_freq);
        self.lm_head = try self.mat(c.vocab, H);
        try mats.append(gpa, &self.lm_head);
        const B = max_branches;
        const S: usize = c.splits;
        self.x = try cuda.DeviceBuffer.alloc(d, B * H * 2);
        try made.append(gpa, &self.x);
        self.xn = try cuda.DeviceBuffer.alloc(d, B * H * 2);
        try made.append(gpa, &self.xn);
        self.qkv = try cuda.DeviceBuffer.alloc(d, B * qkv_rows * 2);
        try made.append(gpa, &self.qkv);
        self.q = try cuda.DeviceBuffer.alloc(d, B * inner * 2);
        try made.append(gpa, &self.q);
        self.att = try cuda.DeviceBuffer.alloc(d, B * inner * 2);
        try made.append(gpa, &self.att);
        self.gu = try cuda.DeviceBuffer.alloc(d, B * @as(usize, c.mlp) * 2);
        try made.append(gpa, &self.gu);
        self.ml = try cuda.DeviceBuffer.alloc(d, B * @as(usize, c.heads) * S * 2 * 4);
        try made.append(gpa, &self.ml);
        self.acc = try cuda.DeviceBuffer.alloc(d, B * @as(usize, c.heads) * S * head_dim * 4);
        try made.append(gpa, &self.acc);
        for (mats.items) |m| self.weight_bytes += m.w.len + (if (m.sc) |s| s.len else 0);
        self.weight_bytes += self.embed.len;
        return self;
    }

    fn mat(self: *Model, n: usize, k: usize) !Mat {
        const d = &self.driver;
        switch (self.format) {
            .bf16 => return .{ .w = try cuda.DeviceBuffer.alloc(d, n * k * 2), .n = n, .k = k },
            .int8_rows, .int8_groups => {
                var w = try cuda.DeviceBuffer.alloc(d, n * k);
                errdefer w.free();
                const scales = if (self.format == .int8_rows) n else n * (k / 128);
                return .{ .w = w, .sc = try cuda.DeviceBuffer.alloc(d, scales * 4), .n = n, .k = k };
            },
        }
    }

    pub fn close(self: *Model) void {
        self.ctx.makeCurrent() catch {};
        self.driver.synchronize() catch {};
        for (&self.graphs) |*g| if (g.*) |*b| {
            b.exec.deinit();
            g.* = null;
        };
        for (self.layers) |*l| {
            l.norms.free();
            l.qkv.free();
            l.o.free();
            l.gate_up.free();
            l.down.free();
        }
        self.gpa.free(self.layers);
        self.lm_head.free();
        for ([_]*cuda.DeviceBuffer{ &self.embed, &self.norm, &self.inv_freq, &self.x, &self.xn, &self.qkv, &self.q, &self.att, &self.gu, &self.ml, &self.acc }) |b| b.free();
        self.capture.deinit();
        self.module.unload();
        self.ctx.deinit();
        self.driver.close();
        self.gpa.destroy(self);
    }

    fn function(self: *const Model, comptime name: []const u8) cuda.Function {
        return self.functions[kernelIndex(name)];
    }

    fn go(_: *const Model, f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: usize, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block) } }, s, args);
    }

    /// Every weight the step reads has been set.
    pub fn ready(self: *const Model) bool {
        const layer_all: u16 = (1 << 11) - 1;
        for (self.layers) |l| if (l.set != layer_all) return false;
        const model_all: u16 = (1 << Which.embed) | (1 << Which.norm) | (1 << Which.lm_head) | (1 << Which.inv_freq);
        return self.model_set == model_all;
    }

    /// One of the host's tensors (bf16 [rows, cols] on the device; inv_freq fp32 [cols]) into the engine's copy.
    pub fn setWeight(self: *Model, layer: i32, which: u32, src: u64, rows: usize, cols: usize) !void {
        if (src == 0) return error.BadAddress;
        try self.ctx.makeCurrent();
        const c = &self.cfg;
        const H: usize = c.hidden;
        const s = self.capture;
        if (layer < 0) {
            switch (which) {
                Which.embed => {
                    if (rows != c.vocab or cols != H) return error.WrongShape;
                    try self.embed.copyFrom(0, src, rows * cols * 2, s.handle);
                },
                Which.norm => {
                    if (rows != 1 or cols != H) return error.WrongShape;
                    try self.norm.copyFrom(0, src, H * 2, s.handle);
                },
                Which.lm_head => {
                    if (rows != c.vocab or cols != H) return error.WrongShape;
                    try self.put(&self.lm_head, 0, src, rows);
                },
                Which.inv_freq => {
                    if (rows != 1 or cols != head_dim / 2) return error.WrongShape;
                    try self.inv_freq.copyFrom(0, src, cols * 4, s.handle);
                },
                else => return error.BadWhich,
            }
            try s.synchronize();
            self.model_set |= @as(u16, 1) << @intCast(which);
            return;
        }
        if (layer >= self.layers.len or which >= Which.embed) return error.BadWhich;
        const l = &self.layers[@intCast(layer)];
        const inner: usize = c.heads * head_dim;
        const kvw: usize = c.kv_heads * head_dim;
        switch (which) {
            Which.input_norm, Which.post_norm => {
                if (rows != 1 or cols != H) return error.WrongShape;
                try l.norms.copyFrom(if (which == Which.input_norm) 0 else H * 2, src, H * 2, s.handle);
            },
            Which.q_norm, Which.k_norm => {
                if (rows != 1 or cols != head_dim) return error.WrongShape;
                try l.norms.copyFrom(2 * H * 2 + (if (which == Which.q_norm) @as(usize, 0) else head_dim * 2), src, head_dim * 2, s.handle);
            },
            Which.q => {
                if (rows != inner or cols != H) return error.WrongShape;
                try self.put(&l.qkv, 0, src, rows);
            },
            Which.k, Which.v => {
                if (rows != kvw or cols != H) return error.WrongShape;
                try self.put(&l.qkv, if (which == Which.k) inner else inner + kvw, src, rows);
            },
            Which.o => {
                if (rows != H or cols != inner) return error.WrongShape;
                try self.put(&l.o, 0, src, rows);
            },
            Which.gate, Which.up => {
                if (rows != c.mlp or cols != H) return error.WrongShape;
                try self.put(&l.gate_up, if (which == Which.gate) 0 else c.mlp, src, rows);
            },
            Which.down => {
                if (rows != H or cols != c.mlp) return error.WrongShape;
                try self.put(&l.down, 0, src, rows);
            },
            else => return error.BadWhich,
        }
        try s.synchronize();
        l.set |= @as(u16, 1) << @intCast(which);
    }

    /// bf16 rows [rows, m.k] at `src` into rows row0.. of `m`: copied, or quantized to int8 on the GPU.
    fn put(self: *Model, m: *Mat, row0: usize, src: u64, rows: usize) !void {
        const s = self.capture;
        if (row0 + rows > m.n) return error.WrongShape;
        switch (self.format) {
            .bf16 => try m.w.copyFrom(row0 * m.k * 2, src, rows * m.k * 2, s.handle),
            .int8_rows, .int8_groups => {
                const groups = self.format == .int8_groups;
                var a: cuda.Args = .{};
                a.add(src);
                a.add(@as(c_int, @intCast(m.k)));
                a.add(m.w.ptr + row0 * m.k);
                a.add(m.sc.?.ptr + row0 * (if (groups) m.k / 128 else 1) * 4);
                const f = if (groups) self.function("m3_quant_groups") else self.function("m3_quant_rows");
                try self.go(f, s, .{ rows, 1, 1 }, 256, &a);
            },
        }
    }

    fn gemvKernel(self: *const Model, rows: usize, glu: bool) cuda.Function {
        const two = rows == 2;
        return switch (self.format) {
            .bf16 => if (glu) (if (two) self.function("y2_glu_bf16_2") else self.function("y2_glu_bf16_1")) else (if (two) self.function("y2_gemv_bf16_2") else self.function("y2_gemv_bf16_1")),
            .int8_rows => if (glu) (if (two) self.function("y2_glu_i8_2") else self.function("y2_glu_i8_1")) else (if (two) self.function("y2_gemv_i8_2") else self.function("y2_gemv_i8_1")),
            .int8_groups => if (glu) (if (two) self.function("y2_glu_g8_2") else self.function("y2_glu_g8_1")) else (if (two) self.function("y2_gemv_g8_2") else self.function("y2_gemv_g8_1")),
        };
    }

    const Gemv = struct { row0: usize = 0, n: ?usize = null, glu: bool = false, y: u64, ys: usize = 0, res: u64 = 0 };

    /// y[b] = W x[b] for the branches' rows; with `res` y = round(res + round(W x)); `row0`/`n` a slice of W's rows.
    fn gemv(self: *const Model, s: cuda.Stream, m: *const Mat, x: u64, rows: usize, o: Gemv) !void {
        const n = o.n orelse m.n - o.row0;
        const outs = if (o.glu) n / 2 else n;
        const bytes_a_weight: usize = if (self.format == .bf16) 2 else 1;
        const scales_a_row: usize = if (self.format == .int8_groups) m.k / 128 else 1;
        var a: cuda.Args = .{};
        a.add(m.w.ptr + o.row0 * m.k * bytes_a_weight);
        a.add(if (m.sc) |sc| sc.ptr + o.row0 * scales_a_row * 4 else @as(u64, 0));
        a.add(x);
        a.add(@as(c_int, @intCast(m.k)));
        a.add(@as(c_int, @intCast(m.k)));
        a.add(@as(c_int, @intCast(n)));
        a.add(o.y);
        a.add(@as(c_int, @intCast(if (o.ys != 0) o.ys else outs)));
        a.add(@as(u64, 0));
        a.add(o.res);
        a.add(@as(c_int, 1));
        try cuda.launch.launch(self.gemvKernel(rows, o.glu), .{ .grid = .{ .x = @intCast((outs + 7) / 8) }, .block = .{ .x = 256 }, .pdl = self.pdl }, s, &a);
    }

    /// Checks a step's sizes and addresses before anything is queued.
    pub fn validate(self: *const Model, in: *const Step) !void {
        const c = &self.cfg;
        if (in.branches == 0 or in.branches > max_branches) return error.BadBranches;
        if (in.capacity == 0) return error.BadCapacity;
        if (in.row0 >= in.row1 or in.row1 > c.vocab or in.logits_stride < in.row1) return error.BadRows;
        if (in.layers > c.layers) return error.BadLayers;
        if (in.tokens == 0 or in.positions == 0 or in.keys == 0 or in.values == 0) return error.BadAddress;
        if ((in.layers == 0 or in.layers == c.layers) and in.logits == 0) return error.BadAddress;
        const keys: [*]const u64 = @ptrFromInt(in.keys);
        const values: [*]const u64 = @ptrFromInt(in.values);
        for (0..c.layers) |i| if (keys[i] == 0 or values[i] == 0) return error.BadAddress;
        if (!self.ready()) return error.WeightsMissing;
    }

    /// Queues one step on `s` (eagerly, or into a capture).
    fn enqueue(self: *Model, s: cuda.Stream, in: *const Step) !void {
        const c = &self.cfg;
        const B: usize = in.branches;
        const H: usize = c.hidden;
        const heads: usize = c.heads;
        const kv: usize = c.kv_heads;
        const S: usize = c.splits;
        const run: usize = if (in.layers == 0) c.layers else in.layers;
        const keys: [*]const u64 = @ptrFromInt(in.keys);
        const values: [*]const u64 = @ptrFromInt(in.values);
        var a: cuda.Args = .{};
        a.add(in.tokens);
        a.add(self.embed.ptr);
        a.add(self.x.ptr);
        a.add(@as(c_int, @intCast(H)));
        try self.go(self.function("y2_embed"), s, .{ B, 1, 1 }, 256, &a);
        for (self.layers[0..run], 0..) |*l, li| {
            try self.rmsnorm(s, self.x.ptr, l.norms.ptr, self.xn.ptr, B, H);
            try self.gemv(s, &l.qkv, self.xn.ptr, B, .{ .y = self.qkv.ptr });
            a = .{};
            a.add(self.qkv.ptr);
            a.add(@as(c_int, @intCast(heads)));
            a.add(@as(c_int, @intCast(kv)));
            a.add(l.norms.ptr + 2 * H * 2);
            a.add(l.norms.ptr + 2 * H * 2 + head_dim * 2);
            a.add(self.inv_freq.ptr);
            a.add(in.positions);
            a.add(self.q.ptr);
            a.add(keys[li]);
            a.add(values[li]);
            a.add(@as(c_int, @intCast(in.capacity)));
            a.add(c.eps);
            try self.go(self.function("y2_qk_rope"), s, .{ B, heads + 2 * kv, 1 }, head_dim, &a);
            a = .{};
            a.add(self.q.ptr);
            a.add(keys[li]);
            a.add(values[li]);
            a.add(in.positions);
            a.add(@as(c_int, @intCast(heads)));
            a.add(@as(c_int, @intCast(kv)));
            a.add(@as(c_int, @intCast(in.capacity)));
            a.add(@as(c_int, @intCast(S)));
            a.add(@as(f32, 1.0 / @sqrt(@as(f32, head_dim))));
            a.add(self.ml.ptr);
            a.add(self.acc.ptr);
            try self.go(self.function("y2_attn_part"), s, .{ kv, B, S }, 32 * (heads / kv), &a);
            a = .{};
            a.add(self.ml.ptr);
            a.add(self.acc.ptr);
            a.add(@as(c_int, @intCast(heads)));
            a.add(@as(c_int, @intCast(S)));
            a.add(self.att.ptr);
            try self.go(self.function("y2_attn_merge"), s, .{ B, heads, 1 }, head_dim, &a);
            try self.gemv(s, &l.o, self.att.ptr, B, .{ .y = self.x.ptr, .res = self.x.ptr });
            try self.rmsnorm(s, self.x.ptr, l.norms.ptr + H * 2, self.xn.ptr, B, H);
            try self.gemv(s, &l.gate_up, self.xn.ptr, B, .{ .glu = true, .y = self.gu.ptr });
            try self.gemv(s, &l.down, self.gu.ptr, B, .{ .y = self.x.ptr, .res = self.x.ptr });
        }
        if (in.hidden_out != 0) {
            const out: cuda.DeviceBuffer = .{ .d = &self.driver, .ptr = in.hidden_out, .len = B * H * 2, .borrowed = true };
            try out.copyFrom(0, self.x.ptr, B * H * 2, s.handle);
        }
        if (run == c.layers) {
            try self.rmsnorm(s, self.x.ptr, self.norm.ptr, self.xn.ptr, B, H);
            try self.gemv(s, &self.lm_head, self.xn.ptr, B, .{ .row0 = in.row0, .n = in.row1 - in.row0, .y = in.logits + @as(u64, in.row0) * 2, .ys = in.logits_stride });
        }
        a = .{};
        a.add(in.positions);
        a.add(@as(c_int, @intCast(B)));
        try self.go(self.function("y2_advance"), s, .{ 1, 1, 1 }, 32, &a);
    }

    fn rmsnorm(self: *const Model, s: cuda.Stream, x: u64, w: u64, y: u64, rows: usize, dim: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(y);
        a.add(@as(c_int, @intCast(dim)));
        a.add(self.cfg.eps);
        try self.go(self.function("y2_rmsnorm"), s, .{ rows, 1, 1 }, 256, &a);
    }

    fn matches(b: *const Bound, in: *const Step, keys: []const u64, values: []const u64) bool {
        return std.mem.eql(u8, std.mem.asBytes(&b.step), std.mem.asBytes(in)) and
            std.mem.eql(u64, b.keys[0..keys.len], keys) and std.mem.eql(u64, b.values[0..values.len], values);
    }

    /// One step on the host's stream: a replay of the graph captured for this binding (captured on first use), or
    /// queued eagerly for a check (fewer layers, hidden_out) or with graphs off. Returns once queued.
    pub fn step(self: *Model, in: *const Step) !void {
        try self.validate(in);
        try self.ctx.makeCurrent();
        const host = cuda.Stream.borrowed(&self.driver, @ptrFromInt(in.stream));
        self.steps += 1;
        if (self.cfg.graphs == 0 or in.layers != 0 or in.hidden_out != 0) return self.enqueue(host, in);
        const L: usize = self.cfg.layers;
        const keys = @as([*]const u64, @ptrFromInt(in.keys))[0..L];
        const values = @as([*]const u64, @ptrFromInt(in.values))[0..L];
        self.clock += 1;
        for (&self.graphs) |*g| if (g.*) |*b| if (matches(b, in, keys, values)) {
            b.used = self.clock;
            return b.exec.launchOn(host);
        };
        // the least recently used slot takes the new binding
        var slot: usize = 0;
        for (self.graphs, 0..) |g, i| {
            if (g == null) {
                slot = i;
                break;
            }
            if (g.?.used < self.graphs[slot].?.used) slot = i;
        }
        if (self.graphs[slot]) |*old| {
            old.exec.deinit();
            self.graphs[slot] = null;
        }
        try cuda.graph.beginCapture(self.capture, .thread_local);
        self.enqueue(self.capture, in) catch |e| {
            var g = cuda.graph.endCapture(self.capture) catch return e;
            g.deinit();
            return e;
        };
        var g = try cuda.graph.endCapture(self.capture);
        defer g.deinit();
        var b: Bound = .{ .step = in.*, .keys = @splat(0), .values = @splat(0), .exec = try g.instantiate(), .used = self.clock };
        @memcpy(b.keys[0..L], keys);
        @memcpy(b.values[0..L], values);
        self.graphs[slot] = b;
        self.captures += 1;
        try b.exec.launchOn(host);
    }
};
