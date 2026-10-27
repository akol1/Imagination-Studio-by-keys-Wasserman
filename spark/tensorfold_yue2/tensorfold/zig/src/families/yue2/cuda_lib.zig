//! The YuE2 CUDA family as a C library for a host that shares its CUDA context: tf_yue2_open, tf_yue2_set_weight,
//! tf_yue2_step, tf_yue2_close. The host (PyTorch) owns the prompt pass, the KV cache, the logits and sampling.
const std = @import("std");
const engine = @import("engine.zig");
const Model = engine.Model;

const gpa = std.heap.c_allocator;

/// The layout of Config and Step; a host checks it before anything else.
export fn tf_yue2_abi_version() u32 {
    return 1;
}

/// sizeof(Config), sizeof(Step): a host's structs must match.
export fn tf_yue2_struct_sizes(out: *[2]u32) void {
    out.* = .{ @sizeOf(engine.Config), @sizeOf(engine.Step) };
}

/// An engine for the model `config` describes, its weights unset; null on failure (the reason on stderr).
export fn tf_yue2_open(config: *const engine.Config) ?*Model {
    return Model.open(gpa, config) catch |e| {
        std.debug.print("tf_yue2_open: {t}\n", .{e});
        return null;
    };
}

export fn tf_yue2_close(model: *Model) void {
    model.close();
}

/// One host tensor into the engine (`layer` -1 for embed / norm / lm_head / inv_freq): 0, or -1 when refused.
export fn tf_yue2_set_weight(model: *Model, layer: c_int, which: c_int, ptr: u64, rows: u32, cols: u32) c_int {
    if (which < 0 or which >= engine.Which.count) return -1;
    model.setWeight(layer, @intCast(which), ptr, rows, cols) catch |e| {
        std.debug.print("tf_yue2_set_weight({d}, {d}): {t}\n", .{ layer, which, e });
        return -1;
    };
    return 0;
}

/// 1 once every weight a step reads has been set.
export fn tf_yue2_ready(model: *const Model) c_int {
    return @intFromBool(model.ready());
}

/// The weight bytes held on the device, steps queued and graphs captured.
export fn tf_yue2_info(model: *const Model, out: *[4]u64) void {
    out.* = .{ model.weight_bytes, model.steps, model.captures, model.cfg.weights };
}

/// Queues one decode step on the host's stream; 0, or -1 when refused (nothing queued) or failed.
export fn tf_yue2_step(model: *Model, in: *const engine.Step) c_int {
    model.step(in) catch |e| {
        std.debug.print("tf_yue2_step: {t}\n", .{e});
        return -1;
    };
    return 0;
}

test "yue2 library: struct layout the host mirrors" {
    try std.testing.expectEqual(@as(u32, 1), tf_yue2_abi_version());
    var sizes: [2]u32 = undefined;
    tf_yue2_struct_sizes(&sizes);
    try std.testing.expectEqual([2]u32{ 44, 80 }, sizes);
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(engine.Step, "branches"));
    try std.testing.expectEqual(@as(usize, 72), @offsetOf(engine.Step, "hidden_out"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(engine.Config, "eps"));
}
