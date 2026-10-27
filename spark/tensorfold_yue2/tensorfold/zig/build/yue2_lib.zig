//! `zig build tf-yue2 -Dnvcc=...`: libtf_yue2.so, YuE2-3B's decode step for a host pipeline, with its own image.
//! `-Dyue2-fatbin=<file>` embeds a prebuilt image instead (an empty one compiles the library without nvcc).
const std = @import("std");
const fatbin = @import("cuda_fatbin.zig").fatbin;

pub fn library(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, nvcc: ?[]const u8, sms: []const u8) void {
    const step = b.step("tf-yue2", "libtf_yue2.so: YuE2-3B's autoregressive decode step on CUDA (bf16 or int8 weights)");
    const prebuilt = b.option([]const u8, "yue2-fatbin", "a prebuilt yue2.fatbin to embed in libtf_yue2.so");
    const image: std.Build.LazyPath = if (prebuilt) |file| b.graph.cwdRelativePath(file) else if (nvcc) |tool| blk: {
        const run = b.addSystemCommand(&.{ tool, "--version" });
        run.has_side_effects = true;
        break :blk fatbin(b, tool, run.captureStdOut(.{}), .{ .name = "yue2", .flags = &.{"-O3"} }, sms);
    } else {
        step.dependOn(&b.addFail("tf-yue2 needs -Dnvcc (or -Dyue2-fatbin)").step);
        return;
    };
    const cuda = @import("cuda_runtime.zig").runtime(b, target, optimize, &.{});
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/families/yue2/cuda_lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "cuda", .module = cuda }},
    });
    root.addAnonymousImport("fatbin_yue2", .{ .root_source_file = image });
    const lib = b.addLibrary(.{ .name = "tf_yue2", .linkage = .dynamic, .root_module = root });
    step.dependOn(&b.addInstallArtifact(lib, .{}).step);
    // the library's own layout test, no GPU (an empty image)
    const tmod = b.createModule(.{
        .root_source_file = b.path("zig/src/families/yue2/cuda_lib.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{.{ .name = "cuda", .module = cuda }},
    });
    tmod.addAnonymousImport("fatbin_yue2", .{ .root_source_file = b.addWriteFiles().add("yue2-test.fatbin", "") });
    b.step("test-yue2", "libtf_yue2's struct layout").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tmod })).step);
}
