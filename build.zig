const std = @import("std");

// Pyrit – alles in Zig: Host-Bibliothek, GPU-Kernel (NVIDIA über PTX, AMD über
// amdgcn) und Tests. Der C-Header include/pyrit.h beschreibt nur die ABI für
// Aufrufer in anderen Sprachen.
//
//   zig build                 Bibliothek (statisch + dynamisch) und Header
//   zig build test            Unit-, CPU- und ABI-Tests, AMD-Übersetzung (ohne GPU)
//   zig build kernel-check    PTX mit ptxas für sm_120 prüfen (braucht CUDA, keine GPU)
//   zig build gpu-test        GPU gegen CPU-Referenz (braucht eine freie NVIDIA-GPU)
//   zig build demo            Demo: Minecraft-artige Welt im Zuschauermodus
//   zig build --release=fast  optimiert

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });
    const ptx_arch = b.option([]const u8, "ptx-arch", "PTX-Zielarchitektur (Treiber übersetzt für neuere GPUs)") orelse "sm_75";
    const amd_arch = b.option([]const u8, "amd-arch", "AMD-Zielarchitektur für die Übersetzungsprüfung") orelse "gfx1100";

    // ------------------------------------------------------------------
    // GPU-Kernel für NVIDIA: Zig -> LLVM-IR -> Alias-Korrektur -> PTX
    // ------------------------------------------------------------------
    const nvptx = b.resolveTargetQuery(.{
        .cpu_arch = .nvptx64,
        .os_tag = .cuda,
        .cpu_model = .{ .explicit = cpuModel(.nvptx64, ptx_arch) },
    });
    const kernels_nv = b.addObject(.{ .name = "pyrit_kernels", .root_module = kernelModule(b, nvptx) });

    // Zig stellt Exporte als LLVM-Alias dar; NVPTX erlaubt keine Aliasse auf Kernel.
    const fixup = b.addExecutable(.{
        .name = "ptx_fixup",
        .root_module = b.createModule(.{ .root_source_file = b.path("tools/ptx_fixup.zig"), .target = b.graph.host, .optimize = .Debug }),
    });
    const run_fixup = b.addRunArtifact(fixup);
    run_fixup.addFileArg(kernels_nv.getEmittedLlvmIr());
    const fixed_ir = run_fixup.addOutputFileArg("pyrit_kernels.ll");

    const to_ptx = b.addSystemCommand(&.{ b.graph.zig_exe, "cc", "-target", "nvptx64-cuda", b.fmt("-mcpu={s}", .{ptx_arch}), "-O3", "-S", "-Wno-unused-command-line-argument" });
    to_ptx.addFileArg(fixed_ir);
    to_ptx.addArg("-o");
    const ptx = to_ptx.addOutputFileArg("pyrit_kernels.ptx");
    b.getInstallStep().dependOn(&b.addInstallFile(ptx, "share/pyrit/pyrit_kernels.ptx").step);

    // OptiX-Programme (RT-Cores): nur Traversierung und Strahllisten. Die
    // Schattierung läuft in CUDA (src/device/replay.zig) – mit optixTrace in
    // der Schattierung brauchte OptiX zum Übersetzen Minuten und GB.
    const rt_ptx = ptxOf(b, fixup, nvptx, ptx_arch, "src/rt_kernels.zig", "pyrit_rt");

    // Kernel der Demo (Geländegenerator): derselbe Weg, eigenes Modul. Pyrit
    // kennt ihn nicht – die Demo lädt ihn selbst und hängt ihn über
    // PyrWorldInfo.generate ein.
    const demo_nv = b.addObject(.{ .name = "demo_kernels", .root_module = kernelModuleFrom(b, nvptx, "demo/kernels.zig") });
    const run_demo_fixup = b.addRunArtifact(fixup);
    run_demo_fixup.addFileArg(demo_nv.getEmittedLlvmIr());
    const demo_ir = run_demo_fixup.addOutputFileArg("demo_kernels.ll");
    const demo_to_ptx = b.addSystemCommand(&.{ b.graph.zig_exe, "cc", "-target", "nvptx64-cuda", b.fmt("-mcpu={s}", .{ptx_arch}), "-O3", "-S", "-Wno-unused-command-line-argument" });
    demo_to_ptx.addFileArg(demo_ir);
    demo_to_ptx.addArg("-o");
    const demo_ptx = demo_to_ptx.addOutputFileArg("demo_kernels.ptx");

    // ------------------------------------------------------------------
    // Optional: NVIDIA DLSS (Super Resolution, Ray Reconstruction) über NGX-CUDA.
    // Das SDK (github.com/NVIDIA/DLSS) wird nicht mitgeliefert: Header werden
    // beim Bauen übersetzt, die statische NGX-Bibliothek dazugelinkt.
    // ------------------------------------------------------------------
    const dlss_sdk = b.option([]const u8, "dlss-sdk", "Pfad zum NVIDIA DLSS SDK (aktiviert DLSS und Ray Reconstruction)");
    const options = b.addOptions();
    options.addOption(bool, "dlss", dlss_sdk != null);
    options.addOption([]const u8, "dlss_lib_dir", if (dlss_sdk) |sdk| b.fmt("{s}/lib/Linux_x86_64/rel", .{sdk}) else "");
    var ngx: ?*std.Build.Module = null;
    if (dlss_sdk) |sdk| {
        const wf = b.addWriteFiles();
        const header = wf.add("ngx_wrap.h",
            \\#include <stddef.h>
            \\#include <wchar.h>
            \\#include <vulkan/vulkan.h>
            \\#include "nvsdk_ngx_vk.h"
            \\#include "nvsdk_ngx.h"
            \\#include "nvsdk_ngx_defs.h"
            \\#include "nvsdk_ngx_params.h"
            \\#include "nvsdk_ngx_defs_dlssd.h"
            \\#include "nvsdk_ngx_params_dlssd.h"
            \\#include "nvsdk_ngx_defs_dlssg.h"
            \\
        );
        const tc = b.addTranslateC(.{ .root_source_file = header, .target = target, .optimize = optimize, .link_libc = true });
        tc.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{sdk}) });
        ngx = tc.createModule();
    }
    // Machbarkeitsprobe DLSS Frame Generation über NGX-Vulkan (kopflos, nur Interop)
    if (dlss_sdk) |sdk| {
        const wf2 = b.addWriteFiles();
        const vkh = wf2.add("ngx_vk_wrap.h",
            \\#include <stddef.h>
            \\#include <wchar.h>
            \\#include <vulkan/vulkan.h>
            \\#include "nvsdk_ngx_vk.h"
            \\#include "nvsdk_ngx_defs.h"
            \\#include "nvsdk_ngx_params.h"
            \\#include "nvsdk_ngx_defs_dlssg.h"
            \\
        );
        const tcv = b.addTranslateC(.{ .root_source_file = vkh, .target = target, .optimize = optimize, .link_libc = true });
        tcv.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{sdk}) });
        const pm = b.createModule(.{ .root_source_file = b.path("tools/dlssg_probe.zig"), .target = target, .optimize = optimize, .link_libc = true });
        pm.addImport("ngxvk", tcv.createModule());
        pm.addOptions("build_options", options);
        pm.addObjectFile(.{ .cwd_relative = b.fmt("{s}/lib/Linux_x86_64/libnvsdk_ngx.a", .{sdk}) });
        const stdcxx = std.mem.trim(u8, b.run(&.{ "cc", "-print-file-name=libstdc++.so.6" }), " \n\r\t");
        pm.addObjectFile(.{ .cwd_relative = stdcxx });
        pm.linkSystemLibrary("vulkan", .{});
        const probe = b.addExecutable(.{ .name = "pyrit-dlssg-probe", .root_module = pm });
        b.installArtifact(probe);
        const run_probe = b.addRunArtifact(probe);
        b.step("dlssg-probe", "DLSS Frame Generation über NGX-Vulkan prüfen (braucht -Ddlss-sdk)").dependOn(&run_probe.step);
    }

    const ptx_files = PtxFiles{ .kernels = ptx, .rt = rt_ptx, .options = options, .ngx = ngx, .dlss_sdk = dlss_sdk };

    // ------------------------------------------------------------------
    // Derselbe Kernel-Code für AMD (Übersetzungsprüfung; HIP-Backend folgt)
    // ------------------------------------------------------------------
    const amdgcn = b.resolveTargetQuery(.{
        .cpu_arch = .amdgcn,
        .os_tag = .amdhsa,
        .cpu_model = .{ .explicit = cpuModel(.amdgcn, amd_arch) },
    });
    const kernels_amd = b.addObject(.{ .name = "pyrit_kernels_amd", .root_module = kernelModule(b, amdgcn) });
    const amd_asm = b.addInstallFile(kernels_amd.getEmittedAsm(), "share/pyrit/pyrit_kernels_amdgcn.s");

    // ------------------------------------------------------------------
    // Host-Bibliothek
    // ------------------------------------------------------------------
    const lib = b.addLibrary(.{ .name = "pyrit", .root_module = hostModule(b, target, optimize, ptx_files), .linkage = .static });
    const shared = b.addLibrary(.{ .name = "pyrit", .root_module = hostModule(b, target, optimize, ptx_files), .linkage = .dynamic });
    b.installArtifact(lib);
    b.installArtifact(shared);
    b.installDirectory(.{ .source_dir = b.path("include"), .install_dir = .header, .install_subdir = "" });

    // ------------------------------------------------------------------
    // Tests
    // ------------------------------------------------------------------
    const test_step = b.step("test", "Unit-, CPU- und ABI-Tests, AMD-Übersetzung (ohne GPU)");

    const unit = b.addTest(.{ .root_module = hostModule(b, target, optimize, ptx_files) });
    test_step.dependOn(&b.addRunArtifact(unit).step);

    const cpu = b.addTest(.{ .root_module = demoUserModule(b, "tests/cpu_test.zig", target, optimize, ptx_files, demo_ptx) });
    test_step.dependOn(&b.addRunArtifact(cpu).step);

    const tc = b.addTranslateC(.{ .root_source_file = b.path("include/pyrit.h"), .target = target, .optimize = optimize });
    const abi_mod = testModule(b, "tests/abi_test.zig", target, optimize, ptx_files);
    abi_mod.addImport("c", tc.createModule());
    const abi = b.addTest(.{ .root_module = abi_mod });
    test_step.dependOn(&b.addRunArtifact(abi).step);

    test_step.dependOn(&amd_asm.step);

    // PTX mit ptxas prüfen (CUDA-Toolkit nötig, GPU nicht)
    const ptxas = b.option([]const u8, "ptxas", "Pfad zu ptxas für kernel-check") orelse "/opt/cuda/bin/ptxas";
    const check_arch = b.option([]const u8, "check-arch", "Zielarchitektur für kernel-check") orelse "sm_120";
    const run_ptxas = b.addSystemCommand(&.{ ptxas, "-v", b.fmt("-arch={s}", .{check_arch}) });
    run_ptxas.addFileArg(ptx);
    run_ptxas.addArg("-o");
    _ = run_ptxas.addOutputFileArg("pyrit_kernels.cubin");
    const kernel_check = b.step("kernel-check", "PTX mit ptxas prüfen und AMD-Kernel übersetzen");
    kernel_check.dependOn(&run_ptxas.step);
    kernel_check.dependOn(&amd_asm.step);

    // OptiX-Anbindung gegen die Original-Header prüfen (Header nicht Teil von Pyrit)
    if (b.option([]const u8, "optix-include", "OptiX-SDK-include-Verzeichnis für optix-abi-test")) |optix_inc| {
        const cuda_inc = b.option([]const u8, "cuda-include", "CUDA-include-Verzeichnis") orelse "/opt/cuda/include";
        const otc = b.addTranslateC(.{ .root_source_file = b.path("tests/optix_abi.h"), .target = target, .optimize = optimize });
        otc.addIncludePath(.{ .cwd_relative = optix_inc });
        otc.addIncludePath(.{ .cwd_relative = cuda_inc });
        const om = testModule(b, "tests/optix_abi_test.zig", target, optimize, ptx_files);
        om.addImport("optix_c", otc.createModule());
        const ot = b.addTest(.{ .root_module = om });
        b.step("optix-abi-test", "src/optix.zig gegen die OptiX-Header prüfen").dependOn(&b.addRunArtifact(ot).step);
    }

    // Build-Werkzeug testen
    const fixup_test = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("tools/ptx_fixup.zig"), .target = b.graph.host }) });
    test_step.dependOn(&b.addRunArtifact(fixup_test).step);

    // Render-Werkzeug: zig build render -- [--vox datei.vox] [--out bild.ppm] ...
    // Der Weltmodus rendert die Welt der Demo.
    const render_exe = b.addExecutable(.{ .name = "pyrit-render", .root_module = demoUserModule(b, "tools/pyrit_render.zig", target, optimize, ptx_files, demo_ptx) });
    b.installArtifact(render_exe);
    const run_render = b.addRunArtifact(render_exe);
    if (b.args) |a| run_render.addArgs(a);
    b.step("render", "Bild rendern (GPU): zig build render -- --out bild.ppm").dependOn(&run_render.step);

    // Demo: Minecraft-artige Welt, Zuschauermodus (Wayland-Fenster, dynamisch
    // geladen) oder Aufnahme ohne Fenster (--record)
    const demo_mod = testModule(b, "demo/main.zig", target, optimize, ptx_files);
    demo_mod.addAnonymousImport("demo_ptx", .{ .root_source_file = demo_ptx });
    const demo_exe = b.addExecutable(.{ .name = "pyrit-demo", .root_module = demo_mod });
    b.installArtifact(demo_exe);
    const run_demo = b.addRunArtifact(demo_exe);
    if (b.args) |a| run_demo.addArgs(a);
    b.step("demo", "Demo starten: zig build demo -- --size 1280x720").dependOn(&run_demo.step);

    // GPU-Test: nur auf ausdrücklichen Aufruf
    const gpu_test = b.addExecutable(.{ .name = "pyrit-gpu-test", .root_module = demoUserModule(b, "tests/gpu_test.zig", target, optimize, ptx_files, demo_ptx) });
    const gpu_step = b.step("gpu-test", "GPU gegen CPU-Referenz und Durchsatz (braucht eine freie NVIDIA-GPU)");
    gpu_step.dependOn(&b.addRunArtifact(gpu_test).step);
    const gpu_build = b.step("gpu-test-build", "GPU-Test nur bauen");
    gpu_build.dependOn(&b.addInstallArtifact(gpu_test, .{}).step);
}

fn cpuModel(comptime arch: std.Target.Cpu.Arch, name: []const u8) *const std.Target.Cpu.Model {
    const cpus = switch (arch) {
        .nvptx64 => std.Target.nvptx.cpu,
        .amdgcn => std.Target.amdgcn.cpu,
        else => unreachable,
    };
    inline for (@typeInfo(cpus).@"struct".decls) |d| {
        if (std.mem.eql(u8, d.name, name)) return &@field(cpus, d.name);
    }
    std.debug.panic("unbekannte GPU-Architektur: {s}", .{name});
}

/// Gerätemodul (Traversierung, Szene, Motion Vectors) für ein Ziel
fn deviceModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("src/device/root.zig"), .target = target, .optimize = optimize });
}

/// Eingaben der Host-Bibliothek (eingebettete Kernel, Build-Optionen, NGX)
const PtxFiles = struct {
    kernels: std.Build.LazyPath,
    rt: std.Build.LazyPath,
    options: *std.Build.Step.Options,
    ngx: ?*std.Build.Module = null,
    dlss_sdk: ?[]const u8 = null,
};

fn kernelModule(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Module {
    return kernelModuleFrom(b, target, "src/gpu_kernels.zig");
}

fn kernelModuleFrom(b: *std.Build, target: std.Build.ResolvedTarget, root: []const u8) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path(root), .target = target, .optimize = .ReleaseFast });
    m.addImport("pyrit_device", deviceModule(b, target, .ReleaseFast));
    return m;
}

fn hostModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, ptx: PtxFiles) *std.Build.Module {
    return hostModuleWith(b, target, optimize, ptx, deviceModule(b, target, optimize));
}

fn hostModuleWith(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, ptx: PtxFiles, device: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("src/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("pyrit_device", device);
    m.addAnonymousImport("pyrit_ptx", .{ .root_source_file = ptx.kernels });
    m.addAnonymousImport("pyrit_rt_ptx", .{ .root_source_file = ptx.rt });
    m.addOptions("build_options", ptx.options);
    if (ptx.ngx) |ngx| {
        m.addImport("ngx", ngx);
        m.addObjectFile(.{ .cwd_relative = b.fmt("{s}/lib/Linux_x86_64/libnvsdk_ngx.a", .{ptx.dlss_sdk.?}) });
        // die NGX-Bibliothek ist gegen libstdc++ gebaut (std::__cxx11); Zig würde
        // "stdc++" durch seine libc++ ersetzen, daher die Systembibliothek per Pfad
        const stdcxx = std.mem.trim(u8, b.run(&.{ "cc", "-print-file-name=libstdc++.so.6" }), " \n\r\t");
        m.addObjectFile(.{ .cwd_relative = stdcxx });
    }
    return m;
}

fn testModule(b: *std.Build, path: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, ptx: PtxFiles) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path(path), .target = target, .optimize = optimize, .link_libc = true });
    const device = deviceModule(b, target, optimize);
    m.addImport("pyrit_device", device);
    m.addImport("pyrit", hostModuleWith(b, target, optimize, ptx, device));
    return m;
}

/// Wie testModule, dazu die Szene der Demo als Modul "demo" – mit denselben
/// Instanzen von pyrit und pyrit_device, sonst wären die Typen verschieden.
fn demoUserModule(b: *std.Build, path: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, ptx: PtxFiles, demo_ptx: std.Build.LazyPath) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path(path), .target = target, .optimize = optimize, .link_libc = true });
    const device = deviceModule(b, target, optimize);
    const host = hostModuleWith(b, target, optimize, ptx, device);
    m.addImport("pyrit_device", device);
    m.addImport("pyrit", host);
    const demo = b.createModule(.{ .root_source_file = b.path("demo/scene.zig"), .target = target, .optimize = optimize, .link_libc = true });
    demo.addImport("pyrit_device", device);
    demo.addImport("pyrit", host);
    demo.addAnonymousImport("demo_ptx", .{ .root_source_file = demo_ptx });
    m.addImport("demo", demo);
    return m;
}

/// Zig-Kernel -> LLVM-IR -> Alias-Korrektur -> PTX, installiert unter share/pyrit
fn ptxOf(b: *std.Build, fixup: *std.Build.Step.Compile, nvptx: std.Build.ResolvedTarget, ptx_arch: []const u8, root: []const u8, name: []const u8) std.Build.LazyPath {
    const obj = b.addObject(.{ .name = name, .root_module = kernelModuleFrom(b, nvptx, root) });
    const run = b.addRunArtifact(fixup);
    run.addFileArg(obj.getEmittedLlvmIr());
    const ir = run.addOutputFileArg(b.fmt("{s}.ll", .{name}));
    const to_ptx = b.addSystemCommand(&.{ b.graph.zig_exe, "cc", "-target", "nvptx64-cuda", b.fmt("-mcpu={s}", .{ptx_arch}), "-O3", "-S", "-Wno-unused-command-line-argument" });
    to_ptx.addFileArg(ir);
    to_ptx.addArg("-o");
    const out = to_ptx.addOutputFileArg(b.fmt("{s}.ptx", .{name}));
    b.getInstallStep().dependOn(&b.addInstallFile(out, b.fmt("share/pyrit/{s}.ptx", .{name})).step);
    return out;
}
