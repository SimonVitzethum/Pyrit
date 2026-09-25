//! NVIDIA DLSS Frame Generation über NGX-Vulkan – Vulkan nur als kopflose
//! Interop-Schicht.
//!
//! Über den CUDA-Weg bietet NGX die Frame Generation nicht an: die
//! NVSDK_NGX_CUDA_*-Einsprünge in libnvidia-ngx-dlssg.so sind Stummel
//! (siehe dlss.zig, probeFrameGeneration). Über Vulkan geht es, und zwar ohne
//! Fenster, Swapchain oder Present: Pyrit legt ein Vulkan-Gerät nur für NGX
//! an, teilt vier Bilder mit CUDA (externer Speicher per fd) und
//! synchronisiert über eine Timeline-Semaphore, die beide Seiten sehen.
//!
//!   CUDA:   Bild, Bewegung, Tiefe in die geteilten Bilder  -> signal n
//!   Vulkan: warten auf n, NGX erzeugt das Zwischenbild      -> signal n+1
//!   CUDA:   warten auf n+1, Zwischenbild in die Ausgabe
//!
//! Der Host wartet dabei nie auf die GPU. libvulkan wird erst geladen, wenn
//! die Frame Generation angefordert wird – ohne sie hängt Pyrit an keiner
//! Grafik-API. Gerendert wird weiter nur in CUDA.

const std = @import("std");
const build_options = @import("build_options");
const cuda = @import("cuda.zig");
const diag = @import("diag.zig");
const types = @import("pyrit_device").types;
const Context = @import("context.zig").Context;

const Error = diag.Error;
const fail = diag.fail;

pub const available = build_options.dlss;
const c = if (available) @import("ngx") else struct {};

const project_id = "5b3e1c2a-7d4f-4e8a-9b61-707972697400";

fn ngxOk(r: c_uint) bool {
    return r & 0xFFF0_0000 != 0xBAD0_0000;
}

/// Eingaben eines Aufrufs (alle in Ausgabeauflösung, auf dem Gerät)
pub const Inputs = struct {
    /// fertiges LDR-Bild dieses Frames (4 Bytes je Pixel)
    ldr: u64,
    bgra: bool,
    /// [4]f32 je Pixel: Bewegung (Pixel, Vorframe − jetzt), lineare Tiefe
    motion_depth: u64,
    /// Ziel: Zwischenbild (4 Bytes je Pixel, gleiche Anordnung wie ldr)
    out: u64,
    width: u32,
    height: u32,
    near: f32,
    /// Kamera jetzt und im Vorframe (Zeilenvektor-Form, 4x4)
    world_to_view: [16]f32,
    view_to_clip: [16]f32,
    prev_world_to_view: [16]f32,
    prev_view_to_clip: [16]f32,
    fov_y: f32,
    reset: bool,
    /// Multi Frame Generation: Zahl der Zwischenbilder und welches (1..count)
    multi_count: u32 = 1,
    multi_index: u32 = 1,
};

const Img = struct {
    image: c.VkImage = null,
    mem: c.VkDeviceMemory = null,
    view: c.VkImageView = null,
    ext: cuda.CUexternalMemory = null,
    mip: cuda.CUmipmappedArray = null,
    arr: cuda.CUarray = null,
    format: c.VkFormat = 0,
    bytes_px: u32 = 0,
};

const img_color = 0;
const img_mvec = 1;
const img_depth = 2;
const img_out = 3;

/// Vulkan-Funktionen, per vkGetInstanceProcAddr/vkGetDeviceProcAddr geladen
const Vk = struct {
    vkDestroyInstance: c.PFN_vkDestroyInstance = null,
    vkEnumeratePhysicalDevices: c.PFN_vkEnumeratePhysicalDevices = null,
    vkGetPhysicalDeviceProperties: c.PFN_vkGetPhysicalDeviceProperties = null,
    vkGetPhysicalDeviceQueueFamilyProperties: c.PFN_vkGetPhysicalDeviceQueueFamilyProperties = null,
    vkGetPhysicalDeviceMemoryProperties: c.PFN_vkGetPhysicalDeviceMemoryProperties = null,
    vkCreateDevice: c.PFN_vkCreateDevice = null,
    vkGetDeviceProcAddr: c.PFN_vkGetDeviceProcAddr = null,
    // Gerät
    vkDestroyDevice: c.PFN_vkDestroyDevice = null,
    vkGetDeviceQueue: c.PFN_vkGetDeviceQueue = null,
    vkCreateImage: c.PFN_vkCreateImage = null,
    vkDestroyImage: c.PFN_vkDestroyImage = null,
    vkGetImageMemoryRequirements: c.PFN_vkGetImageMemoryRequirements = null,
    vkAllocateMemory: c.PFN_vkAllocateMemory = null,
    vkFreeMemory: c.PFN_vkFreeMemory = null,
    vkBindImageMemory: c.PFN_vkBindImageMemory = null,
    vkCreateImageView: c.PFN_vkCreateImageView = null,
    vkDestroyImageView: c.PFN_vkDestroyImageView = null,
    vkGetMemoryFdKHR: c.PFN_vkGetMemoryFdKHR = null,
    vkCreateSemaphore: c.PFN_vkCreateSemaphore = null,
    vkDestroySemaphore: c.PFN_vkDestroySemaphore = null,
    vkGetSemaphoreFdKHR: c.PFN_vkGetSemaphoreFdKHR = null,
    vkCreateCommandPool: c.PFN_vkCreateCommandPool = null,
    vkDestroyCommandPool: c.PFN_vkDestroyCommandPool = null,
    vkAllocateCommandBuffers: c.PFN_vkAllocateCommandBuffers = null,
    vkBeginCommandBuffer: c.PFN_vkBeginCommandBuffer = null,
    vkEndCommandBuffer: c.PFN_vkEndCommandBuffer = null,
    vkResetCommandBuffer: c.PFN_vkResetCommandBuffer = null,
    vkCmdPipelineBarrier: c.PFN_vkCmdPipelineBarrier = null,
    vkQueueSubmit: c.PFN_vkQueueSubmit = null,
    vkCreateFence: c.PFN_vkCreateFence = null,
    vkDestroyFence: c.PFN_vkDestroyFence = null,
    vkWaitForFences: c.PFN_vkWaitForFences = null,
    vkResetFences: c.PFN_vkResetFences = null,
    vkDeviceWaitIdle: c.PFN_vkDeviceWaitIdle = null,
};

const instance_fns = [_][]const u8{
    "vkDestroyInstance",                     "vkEnumeratePhysicalDevices",          "vkGetPhysicalDeviceProperties",
    "vkGetPhysicalDeviceQueueFamilyProperties", "vkGetPhysicalDeviceMemoryProperties", "vkCreateDevice",
    "vkGetDeviceProcAddr",
};

const ring_size = 8;

pub const DlssG = struct {
    lib: std.DynLib,
    gipa: c.PFN_vkGetInstanceProcAddr,
    vk: Vk = .{},
    instance: c.VkInstance = null,
    pd: c.VkPhysicalDevice = null,
    device: c.VkDevice = null,
    queue: c.VkQueue = null,
    qfi: u32 = 0,
    pool: c.VkCommandPool = null,
    /// Ring aus Befehlspuffern: mehr als Zwischenbilder je Frame (6x = 5),
    /// sonst wartete der Host beim dritten Aufruf auf den ersten – und der
    /// läuft erst nach dem ganzen Frame auf der GPU
    cmds: [ring_size]c.VkCommandBuffer = .{null} ** ring_size,
    fences: [ring_size]c.VkFence = .{null} ** ring_size,
    ring: u32 = 0,
    sem: c.VkSemaphore = null,
    sem_cuda: cuda.CUexternalSemaphore = null,
    value: u64 = 0,
    params: ?*c.NVSDK_NGX_Parameter = null,
    feature: ?*c.NVSDK_NGX_Handle = null,
    imgs: [4]Img = .{ .{}, .{}, .{}, .{} },
    /// Zwischenpuffer: Bewegung (2 x f32) und Tiefe (f32), linear
    mv_lin: cuda.CUdeviceptr = 0,
    depth_lin: cuda.CUdeviceptr = 0,
    width: u32 = 0,
    height: u32 = 0,
    bgra: bool = false,
    ngx_up: bool = false,
    /// größte Zahl an Zwischenbildern je Frame (Multi Frame Generation)
    multi_max: u32 = 1,

    fn gi(self: *DlssG, instance: c.VkInstance, name: [*:0]const u8) c.PFN_vkVoidFunction {
        return self.gipa.?(instance, name);
    }

    pub fn create(ctx: *Context) Error!*DlssG {
        if (!available) return fail(error.NotFound, "DLSS Frame Generation braucht einen Build mit -Ddlss-sdk", .{});
        const self = ctx.gpa.create(DlssG) catch return fail(error.OutOfMemory, "Host-Speicher", .{});
        errdefer ctx.gpa.destroy(self);
        var lib = std.DynLib.open("libvulkan.so.1") catch return fail(error.NotFound, "DLSS Frame Generation: libvulkan.so.1 nicht gefunden", .{});
        const gipa = lib.lookup(c.PFN_vkGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse return fail(error.NotFound, "vkGetInstanceProcAddr fehlt", .{});
        self.* = .{ .lib = lib, .gipa = gipa };
        errdefer self.destroyVulkan();
        try self.initVulkan(ctx);
        try self.initNgx(ctx);
        return self;
    }

    fn vkCheck(r: c.VkResult, what: []const u8) Error!void {
        if (r != c.VK_SUCCESS) return fail(error.NotFound, "DLSS-FG (Vulkan): {s} = {d}", .{ what, r });
    }

    fn initVulkan(self: *DlssG, ctx: *Context) Error!void {
        _ = ctx;
        const create_instance: c.PFN_vkCreateInstance = @ptrCast(self.gi(null, "vkCreateInstance"));
        const iexts = [_][*:0]const u8{ "VK_KHR_get_physical_device_properties2", "VK_KHR_external_memory_capabilities", "VK_KHR_external_semaphore_capabilities" };
        const app = c.VkApplicationInfo{ .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "pyrit", .apiVersion = c.VK_API_VERSION_1_3 };
        const ici = c.VkInstanceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &app,
            .enabledExtensionCount = iexts.len,
            .ppEnabledExtensionNames = @ptrCast(&iexts),
        };
        try vkCheck(create_instance.?(&ici, null, &self.instance), "vkCreateInstance");
        inline for (instance_fns) |name| {
            @field(self.vk, name) = @ptrCast(self.gi(self.instance, name ++ ""));
        }

        // NVIDIA-Gerät
        var n: u32 = 0;
        try vkCheck(self.vk.vkEnumeratePhysicalDevices.?(self.instance, &n, null), "vkEnumeratePhysicalDevices");
        var pds: [8]c.VkPhysicalDevice = undefined;
        n = @min(n, 8);
        try vkCheck(self.vk.vkEnumeratePhysicalDevices.?(self.instance, &n, &pds), "vkEnumeratePhysicalDevices");
        for (pds[0..n]) |p| {
            var props: c.VkPhysicalDeviceProperties = undefined;
            self.vk.vkGetPhysicalDeviceProperties.?(p, &props);
            if (props.vendorID == 0x10de) {
                self.pd = p;
                break;
            }
        }
        if (self.pd == null) return fail(error.NotFound, "DLSS-FG: kein NVIDIA-Gerät unter Vulkan", .{});

        // Warteschlange mit Grafik und Rechnen
        var nq: u32 = 0;
        self.vk.vkGetPhysicalDeviceQueueFamilyProperties.?(self.pd, &nq, null);
        var qfs: [16]c.VkQueueFamilyProperties = undefined;
        nq = @min(nq, 16);
        self.vk.vkGetPhysicalDeviceQueueFamilyProperties.?(self.pd, &nq, &qfs);
        for (qfs[0..nq], 0..) |q, i| {
            if (q.queueFlags & c.VK_QUEUE_GRAPHICS_BIT != 0 and q.queueFlags & c.VK_QUEUE_COMPUTE_BIT != 0) {
                self.qfi = @intCast(i);
                break;
            }
        }

        // Geräte-Erweiterungen: was NGX für die Frame Generation will, dazu Interop
        const data = [_:0]c.wchar_t{ '/', 't', 'm', 'p' };
        var disc = std.mem.zeroes(c.NVSDK_NGX_FeatureDiscoveryInfo);
        disc.SDKVersion = c.NVSDK_NGX_Version_API;
        disc.FeatureID = c.NVSDK_NGX_Feature_FrameGeneration;
        disc.Identifier.IdentifierType = c.NVSDK_NGX_Application_Identifier_Type_Project_Id;
        disc.Identifier.v.ProjectDesc = .{ .ProjectId = project_id, .EngineType = c.NVSDK_NGX_ENGINE_TYPE_CUSTOM, .EngineVersion = "0.1" };
        disc.ApplicationDataPath = &data;
        var fi = featureInfo();
        disc.FeatureInfo = &fi.info;
        var n_ext: u32 = 0;
        var ext_props: [*c]c.VkExtensionProperties = null;
        _ = c.NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(self.instance, self.pd, &disc, &n_ext, &ext_props);
        var names: [24][*c]const u8 = undefined;
        var count: u32 = 0;
        for (0..@min(n_ext, 16)) |i| {
            names[count] = &ext_props[i].extensionName;
            count += 1;
        }
        for ([_][*:0]const u8{ "VK_KHR_external_memory", "VK_KHR_external_memory_fd", "VK_KHR_external_semaphore", "VK_KHR_external_semaphore_fd", "VK_KHR_timeline_semaphore", "VK_KHR_dedicated_allocation", "VK_KHR_get_memory_requirements2" }) |e| {
            var dup = false;
            for (names[0..count]) |d| {
                if (std.mem.eql(u8, std.mem.span(@as([*:0]const u8, @ptrCast(d))), std.mem.span(e))) dup = true;
            }
            if (!dup) {
                names[count] = e;
                count += 1;
            }
        }
        const prio: f32 = 1;
        const qci = c.VkDeviceQueueCreateInfo{ .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = self.qfi, .queueCount = 1, .pQueuePriorities = &prio };
        var f13 = std.mem.zeroes(c.VkPhysicalDeviceVulkan13Features);
        f13.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        f13.synchronization2 = c.VK_TRUE;
        var f12 = std.mem.zeroes(c.VkPhysicalDeviceVulkan12Features);
        f12.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        f12.timelineSemaphore = c.VK_TRUE;
        f12.bufferDeviceAddress = c.VK_TRUE;
        f12.pNext = &f13;
        const dci = c.VkDeviceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &f12,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &qci,
            .enabledExtensionCount = count,
            .ppEnabledExtensionNames = &names,
        };
        try vkCheck(self.vk.vkCreateDevice.?(self.pd, &dci, null, &self.device), "vkCreateDevice");
        inline for (std.meta.fields(Vk)) |f| {
            if (@field(self.vk, f.name) == null) {
                @field(self.vk, f.name) = @ptrCast(self.vk.vkGetDeviceProcAddr.?(self.device, f.name ++ ""));
            }
        }
        self.vk.vkGetDeviceQueue.?(self.device, self.qfi, 0, &self.queue);

        const pci = c.VkCommandPoolCreateInfo{ .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = self.qfi };
        try vkCheck(self.vk.vkCreateCommandPool.?(self.device, &pci, null, &self.pool), "vkCreateCommandPool");
        const cai = c.VkCommandBufferAllocateInfo{ .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = self.pool, .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = ring_size };
        try vkCheck(self.vk.vkAllocateCommandBuffers.?(self.device, &cai, &self.cmds), "vkAllocateCommandBuffers");
        for (&self.fences) |*f| {
            const fci = c.VkFenceCreateInfo{ .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = c.VK_FENCE_CREATE_SIGNALED_BIT };
            try vkCheck(self.vk.vkCreateFence.?(self.device, &fci, null, f), "vkCreateFence");
        }
    }

    const FeatureInfo = struct {
        wpath: [1024]c.wchar_t,
        paths: [2][*c]const c.wchar_t,
        dot: [2]c.wchar_t,
        info: c.NVSDK_NGX_FeatureCommonInfo,
    };

    /// Suchpfad der NGX-Bausteine (libnvidia-ngx-dlssg.so)
    fn featureInfo() FeatureInfo {
        var f: FeatureInfo = undefined;
        const dir: []const u8 = if (std.c.getenv("PYRIT_DLSS_PATH")) |e| std.mem.span(e) else build_options.dlss_lib_dir;
        const n = @min(dir.len, f.wpath.len - 1);
        for (dir[0..n], 0..) |ch, i| f.wpath[i] = ch;
        f.wpath[n] = 0;
        f.dot = .{ '.', 0 };
        f.paths = .{ &f.wpath, &f.dot };
        f.info = std.mem.zeroes(c.NVSDK_NGX_FeatureCommonInfo);
        f.info.PathListInfo.Path = @ptrCast(&f.paths);
        f.info.PathListInfo.Length = 2;
        f.info.LoggingInfo.MinimumLoggingLevel = c.NVSDK_NGX_LOGGING_LEVEL_OFF;
        // Diagnose PYRIT_NGX_LOG=1: NGX-Meldungen auf stderr (nicht in Dateien)
        if (std.c.getenv("PYRIT_NGX_LOG") != null) {
            f.info.LoggingInfo.MinimumLoggingLevel = c.NVSDK_NGX_LOGGING_LEVEL_VERBOSE;
            f.info.LoggingInfo.LoggingCallback = ngxLog;
            f.info.LoggingInfo.DisableOtherLoggingSinks = true;
        }
        return f;
    }

    fn ngxLog(msg: [*c]const u8, _: c.NVSDK_NGX_Logging_Level, _: c.NVSDK_NGX_Feature) callconv(.c) void {
        std.debug.print("[ngx] {s}", .{std.mem.span(msg)});
    }

    fn initNgx(self: *DlssG, ctx: *Context) Error!void {
        const data = [_:0]c.wchar_t{ '/', 't', 'm', 'p' };
        var fi = featureInfo();
        fi.info.PathListInfo.Path = @ptrCast(&fi.paths);
        fi.paths = .{ &fi.wpath, &fi.dot };
        const gdpa = self.vk.vkGetDeviceProcAddr;
        const r = c.NVSDK_NGX_VULKAN_Init_with_ProjectID(project_id, c.NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1", &data, self.instance, self.pd, self.device, self.gipa, gdpa, &fi.info, c.NVSDK_NGX_Version_API);
        if (!ngxOk(r)) return fail(error.NotFound, "DLSS-FG: NGX_VULKAN_Init 0x{x}", .{r});
        self.ngx_up = true;
        var caps: ?*c.NVSDK_NGX_Parameter = null;
        _ = c.NVSDK_NGX_VULKAN_GetCapabilityParameters(&caps);
        var avail: c_int = 0;
        _ = c.NVSDK_NGX_Parameter_GetI(caps, c.NVSDK_NGX_Parameter_FrameGeneration_Available, &avail);
        if (avail == 0) return fail(error.NotFound, "DLSS Frame Generation wird von GPU/Treiber nicht angeboten", .{});
        var mfc: c_uint = 0;
        if (ngxOk(c.NVSDK_NGX_Parameter_GetUI(caps, c.NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &mfc))) self.multi_max = @max(mfc, 1);
        ctx.logf(3, "Pyrit: DLSS Frame Generation über Vulkan-Interop, bis {d} Zwischenbilder je Frame", .{self.multi_max});
        if (!ngxOk(c.NVSDK_NGX_VULKAN_AllocateParameters(&self.params))) return fail(error.NotFound, "DLSS-FG: Parameter", .{});
    }

    /// Bilder und Feature für die Ausgabegröße anlegen (einmal, bei Größenwechsel neu)
    fn ensure(self: *DlssG, ctx: *Context, w: u32, h: u32, bgra: bool) Error!void {
        if (self.feature != null and self.width == w and self.height == h and self.bgra == bgra) return;
        try vkCheck(self.vk.vkDeviceWaitIdle.?(self.device), "vkDeviceWaitIdle");
        try ctx.check(ctx.drv.cuCtxSynchronize(), "cuCtxSynchronize");
        self.freeImages(ctx);
        if (self.feature) |f| {
            _ = c.NVSDK_NGX_VULKAN_ReleaseFeature(f);
            self.feature = null;
        }
        self.width = w;
        self.height = h;
        self.bgra = bgra;
        const color_fmt: c.VkFormat = if (bgra) c.VK_FORMAT_B8G8R8A8_UNORM else c.VK_FORMAT_R8G8B8A8_UNORM;
        try self.makeImage(ctx, &self.imgs[img_color], color_fmt, 4, cuda.CU_AD_FORMAT_UNSIGNED_INT8, 4);
        try self.makeImage(ctx, &self.imgs[img_mvec], c.VK_FORMAT_R32G32_SFLOAT, 8, cuda.CU_AD_FORMAT_FLOAT, 2);
        try self.makeImage(ctx, &self.imgs[img_depth], c.VK_FORMAT_R32_SFLOAT, 4, cuda.CU_AD_FORMAT_FLOAT, 1);
        try self.makeImage(ctx, &self.imgs[img_out], color_fmt, 4, cuda.CU_AD_FORMAT_UNSIGNED_INT8, 4);
        const n = @as(u64, w) * h;
        self.mv_lin = try ctx.devAlloc(n * 8, "DLSS-FG");
        self.depth_lin = try ctx.devAlloc(n * 4, "DLSS-FG");
        if (self.sem == null) try self.makeSemaphore(ctx);

        // Bilder einmal in GENERAL bringen und das Feature anlegen
        const cmd = self.cmds[0];
        try vkCheck(self.vk.vkWaitForFences.?(self.device, 1, &self.fences[0], c.VK_TRUE, std.math.maxInt(u64)), "vkWaitForFences");
        try vkCheck(self.vk.vkResetFences.?(self.device, 1, &self.fences[0]), "vkResetFences");
        const cbi = c.VkCommandBufferBeginInfo{ .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        try vkCheck(self.vk.vkBeginCommandBuffer.?(cmd, &cbi), "vkBeginCommandBuffer");
        self.barriers(cmd, c.VK_IMAGE_LAYOUT_UNDEFINED, c.VK_QUEUE_FAMILY_IGNORED, c.VK_QUEUE_FAMILY_IGNORED);
        const p = self.params;
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_Parameter_CreationNodeMask, 1);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_Parameter_VisibilityNodeMask, 1);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_Parameter_Width, w);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_Parameter_Height, h);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_DLSSG_Parameter_BackbufferFormat, @intCast(color_fmt));
        const r = c.NVSDK_NGX_VULKAN_CreateFeature1(self.device, cmd, c.NVSDK_NGX_Feature_FrameGeneration, p, &self.feature);
        try vkCheck(self.vk.vkEndCommandBuffer.?(cmd), "vkEndCommandBuffer");
        const si = c.VkSubmitInfo{ .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
        try vkCheck(self.vk.vkQueueSubmit.?(self.queue, 1, &si, self.fences[0]), "vkQueueSubmit");
        if (!ngxOk(r)) return fail(error.NotFound, "DLSS-FG: CreateFeature 0x{x}", .{r});
    }

    fn memoryType(self: *DlssG, bits: u32) u32 {
        var mp: c.VkPhysicalDeviceMemoryProperties = undefined;
        self.vk.vkGetPhysicalDeviceMemoryProperties.?(self.pd, &mp);
        for (0..mp.memoryTypeCount) |i| {
            if (bits & (@as(u32, 1) << @intCast(i)) != 0 and mp.memoryTypes[i].propertyFlags & c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0) return @intCast(i);
        }
        return 0;
    }

    /// Bild in Vulkan anlegen, Speicher als fd ausführen und in CUDA einbinden
    fn makeImage(self: *DlssG, ctx: *Context, img: *Img, format: c.VkFormat, bytes_px: u32, cu_fmt: c_int, channels: u32) Error!void {
        img.* = .{ .format = format, .bytes_px = bytes_px };
        const ext_img = c.VkExternalMemoryImageCreateInfo{ .sType = c.VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, .handleTypes = c.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT };
        const ici = c.VkImageCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = &ext_img,
            .imageType = c.VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = .{ .width = self.width, .height = self.height, .depth = 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = c.VK_SAMPLE_COUNT_1_BIT,
            .tiling = c.VK_IMAGE_TILING_OPTIMAL,
            .usage = c.VK_IMAGE_USAGE_SAMPLED_BIT | c.VK_IMAGE_USAGE_STORAGE_BIT | c.VK_IMAGE_USAGE_TRANSFER_SRC_BIT | c.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
        };
        try vkCheck(self.vk.vkCreateImage.?(self.device, &ici, null, &img.image), "vkCreateImage");
        var req: c.VkMemoryRequirements = undefined;
        self.vk.vkGetImageMemoryRequirements.?(self.device, img.image, &req);
        const ded = c.VkMemoryDedicatedAllocateInfo{ .sType = c.VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .image = img.image };
        const exp = c.VkExportMemoryAllocateInfo{ .sType = c.VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, .pNext = &ded, .handleTypes = c.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT };
        const mai = c.VkMemoryAllocateInfo{ .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &exp, .allocationSize = req.size, .memoryTypeIndex = self.memoryType(req.memoryTypeBits) };
        try vkCheck(self.vk.vkAllocateMemory.?(self.device, &mai, null, &img.mem), "vkAllocateMemory");
        try vkCheck(self.vk.vkBindImageMemory.?(self.device, img.image, img.mem, 0), "vkBindImageMemory");
        const vci = c.VkImageViewCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = img.image,
            .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .subresourceRange = .{ .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 },
        };
        try vkCheck(self.vk.vkCreateImageView.?(self.device, &vci, null, &img.view), "vkCreateImageView");

        // in CUDA einbinden (der fd geht dabei an CUDA über)
        var fd: c_int = -1;
        const gfd = c.VkMemoryGetFdInfoKHR{ .sType = c.VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = img.mem, .handleType = c.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT };
        try vkCheck(self.vk.vkGetMemoryFdKHR.?(self.device, &gfd, &fd), "vkGetMemoryFdKHR");
        const hd = cuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC{ .type = cuda.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD, .handle = .{ .fd = fd }, .size = req.size, .flags = cuda.CUDA_EXTERNAL_MEMORY_DEDICATED };
        try ctx.check(ctx.drv.cuImportExternalMemory(&img.ext, &hd), "cuImportExternalMemory");
        const md = cuda.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC{
            .offset = 0,
            .arrayDesc = .{ .Width = self.width, .Height = self.height, .Depth = 0, .Format = cu_fmt, .NumChannels = channels, .Flags = cuda.CUDA_ARRAY3D_SURFACE_LDST },
            .numLevels = 1,
        };
        try ctx.check(ctx.drv.cuExternalMemoryGetMappedMipmappedArray(&img.mip, img.ext, &md), "cuExternalMemoryGetMappedMipmappedArray");
        try ctx.check(ctx.drv.cuMipmappedArrayGetLevel(&img.arr, img.mip, 0), "cuMipmappedArrayGetLevel");
    }

    fn makeSemaphore(self: *DlssG, ctx: *Context) Error!void {
        const exp = c.VkExportSemaphoreCreateInfo{ .sType = c.VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO, .handleTypes = c.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT };
        const tci = c.VkSemaphoreTypeCreateInfo{ .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .pNext = &exp, .semaphoreType = c.VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0 };
        const sci = c.VkSemaphoreCreateInfo{ .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &tci };
        try vkCheck(self.vk.vkCreateSemaphore.?(self.device, &sci, null, &self.sem), "vkCreateSemaphore");
        var fd: c_int = -1;
        const gfd = c.VkSemaphoreGetFdInfoKHR{ .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR, .semaphore = self.sem, .handleType = c.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT };
        try vkCheck(self.vk.vkGetSemaphoreFdKHR.?(self.device, &gfd, &fd), "vkGetSemaphoreFdKHR");
        const hd = cuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC{ .type = cuda.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD, .handle = .{ .fd = fd }, .flags = 0 };
        try ctx.check(ctx.drv.cuImportExternalSemaphore(&self.sem_cuda, &hd), "cuImportExternalSemaphore");
    }

    /// Barrieren für alle vier Bilder: `from` -> GENERAL, dazu Übergabe
    /// zwischen externer Queue-Familie (CUDA) und unserer
    fn barriers(self: *DlssG, cmd: c.VkCommandBuffer, from: c.VkImageLayout, src_q: u32, dst_q: u32) void {
        var b: [4]c.VkImageMemoryBarrier = undefined;
        for (&b, self.imgs) |*x, img| {
            x.* = .{
                .sType = c.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = c.VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = c.VK_ACCESS_MEMORY_READ_BIT | c.VK_ACCESS_MEMORY_WRITE_BIT,
                .oldLayout = from,
                .newLayout = c.VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = src_q,
                .dstQueueFamilyIndex = dst_q,
                .image = img.image,
                .subresourceRange = .{ .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 },
            };
        }
        self.vk.vkCmdPipelineBarrier.?(cmd, c.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, c.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, null, 0, null, 4, &b);
    }

    fn resource(img: *const Img, w: u32, h: u32, rw: bool) c.NVSDK_NGX_Resource_VK {
        var r = std.mem.zeroes(c.NVSDK_NGX_Resource_VK);
        r.Resource.ImageViewInfo = .{
            .ImageView = img.view,
            .Image = img.image,
            .SubresourceRange = .{ .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 },
            .Format = img.format,
            .Width = w,
            .Height = h,
        };
        r.Type = c.NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW;
        r.ReadWrite = rw;
        return r;
    }

    fn copy2d(ctx: *Context, src_dev: u64, src_arr: cuda.CUarray, dst_dev: u64, dst_arr: cuda.CUarray, row_bytes: u64, rows: u64) Error!void {
        var m = std.mem.zeroes(cuda.CUDA_MEMCPY2D);
        if (src_arr != null) {
            m.srcMemoryType = cuda.CU_MEMORYTYPE_ARRAY;
            m.srcArray = src_arr;
        } else {
            m.srcMemoryType = cuda.CU_MEMORYTYPE_DEVICE;
            m.srcDevice = src_dev;
            m.srcPitch = row_bytes;
        }
        if (dst_arr != null) {
            m.dstMemoryType = cuda.CU_MEMORYTYPE_ARRAY;
            m.dstArray = dst_arr;
        } else {
            m.dstMemoryType = cuda.CU_MEMORYTYPE_DEVICE;
            m.dstDevice = dst_dev;
            m.dstPitch = row_bytes;
        }
        m.WidthInBytes = row_bytes;
        m.Height = rows;
        try ctx.check(ctx.drv.cuMemcpy2DAsync_v2(&m, ctx.activeStream()), "cuMemcpy2DAsync");
    }

    /// Ein Zwischenbild erzeugen (zwischen dem vorigen und diesem Frame)
    pub fn evaluate(self: *DlssG, ctx: *Context, in: *const Inputs) Error!void {
        try self.ensure(ctx, in.width, in.height, in.bgra);
        const w = in.width;
        const h = in.height;
        const n = @as(u64, w) * h;

        // 1. CUDA: Bewegung und Tiefe umformen, alles in die geteilten Bilder
        // (bei weiteren Zwischenbildern desselben Frames liegen sie schon dort)
        if (in.multi_index == 1) {
            var mvd = in.motion_depth;
            var mv = self.mv_lin;
            var dp = self.depth_lin;
            var cnt: u32 = @intCast(n);
            var near = in.near;
            const args = [_]?*anyopaque{ @ptrCast(&mvd), @ptrCast(&mv), @ptrCast(&dp), @ptrCast(&cnt), @ptrCast(&near) };
            try ctx.launch(ctx.fn_dlssg_prepare, .{ @intCast((n + types.update_block - 1) / types.update_block), 1, 1 }, .{ types.update_block, 1, 1 }, &args);
            try copy2d(ctx, in.ldr, null, 0, self.imgs[img_color].arr, @as(u64, w) * 4, h);
            try copy2d(ctx, self.mv_lin, null, 0, self.imgs[img_mvec].arr, @as(u64, w) * 8, h);
            try copy2d(ctx, self.depth_lin, null, 0, self.imgs[img_depth].arr, @as(u64, w) * 4, h);
        }
        // Diagnose PYRIT_DLSSG_DUMP=<Datei>: das geteilte Eingangsbild (so wie
        // NGX es liest) als rohes BGRA anhängen
        if (std.c.getenv("PYRIT_DLSSG_DUMP")) |path| {
            try ctx.check(ctx.drv.cuStreamSynchronize(ctx.activeStream()), "cuStreamSynchronize");
            const bytes = @as(usize, w) * h * 4;
            const hb = ctx.gpa.alloc(u8, bytes) catch return;
            defer ctx.gpa.free(hb);
            var m = std.mem.zeroes(cuda.CUDA_MEMCPY2D);
            m.srcMemoryType = cuda.CU_MEMORYTYPE_ARRAY;
            m.srcArray = self.imgs[img_color].arr;
            m.dstMemoryType = cuda.CU_MEMORYTYPE_HOST;
            m.dstHost = hb.ptr;
            m.dstPitch = @as(usize, w) * 4;
            m.WidthInBytes = @as(usize, w) * 4;
            m.Height = h;
            try ctx.check(ctx.drv.cuMemcpy2DAsync_v2(&m, ctx.activeStream()), "cuMemcpy2D");
            try ctx.check(ctx.drv.cuStreamSynchronize(ctx.activeStream()), "cuStreamSynchronize");
            if (std.c.fopen(path, "ab")) |f| {
                _ = std.c.fwrite(hb.ptr, 1, bytes, f);
                _ = std.c.fclose(f);
            }
        }
        if (std.c.getenv("PYRIT_DLSSG_DEBUG") != null and in.multi_index == 1) {
            // Diagnose: Bewegung und Tiefe, die NGX bekommt (Mittelwerte)
            try ctx.check(ctx.drv.cuStreamSynchronize(ctx.activeStream()), "cuStreamSynchronize");
            const cnt: usize = @intCast(n);
            const mvh = ctx.gpa.alloc([2]f32, cnt) catch return;
            defer ctx.gpa.free(mvh);
            const dh = ctx.gpa.alloc(f32, cnt) catch return;
            defer ctx.gpa.free(dh);
            try ctx.check(ctx.drv.cuMemcpyDtoH_v2(mvh.ptr, self.mv_lin, cnt * 8), "cuMemcpyDtoH");
            try ctx.check(ctx.drv.cuMemcpyDtoH_v2(dh.ptr, self.depth_lin, cnt * 4), "cuMemcpyDtoH");
            var sx: f64 = 0;
            var sy: f64 = 0;
            var sa: f64 = 0;
            var dmin: f32 = 1e30;
            var dmax: f32 = -1e30;
            for (mvh, dh) |m, d| {
                sx += m[0];
                sy += m[1];
                sa += @abs(m[0]) + @abs(m[1]);
                dmin = @min(dmin, d);
                dmax = @max(dmax, d);
            }
            const nf: f64 = @floatFromInt(cnt);
            std.debug.print("[dlssg] MV Mittel ({d:.3}, {d:.3}) px, |MV| {d:.3}, Tiefe {d:.5}..{d:.5}\n", .{ sx / nf, sy / nf, sa / nf, dmin, dmax });
        }
        self.value += 1;
        const sig = [_]cuda.CUDA_EXTERNAL_SEMAPHORE_PARAMS{.{ .fence_value = self.value }};
        const sems = [_]cuda.CUexternalSemaphore{self.sem_cuda};
        try ctx.check(ctx.drv.cuSignalExternalSemaphoresAsync(&sems, &sig, 1, ctx.activeStream()), "cuSignalExternalSemaphoresAsync");

        // 2. Vulkan: NGX erzeugt das Zwischenbild
        const k = self.ring;
        self.ring = (self.ring + 1) % ring_size;
        const cmd = self.cmds[k];
        try vkCheck(self.vk.vkWaitForFences.?(self.device, 1, &self.fences[k], c.VK_TRUE, std.math.maxInt(u64)), "vkWaitForFences");
        try vkCheck(self.vk.vkResetFences.?(self.device, 1, &self.fences[k]), "vkResetFences");
        try vkCheck(self.vk.vkResetCommandBuffer.?(cmd, 0), "vkResetCommandBuffer");
        const cbi = c.VkCommandBufferBeginInfo{ .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        try vkCheck(self.vk.vkBeginCommandBuffer.?(cmd, &cbi), "vkBeginCommandBuffer");
        // von CUDA übernehmen
        self.barriers(cmd, c.VK_IMAGE_LAYOUT_GENERAL, c.VK_QUEUE_FAMILY_EXTERNAL, self.qfi);

        var r_color = resource(&self.imgs[img_color], w, h, false);
        var r_mv = resource(&self.imgs[img_mvec], w, h, false);
        var r_depth = resource(&self.imgs[img_depth], w, h, false);
        var r_out = resource(&self.imgs[img_out], w, h, true);
        const p = self.params;
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_DLSSG_Parameter_Backbuffer, &r_color);
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_DLSSG_Parameter_MVecs, &r_mv);
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_DLSSG_Parameter_Depth, &r_depth);
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_DLSSG_Parameter_OutputInterpolated, &r_out);
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_DLSSG_Parameter_HUDLess, null);
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_DLSSG_Parameter_UI, null);
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_DLSSG_Parameter_OutputReal, null);
        // Kamera: Matrizen in Zeilenvektor-Form, Tiefe umgekehrt unendlich
        var v2c = in.view_to_clip;
        var c2v = invert4(in.view_to_clip);
        // clip -> Vorframe-clip: inv(V2C) · inv(W2V) · W2V_prev · V2C_prev
        var c2p = mul4(mul4(mul4(c2v, invert4(in.world_to_view)), in.prev_world_to_view), in.prev_view_to_clip);
        var p2c = invert4(c2p);
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_DLSSG_Parameter_CameraViewToClip, &v2c);
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_DLSSG_Parameter_ClipToCameraView, &c2v);
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_DLSSG_Parameter_ClipToPrevClip, &c2p);
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_DLSSG_Parameter_PrevClipToClip, &p2c);
        c.NVSDK_NGX_Parameter_SetVoidPointer(p, c.NVSDK_NGX_DLSSG_Parameter_ClipToLensClip, null);
        const v2w = invert4(in.world_to_view);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraPosX, v2w[12]);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraPosY, v2w[13]);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraPosZ, v2w[14]);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraRightX, v2w[0]);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraRightY, v2w[1]);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraRightZ, v2w[2]);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraUpX, v2w[4]);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraUpY, v2w[5]);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraUpZ, v2w[6]);
        // Blick entlang -z
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraFwdX, -v2w[8]);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraFwdY, -v2w[9]);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraFwdZ, -v2w[10]);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraNear, in.near);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraFar, 1e6);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraFOV, in.fov_y);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_CameraAspectRatio, @as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(h)));
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_JitterOffsetX, 0);
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_JitterOffsetY, 0);
        // Bewegung in Pixeln -> Anteil des Bildes
        // Diagnose: PYRIT_DLSSG_MV = Vorzeichen (-1), PYRIT_DLSSG_MVPIX = Pixel statt Bildanteil
        const msign: f32 = if (std.c.getenv("PYRIT_DLSSG_MV")) |e| (if (e[0] == '-') -1 else 1) else 1;
        const pix = std.c.getenv("PYRIT_DLSSG_MVPIX") != null;
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_MvecScaleX, msign * (if (pix) 1 else 1.0 / @as(f32, @floatFromInt(w))));
        c.NVSDK_NGX_Parameter_SetF(p, c.NVSDK_NGX_DLSSG_Parameter_MvecScaleY, msign * (if (pix) 1 else 1.0 / @as(f32, @floatFromInt(h))));
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_DLSSG_Parameter_ColorBuffersHDR, 0);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_DLSSG_Parameter_DepthInverted, 1);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_DLSSG_Parameter_CameraMotionIncluded, 1);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_DLSSG_Parameter_Reset, @intFromBool(in.reset));
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_DLSSG_Parameter_NotRenderingGameFrames, 0);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_DLSSG_Parameter_MultiFrameCount, in.multi_count);
        c.NVSDK_NGX_Parameter_SetUI(p, c.NVSDK_NGX_DLSSG_Parameter_MultiFrameIndex, in.multi_index);
        const r = c.NVSDK_NGX_VULKAN_EvaluateFeature_C(cmd, self.feature, p, null);
        // an CUDA zurückgeben
        self.barriers(cmd, c.VK_IMAGE_LAYOUT_GENERAL, self.qfi, c.VK_QUEUE_FAMILY_EXTERNAL);
        try vkCheck(self.vk.vkEndCommandBuffer.?(cmd), "vkEndCommandBuffer");
        if (!ngxOk(r)) return fail(error.NotFound, "DLSS-FG: EvaluateFeature 0x{x}", .{r});

        const wait_v = self.value;
        self.value += 1;
        const signal_v = self.value;
        const tsi = c.VkTimelineSemaphoreSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = &wait_v,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &signal_v,
        };
        const stage: c.VkPipelineStageFlags = c.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        const si = c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &tsi,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &self.sem,
            .pWaitDstStageMask = &stage,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &self.sem,
        };
        try vkCheck(self.vk.vkQueueSubmit.?(self.queue, 1, &si, self.fences[k]), "vkQueueSubmit");

        // Diagnose PYRIT_DLSSG_IDLE: Host wartet, bis Vulkan ganz fertig ist
        if (std.c.getenv("PYRIT_DLSSG_IDLE") != null) {
            try ctx.check(ctx.drv.cuCtxSynchronize(), "cuCtxSynchronize");
            try vkCheck(self.vk.vkDeviceWaitIdle.?(self.device), "vkDeviceWaitIdle");
        }
        // 3. CUDA: auf NGX warten, Zwischenbild in die Ausgabe
        const wt = [_]cuda.CUDA_EXTERNAL_SEMAPHORE_PARAMS{.{ .fence_value = signal_v }};
        try ctx.check(ctx.drv.cuWaitExternalSemaphoresAsync(&sems, &wt, 1, ctx.activeStream()), "cuWaitExternalSemaphoresAsync");
        try copy2d(ctx, 0, self.imgs[img_out].arr, in.out, null, @as(u64, w) * 4, h);
    }

    fn freeImages(self: *DlssG, ctx: *Context) void {
        for (&self.imgs) |*img| {
            if (img.mip != null) _ = ctx.drv.cuMipmappedArrayDestroy(img.mip);
            if (img.ext != null) _ = ctx.drv.cuDestroyExternalMemory(img.ext);
            if (img.view != null) self.vk.vkDestroyImageView.?(self.device, img.view, null);
            if (img.image != null) self.vk.vkDestroyImage.?(self.device, img.image, null);
            if (img.mem != null) self.vk.vkFreeMemory.?(self.device, img.mem, null);
            img.* = .{};
        }
        for ([_]*cuda.CUdeviceptr{ &self.mv_lin, &self.depth_lin }) |b| {
            if (b.* != 0) _ = ctx.drv.cuMemFree_v2(b.*);
            b.* = 0;
        }
    }

    fn destroyVulkan(self: *DlssG) void {
        if (self.device != null) {
            _ = self.vk.vkDeviceWaitIdle.?(self.device);
            for (self.fences) |f| if (f != null) self.vk.vkDestroyFence.?(self.device, f, null);
            if (self.pool != null) self.vk.vkDestroyCommandPool.?(self.device, self.pool, null);
            if (self.sem != null) self.vk.vkDestroySemaphore.?(self.device, self.sem, null);
            self.vk.vkDestroyDevice.?(self.device, null);
        }
        if (self.instance != null) self.vk.vkDestroyInstance.?(self.instance, null);
        self.lib.close();
    }

    pub fn destroy(self: *DlssG, ctx: *Context) void {
        if (self.device != null) _ = self.vk.vkDeviceWaitIdle.?(self.device);
        _ = ctx.drv.cuCtxSynchronize();
        if (self.feature) |f| _ = c.NVSDK_NGX_VULKAN_ReleaseFeature(f);
        if (self.params) |p| _ = c.NVSDK_NGX_VULKAN_DestroyParameters(p);
        if (self.ngx_up) _ = c.NVSDK_NGX_VULKAN_Shutdown1(self.device);
        self.freeImages(ctx);
        if (self.sem_cuda != null) _ = ctx.drv.cuDestroyExternalSemaphore(self.sem_cuda);
        self.destroyVulkan();
        ctx.gpa.destroy(self);
    }
};

// --- 4x4-Matrizen (Zeilenvektor-Form, zeilenweise) ---------------------------

fn mul4(a: [16]f32, b: [16]f32) [16]f32 {
    var r: [16]f32 = undefined;
    for (0..4) |i| for (0..4) |j| {
        var s: f32 = 0;
        for (0..4) |k| s += a[i * 4 + k] * b[k * 4 + j];
        r[i * 4 + j] = s;
    };
    return r;
}

fn invert4(m: [16]f32) [16]f32 {
    // Gauß-Jordan mit Pivotsuche, in f64
    var a: [4][8]f64 = undefined;
    for (0..4) |i| for (0..4) |j| {
        a[i][j] = m[i * 4 + j];
        a[i][j + 4] = if (i == j) 1 else 0;
    };
    for (0..4) |col| {
        var piv = col;
        for (col + 1..4) |r| {
            if (@abs(a[r][col]) > @abs(a[piv][col])) piv = r;
        }
        const tmp = a[col];
        a[col] = a[piv];
        a[piv] = tmp;
        const d = a[col][col];
        if (@abs(d) < 1e-20) return m;
        for (0..8) |j| a[col][j] /= d;
        for (0..4) |r| {
            if (r == col) continue;
            const f = a[r][col];
            for (0..8) |j| a[r][j] -= f * a[col][j];
        }
    }
    var r: [16]f32 = undefined;
    for (0..4) |i| for (0..4) |j| {
        r[i * 4 + j] = @floatCast(a[i][j + 4]);
    };
    return r;
}
