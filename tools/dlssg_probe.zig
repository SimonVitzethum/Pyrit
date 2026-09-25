//! Machbarkeitsprobe: DLSS Frame Generation über NGX-Vulkan, kopflos.
//!
//!   zig build dlssg-probe -Ddlss-sdk=$PWD/DLSS
//!
//! Über den CUDA-Weg bietet NGX die Frame Generation nicht an (die
//! CUDA-Einsprünge in libnvidia-ngx-dlssg.so sind Stummel). Vulkan dient hier
//! nur als Interop-Schicht: kein Fenster, keine Swapchain, keine Pipeline.
//! Die Probe prüft Schritt für Schritt, wie weit NGX kommt:
//! Erweiterungen, Init, Verfügbarkeit, Anlegen des Features.

const std = @import("std");
const c = @import("ngxvk");

const project_id = "5b3e1c2a-7d4f-4e8a-9b61-707972697400";

fn ok(r: c_uint) bool {
    return r & 0xFFF0_0000 != 0xBAD0_0000;
}

fn vk(r: c.VkResult, what: []const u8) !void {
    if (r != c.VK_SUCCESS) {
        std.debug.print("Vulkan: {s} = {d}\n", .{ what, r });
        return error.Vulkan;
    }
}

pub fn main() !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    const gpa = gpa_state.allocator();

    // Pfad der NGX-Snippets (libnvidia-ngx-dlssg.so) als wchar_t
    const dir: []const u8 = @import("build_options").dlss_lib_dir;
    var wpath: [1024]c.wchar_t = undefined;
    for (dir, 0..) |ch, i| wpath[i] = ch;
    wpath[dir.len] = 0;
    const dot = [_:0]c.wchar_t{'.'};
    const paths = [_][*c]const c.wchar_t{ &wpath, &dot };
    var fi = std.mem.zeroes(c.NVSDK_NGX_FeatureCommonInfo);
    fi.PathListInfo.Path = @ptrCast(@constCast(&paths));
    fi.PathListInfo.Length = paths.len;
    fi.LoggingInfo.MinimumLoggingLevel = c.NVSDK_NGX_LOGGING_LEVEL_OFF;
    const data = [_:0]c.wchar_t{ '/', 't', 'm', 'p' };

    var disc = std.mem.zeroes(c.NVSDK_NGX_FeatureDiscoveryInfo);
    disc.SDKVersion = c.NVSDK_NGX_Version_API;
    disc.FeatureID = c.NVSDK_NGX_Feature_FrameGeneration;
    disc.Identifier.IdentifierType = c.NVSDK_NGX_Application_Identifier_Type_Project_Id;
    disc.Identifier.v.ProjectDesc = .{ .ProjectId = project_id, .EngineType = c.NVSDK_NGX_ENGINE_TYPE_CUSTOM, .EngineVersion = "0.1" };
    disc.ApplicationDataPath = &data;
    disc.FeatureInfo = &fi;

    // 1. Instanz-Erweiterungen, die die Frame Generation will
    var n_iext: u32 = 0;
    var iext_props: [*c]c.VkExtensionProperties = null;
    const r_ie = c.NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&disc, &n_iext, &iext_props);
    std.debug.print("1. Instanz-Erweiterungen: 0x{x}, {d} Stück\n", .{ r_ie, n_iext });
    var inames: std.ArrayList([*c]const u8) = .empty;
    defer inames.deinit(gpa);
    for (0..n_iext) |i| {
        std.debug.print("     {s}\n", .{std.mem.span(@as([*:0]const u8, @ptrCast(&iext_props[i].extensionName)))});
        try inames.append(gpa, &iext_props[i].extensionName);
    }
    try inames.append(gpa, "VK_KHR_external_memory_capabilities");
    try inames.append(gpa, "VK_KHR_external_semaphore_capabilities");
    try inames.append(gpa, "VK_KHR_get_physical_device_properties2");

    const app = c.VkApplicationInfo{
        .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "pyrit-dlssg-probe",
        .apiVersion = c.VK_API_VERSION_1_3,
    };
    const ici = c.VkInstanceCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
        .enabledExtensionCount = @intCast(inames.items.len),
        .ppEnabledExtensionNames = inames.items.ptr,
    };
    var instance: c.VkInstance = null;
    try vk(c.vkCreateInstance(&ici, null, &instance), "vkCreateInstance");

    // 2. NVIDIA-Gerät
    var n_pd: u32 = 0;
    try vk(c.vkEnumeratePhysicalDevices(instance, &n_pd, null), "vkEnumeratePhysicalDevices");
    var pds: [8]c.VkPhysicalDevice = undefined;
    n_pd = @min(n_pd, 8);
    try vk(c.vkEnumeratePhysicalDevices(instance, &n_pd, &pds), "vkEnumeratePhysicalDevices");
    var pd: c.VkPhysicalDevice = null;
    for (pds[0..n_pd]) |p| {
        var props: c.VkPhysicalDeviceProperties = undefined;
        c.vkGetPhysicalDeviceProperties(p, &props);
        if (props.vendorID == 0x10de) {
            pd = p;
            std.debug.print("2. Gerät: {s}\n", .{std.mem.span(@as([*:0]const u8, @ptrCast(&props.deviceName)))});
        }
    }
    if (pd == null) return error.KeinNvidiaGeraet;

    // 3. Anforderungen und Geräte-Erweiterungen
    var req = std.mem.zeroes(c.NVSDK_NGX_FeatureRequirement);
    const r_req = c.NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, pd, &disc, &req);
    std.debug.print("3. Anforderungen: 0x{x}, FeatureSupported = 0x{x} (0 = unterstützt), MinHWArchitecture {d}\n", .{ r_req, req.FeatureSupported, req.MinHWArchitecture });
    var n_dext: u32 = 0;
    var dext_props: [*c]c.VkExtensionProperties = null;
    const r_de = c.NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, pd, &disc, &n_dext, &dext_props);
    std.debug.print("   Geräte-Erweiterungen: 0x{x}, {d} Stück\n", .{ r_de, n_dext });
    var dnames: std.ArrayList([*c]const u8) = .empty;
    defer dnames.deinit(gpa);
    for (0..n_dext) |i| {
        std.debug.print("     {s}\n", .{std.mem.span(@as([*:0]const u8, @ptrCast(&dext_props[i].extensionName)))});
        try dnames.append(gpa, &dext_props[i].extensionName);
    }
    for ([_][*c]const u8{ "VK_KHR_external_memory", "VK_KHR_external_memory_fd", "VK_KHR_external_semaphore", "VK_KHR_external_semaphore_fd", "VK_KHR_timeline_semaphore" }) |e| {
        var dup = false;
        for (dnames.items) |d| {
            if (std.mem.eql(u8, std.mem.span(d), std.mem.span(e))) dup = true;
        }
        if (!dup) try dnames.append(gpa, e);
    }

    // Warteschlange mit Grafik und Rechnen
    var n_qf: u32 = 0;
    c.vkGetPhysicalDeviceQueueFamilyProperties(pd, &n_qf, null);
    var qfs: [16]c.VkQueueFamilyProperties = undefined;
    n_qf = @min(n_qf, 16);
    c.vkGetPhysicalDeviceQueueFamilyProperties(pd, &n_qf, &qfs);
    var qfi: u32 = 0;
    for (qfs[0..n_qf], 0..) |q, i| {
        if (q.queueFlags & c.VK_QUEUE_GRAPHICS_BIT != 0 and q.queueFlags & c.VK_QUEUE_COMPUTE_BIT != 0) {
            qfi = @intCast(i);
            break;
        }
    }
    const prio: f32 = 1;
    const qci = c.VkDeviceQueueCreateInfo{ .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qfi, .queueCount = 1, .pQueuePriorities = &prio };
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
        .enabledExtensionCount = @intCast(dnames.items.len),
        .ppEnabledExtensionNames = dnames.items.ptr,
    };
    var device: c.VkDevice = null;
    try vk(c.vkCreateDevice(pd, &dci, null, &device), "vkCreateDevice");
    std.debug.print("   Gerät angelegt ({d} Erweiterungen)\n", .{dnames.items.len});

    // 4. NGX über Vulkan
    const r_init = c.NVSDK_NGX_VULKAN_Init_with_ProjectID(project_id, c.NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1", &data, instance, pd, device, c.vkGetInstanceProcAddr, c.vkGetDeviceProcAddr, &fi, c.NVSDK_NGX_Version_API);
    std.debug.print("4. NGX_VULKAN_Init: 0x{x}\n", .{r_init});
    if (!ok(r_init)) return;
    var params: ?*c.NVSDK_NGX_Parameter = null;
    _ = c.NVSDK_NGX_VULKAN_GetCapabilityParameters(&params);
    var avail: c_int = -1;
    var init_r: c_int = -1;
    _ = c.NVSDK_NGX_Parameter_GetI(params, c.NVSDK_NGX_Parameter_FrameGeneration_Available, &avail);
    _ = c.NVSDK_NGX_Parameter_GetI(params, c.NVSDK_NGX_Parameter_FrameGeneration_FeatureInitResult, &init_r);
    std.debug.print("5. FrameGeneration.Available = {d}, InitResult = 0x{x}\n", .{ avail, @as(u32, @bitCast(init_r)) });

    // 6. Feature anlegen (in einem Befehlspuffer)
    var queue: c.VkQueue = null;
    c.vkGetDeviceQueue(device, qfi, 0, &queue);
    var pool: c.VkCommandPool = null;
    const pci = c.VkCommandPoolCreateInfo{ .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qfi };
    try vk(c.vkCreateCommandPool(device, &pci, null, &pool), "vkCreateCommandPool");
    var cmd: c.VkCommandBuffer = null;
    const cai = c.VkCommandBufferAllocateInfo{ .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    try vk(c.vkAllocateCommandBuffers(device, &cai, &cmd), "vkAllocateCommandBuffers");
    const cbi = c.VkCommandBufferBeginInfo{ .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    try vk(c.vkBeginCommandBuffer(cmd, &cbi), "vkBeginCommandBuffer");

    var fp: ?*c.NVSDK_NGX_Parameter = null;
    _ = c.NVSDK_NGX_VULKAN_AllocateParameters(&fp);
    c.NVSDK_NGX_Parameter_SetUI(fp, c.NVSDK_NGX_Parameter_Width, 1280);
    c.NVSDK_NGX_Parameter_SetUI(fp, c.NVSDK_NGX_Parameter_Height, 720);
    c.NVSDK_NGX_Parameter_SetUI(fp, c.NVSDK_NGX_DLSSG_Parameter_Width, 1280);
    c.NVSDK_NGX_Parameter_SetUI(fp, c.NVSDK_NGX_DLSSG_Parameter_Height, 720);
    c.NVSDK_NGX_Parameter_SetUI(fp, c.NVSDK_NGX_DLSSG_Parameter_BackbufferFormat, c.VK_FORMAT_R8G8B8A8_UNORM);
    c.NVSDK_NGX_Parameter_SetUI(fp, c.NVSDK_NGX_Parameter_CreationNodeMask, 1);
    c.NVSDK_NGX_Parameter_SetUI(fp, c.NVSDK_NGX_Parameter_VisibilityNodeMask, 1);
    var handle: ?*c.NVSDK_NGX_Handle = null;
    const r_cf = c.NVSDK_NGX_VULKAN_CreateFeature1(device, cmd, c.NVSDK_NGX_Feature_FrameGeneration, fp, &handle);
    std.debug.print("6. VULKAN_CreateFeature1(FrameGeneration) = 0x{x}{s}\n", .{ r_cf, if (ok(r_cf)) "  -> Frame Generation angelegt" else "" });
    try vk(c.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
    const si = c.VkSubmitInfo{ .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
    try vk(c.vkQueueSubmit(queue, 1, &si, null), "vkQueueSubmit");
    try vk(c.vkQueueWaitIdle(queue), "vkQueueWaitIdle");
    if (ok(r_cf)) _ = c.NVSDK_NGX_VULKAN_ReleaseFeature(handle);
    _ = c.NVSDK_NGX_VULKAN_DestroyParameters(fp);
    _ = c.NVSDK_NGX_VULKAN_Shutdown1(device);
    c.vkDestroyCommandPool(device, pool, null);
    c.vkDestroyDevice(device, null);
    c.vkDestroyInstance(instance, null);
}
