using System.Diagnostics;
using System.IO.Compression;
using System.Net.Http;
using System.Net.Sockets;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace OVMS.Manager;

/// <summary>
/// Runtime status snapshot read from runtime.json.
/// </summary>
internal readonly struct RuntimeStatus
{
    public bool Running { get; init; }
    public int Pid { get; init; }
    public string Owner { get; init; }
}

/// <summary>
/// Result of a health check against the REST /v3/models endpoint.
/// </summary>
internal readonly struct HealthResult
{
    public bool Ok { get; init; }
    public int ModelCount { get; init; }
    public string Message { get; init; }
}

internal readonly struct PackageVersionInfo
{
    public string BasePackageVersion { get; init; }
    public string ModelServerVersion { get; init; }
    public string GenAiVersion { get; init; }
    public string PackageVariant { get; init; }
    public string ManagerVersion { get; init; }
}

internal readonly struct UpdateCheckResult
{
    public string BasePackageVersion { get; init; }
    public string LatestBasePackageVersion { get; init; }
    public string ModelServerVersion { get; init; }
    public string LatestModelServerVersion { get; init; }
    public string GenAiVersion { get; init; }
    public string LatestGenAiVersion { get; init; }
    public string PackageVariant { get; init; }
    public bool BasePackageUpdateAvailable { get; init; }
    public bool ModelServerUpdateAvailable { get; init; }
    public bool GenAiUpdateAvailable { get; init; }
    public bool UpdateAvailable { get; init; }
    public bool LatestIsOlderThanInstalled { get; init; }
    public string ModelServerReleaseUrl { get; init; }
    public string GenAiReleaseUrl { get; init; }
    public string Message { get; init; }
}

internal readonly struct ReleaseMetadata
{
    public string Version { get; init; }
    public string Url { get; init; }
    public string PackageUrl { get; init; }
    public string Sha256Url { get; init; }
}

/// <summary>
/// Pure logic layer for the Manager: settings persistence, process control,
/// health checks and diagnostics. Contains no UI code so it can be safely
/// invoked from background threads.
/// </summary>
internal sealed class OvmsController : IDisposable
{
    private static readonly JsonSerializerOptions ReadOptions = new() { PropertyNameCaseInsensitive = true };
    private static readonly JsonSerializerOptions WriteOptions = new() { WriteIndented = true };

    private readonly HttpClient httpClient = new() { Timeout = TimeSpan.FromSeconds(3) };

    public string DataDir { get; } = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "OVMS");

    public string AppDir { get; } = AppContext.BaseDirectory;

    public string SettingsPath => Path.Combine(DataDir, "settings.json");

    public string RuntimePath => Path.Combine(DataDir, "runtime.json");

    public string InstallMarkerPath => Path.Combine(DataDir, "install.json");

    public OvmsSettings LoadSettings()
    {
        if (!File.Exists(SettingsPath))
        {
            var defaults = new OvmsSettings
            {
                InstallDir = AppDir.TrimEnd(Path.DirectorySeparatorChar),
                DataDir = DataDir,
                ModelRepositoryPath = Path.Combine(DataDir, "models"),
                ConfigPath = Path.Combine(DataDir, "models", "config.json"),
                LogPath = Path.Combine(DataDir, "logs", "ovms_server.log"),
                BindAddress = "127.0.0.1",
                RestPort = 8000,
                GrpcPort = 0,
                LogLevel = "INFO",
                RunMode = "user-login",
                StartAtLogin = true,
                ShowTrayIcon = true,
                ServiceAutoStart = false,
                PackageVariant = "python_on"
            };
            return defaults;
        }

        var json = File.ReadAllText(SettingsPath);
        var settings = JsonSerializer.Deserialize<OvmsSettings>(json, ReadOptions);
        return settings ?? throw new InvalidOperationException($"Could not parse settings: {SettingsPath}");
    }

    /// <summary>
    /// Atomically persists settings: writes to a temp file, validates it
    /// round-trips, backs up the previous file, then replaces it.
    /// </summary>
    public void SaveSettings(OvmsSettings settings)
    {
        Directory.CreateDirectory(DataDir);

        var json = JsonSerializer.Serialize(settings, WriteOptions);

        // Validate the JSON round-trips before touching anything on disk.
        var roundTrip = JsonSerializer.Deserialize<OvmsSettings>(json, ReadOptions)
            ?? throw new InvalidOperationException("Settings failed to round-trip; refusing to save.");
        _ = roundTrip;

        var tmpPath = SettingsPath + ".tmp";
        var bakPath = SettingsPath + ".bak";

        File.WriteAllText(tmpPath, json);

        if (File.Exists(SettingsPath))
        {
            File.Replace(tmpPath, SettingsPath, bakPath);
        }
        else
        {
            File.Move(tmpPath, SettingsPath);
        }
    }

    public RuntimeStatus GetRuntimeStatus()
    {
        if (!File.Exists(RuntimePath))
        {
            return new RuntimeStatus { Running = false, Pid = 0, Owner = "" };
        }

        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(RuntimePath));
            var root = doc.RootElement;
            var owner = root.TryGetProperty("owner", out var ownerEl) ? ownerEl.GetString() ?? "" : "";

            if (root.TryGetProperty("pid", out var pidEl) && pidEl.TryGetInt32(out var pid))
            {
                var process = Process.GetProcessById(pid);
                if (string.Equals(process.ProcessName, "ovms", StringComparison.OrdinalIgnoreCase))
                {
                    return new RuntimeStatus { Running = true, Pid = pid, Owner = owner };
                }
            }
        }
        catch
        {
            // Stale or unreadable runtime file: treat as not running.
        }

        return new RuntimeStatus { Running = false, Pid = 0, Owner = "" };
    }

    public void Start()
    {
        RunPowerShell("start-ovms.ps1");
    }

    public void Stop()
    {
        RunPowerShell("stop-ovms.ps1");
    }

    public void Restart()
    {
        Stop();
        Start();
    }

    public string Repair()
    {
        var settings = LoadSettings();
        return RunPowerShell("repair-package.ps1", $"-InstallDir \"{settings.InstallDir}\" -DataDir \"{DataDir}\"", includeDataDirArg: false);
    }

    public string ValidateEnvironment()
    {
        var settings = LoadSettings();
        return RunPowerShell("validate-install.ps1", $"-InstallDir \"{settings.InstallDir}\"", includeDataDirArg: false);
    }

    public PackageVersionInfo GetPackageVersionInfo()
    {
        var settings = LoadSettings();
        var versionOutput = GetOvmsVersionOutput(settings.InstallDir).Trim();
        var parsed = ParseVersionOutput(versionOutput);
        var basePackageVersion = ReadInstallMarkerVersion();
        if (string.IsNullOrWhiteSpace(basePackageVersion) || string.Equals(basePackageVersion, "unknown", StringComparison.OrdinalIgnoreCase))
        {
            basePackageVersion = parsed.ModelServerVersion;
        }

        return new PackageVersionInfo
        {
            BasePackageVersion = string.IsNullOrWhiteSpace(basePackageVersion) ? "unknown" : FirstNonEmptyLine(basePackageVersion),
            ModelServerVersion = string.IsNullOrWhiteSpace(parsed.ModelServerVersion) ? "unknown" : parsed.ModelServerVersion,
            GenAiVersion = string.IsNullOrWhiteSpace(parsed.GenAiVersion) ? DetectGenAiSupport(settings.InstallDir) : parsed.GenAiVersion,
            PackageVariant = settings.PackageVariant,
            ManagerVersion = typeof(OvmsController).Assembly.GetName().Version?.ToString() ?? "unknown"
        };
    }

    public async Task<UpdateCheckResult> CheckForUpdatesAsync()
    {
        var packageInfo = GetPackageVersionInfo();
        var modelServerRelease = await GetLatestGitHubReleaseAsync("openvinotoolkit", "model_server").ConfigureAwait(false);
        var genAiRelease = await GetLatestGitHubReleaseAsync("openvinotoolkit", "openvino.genai").ConfigureAwait(false);

        var baseUpdateAvailable = IsUpdateAvailable(packageInfo.BasePackageVersion, modelServerRelease.Version);
        var modelServerUpdateAvailable = IsUpdateAvailable(packageInfo.ModelServerVersion, modelServerRelease.Version);
        var genAiUpdateAvailable = IsUpdateAvailable(packageInfo.GenAiVersion, genAiRelease.Version);
        var latestIsOlderThanInstalled =
            IsInstalledNewerThanLatest(packageInfo.BasePackageVersion, modelServerRelease.Version)
            || IsInstalledNewerThanLatest(packageInfo.ModelServerVersion, modelServerRelease.Version)
            || IsInstalledNewerThanLatest(packageInfo.GenAiVersion, genAiRelease.Version);
        var updateAvailable = baseUpdateAvailable || modelServerUpdateAvailable || genAiUpdateAvailable;

        return new UpdateCheckResult
        {
            BasePackageVersion = packageInfo.BasePackageVersion,
            LatestBasePackageVersion = modelServerRelease.Version,
            ModelServerVersion = packageInfo.ModelServerVersion,
            LatestModelServerVersion = modelServerRelease.Version,
            GenAiVersion = packageInfo.GenAiVersion,
            LatestGenAiVersion = genAiRelease.Version,
            PackageVariant = packageInfo.PackageVariant,
            BasePackageUpdateAvailable = baseUpdateAvailable,
            ModelServerUpdateAvailable = modelServerUpdateAvailable,
            GenAiUpdateAvailable = genAiUpdateAvailable,
            UpdateAvailable = updateAvailable,
            LatestIsOlderThanInstalled = latestIsOlderThanInstalled,
            ModelServerReleaseUrl = modelServerRelease.Url,
            GenAiReleaseUrl = genAiRelease.Url,
            Message = updateAvailable
                ? "Update available."
                : latestIsOlderThanInstalled ? "Installed version is newer than the latest stable release." : "Up to date."
        };
    }

    public string InstallPackageUpdate()
    {
        var settings = LoadSettings();
        var release = GetLatestGitHubReleaseAsync("openvinotoolkit", "model_server", settings.PackageVariant).GetAwaiter().GetResult();
        if (string.IsNullOrWhiteSpace(release.PackageUrl))
        {
            throw new InvalidOperationException($"No Windows package asset found for variant '{settings.PackageVariant}' in the latest Model Server release.");
        }

        var args = string.Join(' ', new[]
        {
            $"-InstallDir \"{settings.InstallDir}\"",
            $"-DataDir \"{DataDir}\"",
            $"-PackageUrl \"{release.PackageUrl}\"",
            string.IsNullOrWhiteSpace(release.Sha256Url) ? "" : $"-Sha256Url \"{release.Sha256Url}\"",
            $"-Version \"{release.Version}\"",
            $"-PackageVariant \"{settings.PackageVariant}\""
        }.Where(arg => !string.IsNullOrWhiteSpace(arg)));

        return RunPowerShell("upgrade-package.ps1", args, includeDataDirArg: false);
    }

    public async Task<HealthResult> CheckHealthAsync()
    {
        var settings = LoadSettings();
        var url = $"http://{settings.BindAddress}:{settings.RestPort}/v3/models";

        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(3));
            using var response = await httpClient.GetAsync(url, cts.Token).ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                return new HealthResult { Ok = false, ModelCount = 0, Message = $"HTTP {(int)response.StatusCode}" };
            }

            var body = await response.Content.ReadAsStringAsync(cts.Token).ConfigureAwait(false);
            var modelCount = 0;
            try
            {
                using var doc = JsonDocument.Parse(body);
                if (doc.RootElement.TryGetProperty("data", out var data) && data.ValueKind == JsonValueKind.Array)
                {
                    modelCount = data.GetArrayLength();
                }
            }
            catch
            {
                // Tolerate unparsable bodies; we still know the endpoint responded OK.
            }

            return new HealthResult { Ok = true, ModelCount = modelCount, Message = "OK" };
        }
        catch (Exception ex)
        {
            return new HealthResult { Ok = false, ModelCount = 0, Message = ex.Message };
        }
    }

    public static bool IsPortFree(int port)
    {
        try
        {
            using var client = new TcpClient();
            var connectTask = client.ConnectAsync("127.0.0.1", port);
            var completed = connectTask.Wait(TimeSpan.FromMilliseconds(300));
            return !(completed && client.Connected);
        }
        catch
        {
            return true;
        }
    }

    public string EffectiveCommandLine()
    {
        var settings = LoadSettings();
        var ovmsExe = Path.Combine(settings.InstallDir, "ovms.exe");

        var args = new List<string>
        {
            "--rest_port", settings.RestPort.ToString(),
            "--rest_bind_address", settings.BindAddress,
            "--config_path", settings.ConfigPath,
            "--log_level", settings.LogLevel,
            "--log_path", settings.LogPath
        };

        if (settings.GrpcPort > 0)
        {
            args.Add("--port");
            args.Add(settings.GrpcPort.ToString());
        }

        return $"\"{ovmsExe}\" {string.Join(' ', args.Select(a => a.Contains(' ') ? $"\"{a}\"" : a))}";
    }

    public IReadOnlyList<string> TailLog(int lines)
    {
        var settings = LoadSettings();
        if (string.IsNullOrEmpty(settings.LogPath) || !File.Exists(settings.LogPath))
        {
            return Array.Empty<string>();
        }

        using var stream = new FileStream(settings.LogPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        using var reader = new StreamReader(stream);

        var buffer = new Queue<string>(lines + 1);
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (buffer.Count == lines)
            {
                buffer.Dequeue();
            }
            buffer.Enqueue(line);
        }

        return buffer.ToList();
    }

    /// <summary>
    /// Builds a diagnostics zip. The timestamp is supplied by the caller so
    /// this method stays deterministic/testable.
    /// </summary>
    public string ExportDiagnostics(string stamp)
    {
        var diagDir = Path.Combine(DataDir, "diagnostics");
        Directory.CreateDirectory(diagDir);

        var zipPath = Path.Combine(diagDir, $"ovms-diagnostics-{stamp}.zip");
        if (File.Exists(zipPath))
        {
            File.Delete(zipPath);
        }

        var settings = LoadSettings();

        using var archive = ZipFile.Open(zipPath, ZipArchiveMode.Create);

        AddFileIfExists(archive, SettingsPath, "settings.json");
        AddFileIfExists(archive, settings.ConfigPath, "models/config.json");
        AddFileIfExists(archive, RuntimePath, "runtime.json");

        var versionsEntry = archive.CreateEntry("versions.txt");
        using (var writer = new StreamWriter(versionsEntry.Open()))
        {
            writer.WriteLine($"Manager version: {typeof(OvmsController).Assembly.GetName().Version}");
            writer.WriteLine($"ovms.exe --version output:");
            writer.WriteLine(GetOvmsVersionOutput(settings.InstallDir));
        }

        var logLines = TailLog(500);
        if (logLines.Count > 0)
        {
            var logEntry = archive.CreateEntry("server.log.tail.txt");
            using var writer = new StreamWriter(logEntry.Open());
            foreach (var line in logLines)
            {
                writer.WriteLine(line);
            }
        }

        return zipPath;
    }

    private static void AddFileIfExists(ZipArchive archive, string path, string entryName)
    {
        if (!string.IsNullOrEmpty(path) && File.Exists(path))
        {
            archive.CreateEntryFromFile(path, entryName);
        }
    }

    private static string GetOvmsVersionOutput(string installDir)
    {
        try
        {
            var ovmsExe = Path.Combine(installDir, "ovms.exe");
            if (!File.Exists(ovmsExe))
            {
                return "(ovms.exe not found)";
            }

            var startInfo = new ProcessStartInfo
            {
                FileName = ovmsExe,
                Arguments = "--version",
                CreateNoWindow = true,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };

            using var process = Process.Start(startInfo);
            if (process is null)
            {
                return "(failed to start ovms.exe)";
            }

            var stdout = process.StandardOutput.ReadToEnd();
            var stderr = process.StandardError.ReadToEnd();
            process.WaitForExit(5000);

            return string.IsNullOrWhiteSpace(stdout) ? stderr : stdout;
        }
        catch (Exception ex)
        {
            return $"(error: {ex.Message})";
        }
    }

    private string ReadInstallMarkerVersion()
    {
        if (!File.Exists(InstallMarkerPath))
        {
            return "";
        }

        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(InstallMarkerPath));
            return doc.RootElement.TryGetProperty("version", out var versionEl) ? versionEl.GetString() ?? "" : "";
        }
        catch
        {
            return "";
        }
    }

    private static (string ModelServerVersion, string GenAiVersion) ParseVersionOutput(string versionOutput)
    {
        var modelServerVersion = "";
        var genAiVersion = "";

        foreach (var line in versionOutput.Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries).Select(l => l.Trim()))
        {
            if (line.StartsWith("OpenVINO Model Server", StringComparison.OrdinalIgnoreCase))
            {
                modelServerVersion = line;
            }
            else if (line.StartsWith("OpenVINO GenAI backend", StringComparison.OrdinalIgnoreCase))
            {
                genAiVersion = line;
            }
        }

        return (modelServerVersion, genAiVersion);
    }

    private static string DetectGenAiSupport(string installDir)
    {
        return File.Exists(Path.Combine(installDir, "openvino_genai.dll")) ? "Installed" : "Missing";
    }

    private async Task<ReleaseMetadata> GetLatestGitHubReleaseAsync(string owner, string repo, string? packageVariant = null)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, $"https://api.github.com/repos/{owner}/{repo}/releases/latest");
        request.Headers.UserAgent.ParseAdd("OVMS-Manager");

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        using var response = await httpClient.SendAsync(request, cts.Token).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var body = await response.Content.ReadAsStringAsync(cts.Token).ConfigureAwait(false);
        using var doc = JsonDocument.Parse(body);
        var root = doc.RootElement;
        var tag = root.TryGetProperty("tag_name", out var tagEl) ? tagEl.GetString() ?? "" : "";
        var htmlUrl = root.TryGetProperty("html_url", out var urlEl) ? urlEl.GetString() ?? "" : "";
        var packageUrl = "";
        var sha256Url = "";

        if (!string.IsNullOrWhiteSpace(packageVariant) && root.TryGetProperty("assets", out var assets) && assets.ValueKind == JsonValueKind.Array)
        {
            foreach (var asset in assets.EnumerateArray())
            {
                var name = asset.TryGetProperty("name", out var nameEl) ? nameEl.GetString() ?? "" : "";
                var url = asset.TryGetProperty("browser_download_url", out var assetUrlEl) ? assetUrlEl.GetString() ?? "" : "";
                if (name.StartsWith("ovms_windows_", StringComparison.OrdinalIgnoreCase)
                    && name.EndsWith($"_{packageVariant}.zip", StringComparison.OrdinalIgnoreCase))
                {
                    packageUrl = url;
                }
                else if (name.StartsWith("ovms_windows_", StringComparison.OrdinalIgnoreCase)
                    && (name.EndsWith($"_{packageVariant}.zip.sha256", StringComparison.OrdinalIgnoreCase)
                        || name.EndsWith($"_{packageVariant}.sha256", StringComparison.OrdinalIgnoreCase)))
                {
                    sha256Url = url;
                }
            }
        }

        return new ReleaseMetadata
        {
            Version = string.IsNullOrWhiteSpace(tag) ? "unknown" : tag,
            Url = htmlUrl,
            PackageUrl = packageUrl,
            Sha256Url = sha256Url
        };
    }

    private static bool IsUpdateAvailable(string installed, string latest)
    {
        return CompareVersionStrings(NormalizeVersion(installed), NormalizeVersion(latest)) < 0;
    }

    private static bool IsInstalledNewerThanLatest(string installed, string latest)
    {
        return CompareVersionStrings(NormalizeVersion(installed), NormalizeVersion(latest)) > 0;
    }

    private static string FirstNonEmptyLine(string value)
    {
        return value.Split(new[] { "\r\n", "\n" }, StringSplitOptions.None)
            .Select(line => line.Trim())
            .FirstOrDefault(line => !string.IsNullOrWhiteSpace(line)) ?? "unknown";
    }

    private static string NormalizeVersion(string value)
    {
        var match = Regex.Match(value, @"\d+(?:\.\d+)+(?:[.-][A-Za-z0-9]+)*");
        return match.Success ? match.Value.TrimStart('v', 'V') : "";
    }

    private static int CompareVersionStrings(string installed, string latest)
    {
        if (string.IsNullOrWhiteSpace(installed) || string.IsNullOrWhiteSpace(latest))
        {
            return 0;
        }

        var installedParts = Regex.Matches(installed, @"\d+").Select(m => int.Parse(m.Value)).ToArray();
        var latestParts = Regex.Matches(latest, @"\d+").Select(m => int.Parse(m.Value)).ToArray();
        var length = Math.Max(installedParts.Length, latestParts.Length);
        for (var i = 0; i < length; i++)
        {
            var installedPart = i < installedParts.Length ? installedParts[i] : 0;
            var latestPart = i < latestParts.Length ? latestParts[i] : 0;
            if (installedPart != latestPart)
            {
                return installedPart.CompareTo(latestPart);
            }
        }

        return 0;
    }

    private string RunPowerShell(string scriptName, string? extraArgs = null, bool includeDataDirArg = true)
    {
        var scriptPath = Path.Combine(AppDir, "scripts", scriptName);
        if (!File.Exists(scriptPath))
        {
            throw new FileNotFoundException("Missing control script.", scriptPath);
        }

        var arguments = $"-NoProfile -ExecutionPolicy Bypass -File \"{scriptPath}\"";
        if (includeDataDirArg)
        {
            arguments += $" -DataDir \"{DataDir}\"";
        }
        if (!string.IsNullOrEmpty(extraArgs))
        {
            arguments += " " + extraArgs;
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = "powershell.exe",
            Arguments = arguments,
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardError = true,
            RedirectStandardOutput = true
        };

        using var process = Process.Start(startInfo) ?? throw new InvalidOperationException("Failed to start PowerShell.");
        var stdout = process.StandardOutput.ReadToEnd();
        var stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(string.IsNullOrWhiteSpace(stderr) ? stdout : stderr);
        }

        return string.IsNullOrWhiteSpace(stdout) ? stderr : stdout;
    }

    public void Dispose()
    {
        httpClient.Dispose();
    }
}
