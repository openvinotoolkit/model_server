using System.Diagnostics;
using System.IO.Compression;
using System.Net.Http;
using System.Net.Sockets;
using System.Text.Json;

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
