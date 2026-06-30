using System.Diagnostics;
using System.Drawing;
using System.Text.Json;
using System.Windows.Forms;

namespace OVMS.Manager;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        ApplicationConfiguration.Initialize();
        using var app = new TrayApplicationContext();
        Application.Run(app);
    }
}

internal sealed class TrayApplicationContext : ApplicationContext
{
    private readonly NotifyIcon notifyIcon;
    private readonly ToolStripMenuItem startItem;
    private readonly ToolStripMenuItem stopItem;
    private readonly OvmsController controller;

    public TrayApplicationContext()
    {
        controller = new OvmsController();

        startItem = new ToolStripMenuItem("Start Server", null, (_, _) => RunAction("Start Server", controller.Start));
        stopItem = new ToolStripMenuItem("Stop Server", null, (_, _) => RunAction("Stop Server", controller.Stop));

        var openItem = new ToolStripMenuItem("Open OVMS Manager", null, (_, _) => OpenManager());
        var menu = new ContextMenuStrip();
        menu.Items.Add(openItem);
        menu.Items.Add(startItem);
        menu.Items.Add(stopItem);

        notifyIcon = new NotifyIcon
        {
            Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath) ?? SystemIcons.Application,
            Text = "OpenVINO Model Server",
            ContextMenuStrip = menu,
            Visible = true
        };
        notifyIcon.MouseClick += (_, args) =>
        {
            if (args.Button == MouseButtons.Left)
            {
                OpenManager();
            }
        };
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            notifyIcon.Visible = false;
            notifyIcon.Dispose();
            controller.Dispose();
        }
        base.Dispose(disposing);
    }

    private void OpenManager()
    {
        var status = controller.GetStatus();
        MessageBox.Show(status, "OpenVINO Model Server", MessageBoxButtons.OK, MessageBoxIcon.Information);
    }

    private void RunAction(string title, Action action)
    {
        try
        {
            startItem.Enabled = false;
            stopItem.Enabled = false;
            action();
            notifyIcon.ShowBalloonTip(3000, "OpenVINO Model Server", $"{title} completed.", ToolTipIcon.Info);
        }
        catch (Exception ex)
        {
            MessageBox.Show(ex.Message, title, MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            startItem.Enabled = true;
            stopItem.Enabled = true;
        }
    }
}

internal sealed class OvmsController : IDisposable
{
    private readonly string dataDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "OVMS");
    private readonly string appDir = AppContext.BaseDirectory;

    public string GetStatus()
    {
        var settings = LoadSettings();
        var runtimePath = Path.Combine(dataDir, "runtime.json");
        var status = "Stopped";

        if (File.Exists(runtimePath))
        {
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(runtimePath));
                if (doc.RootElement.TryGetProperty("pid", out var pidElement) && pidElement.TryGetInt32(out var pid))
                {
                    var process = Process.GetProcessById(pid);
                    if (string.Equals(process.ProcessName, "ovms", StringComparison.OrdinalIgnoreCase))
                    {
                        status = $"Running, pid {pid}";
                    }
                }
            }
            catch
            {
                status = "Runtime state is stale or unreadable";
            }
        }

        return string.Join(Environment.NewLine, new[]
        {
            $"Status: {status}",
            $"REST: http://{settings.BindAddress}:{settings.RestPort}",
            $"Install: {settings.InstallDir}",
            $"Config: {settings.ConfigPath}",
            $"Logs: {settings.LogPath}"
        });
    }

    public void Start()
    {
        RunPowerShell("start-ovms.ps1");
    }

    public void Stop()
    {
        RunPowerShell("stop-ovms.ps1");
    }

    public void Dispose()
    {
    }

    private OvmsSettings LoadSettings()
    {
        var settingsPath = Path.Combine(dataDir, "settings.json");
        if (!File.Exists(settingsPath))
        {
            return new OvmsSettings
            {
                InstallDir = appDir.TrimEnd(Path.DirectorySeparatorChar),
                DataDir = dataDir,
                ConfigPath = Path.Combine(dataDir, "models", "config.json"),
                LogPath = Path.Combine(dataDir, "logs", "ovms_server.log"),
                BindAddress = "127.0.0.1",
                RestPort = 8000,
                GrpcPort = 0
            };
        }

        var settings = JsonSerializer.Deserialize<OvmsSettings>(File.ReadAllText(settingsPath), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        });

        return settings ?? throw new InvalidOperationException($"Could not parse settings: {settingsPath}");
    }

    private void RunPowerShell(string scriptName)
    {
        var scriptPath = Path.Combine(appDir, "scripts", scriptName);
        if (!File.Exists(scriptPath))
        {
            throw new FileNotFoundException("Missing control script.", scriptPath);
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = "powershell.exe",
            Arguments = $"-NoProfile -ExecutionPolicy Bypass -File \"{scriptPath}\" -DataDir \"{dataDir}\"",
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
    }
}

internal sealed class OvmsSettings
{
    public string InstallDir { get; set; } = "";
    public string DataDir { get; set; } = "";
    public string ConfigPath { get; set; } = "";
    public string LogPath { get; set; } = "";
    public string BindAddress { get; set; } = "127.0.0.1";
    public int RestPort { get; set; } = 8000;
    public int GrpcPort { get; set; } = 9000;
}
