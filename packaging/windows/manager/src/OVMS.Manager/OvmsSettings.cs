namespace OVMS.Manager;

/// <summary>
/// Plain settings POCO persisted to %LOCALAPPDATA%\OVMS\settings.json.
/// Property names match the JSON keys case-insensitively (System.Text.Json
/// is configured with PropertyNameCaseInsensitive = true wherever this is
/// deserialized).
/// </summary>
internal sealed class OvmsSettings
{
    public string InstallDir { get; set; } = "";
    public string DataDir { get; set; } = "";
    public string ModelRepositoryPath { get; set; } = "";
    public string ConfigPath { get; set; } = "";
    public string LogPath { get; set; } = "";
    public int RestPort { get; set; } = 8000;
    public int GrpcPort { get; set; } = 0;
    public string BindAddress { get; set; } = "127.0.0.1";
    public string LogLevel { get; set; } = "INFO";
    public string RunMode { get; set; } = "user-login";
    public bool StartAtLogin { get; set; } = true;
    public bool ShowTrayIcon { get; set; } = true;
    public bool ServiceAutoStart { get; set; } = false;
    public string PackageVariant { get; set; } = "python_on";
}
