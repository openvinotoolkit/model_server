using System.Diagnostics;
using System.Drawing;
using Microsoft.Win32;

namespace OVMS.Manager;

/// <summary>
/// Main GUI window: Dashboard / Settings / Logs / Advanced tabs.
/// All process/HTTP work is dispatched off the UI thread via Task.Run and
/// marshalled back through the WinForms SynchronizationContext (async/await).
/// </summary>
internal sealed class MainForm : Form
{
    private const string RunKeyPath = @"Software\Microsoft\Windows\CurrentVersion\Run";
    private const string RunValueName = "OVMS Manager";

    private readonly OvmsController controller;

    /// <summary>
    /// When true, FormClosing will let the close proceed instead of hiding
    /// to tray. Set by the tray context's Exit handler.
    /// </summary>
    public bool AllowExit { get; set; }

    // Dashboard controls
    private Label statusValueLabel = null!;
    private Label runModeValueLabel = null!;
    private Label restUrlValueLabel = null!;
    private Label grpcValueLabel = null!;
    private Label bindValueLabel = null!;
    private Label variantValueLabel = null!;
    private Label healthValueLabel = null!;
    private Label modelsValueLabel = null!;
    private Button startButton = null!;
    private Button stopButton = null!;
    private Button restartButton = null!;
    private Button refreshButton = null!;

    // Settings controls
    private NumericUpDown restPortInput = null!;
    private NumericUpDown grpcPortInput = null!;
    private TextBox bindAddressInput = null!;
    private ComboBox logLevelInput = null!;
    private TextBox logPathInput = null!;
    private TextBox modelRepoInput = null!;
    private ComboBox runModeInput = null!;
    private CheckBox showTrayCheckBox = null!;
    private CheckBox startAtLoginCheckBox = null!;

    // Logs controls
    private TextBox logTextBox = null!;

    // Advanced controls
    private TextBox advancedTextBox = null!;

    public MainForm(OvmsController controller)
    {
        this.controller = controller;

        Text = "OpenVINO Model Server Manager";
        MinimumSize = new Size(720, 520);
        StartPosition = FormStartPosition.CenterScreen;
        try
        {
            Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
        }
        catch
        {
            Icon = SystemIcons.Application;
        }

        var tabs = new TabControl { Dock = DockStyle.Fill };
        tabs.TabPages.Add(BuildDashboardTab());
        tabs.TabPages.Add(BuildSettingsTab());
        tabs.TabPages.Add(BuildLogsTab());
        tabs.TabPages.Add(BuildAdvancedTab());
        Controls.Add(tabs);

        FormClosing += OnFormClosing;
        Shown += async (_, _) => await RefreshDashboardAsync();
    }

    private void OnFormClosing(object? sender, FormClosingEventArgs e)
    {
        if (e.CloseReason == CloseReason.UserClosing && !AllowExit)
        {
            var settings = TryLoadSettings();
            if (settings is { ShowTrayIcon: true })
            {
                e.Cancel = true;
                Hide();
            }
        }
    }

    private OvmsSettings? TryLoadSettings()
    {
        try
        {
            return controller.LoadSettings();
        }
        catch
        {
            return null;
        }
    }

    // ---------------------------------------------------------------
    // Dashboard
    // ---------------------------------------------------------------

    private TabPage BuildDashboardTab()
    {
        var page = new TabPage("Dashboard");
        var layout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            Padding = new Padding(12),
            AutoSize = true
        };
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

        (Label caption, Label value) AddRow(string caption)
        {
            var captionLabel = new Label { Text = caption, AutoSize = true, Margin = new Padding(3, 8, 12, 3), Font = new Font(Font, FontStyle.Bold) };
            var valueLabel = new Label { Text = "-", AutoSize = true, Margin = new Padding(3, 8, 3, 3) };
            layout.Controls.Add(captionLabel);
            layout.Controls.Add(valueLabel);
            return (captionLabel, valueLabel);
        }

        (_, statusValueLabel) = AddRow("Status:");
        (_, runModeValueLabel) = AddRow("Runtime mode:");
        (_, restUrlValueLabel) = AddRow("REST URL:");
        (_, grpcValueLabel) = AddRow("gRPC port:");
        (_, bindValueLabel) = AddRow("Bind address:");
        (_, variantValueLabel) = AddRow("Package variant:");
        (_, healthValueLabel) = AddRow("Health:");
        (_, modelsValueLabel) = AddRow("Served models:");

        var buttonPanel = new FlowLayoutPanel { Dock = DockStyle.Bottom, Height = 48, Padding = new Padding(8) };
        startButton = new Button { Text = "Start", AutoSize = true };
        stopButton = new Button { Text = "Stop", AutoSize = true };
        restartButton = new Button { Text = "Restart", AutoSize = true };
        refreshButton = new Button { Text = "Refresh", AutoSize = true };
        var openLogsButton = new Button { Text = "Open Logs", AutoSize = true };
        var openModelFolderButton = new Button { Text = "Open Model Folder", AutoSize = true };

        startButton.Click += async (_, _) => await RunControlActionAsync("Start", controller.Start);
        stopButton.Click += async (_, _) => await RunControlActionAsync("Stop", controller.Stop);
        restartButton.Click += async (_, _) => await RunControlActionAsync("Restart", controller.Restart);
        refreshButton.Click += async (_, _) => await RefreshDashboardAsync();
        openLogsButton.Click += (_, _) => OpenFolderFor(TryLoadSettings()?.LogPath);
        openModelFolderButton.Click += (_, _) => OpenFolder(TryLoadSettings()?.ModelRepositoryPath);

        buttonPanel.Controls.Add(startButton);
        buttonPanel.Controls.Add(stopButton);
        buttonPanel.Controls.Add(restartButton);
        buttonPanel.Controls.Add(refreshButton);
        buttonPanel.Controls.Add(openLogsButton);
        buttonPanel.Controls.Add(openModelFolderButton);

        page.Controls.Add(layout);
        page.Controls.Add(buttonPanel);
        return page;
    }

    private async Task RunControlActionAsync(string title, Action action)
    {
        SetControlButtonsEnabled(false);
        try
        {
            await Task.Run(action);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, title, MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            SetControlButtonsEnabled(true);
            await RefreshDashboardAsync();
        }
    }

    private void SetControlButtonsEnabled(bool enabled)
    {
        startButton.Enabled = enabled;
        stopButton.Enabled = enabled;
        restartButton.Enabled = enabled;
        refreshButton.Enabled = enabled;
    }

    public async Task RefreshDashboardAsync()
    {
        var settings = TryLoadSettings();
        if (settings is null)
        {
            statusValueLabel.Text = "Settings unavailable";
            return;
        }

        var runtimeStatus = await Task.Run(controller.GetRuntimeStatus);
        statusValueLabel.Text = runtimeStatus.Running ? $"Running (pid {runtimeStatus.Pid})" : "Stopped";
        runModeValueLabel.Text = settings.RunMode;
        restUrlValueLabel.Text = $"http://{settings.BindAddress}:{settings.RestPort}";
        grpcValueLabel.Text = settings.GrpcPort > 0 ? settings.GrpcPort.ToString() : "disabled";
        bindValueLabel.Text = settings.BindAddress;
        variantValueLabel.Text = settings.PackageVariant;

        healthValueLabel.Text = "Checking...";
        modelsValueLabel.Text = "-";

        var health = await controller.CheckHealthAsync();
        healthValueLabel.Text = health.Ok ? "Healthy" : $"Unreachable ({health.Message})";
        modelsValueLabel.Text = health.Ok ? health.ModelCount.ToString() : "-";
    }

    private static void OpenFolderFor(string? filePath)
    {
        if (string.IsNullOrEmpty(filePath))
        {
            return;
        }
        OpenFolder(Path.GetDirectoryName(filePath));
    }

    private static void OpenFolder(string? folderPath)
    {
        if (string.IsNullOrEmpty(folderPath))
        {
            return;
        }
        try
        {
            Directory.CreateDirectory(folderPath);
            Process.Start(new ProcessStartInfo { FileName = folderPath, UseShellExecute = true });
        }
        catch
        {
            // Best effort; non-fatal if the folder cannot be opened.
        }
    }

    // ---------------------------------------------------------------
    // Settings
    // ---------------------------------------------------------------

    private TabPage BuildSettingsTab()
    {
        var page = new TabPage("Settings");
        var layout = new TableLayoutPanel
        {
            Dock = DockStyle.Top,
            ColumnCount = 2,
            AutoSize = true,
            Padding = new Padding(12)
        };
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

        void AddLabel(string text)
        {
            layout.Controls.Add(new Label { Text = text, AutoSize = true, Margin = new Padding(3, 8, 12, 3) });
        }

        AddLabel("REST port:");
        restPortInput = new NumericUpDown { Minimum = 1, Maximum = 65535, Width = 120 };
        layout.Controls.Add(restPortInput);

        AddLabel("gRPC port (0 = disabled):");
        grpcPortInput = new NumericUpDown { Minimum = 0, Maximum = 65535, Width = 120 };
        layout.Controls.Add(grpcPortInput);

        AddLabel("Bind address:");
        bindAddressInput = new TextBox { Width = 220 };
        layout.Controls.Add(bindAddressInput);

        AddLabel("Log level:");
        logLevelInput = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 120 };
        logLevelInput.Items.AddRange(new object[] { "ERROR", "WARNING", "INFO", "DEBUG", "TRACE" });
        layout.Controls.Add(logLevelInput);

        AddLabel("Log path:");
        logPathInput = new TextBox { Width = 400 };
        layout.Controls.Add(logPathInput);

        AddLabel("Model repository path:");
        modelRepoInput = new TextBox { Width = 400 };
        layout.Controls.Add(modelRepoInput);

        AddLabel("Startup mode:");
        runModeInput = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 160 };
        runModeInput.Items.AddRange(new object[] { "user-login", "service", "manual" });
        layout.Controls.Add(runModeInput);

        AddLabel("Show tray icon:");
        showTrayCheckBox = new CheckBox();
        layout.Controls.Add(showTrayCheckBox);

        AddLabel("Start at login:");
        startAtLoginCheckBox = new CheckBox();
        layout.Controls.Add(startAtLoginCheckBox);

        var saveButton = new Button { Text = "Save", AutoSize = true, Margin = new Padding(3, 16, 3, 3) };
        saveButton.Click += async (_, _) => await SaveSettingsAsync();
        layout.Controls.Add(new Label());
        layout.Controls.Add(saveButton);

        page.Controls.Add(layout);
        page.Enter += (_, _) => LoadSettingsIntoForm();
        return page;
    }

    private void LoadSettingsIntoForm()
    {
        var settings = TryLoadSettings();
        if (settings is null)
        {
            return;
        }

        restPortInput.Value = Math.Clamp(settings.RestPort, (int)restPortInput.Minimum, (int)restPortInput.Maximum);
        grpcPortInput.Value = Math.Clamp(settings.GrpcPort, (int)grpcPortInput.Minimum, (int)grpcPortInput.Maximum);
        bindAddressInput.Text = settings.BindAddress;
        logLevelInput.SelectedItem = settings.LogLevel;
        logPathInput.Text = settings.LogPath;
        modelRepoInput.Text = settings.ModelRepositoryPath;
        runModeInput.SelectedItem = settings.RunMode;
        showTrayCheckBox.Checked = settings.ShowTrayIcon;
        startAtLoginCheckBox.Checked = settings.StartAtLogin;
    }

    private async Task SaveSettingsAsync()
    {
        var current = TryLoadSettings();
        if (current is null)
        {
            MessageBox.Show(this, "Could not load existing settings.", "Save", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        if (string.IsNullOrWhiteSpace(bindAddressInput.Text))
        {
            MessageBox.Show(this, "Bind address cannot be empty.", "Save", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        var updated = new OvmsSettings
        {
            InstallDir = current.InstallDir,
            DataDir = current.DataDir,
            ModelRepositoryPath = modelRepoInput.Text,
            ConfigPath = current.ConfigPath,
            LogPath = logPathInput.Text,
            RestPort = (int)restPortInput.Value,
            GrpcPort = (int)grpcPortInput.Value,
            BindAddress = bindAddressInput.Text,
            LogLevel = logLevelInput.SelectedItem as string ?? current.LogLevel,
            RunMode = runModeInput.SelectedItem as string ?? current.RunMode,
            StartAtLogin = startAtLoginCheckBox.Checked,
            ShowTrayIcon = showTrayCheckBox.Checked,
            ServiceAutoStart = current.ServiceAutoStart,
            PackageVariant = current.PackageVariant
        };

        var commandLineAffectingChanged =
            current.RestPort != updated.RestPort ||
            current.GrpcPort != updated.GrpcPort ||
            current.BindAddress != updated.BindAddress ||
            current.LogPath != updated.LogPath ||
            current.ConfigPath != updated.ConfigPath ||
            current.LogLevel != updated.LogLevel;

        try
        {
            await Task.Run(() => controller.SaveSettings(updated));
            ApplyStartAtLoginRegistration(updated.StartAtLogin);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Save", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        if (commandLineAffectingChanged)
        {
            var runtimeStatus = await Task.Run(controller.GetRuntimeStatus);
            if (runtimeStatus.Running)
            {
                var result = MessageBox.Show(this, "Settings that affect the server command line changed. Restart server to apply?",
                    "Restart required", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                if (result == DialogResult.Yes)
                {
                    await RunControlActionAsync("Restart", controller.Restart);
                }
            }
        }

        await RefreshDashboardAsync();
        MessageBox.Show(this, "Settings saved.", "Save", MessageBoxButtons.OK, MessageBoxIcon.Information);
    }

    private static void ApplyStartAtLoginRegistration(bool enabled)
    {
        try
        {
            using var key = Registry.CurrentUser.OpenSubKey(RunKeyPath, writable: true)
                ?? Registry.CurrentUser.CreateSubKey(RunKeyPath, writable: true);

            if (key is null)
            {
                return;
            }

            if (enabled)
            {
                var exePath = Application.ExecutablePath;
                key.SetValue(RunValueName, $"\"{exePath}\" --tray");
            }
            else
            {
                if (key.GetValue(RunValueName) is not null)
                {
                    key.DeleteValue(RunValueName, throwOnMissingValue: false);
                }
            }
        }
        catch
        {
            // Non-fatal: registry write may fail under restricted environments.
        }
    }

    // ---------------------------------------------------------------
    // Logs
    // ---------------------------------------------------------------

    private TabPage BuildLogsTab()
    {
        var page = new TabPage("Logs");

        logTextBox = new TextBox
        {
            Dock = DockStyle.Fill,
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Both,
            WordWrap = false,
            Font = new Font(FontFamily.GenericMonospace, 9f)
        };

        var buttonPanel = new FlowLayoutPanel { Dock = DockStyle.Bottom, Height = 40, Padding = new Padding(8) };
        var refreshLogsButton = new Button { Text = "Refresh", AutoSize = true };
        var openFolderButton = new Button { Text = "Open Folder", AutoSize = true };

        refreshLogsButton.Click += async (_, _) => await RefreshLogsAsync();
        openFolderButton.Click += (_, _) => OpenFolderFor(TryLoadSettings()?.LogPath);

        buttonPanel.Controls.Add(refreshLogsButton);
        buttonPanel.Controls.Add(openFolderButton);

        page.Controls.Add(logTextBox);
        page.Controls.Add(buttonPanel);
        page.Enter += async (_, _) => await RefreshLogsAsync();
        return page;
    }

    private async Task RefreshLogsAsync()
    {
        var lines = await Task.Run(() => controller.TailLog(500));
        logTextBox.Text = string.Join(Environment.NewLine, lines);
        logTextBox.SelectionStart = logTextBox.Text.Length;
        logTextBox.ScrollToCaret();
    }

    // ---------------------------------------------------------------
    // Advanced
    // ---------------------------------------------------------------

    private TabPage BuildAdvancedTab()
    {
        var page = new TabPage("Advanced");

        advancedTextBox = new TextBox
        {
            Dock = DockStyle.Fill,
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Both,
            WordWrap = false,
            Font = new Font(FontFamily.GenericMonospace, 9f)
        };

        var buttonPanel = new FlowLayoutPanel { Dock = DockStyle.Bottom, Height = 40, Padding = new Padding(8) };
        var repairButton = new Button { Text = "Repair Package", AutoSize = true };
        var exportButton = new Button { Text = "Export Diagnostics", AutoSize = true };
        var validateButton = new Button { Text = "Validate Environment", AutoSize = true };

        repairButton.Click += async (_, _) => await RunAdvancedActionAsync("Repair", () => controller.Repair());
        validateButton.Click += async (_, _) => await RunAdvancedActionAsync("Validate Environment", () => controller.ValidateEnvironment());
        exportButton.Click += async (_, _) => await ExportDiagnosticsAsync();

        buttonPanel.Controls.Add(repairButton);
        buttonPanel.Controls.Add(exportButton);
        buttonPanel.Controls.Add(validateButton);

        page.Controls.Add(advancedTextBox);
        page.Controls.Add(buttonPanel);
        page.Enter += (_, _) => RefreshAdvancedInfo();
        return page;
    }

    private void RefreshAdvancedInfo()
    {
        var settings = TryLoadSettings();
        if (settings is null)
        {
            advancedTextBox.Text = "Settings unavailable.";
            return;
        }

        string commandLine;
        try
        {
            commandLine = controller.EffectiveCommandLine();
        }
        catch (Exception ex)
        {
            commandLine = $"(error building command line: {ex.Message})";
        }

        advancedTextBox.Text = string.Join(Environment.NewLine, new[]
        {
            $"Install dir: {settings.InstallDir}",
            $"Data dir: {controller.DataDir}",
            "",
            "Effective command line:",
            commandLine
        });
    }

    private async Task RunAdvancedActionAsync(string title, Func<string> action)
    {
        try
        {
            var output = await Task.Run(action);
            MessageBox.Show(this, string.IsNullOrWhiteSpace(output) ? "Completed." : output, title, MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, title, MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            RefreshAdvancedInfo();
        }
    }

    private async Task ExportDiagnosticsAsync()
    {
        var stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
        try
        {
            var zipPath = await Task.Run(() => controller.ExportDiagnostics(stamp));
            MessageBox.Show(this, $"Diagnostics exported to:{Environment.NewLine}{zipPath}", "Export Diagnostics", MessageBoxButtons.OK, MessageBoxIcon.Information);
            OpenFolderFor(zipPath);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Export Diagnostics", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}
