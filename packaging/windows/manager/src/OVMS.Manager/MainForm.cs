using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using Microsoft.Win32;

namespace OVMS.Manager;

/// <summary>
/// Main GUI window: app-shell layout with a left nav rail (Dashboard /
/// Settings / Logs / Advanced) and a content area on the right. Pages are
/// plain Panels toggled via Visible, not a TabControl.
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

    // Shell controls
    private Panel navRail = null!;
    private Panel contentHost = null!;
    private Label pageTitleLabel = null!;
    private Button headerRefreshButton = null!;
    private readonly List<NavButton> navButtons = new();
    private readonly Dictionary<string, Panel> pages = new();
    private Label railStatusDot = null!;
    private Label railStatusText = null!;
    private System.Windows.Forms.Timer? dashboardTimer;
    private string currentPage = "";

    // Dashboard controls
    private Label statusValueLabel = null!;
    private Label statusDotLabel = null!;
    private Label runModeValueLabel = null!;
    private Label restUrlValueLabel = null!;
    private Label grpcValueLabel = null!;
    private Label variantValueLabel = null!;
    private Label healthValueLabel = null!;
    private Label healthDotLabel = null!;
    private Label modelsValueLabel = null!;
    private Button startButton = null!;
    private Button stopButton = null!;
    private Button restartButton = null!;

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
    private Label advancedStatusLabel = null!;

    public MainForm(OvmsController controller)
    {
        this.controller = controller;

        Text = "OpenVINO Model Server Manager";
        MinimumSize = new Size(860, 600);
        Size = new Size(960, 640);
        StartPosition = FormStartPosition.CenterScreen;
        BackColor = Theme.WindowBackground;
        Font = Theme.BaseFont;
        SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.OptimizedDoubleBuffer, true);

        try
        {
            Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
        }
        catch
        {
            Icon = SystemIcons.Application;
        }

        BuildShell();

        FormClosing += OnFormClosing;
        FormClosed += (_, _) => StopDashboardTimer();
        Shown += async (_, _) => await RefreshDashboardAsync();

        SelectPage("Dashboard");
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
    // Shell: nav rail + content header + page hosting
    // ---------------------------------------------------------------

    private void BuildShell()
    {
        navRail = new Panel
        {
            Dock = DockStyle.Left,
            Width = 210,
            BackColor = Theme.Rail
        };

        var railBorder = new Panel { Dock = DockStyle.Right, Width = 1, BackColor = Theme.Border };
        navRail.Controls.Add(railBorder);

        var headerBlock = new Panel { Dock = DockStyle.Top, Height = 76, BackColor = Theme.Rail };
        var titleLabel = new Label
        {
            Text = "OVMS Manager",
            Font = Theme.SemiboldFont,
            ForeColor = Theme.Text,
            AutoSize = false,
            TextAlign = ContentAlignment.MiddleLeft,
            Dock = DockStyle.Fill,
            Padding = new Padding(18, 0, 8, 0)
        };
        var ringDot = new Label
        {
            Text = "●",
            ForeColor = Theme.Accent,
            Font = new Font("Segoe UI", 14f),
            AutoSize = true,
            Location = new Point(18, 14)
        };
        headerBlock.Controls.Add(titleLabel);
        headerBlock.Controls.Add(ringDot);
        titleLabel.Padding = new Padding(40, 0, 8, 0);
        titleLabel.Location = new Point(0, 14);
        titleLabel.Size = new Size(210, 40);

        var navFlow = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            FlowDirection = FlowDirection.TopDown,
            WrapContents = false,
            AutoSize = true,
            Top = 76
        };
        navFlow.Controls.Add(CreateNavButton("Dashboard", Glyphs.Dashboard));
        navFlow.Controls.Add(CreateNavButton("Settings", Glyphs.Settings));
        navFlow.Controls.Add(CreateNavButton("Logs", Glyphs.Logs));
        navFlow.Controls.Add(CreateNavButton("Advanced", Glyphs.Advanced));

        var statusPill = new Panel { Dock = DockStyle.Bottom, Height = 56, BackColor = Theme.Rail, Padding = new Padding(16, 12, 16, 12) };
        var pillInner = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
        railStatusDot = new Label { Text = "●", ForeColor = Theme.Muted, Font = new Font("Segoe UI", 10f), AutoSize = true, Margin = new Padding(0, 3, 6, 0) };
        railStatusText = new Label { Text = "Checking...", ForeColor = Theme.Muted, Font = Theme.BaseFont, AutoSize = true, Margin = new Padding(0, 4, 0, 0) };
        pillInner.Controls.Add(railStatusDot);
        pillInner.Controls.Add(railStatusText);
        statusPill.Controls.Add(pillInner);

        navRail.Controls.Add(navFlow);
        navRail.Controls.Add(statusPill);
        navRail.Controls.Add(headerBlock);

        // Content area
        var contentArea = new Panel { Dock = DockStyle.Fill, BackColor = Theme.WindowBackground };

        var contentHeader = new Panel { Dock = DockStyle.Top, Height = 56, BackColor = Theme.WindowBackground, Padding = new Padding(24, 12, 24, 12) };
        pageTitleLabel = new Label { Text = "Dashboard", Font = Theme.PageTitleFont, ForeColor = Theme.Text, AutoSize = true, Dock = DockStyle.Left };
        headerRefreshButton = CreateIconButton(Glyphs.Refresh, "Refresh");
        headerRefreshButton.Dock = DockStyle.Right;
        headerRefreshButton.Click += async (_, _) => await RefreshCurrentPageAsync();
        contentHeader.Controls.Add(pageTitleLabel);
        contentHeader.Controls.Add(headerRefreshButton);

        contentHost = new Panel { Dock = DockStyle.Fill, BackColor = Theme.WindowBackground, Padding = new Padding(24, 12, 24, 24) };
        SetDoubleBuffered(contentHost);

        pages["Dashboard"] = BuildDashboardPage();
        pages["Settings"] = BuildSettingsPage();
        pages["Logs"] = BuildLogsPage();
        pages["Advanced"] = BuildAdvancedPage();

        foreach (var page in pages.Values)
        {
            page.Dock = DockStyle.Fill;
            page.Visible = false;
            contentHost.Controls.Add(page);
        }

        contentArea.Controls.Add(contentHost);
        contentArea.Controls.Add(contentHeader);

        Controls.Add(contentArea);
        Controls.Add(navRail);
    }

    private static void SetDoubleBuffered(Control control)
    {
        typeof(Control).InvokeMember("DoubleBuffered",
            System.Reflection.BindingFlags.SetProperty | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic,
            null, control, new object[] { true });
    }

    private NavButton CreateNavButton(string name, string glyph)
    {
        var button = new NavButton(name, glyph);
        button.Click += (_, _) => SelectPage(name);
        navButtons.Add(button);
        return button;
    }

    private void SelectPage(string name)
    {
        if (!pages.ContainsKey(name))
        {
            return;
        }

        currentPage = name;
        pageTitleLabel.Text = name;

        foreach (var page in pages)
        {
            page.Value.Visible = page.Key == name;
        }

        foreach (var nav in navButtons)
        {
            nav.Selected = nav.PageName == name;
        }

        if (name == "Dashboard")
        {
            StartDashboardTimer();
            _ = RefreshDashboardAsync();
        }
        else
        {
            StopDashboardTimer();
            if (name == "Settings")
            {
                LoadSettingsIntoForm();
            }
            else if (name == "Logs")
            {
                _ = RefreshLogsAsync();
            }
            else if (name == "Advanced")
            {
                RefreshAdvancedInfo();
            }
        }
    }

    private async Task RefreshCurrentPageAsync()
    {
        switch (currentPage)
        {
            case "Dashboard":
                await RefreshDashboardAsync();
                break;
            case "Logs":
                await RefreshLogsAsync();
                break;
            case "Advanced":
                RefreshAdvancedInfo();
                break;
            case "Settings":
                LoadSettingsIntoForm();
                break;
        }
    }

    private void StartDashboardTimer()
    {
        if (dashboardTimer != null)
        {
            return;
        }
        dashboardTimer = new System.Windows.Forms.Timer { Interval = 5000 };
        dashboardTimer.Tick += async (_, _) =>
        {
            if (currentPage == "Dashboard" && Visible)
            {
                await RefreshDashboardAsync();
            }
        };
        dashboardTimer.Start();
    }

    private void StopDashboardTimer()
    {
        if (dashboardTimer is null)
        {
            return;
        }
        dashboardTimer.Stop();
        dashboardTimer.Dispose();
        dashboardTimer = null;
    }

    // ---------------------------------------------------------------
    // Shared UI helpers
    // ---------------------------------------------------------------

    private static Button CreateIconButton(string glyph, string tooltip)
    {
        var button = new Button
        {
            Text = glyph,
            Font = Theme.IconFont(13f),
            FlatStyle = FlatStyle.Flat,
            ForeColor = Theme.Muted,
            BackColor = Theme.Surface,
            Size = new Size(32, 32),
            Cursor = Cursors.Hand,
            UseVisualStyleBackColor = false
        };
        button.FlatAppearance.BorderSize = 1;
        button.FlatAppearance.BorderColor = Theme.Border;
        button.FlatAppearance.MouseOverBackColor = Theme.HoverTint;
        var tip = new ToolTip();
        tip.SetToolTip(button, tooltip);
        return button;
    }

    private static Button CreatePrimaryButton(string glyph, string text, Color fillColor)
    {
        var button = new Button
        {
            Text = "  " + text,
            Font = Theme.NavFont,
            FlatStyle = FlatStyle.Flat,
            ForeColor = Color.White,
            BackColor = fillColor,
            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowAndShrink,
            Padding = new Padding(12, 8, 14, 8),
            Cursor = Cursors.Hand,
            UseVisualStyleBackColor = false,
            TextImageRelation = TextImageRelation.ImageBeforeText
        };
        button.FlatAppearance.BorderSize = 0;
        button.FlatAppearance.MouseOverBackColor = ControlPaint.Dark(fillColor, 0.08f);
        return button;
    }

    private static Button CreateSecondaryButton(string text)
    {
        var button = new Button
        {
            Text = "  " + text,
            Font = Theme.NavFont,
            FlatStyle = FlatStyle.Flat,
            ForeColor = Theme.Text,
            BackColor = Theme.Surface,
            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowAndShrink,
            Padding = new Padding(12, 8, 14, 8),
            Cursor = Cursors.Hand,
            UseVisualStyleBackColor = false
        };
        button.FlatAppearance.BorderSize = 1;
        button.FlatAppearance.BorderColor = Theme.Border;
        button.FlatAppearance.MouseOverBackColor = Theme.HoverTint;
        return button;
    }

    private static CardPanel CreateCard(int width, int height)
    {
        return new CardPanel
        {
            Width = width,
            Height = height,
            Margin = new Padding(0, 0, 16, 16)
        };
    }

    // ---------------------------------------------------------------
    // Dashboard
    // ---------------------------------------------------------------

    private Panel BuildDashboardPage()
    {
        var page = new Panel { BackColor = Theme.WindowBackground };

        var cardFlow = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = true,
            AutoSize = true,
            Padding = new Padding(0, 0, 0, 8)
        };

        (Label dot, Label value) AddStatusCard(string title, string initialValue, bool withDot)
        {
            var card = CreateCard(220, 92);
            var titleLabel = new Label { Text = title, Font = Theme.CardTitleFont, ForeColor = Theme.Muted, AutoSize = true, Location = new Point(16, 14) };
            Label? dotLabel = null;
            var valueLabel = new Label { Text = initialValue, Font = Theme.CardValueFont, ForeColor = Theme.Text, AutoSize = true, Location = new Point(withDot ? 32 : 16, 42) };
            if (withDot)
            {
                dotLabel = new Label { Text = "●", Font = new Font("Segoe UI", 11f), ForeColor = Theme.Muted, AutoSize = true, Location = new Point(16, 44) };
                card.Controls.Add(dotLabel);
            }
            card.Controls.Add(titleLabel);
            card.Controls.Add(valueLabel);
            cardFlow.Controls.Add(card);
            return (dotLabel!, valueLabel);
        }

        (statusDotLabel, statusValueLabel) = AddStatusCard("Status", "-", withDot: true);
        (_, restUrlValueLabel) = AddStatusCard("REST endpoint", "-", withDot: false);
        (_, grpcValueLabel) = AddStatusCard("gRPC port", "-", withDot: false);
        (_, modelsValueLabel) = AddStatusCard("Models served", "-", withDot: false);
        (healthDotLabel, healthValueLabel) = AddStatusCard("Health", "-", withDot: true);
        (_, variantValueLabel) = AddStatusCard("Package variant", "-", withDot: false);
        (_, runModeValueLabel) = AddStatusCard("Runtime mode", "-", withDot: false);

        var actionsCard = new CardPanel { Dock = DockStyle.Top, Height = 76, Margin = new Padding(0, 8, 0, 0) };
        var actionsFlow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = true, Padding = new Padding(4) };

        startButton = CreatePrimaryButton(Glyphs.Play, "Start", Theme.Success);
        stopButton = CreatePrimaryButton(Glyphs.Stop, "Stop", Theme.Danger);
        restartButton = CreatePrimaryButton(Glyphs.Restart, "Restart", Theme.Accent);
        var openLogsButton = CreateSecondaryButton("Open Logs");
        var openModelFolderButton = CreateSecondaryButton("Open Model Folder");

        foreach (var b in new[] { startButton, stopButton, restartButton, openLogsButton, openModelFolderButton })
        {
            b.Margin = new Padding(0, 0, 10, 0);
        }

        startButton.Click += async (_, _) => await RunControlActionAsync("Start", controller.Start);
        stopButton.Click += async (_, _) => await RunControlActionAsync("Stop", controller.Stop);
        restartButton.Click += async (_, _) => await RunControlActionAsync("Restart", controller.Restart);
        openLogsButton.Click += (_, _) => OpenFolderFor(TryLoadSettings()?.LogPath);
        openModelFolderButton.Click += (_, _) => OpenFolder(TryLoadSettings()?.ModelRepositoryPath);

        actionsFlow.Controls.Add(startButton);
        actionsFlow.Controls.Add(stopButton);
        actionsFlow.Controls.Add(restartButton);
        actionsFlow.Controls.Add(openLogsButton);
        actionsFlow.Controls.Add(openModelFolderButton);
        actionsCard.Controls.Add(actionsFlow);

        // Dock order: bottom-most added is at the top visually for DockStyle.Top stacking,
        // so add actionsCard after cardFlow to place it beneath.
        page.Controls.Add(actionsCard);
        page.Controls.Add(cardFlow);
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
        headerRefreshButton.Enabled = enabled;
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
        statusDotLabel.ForeColor = runtimeStatus.Running ? Theme.Success : Theme.Muted;
        railStatusDot.ForeColor = runtimeStatus.Running ? Theme.Success : Theme.Muted;
        railStatusText.Text = runtimeStatus.Running ? "Running" : "Stopped";
        railStatusText.ForeColor = runtimeStatus.Running ? Theme.Text : Theme.Muted;

        runModeValueLabel.Text = settings.RunMode;
        restUrlValueLabel.Text = $"{settings.BindAddress}:{settings.RestPort}";
        grpcValueLabel.Text = settings.GrpcPort > 0 ? settings.GrpcPort.ToString() : "disabled";
        variantValueLabel.Text = settings.PackageVariant;

        healthValueLabel.Text = "Checking...";
        healthDotLabel.ForeColor = Theme.Muted;
        modelsValueLabel.Text = "-";

        var health = await controller.CheckHealthAsync();
        healthValueLabel.Text = health.Ok ? "Healthy" : "Unreachable";
        healthDotLabel.ForeColor = health.Ok ? Theme.Success : Theme.Danger;
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

    private Panel BuildSettingsPage()
    {
        var page = new Panel { BackColor = Theme.WindowBackground, AutoScroll = true };

        var card = new CardPanel { Dock = DockStyle.Top, Height = 430 };
        var cardLayout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            AutoSize = false,
            Padding = new Padding(4, 4, 4, 4)
        };
        cardLayout.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        cardLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

        var sectionTitle = new Label { Text = "Server settings", Font = Theme.CardTitleFont, ForeColor = Theme.Text, AutoSize = true, Margin = new Padding(0, 0, 0, 12) };
        cardLayout.Controls.Add(sectionTitle, 0, 0);
        cardLayout.SetColumnSpan(sectionTitle, 2);

        void AddLabel(string text, int row)
        {
            cardLayout.Controls.Add(new Label { Text = text, AutoSize = true, ForeColor = Theme.Text, Margin = new Padding(3, 10, 16, 3) }, 0, row);
        }

        var row = 1;
        AddLabel("REST port:", row);
        restPortInput = new NumericUpDown { Minimum = 1, Maximum = 65535, Width = 120, Margin = new Padding(3, 6, 3, 3) };
        cardLayout.Controls.Add(restPortInput, 1, row++);

        AddLabel("gRPC port (0 = disabled):", row);
        grpcPortInput = new NumericUpDown { Minimum = 0, Maximum = 65535, Width = 120, Margin = new Padding(3, 6, 3, 3) };
        cardLayout.Controls.Add(grpcPortInput, 1, row++);

        AddLabel("Bind address:", row);
        bindAddressInput = new TextBox { Width = 240, Margin = new Padding(3, 6, 3, 3) };
        cardLayout.Controls.Add(bindAddressInput, 1, row++);

        AddLabel("Log level:", row);
        logLevelInput = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 140, Margin = new Padding(3, 6, 3, 3) };
        logLevelInput.Items.AddRange(new object[] { "ERROR", "WARNING", "INFO", "DEBUG", "TRACE" });
        cardLayout.Controls.Add(logLevelInput, 1, row++);

        AddLabel("Log path:", row);
        logPathInput = new TextBox { Width = 420, Margin = new Padding(3, 6, 3, 3) };
        cardLayout.Controls.Add(logPathInput, 1, row++);

        AddLabel("Model repository path:", row);
        modelRepoInput = new TextBox { Width = 420, Margin = new Padding(3, 6, 3, 3) };
        cardLayout.Controls.Add(modelRepoInput, 1, row++);

        AddLabel("Startup mode:", row);
        runModeInput = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 180, Margin = new Padding(3, 6, 3, 3) };
        runModeInput.Items.AddRange(new object[] { "user-login", "service", "manual" });
        cardLayout.Controls.Add(runModeInput, 1, row++);

        AddLabel("Show tray icon:", row);
        showTrayCheckBox = new CheckBox { Margin = new Padding(3, 10, 3, 3) };
        cardLayout.Controls.Add(showTrayCheckBox, 1, row++);

        AddLabel("Start at login:", row);
        startAtLoginCheckBox = new CheckBox { Margin = new Padding(3, 10, 3, 3) };
        cardLayout.Controls.Add(startAtLoginCheckBox, 1, row++);

        var saveButton = CreatePrimaryButton(Glyphs.Save, "Save", Theme.Accent);
        saveButton.Margin = new Padding(3, 18, 3, 3);
        saveButton.Click += async (_, _) => await SaveSettingsAsync();
        cardLayout.Controls.Add(new Label(), 0, row);
        cardLayout.Controls.Add(saveButton, 1, row++);

        card.Controls.Add(cardLayout);
        page.Controls.Add(card);
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

    private Panel BuildLogsPage()
    {
        var page = new Panel { BackColor = Theme.WindowBackground };

        var toolbar = new FlowLayoutPanel { Dock = DockStyle.Top, Height = 40, FlowDirection = FlowDirection.LeftToRight, Padding = new Padding(0, 0, 0, 8) };
        var refreshLogsButton = CreateIconButton(Glyphs.Refresh, "Refresh");
        var openFolderButton = CreateIconButton(Glyphs.OpenFolder, "Open Folder");
        refreshLogsButton.Margin = new Padding(0, 0, 8, 0);

        refreshLogsButton.Click += async (_, _) => await RefreshLogsAsync();
        openFolderButton.Click += (_, _) => OpenFolderFor(TryLoadSettings()?.LogPath);

        toolbar.Controls.Add(refreshLogsButton);
        toolbar.Controls.Add(openFolderButton);

        var logCard = new CardPanel { Dock = DockStyle.Fill, Padding = new Padding(1), CardBackColor = Theme.LogBackground, BorderColor = Theme.LogBackground };
        logTextBox = new TextBox
        {
            Dock = DockStyle.Fill,
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Both,
            WordWrap = false,
            BorderStyle = BorderStyle.None,
            BackColor = Theme.LogBackground,
            ForeColor = Theme.LogForeground,
            Font = Theme.MonoFont
        };
        logCard.Controls.Add(logTextBox);

        page.Controls.Add(logCard);
        page.Controls.Add(toolbar);
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

    private Panel BuildAdvancedPage()
    {
        var page = new Panel { BackColor = Theme.WindowBackground };

        var card = new CardPanel { Dock = DockStyle.Top, Height = 260 };
        var cardLayout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3 };
        cardLayout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        cardLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        cardLayout.RowStyles.Add(new RowStyle(SizeType.AutoSize));

        var sectionTitle = new Label { Text = "Effective command line", Font = Theme.CardTitleFont, ForeColor = Theme.Text, AutoSize = true, Margin = new Padding(0, 0, 0, 8) };
        cardLayout.Controls.Add(sectionTitle, 0, 0);

        advancedTextBox = new TextBox
        {
            Dock = DockStyle.Fill,
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Both,
            WordWrap = false,
            BackColor = Theme.WindowBackground,
            BorderStyle = BorderStyle.FixedSingle,
            Font = Theme.MonoFont
        };
        cardLayout.Controls.Add(advancedTextBox, 0, 1);

        advancedStatusLabel = new Label { Text = "", ForeColor = Theme.Muted, AutoSize = true, Margin = new Padding(0, 8, 0, 0) };
        cardLayout.Controls.Add(advancedStatusLabel, 0, 2);

        card.Controls.Add(cardLayout);

        var buttonPanel = new FlowLayoutPanel { Dock = DockStyle.Top, Height = 56, Padding = new Padding(0, 16, 0, 0) };
        var repairButton = CreatePrimaryButton(Glyphs.Repair, "Repair Package", Theme.Accent);
        var exportButton = CreateSecondaryButton("Export Diagnostics");
        var validateButton = CreateSecondaryButton("Validate Environment");
        repairButton.Margin = new Padding(0, 0, 10, 0);
        exportButton.Margin = new Padding(0, 0, 10, 0);

        repairButton.Click += async (_, _) => await RunAdvancedActionAsync("Repair", () => controller.Repair());
        validateButton.Click += async (_, _) => await RunAdvancedActionAsync("Validate Environment", () => controller.ValidateEnvironment());
        exportButton.Click += async (_, _) => await ExportDiagnosticsAsync();

        buttonPanel.Controls.Add(repairButton);
        buttonPanel.Controls.Add(exportButton);
        buttonPanel.Controls.Add(validateButton);

        page.Controls.Add(buttonPanel);
        page.Controls.Add(card);
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
            advancedStatusLabel.ForeColor = Theme.Success;
            advancedStatusLabel.Text = $"{title}: completed.";
            MessageBox.Show(this, string.IsNullOrWhiteSpace(output) ? "Completed." : output, title, MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
        catch (Exception ex)
        {
            advancedStatusLabel.ForeColor = Theme.Danger;
            advancedStatusLabel.Text = $"{title}: failed.";
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
            advancedStatusLabel.ForeColor = Theme.Success;
            advancedStatusLabel.Text = "Export Diagnostics: completed.";
            MessageBox.Show(this, $"Diagnostics exported to:{Environment.NewLine}{zipPath}", "Export Diagnostics", MessageBoxButtons.OK, MessageBoxIcon.Information);
            OpenFolderFor(zipPath);
        }
        catch (Exception ex)
        {
            advancedStatusLabel.ForeColor = Theme.Danger;
            advancedStatusLabel.Text = "Export Diagnostics: failed.";
            MessageBox.Show(this, ex.Message, "Export Diagnostics", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}

/// <summary>
/// Flat nav-rail button: glyph + label, with a 4px accent left-bar and
/// tinted background when selected, subtle hover highlight otherwise.
/// </summary>
internal sealed class NavButton : Panel
{
    public string PageName { get; }
    private readonly string glyph;
    private readonly Label glyphLabel;
    private readonly Label textLabel;
    private bool selected;
    private bool hovered;

    public bool Selected
    {
        get => selected;
        set
        {
            selected = value;
            ApplyState();
        }
    }

    public NavButton(string pageName, string glyph)
    {
        PageName = pageName;
        this.glyph = glyph;

        Dock = DockStyle.Top;
        Height = 40;
        Cursor = Cursors.Hand;
        SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.OptimizedDoubleBuffer, true);

        glyphLabel = new Label
        {
            Text = glyph,
            Font = Theme.IconFont(14f),
            AutoSize = false,
            Size = new Size(36, 40),
            TextAlign = ContentAlignment.MiddleCenter,
            Location = new Point(10, 0),
            BackColor = Color.Transparent
        };
        textLabel = new Label
        {
            Text = pageName,
            Font = Theme.NavFont,
            AutoSize = false,
            Size = new Size(150, 40),
            TextAlign = ContentAlignment.MiddleLeft,
            Location = new Point(44, 0),
            BackColor = Color.Transparent
        };

        Controls.Add(glyphLabel);
        Controls.Add(textLabel);

        foreach (var c in new Control[] { this, glyphLabel, textLabel })
        {
            c.Click += (_, _) => OnClick(EventArgs.Empty);
            c.MouseEnter += (_, _) => { hovered = true; ApplyState(); };
            c.MouseLeave += (_, _) => { hovered = false; ApplyState(); };
        }

        ApplyState();
    }

    private void ApplyState()
    {
        if (selected)
        {
            BackColor = Theme.AccentTint;
            glyphLabel.ForeColor = Theme.Accent;
            textLabel.ForeColor = Theme.Accent;
            textLabel.Font = Theme.SemiboldFont;
        }
        else
        {
            BackColor = hovered ? Theme.HoverTint : Color.Transparent;
            glyphLabel.ForeColor = Theme.Muted;
            textLabel.ForeColor = Theme.Muted;
            textLabel.Font = Theme.NavFont;
        }
        Invalidate();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        if (selected)
        {
            using var brush = new SolidBrush(Theme.Accent);
            e.Graphics.FillRectangle(brush, 0, 0, 4, Height);
        }
    }
}
