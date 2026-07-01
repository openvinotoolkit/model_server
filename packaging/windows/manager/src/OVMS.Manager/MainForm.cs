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
    private Label settingsStatusLabel = null!;

    // Logs controls
    private TextBox logTextBox = null!;
    private Label logPathLabel = null!;
    private CheckBox autoScrollCheckBox = null!;
    private Label logStatusLabel = null!;

    // Advanced controls
    private TextBox advancedTextBox = null!;
    private Label advancedStatusLabel = null!;

    // Updates controls
    private Label updatesBaseInstalledVersionValue = null!;
    private Label updatesBaseLatestVersionValue = null!;
    private Label updatesModelServerInstalledVersionValue = null!;
    private Label updatesModelServerLatestVersionValue = null!;
    private Label updatesGenAiInstalledVersionValue = null!;
    private Label updatesGenAiLatestVersionValue = null!;
    private Label updatesVariantValue = null!;
    private Label updatesManagerVersionValue = null!;
    private Label updatesStatusLabel = null!;
    private CardPanel updatesBaseCard = null!;
    private CardPanel updatesModelServerCard = null!;
    private CardPanel updatesGenAiCard = null!;
    private CollapsibleActionRow updatesBaseInstallRow;
    private CollapsibleActionRow updatesModelServerInstallRow;
    private CollapsibleActionRow updatesGenAiInstallRow;
    private Button updatesCheckButton = null!;
    private Button updatesBaseInstallButton = null!;
    private Button updatesModelServerInstallButton = null!;
    private Button updatesGenAiInstallButton = null!;
    private Button updatesModelServerReleaseButton = null!;
    private Button updatesGenAiReleaseButton = null!;
    private string updatesModelServerReleaseUrl = "";
    private string updatesGenAiReleaseUrl = "";

    public MainForm(OvmsController controller)
    {
        this.controller = controller;

        Text = "OpenVINO Model Server Manager";
        MinimumSize = new Size(1020, 680);
        Size = new Size(1220, 780);
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
            Dock = DockStyle.Fill,
            FlowDirection = FlowDirection.TopDown,
            WrapContents = false,
            AutoSize = false,
            AutoScroll = true,
            BackColor = Theme.Rail,
            Padding = new Padding(0, 8, 0, 0)
        };
        navFlow.Controls.Add(CreateNavButton("Dashboard", Glyphs.Dashboard));
        navFlow.Controls.Add(CreateNavButton("Settings", Glyphs.Settings));
        navFlow.Controls.Add(CreateNavButton("Logs", Glyphs.Logs));
        navFlow.Controls.Add(CreateNavButton("Advanced", Glyphs.Advanced));
        navFlow.Controls.Add(CreateNavButton("Updates", Glyphs.Update));

        var statusPill = new Panel { Dock = DockStyle.Bottom, Height = 56, BackColor = Theme.Rail, Padding = new Padding(16, 12, 16, 12) };
        var pillInner = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
        railStatusDot = new Label { Text = "●", ForeColor = Theme.Muted, Font = new Font("Segoe UI", 10f), AutoSize = true, Margin = new Padding(0, 3, 6, 0) };
        railStatusText = new Label { Text = "Checking...", ForeColor = Theme.Muted, Font = Theme.BaseFont, AutoSize = true, Margin = new Padding(0, 4, 0, 0) };
        pillInner.Controls.Add(railStatusDot);
        pillInner.Controls.Add(railStatusText);
        statusPill.Controls.Add(pillInner);

        // Dock carving order matters: WinForms carves docked children from the
        // LAST-added control backwards. We add navFlow (Fill) FIRST so it is
        // carved first and would normally claim the whole client area; but
        // controls added AFTER it (statusPill=Bottom, headerBlock=Top) are
        // carved out of navRail's bounds BEFORE navFlow's Fill is resolved,
        // because Fill is always resolved last regardless of add order for
        // DockStyle.Fill vs the other dock styles -- WinForms processes
        // Top/Bottom/Left/Right docked siblings first (in reverse add order)
        // and only then gives the remaining space to the Fill control. So the
        // actual requirement is just that statusPill/headerBlock are added
        // AFTER navFlow so they end up with a higher z-order and get their
        // Top/Bottom slices carved out of the rail before Fill claims the rest.
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

        pages["Dashboard"] = BuildResponsiveDashboardPage();
        pages["Settings"] = BuildSettingsPage();
        pages["Logs"] = BuildLogsPage();
        pages["Advanced"] = BuildAdvancedPage();
        pages["Updates"] = BuildUpdatesPage();

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
            else if (name == "Updates")
            {
                RefreshUpdatesInfo();
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
            case "Updates":
                RefreshUpdatesInfo();
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
        var button = new RuntimeIconButton(glyph, Theme.Muted)
        {
            BackColor = Theme.Surface,
            Size = new Size(42, 38),
            MinimumSize = new Size(42, 38),
            MaximumSize = new Size(42, 38),
            Margin = new Padding(0)
        };
        button.Font = Theme.IconFont(12.5f);
        var tip = new ToolTip();
        tip.SetToolTip(button, tooltip);
        return button;
    }

    private static Button CreatePrimaryButton(string glyph, string text, Color fillColor)
    {
        var button = new Button
        {
            Text = string.IsNullOrEmpty(glyph) ? text : glyph + "   " + text,
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

    private static Button CreateRuntimeIconButton(string glyph, string tooltip, Color fillColor)
    {
        var button = new RuntimeIconButton(glyph, fillColor)
        {
            BackColor = Theme.Surface,
            Size = new Size(58, 52),
            MinimumSize = new Size(58, 52),
            MaximumSize = new Size(58, 52),
            Margin = new Padding(4, 0, 0, 0)
        };
        var tip = new ToolTip();
        tip.SetToolTip(button, tooltip);
        return button;
    }

    private static void SetRuntimeIconButtonState(Button button, bool enabled)
    {
        var activeColor = button.Tag is Color color ? color : Theme.Text;
        button.Enabled = enabled;
        button.ForeColor = enabled ? activeColor : Color.FromArgb(145, Theme.Muted);
        button.BackColor = Theme.Surface;
        button.Cursor = enabled ? Cursors.Hand : Cursors.Default;
        if (button is RuntimeIconButton iconButton)
        {
            iconButton.ActiveColor = activeColor;
            iconButton.DisabledColor = Color.FromArgb(145, Theme.Muted);
            iconButton.Invalidate();
        }
        else
        {
            button.FlatAppearance.MouseOverBackColor = enabled ? Theme.HoverTint : Theme.Surface;
            button.FlatAppearance.MouseDownBackColor = enabled ? Theme.WindowBackground : Theme.Surface;
        }
    }

    private static Button CreateSecondaryButton(string text)
    {
        var button = new Button
        {
            Text = text,
            Font = Theme.NavFont,
            FlatStyle = FlatStyle.Flat,
            ForeColor = Theme.Text,
            BackColor = Theme.Surface,
            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowAndShrink,
            Padding = new Padding(16, 8, 16, 8),
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

    private Panel BuildResponsiveDashboardPage()
    {
        var page = new Panel { BackColor = Theme.WindowBackground, AutoScroll = true };

        var dashboardStack = new TableLayoutPanel
        {
            Location = new Point(0, 0),
            AutoSize = false,
            ColumnCount = 1,
            RowCount = 1,
            Height = 456,
            Padding = new Padding(0),
            Margin = new Padding(0)
        };
        dashboardStack.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        dashboardStack.RowStyles.Add(new RowStyle(SizeType.Absolute, 456));

        var metricGrid = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            AutoSize = false,
            ColumnCount = 2,
            RowCount = 4,
            Padding = new Padding(0),
            Margin = new Padding(0)
        };
        for (var i = 0; i < 2; i++)
        {
            metricGrid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        }

        for (var i = 0; i < 4; i++)
        {
            metricGrid.RowStyles.Add(new RowStyle(SizeType.Absolute, 112));
        }

        startButton = CreateRuntimeIconButton(Glyphs.Play, "Start server", Theme.Success);
        stopButton = CreateRuntimeIconButton(Glyphs.Stop, "Stop server", Theme.Danger);
        restartButton = CreateRuntimeIconButton(Glyphs.Refresh, "Restart server", Theme.Accent);

        startButton.Click += async (_, _) => await RunControlActionAsync("Start", controller.Start);
        stopButton.Click += async (_, _) => await RunControlActionAsync("Stop", controller.Stop);
        restartButton.Click += async (_, _) => await RunControlActionAsync("Restart", controller.Restart);

        (Label dot, Label value) AddControlStatusCard(int column, int row)
        {
            var card = new CardPanel
            {
                Dock = DockStyle.Fill,
                Margin = new Padding(0, 0, 16, 16),
                Padding = new Padding(18, 14, 18, 14)
            };

            var cardLayout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2,
                RowCount = 1,
                BackColor = Color.Transparent,
                Margin = new Padding(0),
                Padding = new Padding(0)
            };
            cardLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            cardLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 230));
            cardLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

            var statusLayout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2,
                RowCount = 2,
                BackColor = Color.Transparent,
                Margin = new Padding(0),
                Padding = new Padding(0)
            };
            statusLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 20));
            statusLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            statusLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 28));
            statusLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

            var titleLabel = new Label
            {
                Text = "Status",
                Font = Theme.CardTitleFont,
                ForeColor = Theme.Muted,
                AutoEllipsis = true,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleLeft,
                Margin = new Padding(0)
            };
            var dotLabel = new Label
            {
                Text = "\u25CF",
                Font = new Font("Segoe UI", 10.5f),
                ForeColor = Theme.Muted,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleLeft,
                Margin = new Padding(0, 5, 0, 0)
            };
            var valueLabel = new Label
            {
                Text = "-",
                Font = Theme.CardValueFont,
                ForeColor = Theme.Text,
                AutoEllipsis = true,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleLeft,
                Margin = new Padding(0, 2, 0, 0)
            };

            statusLayout.Controls.Add(titleLabel, 0, 0);
            statusLayout.SetColumnSpan(titleLabel, 2);
            statusLayout.Controls.Add(dotLabel, 0, 1);
            statusLayout.Controls.Add(valueLabel, 1, 1);

            var actionLayout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 4,
                RowCount = 3,
                BackColor = Color.Transparent,
                Margin = new Padding(0),
                Padding = new Padding(0, 0, 6, 0)
            };
            actionLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            actionLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 62));
            actionLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 62));
            actionLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 62));
            actionLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
            actionLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 52));
            actionLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
            actionLayout.Controls.Add(startButton, 1, 1);
            actionLayout.Controls.Add(stopButton, 2, 1);
            actionLayout.Controls.Add(restartButton, 3, 1);

            cardLayout.Controls.Add(statusLayout, 0, 0);
            cardLayout.Controls.Add(actionLayout, 1, 0);
            card.Controls.Add(cardLayout);
            metricGrid.Controls.Add(card, column, row);
            return (dotLabel, valueLabel);
        }

        (Label dot, Label value) AddStatusCard(string title, string initialValue, bool withDot, int column, int row)
        {
            var card = new CardPanel
            {
                Dock = DockStyle.Fill,
                Margin = new Padding(0, 0, 16, 16),
                Padding = new Padding(18, 14, 18, 14)
            };

            var cardLayout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = withDot ? 2 : 1,
                RowCount = 2,
                BackColor = Color.Transparent,
                Margin = new Padding(0),
                Padding = new Padding(0)
            };
            if (withDot)
            {
                cardLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 20));
                cardLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            }
            else
            {
                cardLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            }
            cardLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 28));
            cardLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

            var titleLabel = new Label
            {
                Text = title,
                Font = Theme.CardTitleFont,
                ForeColor = Theme.Muted,
                AutoEllipsis = true,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleLeft,
                Margin = new Padding(0)
            };
            var valueLabel = new Label
            {
                Text = initialValue,
                Font = Theme.CardValueFont,
                ForeColor = Theme.Text,
                AutoEllipsis = true,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleLeft,
                Margin = new Padding(0, 2, 0, 0)
            };

            Label? dotLabel = null;
            if (withDot)
            {
                dotLabel = new Label
                {
                    Text = "\u25CF",
                    Font = new Font("Segoe UI", 10.5f),
                    ForeColor = Theme.Muted,
                    Dock = DockStyle.Fill,
                    TextAlign = ContentAlignment.MiddleLeft,
                    Margin = new Padding(0, 5, 0, 0)
                };
                cardLayout.Controls.Add(titleLabel, 0, 0);
                cardLayout.SetColumnSpan(titleLabel, 2);
                cardLayout.Controls.Add(dotLabel, 0, 1);
                cardLayout.Controls.Add(valueLabel, 1, 1);
            }
            else
            {
                cardLayout.Controls.Add(titleLabel, 0, 0);
                cardLayout.Controls.Add(valueLabel, 0, 1);
            }

            card.Controls.Add(cardLayout);
            metricGrid.Controls.Add(card, column, row);
            return (dotLabel!, valueLabel);
        }

        (statusDotLabel, statusValueLabel) = AddControlStatusCard(0, 0);
        (_, restUrlValueLabel) = AddStatusCard("REST endpoint", "-", withDot: false, 1, 0);
        (_, grpcValueLabel) = AddStatusCard("gRPC port", "-", withDot: false, 0, 1);
        (_, modelsValueLabel) = AddStatusCard("Models served", "-", withDot: false, 1, 1);
        (healthDotLabel, healthValueLabel) = AddStatusCard("Health", "-", withDot: true, 0, 2);
        (_, variantValueLabel) = AddStatusCard("Package variant", "-", withDot: false, 1, 2);
        (_, runModeValueLabel) = AddStatusCard("Runtime mode", "-", withDot: false, 0, 3);

        dashboardStack.Controls.Add(metricGrid, 0, 0);
        page.Controls.Add(dashboardStack);

        void ResizeDashboard()
        {
            var availableWidth = ClientSize.Width - navRail.Width - contentHost.Padding.Horizontal - 28;
            dashboardStack.Width = Math.Max(560, availableWidth);
        }

        page.Resize += (_, _) => ResizeDashboard();
        Resize += (_, _) => ResizeDashboard();
        ResizeDashboard();
        return page;
    }

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
            var card = CreateCard(250, 96);
            var titleLabel = new Label { Text = title, Font = Theme.CardTitleFont, ForeColor = Theme.Muted, AutoSize = true, Location = new Point(16, 14) };
            Label? dotLabel = null;
            var valueLabel = new Label { Text = initialValue, Font = Theme.CardValueFont, ForeColor = Theme.Text, AutoSize = true, Location = new Point(withDot ? 34 : 16, 46) };
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

        var actionsCard = new CardPanel { Dock = DockStyle.Top, Height = 84, Margin = new Padding(0, 8, 0, 0) };
        var actionsFlow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false, AutoSize = false, AutoScroll = true, Padding = new Padding(12, 10, 12, 10) };

        startButton = CreatePrimaryButton("▶", "Start", Theme.Success);
        stopButton = CreatePrimaryButton("■", "Stop", Theme.Danger);
        restartButton = CreatePrimaryButton("↻", "Restart", Theme.Accent);
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
        if (!enabled)
        {
            SetRuntimeIconButtonState(startButton, false);
            SetRuntimeIconButtonState(stopButton, false);
            SetRuntimeIconButtonState(restartButton, false);
        }
        headerRefreshButton.Enabled = enabled;
    }

    public async Task RefreshDashboardAsync()
    {
        var settings = TryLoadSettings();
        if (settings is null)
        {
            statusValueLabel.Text = "Settings unavailable";
            SetRuntimeIconButtonState(startButton, false);
            SetRuntimeIconButtonState(stopButton, false);
            SetRuntimeIconButtonState(restartButton, false);
            return;
        }

        var runtimeStatus = await Task.Run(controller.GetRuntimeStatus);
        statusValueLabel.Text = runtimeStatus.Running ? $"Running (pid {runtimeStatus.Pid})" : "Stopped";
        statusDotLabel.ForeColor = runtimeStatus.Running ? Theme.Success : Theme.Muted;
        railStatusDot.ForeColor = runtimeStatus.Running ? Theme.Success : Theme.Muted;
        railStatusText.Text = runtimeStatus.Running ? "Running" : "Stopped";
        railStatusText.ForeColor = runtimeStatus.Running ? Theme.Text : Theme.Muted;
        SetRuntimeIconButtonState(startButton, !runtimeStatus.Running);
        SetRuntimeIconButtonState(stopButton, runtimeStatus.Running);
        SetRuntimeIconButtonState(restartButton, runtimeStatus.Running);

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
        var page = new Panel { BackColor = Theme.WindowBackground, AutoScroll = true, Padding = new Padding(0, 0, 10, 0) };

        var serverCard = CreateSettingsCard(650);
        var serverLayout = CreateSettingsTable();

        var row = 0;
        AddSettingsCardHeader(serverLayout, "Server", "Configure endpoints, logging, and where OVMS loads models from.", row++);
        AddSettingsSectionHeader(serverLayout, "Network", row++);
        restPortInput = new NumericUpDown { Minimum = 1, Maximum = 65535, Width = 150, Anchor = AnchorStyles.Left };
        StyleSettingsField(restPortInput);
        AddSettingsRow(serverLayout, "REST port", restPortInput, row++);
        grpcPortInput = new NumericUpDown { Minimum = 0, Maximum = 65535, Width = 150, Anchor = AnchorStyles.Left };
        StyleSettingsField(grpcPortInput);
        AddSettingsRow(serverLayout, "gRPC port", grpcPortInput, row++);
        AddHelperRow(serverLayout, "Set to 0 to disable the gRPC endpoint.", row++);
        bindAddressInput = new TextBox { Anchor = AnchorStyles.Left | AnchorStyles.Right };
        StyleSettingsField(bindAddressInput);
        AddSettingsRow(serverLayout, "Bind address", bindAddressInput, row++);
        AddHelperRow(serverLayout, "Use 127.0.0.1 for local-only access. Other addresses may expose the server to your network.", row++);

        AddSettingsSectionHeader(serverLayout, "Logging and models", row++);
        logLevelInput = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 170, Anchor = AnchorStyles.Left, FlatStyle = FlatStyle.Standard };
        StyleSettingsField(logLevelInput);
        logLevelInput.Items.AddRange(new object[] { "ERROR", "WARNING", "INFO", "DEBUG", "TRACE" });
        AddSettingsRow(serverLayout, "Log level", logLevelInput, row++);

        logPathInput = new TextBox { Anchor = AnchorStyles.Left | AnchorStyles.Right };
        StyleSettingsField(logPathInput);
        var browseLogButton = CreateSecondaryButton("Browse...");
        browseLogButton.Click += (_, _) =>
        {
            using var dialog = new SaveFileDialog
            {
                Title = "Select log file",
                CheckFileExists = false,
                OverwritePrompt = false,
                FileName = Path.GetFileName(logPathInput.Text),
                InitialDirectory = SafeDirectoryName(logPathInput.Text),
                Filter = "Log files (*.log)|*.log|All files (*.*)|*.*"
            };
            if (dialog.ShowDialog(this) == DialogResult.OK)
            {
                logPathInput.Text = dialog.FileName;
            }
        };
        AddSettingsRowWithButton(serverLayout, "Log path", logPathInput, browseLogButton, row++);

        modelRepoInput = new TextBox { Anchor = AnchorStyles.Left | AnchorStyles.Right };
        StyleSettingsField(modelRepoInput);
        var browseModelRepoButton = CreateSecondaryButton("Browse...");
        browseModelRepoButton.Click += (_, _) =>
        {
            using var dialog = new FolderBrowserDialog
            {
                Description = "Select model repository folder",
                SelectedPath = Directory.Exists(modelRepoInput.Text) ? modelRepoInput.Text : ""
            };
            if (dialog.ShowDialog(this) == DialogResult.OK)
            {
                modelRepoInput.Text = dialog.SelectedPath;
            }
        };
        var openModelRepoButton = CreateSecondaryButton("Open");
        openModelRepoButton.Click += (_, _) => OpenFolder(modelRepoInput.Text);
        AddSettingsRowWithButtons(serverLayout, "Model folder", modelRepoInput, new[] { browseModelRepoButton, openModelRepoButton }, row++);

        serverCard.Controls.Add(serverLayout);

        var startupCard = CreateSettingsCard(330);
        var startupLayout = CreateSettingsTable();

        row = 0;
        AddSettingsCardHeader(startupLayout, "Startup tray", "Choose how the manager and server behave when Windows starts.", row++);
        AddSettingsSectionHeader(startupLayout, "Launch behavior", row++);
        runModeInput = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 220, Anchor = AnchorStyles.Left, FlatStyle = FlatStyle.Standard };
        StyleSettingsField(runModeInput);
        runModeInput.Items.AddRange(new object[] { "user-login", "service", "manual" });
        AddSettingsRow(startupLayout, "Startup mode", runModeInput, row++);

        showTrayCheckBox = new CheckBox { Text = "Show OVMS in the Windows notification area", AutoSize = true, Anchor = AnchorStyles.Left, Margin = new Padding(3, 8, 3, 5), FlatStyle = FlatStyle.System, BackColor = Theme.Surface };
        AddSettingsRow(startupLayout, "Tray icon", showTrayCheckBox, row++);

        startAtLoginCheckBox = new CheckBox { Text = "Launch OVMS Manager when you sign in", AutoSize = true, Anchor = AnchorStyles.Left, Margin = new Padding(3, 8, 3, 5), FlatStyle = FlatStyle.System, BackColor = Theme.Surface };
        AddSettingsRow(startupLayout, "Start at login", startAtLoginCheckBox, row++);

        startupCard.Controls.Add(startupLayout);

        var footer = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            Height = 82,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false,
            Padding = new Padding(18, 16, 18, 14),
            Margin = new Padding(0, 2, 0, 0),
            BackColor = Theme.WindowBackground
        };
        var footerRule = new Panel { Dock = DockStyle.Top, Height = 1, BackColor = Theme.Border, Margin = new Padding(0) };
        var saveButton = CreatePrimaryButton("", "Save changes", Theme.Accent);
        var resetButton = CreateSecondaryButton("Reset");
        saveButton.Margin = new Padding(0, 0, 10, 0);
        resetButton.Margin = new Padding(0, 0, 16, 0);
        settingsStatusLabel = new Label
        {
            Text = "",
            AutoSize = true,
            Font = Theme.SemiboldFont,
            Margin = new Padding(0, 10, 0, 0)
        };

        saveButton.Click += async (_, _) => await SaveSettingsAsync();
        resetButton.Click += (_, _) =>
        {
            LoadSettingsIntoForm();
            settingsStatusLabel.ForeColor = Theme.Muted;
            settingsStatusLabel.Text = "Reloaded from disk.";
        };

        footer.Controls.Add(saveButton);
        footer.Controls.Add(resetButton);
        footer.Controls.Add(settingsStatusLabel);

        // Dock-Top stacking: add bottom-most visual element first so later
        // Top-docked controls are carved above it (see BuildShell comment).
        page.Controls.Add(footer);
        page.Controls.Add(footerRule);
        page.Controls.Add(startupCard);
        page.Controls.Add(serverCard);
        return page;
    }

    private static CardPanel CreateSettingsCard(int height)
    {
        return new CardPanel
        {
            Dock = DockStyle.Top,
            Height = height,
            Margin = new Padding(0, 0, 0, 26),
            Padding = new Padding(26, 22, 26, 24),
            CardBackColor = Theme.Surface,
            BorderColor = Color.FromArgb(222, 226, 232)
        };
    }

    private static Panel CreateCardSpacer(int height = 18)
    {
        return new Panel
        {
            Dock = DockStyle.Top,
            Height = height,
            BackColor = Theme.WindowBackground
        };
    }

    private static TableLayoutPanel CreateSettingsTable()
    {
        var table = new TableLayoutPanel
        {
            Dock = DockStyle.Top,
            ColumnCount = 2,
            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowAndShrink,
            Padding = new Padding(0),
            BackColor = Theme.Surface
        };
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 190));
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        return table;
    }

    private static void AddSettingsCardHeader(TableLayoutPanel table, string title, string subtitle, int row)
    {
        table.RowStyles.Add(new RowStyle(SizeType.Absolute, 78));
        var header = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 2,
            Margin = new Padding(0),
            BackColor = Theme.Surface
        };
        header.RowStyles.Add(new RowStyle(SizeType.Absolute, 28));
        header.RowStyles.Add(new RowStyle(SizeType.Absolute, 30));
        header.Controls.Add(new Label
        {
            Text = title,
            AutoSize = false,
            Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.MiddleLeft,
            Font = Theme.CardTitleFont,
            ForeColor = Theme.Text,
            BackColor = Theme.Surface,
            Margin = new Padding(0)
        }, 0, 0);
        header.Controls.Add(new Label
        {
            Text = subtitle,
            AutoSize = false,
            Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.TopLeft,
            Font = new Font(Theme.BaseFont.FontFamily, 9.25f),
            ForeColor = Theme.Muted,
            BackColor = Theme.Surface,
            Margin = new Padding(0)
        }, 0, 1);

        table.Controls.Add(header, 0, row);
        table.SetColumnSpan(header, 2);
    }

    private static void AddSettingsSectionHeader(TableLayoutPanel table, string text, int row)
    {
        table.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
        var header = new Label
        {
            Text = text,
            AutoSize = false,
            Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.MiddleLeft,
            Font = new Font(Theme.BaseFont.FontFamily, 8.25f, FontStyle.Bold),
            ForeColor = Theme.Accent,
            BackColor = Theme.Surface,
            Margin = new Padding(3, 10, 0, 6)
        };
        table.Controls.Add(header, 0, row);
        table.SetColumnSpan(header, 2);
    }

    private static void AddSettingsRow(TableLayoutPanel table, string label, Control field, int row)
    {
        table.RowStyles.Add(new RowStyle(SizeType.Absolute, 58));
        table.Controls.Add(CreateSettingsLabel(label), 0, row);
        table.Controls.Add(field is CheckBox ? field : CreateCappedSettingsField(field), 1, row);
    }

    private static Label AddReadOnlySettingsRow(TableLayoutPanel table, string label, int row)
    {
        var value = CreateReadOnlyValueLabel();
        AddSettingsRow(table, label, value, row);
        return value;
    }

    private static void AddSettingsRowWithButton(TableLayoutPanel table, string label, Control field, Control button, int row)
    {
        AddSettingsRowWithButtons(table, label, field, new[] { button }, row);
    }

    private static void AddSettingsRowWithButtons(TableLayoutPanel table, string label, Control field, IReadOnlyList<Control> buttons, int row)
    {
        table.RowStyles.Add(new RowStyle(SizeType.Absolute, 84));

        var inline = new Panel
        {
            Dock = DockStyle.Fill,
            Margin = new Padding(0),
            BackColor = Theme.Surface
        };
        var fieldHost = CreateSettingsFieldHost(field);
        fieldHost.Location = new Point(3, 8);
        inline.Controls.Add(fieldHost);

        for (var i = 0; i < buttons.Count; i++)
        {
            buttons[i].Anchor = AnchorStyles.Top | AnchorStyles.Left;
            buttons[i].AutoSize = false;
            buttons[i].Size = new Size(buttons[i].Text == "Open" ? 110 : 150, 52);
            buttons[i].MinimumSize = buttons[i].Size;
            buttons[i].Margin = new Padding(0);
            buttons[i].Location = new Point(0, 6);
            inline.Controls.Add(buttons[i]);
        }

        void LayoutInline()
        {
            var buttonWidth = buttons.Count * 10;
            foreach (var button in buttons)
            {
                buttonWidth += button.Width;
            }
            var fieldWidth = Math.Min(700, Math.Max(420, inline.ClientSize.Width - buttonWidth - 16));
            fieldHost.Size = new Size(fieldWidth, 40);

            var x = fieldHost.Right + 10;
            foreach (var button in buttons)
            {
                button.Location = new Point(x, 6);
                x += button.Width + 10;
            }
        }

        inline.Resize += (_, _) => LayoutInline();
        LayoutInline();

        table.Controls.Add(CreateSettingsLabel(label), 0, row);
        table.Controls.Add(inline, 1, row);
    }

    private static void AddHelperRow(TableLayoutPanel table, string text, int row)
    {
        table.RowStyles.Add(new RowStyle(SizeType.Absolute, 32));
        var helper = new Label
        {
            Text = text,
            AutoSize = false,
            Dock = DockStyle.Fill,
            Font = new Font(Theme.BaseFont.FontFamily, 8.25f),
            ForeColor = Theme.Muted,
            BackColor = Theme.Surface,
            Margin = new Padding(3, 0, 12, 12)
        };
        table.Controls.Add(helper, 1, row);
    }

    private static void AddStatusRow(TableLayoutPanel table, Label statusLabel, int row)
    {
        table.RowStyles.Add(new RowStyle(SizeType.Absolute, 54));
        statusLabel.AutoSize = false;
        statusLabel.Dock = DockStyle.Fill;
        statusLabel.TextAlign = ContentAlignment.MiddleLeft;
        statusLabel.BackColor = Theme.Surface;
        statusLabel.Margin = new Padding(3, 4, 12, 4);
        table.Controls.Add(statusLabel, 1, row);
    }

    private static CollapsibleActionRow AddActionRow(TableLayoutPanel table, string label, string helperText, IReadOnlyList<Control> buttons, int row)
    {
        var stacked = buttons.Count > 1;
        var height = stacked ? 118 : 84;
        var rowStyle = new RowStyle(SizeType.Absolute, height);
        table.RowStyles.Add(rowStyle);
        var labelControl = CreateSettingsLabel(label);
        table.Controls.Add(labelControl, 0, row);

        var content = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = stacked ? 1 : 3,
            RowCount = stacked ? 2 : 1,
            Margin = new Padding(0),
            BackColor = Theme.Surface
        };
        if (stacked)
        {
            content.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            content.RowStyles.Add(new RowStyle(SizeType.Absolute, 36));
            content.RowStyles.Add(new RowStyle(SizeType.Absolute, 66));
        }
        else
        {
            content.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 360));
            content.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            content.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            content.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        }
        content.Controls.Add(new Label
        {
            Text = helperText,
            AutoSize = false,
            Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.MiddleLeft,
            Font = new Font(Theme.BaseFont.FontFamily, 8.25f),
            ForeColor = Theme.Muted,
            BackColor = Theme.Surface,
            Margin = new Padding(3, 0, stacked ? 12 : 18, 0)
        }, 0, 0);

        var buttonFlow = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false,
            BackColor = Theme.Surface,
            Margin = new Padding(0),
            Padding = new Padding(stacked ? 3 : 0, stacked ? 6 : 14, 0, 0),
            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowAndShrink
        };
        foreach (var button in buttons)
        {
            button.AutoSize = false;
            button.Height = 52;
            button.Padding = new Padding(14, 4, 14, 6);
            var preferredWidth = TextRenderer.MeasureText(button.Text, button.Font).Width + 56;
            var minimumWidth = string.Equals(button.Text, "Install Package Update", StringComparison.OrdinalIgnoreCase)
                ? 230
                : button.Text.Length > 12 ? 170 : 140;
            button.Width = Math.Clamp(preferredWidth, minimumWidth, 240);
            button.Margin = new Padding(0, 0, 10, 0);
            buttonFlow.Controls.Add(button);
        }
        content.Controls.Add(buttonFlow, stacked ? 0 : 1, stacked ? 1 : 0);

        table.Controls.Add(content, 1, row);
        return new CollapsibleActionRow(rowStyle, labelControl, content, height);
    }

    private static Label CreateSettingsLabel(string text)
    {
        return new Label
        {
            Text = text,
            AutoSize = false,
            Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.MiddleLeft,
            ForeColor = Color.FromArgb(55, 65, 81),
            Font = new Font(Theme.BaseFont.FontFamily, 9.25f, FontStyle.Bold),
            BackColor = Theme.Surface,
            Margin = new Padding(3, 0, 18, 0)
        };
    }

    private static Label CreateReadOnlyValueLabel()
    {
        return new Label
        {
            Text = "",
            AutoSize = false,
            AutoEllipsis = true,
            Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.MiddleLeft,
            ForeColor = Theme.Text,
            Font = new Font(Theme.BaseFont.FontFamily, 10f),
            BackColor = Theme.FieldBackground,
            Margin = new Padding(10, 0, 10, 0)
        };
    }

    private static Control CreateCappedSettingsField(Control field)
    {
        var host = CreateSettingsFieldHost(field);
        if (field is NumericUpDown or ComboBox)
        {
            return host;
        }

        var wrapper = new Panel
        {
            Dock = DockStyle.Fill,
            BackColor = Theme.Surface,
            Margin = new Padding(0)
        };
        host.Width = 900;
        host.MaximumSize = new Size(900, 40);
        host.Anchor = AnchorStyles.Left | AnchorStyles.Top;
        wrapper.Controls.Add(host);
        return wrapper;
    }

    private static Control CreateSettingsFieldHost(Control field)
    {
        var fixedWidth = field is NumericUpDown or ComboBox;
        var host = new SettingsFieldHost
        {
            Dock = fixedWidth ? DockStyle.None : DockStyle.Top,
            Anchor = fixedWidth ? AnchorStyles.Left : AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Top,
            Height = 40,
            Size = fixedWidth ? new Size(field.Width + 22, 40) : new Size(0, 40),
            MinimumSize = fixedWidth ? new Size(field.Width + 22, 40) : new Size(0, 40),
            Margin = new Padding(3, 8, 12, 8)
        };

        field.Dock = DockStyle.Fill;
        field.Margin = field is ComboBox or NumericUpDown
            ? new Padding(10, 5, 8, 4)
            : new Padding(10, 8, 10, 6);

        if (field is TextBox textBox)
        {
            textBox.BorderStyle = BorderStyle.None;
            textBox.BackColor = Theme.FieldBackground;
        }
        else if (field is NumericUpDown numeric)
        {
            numeric.BorderStyle = BorderStyle.None;
            numeric.BackColor = Theme.FieldBackground;
        }
        else if (field is ComboBox combo)
        {
            combo.FlatStyle = FlatStyle.Flat;
            combo.BackColor = Theme.FieldBackground;
        }

        host.Controls.Add(field);
        return host;
    }

    private static void StyleSettingsField(Control control)
    {
        control.Font = new Font(Theme.BaseFont.FontFamily, 10f);
        control.ForeColor = Theme.Text;
        control.BackColor = Theme.FieldBackground;
        control.Margin = new Padding(0);

        if (control is TextBox textBox)
        {
            textBox.BorderStyle = BorderStyle.None;
        }
        else if (control is NumericUpDown numeric)
        {
            numeric.BorderStyle = BorderStyle.None;
        }
    }

    private static string SafeDirectoryName(string path)
    {
        try
        {
            var dir = Path.GetDirectoryName(path);
            return string.IsNullOrEmpty(dir) ? "" : dir;
        }
        catch
        {
            return "";
        }
    }

    private static bool IsLocalBindAddress(string bindAddress)
    {
        return string.Equals(bindAddress, "127.0.0.1", StringComparison.OrdinalIgnoreCase)
            || string.Equals(bindAddress, "localhost", StringComparison.OrdinalIgnoreCase)
            || string.Equals(bindAddress, "::1", StringComparison.OrdinalIgnoreCase);
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
            settingsStatusLabel.ForeColor = Theme.Danger;
            settingsStatusLabel.Text = "Error: could not load existing settings.";
            MessageBox.Show(this, "Could not load existing settings.", "Save", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        if (string.IsNullOrWhiteSpace(bindAddressInput.Text))
        {
            settingsStatusLabel.ForeColor = Theme.Danger;
            settingsStatusLabel.Text = "Error: bind address cannot be empty.";
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
            settingsStatusLabel.ForeColor = Theme.Danger;
            settingsStatusLabel.Text = $"Error: {ex.Message}";
            MessageBox.Show(this, ex.Message, "Save", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        settingsStatusLabel.ForeColor = Theme.Success;
        settingsStatusLabel.Text = "Saved.";
        if (!IsLocalBindAddress(updated.BindAddress))
        {
            settingsStatusLabel.ForeColor = Theme.Warning;
            settingsStatusLabel.Text = "Saved. Non-local bind address - server may be reachable from other devices.";
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

        var toolbar = new TableLayoutPanel
        {
            Dock = DockStyle.Top,
            Height = 72,
            ColumnCount = 5,
            RowCount = 1,
            Padding = new Padding(0, 0, 0, 10),
            BackColor = Theme.WindowBackground
        };
        toolbar.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 66));
        toolbar.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 66));
        toolbar.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 78));
        toolbar.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 172));
        toolbar.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        toolbar.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

        var openFolderButton = CreateIconButton(Glyphs.OpenFolder, "Open Logs");
        var changeLogPathButton = CreateIconButton(Glyphs.Edit, "Change log location");
        var clearLogsButton = CreateIconButton(Glyphs.Cancel, "Clear (display only)");
        foreach (var button in new[] { openFolderButton, changeLogPathButton, clearLogsButton })
        {
            button.Size = new Size(58, 52);
            button.MinimumSize = new Size(58, 52);
            button.MaximumSize = new Size(58, 52);
        }
        openFolderButton.Margin = new Padding(0, 0, 8, 0);
        changeLogPathButton.Margin = new Padding(0, 0, 8, 0);
        clearLogsButton.Margin = new Padding(0, 0, 18, 0);

        autoScrollCheckBox = new CheckBox
        {
            Text = "Auto-scroll",
            Checked = true,
            AutoSize = true,
            Anchor = AnchorStyles.Left,
            BackColor = Theme.WindowBackground,
            Margin = new Padding(0, 16, 16, 0),
            MinimumSize = new Size(140, 24)
        };

        openFolderButton.Click += (_, _) => OpenFolderFor(TryLoadSettings()?.LogPath);
        changeLogPathButton.Click += async (_, _) => await ChangeLogLocationAsync();
        clearLogsButton.Click += (_, _) =>
        {
            logTextBox.Clear();
            logStatusLabel.Text = "Display cleared (file untouched).";
        };

        logPathLabel = new Label
        {
            Text = "",
            AutoSize = true,
            ForeColor = Theme.Muted,
            AutoEllipsis = true,
            Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.MiddleRight,
            Margin = new Padding(8, 10, 0, 0)
        };

        toolbar.Controls.Add(openFolderButton, 0, 0);
        toolbar.Controls.Add(changeLogPathButton, 1, 0);
        toolbar.Controls.Add(clearLogsButton, 2, 0);
        toolbar.Controls.Add(autoScrollCheckBox, 3, 0);
        toolbar.Controls.Add(logPathLabel, 4, 0);

        var statusBar = new Panel { Dock = DockStyle.Bottom, Height = 34, Padding = new Padding(0, 7, 0, 0) };
        logStatusLabel = new Label { Text = "", AutoSize = false, AutoEllipsis = true, ForeColor = Theme.Muted, Dock = DockStyle.Fill };
        statusBar.Controls.Add(logStatusLabel);

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

        // Dock-Top stacking: add bottom-most visual element first (see
        // BuildShell comment for why this ordering yields correct results).
        page.Controls.Add(logCard);
        page.Controls.Add(statusBar);
        page.Controls.Add(toolbar);
        return page;
    }

    private async Task RefreshLogsAsync()
    {
        var settings = TryLoadSettings();
        logPathLabel.Text = settings?.LogPath ?? "";

        if (settings is null || string.IsNullOrEmpty(settings.LogPath) || !File.Exists(settings.LogPath))
        {
            logTextBox.Text = "(log file not found)";
            logStatusLabel.Text = $"0 lines - last refreshed {DateTime.Now:T}";
            return;
        }

        var lines = await Task.Run(() => controller.TailLog(500));
        logTextBox.Text = string.Join(Environment.NewLine, lines);
        if (autoScrollCheckBox.Checked)
        {
            logTextBox.SelectionStart = logTextBox.Text.Length;
            logTextBox.ScrollToCaret();
        }

        logStatusLabel.Text = $"{lines.Count} lines - last refreshed {DateTime.Now:T}";
    }

    private async Task ChangeLogLocationAsync()
    {
        var current = TryLoadSettings();
        if (current is null)
        {
            MessageBox.Show(this, "Could not load existing settings.", "Change Log Location", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        using var dialog = new SaveFileDialog
        {
            Title = "Select log file",
            CheckFileExists = false,
            OverwritePrompt = false,
            FileName = Path.GetFileName(current.LogPath),
            InitialDirectory = SafeDirectoryName(current.LogPath),
            Filter = "Log files (*.log)|*.log|All files (*.*)|*.*"
        };

        if (dialog.ShowDialog(this) != DialogResult.OK)
        {
            return;
        }

        var updated = new OvmsSettings
        {
            InstallDir = current.InstallDir,
            DataDir = current.DataDir,
            ModelRepositoryPath = current.ModelRepositoryPath,
            ConfigPath = current.ConfigPath,
            LogPath = dialog.FileName,
            RestPort = current.RestPort,
            GrpcPort = current.GrpcPort,
            BindAddress = current.BindAddress,
            LogLevel = current.LogLevel,
            RunMode = current.RunMode,
            StartAtLogin = current.StartAtLogin,
            ShowTrayIcon = current.ShowTrayIcon,
            ServiceAutoStart = current.ServiceAutoStart,
            PackageVariant = current.PackageVariant
        };
        try
        {
            await Task.Run(() => controller.SaveSettings(updated));
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Change Log Location", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        logPathInput.Text = updated.LogPath;
        logPathLabel.Text = updated.LogPath;
        logStatusLabel.Text = "Log location saved.";

        var runtimeStatus = await Task.Run(controller.GetRuntimeStatus);
        if (runtimeStatus.Running)
        {
            var result = MessageBox.Show(this, "The log path changed. Restart server to apply the new log location?",
                "Restart required", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
            if (result == DialogResult.Yes)
            {
                await RunControlActionAsync("Restart", controller.Restart);
            }
        }

        await RefreshLogsAsync();
    }

    // ---------------------------------------------------------------
    // Advanced
    // ---------------------------------------------------------------

    private Label advancedInstallDirValue = null!;
    private Label advancedDataDirValue = null!;
    private Label advancedVersionValue = null!;
    private Label advancedConfigPathValue = null!;

    private Panel BuildAdvancedPage()
    {
        var page = new Panel { BackColor = Theme.WindowBackground, AutoScroll = true, Padding = new Padding(0, 0, 10, 0) };

        var envCard = CreateSettingsCard(500);
        var envLayout = CreateSettingsTable();
        var envRow = 0;
        AddSettingsCardHeader(envLayout, "Environment", "Review the installed manager paths and the command OVMS will run.", envRow++);
        AddSettingsSectionHeader(envLayout, "Paths and versions", envRow++);
        advancedInstallDirValue = AddReadOnlySettingsRow(envLayout, "Install dir", envRow++);
        advancedDataDirValue = AddReadOnlySettingsRow(envLayout, "Data dir", envRow++);
        advancedVersionValue = AddReadOnlySettingsRow(envLayout, "Manager version", envRow++);
        advancedConfigPathValue = AddReadOnlySettingsRow(envLayout, "Config path", envRow++);

        advancedTextBox = new TextBox
        {
            Anchor = AnchorStyles.Left | AnchorStyles.Right,
            ReadOnly = true,
            Font = Theme.MonoFont,
            BackColor = Theme.FieldBackground,
            ForeColor = Theme.Text
        };
        StyleSettingsField(advancedTextBox);

        var copyButton = CreateSecondaryButton("Copy");
        copyButton.Click += (_, _) =>
        {
            try
            {
                var commandLine = advancedTextBox.Text;
                if (!string.IsNullOrEmpty(commandLine))
                {
                    Clipboard.SetText(commandLine);
                }
            }
            catch
            {
                // Best effort; clipboard access can fail in restricted environments.
            }
        };

        AddSettingsSectionHeader(envLayout, "Command", envRow++);
        AddSettingsRowWithButton(envLayout, "Effective command line", advancedTextBox, copyButton, envRow++);

        envCard.Controls.Add(envLayout);

        var maintenanceCard = CreateSettingsCard(450);
        var maintenanceLayout = CreateSettingsTable();

        var repairButton = CreateSecondaryButton("Repair Package");
        var validateButton = CreateSecondaryButton("Validate Environment");
        var exportButton = CreateSecondaryButton("Export Diagnostics");

        repairButton.Click += async (_, _) => await RunAdvancedActionAsync("Repair", () => controller.Repair());
        validateButton.Click += async (_, _) => await RunAdvancedActionAsync("Validate Environment", () => controller.ValidateEnvironment());
        exportButton.Click += async (_, _) => await ExportDiagnosticsAsync();

        var maintenanceRow = 0;
        AddSettingsCardHeader(maintenanceLayout, "Maintenance", "Repair the installation, validate the environment, or export a diagnostics bundle.", maintenanceRow++);
        AddSettingsSectionHeader(maintenanceLayout, "Package tools", maintenanceRow++);
        AddActionRow(maintenanceLayout, "Repair package", "Verify installed files and restore any that are missing or corrupt.", new[] { repairButton }, maintenanceRow++);
        AddActionRow(maintenanceLayout, "Validate environment", "Check that ovms.exe and required files are present and runnable.", new[] { validateButton }, maintenanceRow++);
        AddActionRow(maintenanceLayout, "Diagnostics", "Save a zip with settings, logs, versions, and runtime state.", new[] { exportButton }, maintenanceRow++);

        advancedStatusLabel = new Label
        {
            Text = "",
            ForeColor = Theme.Muted,
            Font = new Font(Theme.SemiboldFont.FontFamily, 10.5f, FontStyle.Bold)
        };
        AddStatusRow(maintenanceLayout, advancedStatusLabel, maintenanceRow++);

        maintenanceCard.Controls.Add(maintenanceLayout);

        // Dock-Top stacking: add bottom-most visual element first (see
        // BuildShell comment for why this ordering yields correct results).
        page.Controls.Add(maintenanceCard);
        page.Controls.Add(envCard);
        return page;
    }

    private void RefreshAdvancedInfo()
    {
        var settings = TryLoadSettings();
        if (settings is null)
        {
            advancedInstallDirValue.Text = "-";
            advancedDataDirValue.Text = "-";
            advancedVersionValue.Text = typeof(MainForm).Assembly.GetName().Version?.ToString() ?? "-";
            advancedConfigPathValue.Text = "-";
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

        advancedInstallDirValue.Text = settings.InstallDir;
        advancedDataDirValue.Text = controller.DataDir;
        advancedVersionValue.Text = typeof(MainForm).Assembly.GetName().Version?.ToString() ?? "-";
        advancedConfigPathValue.Text = settings.ConfigPath;
        advancedTextBox.Text = commandLine;
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

    // ---------------------------------------------------------------
    // Updates
    // ---------------------------------------------------------------

    private Panel BuildUpdatesPage()
    {
        var page = new Panel { BackColor = Theme.WindowBackground, AutoScroll = true, Padding = new Padding(0, 0, 10, 0) };

        var actionCard = CreateSettingsCard(320);
        var actionLayout = CreateSettingsTable();

        updatesCheckButton = CreateSecondaryButton("Check for Updates");
        updatesCheckButton.Click += async (_, _) => await CheckForUpdatesAsync();

        updatesModelServerReleaseButton = CreateSecondaryButton("Model Server Notes");
        updatesModelServerReleaseButton.Enabled = false;
        updatesModelServerReleaseButton.Click += (_, _) => OpenReleaseNotes(updatesModelServerReleaseUrl, "Model Server Release Notes");

        updatesGenAiReleaseButton = CreateSecondaryButton("GenAI Notes");
        updatesGenAiReleaseButton.Enabled = false;
        updatesGenAiReleaseButton.Click += (_, _) => OpenReleaseNotes(updatesGenAiReleaseUrl, "GenAI Release Notes");

        var actionRow = 0;
        AddSettingsCardHeader(actionLayout, "Update check", "Check for a newer compatible release. This will not download or install anything.", actionRow++);
        AddSettingsSectionHeader(actionLayout, "Manual check", actionRow++);
        AddActionRow(actionLayout, "Available updates", "Compare the package, Model Server, and GenAI versions with release metadata.", new[] { updatesCheckButton, updatesModelServerReleaseButton, updatesGenAiReleaseButton }, actionRow++);
        updatesStatusLabel = new Label
        {
            Text = "Ready to check for updates.",
            ForeColor = Theme.Muted,
            Font = new Font(Theme.SemiboldFont.FontFamily, 10.5f, FontStyle.Bold)
        };
        AddStatusRow(actionLayout, updatesStatusLabel, actionRow++);

        actionCard.Controls.Add(actionLayout);

        updatesBaseCard = CreateSettingsCard(360);
        var baseLayout = CreateSettingsTable();
        var baseRow = 0;
        AddSettingsCardHeader(baseLayout, "Base Package", "Installed package metadata and matching package release.", baseRow++);
        AddSettingsSectionHeader(baseLayout, "Version details", baseRow++);
        updatesBaseInstalledVersionValue = AddReadOnlySettingsRow(baseLayout, "Installed version", baseRow++);
        updatesBaseLatestVersionValue = AddReadOnlySettingsRow(baseLayout, "Latest version", baseRow++);
        updatesVariantValue = AddReadOnlySettingsRow(baseLayout, "Package variant", baseRow++);
        updatesManagerVersionValue = AddReadOnlySettingsRow(baseLayout, "Manager version", baseRow++);
        updatesBaseInstallButton = CreateSecondaryButton("Install Package Update");
        updatesBaseInstallButton.Click += async (_, _) => await InstallPackageUpdateAsync("Base Package");
        updatesBaseInstallRow = AddActionRow(baseLayout, "Install update", "Download, verify, stage, and replace the installed package.", new[] { updatesBaseInstallButton }, baseRow++);
        updatesBaseCard.Controls.Add(baseLayout);

        updatesGenAiCard = CreateSettingsCard(260);
        var genAiLayout = CreateSettingsTable();
        var genAiRow = 0;
        AddSettingsCardHeader(genAiLayout, "GenAI", "OpenVINO GenAI backend bundled with this package.", genAiRow++);
        AddSettingsSectionHeader(genAiLayout, "Version details", genAiRow++);
        updatesGenAiInstalledVersionValue = AddReadOnlySettingsRow(genAiLayout, "Installed version", genAiRow++);
        updatesGenAiLatestVersionValue = AddReadOnlySettingsRow(genAiLayout, "Latest version", genAiRow++);
        updatesGenAiInstallButton = CreateSecondaryButton("Install Package Update");
        updatesGenAiInstallButton.Click += async (_, _) => await InstallPackageUpdateAsync("GenAI");
        updatesGenAiInstallRow = AddActionRow(genAiLayout, "Install update", "Install the matching package update so GenAI stays version-aligned.", new[] { updatesGenAiInstallButton }, genAiRow++);
        updatesGenAiCard.Controls.Add(genAiLayout);

        updatesModelServerCard = CreateSettingsCard(260);
        var modelServerLayout = CreateSettingsTable();
        var modelServerRow = 0;
        AddSettingsCardHeader(modelServerLayout, "Model Server", "OpenVINO Model Server runtime bundled with this package.", modelServerRow++);
        AddSettingsSectionHeader(modelServerLayout, "Version details", modelServerRow++);
        updatesModelServerInstalledVersionValue = AddReadOnlySettingsRow(modelServerLayout, "Installed version", modelServerRow++);
        updatesModelServerLatestVersionValue = AddReadOnlySettingsRow(modelServerLayout, "Latest version", modelServerRow++);
        updatesModelServerInstallButton = CreateSecondaryButton("Install Package Update");
        updatesModelServerInstallButton.Click += async (_, _) => await InstallPackageUpdateAsync("Model Server");
        updatesModelServerInstallRow = AddActionRow(modelServerLayout, "Install update", "Install the matching package update so Model Server and dependencies stay aligned.", new[] { updatesModelServerInstallButton }, modelServerRow++);
        updatesModelServerCard.Controls.Add(modelServerLayout);

        SetInstallRowsVisible(basePackage: false, modelServer: false, genAi: false);

        // Dock-Top stacking: add bottom-most visual element first.
        page.Controls.Add(updatesModelServerCard);
        page.Controls.Add(CreateCardSpacer(20));
        page.Controls.Add(updatesGenAiCard);
        page.Controls.Add(CreateCardSpacer(20));
        page.Controls.Add(updatesBaseCard);
        page.Controls.Add(CreateCardSpacer(20));
        page.Controls.Add(actionCard);
        return page;
    }

    private void RefreshUpdatesInfo()
    {
        try
        {
            var info = controller.GetPackageVersionInfo();
            updatesBaseInstalledVersionValue.Text = info.BasePackageVersion;
            updatesBaseLatestVersionValue.Text = "-";
            updatesVariantValue.Text = info.PackageVariant;
            updatesManagerVersionValue.Text = info.ManagerVersion;
            updatesModelServerInstalledVersionValue.Text = info.ModelServerVersion;
            updatesModelServerLatestVersionValue.Text = "-";
            updatesGenAiInstalledVersionValue.Text = info.GenAiVersion;
            updatesGenAiLatestVersionValue.Text = "-";
            updatesStatusLabel.ForeColor = Theme.Muted;
            updatesStatusLabel.Text = "Ready to check for updates.";
            updatesModelServerReleaseUrl = "";
            updatesGenAiReleaseUrl = "";
            updatesModelServerReleaseButton.Enabled = false;
            updatesGenAiReleaseButton.Enabled = false;
            SetInstallRowsVisible(basePackage: false, modelServer: false, genAi: false);
        }
        catch (Exception ex)
        {
            updatesBaseInstalledVersionValue.Text = "-";
            updatesBaseLatestVersionValue.Text = "-";
            updatesVariantValue.Text = "-";
            updatesManagerVersionValue.Text = typeof(MainForm).Assembly.GetName().Version?.ToString() ?? "-";
            updatesModelServerInstalledVersionValue.Text = "-";
            updatesModelServerLatestVersionValue.Text = "-";
            updatesGenAiInstalledVersionValue.Text = "-";
            updatesGenAiLatestVersionValue.Text = "-";
            updatesStatusLabel.ForeColor = Theme.Danger;
            updatesStatusLabel.Text = $"Cannot read package information: {ex.Message}";
            updatesModelServerReleaseUrl = "";
            updatesGenAiReleaseUrl = "";
            updatesModelServerReleaseButton.Enabled = false;
            updatesGenAiReleaseButton.Enabled = false;
            SetInstallRowsVisible(basePackage: false, modelServer: false, genAi: false);
        }
    }

    private async Task CheckForUpdatesAsync()
    {
        updatesCheckButton.Enabled = false;
        updatesStatusLabel.ForeColor = Theme.Muted;
        updatesStatusLabel.Text = "Checking for updates...";

        try
        {
            var result = await controller.CheckForUpdatesAsync();
            updatesBaseInstalledVersionValue.Text = result.BasePackageVersion;
            updatesBaseLatestVersionValue.Text = result.LatestBasePackageVersion;
            updatesModelServerInstalledVersionValue.Text = result.ModelServerVersion;
            updatesModelServerLatestVersionValue.Text = result.LatestModelServerVersion;
            updatesGenAiInstalledVersionValue.Text = result.GenAiVersion;
            updatesGenAiLatestVersionValue.Text = result.LatestGenAiVersion;
            updatesVariantValue.Text = result.PackageVariant;
            updatesModelServerReleaseUrl = result.ModelServerReleaseUrl;
            updatesGenAiReleaseUrl = result.GenAiReleaseUrl;
            updatesModelServerReleaseButton.Enabled = !string.IsNullOrWhiteSpace(updatesModelServerReleaseUrl);
            updatesGenAiReleaseButton.Enabled = !string.IsNullOrWhiteSpace(updatesGenAiReleaseUrl);
            SetInstallRowsVisible(result.BasePackageUpdateAvailable, result.ModelServerUpdateAvailable, result.GenAiUpdateAvailable);
            updatesStatusLabel.ForeColor = result.UpdateAvailable
                ? Theme.Warning
                : result.LatestIsOlderThanInstalled ? Theme.Muted : Theme.Success;
            updatesStatusLabel.Text = result.Message;
        }
        catch (Exception ex)
        {
            updatesStatusLabel.ForeColor = Theme.Danger;
            updatesStatusLabel.Text = $"Cannot check for updates: {ex.Message}";
            updatesModelServerReleaseUrl = "";
            updatesGenAiReleaseUrl = "";
            updatesModelServerReleaseButton.Enabled = false;
            updatesGenAiReleaseButton.Enabled = false;
            SetInstallRowsVisible(basePackage: false, modelServer: false, genAi: false);
        }
        finally
        {
            updatesCheckButton.Enabled = true;
        }
    }

    private async Task InstallPackageUpdateAsync(string sectionName)
    {
        var result = MessageBox.Show(this,
            $"{sectionName} updates are installed by replacing the matched OVMS Windows package as a unit. This keeps Base Package, Model Server, and GenAI versions aligned.{Environment.NewLine}{Environment.NewLine}Download and install the latest package now?",
            "Install Package Update",
            MessageBoxButtons.YesNo,
            MessageBoxIcon.Question);
        if (result != DialogResult.Yes)
        {
            return;
        }

        SetUpdateInstallButtonsEnabled(false);
        updatesStatusLabel.ForeColor = Theme.Muted;
        updatesStatusLabel.Text = "Installing package update...";

        try
        {
            var output = await Task.Run(controller.InstallPackageUpdate);
            updatesStatusLabel.ForeColor = Theme.Success;
            updatesStatusLabel.Text = "Package update installed.";
            MessageBox.Show(this, string.IsNullOrWhiteSpace(output) ? "Package update installed." : output, "Install Package Update", MessageBoxButtons.OK, MessageBoxIcon.Information);
            RefreshUpdatesInfo();
        }
        catch (Exception ex)
        {
            updatesStatusLabel.ForeColor = Theme.Danger;
            updatesStatusLabel.Text = "Package update failed.";
            MessageBox.Show(this, ex.Message, "Install Package Update", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            SetUpdateInstallButtonsEnabled(true);
        }
    }

    private void SetUpdateInstallButtonsEnabled(bool enabled)
    {
        updatesCheckButton.Enabled = enabled;
        updatesBaseInstallButton.Enabled = enabled;
        updatesModelServerInstallButton.Enabled = enabled;
        updatesGenAiInstallButton.Enabled = enabled;
    }

    private void SetInstallRowsVisible(bool basePackage, bool modelServer, bool genAi)
    {
        updatesBaseInstallRow.SetVisible(basePackage);
        updatesModelServerInstallRow.SetVisible(modelServer);
        updatesGenAiInstallRow.SetVisible(genAi);

        updatesBaseCard.Height = basePackage ? 444 : 360;
        updatesModelServerCard.Height = modelServer ? 344 : 260;
        updatesGenAiCard.Height = genAi ? 344 : 260;
    }

    private void OpenReleaseNotes(string releaseUrl, string title)
    {
        if (string.IsNullOrWhiteSpace(releaseUrl))
        {
            return;
        }

        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = releaseUrl,
                UseShellExecute = true
            });
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, title, MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}

internal readonly struct CollapsibleActionRow
{
    private readonly RowStyle rowStyle;
    private readonly Control label;
    private readonly Control content;
    private readonly float expandedHeight;

    public CollapsibleActionRow(RowStyle rowStyle, Control label, Control content, float expandedHeight)
    {
        this.rowStyle = rowStyle;
        this.label = label;
        this.content = content;
        this.expandedHeight = expandedHeight;
    }

    public void SetVisible(bool visible)
    {
        rowStyle.Height = visible ? expandedHeight : 0;
        label.Visible = visible;
        content.Visible = visible;
    }
}

internal sealed class RuntimeIconButton : Button
{
    private bool hovered;
    private bool pressed;

    public Color ActiveColor { get; set; }
    public Color DisabledColor { get; set; } = Theme.Muted;

    public RuntimeIconButton(string glyph, Color activeColor)
    {
        Text = glyph;
        ActiveColor = activeColor;
        Tag = activeColor;
        Font = new Font("Segoe MDL2 Assets", 13.5f, FontStyle.Regular, GraphicsUnit.Point);
        FlatStyle = FlatStyle.Flat;
        FlatAppearance.BorderSize = 0;
        UseVisualStyleBackColor = false;
        TextAlign = ContentAlignment.MiddleCenter;
        Padding = new Padding(0);
        TabStop = false;
        Cursor = Cursors.Hand;
        SetStyle(
            ControlStyles.AllPaintingInWmPaint |
            ControlStyles.UserPaint |
            ControlStyles.OptimizedDoubleBuffer |
            ControlStyles.ResizeRedraw,
            true);
    }

    protected override void OnMouseEnter(EventArgs e)
    {
        hovered = true;
        Invalidate();
        base.OnMouseEnter(e);
    }

    protected override void OnMouseLeave(EventArgs e)
    {
        hovered = false;
        pressed = false;
        Invalidate();
        base.OnMouseLeave(e);
    }

    protected override void OnMouseDown(MouseEventArgs mevent)
    {
        if (Enabled && mevent.Button == MouseButtons.Left)
        {
            pressed = true;
            Invalidate();
        }
        base.OnMouseDown(mevent);
    }

    protected override void OnMouseUp(MouseEventArgs mevent)
    {
        pressed = false;
        Invalidate();
        base.OnMouseUp(mevent);
    }

    protected override void OnEnabledChanged(EventArgs e)
    {
        hovered = false;
        pressed = false;
        Invalidate();
        base.OnEnabledChanged(e);
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        e.Graphics.TextRenderingHint = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;
        e.Graphics.Clear(ResolveCanvasColor(this));

        var buttonRect = new Rectangle(7, 6, Width - 14, Height - 13);
        if (pressed)
        {
            buttonRect.Offset(0, 1);
        }

        if (Enabled && hovered)
        {
            using var ambientPath = RoundedRect(new Rectangle(buttonRect.X - 2, buttonRect.Y, buttonRect.Width + 4, buttonRect.Height + 4), 12);
            using var dropPath = RoundedRect(new Rectangle(buttonRect.X - 1, buttonRect.Y + 5, buttonRect.Width + 2, buttonRect.Height), 12);
            using var ambientBrush = new SolidBrush(Color.FromArgb(18, 13, 24, 38));
            using var dropBrush = new SolidBrush(Color.FromArgb(48, 13, 24, 38));
            e.Graphics.FillPath(ambientBrush, ambientPath);
            e.Graphics.FillPath(dropBrush, dropPath);
        }

        var borderColor = Enabled
            ? (hovered ? Color.FromArgb(86, ActiveColor) : Color.FromArgb(34, 13, 24, 38))
            : Color.FromArgb(32, Theme.Muted);
        var fillColor = Enabled
            ? (hovered ? Color.FromArgb(254, 255, 255) : Color.FromArgb(248, 249, 251))
            : Color.FromArgb(246, 247, 249);

        using (var buttonPath = RoundedRect(buttonRect, 11))
        using (var fillBrush = new SolidBrush(fillColor))
        using (var borderPen = new Pen(borderColor))
        {
            e.Graphics.FillPath(fillBrush, buttonPath);
            e.Graphics.DrawPath(borderPen, buttonPath);
        }

        var color = Enabled ? ActiveColor : DisabledColor;
        using var textBrush = new SolidBrush(color);
        using var iconPath = new GraphicsPath();
        using var format = StringFormat.GenericTypographic;
        iconPath.AddString(Text, Font.FontFamily, (int)Font.Style, e.Graphics.DpiY * Font.Size / 72f, Point.Empty, format);
        var bounds = iconPath.GetBounds();
        var targetCenter = new PointF(buttonRect.Left + buttonRect.Width / 2f, buttonRect.Top + buttonRect.Height / 2f + (pressed ? 1f : 0f));
        using var transform = new Matrix();
        transform.Translate(targetCenter.X - (bounds.Left + bounds.Width / 2f), targetCenter.Y - (bounds.Top + bounds.Height / 2f));
        iconPath.Transform(transform);
        e.Graphics.FillPath(textBrush, iconPath);
    }

    private static GraphicsPath RoundedRect(Rectangle bounds, int radius)
    {
        var diameter = radius * 2;
        var path = new GraphicsPath();
        path.AddArc(bounds.Left, bounds.Top, diameter, diameter, 180, 90);
        path.AddArc(bounds.Right - diameter, bounds.Top, diameter, diameter, 270, 90);
        path.AddArc(bounds.Right - diameter, bounds.Bottom - diameter, diameter, diameter, 0, 90);
        path.AddArc(bounds.Left, bounds.Bottom - diameter, diameter, diameter, 90, 90);
        path.CloseFigure();
        return path;
    }

    private static Color ResolveCanvasColor(Control control)
    {
        for (var current = control; current is not null; current = current.Parent)
        {
            if (current.BackColor != Color.Transparent && current.BackColor.A > 0)
            {
                return current.BackColor;
            }
        }

        return Theme.Surface;
    }
}

internal sealed class SettingsFieldHost : Panel
{
    public SettingsFieldHost()
    {
        BackColor = Theme.Surface;
        Padding = new Padding(1);
        SetStyle(
            ControlStyles.AllPaintingInWmPaint |
            ControlStyles.UserPaint |
            ControlStyles.OptimizedDoubleBuffer |
            ControlStyles.ResizeRedraw,
            true);
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        e.Graphics.Clear(Parent?.BackColor ?? Theme.Surface);

        var rect = new Rectangle(0, 0, Width - 1, Height - 1);
        using var path = RoundedRect(rect, 7);
        using var fillBrush = new SolidBrush(Theme.FieldBackground);
        using var borderPen = new Pen(Theme.FieldBorder);
        e.Graphics.FillPath(fillBrush, path);
        e.Graphics.DrawPath(borderPen, path);

        base.OnPaint(e);
    }

    private static GraphicsPath RoundedRect(Rectangle bounds, int radius)
    {
        var diameter = radius * 2;
        var path = new GraphicsPath();
        path.AddArc(bounds.Left, bounds.Top, diameter, diameter, 180, 90);
        path.AddArc(bounds.Right - diameter, bounds.Top, diameter, diameter, 270, 90);
        path.AddArc(bounds.Right - diameter, bounds.Bottom - diameter, diameter, diameter, 0, 90);
        path.AddArc(bounds.Left, bounds.Bottom - diameter, diameter, diameter, 90, 90);
        path.CloseFigure();
        return path;
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

        Width = 210;
        Height = 44;
        Margin = new Padding(0);
        Cursor = Cursors.Hand;
        SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.OptimizedDoubleBuffer, true);

        glyphLabel = new Label
        {
            Text = glyph,
            Font = Theme.IconFont(14f),
            AutoSize = false,
            Size = new Size(36, 44),
            TextAlign = ContentAlignment.MiddleCenter,
            Location = new Point(10, 0),
            BackColor = Color.Transparent
        };
        textLabel = new Label
        {
            Text = pageName,
            Font = Theme.NavFont,
            AutoSize = false,
            Size = new Size(158, 44),
            TextAlign = ContentAlignment.MiddleLeft,
            Location = new Point(44, 0),
            BackColor = Color.Transparent
        };

        Controls.Add(glyphLabel);
        Controls.Add(textLabel);

        // Only forward the child labels' clicks to the panel's OnClick. Do NOT
        // subscribe the panel's own Click to OnClick -- OnClick raises Click,
        // which would re-enter the handler and recurse until a stack overflow.
        glyphLabel.Click += (_, _) => OnClick(EventArgs.Empty);
        textLabel.Click += (_, _) => OnClick(EventArgs.Empty);

        foreach (var c in new Control[] { this, glyphLabel, textLabel })
        {
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
