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
        var button = new Button
        {
            Text = glyph,
            Font = new Font("Segoe UI Symbol", 13.5f, FontStyle.Regular, GraphicsUnit.Point),
            FlatStyle = FlatStyle.Flat,
            ForeColor = Color.White,
            BackColor = fillColor,
            Size = new Size(44, 42),
            MinimumSize = new Size(44, 42),
            MaximumSize = new Size(44, 42),
            Cursor = Cursors.Hand,
            UseVisualStyleBackColor = false,
            TextAlign = ContentAlignment.MiddleCenter,
            Padding = new Padding(0),
            Margin = new Padding(8, 0, 0, 0)
        };
        button.FlatAppearance.BorderSize = 0;
        button.FlatAppearance.MouseOverBackColor = ControlPaint.Dark(fillColor, 0.08f);
        var tip = new ToolTip();
        tip.SetToolTip(button, tooltip);
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

    private Panel BuildResponsiveDashboardPage()
    {
        var page = new Panel { BackColor = Theme.WindowBackground, AutoScroll = true };

        var dashboardStack = new TableLayoutPanel
        {
            Location = new Point(0, 0),
            AutoSize = false,
            ColumnCount = 1,
            RowCount = 2,
            Height = 656,
            Padding = new Padding(0),
            Margin = new Padding(0)
        };
        dashboardStack.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        dashboardStack.RowStyles.Add(new RowStyle(SizeType.Absolute, 456));
        dashboardStack.RowStyles.Add(new RowStyle(SizeType.Absolute, 200));

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

        startButton = CreateRuntimeIconButton("\u25B6", "Start server", Theme.Success);
        stopButton = CreateRuntimeIconButton("\u25A0", "Stop server", Theme.Danger);
        restartButton = CreateRuntimeIconButton("\u21BB", "Restart server", Theme.Accent);

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
            cardLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 248));
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
                ColumnCount = 3,
                RowCount = 3,
                BackColor = Color.Transparent,
                Margin = new Padding(0),
                Padding = new Padding(0)
            };
            actionLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.333f));
            actionLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.333f));
            actionLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.333f));
            actionLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
            actionLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            actionLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
            actionLayout.Controls.Add(startButton, 0, 1);
            actionLayout.Controls.Add(stopButton, 1, 1);
            actionLayout.Controls.Add(restartButton, 2, 1);

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

        var actionsCard = new CardPanel { Dock = DockStyle.Fill, Margin = new Padding(0, 10, 0, 0), Padding = new Padding(14) };
        var actionsLayout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 3,
            RowCount = 2,
            Margin = new Padding(0),
            Padding = new Padding(0)
        };
        for (var i = 0; i < 3; i++)
        {
            actionsLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.333f));
        }
        actionsLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
        actionsLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 50));

        var openLogsButton = CreateSecondaryButton("Open Logs");
        var openModelFolderButton = CreateSecondaryButton("Open Model Folder");

        foreach (var b in new[] { openLogsButton, openModelFolderButton })
        {
            b.AutoSize = false;
            b.Anchor = AnchorStyles.None;
            b.Margin = new Padding(0, 0, 10, 10);
            b.MinimumSize = new Size(120, 48);
        }

        openLogsButton.Click += (_, _) => OpenFolderFor(TryLoadSettings()?.LogPath);
        openModelFolderButton.Click += (_, _) => OpenFolder(TryLoadSettings()?.ModelRepositoryPath);

        actionsLayout.Controls.Add(openLogsButton, 0, 0);
        actionsLayout.Controls.Add(openModelFolderButton, 1, 0);
        actionsLayout.SetColumnSpan(openModelFolderButton, 2);
        actionsCard.Controls.Add(actionsLayout);

        dashboardStack.Controls.Add(metricGrid, 0, 0);
        dashboardStack.Controls.Add(actionsCard, 0, 1);
        page.Controls.Add(dashboardStack);

        void SizeActionButtons()
        {
            foreach (var button in new[] { openLogsButton, openModelFolderButton })
            {
                var preferred = TextRenderer.MeasureText(button.Text, button.Font).Width + 72;
                button.Size = new Size(Math.Clamp(preferred, 120, 280), 52);
            }
        }

        void ResizeDashboard()
        {
            var availableWidth = ClientSize.Width - navRail.Width - contentHost.Padding.Horizontal - 28;
            dashboardStack.Width = Math.Max(560, availableWidth);
            SizeActionButtons();
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
            startButton.Visible = false;
            stopButton.Visible = false;
            restartButton.Visible = false;
            return;
        }

        var runtimeStatus = await Task.Run(controller.GetRuntimeStatus);
        statusValueLabel.Text = runtimeStatus.Running ? $"Running (pid {runtimeStatus.Pid})" : "Stopped";
        statusDotLabel.ForeColor = runtimeStatus.Running ? Theme.Success : Theme.Muted;
        railStatusDot.ForeColor = runtimeStatus.Running ? Theme.Success : Theme.Muted;
        railStatusText.Text = runtimeStatus.Running ? "Running" : "Stopped";
        railStatusText.ForeColor = runtimeStatus.Running ? Theme.Text : Theme.Muted;
        startButton.Visible = !runtimeStatus.Running;
        stopButton.Visible = runtimeStatus.Running;
        restartButton.Visible = runtimeStatus.Running;

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

        var serverCard = CreateTitledCard("Server", 320);
        var serverLayout = CreateSettingsTable();

        var row = 0;
        AddSettingsRow(serverLayout, "REST port:", restPortInput = new NumericUpDown { Minimum = 1, Maximum = 65535, Width = 120, Margin = new Padding(3, 6, 3, 3) }, row++);
        AddSettingsRow(serverLayout, "gRPC port (0 = disabled):", grpcPortInput = new NumericUpDown { Minimum = 0, Maximum = 65535, Width = 120, Margin = new Padding(3, 6, 3, 3) }, row++);
        AddSettingsRow(serverLayout, "Bind address:", bindAddressInput = new TextBox { Width = 240, Margin = new Padding(3, 6, 3, 3) }, row++);

        logLevelInput = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 140, Margin = new Padding(3, 6, 3, 3) };
        logLevelInput.Items.AddRange(new object[] { "ERROR", "WARNING", "INFO", "DEBUG", "TRACE" });
        AddSettingsRow(serverLayout, "Log level:", logLevelInput, row++);

        logPathInput = new TextBox { Width = 360, Margin = new Padding(3, 6, 3, 3) };
        var browseLogButton = CreateSecondaryButton("Browse...");
        browseLogButton.Margin = new Padding(6, 3, 3, 3);
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
        AddSettingsRowWithButton(serverLayout, "Log path:", logPathInput, browseLogButton, row++);

        modelRepoInput = new TextBox { Width = 360, Margin = new Padding(3, 6, 3, 3) };
        var browseModelRepoButton = CreateSecondaryButton("Browse...");
        browseModelRepoButton.Margin = new Padding(6, 3, 3, 3);
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
        AddSettingsRowWithButton(serverLayout, "Model repository path:", modelRepoInput, browseModelRepoButton, row++);

        serverCard.Controls.Add(serverLayout);

        var startupCard = CreateTitledCard("Startup & tray", 200);
        var startupLayout = CreateSettingsTable();

        row = 0;
        runModeInput = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 180, Margin = new Padding(3, 6, 3, 3) };
        runModeInput.Items.AddRange(new object[] { "user-login", "service", "manual" });
        AddSettingsRow(startupLayout, "Startup mode:", runModeInput, row++);

        showTrayCheckBox = new CheckBox { Margin = new Padding(3, 10, 3, 3) };
        AddSettingsRow(startupLayout, "Show tray icon:", showTrayCheckBox, row++);

        startAtLoginCheckBox = new CheckBox { Margin = new Padding(3, 10, 3, 3) };
        AddSettingsRow(startupLayout, "Start at login:", startAtLoginCheckBox, row++);

        startupCard.Controls.Add(startupLayout);

        var footer = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            Height = 56,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false,
            Padding = new Padding(0, 14, 0, 0),
            Margin = new Padding(0)
        };
        var saveButton = CreatePrimaryButton("", "Save", Theme.Accent);
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
        page.Controls.Add(startupCard);
        page.Controls.Add(serverCard);
        return page;
    }

    private static CardPanel CreateTitledCard(string title, int height)
    {
        var card = new CardPanel { Dock = DockStyle.Top, Height = height, Margin = new Padding(0, 0, 0, 16) };
        var titleLabel = new Label
        {
            Text = title,
            Font = Theme.CardTitleFont,
            ForeColor = Theme.Text,
            AutoSize = true,
            Dock = DockStyle.Top,
            Margin = new Padding(0, 0, 0, 12)
        };
        card.Controls.Add(titleLabel);
        return card;
    }

    private static TableLayoutPanel CreateSettingsTable()
    {
        var table = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            AutoSize = false,
            Padding = new Padding(0, 32, 0, 0)
        };
        table.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        return table;
    }

    private static void AddSettingsRow(TableLayoutPanel table, string label, Control field, int row)
    {
        table.Controls.Add(new Label { Text = label, AutoSize = true, ForeColor = Theme.Text, Margin = new Padding(3, 10, 16, 3) }, 0, row);
        table.Controls.Add(field, 1, row);
    }

    private static void AddSettingsRowWithButton(TableLayoutPanel table, string label, Control field, Control button, int row)
    {
        var inline = new FlowLayoutPanel
        {
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false,
            AutoSize = true,
            Margin = new Padding(0)
        };
        inline.Controls.Add(field);
        inline.Controls.Add(button);
        table.Controls.Add(new Label { Text = label, AutoSize = true, ForeColor = Theme.Text, Margin = new Padding(3, 10, 16, 3) }, 0, row);
        table.Controls.Add(inline, 1, row);
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

        var toolbar = new FlowLayoutPanel { Dock = DockStyle.Top, Height = 40, FlowDirection = FlowDirection.LeftToRight, Padding = new Padding(0, 0, 0, 8) };
        var refreshLogsButton = CreateIconButton(Glyphs.Refresh, "Refresh");
        var openFolderButton = CreateIconButton(Glyphs.OpenFolder, "Open Folder");
        var clearLogsButton = CreateIconButton("✕", "Clear (display only)");
        refreshLogsButton.Margin = new Padding(0, 0, 8, 0);
        openFolderButton.Margin = new Padding(0, 0, 8, 0);
        clearLogsButton.Margin = new Padding(0, 0, 16, 0);

        autoScrollCheckBox = new CheckBox { Text = "Auto-scroll", Checked = true, AutoSize = true, Margin = new Padding(0, 8, 16, 0) };

        refreshLogsButton.Click += async (_, _) => await RefreshLogsAsync();
        openFolderButton.Click += (_, _) => OpenFolderFor(TryLoadSettings()?.LogPath);
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
            TextAlign = ContentAlignment.MiddleRight,
            Anchor = AnchorStyles.Right,
            Margin = new Padding(8, 8, 0, 0)
        };

        toolbar.Controls.Add(refreshLogsButton);
        toolbar.Controls.Add(openFolderButton);
        toolbar.Controls.Add(clearLogsButton);
        toolbar.Controls.Add(autoScrollCheckBox);
        toolbar.Controls.Add(logPathLabel);

        var statusBar = new Panel { Dock = DockStyle.Bottom, Height = 24, Padding = new Padding(0, 4, 0, 0) };
        logStatusLabel = new Label { Text = "", AutoSize = true, ForeColor = Theme.Muted, Dock = DockStyle.Left };
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

    // ---------------------------------------------------------------
    // Advanced
    // ---------------------------------------------------------------

    private Panel BuildAdvancedPage()
    {
        var page = new Panel { BackColor = Theme.WindowBackground, AutoScroll = true };

        var envCard = CreateTitledCard("Environment", 260);
        var envLayout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2, Padding = new Padding(0, 32, 0, 0) };
        envLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        envLayout.RowStyles.Add(new RowStyle(SizeType.AutoSize));

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
        envLayout.Controls.Add(advancedTextBox, 0, 0);

        var copyButton = CreateSecondaryButton("Copy");
        copyButton.Margin = new Padding(0, 8, 0, 0);
        copyButton.Click += (_, _) =>
        {
            try
            {
                var settings = TryLoadSettings();
                var commandLine = settings is null ? "" : controller.EffectiveCommandLine();
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
        envLayout.Controls.Add(copyButton, 0, 1);

        envCard.Controls.Add(envLayout);

        var maintenanceCard = CreateTitledCard("Maintenance", 170);
        var buttonPanel = new FlowLayoutPanel { Dock = DockStyle.Top, AutoSize = true, Padding = new Padding(0, 32, 0, 0) };
        var repairButton = CreatePrimaryButton("", "Repair Package", Theme.Accent);
        var validateButton = CreateSecondaryButton("Validate Environment");
        var exportButton = CreateSecondaryButton("Export Diagnostics");
        repairButton.Margin = new Padding(0, 0, 10, 0);
        validateButton.Margin = new Padding(0, 0, 10, 0);

        repairButton.Click += async (_, _) => await RunAdvancedActionAsync("Repair", () => controller.Repair());
        validateButton.Click += async (_, _) => await RunAdvancedActionAsync("Validate Environment", () => controller.ValidateEnvironment());
        exportButton.Click += async (_, _) => await ExportDiagnosticsAsync();

        buttonPanel.Controls.Add(repairButton);
        buttonPanel.Controls.Add(validateButton);
        buttonPanel.Controls.Add(exportButton);

        advancedStatusLabel = new Label
        {
            Text = "",
            ForeColor = Theme.Muted,
            Font = new Font(Theme.SemiboldFont.FontFamily, 11f, FontStyle.Bold),
            AutoSize = true,
            Dock = DockStyle.Top,
            Margin = new Padding(0, 14, 0, 0)
        };

        maintenanceCard.Controls.Add(advancedStatusLabel);
        maintenanceCard.Controls.Add(buttonPanel);

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

        var managerVersion = typeof(MainForm).Assembly.GetName().Version;

        advancedTextBox.Text = string.Join(Environment.NewLine, new[]
        {
            $"Install dir: {settings.InstallDir}",
            $"Data dir: {controller.DataDir}",
            $"Manager version: {managerVersion}",
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
