using System.Drawing;

namespace OVMS.Manager;

/// <summary>
/// Owns the NotifyIcon and a single lazily-created MainForm instance.
/// </summary>
internal sealed class TrayApplicationContext : ApplicationContext
{
    private readonly NotifyIcon notifyIcon;
    private readonly ToolStripMenuItem startItem;
    private readonly ToolStripMenuItem stopItem;
    private readonly OvmsController controller;
    private MainForm? mainForm;
    private readonly bool startHidden;

    public TrayApplicationContext(bool startHidden = false)
    {
        this.startHidden = startHidden;
        controller = new OvmsController();

        startItem = new ToolStripMenuItem("Start Server", null, async (_, _) => await RunTrayActionAsync("Start Server", controller.Start));
        stopItem = new ToolStripMenuItem("Stop Server", null, async (_, _) => await RunTrayActionAsync("Stop Server", controller.Stop));

        var openItem = new ToolStripMenuItem("Open OVMS Manager", null, (_, _) => OpenManager());
        var exitItem = new ToolStripMenuItem("Exit", null, (_, _) => ExitApplication());

        var menu = new ContextMenuStrip();
        menu.Items.Add(openItem);
        menu.Items.Add(startItem);
        menu.Items.Add(stopItem);
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add(exitItem);

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

        if (!startHidden)
        {
            OpenManager();
        }
    }

    private MainForm GetOrCreateMainForm()
    {
        if (mainForm is null || mainForm.IsDisposed)
        {
            mainForm = new MainForm(controller);
        }
        return mainForm;
    }

    private void OpenManager()
    {
        var form = GetOrCreateMainForm();

        if (!form.Visible)
        {
            form.Show();
        }

        if (form.WindowState == FormWindowState.Minimized)
        {
            form.WindowState = FormWindowState.Normal;
        }

        form.Activate();
        form.BringToFront();
    }

    private async Task RunTrayActionAsync(string title, Action action)
    {
        startItem.Enabled = false;
        stopItem.Enabled = false;
        try
        {
            await Task.Run(action);
            notifyIcon.ShowBalloonTip(3000, "OpenVINO Model Server", $"{title} completed.", ToolTipIcon.Info);
            if (mainForm is { IsDisposed: false })
            {
                await mainForm.RefreshDashboardAsync();
            }
        }
        catch (Exception ex)
        {
            notifyIcon.ShowBalloonTip(5000, "OpenVINO Model Server", $"{title} failed: {ex.Message}", ToolTipIcon.Error);
        }
        finally
        {
            startItem.Enabled = true;
            stopItem.Enabled = true;
        }
    }

    private void ExitApplication()
    {
        notifyIcon.Visible = false;
        if (mainForm is { IsDisposed: false })
        {
            mainForm.AllowExit = true;
            mainForm.Close();
        }
        ExitThread();
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
}
