namespace OVMS.Manager;

internal static class Program
{
    private const string SingleInstanceMutexName = "OVMS.Manager.SingleInstance";

    [STAThread]
    private static void Main(string[] args)
    {
        using var singleInstanceMutex = new Mutex(initiallyOwned: true, SingleInstanceMutexName, out var createdNew);
        if (!createdNew)
        {
            // Another instance already owns the Manager; exit quietly.
            return;
        }

        var startHidden = args.Any(a =>
            string.Equals(a, "--tray", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(a, "/tray", StringComparison.OrdinalIgnoreCase));

        ApplicationConfiguration.Initialize();
        using var app = new TrayApplicationContext(startHidden);
        Application.Run(app);

        GC.KeepAlive(singleInstanceMutex);
    }
}
