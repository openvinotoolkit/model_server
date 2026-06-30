using System.Drawing;

namespace OVMS.Manager;

/// <summary>
/// Centralized color/font tokens for the Manager's light theme.
/// </summary>
internal static class Theme
{
    public static readonly Color WindowBackground = ColorTranslator.FromHtml("#F3F4F6");
    public static readonly Color Surface = ColorTranslator.FromHtml("#FFFFFF");
    public static readonly Color Rail = ColorTranslator.FromHtml("#F8F7FC");
    public static readonly Color Accent = ColorTranslator.FromHtml("#6C2BD9");
    public static readonly Color AccentHover = ColorTranslator.FromHtml("#5B21B6");
    public static readonly Color Text = ColorTranslator.FromHtml("#1F2937");
    public static readonly Color Muted = ColorTranslator.FromHtml("#6B7280");
    public static readonly Color Border = ColorTranslator.FromHtml("#E5E7EB");
    public static readonly Color Success = ColorTranslator.FromHtml("#16A34A");
    public static readonly Color Danger = ColorTranslator.FromHtml("#DC2626");
    public static readonly Color Warning = ColorTranslator.FromHtml("#D97706");

    public static readonly Color AccentTint = ColorTranslator.FromHtml("#EFE6FB");
    public static readonly Color HoverTint = ColorTranslator.FromHtml("#F1F0F4");

    public static readonly Color LogBackground = ColorTranslator.FromHtml("#1E1E1E");
    public static readonly Color LogForeground = ColorTranslator.FromHtml("#D4D4D4");

    public static Font BaseFont { get; } = new("Segoe UI", 9.75f);
    public static Font SemiboldFont { get; } = new("Segoe UI Semibold", 9.75f);
    public static Font PageTitleFont { get; } = new("Segoe UI Semibold", 16f);
    public static Font CardTitleFont { get; } = new("Segoe UI Semibold", 9.75f);
    public static Font CardValueFont { get; } = new("Segoe UI Semibold", 13f);
    public static Font NavFont { get; } = new("Segoe UI", 10f);
    public static Font MonoFont { get; } = new("Consolas", 9.5f);

    private static Font? iconFont;

    /// <summary>
    /// Returns a font for rendering Segoe Fluent/MDL2 glyphs. If neither
    /// family is installed, GDI silently substitutes a fallback font, so
    /// this never throws -- worst case the glyph renders as a box/blank.
    /// </summary>
    public static Font IconFont(float size = 13f)
    {
        if (iconFont != null && Math.Abs(iconFont.Size - size) < 0.01f)
        {
            return iconFont;
        }

        Font font;
        try
        {
            font = new Font("Segoe Fluent Icons", size, FontStyle.Regular, GraphicsUnit.Point);
            if (!string.Equals(font.Name, "Segoe Fluent Icons", StringComparison.OrdinalIgnoreCase))
            {
                font.Dispose();
                font = new Font("Segoe MDL2 Assets", size, FontStyle.Regular, GraphicsUnit.Point);
            }
        }
        catch
        {
            font = new Font("Segoe MDL2 Assets", size, FontStyle.Regular, GraphicsUnit.Point);
        }

        iconFont = font;
        return iconFont;
    }
}

/// <summary>
/// Segoe Fluent / MDL2 Assets glyph code points used throughout the UI.
/// Written as \uXXXX escapes (Private Use Area) to avoid any source-encoding
/// ambiguity.
/// </summary>
internal static class Glyphs
{
    public const string Dashboard = ""; // Home
    public const string Settings = "";  // Settings gear
    public const string Logs = "";      // Diagnostic
    public const string Advanced = "";  // Repair/tools (Build)

    public const string Play = "";       // Play
    public const string Stop = "";       // Stop
    public const string Restart = "";    // Sync/restart
    public const string Refresh = "";    // Refresh
    public const string OpenFolder = ""; // OpenFolderHorizontal
    public const string Repair = "";     // Repair/build tools
    public const string Save = "";       // Save
    public const string Health = "";     // HealthSolid
    public const string Export = "";     // Export
}
