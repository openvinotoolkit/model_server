#define AppName "OpenVINO Model Server"
#define AppPublisher "OpenVINO"
#define AppExeName "OVMS.Manager.exe"
#ifndef SourceRoot
#define SourceRoot "..\..\.."
#endif
#ifndef OvmsSourceDir
#define OvmsSourceDir SourceRoot + "\dist\windows\ovms"
#endif
#ifndef ManagerPublishDir
#define ManagerPublishDir SourceRoot + "\packaging\windows\manager\artifacts\publish"
#endif
#ifndef OutputDir
#define OutputDir SourceRoot + "\dist\windows"
#endif

[Setup]
AppId={{9F35B167-C72A-4E1D-93E9-448CE4AE5270}
AppName={#AppName}
AppVersion=1.0.0
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\OVMS
DefaultGroupName=OpenVINO Model Server
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=OVMS-Setup
Compression=lzma
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\OVMS.Manager.exe
CloseApplications=yes
RestartApplications=no
WizardStyle=modern
WizardSizePercent=120
WizardImageFile=assets\ovms-welcome.bmp
WizardSmallImageFile=assets\ovms-small.bmp
WizardImageStretch=yes
DisableWelcomePage=no
ShowLanguageDialog=no
DisableReadyPage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "path"; Description: "Add OVMS to PATH"; GroupDescription: "Optional configuration:"; Flags: checkedonce
Name: "startup"; Description: "Start tray app when I sign in"; GroupDescription: "Optional configuration:"; Flags: checkedonce

[Files]
Source: "{#OvmsSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#ManagerPublishDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceRoot}\packaging\windows\scripts\*.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#SourceRoot}\packaging\windows\settings\settings.template.json"; DestDir: "{app}\templates"; Flags: ignoreversion
Source: "{#OvmsSourceDir}\*"; DestDir: "{localappdata}\OVMS\packages\source"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Open OVMS Manager"; Filename: "{app}\OVMS.Manager.exe"
Name: "{group}\Start OVMS Server"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\start-ovms.ps1"" -DataDir ""{localappdata}\OVMS"""
Name: "{group}\Stop OVMS Server"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\stop-ovms.ps1"" -DataDir ""{localappdata}\OVMS"""
Name: "{group}\Repair OVMS Server"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\repair-package.ps1"" -InstallDir ""{app}"" -DataDir ""{localappdata}\OVMS"""
Name: "{group}\Uninstall OpenVINO Model Server"; Filename: "{uninstallexe}"

[Registry]
Root: HKCU; Subkey: "Software\OpenVINO\OVMS"; ValueType: string; ValueName: "InstallDir"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\OpenVINO\OVMS"; ValueType: string; ValueName: "DataDir"; ValueData: "{localappdata}\OVMS"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\OpenVINO\OVMS"; ValueType: string; ValueName: "Version"; ValueData: "1.0.0"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "OVMS Manager"; ValueData: """{app}\OVMS.Manager.exe"""; Tasks: startup; Flags: uninsdeletevalue

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\configure-ovms.ps1"" -InstallDir ""{app}"" -DataDir ""{localappdata}\OVMS"" -PackageVariant ""python_on"""; Flags: runhidden waituntilterminated
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\set-path.ps1"" -InstallDir ""{app}"" -Action Add"; Tasks: path; Flags: runhidden waituntilterminated
Filename: "{app}\OVMS.Manager.exe"; Description: "Launch OVMS Manager"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\uninstall-ovms.ps1"" -InstallDir ""{app}"" -DataDir ""{localappdata}\OVMS"" -DataMode ""{code:GetDataMode}"""; Flags: runhidden waituntilterminated; RunOnceId: "OVMSCleanup"

[Code]
var
  DataModeChoice: String;
  // Existing-install detection state, populated in InitializeSetup and
  // consumed by InitializeWizard/NextButtonClick to drive the custom page.
  ExistingInstallDetected: Boolean;
  ExistingInstallDir: String;
  ExistingUninstallString: String;
  ExistingInstallPage: TWizardPage;
  RbRepair: TNewRadioButton;
  RbUninstall: TNewRadioButton;
  // When True, the "Exit Setup?" confirmation is suppressed so Setup can
  // close cleanly after handing off to the existing uninstaller.
  SuppressCancelConfirm: Boolean;

function GetDataMode(Param: String): String;
begin
  Result := DataModeChoice;
end;
function QueryExistingManagedInstall(var ExistingDir: String; var UninstallString: String): Boolean;
begin
  Result := False;
  ExistingDir := '';
  UninstallString := '';

  if RegQueryStringValue(HKCU, 'Software\OpenVINO\OVMS', 'InstallDir', ExistingDir) then
  begin
    RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{9F35B167-C72A-4E1D-93E9-448CE4AE5270}_is1', 'UninstallString', UninstallString);
    Result := True;
    exit;
  end;

  if RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{9F35B167-C72A-4E1D-93E9-448CE4AE5270}_is1', 'InstallLocation', ExistingDir) then
  begin
    RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{9F35B167-C72A-4E1D-93E9-448CE4AE5270}_is1', 'UninstallString', UninstallString);
    Result := True;
    exit;
  end;

  if DirExists(ExpandConstant('{localappdata}\Programs\OVMS')) then
  begin
    ExistingDir := ExpandConstant('{localappdata}\Programs\OVMS');
    RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{9F35B167-C72A-4E1D-93E9-448CE4AE5270}_is1', 'UninstallString', UninstallString);
    Result := True;
  end;
end;

function HasRunningOvmsProcess(): Boolean;
var
  ResultCode: Integer;
  MarkerPath: String;
begin
  MarkerPath := ExpandConstant('{tmp}\ovms-running.txt');
  DeleteFile(MarkerPath);
  Exec('powershell.exe',
    '-NoProfile -ExecutionPolicy Bypass -Command "if (Get-Process ovms -ErrorAction SilentlyContinue) { ''''running'''' | Set-Content -Path ''''' + MarkerPath + ''''' }"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := FileExists(MarkerPath);
  DeleteFile(MarkerPath);
end;

function RunExistingUninstaller(UninstallString: String): Boolean;
var
  ResultCode: Integer;
begin
  Result := False;
  if UninstallString = '' then
  begin
    MsgBox('An existing OpenVINO Model Server install was detected, but its uninstaller could not be found.', mbError, MB_OK);
    exit;
  end;

  Exec('cmd.exe', '/C ' + UninstallString, '', SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode);
  Result := ResultCode = 0;
end;

procedure StopRunningManager();
var
  ResultCode: Integer;
begin
  Exec('taskkill.exe', '/IM OVMS.Manager.exe /F', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  StopRunningManager();

  // Existing-install detection: just stash the result in globals. The actual
  // repair-vs-uninstall choice is now presented as a custom wizard page
  // (see InitializeWizard/NextButtonClick below) instead of a MsgBox here.
  // For silent installs there is no UI at all, so default to repair/upgrade
  // (i.e. just proceed) -- ExistingInstallDetected is still set, but since
  // the custom page is never shown/navigated in silent mode, RbRepair's
  // default behavior (proceed) is effectively what happens.
  ExistingInstallDetected := QueryExistingManagedInstall(ExistingInstallDir, ExistingUninstallString);

  if HasRunningOvmsProcess() then
  begin
    if WizardSilent() then
    begin
      Result := True;
      exit;
    end;

    if MsgBox(
      'A running ovms.exe process was detected, but it does not appear to be managed by this installer.' + #13#10#13#10 +
      'Setup will not repair or uninstall unmanaged portable OVMS processes. Stop the running server before installing if it uses the same ports.' + #13#10#13#10 +
      'Continue setup?',
      mbConfirmation, MB_YESNO) = IDNO then
    begin
      Result := False;
    end;
  end;
end;

procedure InitializeWizard();
var
  Lbl: TNewStaticText;
begin
  if not ExistingInstallDetected then
    exit;

  ExistingInstallPage := CreateCustomPage(wpWelcome,
    'Existing Installation Found',
    'OpenVINO Model Server is already installed on this computer.');

  Lbl := TNewStaticText.Create(ExistingInstallPage);
  Lbl.Parent := ExistingInstallPage.Surface;
  Lbl.Left := 0;
  Lbl.Top := 0;
  Lbl.Width := ExistingInstallPage.SurfaceWidth;
  Lbl.AutoSize := False;
  Lbl.WordWrap := True;
  Lbl.Caption :=
    'An existing install was found at:' + #13#10 +
    ExistingInstallDir + #13#10#13#10 +
    'Choose how to continue:';
  Lbl.Height := ScaleY(60);

  RbRepair := TNewRadioButton.Create(ExistingInstallPage);
  RbRepair.Parent := ExistingInstallPage.Surface;
  RbRepair.Left := 0;
  RbRepair.Top := Lbl.Top + Lbl.Height + ScaleY(8);
  RbRepair.Width := ExistingInstallPage.SurfaceWidth;
  RbRepair.Height := ScaleY(17);
  RbRepair.Caption := 'Repair or upgrade the existing installation (recommended)';
  RbRepair.Checked := True;

  RbUninstall := TNewRadioButton.Create(ExistingInstallPage);
  RbUninstall.Parent := ExistingInstallPage.Surface;
  RbUninstall.Left := 0;
  RbUninstall.Top := RbRepair.Top + RbRepair.Height + ScaleY(8);
  RbUninstall.Width := ExistingInstallPage.SurfaceWidth;
  RbUninstall.Height := ScaleY(17);
  RbUninstall.Caption := 'Uninstall the existing installation, then exit';
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;

  if (ExistingInstallPage <> nil) and (CurPageID = ExistingInstallPage.ID) then
  begin
    if RbUninstall.Checked then
    begin
      RunExistingUninstaller(ExistingUninstallString);
      // Hand-off to the existing uninstaller is done; close Setup cleanly.
      // SuppressCancelConfirm tells CancelButtonClick to skip the normal
      // "Setup is not complete. Exit Setup?" confirmation, so the user is
      // not nagged after deliberately choosing to uninstall and exit.
      MsgBox('The existing OpenVINO Model Server installation has been removed.' + #13#10#13#10 +
        'Setup will now close. Run it again to install a fresh copy.',
        mbInformation, MB_OK);
      SuppressCancelConfirm := True;
      WizardForm.Close;
      Result := False;
      exit;
    end;

    // RbRepair.Checked (default): proceed with repair/upgrade as normal.
    Result := True;
  end;
end;

procedure CancelButtonClick(CurPageID: Integer; var Cancel, Confirm: Boolean);
begin
  // After the "Uninstall existing, then exit" path triggers WizardForm.Close,
  // skip the "Setup is not complete" confirmation prompt.
  if SuppressCancelConfirm then
    Confirm := False;
end;

function InitializeUninstall(): Boolean;
var
  DataDir: String;
  KeepEverything: Integer;
  RemoveModelsToo: Integer;
begin
  Result := True;
  DataDir := ExpandConstant('{localappdata}\OVMS');
  DataModeChoice := 'PreserveAll';

  if UninstallSilent() then
  begin
    exit;
  end;

  KeepEverything := MsgBox(
    'Keep your downloaded models and data under:' + #13#10 +
    DataDir + #13#10#13#10 +
    'Yes = keep everything (models, settings, and logs)' + #13#10 +
    'No = choose what to remove',
    mbConfirmation, MB_YESNO);

  if KeepEverything = IDYES then
  begin
    DataModeChoice := 'PreserveAll';
    exit;
  end;

  RemoveModelsToo := MsgBox(
    'Also delete your downloaded MODELS?' + #13#10#13#10 +
    'Yes = remove everything, including models' + #13#10 +
    'No = keep models, but remove settings and logs',
    mbConfirmation, MB_YESNO);

  if RemoveModelsToo = IDYES then
  begin
    DataModeChoice := 'RemoveAll';
  end
  else
  begin
    DataModeChoice := 'RemoveSettingsKeepModels';
  end;
end;
