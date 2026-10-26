#define AppName "OpenVINO Model Server"
#define AppPublisher "OpenVINO"
#ifndef SourceRoot
#define SourceRoot "..\..\.."
#endif
#ifndef OvmsSourceDir
#define OvmsSourceDir SourceRoot + "\dist\windows\ovms"
#endif
#ifndef OutputDir
#define OutputDir SourceRoot + "\dist\windows"
#endif

[Setup]
AppId={{9F35B167-C72A-4E1D-93E9-448CE4AE5270}
AppName={#AppName}
AppVersion=2026.5.0
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

[Components]
Name: "core"; Description: "Core OVMS installation"; Types: full compact custom; Flags: fixed
Name: "python_system"; Description: "Use existing Python on this system and install Jinja, Setuptools and MarkupSafe"; Types: full custom; Flags: exclusive
Name: "python_embedded"; Description: "Bundle embedded Python runtime"; Types: full custom; Flags: exclusive
Name: "espeak"; Description: "espeak / Kokoro support"; Types: full custom
Name: "gstreamer"; Description: "GStreamer for media handling"; Types: full custom
Name: "npu"; Description: "NPU OpenVINO backend"; Types: full custom
Name: "gpu"; Description: "Intel GPU driver / memory guidance"; Types: full custom
Name: "cloud"; Description: "Cloud storage adapters (S3, Azure, GCS)"; Types: custom; Flags: fixed
Name: "opencv_mp"; Description: "OpenCV for MediaPipe calculator"; Types: full custom
Name: "gguf_frontend"; Description: "OpenVINO GGUF frontend"; Types: full custom
Name: "ir_frontend"; Description: "OpenVINO IR frontend"; Types: full custom
Name: "onnx_frontend"; Description: "OpenVINO ONNX frontend"; Types: full custom
Name: "paddle_frontend"; Description: "OpenVINO PaddlePaddle frontend"; Types: full custom
Name: "pytorch_frontend"; Description: "OpenVINO PyTorch frontend"; Types: full custom
Name: "tensorflow_frontend"; Description: "OpenVINO TensorFlow frontend"; Types: full custom
Name: "tensorflow_lite_frontend"; Description: "OpenVINO TensorFlow Lite frontend"; Types: full custom

[Files]
Source: "{#OvmsSourceDir}\ovms.exe"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\install_ovms_service.bat"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\setupvars.bat"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\setupvars.ps1"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\LICENSE"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_c.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_genai.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_genai_c.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_tokenizers.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_auto_plugin.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_auto_batch_plugin.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_hetero_plugin.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_intel_cpu_plugin.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\opencv_world4140.dll"; DestDir: "{app}"; Components: opencv_mp; Flags: ignoreversion
Source: "{#OvmsSourceDir}\libcurl-x64.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\libovmspython.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\libpython_calculators.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\git2.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\tbb12.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\thirdparty-licenses\*"; DestDir: "{app}\thirdparty-licenses"; Components: core; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#OvmsSourceDir}\python\*"; DestDir: "{app}\python"; Components: python_embedded; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#OvmsSourceDir}\espeak-ng.dll"; DestDir: "{app}"; Components: espeak; Flags: ignoreversion
Source: "{#OvmsSourceDir}\espeak-ng-data\*"; DestDir: "{app}\espeak-ng-data"; Components: espeak; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#OvmsSourceDir}\openvino_intel_gpu_plugin.dll"; DestDir: "{app}"; Components: gpu; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_intel_npu_compiler.dll"; DestDir: "{app}"; Components: npu; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_intel_npu_compiler_loader.dll"; DestDir: "{app}"; Components: npu; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_intel_npu_plugin.dll"; DestDir: "{app}"; Components: npu; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_intel_npu_vm_runtime.dll"; DestDir: "{app}"; Components: npu; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_gguf_frontend.dll"; DestDir: "{app}"; Components: gguf_frontend; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_ir_frontend.dll"; DestDir: "{app}"; Components: ir_frontend; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_onnx_frontend.dll"; DestDir: "{app}"; Components: onnx_frontend; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_paddle_frontend.dll"; DestDir: "{app}"; Components: paddle_frontend; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_pytorch_frontend.dll"; DestDir: "{app}"; Components: pytorch_frontend; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_tensorflow_frontend.dll"; DestDir: "{app}"; Components: tensorflow_frontend; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_tensorflow_lite_frontend.dll"; DestDir: "{app}"; Components: tensorflow_lite_frontend; Flags: ignoreversion
Source: "{#SourceRoot}\packaging\windows\scripts\*.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#SourceRoot}\packaging\windows\settings\settings.template.json"; DestDir: "{app}\templates"; Flags: ignoreversion
Source: "{#OvmsSourceDir}\*"; DestDir: "{localappdata}\OVMS\packages\source"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Start OVMS Server"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\start-ovms.ps1"" -DataDir ""{localappdata}\OVMS"""
Name: "{group}\Stop OVMS Server"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\stop-ovms.ps1"" -DataDir ""{localappdata}\OVMS"""
Name: "{group}\Repair OVMS Server"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\repair-package.ps1"" -InstallDir ""{app}"" -DataDir ""{localappdata}\OVMS"""
Name: "{group}\Uninstall OpenVINO Model Server"; Filename: "{uninstallexe}"

[Registry]
Root: HKCU; Subkey: "Software\OpenVINO\OVMS"; ValueType: string; ValueName: "InstallDir"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\OpenVINO\OVMS"; ValueType: string; ValueName: "DataDir"; ValueData: "{localappdata}\OVMS"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\OpenVINO\OVMS"; ValueType: string; ValueName: "Version"; ValueData: "1.0.0"; Flags: uninsdeletekey

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\set-path.ps1"" -InstallDir ""{app}"" -Action Add"; Tasks: path; Flags: runhidden waituntilterminated

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
  PythonSelectionPage: TWizardPage;
  PythonSelectionLabel: TNewStaticText;
  PythonSelectionMode: String;
  // When True, the "Exit Setup?" confirmation is suppressed so Setup can
  // close cleanly after handing off to the existing uninstaller.
  SuppressCancelConfirm: Boolean;

function BoolToJson(Value: Boolean): String;
begin
  if Value then
    Result := 'true'
  else
    Result := 'false';
end;

function GetPythonModeValue(Param: String): String;
begin
  if WizardIsComponentSelected('python_system') then
    Result := 'system'
  else
    Result := 'embedded';
end;

function DetectSystemPythonPath: String;
begin
  Result := '';

  if FileExists(ExpandConstant('{commonpf64}\Python312\python.exe')) then
    Result := ExpandConstant('{commonpf64}\Python312\python.exe')
  else if FileExists(ExpandConstant('{localappdata}\Programs\Python\Python312\python.exe')) then
    Result := ExpandConstant('{localappdata}\Programs\Python\Python312\python.exe')
  else if FileExists(ExpandConstant('{win}\python.exe')) then
    Result := ExpandConstant('{win}\python.exe');
end;

function GetPythonDetectionSummary: String;
var
  PythonPath: String;
begin
  PythonPath := DetectSystemPythonPath;

  if PythonPath = '' then
  begin
    Result :=
      'ERROR: No compatible Python 3.12 interpreter was detected on this system.' + #13#10 +
      'OVMS cannot use the system-Python option in this environment.' + #13#10 + #13#10 +
      'Package installation status: not available for this interpreter.' + #13#10 +
      'If you want a self-contained install, press Back and choose the embedded Python option.';
  end
  else
  begin
    Result :=
      'Detected system Python: ' + PythonPath + #13#10 +
      'OVMS will validate this interpreter and install the required Python packages if needed.' + #13#10 + #13#10 +
      'Package installation status: the installer will report success or failure during configuration.' + #13#10 +
      'If this interpreter is not the one you want, press Back and select the embedded Python option instead.';
  end;
end;

function GetSelectedPythonVariant: String;
begin
  if PythonSelectionMode <> '' then
    Result := PythonSelectionMode
  else if WizardIsComponentSelected('python_system') then
    Result := 'system'
  else
    Result := 'embedded';
end;

function IsSystemPythonAvailable: Boolean;
begin
  Result := DetectSystemPythonPath <> '';
end;

function GetDataMode(Param: String): String;
begin
  Result := DataModeChoice;
end;

function IsGstreamerAvailable: Boolean;
begin
  Result :=
    DirExists(ExpandConstant('{sd}\gstreamer')) or
    DirExists(ExpandConstant('{commonpf64}\GStreamer')) or
    DirExists(ExpandConstant('{commonpf32}\GStreamer')) or
    FileExists(ExpandConstant('{sd}\gstreamer\1.0\x86_64\bin\gst-launch-1.0.exe')) or
    FileExists(ExpandConstant('{commonpf64}\GStreamer\1.0\x86_64\bin\gst-launch-1.0.exe')) or
    FileExists(ExpandConstant('{commonpf32}\GStreamer\1.0\x86_64\bin\gst-launch-1.0.exe'));
end;

function IsIntelGpuDriverAvailable: Boolean;
begin
  Result :=
    FileExists(ExpandConstant('{sys}\igfxicd64.dll')) or
    FileExists(ExpandConstant('{sys}\igfxicd64.exe')) or
    DirExists(ExpandConstant('{commonpf64}\Intel')) or
    DirExists(ExpandConstant('{commonpf32}\Intel'));
end;

function IsNpuDriverAvailable: Boolean;
begin
  Result :=
    FileExists(ExpandConstant('{sys}\IntelVpuDriver.dll')) or
    DirExists(ExpandConstant('{commonpf64}\Intel\NPU')) or
    DirExists(ExpandConstant('{commonpf32}\Intel\NPU'));
end;

procedure WriteOptionalInstallSelection;
var
  FilePath: String;
  Json: String;
begin
  FilePath := ExpandConstant('{localappdata}\OVMS\install-options.json');
  CreateDir(ExpandConstant('{localappdata}\OVMS'));

  Json := '{' +
    '"pythonMode":"' + GetPythonModeValue('') +
    '",' +
    '"espeak":' + BoolToJson(WizardIsComponentSelected('espeak')) + ',' +
    '"gstreamer":' + BoolToJson(WizardIsComponentSelected('gstreamer')) + ',' +
    '"npu":' + BoolToJson(WizardIsComponentSelected('npu')) + ',' +
    '"gpu":' + BoolToJson(WizardIsComponentSelected('gpu')) + ',' +
    '"opencv_mp":' + BoolToJson(WizardIsComponentSelected('opencv_mp')) + ',' +
    '"cloud_storage":false,' +
    '"openvino_frontends":{' +
      '"gguf":' + BoolToJson(WizardIsComponentSelected('gguf_frontend')) + ',' +
      '"ir":' + BoolToJson(WizardIsComponentSelected('ir_frontend')) + ',' +
      '"onnx":' + BoolToJson(WizardIsComponentSelected('onnx_frontend')) + ',' +
      '"paddle":' + BoolToJson(WizardIsComponentSelected('paddle_frontend')) + ',' +
      '"pytorch":' + BoolToJson(WizardIsComponentSelected('pytorch_frontend')) + ',' +
      '"tensorflow":' + BoolToJson(WizardIsComponentSelected('tensorflow_frontend')) + ',' +
      '"tensorflow_lite":' + BoolToJson(WizardIsComponentSelected('tensorflow_lite_frontend')) +
    '}' +
    '}';

  SaveStringToFile(FilePath, Json, False);
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

function HasRunningOvmsProcess: Boolean;
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

function InitializeSetup: Boolean;
begin
  Result := True;

  // Existing-install detection: just stash the result in globals. The actual
  // repair-vs-uninstall choice is now presented as a custom wizard page
  // (see InitializeWizard/NextButtonClick below) instead of a MsgBox here.
  // For silent installs there is no UI at all, so default to repair/upgrade
  // (i.e. just proceed) -- ExistingInstallDetected is still set, but since
  // the custom page is never shown/navigated in silent mode, RbRepair's
  // default behavior (proceed) is effectively what happens.
  ExistingInstallDetected := QueryExistingManagedInstall(ExistingInstallDir, ExistingUninstallString);

  if HasRunningOvmsProcess then
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

procedure InitializeWizard;
var
  Lbl: TNewStaticText;
begin
  PythonSelectionMode := 'embedded';
  if WizardIsComponentSelected('python_system') then
    PythonSelectionMode := 'system'
  else if WizardIsComponentSelected('python_embedded') then
    PythonSelectionMode := 'embedded';

  if ExistingInstallDetected then
  begin
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

  if WizardIsComponentSelected('python_system') or WizardIsComponentSelected('python_embedded') then
  begin
    PythonSelectionPage := CreateCustomPage(wpSelectComponents,
      'Python detection',
      'OVMS Python configuration');

    PythonSelectionLabel := TNewStaticText.Create(PythonSelectionPage);
    PythonSelectionLabel.Parent := PythonSelectionPage.Surface;
    PythonSelectionLabel.Left := 0;
    PythonSelectionLabel.Top := 0;
    PythonSelectionLabel.Width := PythonSelectionPage.SurfaceWidth;
    PythonSelectionLabel.AutoSize := False;
    PythonSelectionLabel.WordWrap := True;

    if GetSelectedPythonVariant = 'system' then
      PythonSelectionLabel.Caption := GetPythonDetectionSummary
    else
      PythonSelectionLabel.Caption :=
        'Embedded Python runtime: selected.' + #13#10 +
        'OVMS will use the bundled Python installation under the application directory.' + #13#10 + #13#10 +
        'No system Python detection is required for this option.' + #13#10 +
        'If you want to use an existing system Python instead, use the Back button and choose the system-Python option.';
    PythonSelectionLabel.Height := ScaleY(110);

    if GetSelectedPythonVariant = 'system' then
    begin
      if not IsSystemPythonAvailable then
      begin
        Lbl := TNewStaticText.Create(PythonSelectionPage);
        Lbl.Parent := PythonSelectionPage.Surface;
        Lbl.Left := 0;
        Lbl.Top := PythonSelectionLabel.Top + PythonSelectionLabel.Height + ScaleY(10);
        Lbl.Width := PythonSelectionPage.SurfaceWidth;
        Lbl.AutoSize := False;
        Lbl.WordWrap := True;
        Lbl.Caption :=
          'ERROR: No compatible Python 3.12 interpreter was detected on this system.' + #13#10 +
          'Use the Back button to return to the component selection page and choose the embedded Python option instead.';
        Lbl.Height := ScaleY(60);
      end
      else
      begin
        Lbl := TNewStaticText.Create(PythonSelectionPage);
        Lbl.Parent := PythonSelectionPage.Surface;
        Lbl.Left := 0;
        Lbl.Top := PythonSelectionLabel.Top + PythonSelectionLabel.Height + ScaleY(10);
        Lbl.Width := PythonSelectionPage.SurfaceWidth;
        Lbl.AutoSize := False;
        Lbl.WordWrap := True;
        Lbl.Caption :=
          'Package installation status is checked during setup. If the required Python packages fail to install, the setup will stop and show the error.' + #13#10 +
          'If you want to avoid modifying your system Python, use the Back button and select the embedded Python option.';
        Lbl.Height := ScaleY(80);
      end;
    end
    else
    begin
      Lbl := TNewStaticText.Create(PythonSelectionPage);
      Lbl.Parent := PythonSelectionPage.Surface;
      Lbl.Left := 0;
      Lbl.Top := PythonSelectionLabel.Top + PythonSelectionLabel.Height + ScaleY(10);
      Lbl.Width := PythonSelectionPage.SurfaceWidth;
      Lbl.AutoSize := False;
      Lbl.WordWrap := True;
      Lbl.Caption :=
        'The bundled Python runtime will be installed with OVMS.' + #13#10 +
        'This keeps the installation self-contained and avoids changing the system Python environment.';
      Lbl.Height := ScaleY(60);
    end;
  end;
end;

function RunPythonConfigurationForSelectedVariant: Boolean;
var
  Variant: String;
  ResultCode: Integer;
  LogPath: String;
  Message: String;
begin
  Result := True;

  Variant := GetSelectedPythonVariant;
  if Variant = '' then
    exit;

  LogPath := ExpandConstant('{src}\ovms-python-config.log');
  DeleteFile(LogPath);

  Exec('cmd.exe',
    '/C powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{app}\scripts\configure-ovms.ps1" -InstallDir "{app}" -DataDir "{localappdata}\OVMS" -PackageVariant "' + Variant + '" 1> "' + LogPath + '" 2>&1',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

  if ResultCode = 0 then
  begin
    if Variant = 'system' then
      Message := 'System Python configuration succeeded.' + #13#10 + 'The detected Python 3.12 environment is ready for OVMS.'
    else
      Message := 'Embedded Python configuration succeeded.';

    if FileExists(LogPath) then
      Message := Message + #13#10 + #13#10 + 'Log file: ' + LogPath;

    MsgBox(Message, mbInformation, MB_OK);
    Result := True;
  end
  else
  begin
    Message := 'OVMS configuration failed for the selected ' + Variant + ' Python mode.' + #13#10 +
      'The detailed output was written to:' + #13#10 + LogPath;
    if not FileExists(LogPath) then
      Message := 'OVMS configuration failed for the selected ' + Variant + ' Python mode.' + #13#10 +
        'No detailed output was captured. Please check the installer log and try again.';
    MsgBox(Message, mbError, MB_OK);
    Result := False;
  end;
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
      SuppressCancelConfirm := True;
      WizardForm.Close;
      Result := False;
      exit;
    end;

    // RbRepair.Checked (default): proceed with repair/upgrade as normal.
    Result := True;
  end;

  if (PythonSelectionPage <> nil) and (CurPageID = PythonSelectionPage.ID) then
  begin
    Result := RunPythonConfigurationForSelectedVariant;
    if not Result then
      exit;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    WriteOptionalInstallSelection;
end;

procedure CancelButtonClick(CurPageID: Integer; var Cancel, Confirm: Boolean);
begin
  // After the "Uninstall existing, then exit" path triggers WizardForm.Close,
  // skip the "Setup is not complete" confirmation prompt.
  if SuppressCancelConfirm then
    Confirm := False;
end;

function InitializeUninstall: Boolean;
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
