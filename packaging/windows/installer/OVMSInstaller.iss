; Copyright (c) 2026 Intel Corporation
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;      http://www.apache.org/licenses/LICENSE-2.0
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS,
; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; See the License for the specific language governing permissions and
; limitations under the License.

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
#define MyAppURL "https://github.com/openvinotoolkit/model_server"

#ifndef OvmsVersion
#define OvmsVersion "2026.5.1"
#endif

[Setup]
AppId={{9F35B167-C72A-4E1D-93E9-448CE4AE5270}
AppName={#AppName}
AppVersion={#OvmsVersion}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
AppPublisher={#AppPublisher}
LicenseFile={#SourceRoot}\LICENSE
DefaultDirName={autopf}\OVMS
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

[Components]
Name: "core"; Description: "Core OVMS installation"; Types: full compact custom; Flags: fixed
Name: "python_system"; Description: "Use existing Python on this system and install Jinja, Setuptools and MarkupSafe"; Types: full custom; Flags: exclusive
Name: "python_embedded"; Description: "Bundle embedded Python runtime"; Types: full custom; Flags: exclusive
Name: "python_none"; Description: "Do not install bundled Python and do not use a system Python"; Types: full custom; Flags: exclusive
Name: "espeak"; Description: "eSpeak / Kokoro support"; Types: full custom
Name: "npu"; Description: "NPU OpenVINO backend"; Types: full custom
Name: "gpu"; Description: "Intel GPU driver / memory guidance"; Types: full custom
Name: "opencv_mp"; Description: "OpenCV for MediaPipe calculator"; Types: full custom
Name: "onnx_frontend"; Description: "OpenVINO ONNX frontend"; Types: full custom
Name: "paddle_frontend"; Description: "OpenVINO PaddlePaddle frontend"; Types: full custom
Name: "tensorflow_frontend"; Description: "OpenVINO TensorFlow and TensorFlow Lite frontends"; Types: full custom

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
Source: "{#OvmsSourceDir}\ovms_opencv_calculators.dll"; DestDir: "{app}"; Components: opencv_mp; Flags: ignoreversion
Source: "{#OvmsSourceDir}\ovms_opencv_image_decoder.dll"; DestDir: "{app}"; Components: opencv_mp; Flags: ignoreversion
Source: "{#OvmsSourceDir}\libcurl-x64.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\libovmspython.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\libpython_calculators.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\git2.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\tbb12.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#OvmsSourceDir}\thirdparty-licenses\*"; DestDir: "{app}\thirdparty-licenses"; Components: core; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#OvmsSourceDir}\python\pyovms.pyd"; DestDir: "{app}\python"; Components: python_system python_embedded; Flags: ignoreversion
Source: "{#OvmsSourceDir}\python\*"; DestDir: "{app}\python"; Components: python_embedded; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#OvmsSourceDir}\espeak-ng.dll"; DestDir: "{app}"; Components: espeak; Flags: ignoreversion
Source: "{#OvmsSourceDir}\espeak-ng-data\*"; DestDir: "{app}\espeak-ng-data"; Components: espeak; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#OvmsSourceDir}\openvino_intel_gpu_plugin.dll"; DestDir: "{app}"; Components: gpu; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_intel_npu_compiler.dll"; DestDir: "{app}"; Components: npu; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_intel_npu_compiler_loader.dll"; DestDir: "{app}"; Components: npu; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_intel_npu_plugin.dll"; DestDir: "{app}"; Components: npu; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_intel_npu_vm_runtime.dll"; DestDir: "{app}"; Components: npu; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_onnx_frontend.dll"; DestDir: "{app}"; Components: onnx_frontend; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_paddle_frontend.dll"; DestDir: "{app}"; Components: paddle_frontend; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_tensorflow_frontend.dll"; DestDir: "{app}"; Components: tensorflow_frontend; Flags: ignoreversion
Source: "{#OvmsSourceDir}\openvino_tensorflow_lite_frontend.dll"; DestDir: "{app}"; Components: tensorflow_frontend; Flags: ignoreversion
Source: "{#SourceRoot}\packaging\windows\scripts\configure-ovms.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#SourceRoot}\packaging\windows\scripts\install-service.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#SourceRoot}\packaging\windows\scripts\ovms-env.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#SourceRoot}\packaging\windows\scripts\repair-package.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#SourceRoot}\packaging\windows\scripts\set-path.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#SourceRoot}\packaging\windows\scripts\start-ovms.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#SourceRoot}\packaging\windows\scripts\stop-ovms.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#SourceRoot}\packaging\windows\scripts\uninstall-ovms.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#SourceRoot}\packaging\windows\scripts\uninstall-service.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#SourceRoot}\packaging\windows\scripts\run-ovms.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion

[Icons]
Name: "{group}\Start OVMS Server"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\start-ovms.ps1"" -DataDir ""{localappdata}\OVMS"""
Name: "{group}\Stop OVMS Server"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\stop-ovms.ps1"" -DataDir ""{localappdata}\OVMS"""
Name: "{group}\Repair OVMS Server"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\repair-package.ps1"" -InstallDir ""{app}"" -DataDir ""{localappdata}\OVMS"""
Name: "{group}\Uninstall OpenVINO Model Server"; Filename: "{uninstallexe}"

[Registry]
Root: HKCU; Subkey: "Software\OpenVINO\OVMS"; ValueType: string; ValueName: "InstallDir"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\OpenVINO\OVMS"; ValueType: string; ValueName: "DataDir"; ValueData: "{localappdata}\OVMS"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\OpenVINO\OVMS"; ValueType: string; ValueName: "Version"; ValueData: "{#OvmsVersion}"; Flags: uninsdeletekey

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\uninstall-ovms.ps1"" -InstallDir ""{app}"" -DataDir ""{localappdata}\OVMS"" -DataMode ""{code:GetDataMode}"""; Flags: runhidden waituntilterminated; RunOnceId: "OVMSCleanup"

[Code]
var
  DataModeChoice: String;
  // Existing-install detection state, populated in InitializeSetup and
  // consumed by InitializeWizard/NextButtonClick to drive the custom page.
  ExistingInstallDetected: Boolean;
  ExistingInstallDir: String;
  ExistingInstallVersion: String;
  ExistingUninstallString: String;
  ExistingInstallPage: TWizardPage;
  RbRepair: TNewRadioButton;
  RbUninstall: TNewRadioButton;
  RuntimeSummaryPage: TWizardPage;
  SettingsSummaryEdit: TNewMemo;
  AdditionalTasksPage: TWizardPage;
  AddOvmsToPathCheck: TNewCheckBox;
  ModelRepositoryPathEdit: TNewEdit;
  OvmsServiceModeRadio: TNewRadioButton;
  OvmsTerminalModeRadio: TNewRadioButton;
  OvmsServiceStatusLabel: TNewStaticText;
  StopOvmsServiceButton: TNewButton;
  OvmsProcessStatusLabel: TNewStaticText;
  StopOvmsProcessButton: TNewButton;
  // When True, the "Exit Setup?" confirmation is suppressed so Setup can
  // close cleanly after handing off to the existing uninstaller.
  SuppressCancelConfirm: Boolean;
  LegacyInstallMigrationPending: Boolean;
  LegacyInstallDirToMigrate: String;
  LegacyUninstallStringToMigrate: String;

procedure StopOvmsProcessButtonClick(Sender: TObject); forward;
procedure UpdateOvmsProcessStatus; forward;
function RunExistingUninstaller(UninstallString: String): Boolean; forward;

function BoolToJson(Value: Boolean): String;
begin
  if Value then
    Result := 'true'
  else
    Result := 'false';
end;

function GetSetupLogPath: String;
begin
  Result := AddBackslash(ExtractFileDir(ExpandConstant('{srcexe}'))) + 'OVMSSetup.log';
end;

procedure InitializeSetupLog;
var
  LogPath: String;
  MarkerText: String;
begin
  LogPath := GetSetupLogPath;
  if FileExists(LogPath) then
    DeleteFile(LogPath);

  MarkerText := '[SETUP] ==================== OVMS Setup started ====================' + #13#10;
  SaveStringToFile(LogPath, MarkerText, False);
end;

function GetPythonModeValue(Param: String): String;
begin
  if WizardIsComponentSelected('python_system') then
    Result := 'system'
  else if WizardIsComponentSelected('python_embedded') then
    Result := 'embedded'
  else
    Result := 'none';
end;

function DetectSystemPythonPath: String;
begin
  Result := '';

  if FileExists(ExpandConstant('{commonpf64}\Python314\python.exe')) then
    Result := ExpandConstant('{commonpf64}\Python314\python.exe')
  else if FileExists(ExpandConstant('{localappdata}\Programs\Python\Python314\python.exe')) then
    Result := ExpandConstant('{localappdata}\Programs\Python\Python314\python.exe')
  else if FileExists(ExpandConstant('{sd}\opt\Python314\python.exe')) then
    Result := ExpandConstant('{sd}\opt\Python314\python.exe')
  else if FileExists(ExpandConstant('{win}\python.exe')) then
    Result := ExpandConstant('{win}\python.exe');
end;

function GetSelectedPythonVariant: String;
begin
  if WizardIsComponentSelected('python_system') then
    Result := 'system'
  else if WizardIsComponentSelected('python_embedded') then
    Result := 'embedded'
  else if WizardIsComponentSelected('python_none') then
    Result := 'none'
  else
    Result := 'none';
end;

function GetDataMode(Param: String): String;
begin
  Result := DataModeChoice;
end;

function NormalizeVersionText(Value: String): String;
var
  Index: Integer;
  Ch: Char;
begin
  Result := '';
  for Index := 1 to Length(Value) do
  begin
    Ch := Value[Index];
    if ((Ch >= '0') and (Ch <= '9')) or (Ch = '.') then
      Result := Result + Ch;
  end;

  if Result = '' then
    Result := '0';
end;

function CompareVersionStrings(Left: String; Right: String): Integer;
var
  LeftValue: String;
  RightValue: String;
  LeftNum: Integer;
  RightNum: Integer;
  LeftPos: Integer;
  RightPos: Integer;
begin
  Result := 0;

  LeftValue := NormalizeVersionText(Left);
  RightValue := NormalizeVersionText(Right);

  while (LeftValue <> '') or (RightValue <> '') do
  begin
    LeftPos := Pos('.', LeftValue);
    RightPos := Pos('.', RightValue);

    if LeftPos = 0 then
    begin
      LeftNum := StrToIntDef(LeftValue, 0);
      LeftValue := '';
    end
    else
    begin
      LeftNum := StrToIntDef(Copy(LeftValue, 1, LeftPos - 1), 0);
      Delete(LeftValue, 1, LeftPos);
    end;

    if RightPos = 0 then
    begin
      RightNum := StrToIntDef(RightValue, 0);
      RightValue := '';
    end
    else
    begin
      RightNum := StrToIntDef(Copy(RightValue, 1, RightPos - 1), 0);
      Delete(RightValue, 1, RightPos);
    end;

    if LeftNum < RightNum then
    begin
      Result := -1;
      exit;
    end;

    if LeftNum > RightNum then
    begin
      Result := 1;
      exit;
    end;
  end;
end;

function GetInstalledOvmsVersion: String;
var
  VersionValue: String;
begin
  Result := '';

  if RegQueryStringValue(HKCU, 'Software\OpenVINO\OVMS', 'Version', VersionValue) then
    Result := VersionValue;

  if Result = '' then
    if RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{9F35B167-C72A-4E1D-93E9-448CE4AE5270}_is1', 'DisplayVersion', VersionValue) then
      Result := VersionValue;

  if Result = '' then
    if RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{9F35B167-C72A-4E1D-93E9-448CE4AE5270}_is1', 'Version', VersionValue) then
      Result := VersionValue;
end;

function IsInstalledOvmsOlderThanCurrent: Boolean;
begin
  Result := False;
  ExistingInstallVersion := GetInstalledOvmsVersion;
  if ExistingInstallVersion = '' then
    exit;

  Result := CompareVersionStrings(ExistingInstallVersion, '{#OvmsVersion}') < 0;
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

function FindDriverFileInDirectory(Path: String; FileName: String): String;
var
  FindRec: TFindRec;
  CandidatePath: String;
begin
  Result := '';
  if not DirExists(Path) then
    Exit;

  if FindFirst(AddBackslash(Path) + '*', FindRec) then
  begin
    try
      repeat
        if (FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0 then
        begin
          if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
          begin
            CandidatePath := AddBackslash(Path) + FindRec.Name;
            Result := FindDriverFileInDirectory(CandidatePath, FileName);
            if Result <> '' then
              Exit;
          end;
        end
        else if CompareText(FindRec.Name, FileName) = 0 then
        begin
          Result := AddBackslash(Path) + FindRec.Name;
          Exit;
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

function GetIntelGpuDriverPath: String;
var
  SearchPath: String;
  Candidate: String;
begin
  Result := '';

  if FileExists(ExpandConstant('{sys}\igfxicd64.dll')) then
    Result := ExpandConstant('{sys}\igfxicd64.dll')
  else if FileExists(ExpandConstant('{sys}\igfxicd64.exe')) then
    Result := ExpandConstant('{sys}\igfxicd64.exe')
  else if FileExists(ExpandConstant('{commonpf64}\Intel\igfxicd64.dll')) then
    Result := ExpandConstant('{commonpf64}\Intel\igfxicd64.dll')
  else if FileExists(ExpandConstant('{commonpf64}\Intel\igfxicd64.exe')) then
    Result := ExpandConstant('{commonpf64}\Intel\igfxicd64.exe')
  else if FileExists(ExpandConstant('{commonpf32}\Intel\igfxicd64.dll')) then
    Result := ExpandConstant('{commonpf32}\Intel\igfxicd64.dll')
  else if FileExists(ExpandConstant('{commonpf32}\Intel\igfxicd64.exe')) then
    Result := ExpandConstant('{commonpf32}\Intel\igfxicd64.exe')
  else
  begin
    if DirExists(ExpandConstant('{commonpf64}\Intel')) then
    begin
      SearchPath := ExpandConstant('{commonpf64}\Intel');
      Candidate := FindDriverFileInDirectory(SearchPath, 'igfxicd64.dll');
      if Candidate = '' then
        Candidate := FindDriverFileInDirectory(SearchPath, 'igfxicd64.exe');
      if Candidate <> '' then
        Result := Candidate
      else
        Result := AddBackslash(SearchPath) + 'igfxicd64.dll';
      Exit;
    end;

    if DirExists(ExpandConstant('{commonpf32}\Intel')) then
    begin
      SearchPath := ExpandConstant('{commonpf32}\Intel');
      Candidate := FindDriverFileInDirectory(SearchPath, 'igfxicd64.dll');
      if Candidate = '' then
        Candidate := FindDriverFileInDirectory(SearchPath, 'igfxicd64.exe');
      if Candidate <> '' then
        Result := Candidate
      else
        Result := AddBackslash(SearchPath) + 'igfxicd64.dll';
    end;
  end;
end;

function GetNpuDriverPath: String;
var
  SearchPath: String;
  Candidate: String;
begin
  Result := '';

  if FileExists(ExpandConstant('{sys}\IntelVpuDriver.dll')) then
    Result := ExpandConstant('{sys}\IntelVpuDriver.dll')
  else
  begin
    if DirExists(ExpandConstant('{commonpf64}\Intel')) then
    begin
      SearchPath := ExpandConstant('{commonpf64}\Intel');
      Candidate := FindDriverFileInDirectory(SearchPath, 'IntelVpuDriver.dll');
      if Candidate <> '' then
        Result := Candidate
      else if DirExists(ExpandConstant('{commonpf64}\Intel\NPU')) then
        Result := ExpandConstant('{commonpf64}\Intel\NPU');
      Exit;
    end;

    if DirExists(ExpandConstant('{commonpf32}\Intel')) then
    begin
      SearchPath := ExpandConstant('{commonpf32}\Intel');
      Candidate := FindDriverFileInDirectory(SearchPath, 'IntelVpuDriver.dll');
      if Candidate <> '' then
        Result := Candidate
      else if DirExists(ExpandConstant('{commonpf32}\Intel\NPU')) then
        Result := ExpandConstant('{commonpf32}\Intel\NPU');
    end;
  end;
end;

function GetIntelCpuName: String;
var
  CpuName: String;
begin
  Result := '';

  if RegQueryStringValue(HKLM, 'HARDWARE\DESCRIPTION\System\CentralProcessor\0', 'ProcessorNameString', CpuName) then
    Result := CpuName;
end;

function GetPythonStatusLine: String;
begin
  if GetSelectedPythonVariant = 'system' then
  begin
    if DetectSystemPythonPath <> '' then
      Result := 'Python: system (' + DetectSystemPythonPath + ')'
    else
      Result := 'Python: system (not available)';
  end
  else if GetSelectedPythonVariant = 'embedded' then
    Result := 'Python: bundled runtime'
  else
    Result := 'Python: Not installing';
end;

function GetIntelDriverStatusLine(DriverName: String; DriverPresent: Boolean; DriverPath: String): String;
begin
  if DriverPresent then
  begin
    if DriverPath <> '' then
      Result := DriverName + ': present (' + DriverPath + ')'
    else
      Result := DriverName + ': present';
  end
  else
    Result := DriverName + ': not detected';
end;

function GetGpuDeviceNames: String;
var
  MarkerPath: String;
  ResultCode: Integer;
  Lines: TArrayOfString;
  Index: Integer;
begin
  Result := '';
  MarkerPath := ExpandConstant('{tmp}\ovms-gpu-devices.txt');
  DeleteFile(MarkerPath);

  Exec('powershell.exe',
    '-NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_VideoController | Where-Object { $_.Name -and (($_.Name -match ''Intel'') -or ($_.AdapterCompatibility -match ''Intel'')) } | Select-Object -ExpandProperty Name | Sort-Object -Unique | Set-Content -LiteralPath ''' + MarkerPath + ''' -Encoding ASCII"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

  if (ResultCode = 0) and LoadStringsFromFile(MarkerPath, Lines) then
  begin
    for Index := 0 to GetArrayLength(Lines) - 1 do
    begin
      if Result <> '' then
        Result := Result + #13#10;
      Result := Result + Lines[Index];
    end;
  end;

  DeleteFile(MarkerPath);
end;

function GetNpuDeviceNames: String;
var
  MarkerPath: String;
  ResultCode: Integer;
  Lines: TArrayOfString;
  Index: Integer;
begin
  Result := '';
  MarkerPath := ExpandConstant('{tmp}\ovms-npu-devices.txt');
  DeleteFile(MarkerPath);

  Exec('powershell.exe',
    '-NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_PnPEntity | Where-Object { (($_.Name -match ''Intel'' ) -or ($_.PNPDeviceID -match ''VEN_8086'')) -and ($_.Name -match ''NPU|Neural|AI Boost|VPU'' ) } | Select-Object -ExpandProperty Name | Sort-Object -Unique | Set-Content -LiteralPath ''' + MarkerPath + ''' -Encoding ASCII"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

  if (ResultCode = 0) and LoadStringsFromFile(MarkerPath, Lines) then
  begin
    for Index := 0 to GetArrayLength(Lines) - 1 do
    begin
      if Result <> '' then
        Result := Result + #13#10;
      Result := Result + Lines[Index];
    end;
  end;

  DeleteFile(MarkerPath);
end;

function GetFirstDeviceName(Names: String): String;
var
  SeparatorPosition: Integer;
begin
  Result := Trim(Names);
  SeparatorPosition := Pos(#13#10, Result);
  if SeparatorPosition > 0 then
    Result := Trim(Copy(Result, 1, SeparatorPosition - 1));
end;

function GetOvmsSettingsSummary: String;
var
  CpuName: String;
  GpuName: String;
  NpuName: String;
begin
  GpuName := GetFirstDeviceName(GetGpuDeviceNames);
  NpuName := GetFirstDeviceName(GetNpuDeviceNames);

  Result := GetPythonStatusLine;

  CpuName := GetIntelCpuName;
  if CpuName <> '' then
    Result := Result + #13#10 + 'Intel CPU: ' + CpuName
  else
    Result := Result + #13#10 + 'Intel CPU: not detected';

  if GpuName <> '' then
    Result := Result + #13#10 + 'Intel GPU: ' + GpuName
  else
    Result := Result + #13#10 + 'Intel GPU: not detected';

  Result := Result + #13#10 +
    GetIntelDriverStatusLine('Intel GPU driver', IsIntelGpuDriverAvailable, GetIntelGpuDriverPath);

  if IsIntelGpuDriverAvailable then
    Result := Result + #13#10 + ' - It is recommended to increase RAM allocation for GPU';

  if NpuName <> '' then
  begin
    Result := Result + #13#10 + 'Intel NPU: ' + NpuName;
    if IsNpuDriverAvailable then
      Result := Result + #13#10 + GetIntelDriverStatusLine('Intel NPU driver', True, GetNpuDriverPath)
    else
      Result := Result + #13#10 + 'Intel NPU driver: not detected';
  end
  else
  begin
    Result := Result + #13#10 + 'Intel NPU: not detected';
    Result := Result + #13#10 + 'Intel NPU driver: not detected';
  end;
end;

function IsOvmsServiceRunning: Boolean;
var
  ResultCode: Integer;
begin
  Result := False;
  Exec('powershell.exe',
    '-NoProfile -ExecutionPolicy Bypass -Command "$svc = Get-Service ovms -ErrorAction SilentlyContinue; if ($svc -and $svc.Status -eq ''Running'') { exit 0 } else { exit 1 }"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := ResultCode = 0;
end;

procedure UpdateOvmsServiceStatus;
begin
  if not Assigned(OvmsServiceStatusLabel) then
    Exit;

  if IsOvmsServiceRunning then
  begin
    OvmsServiceStatusLabel.Caption := 'OVMS Windows service: running';
    if Assigned(StopOvmsServiceButton) then
      StopOvmsServiceButton.Enabled := True;
  end
  else
  begin
    OvmsServiceStatusLabel.Caption := 'OVMS Windows service: stopped or not installed';
    if Assigned(StopOvmsServiceButton) then
      StopOvmsServiceButton.Enabled := False;
  end;
end;

procedure StopOvmsServiceButtonClick(Sender: TObject);
var
  ResultCode: Integer;
  Message: String;
begin
  if not IsOvmsServiceRunning then
  begin
    UpdateOvmsServiceStatus;
    Exit;
  end;

  Exec('powershell.exe',
    '-NoProfile -ExecutionPolicy Bypass -Command "Stop-Service ovms -Force -ErrorAction SilentlyContinue; $svc = Get-Service ovms -ErrorAction SilentlyContinue; if ($svc -and $svc.Status -eq ''Stopped'') { exit 0 } else { exit 1 }"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

  if ResultCode = 0 then
  begin
    Message := 'The OVMS Windows service was stopped successfully.';
    MsgBox(Message, mbInformation, MB_OK);
  end
  else
  begin
    Message := 'OVMS service could not be stopped. Please stop it manually and then continue with installation.';
    MsgBox(Message, mbError, MB_OK);
  end;

  UpdateOvmsServiceStatus;
end;

function GetDefaultModelRepositoryPath: String;
begin
  Result := 'C:\models';
end;

function GetSavedInstallOptionBool(Key: String): Boolean;
var
  FilePath: String;
  MarkerPath: String;
  ResultCode: Integer;
  Command: String;
begin
  Result := False;
  FilePath := ExpandConstant('{localappdata}\OVMS\install-options.json');
  if not FileExists(FilePath) then
    Exit;

  MarkerPath := ExpandConstant('{tmp}\ovms-install-option-' + Key + '.txt');
  DeleteFile(MarkerPath);

  Command :=
    '-NoProfile -ExecutionPolicy Bypass -Command "$p = ''' + FilePath + '''; if ((Get-Content -Raw $p) -match ''"'' + Key + ''"\s*:\s*true'') { ''true'' | Set-Content -Path ''' + MarkerPath + ''' }"';

  Exec('powershell.exe', Command, '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := FileExists(MarkerPath);
  DeleteFile(MarkerPath);
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
    '"npu":' + BoolToJson(WizardIsComponentSelected('npu')) + ',' +
    '"gpu":' + BoolToJson(WizardIsComponentSelected('gpu')) + ',' +
    '"opencv_mp":' + BoolToJson(WizardIsComponentSelected('opencv_mp')) + ',' +
    '"openvino_frontends":{' +
      '"onnx":' + BoolToJson(WizardIsComponentSelected('onnx_frontend')) + ',' +
      '"paddle":' + BoolToJson(WizardIsComponentSelected('paddle_frontend')) + ',' +
      '"tensorflow":' + BoolToJson(WizardIsComponentSelected('tensorflow_frontend')) + ',' +
      '"tensorflow_lite":' + BoolToJson(WizardIsComponentSelected('tensorflow_frontend')) +
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

function GetExternalRunningOvmsProcesses: String;
var
  MarkerPath: String;
  ResultCode: Integer;
begin
  Result := '';
  MarkerPath := ExpandConstant('{tmp}\ovms-external-processes.txt');
  DeleteFile(MarkerPath);

  Exec('powershell.exe',
    '-NoProfile -ExecutionPolicy Bypass -Command "if (Get-Process ovms -ErrorAction SilentlyContinue) { ''running'' | Set-Content -LiteralPath ''' + MarkerPath + ''' -Encoding ASCII }"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

  if FileExists(MarkerPath) then
  begin
    Result := MarkerPath;
  end;
  DeleteFile(MarkerPath);
end;

function HasRunningOvmsProcess: Boolean;
begin
  Result := GetExternalRunningOvmsProcesses <> '';
end;

function IsOvmsProcessAtDestinationRunning: Boolean;
var
  ResultCode: Integer;
  DestinationPath: String;
begin
  Result := False;

  DestinationPath := '';
  if Assigned(WizardForm) and Assigned(WizardForm.DirEdit) then
    DestinationPath := WizardForm.DirEdit.Text;
  if DestinationPath = '' then
    DestinationPath := ExistingInstallDir;
  if DestinationPath = '' then
    Exit;

  DestinationPath := AddBackslash(DestinationPath) + 'ovms.exe';
  Exec('powershell.exe',
    '-NoProfile -ExecutionPolicy Bypass -Command "$dest=[IO.Path]::GetFullPath(''' + DestinationPath + ''').TrimEnd(''\\''); Get-Process ovms -ErrorAction SilentlyContinue | Where-Object { try { [IO.Path]::GetFullPath($_.Path).TrimEnd(''\\'').Equals($dest,[StringComparison]::OrdinalIgnoreCase) } catch { $false } } | Select-Object -First 1 | ForEach-Object { exit 0 }; exit 1"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
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
  ExistingInstallVersion := GetInstalledOvmsVersion;

  if HasRunningOvmsProcess then
  begin
    if WizardSilent() then
    begin
      Result := True;
      exit;
    end;

    if MsgBox(
      'Other running ovms.exe processes were detected outside this installer destination.' + #13#10#13#10 +
      'This setup does not stop them. Stop external servers manually if they use conflicting ports.' + #13#10#13#10 +
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
  InitializeSetupLog;

  LegacyInstallMigrationPending := False;
  LegacyInstallDirToMigrate := '';
  LegacyUninstallStringToMigrate := '';

  if ExistingInstallDetected then
  begin
    if Length(ExistingInstallDir) > 0 then
    begin
      WizardForm.DirEdit.Text := ExistingInstallDir;
    end;
  end;

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
      'Installed version: ' + ExistingInstallVersion + #13#10 +
      'Installer version: {#OvmsVersion}' + #13#10#13#10 +
      'Choose how to continue:';
    Lbl.Height := ScaleY(80);

    if IsInstalledOvmsOlderThanCurrent then
    begin
      Lbl.Caption := Lbl.Caption + #13#10 +
        'This installation is older than the current OVMS package. Upgrade is recommended.';
    end;

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

  if WizardIsComponentSelected('python_system') or WizardIsComponentSelected('python_embedded') or WizardIsComponentSelected('python_none') then
  begin
    RuntimeSummaryPage := CreateCustomPage(wpSelectComponents,
      'Ovms runtime configuration',
      '');

    SettingsSummaryEdit := TNewMemo.Create(RuntimeSummaryPage);
    SettingsSummaryEdit.Parent := RuntimeSummaryPage.Surface;
    SettingsSummaryEdit.Left := 0;
    SettingsSummaryEdit.Top := 0;
    SettingsSummaryEdit.Width := RuntimeSummaryPage.SurfaceWidth;
    SettingsSummaryEdit.Height := ScaleY(110);
    SettingsSummaryEdit.ReadOnly := True;
    SettingsSummaryEdit.WordWrap := True;
    SettingsSummaryEdit.Text := GetOvmsSettingsSummary;

    OvmsServiceStatusLabel := TNewStaticText.Create(RuntimeSummaryPage);
    OvmsServiceStatusLabel.Parent := RuntimeSummaryPage.Surface;
    OvmsServiceStatusLabel.Left := 0;
    OvmsServiceStatusLabel.Top := SettingsSummaryEdit.Top + SettingsSummaryEdit.Height + ScaleY(12);
    OvmsServiceStatusLabel.Width := RuntimeSummaryPage.SurfaceWidth;
    OvmsServiceStatusLabel.AutoSize := False;
    OvmsServiceStatusLabel.WordWrap := True;
    OvmsServiceStatusLabel.Caption := 'OVMS Windows service: checking...';
    OvmsServiceStatusLabel.Height := ScaleY(20);

    StopOvmsServiceButton := TNewButton.Create(RuntimeSummaryPage);
    StopOvmsServiceButton.Parent := RuntimeSummaryPage.Surface;
    StopOvmsServiceButton.Left := 0;
    StopOvmsServiceButton.Top := OvmsServiceStatusLabel.Top + OvmsServiceStatusLabel.Height + ScaleY(8);
    StopOvmsServiceButton.Width := ScaleX(150);
    StopOvmsServiceButton.Height := ScaleY(23);
    StopOvmsServiceButton.Caption := 'Stop OVMS service';
    StopOvmsServiceButton.OnClick := @StopOvmsServiceButtonClick;
    UpdateOvmsServiceStatus;

    OvmsProcessStatusLabel := TNewStaticText.Create(RuntimeSummaryPage);
    OvmsProcessStatusLabel.Parent := RuntimeSummaryPage.Surface;
    OvmsProcessStatusLabel.Left := 0;
    OvmsProcessStatusLabel.Top := StopOvmsServiceButton.Top + StopOvmsServiceButton.Height + ScaleY(8);
    OvmsProcessStatusLabel.Width := RuntimeSummaryPage.SurfaceWidth;
    OvmsProcessStatusLabel.AutoSize := False;
    OvmsProcessStatusLabel.WordWrap := True;
    OvmsProcessStatusLabel.Caption := 'OVMS process from this installation: checking...';
    OvmsProcessStatusLabel.Height := ScaleY(20);

    StopOvmsProcessButton := TNewButton.Create(RuntimeSummaryPage);
    StopOvmsProcessButton.Parent := RuntimeSummaryPage.Surface;
    StopOvmsProcessButton.Left := 0;
    StopOvmsProcessButton.Top := OvmsProcessStatusLabel.Top + OvmsProcessStatusLabel.Height + ScaleY(8);
    StopOvmsProcessButton.Width := ScaleX(150);
    StopOvmsProcessButton.Height := ScaleY(23);
    StopOvmsProcessButton.Caption := 'Stop OVMS process';
    StopOvmsProcessButton.OnClick := @StopOvmsProcessButtonClick;
    UpdateOvmsProcessStatus;
  end;

  // Keep the install-options page available regardless of the selected Python
  // flavor so the model-download shortcut toggles remain visible to the user.
  AdditionalTasksPage := CreateCustomPage(wpSelectTasks,
    'Additional tasks',
    'OVMS install options');

  AddOvmsToPathCheck := TNewCheckBox.Create(AdditionalTasksPage);
  AddOvmsToPathCheck.Parent := AdditionalTasksPage.Surface;
  AddOvmsToPathCheck.Left := 0;
  AddOvmsToPathCheck.Top := 0;
  AddOvmsToPathCheck.Width := AdditionalTasksPage.SurfaceWidth;
  AddOvmsToPathCheck.Checked := True;
  AddOvmsToPathCheck.Caption := 'Add OVMS to PATH';

  Lbl := TNewStaticText.Create(AdditionalTasksPage);
  Lbl.Parent := AdditionalTasksPage.Surface;
  Lbl.Left := 0;
  Lbl.Top := AddOvmsToPathCheck.Top + AddOvmsToPathCheck.Height + ScaleY(8);
  Lbl.Width := AdditionalTasksPage.SurfaceWidth;
  Lbl.AutoSize := False;
  Lbl.WordWrap := True;
  Lbl.Caption := 'Model repository path used for OVMS_MODEL_REPOSITORY_PATH';
  Lbl.Height := ScaleY(20);

  ModelRepositoryPathEdit := TNewEdit.Create(AdditionalTasksPage);
  ModelRepositoryPathEdit.Parent := AdditionalTasksPage.Surface;
  ModelRepositoryPathEdit.Left := 0;
  ModelRepositoryPathEdit.Top := Lbl.Top + Lbl.Height + ScaleY(4);
  ModelRepositoryPathEdit.Width := AdditionalTasksPage.SurfaceWidth;
  ModelRepositoryPathEdit.Text := GetDefaultModelRepositoryPath;

  Lbl := TNewStaticText.Create(AdditionalTasksPage);
  Lbl.Parent := AdditionalTasksPage.Surface;
  Lbl.Left := 0;
  Lbl.Top := ModelRepositoryPathEdit.Top + ModelRepositoryPathEdit.Height + ScaleY(8);
  Lbl.Width := AdditionalTasksPage.SurfaceWidth;
  Lbl.AutoSize := False;
  Lbl.WordWrap := True;
  Lbl.Caption := 'Ovms start mode';
  Lbl.Height := ScaleY(20);

  OvmsServiceModeRadio := TNewRadioButton.Create(AdditionalTasksPage);
  OvmsServiceModeRadio.Parent := AdditionalTasksPage.Surface;
  OvmsServiceModeRadio.Left := 0;
  OvmsServiceModeRadio.Top := Lbl.Top + Lbl.Height + ScaleY(4);
  OvmsServiceModeRadio.Width := AdditionalTasksPage.SurfaceWidth;
  OvmsServiceModeRadio.Height := ScaleY(17);
  OvmsServiceModeRadio.Caption := 'Automatic - start ovms service';
  OvmsServiceModeRadio.Checked := False;

  Lbl := TNewStaticText.Create(AdditionalTasksPage);
  Lbl.Parent := AdditionalTasksPage.Surface;
  Lbl.Left := ScaleX(18);
  Lbl.Top := OvmsServiceModeRadio.Top + OvmsServiceModeRadio.Height + ScaleY(2);
  Lbl.Width := AdditionalTasksPage.SurfaceWidth - ScaleX(18);
  Lbl.AutoSize := False;
  Lbl.WordWrap := True;
  Lbl.Caption := 'Will stop and reinstall the existing OVMS service.';
  Lbl.Height := ScaleY(26);

  OvmsTerminalModeRadio := TNewRadioButton.Create(AdditionalTasksPage);
  OvmsTerminalModeRadio.Parent := AdditionalTasksPage.Surface;
  OvmsTerminalModeRadio.Left := 0;
  OvmsTerminalModeRadio.Top := Lbl.Top + Lbl.Height + ScaleY(8);
  OvmsTerminalModeRadio.Width := AdditionalTasksPage.SurfaceWidth;
  OvmsTerminalModeRadio.Height := ScaleY(17);
  OvmsTerminalModeRadio.Caption := 'Manual - start ovms in terminal';
  OvmsTerminalModeRadio.Checked := True;

  Lbl := TNewStaticText.Create(AdditionalTasksPage);
  Lbl.Parent := AdditionalTasksPage.Surface;
  Lbl.Left := ScaleX(18);
  Lbl.Top := OvmsTerminalModeRadio.Top + OvmsTerminalModeRadio.Height + ScaleY(2);
  Lbl.Width := AdditionalTasksPage.SurfaceWidth - ScaleX(18);
  Lbl.AutoSize := False;
  Lbl.WordWrap := True;
  Lbl.Caption := 'Run "Start OVMS Server" from Start Windows Menu.';
  Lbl.Height := ScaleY(26);
end;

function GetSelectedModelRepositoryPath: String;
begin
  Result := '';
  if Assigned(ModelRepositoryPathEdit) then
    Result := Trim(ModelRepositoryPathEdit.Text);

  if Result = '' then
    Result := GetDefaultModelRepositoryPath;
end;

procedure PersistModelRepositoryPath;
var
  ModelRepositoryPath: String;
  ResultCode: Integer;
  Message: String;
begin
  if not Assigned(ModelRepositoryPathEdit) then
    Exit;

  ModelRepositoryPath := GetSelectedModelRepositoryPath;

  ResultCode := 0;
  Exec('cmd.exe',
    '/C setx OVMS_MODEL_REPOSITORY_PATH "' + ModelRepositoryPath + '"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

  if ResultCode <> 0 then
  begin
    Message :=
      'OVMS could not persist the model repository path to the user environment.' + #13#10 +
      'Path: ' + ModelRepositoryPath + #13#10 +
      'The installation will continue, but you may need to set OVMS_MODEL_REPOSITORY_PATH manually.';
    MsgBox(Message, mbError, MB_OK);
  end;
end;

function GetOvmsDataDirForRunMode(RunMode: String): String;
begin
  if CompareText(RunMode, 'service') = 0 then
    Result := ExpandConstant('{commonappdata}\OVMS')
  else
    Result := ExpandConstant('{localappdata}\OVMS');
end;

function RunPythonConfigurationForSelectedVariant: Boolean;
var
  Variant: String;
  RunMode: String;
  ModelRepositoryPath: String;
  DataDir: String;
  ResultCode: Integer;
  LogPath: String;
  Message: String;
  ScriptPath: String;
begin
  Result := True;

  Variant := GetSelectedPythonVariant;
  if (Variant = '') then
    exit;

  RunMode := 'user-login';
  if Assigned(OvmsServiceModeRadio) and OvmsServiceModeRadio.Checked then
    RunMode := 'service';

  DataDir := GetOvmsDataDirForRunMode(RunMode);

  ModelRepositoryPath := GetSelectedModelRepositoryPath;

  ScriptPath := ExpandConstant('{app}\scripts\configure-ovms.ps1');
  if not FileExists(ScriptPath) then
  begin
    MsgBox('The OVMS Python configuration helper script was not installed with the package. Expected path:' + #13#10 + ScriptPath + #13#10#13#10 + 'Rebuild the installer or reinstall the package with the full scripts bundle.', mbError, MB_OK);
    Result := False;
    exit;
  end;

  LogPath := GetSetupLogPath;

  Exec('cmd.exe',
    '/C powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + ScriptPath + '" -InstallDir "' + ExpandConstant('{app}') + '" -DataDir "' + DataDir + '" -ModelRepositoryPath "' + ModelRepositoryPath + '" -PythonVariant "' + Variant + '" -RunMode "' + RunMode + '" -SetupLogPath "' + LogPath + '"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

  if ResultCode = 0 then
  begin
    if Variant = 'system' then
      Message := 'System Python configuration succeeded.' + #13#10 + 'The detected Python 3.14 environment is ready for OVMS.'
    else if Variant = 'none' then
      Message := 'OVMS runtime configuration succeeded.' + #13#10 + 'Python support is disabled for this installation.'
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

procedure CurPageChanged(CurPageID: Integer);
begin
  if Assigned(RuntimeSummaryPage) and (CurPageID = RuntimeSummaryPage.ID) then
  begin
    if Assigned(SettingsSummaryEdit) then
      SettingsSummaryEdit.Text := GetOvmsSettingsSummary;
    UpdateOvmsServiceStatus;
    UpdateOvmsProcessStatus;
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

  if (CurPageID = RuntimeSummaryPage.ID) then
  begin
    if IsOvmsServiceRunning then
    begin
      MsgBox('The OVMS Windows service is still running. Stop it before continuing with the installation.', mbError, MB_OK);
      Result := False;
      Exit;
    end;

    if IsOvmsProcessAtDestinationRunning then
    begin
      MsgBox('An OVMS process from this installation is still running. Stop it before continuing.', mbError, MB_OK);
      Result := False;
      Exit;
    end;
  end;

  if (CurPageID = wpSelectDir) and ExistingInstallDetected then
  begin
    if (Length(ExistingInstallDir) > 0) and (CompareText(ExistingInstallDir, WizardForm.DirEdit.Text) <> 0) then
    begin
      // Do not delete the previous installation at directory-selection time.
      // Keep the old install until the replacement is fully installed.
      LegacyInstallMigrationPending := True;
      LegacyInstallDirToMigrate := ExistingInstallDir;
      LegacyUninstallStringToMigrate := ExistingUninstallString;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ServiceScriptPath: String;
  ConfigureScriptPath: String;
  RepairScriptPath: String;
  PythonVariant: String;
  ResultCode: Integer;
  RecoveryResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    SaveStringToFile(GetSetupLogPath, '[SETUP] Starting post-install stage markers.' + #13#10, True);
    WriteOptionalInstallSelection;

    if ExistingInstallDetected and Assigned(RbRepair) and RbRepair.Checked then
    begin
      SaveStringToFile(GetSetupLogPath, '[SETUP] Stage: repair existing installation' + #13#10, True);
      RepairScriptPath := ExpandConstant('{app}\scripts\repair-package.ps1');
      if FileExists(RepairScriptPath) then
      begin
        Exec('cmd.exe',
          '/C powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + RepairScriptPath + '" -InstallDir "' + ExpandConstant('{app}') + '" -DataDir "' + ExpandConstant('{localappdata}\OVMS') + '" -PackageSource "' + ExpandConstant('{app}') + '" >> "' + GetSetupLogPath + '" 2>&1',
          '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
        if ResultCode <> 0 then
        begin
          SaveStringToFile(GetSetupLogPath, '[SETUP] Repair cleanup failed with exit code ' + IntToStr(ResultCode) + #13#10, True);
          MsgBox('Repair cleanup failed for the existing installation. Some deselected components may still be present.' + #13#10 + 'See the setup log for details: ' + GetSetupLogPath, mbError, MB_OK);
        end;
      end;
    end;

    PersistModelRepositoryPath;

    ResultCode := 0;

    if Assigned(AddOvmsToPathCheck) and AddOvmsToPathCheck.Checked then
    begin
      SaveStringToFile(GetSetupLogPath, '[SETUP] Stage: update PATH environment variable' + #13#10, True);
      Exec('cmd.exe',
        '/C powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\scripts\set-path.ps1') + '" -InstallDir "' + ExpandConstant('{app}') + '" -Action Add -SetupLogPath "' + GetSetupLogPath + '"',
        '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
      if ResultCode <> 0 then
      begin
        MsgBox('OVMS PATH configuration failed. See the setup log for details: ' + GetSetupLogPath, mbError, MB_OK);
      end;
    end;

    if (WizardIsComponentSelected('python_system') or WizardIsComponentSelected('python_embedded') or WizardIsComponentSelected('python_none')) then
    begin
      SaveStringToFile(GetSetupLogPath, '[SETUP] Stage: configure Python runtime' + #13#10, True);
      RunPythonConfigurationForSelectedVariant;
    end;

    if Assigned(OvmsServiceModeRadio) and OvmsServiceModeRadio.Checked then
    begin
      SaveStringToFile(GetSetupLogPath, '[SETUP] Stage: configure OVMS Windows service' + #13#10, True);
      ServiceScriptPath := ExpandConstant('{app}\scripts\install-service.ps1');
      if FileExists(ServiceScriptPath) then
      begin
        Exec('cmd.exe',
          '/C powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + ServiceScriptPath + '" -InstallDir "' + ExpandConstant('{app}') + '" -DataDir "' + ExpandConstant('{commonappdata}\OVMS') + '" -SetupLogPath "' + GetSetupLogPath + '"',
          '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
        if ResultCode <> 0 then
        begin
          SaveStringToFile(GetSetupLogPath, '[SETUP] Service setup failed. Attempting run-mode recovery to user-login.' + #13#10, True);

          PythonVariant := GetSelectedPythonVariant;
          if PythonVariant = '' then
            PythonVariant := 'none';

          ConfigureScriptPath := ExpandConstant('{app}\scripts\configure-ovms.ps1');
          RecoveryResultCode := 1;
          if FileExists(ConfigureScriptPath) then
          begin
            Exec('cmd.exe',
              '/C powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + ConfigureScriptPath + '" -InstallDir "' + ExpandConstant('{app}') + '" -DataDir "' + ExpandConstant('{localappdata}\OVMS') + '" -ModelRepositoryPath "' + GetSelectedModelRepositoryPath + '" -PythonVariant "' + PythonVariant + '" -RunMode "user-login" -SetupLogPath "' + GetSetupLogPath + '"',
              '', SW_HIDE, ewWaitUntilTerminated, RecoveryResultCode);
          end;

          if RecoveryResultCode = 0 then
          begin
            MsgBox('OVMS service mode was selected, but elevated service setup failed.' + #13#10 +
              'The installer recovered run mode to Manual - start in terminal.' + #13#10 +
              'Details were logged to:' + #13#10 + GetSetupLogPath,
              mbError, MB_OK);
          end
          else
          begin
            MsgBox('OVMS service mode was selected, but elevated service setup failed.' + #13#10 +
              'Run-mode recovery to Manual mode also failed, so settings may still indicate service mode.' + #13#10 +
              'Please run configure-ovms.ps1 manually with -RunMode user-login.' + #13#10 +
              'Details were logged to:' + #13#10 + GetSetupLogPath,
              mbError, MB_OK);
          end;
        end;
      end;
    end;

    if LegacyInstallMigrationPending and (LegacyInstallDirToMigrate <> '') and
       (CompareText(LegacyInstallDirToMigrate, ExpandConstant('{app}')) <> 0) then
    begin
      SaveStringToFile(GetSetupLogPath,
        '[SETUP] Legacy install preserved for controlled migration: ' + LegacyInstallDirToMigrate + #13#10, True);

      if LegacyUninstallStringToMigrate <> '' then
      begin
        MsgBox(
          'A previous OVMS installation was preserved at:' + #13#10 +
          LegacyInstallDirToMigrate + #13#10#13#10 +
          'The new installation completed successfully.' + #13#10 +
          'To retire only installer-managed files from the old location, run its uninstaller:' + #13#10 +
          LegacyUninstallStringToMigrate,
          mbInformation, MB_OK);
      end
      else
      begin
        MsgBox(
          'A previous OVMS installation was preserved at:' + #13#10 +
          LegacyInstallDirToMigrate + #13#10#13#10 +
          'The new installation completed successfully.' + #13#10 +
          'No registered old uninstaller was found. If you want to retire the old installation, remove only installer-managed files manually.',
          mbInformation, MB_OK);
      end;
    end;

    SaveStringToFile(GetSetupLogPath, '[SETUP] ==================== OVMS Setup completed successfully ====================' + #13#10, True);
  end;
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

procedure StopOvmsProcessButtonClick(Sender: TObject);
var
  ResultCode: Integer;
  DestinationPath: String;
begin
  DestinationPath := ExpandConstant('{app}') + '\ovms.exe';
  Exec('powershell.exe',
    '-NoProfile -ExecutionPolicy Bypass -Command "$dest=[IO.Path]::GetFullPath(''' + DestinationPath + ''').TrimEnd(''\\''); Get-Process ovms -ErrorAction SilentlyContinue | Where-Object { try { [IO.Path]::GetFullPath($_.Path).TrimEnd(''\\'').Equals($dest,[StringComparison]::OrdinalIgnoreCase) } catch { $false } } | Stop-Process -Force -ErrorAction SilentlyContinue"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  UpdateOvmsProcessStatus;
  MsgBox('Matching OVMS processes were stopped.', mbInformation, MB_OK);
end;

procedure UpdateOvmsProcessStatus;
begin
  if not Assigned(OvmsProcessStatusLabel) then
    Exit;

  if IsOvmsProcessAtDestinationRunning then
  begin
    OvmsProcessStatusLabel.Caption := 'OVMS process from this installation: running';
    if Assigned(StopOvmsProcessButton) then
      StopOvmsProcessButton.Enabled := True;
  end
  else
  begin
    OvmsProcessStatusLabel.Caption := 'OVMS process from this installation: stopped';
    if Assigned(StopOvmsProcessButton) then
      StopOvmsProcessButton.Enabled := False;
  end;
end;

