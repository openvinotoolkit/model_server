<#
ИНСТРУКЦИЯ ПО АДАПТАЦИИ СКРИПТОВ ЗАПУСКА (LFM2.5)
1. package — каталог RC4 extracted-py_off\ovms: DLL OpenVINO/GenAI и MANIFEST.
   exe — отдельно Task-3 ovms.exe с lfm2 parser (не подменяйте DLL пакета).
   modelPath — OpenVINO IR LFM2.5-8B; modelName — имя в API/карточке OMP.
2. WEIGHTS_PATH не используется: OFFLOAD_RATIO=0 (Pack/offload отложены).
   Веса — openvino_model.bin (не openvino_language_model.bin).
3. servingDir — рабочая копия IR: .bin через hardlink, остальное копируется,
   chat_template.jinja берётся из docs/lfm/templates override. Скачанный
   C:\llm\models\* не изменяется. Каталоги configs/logs создаются сами.
4. Pipeline LM (не VLM_CB). tool_parser и reasoning_parser — lfm2.
   Не переносите чужой chat_template. MTP/speculative не включается.
5. Профиль как в GPU smoke Task 5: 4 слота, batch 1024, cache_size=1 GiB,
   prefix caching выключен. cache_size — пул KV в гигабайтах, не вся RAM.
6. Порт 8137 общий с Qwen/Gemma/Ornith: запускать по очереди либо сменить порт.
   Скрипт чужие процессы не останавливает. -ConfigureOnly только готовит serving.
7. Файл — UTF-8 с BOM для PS 5.1. Это запуск сервера, не автотест.
#>
param(
    [switch]$ConfigureOnly,
    [ValidateRange(1,65535)][int]$Port = 8137
)
$ErrorActionPreference = 'Stop'
$package = 'C:\gm\qwenmonster\releases\gemmamonster-2026.4-rc4\extracted-py_off\ovms'
$exe = 'C:\gm\lfm-2026.4\serving\ovms.exe'
$expectedExeSha256 = 'f484927cc5bacbb979f9edb11df85c494b6cdf369d7c179b29c48d6807dfc877'
$modelPath = 'C:\llm\models\LFM2.5-8B-A1B-int4-ov'
$modelName = 'lfm25-8b-a1b-int4'
$templatePath = 'C:\git\gemmamonster-lfm-2026.4\docs\lfm\templates\LFM2.5-8B-A1B-int4-ov.jinja'
$servingDir = 'C:\gm\lfm-2026.4\serving-models\LFM2.5-8B-A1B-int4-ov'

foreach ($requiredFile in @(
    (Join-Path $modelPath 'openvino_model.bin'),
    (Join-Path $modelPath 'config.json'),
    (Join-Path $modelPath 'openvino_model.xml'),
    (Join-Path $modelPath 'openvino_tokenizer.xml'),
    (Join-Path $modelPath 'openvino_tokenizer.bin'),
    $templatePath,
    $exe
)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw ('Required file missing: ' + $requiredFile)
    }
}

$exeStream = [IO.File]::OpenRead($exe)
$exeHash = [Security.Cryptography.SHA256]::Create()
try {
    $actualExe = [BitConverter]::ToString($exeHash.ComputeHash($exeStream)).Replace('-', '').ToLower()
} finally {
    $exeStream.Dispose(); $exeHash.Dispose()
}
if ($actualExe -ne $expectedExeSha256) {
    throw ('Serving ovms.exe hash drift: expected ' + $expectedExeSha256 + ' got ' + $actualExe)
}

# Baseline DLLs: gemmamonster-2026.4-rc4; serving EXE: Task 3 LFM parser rebuild.
$manifest = Get-Content (Join-Path $package 'MANIFEST.json') -Raw | ConvertFrom-Json
foreach ($row in $manifest.files) {
    if ($row.path -match '\.dll$') {
        $file = Join-Path $package $row.path
        $stream = [IO.File]::OpenRead($file)
        $hash = [Security.Cryptography.SHA256]::Create()
        try {
            $actual = [BitConverter]::ToString($hash.ComputeHash($stream)).Replace('-', '').ToLower()
        } finally {
            $stream.Dispose(); $hash.Dispose()
        }
        if ($actual -ne $row.sha256) { throw ('Release file changed: ' + $file) }
    }
}

if (Test-Path -LiteralPath $servingDir) {
    Remove-Item -LiteralPath $servingDir -Recurse -Force
}
[void][IO.Directory]::CreateDirectory($servingDir)
Get-ChildItem -LiteralPath $modelPath -File | ForEach-Object {
    $target = Join-Path $servingDir $_.Name
    if ($_.Extension -eq '.bin') {
        New-Item -ItemType HardLink -Path $target -Target $_.FullName | Out-Null
    } else {
        Copy-Item -LiteralPath $_.FullName -Destination $target -Force
    }
}
Copy-Item -LiteralPath $templatePath -Destination (Join-Path $servingDir 'chat_template.jinja') -Force
Write-Host ('Serving model: ' + $servingDir)

$configDir = Join-Path $PSScriptRoot 'configs\lfm25'
[void][IO.Directory]::CreateDirectory($configDir)
$commandPath = Join-Path $configDir 'command.json'
$gpuPluginConfig = '{"OFFLOAD_RATIO":0}'
if ($PSVersionTable.PSVersion.Major -le 5 -or $PSNativeCommandArgumentPassing -eq 'Legacy') {
    $gpuPluginConfigEscaped = $gpuPluginConfig.Replace('"', '\"')
} else {
    $gpuPluginConfigEscaped = $gpuPluginConfig
}
$command = @(
    $exe,
    '--rest_port', "$Port",
    '--rest_bind_address', '127.0.0.1',
    '--model_path', $servingDir,
    '--model_name', $modelName,
    '--task', 'text_generation',
    '--pipeline_type', 'LM',
    '--target_device', 'GPU',
    '--plugin_config', $gpuPluginConfig,
    '--tool_parser', 'lfm2',
    '--reasoning_parser', 'lfm2',
    '--enable_tool_guided_generation', 'true',
    '--max_num_seqs', '4',
    '--max_num_batched_tokens', '1024',
    '--cache_size', '1',
    '--enable_prefix_caching', 'false',
    '--log_level', 'INFO'
)
$utf8 = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText($commandPath, ($command | ConvertTo-Json -Compress), $utf8)
Write-Host ('Command receipt: ' + $commandPath)
if ($ConfigureOnly) { exit 0 }

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw ('Port ' + $Port + ' is occupied. Stop the current model before starting LFM2.5; no processes were stopped.')
}

$env:PYTHONHOME = $null
$env:PYTHONPATH = $null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PATH = "$package;$env:SystemRoot\System32;$env:SystemRoot"
$logDir = Join-Path $PSScriptRoot 'logs'
[void][IO.Directory]::CreateDirectory($logDir)
$logPath = Join-Path $logDir ($modelName + '-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '.log')
Write-Host ('Model: ' + $modelName + ' -> ' + $servingDir)
Write-Host ('OVMS log: ' + $logPath)
& $exe `
    --rest_port $Port `
    --rest_bind_address 127.0.0.1 `
    --model_path $servingDir `
    --model_name $modelName `
    --task text_generation `
    --pipeline_type LM `
    --target_device GPU `
    --plugin_config $gpuPluginConfigEscaped `
    --tool_parser lfm2 `
    --reasoning_parser lfm2 `
    --enable_tool_guided_generation true `
    --max_num_seqs 4 `
    --max_num_batched_tokens 1024 `
    --cache_size 1 `
    --enable_prefix_caching false `
    --log_level INFO `
    --log_path $logPath
exit $LASTEXITCODE