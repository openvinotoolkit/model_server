<#
ИНСТРУКЦИЯ ПО АДАПТАЦИИ СКРИПТОВ ЗАПУСКА (LFM2-24B, long)
1. package — каталог RC4 extracted-py_off\ovms: DLL OpenVINO/GenAI и MANIFEST.
   exe — candidate-final ovms.exe с LFM retainer/policy (не подменяйте DLL пакета).
   modelPath — OpenVINO IR LFM2-24B-A2B; modelName — имя в API/карточке OMP.
2. WEIGHTS_PATH не используется: OFFLOAD_RATIO=0 (Pack/offload отложены).
   Веса — openvino_model.bin. Нужен запас host RAM (admission ~ weights+2.5 GiB).
3. servingDir — рабочая копия IR с override chat_template (экспорт без renderer).
   Скачанный C:\llm\models\* не изменяется. configs/logs создаются сами.
4. Pipeline LM. tool_parser — lfm2. reasoning_parser для LFM2-24B не задаём.
   Штатный parallel_tool_calls=true на стороне API (поле можно не слать).
5. Профиль ДЛИННОЙ генерации (не smoke Task5): 2 слота, batch 4096,
   cache_size=0 (динамический KV), prefix caching ON, interval 256,
   dynamic_split_fuse ON, rest_workers=1. cache_size=0 не ограничивает всю RAM
   процесса — следите за свободной памятью на Arc 140V / UMA.
   После смены scheduler параметров считайте это новой конфигурацией.
6. Порт 8137 общий с Qwen/Gemma/Ornith: по очереди либо сменить порт.
   Скрипт чужие процессы не останавливает. -ConfigureOnly только готовит serving.
7. Файл — UTF-8 с BOM для PS 5.1. Это запуск сервера, не автотест.
#>
param(
    [switch]$ConfigureOnly,
    [ValidateRange(1,65535)][int]$Port = 8137
)
$ErrorActionPreference = 'Stop'
$package = 'C:\gm\qwenmonster\releases\gemmamonster-2026.4-rc4\extracted-py_off\ovms'
$exe = 'C:\gm\lfm-2026.4\tool-use-20261027\candidate-final\build-py_off\bin\ovms.exe'
$expectedExeSha256 = '95fb62b5df7d68a880457378a4d3503cc606b5f06a82c23e70010d96358272ba'
$modelPath = 'C:\llm\models\LFM2-24B-A2B-int4-ov'
$modelName = 'lfm2-24b-a2b-int4'
$templatePath = 'C:\git\gemmamonster-lfm-2026.4\docs\lfm\templates\LFM2-24B-A2B-int4-ov.jinja'
$servingDir = 'C:\gm\lfm-2026.4\serving-models\LFM2-24B-A2B-int4-ov'

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

# Baseline DLLs: gemmamonster-2026.4-rc4; EXE: tool-use candidate-final.
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

$configDir = Join-Path $PSScriptRoot 'configs\lfm2-24b-long'
[void][IO.Directory]::CreateDirectory($configDir)
$commandPath = Join-Path $configDir 'command.json'
$gpuPluginConfig = '{"OFFLOAD_RATIO":0}'
if ($PSVersionTable.PSVersion.Major -le 5 -or $PSNativeCommandArgumentPassing -eq 'Legacy') {
    $gpuPluginConfigEscaped = $gpuPluginConfig.Replace('"', '\"')
} else {
    $gpuPluginConfigEscaped = $gpuPluginConfig
}
# Long profile: fewer concurrent seqs, larger batch, dynamic KV (cache_size=0).
$command = @(
    $exe,
    '--rest_port', "$Port",
    '--rest_bind_address', '127.0.0.1',
    '--rest_workers', '1',
    '--model_path', $servingDir,
    '--model_name', $modelName,
    '--task', 'text_generation',
    '--pipeline_type', 'LM',
    '--target_device', 'GPU',
    '--plugin_config', $gpuPluginConfig,
    '--tool_parser', 'lfm2',
    '--enable_tool_guided_generation', 'true',
    '--max_num_seqs', '2',
    '--max_num_batched_tokens', '4096',
    '--cache_size', '0',
    '--cache_interval_multiplier', '256',
    '--enable_prefix_caching', 'true',
    '--dynamic_split_fuse', 'true',
    '--log_level', 'INFO'
)
$utf8 = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText($commandPath, ($command | ConvertTo-Json -Compress), $utf8)
Write-Host ('Command receipt: ' + $commandPath)
Write-Host 'Profile: LFM2-24B long (seqs=2 batch=4096 cache_size=0 prefix=on interval=256)'
if ($ConfigureOnly) { exit 0 }

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw ('Port ' + $Port + ' is occupied. Stop the current model before starting LFM2-24B; no processes were stopped.')
}

$env:PYTHONHOME = $null
$env:PYTHONPATH = $null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PATH = "$package;$env:SystemRoot\System32;$env:SystemRoot"
$logDir = Join-Path $PSScriptRoot 'logs'
[void][IO.Directory]::CreateDirectory($logDir)
$logPath = Join-Path $logDir ($modelName + '-long-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '.log')
Write-Host ('Model: ' + $modelName + ' -> ' + $servingDir)
Write-Host ('OVMS log: ' + $logPath)
& $exe `
    --rest_port $Port `
    --rest_bind_address 127.0.0.1 `
    --rest_workers 1 `
    --model_path $servingDir `
    --model_name $modelName `
    --task text_generation `
    --pipeline_type LM `
    --target_device GPU `
    --plugin_config $gpuPluginConfigEscaped `
    --tool_parser lfm2 `
    --enable_tool_guided_generation true `
    --max_num_seqs 2 `
    --max_num_batched_tokens 4096 `
    --cache_size 0 `
    --cache_interval_multiplier 256 `
    --enable_prefix_caching true `
    --dynamic_split_fuse true `
    --log_level INFO `
    --log_path $logPath
exit $LASTEXITCODE