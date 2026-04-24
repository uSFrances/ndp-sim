[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$ramulatorRoot = Join-Path $RepoRoot "third_party\ramulator2"
$cmake = "C:\Program Files\CMake\bin\cmake.exe"
$ninja = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$llvmRc = "C:\Program Files\LLVM\bin\llvm-rc.exe"

if (-not (Test-Path $ramulatorRoot)) {
    git clone https://github.com/CMU-SAFARI/ramulator2.git $ramulatorRoot
}

if (-not (Test-Path (Join-Path $ramulatorRoot "ext\yaml-cpp\.git"))) {
    git clone --depth 1 --branch yaml-cpp-0.7.0 https://github.com/jbeder/yaml-cpp.git (Join-Path $ramulatorRoot "ext\yaml-cpp")
}
if (-not (Test-Path (Join-Path $ramulatorRoot "ext\spdlog\.git"))) {
    git clone --depth 1 --branch v1.11.0 https://github.com/gabime/spdlog.git (Join-Path $ramulatorRoot "ext\spdlog")
}
if (-not (Test-Path (Join-Path $ramulatorRoot "ext\argparse\.git"))) {
    git clone --depth 1 --branch v2.9 https://github.com/p-ranav/argparse.git (Join-Path $ramulatorRoot "ext\argparse")
}

if ($SkipBuild) {
    Write-Host "Dependencies are ready under $ramulatorRoot"
    exit 0
}

$cmdFile = Join-Path $env:TEMP "build_ramulator_windows.cmd"
@"
@echo off
call "$vcvars"
if exist "$ramulatorRoot\build_msvc" rmdir /s /q "$ramulatorRoot\build_msvc"
"$cmake" -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5 -S "$ramulatorRoot" -B "$ramulatorRoot\build_msvc" -G Ninja -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_RC_COMPILER="$llvmRc" -DCMAKE_MAKE_PROGRAM="$ninja"
if errorlevel 1 exit /b 1
"$cmake" --build "$ramulatorRoot\build_msvc" --parallel 4
"@ | Set-Content -Path $cmdFile -Encoding ascii

cmd.exe /c $cmdFile
