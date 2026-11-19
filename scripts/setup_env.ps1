# =====================================================
# chinese-sentiment-bert 一键环境配置脚本（Windows）
#
# 用法：右键"使用 PowerShell 运行"，或：
#   powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
#
# 功能：
#   1. 在项目根目录创建 .venv 虚拟环境（已存在则复用）
#   2. 检测 NVIDIA GPU：有则安装 CUDA 版 PyTorch（cu124），无则安装 CPU 版
#   3. 安装 requirements.txt 中的其余依赖（使用清华镜像加速）
#   4. 校验关键库可用性
# =====================================================

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " chinese-sentiment-bert 环境配置" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# ---------- 1. 虚拟环境 ----------
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[1/4] 创建虚拟环境 .venv ..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Host "创建虚拟环境失败，请确认已安装 Python 3.10+ 并加入 PATH。" -ForegroundColor Red; exit 1 }
} else {
    Write-Host "[1/4] 已存在 .venv，复用。" -ForegroundColor Yellow
}

$Py = ".venv\Scripts\python.exe"
$Pip = ".venv\Scripts\pip.exe"

# ---------- 2. 检测 GPU ----------
Write-Host "[2/4] 检测 GPU ..." -ForegroundColor Yellow
$HasGPU = $false
try {
    $nvidia = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $nvidia) {
        $HasGPU = $true
        Write-Host "      检测到 NVIDIA GPU: $($nvidia -join ', ')" -ForegroundColor Green
    }
} catch { }

# ---------- 3. 安装依赖 ----------
Write-Host "[3/4] 安装依赖（清华镜像，可能需要几分钟）..." -ForegroundColor Yellow

$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"

if ($HasGPU) {
    Write-Host "      安装 CUDA 版 PyTorch（download.pytorch.org）..."
    & $Pip install torch --index-url https://download.pytorch.org/whl/cu124 --timeout 60
    if ($LASTEXITCODE -ne 0) {
        Write-Host "      CUDA 版安装失败，回退到默认版 torch ..." -ForegroundColor Yellow
        & $Pip install torch --index-url $Mirror --timeout 60
    }
} else {
    Write-Host "      未检测到 GPU，安装 CPU 版 PyTorch ..."
    & $Pip install torch --index-url $Mirror --timeout 60
}

& $Pip install --index-url $Mirror --timeout 30 -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "依赖安装失败，请检查网络后重试。" -ForegroundColor Red; exit 1 }

# ---------- 4. 校验 ----------
Write-Host "[4/4] 校验安装 ..." -ForegroundColor Yellow
& $Py -c "import torch, transformers, sklearn, jieba, flask; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available()); print('transformers', transformers.__version__); print('sklearn', sklearn.__version__); print('flask', flask.__version__); print('jieba OK'); print('ALL OK')"
if ($LASTEXITCODE -ne 0) { Write-Host "校验失败，请手动检查。" -ForegroundColor Red; exit 1 }

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " 环境配置完成！后续命令示例：" -ForegroundColor Green
Write-Host "   .venv\Scripts\python.exe scripts\prepare_data.py   # 下载并整理数据集"
Write-Host "   .venv\Scripts\python.exe src\train.py              # 训练"
Write-Host "==============================================" -ForegroundColor Cyan
