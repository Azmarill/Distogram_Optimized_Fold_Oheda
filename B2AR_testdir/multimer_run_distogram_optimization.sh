#!/bin/bash -l
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -g gn46
#PJM -L elapse=02:00:00
#PJM -j
#PJM -o /work/gn46/n46001/tools/openfold/logs/egf_refine_%j.log

# --- 1. モジュールとConda環境の有効化 ---
echo "Loading modules and activating Conda environment..."
#module load cuda/11.8
module load cuda/12.2
#module load gcc-toolset/10

# 計算ノード上でCondaコマンドを使えるように初期化
CONDA_BASE=/work/04/gn46/share/conda/miniconda3
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# EGF用のConda環境を有効化
#conda activate openfold-venv
conda activate egf_env
#export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH

export LD_PRELOAD=/work/gn46/share/conda/miniconda3/envs/egf_env/x86_64-conda-linux-gnu/lib64/libstdc++.so.6

# --- 2. 作業ディレクトリとキャッシュの設定 ---
# EGFのコードが置いてあるディレクトリに移動
EGF_DIR="/work/gn46/share/do_fold/Distogram_Optimized_Fold_Oheda/" # このパスはご自身の環境に合わせてください
cd "$EGF_DIR"

export LD_LIBRARY_PATH=/work/gn46/share/conda/miniconda3/envs/egf_env/lib:$LD_LIBRARY_PATH

# ホームディレクトリへの書き込みエラーを防ぐため、キャッシュの場所を指定
# 出力ディレクトリ内にキャッシュ用のサブディレクトリを作成
CACHE_DIR="/work/gn46/n46001/tools/openfold/outputs/cache_${PJM_JOBID}"
mkdir -p "$CACHE_DIR"
export TRITON_CACHE_DIR="$CACHE_DIR/.triton"
mkdir -p "$TRITON_CACHE_DIR"
export MPLCONFIGDIR="$CACHE_DIR/.config/matplotlib" # Matplotlibのキャッシュ
export XDG_CACHE_HOME="$CACHE_DIR/.cache"           # その他のキャッシュ
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME"

echo "Working directory: $(pwd)"
echo "Cache directory: ${CACHE_DIR}"

# --- 3. EGFの実行 ---
echo "Starting EGF refinement..."

# あなたが実行したいコマンド
python main.py --config-name=multimer_refine_B2AR_inactive

echo "Distogram Optimization script finished."

# --- 4. 環境の非アクティブ化 ---
conda deactivate
