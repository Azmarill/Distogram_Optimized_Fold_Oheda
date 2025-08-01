#!/bin/bash

# Pythonスクリプトのパス
PROCESS_SCRIPT_PATH="/work/gn46/share/egf/EGF/create_truth_distances.py"

# 出力先ディレクトリ（なければ作成）
OUTPUT_DIR="./distogram_outputs"
mkdir -p "$OUTPUT_DIR"

# カレントディレクトリにある全ての.pdbファイルでループ
for pdb_file in ./*.pdb; do
  # ファイルが存在しない場合はスキップ (ファイルが一つもない場合のエラー防止)
  [ -e "$pdb_file" ] || continue

  # ファイル名(拡張子なし)を取得
  basename=$(basename "${pdb_file}" .pdb)

  # 出力ファイル名を定義
  output_npy_file="${OUTPUT_DIR}/${basename}_truth_distogram.npy"
  
  # Pythonスクリプトを実行
  python "${PROCESS_SCRIPT_PATH}" "${pdb_file}" "${output_npy_file}"
done

echo "All PDB files have been processed."
