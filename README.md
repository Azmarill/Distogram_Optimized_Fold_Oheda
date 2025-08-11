# Distogram Optimized Fold (DOF) - 複合体の一部を精密化するEGF拡張ツール

## 概要

このプロジェクトは、[Entropy Guided Fold (EGF) [cite: 1]](https://www.biorxiv.org/content/10.1101/2025.04.26.650728v1) の独創的なアイデアに基づき、その機能を大幅に拡張したものです。元のEGFが単量体（Monomer）のコンフォメーション変化の予測に焦点を当てていたのに対し、このツールは**タンパク質複合体（Multimer）**、特に**複合体の一方の鎖の構造だけを、もう一方を固定したまま意図的に特定のコンフォメーションへと誘導する**ことを可能にします。

このREADMEでは、元のEGFコードから加えられた主要な変更点、このツール独自の機能、そして具体的な使用方法について詳細に解説します。

## 背景：なぜこのツールが必要か

AlphaFold2-Multimerは、共進化情報（MSA）に基づいて複合体の最も安定な構造を予測することに長けています。しかし、以下のような特殊なケースでは、その能力に限界がありました。

1.  **共進化情報のない複合体**: Nanobodyのように、自然界では共存しない人工的なタンパク質との複合体を予測する場合、モデルは正しい結合様式を知るための手がかりを失い、不正確な予測を行うことがあります。
2.  **特定のコンフォメーションへの誘導**: MSAが強く示唆する構造（例：活性型GPCR）とは異なる、別の状態（例：不活性型GPCR）との複合体を予測しようとすると、モデル内部で情報の矛盾が生じ、パートナーの構造が破綻してしまう問題がありました。

本ツールは、これらの問題を解決するために、EGFの「勾配降下による内部表現の修正」というアイデアを、より高度で精密なレベルへと昇華させたものです。

## 工夫点・実装した独自機能

このツールは、元のEGFのコードに対して、以下の重要な改造が加えられています。

### 1\. 完全な多量体（Multimer）対応

EGFのコードは元々、多量体のデータ処理パイプラインが不完全でした。FASTAパーサーの置き換え、`preprocess`メソッドの全面的な書き直し、そして最終的なPDB/CIF出力機能のバグ修正を行い、単量体と多量体の両方をシームレスに扱えるようにしました。

### 2\. 精密な部分構造の誘導（このツールの核心）

「複合体の片方だけを動かしたい」という目的を達成するため、勾配降下のプロセスに詳細な調整を可能にする、**更新マスク**の概念を導入しました。

  * **ファイル**: `openfold/model/model.py`
  * **新規メソッド**: `_build_update_masks`
  * **改造メソッド**: `opt`

#### **更新マスク (`_build_update_masks`)**

この新しい関数は、`.yaml`設定ファイルからの指示に基づき、どの部分の内部表現を更新してよいかを制御するマスクを動的に生成します。

  * **`target_chain_ids`**: どの鎖を誘導のターゲットにするかを指定します。
  * **`interchain_mode`**: 結合界面を「完全に固定（`freeze`）」するか、「ターゲットの変化に合わせて再調整（`refine`）させる」かを選択できます。

#### **マスクの適用 (`opt`メソッド)**

最適化ループの中で、計算された勾配（`m_param.grad`, `z_param.grad`）に、この更新マスクを掛け合わせます。

```python
m_param.grad *= m_up_mask
z_param.grad *= z_up_mask
```

これにより、例えば**GPCRの内部構造だけ**を更新し、**Nanobodyと結合界面は一切変更しない**、といった外科手術のような精密な制御が可能になりました。

### 3\. 高度な損失関数の設計

多量体の精密化をより細かく制御するため、`gt_distogram_loss`関数を大幅に改造しました。

  * **ファイル**: `openfold/model/model.py`
  * **改造メソッド**: `gt_distogram_loss`

この新しい損失関数は、マスクを利用して損失の計算範囲を限定するだけでなく、\*\*鎖間（inter-chain）**と**鎖内（intra-chain）\*\*の損失を分離し、それぞれに異なる重みを与えることができます。

```python
loss = loss_inter + w_intra * loss_intra
```

これにより、「鎖の内部構造はできるだけ維持しつつ、鎖間の相互作用だけを重点的に精密化する」といった、より高度な実験設計が可能になりました。

### 4\. 信頼度スコア（ipTM/pTM）の計算機能

元のEGFでは省略されていた、複合体の信頼度を評価するために不可欠な**ipTM**と**pTM**のスコア計算機能を、`postprocess`メソッドに新たに実装しました。これにより、誘導された構造の信頼性を定量的に評価できます。

## 使用方法

### ステップ1：入力ファイルの準備

1.  **FASTAファイル**: 予測したい複合体のFASTAファイル（単量体または多量体）を準備します。
2.  **MSAファイル（事前計算済み）**: 各鎖に対応するペアリング済みのMSAファイル（`.sto`または`.a3m`）を準備し、正しいディレクトリ構造に配置します。MSAの\*\*深さ（配列数）\*\*は、全ての鎖で完全に一致している必要があります。
3.  **正解データ（Ground Truth）**: 誘導の目標となる構造のPDBファイルから、`create_truth_distances.py`スクリプトを使って、生の距離行列（`.npy`）を作成します。
4.  **マスクファイル（多量体の場合）**: `make_mask_for_multimer.py`のようなスクリプトを使い、どの部分を誘導の対象にするかを示すマスクファイル（`.npy`）を作成します。

### ステップ2：設定ファイルの作成

`configs/`ディレクトリに、あなたの実験用の`.yaml`設定ファイルを作成します。以下は、多量体の一部だけを誘導する場合の設定例です。

```yaml
# defaults: ...

fasta_dir: /path/to/your/fastas/
alignment_dir: /path/to/your/alignments/
output_dir: /path/to/your/results/
structure_dir: /path/to/your/ground_truth_pdbs/

base:
  config_preset: model_1_multimer_v3
  jax_param_path: /path/to/your/params_model_1_multimer_v3.npz
  # ... (max_template_dateなどの必須項目)

guide_config:
  tag_cluster_mapping_path: /path/to/your/tag_mapping.json
  info_dir: ${output_dir}/info
  iter_0:
    opt: true
    num_gradient_steps: 10
    guidance_lr: 0.01
    
    # --- 損失に関する設定 ---
    gt_distogram_weight: 1.0 # 誘導の力の強さ
    gt_distances_path: /path/to/your/ground_truth_distances.npy
    
    # --- あなたが追加した新しい制御スイッチ ---
    gt_mask_path: /path/to/your/multimer_mask.npy # 使うマスク
    target_chain_ids: [1]                       # 1番目の鎖(GPCR)をターゲットに
    interchain_mode: 'freeze'                   # 結合界面は完全に固定

  # (max_recycling_iters や iter_1 以降の設定)
```

### ステップ3：実行

準備した設定ファイルを指定して、メインスクリプトを実行します。

```bash
python main.py --config-name=your_experiment_config
```

## 結論と今後の展望

このツールは、EGFの強力なアイデアを、より複雑で現実的な生物学的問題へと応用するための、柔軟な研究プラットフォームです。特に、複合体の一部分だけを選択的に操作する機能は、薬剤設計やタンパク質間相互作用のメカニズム解明において、新しい可能性を切り開くものです。

今後の展望として、このフレームワークをAlphaFold 3のアーキテクチャに移植することや、`m_param`/`z_param`を直接予測するメタ学習モデルを構築することが考えられます。

-----
