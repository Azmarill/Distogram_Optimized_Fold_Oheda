import numpy as np

# ロード
distogram = np.load("FFA2_inactive_do_truth_distogram_full.npy")

# 最大値を考慮したbinの作成（例えば64個のbin）
max_dist = 62.2
num_bins = 64
bin_edges = np.linspace(0, max_dist, num_bins - 1)

# 距離をbin番号に変換（整数化）
distogram_bins = np.digitize(distogram, bin_edges).astype(np.int64)

# binの番号を確認（0〜63になることを確認）
print("binの最小値:", distogram_bins.min())
print("binの最大値:", distogram_bins.max())

# bin化したdistogramを保存
np.save("FFA2_inactive_do_truth_distogram_full_bins.npy", distogram_bins)

