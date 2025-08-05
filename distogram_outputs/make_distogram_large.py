import numpy as np

# 元のdistogram (263x263)
distogram = np.load("FFA2_inactive_do_truth_distogram.npy")

# 新しい大きさ(617x617)で初期化
distogram_full = np.zeros((617, 617), dtype=distogram.dtype)

# FFA2の領域(左上263×263)に元のdistogramをコピー
distogram_full[:263, :263] = distogram

# 新しいdistogramを保存
np.save("FFA2_inactive_do_truth_distogram_full.npy", distogram_full)

