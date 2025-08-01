import numpy as np

# FFA2の長さを取得 (例)
len_ffa2 = 278 
# 複合体全体の長さを取得 (例)
len_total = 632

# まずは全て0で初期化
mask = np.zeros((len_total, len_total), dtype=bool)

# FFA2の範囲 (0からlen_ffa2-1まで) だけをTrueにする
mask[:len_ffa2, :len_ffa2] = True

# マスクを.npyファイルとして保存
np.save("ffa2_multimer_mask.npy", mask)
