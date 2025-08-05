import numpy as np

#len_ffa2 = 263 
#len_total = 617

# マスクをboolではなくfloat32で作成
#mask = np.zeros((len_total, len_total), dtype=np.float32)
#mask[:len_ffa2, :len_ffa2] = 1.0  # 1.0でTrueを表現

#np.save("ffa2_multimer_mask.npy", mask)

distogram = np.load("FFA2_inactive_do_truth_distogram.npy").astype(np.float32)

distogram_full = np.zeros((617,617), dtype=np.float32)
distogram_full[:263,:263] = distogram

np.save("FFA2_inactive_do_truth_distogram_full.npy", distogram_full)

