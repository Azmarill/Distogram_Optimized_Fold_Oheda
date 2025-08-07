import numpy as np

len_gpcr = 388
len_total = 508

mask = np.zeros((len_total, len_total), dtype=bool)

mask[:len_gpcr, :len_gpcr] = True

np.save("B2AR_active_multimer_mask.npy", mask)
