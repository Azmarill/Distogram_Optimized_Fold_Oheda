import numpy as np

len_gpcr = 285
len_total = 410

mask = np.zeros((len_total, len_total), dtype=bool)

mask[:len_gpcr, :len_gpcr] = True

np.save("Sec_B2AR_inactive_multimer_mask.npy", mask)
