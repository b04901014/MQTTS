import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import linalg
import random

vaddir_tts = 'cache/{...}'
vaddir_gt = 'cache/{...}'
normalize = lambda x: (x - x.mean(0)) / x.std(0)

def fid():
    clus_tts = np.load(vaddir_tts) * 10
    print (clus_tts.shape)
    clus_gt = np.load(vaddir_gt) * 10
    clus_gt = clus_gt[50000:]
    mu_clus_tts = np.mean(clus_tts, 0)
    mu_clus_gt = np.mean(clus_gt, 0)
    sigma_tts = np.cov(clus_tts, rowvar=False)
    sigma_gt = np.cov(clus_gt, rowvar=False)
    sigma_tts = np.atleast_2d(sigma_tts)
    sigma_gt = np.atleast_2d(sigma_gt)
    diff = mu_clus_tts - mu_clus_gt
    covmean, _ = linalg.sqrtm(sigma_tts.dot(sigma_gt), disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print (msg)
        offset = np.eye(mu_clus_tts.shape[0]) * eps
        covmean = linalg.sqrtm((mu_clus_tts + offset).dot(sigma_gt + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    print (diff.dot(diff), np.trace(sigma_tts), np.trace(sigma_gt), tr_covmean)
    fid = diff.dot(diff) + np.trace(sigma_tts) + np.trace(sigma_gt) - 2 * tr_covmean
    return fid
print (fid())
