#!/usr/bin/env python3

from nemo.collections.asr.parts.utils.offline_clustering import (
    NMESC,
    getCosAffinityMatrix,
    getAffinityGraphMat,
    SpectralClustering,
)


def clustering(
    embeddings,
    num_speakers=-1,
    max_num_speakers=6,
    max_rp_threshold=0.15,
    sparse_search=True,
    sparse_search_volume=10,
    fixed_thres=-1.0,
    nme_mat_size=300,
    cuda=False,
):
    """NMESC-based clustering"""
    mat = getCosAffinityMatrix(embeddings)
    nmesc = NMESC(
        mat,
        max_num_speakers=max_num_speakers,
        max_rp_threshold=max_rp_threshold,
        sparse_search=sparse_search,
        sparse_search_volume=sparse_search_volume,
        fixed_thres=fixed_thres,
        nme_mat_size=nme_mat_size,
        cuda=cuda,
    )
    try:
        est_num_of_spk, p_hat_value = nmesc.forward()
        affinity_mat = getAffinityGraphMat(mat, p_hat_value)
        if num_speakers > 0:
            est_num_of_spk = num_speakers
        spectral_model = SpectralClustering(n_clusters=est_num_of_spk, cuda=cuda)
        Y = spectral_model.forward(affinity_mat)
    except Exception as err:
        print(
            "NMESC-based speaker counting failed, which usually happens when there's only 1 segment."
        )
        print(embeddings.shape)
        print(err)
        Y = [0 for _ in range(embeddings.shape[0])]

    return Y
