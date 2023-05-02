import tensorly as tl
from tensorly.tenalg.svd import svd_interface
from tensorly.base import unfold
from tensorly.tenalg import multi_mode_dot
from math import sqrt
import numpy as np
from structures import PruningStructure


def initialize_tucker(
    tensor,
    rank,
    modes,
    svd_mask_repeats=5,
):
    factors = []
    for index, mode in enumerate(modes):
        U, _, _ = svd_interface(
            unfold(tensor, mode),
            n_eigenvecs=rank[index],
            method="truncated_svd",
            non_negative=False,
            mask=None,
            n_iter_mask_imputation=svd_mask_repeats,
            random_state=None,
        )

        factors.append(U)
    core = multi_mode_dot(tensor, factors, modes=modes, transpose=True)
    return core, factors


def partial_tucker_spar(
    tensor,
    rank,
    modes=(2, 3),
    spar=90,
    ps: PruningStructure = PruningStructure(),
    n_iter_max=100,
    tol=1e-3,
    svd_mask_repeats=5,
):
    if isinstance(rank, int):
        rank = (rank, rank)
    else:
        rank = tuple(rank)
    core, factors = initialize_tucker(
        tensor,
        rank,
        modes,
        svd_mask_repeats=svd_mask_repeats,
    )
    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)
    t_copy = tensor.copy()
    diff = abs(t_copy)
    t = np.percentile(diff, spar)
    t_copy[diff > t] = 0
    for iteration in range(n_iter_max):
        for index, mode in enumerate(modes):
            core_approximation = multi_mode_dot(
                t_copy, factors, modes=modes, skip=index, transpose=True
            )
            eigenvecs, _, _ = svd_interface(
                unfold(core_approximation, mode),
                n_eigenvecs=rank[index],
                random_state=None,
            )
            factors[index] = eigenvecs
        core = multi_mode_dot(t_copy, factors, modes=modes, transpose=True)
        t_approx = multi_mode_dot(core, factors, modes=modes)
        diff = ps.reduce_ker(abs(t_approx - tensor))
        t = np.percentile(diff, spar)
        mask = ps.transform_mask(diff > t, tensor.shape)
        t_copy = tensor.copy()
        t_copy[mask] = t_approx[mask]
        rec_error = sqrt(abs(norm_tensor**2 - tl.norm(core, 2) ** 2)) / norm_tensor
        rec_errors.append(rec_error)
        if iteration > 1:
            if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                break
    sp = np.zeros(tensor.shape, dtype=np.float32)
    sp[mask] = tensor[mask] - t_approx[mask]
    return core, *factors, sp
