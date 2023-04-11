import numpy as np
import tensorly as tl
import warnings
from math import sqrt
from tensorly.tenalg import multi_mode_dot
from tensorly.base import unfold
from tensorly.tenalg.svd import svd_interface


def initialize_tucker(
    tensor,
    rank,
    modes,
    random_state,
    init="svd",
    svd="truncated_svd",
    non_negative=False,
    mask=None,
    svd_mask_repeats=5,
):
    # Initialisation
    if init == "svd":
        factors = []
        for index, mode in enumerate(modes):
            mask_unfold = None if mask is None else unfold(mask, mode)
            U, _, _ = svd_interface(
                unfold(tensor, mode),
                n_eigenvecs=rank[index],
                method=svd,
                non_negative=non_negative,
                mask=mask_unfold,
                n_iter_mask_imputation=svd_mask_repeats,
                random_state=random_state,
            )

            factors.append(U)
        # The initial core approximation is needed here for the masking step
        core = multi_mode_dot(tensor, factors, modes=modes, transpose=True)

    elif init == "random":
        rng = tl.check_random_state(random_state)
        core = tl.tensor(
            rng.random_sample(rank) + 0.01, **tl.context(tensor)
        )  # Check this
        factors = [
            tl.tensor(rng.random_sample(s), **tl.context(tensor))
            for s in zip(tl.shape(tensor), rank)
        ]

    else:
        (core, factors) = init

    if non_negative is True:
        factors = [tl.abs(f) for f in factors]
        core = tl.abs(core)

    return core, factors


def partial_tucker(
    tensor,
    rank,
    modes=None,
    n_iter_max=100,
    init="svd",
    tol=10e-5,
    svd="truncated_svd",
    random_state=None,
    verbose=False,
    svd_mask_repeats=5,
    percentage=90,
):
    mask = np.zeros(tensor.shape)
    if modes is None:
        modes = list(range(tl.ndim(tensor)))

    if rank is None:
        message = "No value given for 'rank'. The decomposition will preserve the original size."
        warnings.warn(message, Warning)
        rank = [tl.shape(tensor)[mode] for mode in modes]
    elif isinstance(rank, int):
        message = f"Given only one int for 'rank' instead of a list of {len(modes)} modes. Using this rank for all modes."
        warnings.warn(message, Warning)
        rank = tuple(rank for _ in modes)
    else:
        rank = tuple(rank)

    # SVD init
    core, factors = initialize_tucker(
        tensor,
        rank,
        modes,
        init=init,
        svd=svd,
        random_state=random_state,
        mask=mask,
        svd_mask_repeats=svd_mask_repeats,
    )

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    for iteration in range(n_iter_max):
        tensor = tensor * mask + multi_mode_dot(
            core, factors, modes=modes, transpose=False
        ) * (1 - mask)

        for index, mode in enumerate(modes):
            core_approximation = multi_mode_dot(
                tensor, factors, modes=modes, skip=index, transpose=True
            )
            eigenvecs, _, _ = svd_interface(
                unfold(core_approximation, mode),
                n_eigenvecs=rank[index],
                random_state=random_state,
            )
            factors[index] = eigenvecs

        core = multi_mode_dot(tensor, factors, modes=modes, transpose=True)

        # The factors are orthonormal and therefore do not affect the reconstructed tensor's norm
        rec_error = sqrt(abs(norm_tensor**2 - tl.norm(core, 2) ** 2)) / norm_tensor
        rec_errors.append(rec_error)

        if iteration > 1:
            if verbose:
                print(
                    f"reconstruction error={rec_errors[-1]}, variation={rec_errors[-2] - rec_errors[-1]}."
                )

            if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                if verbose:
                    print(f"converged in {iteration} iterations.")
                break

    return (core, factors), rec_errors
