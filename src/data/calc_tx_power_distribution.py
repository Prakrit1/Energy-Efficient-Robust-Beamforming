import numpy as np

def calc_tx_power_distribution(
    w_precoder: np.ndarray,
) -> np.ndarray:
    """TODO: Comment"""

    user_nr = w_precoder.shape[1]

    power_per_user = np.zeros(user_nr)

    for user_id in range(user_nr):

        norm = np.linalg.norm(w_precoder[:, user_id])

        power_per_user[user_id] = norm**2

    return power_per_user
