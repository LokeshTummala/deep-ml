import numpy as np


def gradient_direction_magnitude(gradient: list) -> dict:
  """
    Calculate the magnitude and direction of a gradient vector.
    """
  grad_arr = np.array(gradient, dtype=float)
  l2_norm = float(np.linalg.norm(grad_arr, ord=2))

  if l2_norm == 0:
    zeros = [0.0] * len(gradient)
    return {
        'magnitude': 0.0,
        'direction': zeros,
        'descent_direction': zeros,
    }

  direction = grad_arr / l2_norm
  descent_direction = -direction

  return {
      'magnitude': l2_norm,
      'direction': direction.tolist(),
      'descent_direction': descent_direction.tolist(),
  }
