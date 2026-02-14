import jax
import jax.numpy as jnp

def fit_circle_algebraic(points, valid):
    mean = jnp.sum(jnp.where(valid, points, 0), axis=1) / jnp.sum(valid)
    x, y = jnp.unstack(jnp.where(valid, points - mean[:, None], 0), axis=0)

    a = jnp.stack([2 * x, 2 * y, valid], axis=1)
    b = x ** 2 + y ** 2

    ata, atb = a.T @ a, a.T @ b
    cols = jnp.unstack(ata, axis=1)
    cramer = lambda idx: jnp.stack(cols[:idx] + (atb,) + cols[idx + 1:], axis=1)

    det = jnp.linalg.det(ata)
    det = jnp.where(jnp.abs(det) < 1e-10, 1e-10, det)

    a_, b_, c = (jnp.linalg.det(cramer(i)) / det for i in range(3))
    return a_ + mean[0], b_ + mean[1], jnp.sqrt(c + a_ ** 2 + b_ ** 2)

# Example Usage
points_data = jnp.array([
    [1.0, 7.0], [2.0, 6.0], [5.0, 8.0],
    [7.0, 7.0], [9.0, 5.0], [3.0, 2.0],
    [-1.0, -1.0]
]).T
valid = jnp.array([True, True, True, True, True, True, False])

center_x, center_y, radius = fit_circle_algebraic(points_data, valid)

# print(jax.jit(fit_circle_algebraic).lower(points_data, valid).as_text())

print(f"Center: ({center_x:.4f}, {center_y:.4f}) ref (4.8166, 4.8077)")
print(f"Radius: {radius:.4f} ref 3.5881")
