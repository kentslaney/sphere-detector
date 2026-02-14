import jax
import jax.numpy as jnp

@jax.jit
def fit_circle_algebraic(points):
    mean = jnp.mean(points, axis=0)
    x, y = jnp.unstack(points - mean, axis=1)

    a = jnp.stack([2 * x, 2 * y, jnp.ones_like(x)], axis=1)
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
    [7.0, 7.0], [9.0, 5.0], [3.0, 2.0]
])

print(fit_circle_algebraic.lower(points_data).as_text())
center_x, center_y, radius = fit_circle_algebraic(points_data)

print(f"Center: ({center_x:.4f}, {center_y:.4f})")
print(f"Radius: {radius:.4f}")
