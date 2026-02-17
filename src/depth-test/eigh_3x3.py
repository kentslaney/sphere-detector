import jax
import jax.numpy as jnp

def get_eigenvalues_3x3(A):
    """
    Computes the eigenvalues of a 3x3 symmetric matrix A using the trigonometric formula.
    A is expected to be a symmetric matrix of shape (3, 3).
    """

    # 1. Characteristic polynomial: det(A - lambda I) = 0
    # lambda^3 - tr(A) * lambda^2 + 0.5 * (tr(A)^2 - tr(A^2)) * lambda - det(A) = 0

    # Coefficients for x^3 + a*x^2 + b*x + c = 0
    # But notice signs: standard form x^3 + ...
    # From det(M - lambda I) = -lambda^3 + tr(M)lambda^2 + ...
    # Let lambda = x.
    # P(x) = x^3 - tr(A)x^2 + 0.5(tr(A)^2 - tr(A^2))x - det(A)

    trace_A = jnp.trace(A)
    # Using slightly more direct calculation for b and c
    # b = sum of principal minors of order 2
    # c = -det(A)

    # Let's shift the matrix to have zero trace for better stability?
    # No, trigonometric formula handles x^3 + px + q = 0.
    # Let x = y + tr(A)/3

    p1 = jnp.trace(A @ A)
    det_A = jnp.linalg.det(A)

    # Coefficients of characteristic polynomial:
    # x^3 + c2*x^2 + c1*x + c0 = 0
    c2 = -trace_A
    c1 = 0.5 * (trace_A**2 - p1)
    c0 = -det_A

    # Reduce to depressed cubic linear form: y^3 + p*y + q = 0
    # Substitution x = y - c2/3 = y + trace_A/3
    tr_div_3 = trace_A / 3.0

    # p = c1 - c2^2 / 3
    p = c1 - (c2**2) / 3.0

    # q = c0 - c1*c2/3 + 2*c2^3/27
    q = c0 - (c1 * c2) / 3.0 + 2.0 * (c2**3) / 27.0

    # Solutions for y (using trigonometric solution for real roots):
    # If p=0, then y^3 + q = 0 => y = -cbrt(q). All roots equal?
    # For symmetric matrices, roots are real.
    # If p < 0, we can use trigonometric form.
    # If p > 0, something is wrong? (Symmetric matrix eigenvalues must be real).
    # p = c1 - c2^2/3 = 1/2(tr(A)^2 - tr(A^2)) - tr(A)^2/3
    #   = 1/6 tr(A)^2 - 1/2 tr(A^2)
    # By Cauchy-Schwarz, tr(A^2) >= tr(A)^2 / 3 for 3x3?
    # Let's check: sum(lambda_i^2) >= (sum lambda_i)^2 / 3.
    # 3 * sum(lambda^2) - (sum lambda)^2 = 3 sum lambda^2 - (sum lambda^2 + 2 sum mixed)
    # = 2 sum lambda^2 - 2 sum mixed = sum (lambda_i - lambda_j)^2 >= 0.
    # So p is always <= 0.

    # Use p_sqrt = sqrt(-p) to avoid NaN if p is slightly positive due to numerical error.
    # Enforce p <= 0.
    p = jnp.minimum(p, 0.0)
    limit_p = 1e-20 # Avoid division by zero

    # safe_p for division
    p_safe = jnp.minimum(p, -limit_p)

    # Formula: y_k = 2 * sqrt(-p/3) * cos((theta + 2k*pi)/3)
    # where cos(theta) = 3q / (2p) * sqrt(-3/p)
    # Let r = sqrt(-p/3)
    r = jnp.sqrt(-p_safe / 3.0)

    # cos_theta = -q / (2 * r^3)
    # Wait, substitution leads to 4 cos^3 - 3 cos = ...
    # y = 2r cos(phi)
    # (2r cos phi)^3 + p(2r cos phi) + q = 0
    # 8r^3 cos^3 phi + 2pr cos phi + q = 0
    # 4 cos^3 phi + (p/r^2) cos phi + q/(2r^3) = 0
    # p/r^2 = p / (-p/3) = -3
    # 4 cos^3 phi - 3 cos phi = -q / (2r^3)
    # cos(3phi) = -q / (2r^3)

    cos_3phi = -q / (2.0 * r**3)
    # Clamp cos_3phi to [-1, 1] for acos
    cos_3phi = jnp.clip(cos_3phi, -1.0, 1.0)

    phi_0 = jnp.arccos(cos_3phi) / 3.0

    # Roots in y
    y0 = 2.0 * r * jnp.cos(phi_0)
    y1 = 2.0 * r * jnp.cos(phi_0 + 2.0 * jnp.pi / 3.0)
    y2 = 2.0 * r * jnp.cos(phi_0 + 4.0 * jnp.pi / 3.0)

    # Roots in x (eigenvalues)
    x0 = y0 + tr_div_3
    x1 = y1 + tr_div_3
    x2 = y2 + tr_div_3

    # Handle the p ~ 0 case (all eigenvalues equal)
    # If p is very small, r is small, y is small.
    # x ~= tr_div_3.
    # We can just return x0, x1, x2.

    # Sort eigenvalues?
    # Result of arccos is in [0, pi].
    # phi_0 in [0, pi/3].
    # cos(phi_0) >= cos(phi_0 + 2pi/3) ...?
    # Let's just standard sort them at the end.

    lambdas = jnp.stack([x0, x1, x2])
    lambdas = jnp.sort(lambdas) # Ascending order usually

    return lambdas

def get_eigenvectors_3x3(A, lambdas):
    """
    Computes eigenvectors for given eigenvalues.
    Uses cross-product method.
    """

    # We need to find v such that (A - lambda_i I) v = 0.
    # Method: Rows of cofactor matrix of (A - lambda_i I) are proportional to v.
    # C_jk = (-1)^{j+k} M_{jk} where M is minor.
    # Or simply: cross product of any two rows of (A - lambda_i I) is an eigenvector.
    # To be robust, we compute all 3 cross products and take the one with largest norm.

    # Unpack lambdas
    l0, l1, l2 = lambdas[0], lambdas[1], lambdas[2]

    def compute_ev_for_lambda(l):
        # M = A - l * I
        M = A - l * jnp.eye(3)

        # Rows
        r0 = M[0, :]
        r1 = M[1, :]
        r2 = M[2, :]

        # Cross products
        n0 = jnp.cross(r0, r1)
        n1 = jnp.cross(r1, r2)
        n2 = jnp.cross(r2, r0)

        # Norms squared
        d0 = jnp.sum(n0**2)
        d1 = jnp.sum(n1**2)
        d2 = jnp.sum(n2**2)

        # Pick largest
        # Since this is JAX, we can use argmax or just a weighted sum if we know only 1 is non-zero
        # But for robustness, just pick max.

        best_idx = jnp.argmax(jnp.array([d0, d1, d2]))

        # Using switch/cond
        v = jnp.where(best_idx == 0, n0,
              jnp.where(best_idx == 1, n1, n2))

        # Normalize
        norm_v = jnp.linalg.norm(v)

        # If norm is zero, we have a degenerate case (rank < 2).
        # This happens if l is repeated.
        # Fallback will be handled outside or via a safe vector.
        # Return vector and whether it's valid (norm > tol)
        return v, norm_v

    v0, norm0 = compute_ev_for_lambda(l0)
    v1, norm1 = compute_ev_for_lambda(l1)
    v2, norm2 = compute_ev_for_lambda(l2)

    # Logic for degenerate cases or near-degenerate
    tol = 1e-6 # Numerical tolerance for cross product vanishing

    # If distinct eigenvalues, all norms should be non-zero.
    # If l0 == l1 != l2 (e.g. 1, 1, 2)
    # v2 should be fine.
    # v0 and v1 calculation might involve rank-1 matrix A-l0*I.
    # Rank 1 => rows are proportional => cross product is zero.

    # We need a robust fallback.
    # Re-orthogonalization step or different construction.

    # Case 1: All distinct. v0, v1, v2 valid.
    # Case 2: l0 == l1. v0_calc, v1_calc might be zero. v2 valid.
    #         Need to find v0, v1 such that they are orthogonal to v2 and each other.
    # Case 3: l1 == l2.
    # Case 4: All equal. Identity. Any basis works.

    # Let's clean v2 first.
    v2_safe = jnp.where(norm2 > tol, v2 / (norm2 + 1e-20), jnp.array([0., 0., 1.]))
    # This fallback is only if v2 itself failed (rank 0? A-l2 I = 0 => A diagonal).
    # If A diagonal, cross products are 0?
    # Yes, if A is diagonal 3x3, A-lI has rows with 1 zero.
    # E.g. M = diag(d, 0, 0). r0=(d,0,0), r1=0. Cross=0.
    # My cross product logic fails for diagonal matrices if multiple entries are zero?
    # Actually if A is diagonal [1, 2, 3]. l=1. M=[0, 1, 2].
    # rows: (0,0,0), (0,1,0), (0,0,2).
    # r0xr1 = 0. r1xr2 = (2,0,0). Correct (eigenvector e1).
    # So for distinct diagonal, it works.

    # Fallback Handling:
    # We construct the basis vectors iteratively to ensure orthogonality.

    # 1. Select the "best" eigenvector from the 3 computed.
    #    The one corresponding to the most isolated eigenvalue is usually most stable?
    #    Or just the one with largest cross-product norm.

    norms = jnp.array([norm0, norm1, norm2])
    best_overall = jnp.argmax(norms) # Index of most reliable vector

    # Let's pick vectors.
    # Ideally v0, v1, v2 are mutually orthogonal.

    # Strategy:
    # Take the valid vectors.
    # Use Gram-Schmidt or similar?
    # To avoid branching, maybe just:
    # v0 = normalize(v0) if valid else random_ortho?

    # Better Strategy for JAX friendly code:
    # Compute v0, v1, v2.
    # Mask invalid ones (norm < tol).
    # If v0, v1, v2 all valid, return them.
    # If not, generate arbitrary orthogonal vectors.

    # Since specific "closed form" was requested, let's use the property:
    # V = [v0, v1, v2].
    # If l0=l1, then we can pick v0 arbitrarily in plane orthogonal to v2.
    # v0_new = cross(v2, arbitrary_vec).
    # v1_new = cross(v2, v0_new).

    # Let's standardize the output vectors.
    # Normalize with safe division
    v0_safe = v0 / (norm0 + 1e-20)
    v1_safe = v1 / (norm1 + 1e-20)
    v2_safe = v2 / (norm2 + 1e-20)

    # Check validity masks
    valid0 = norm0 > tol
    valid1 = norm1 > tol
    valid2 = norm2 > tol

    # If lambdas are distinct, all valid.

    # What if l0 == l1? v0 and v1 might be invalid (rank 1 matrix). v2 valid.
    # Then v0 = cross(v2, axis_x). If zero, try cross(v2, axis_y).
    # Then v1 = cross(v2, v0).

    # We can perform a "fixup" pass.
    # We rely on sorting l0 <= l1 <= l2.
    # If l0 approx l1, we trust v2 most?
    # Not necessarily. The separation det(A-lI) handles it?

    # Let's perform a simple prioritized fixup.
    # Order of trust: valid2 (most distinct typically if l0=l1?)
    # Wait, if l1=l2, then v0 is the distinct one.

    # If l0 is distinct from l1 (gap), v0 is valid.
    # If l1 is distinct from l2 (gap), v2 is valid. (l2 distinct from l1 implies l2 distinct from l0).

    # Let's assume we can pick ONE good vector always (unless A=I).
    # Find index of max norm.
    k = jnp.argmax(norms)
    vk = jnp.stack([v0_safe, v1_safe, v2_safe])[k] # The most robustly computed vector
    is_valid_k = norms[k] > tol

    # If even the best one is 0, then A is scalar * I. Return Identity.
    identity_ev = jnp.eye(3)

    # Else, we have vk.
    # Let's find a vector orthogonal to vk.
    # Try cross(vk, e1). If parallel, cross(vk, e2).
    # u1 = cross(vk, e1)
    e1 = jnp.array([1., 0., 0.])
    e2 = jnp.array([0., 1., 0.])

    u1 = jnp.cross(vk, e1)
    norm_u1 = jnp.linalg.norm(u1)

    u2 = jnp.cross(vk, e2)
    norm_u2 = jnp.linalg.norm(u2)

    # Pick better u
    u = jnp.where(norm_u1 > norm_u2, u1, u2)
    u = u / jnp.linalg.norm(u) # Normalized first orthogonal vector

    # Second orthogonal vector
    w = jnp.cross(vk, u) # Already normalized

    # Now we have basis {vk, u, w}.
    # We need to map them to the corresponding eigenvalues.
    # If all distinct, we should have used the original v0, v1, v2.
    # Using the "one good vector + completion" only works if the other two are degenerate (l1=l2).

    # So we need to detect degeneracy to switch strategies?
    # Or can we blend?

    # Let's try to construct final [v0, v1, v2] by selecting valid ones.

    # Final robust assembly:
    # 1. Initialize V = [v0_safe, v1_safe, v2_safe]
    # 2. Iterate to fill invalid ones? Hard in JAX without scan/loop.

    # Let's use the explicit gap checks.
    # diff10 = l1 - l0
    # diff21 = l2 - l1

    # If diff10 is small, recompute v0, v1.
    # If diff21 is small, recompute v1, v2.

    # Actually, simpler:
    # If v0 is invalid, it means l0 is degenerate with someone.
    # If v0 valid, keep it.

    # Algorithm:
    # 1. Keep valid vectors.
    # 2. If we have 3 valid vectors, orthogonalize them (Graham Schmidt) to be safe against numerical error?
    #    Actually cross product method produces orthogonal vectors for distinct eigenvalues automatically.

    # 3. If missing vectors:
    #    If only 1 valid (say v2), generate u, w as above.
    #    Assign them to v0, v1.
    #    This assumes l0=l1.

    #    What if v1 is valid (l1 distinct from l0 and l2?? Impossible for sorted).
    #    Sorted: l0 <= l1 <= l2.
    #    Degeneracies can be:
    #    a) l0 = l1 < l2. (v0, v1 invalid? v2 valid).
    #    b) l0 < l1 = l2. (v0 valid, v1, v2 invalid?).
    #    c) l0 = l1 = l2. (All invalid).

    #    So we count valids.
    #    If 0 valid => Identity.
    #    If 1 valid => Expand orthogonal basis.
    #    If 3 valid => Return them.
    #    What about 2 valid? (e.g. l0 < l1 < l2 but numerical noise killed one?).
    #    Unlikely with cross product Max Row logic.

    # Let's implement this logic.

    # Count valid
    valid_count = (valid0.astype(int) + valid1.astype(int) + valid2.astype(int))

    # Case All Valid (3)
    # Just return stacked.

    # Case 1 Valid
    # Find which one is valid. Construct orthogonal complement.
    # If k=0 valid, v0 is good. v1, v2 bad (l1=l2). v1=u, v2=w.
    # If k=1 valid... (should not happen for sorted?)
    # If k=2 valid, v2 good. v0, v1 bad (l0=l1). v0=u, v1=w.

    # Case 0 Valid
    # I3.

    # Case 2 Valid?
    # Say v0, v2 valid. v1 invalid?
    # If l0 < l1 < l2, all should be valid.
    # If v1 invalid, maybe l1 is close to l0 or l2?
    # If v0, v2 valid, we can construct v1 = cross(v2, v0).

    # Consolidated logic:
    # Default: V_out = [v0_safe, v1_safe, v2_safe]

    # If (!valid0 or !valid1 or !valid2):
    #   Find stable vector 'primary'.
    #   primary = v[argmax(norms)]
    #   Find u, w orthogonal to primary.
    #
    #   BUT we must assign u, w to the correct slots!
    #   If l0=l1, then v0, v1 are the degenerate ones.
    #   If v2 is primary, then v0=u, v1=w.
    #   If l1=l2, then v1, v2 degenerate.
    #   If v0 is primary, then v1=u, v2=w.

    #   How to distiguish l0=l1 vs l1=l2 if we just rely on validity?
    #   Check indices of valid vectors?

    #   Actually, if we just fill invalid ones?
    #   If v1 invalid, set v1 = cross(v2, v0)?
    #   If v0 invalid, set v0 = cross(v1, v2)?

    #   This dependency loop is tricky.

    #   Robust single pass:
    #   v0_final, v1_final, v2_final.

    #   Refined "1 Valid" Logic:
    #   If (l0 close to l1):
    #       Treat as degenerate pair.
    #       Primary = v2.
    #       v0 = u, v1 = w.
    #   Else If (l1 close to l2):
    #       Treat as degenerate pair.
    #       Primary = v0.
    #       v1 = u, v2 = w.
    #   Else:
    #       Assume all distinct.

    gap01 = jnp.abs(l1 - l0)
    gap12 = jnp.abs(l2 - l1)

    # Heuristic for "close": relative to spectral radius?
    # Or just fixed tolerance?
    # Use the computed norms!
    # If norm0 is small, it implies l0 is part of a degeneracy (or A=0).

    # Let's try "Fixup" based on validity.

    # Fallback Basis (if nothing valid)
    basis_fallback = jnp.eye(3)

    # If all 3 valid:
    # result = [v0, v1, v2]

    # If degradation:
    # Identify the "Primary" vector.
    # If norm2 is good (likely isolated max eigenvalue), use it.
    # If norm0 is good (likely isolated min eigenvalue), use it.

    # Let's try to construct [out0, out1, out2]

    # Scenario A: v0, v2 good. v1 bad? -> linear interpolate? No.
    # If v0, v2 good, v1 = cross(v2, v0).

    # Scenario B: Only v2 good (l0=l1).
    # v0, v1 = orthogonal basis to v2.

    # Scenario C: Only v0 good (l1=l2).
    # v1, v2 = orthogonal basis to v0.

    # Scenario D: None good (l0=l1=l2).
    # Identity.

    # Let's implement this selection.

    # primary_idx = argmax(norms)
    # primary_vec = v[primary_idx]
    # u, w = orthogonal_complement(primary_vec)

    # If !valid0 and !valid1 and valid2:
    #    ans = [u, w, primary]
    # If valid0 and !valid1 and !valid2:
    #    ans = [primary, u, w]

    # What if valid0 and valid1 and valid2? Use original.

    # What if valid0 and valid2 but !valid1? (Intermediate eigenvalue problem).
    # ans = [v0, cross(v2, v0), v2].

    # How to control all this without explicit Python `if`? `jnp.select` or nested `where`.

    cond_all_valid = (valid0 & valid1 & valid2)
    cond_l0_l1_degen = (~valid0 & ~valid1 & valid2) # Approx
    cond_l1_l2_degen = (valid0 & ~valid1 & ~valid2) # Approx
    cond_all_degen = (~valid0 & ~valid1 & ~valid2)

    # Middle case: valid0 and valid2 but v1 bad?
    # Can happen?

    # Let's keep it simple.
    # if valid0 & valid2:
    #     out0 = v0
    #     out2 = v2
    #     out1 = cross(v2, v0)
    # elif valid2: # implies l0, l1 bad
    #     out2 = v2
    #     out0 = u_of_v2
    #     out1 = w_of_v2
    # elif valid0: # implies l1, l2 bad
    #     out0 = v0
    #     out1 = u_of_v0
    #     out2 = w_of_v0
    # else:
    #     Identity

    # Helper to clean up
    def get_ortho(k):
        # returns u, w orthogonal to vk
        vec = jnp.stack([v0_safe, v1_safe, v2_safe])[k]
        u1 = jnp.cross(vec, e1)
        u2 = jnp.cross(vec, e2)
        u = jnp.where(jnp.sum(u1**2) > jnp.sum(u2**2), u1, u2)
        u = u / (jnp.linalg.norm(u) + 1e-20)
        w = jnp.cross(vec, u)
        w = w / (jnp.linalg.norm(w) + 1e-20)
        return u, w

    u_v2, w_v2 = get_ortho(2)
    u_v0, w_v0 = get_ortho(0)

    # Candidates
    # 1. All Valid
    c1_v0, c1_v1, c1_v2 = v0_safe, v1_safe, v2_safe

    # 2. v0, v2 valid (Fix v1)
    c2_v0 = v0_safe
    c2_v2 = v2_safe
    c2_v1 = jnp.cross(c2_v2, c2_v0) # v2 x v0
    c2_v1 = c2_v1 / (jnp.linalg.norm(c2_v1)+1e-20)

    # 3. Only v2 valid (l0=l1)
    c3_v2 = v2_safe
    c3_v0 = u_v2
    c3_v1 = w_v2

    # 4. Only v0 valid (l1=l2)
    c4_v0 = v0_safe
    c4_v1 = u_v0
    c4_v2 = w_v0

    # 5. None valid
    c5_v0 = e1
    c5_v1 = e2
    c5_v2 = jnp.cross(e1, e2)

    # Selection Logic
    # Preference:
    # If v0, v2 valid -> Use C2 (Recalculate v1 to ensure orthogonality)
    # Else If v2 valid -> Use C3
    # Else If v0 valid -> Use C4
    # Else -> C5

    # Why not C1? C2 computes v1 from v0, v2 so it's perfectly orthogonal.
    # C1 relies on v1 being computed independently.
    # If v1 is valid but noisy, C2 might be better?
    # Actually, for distinct eigenvalues, v1 is orthogonal to v0, v2 analytically.
    # But C2 is safe. I will use C2 if v0, v2 are valid.

    # Only exception: if v2 invalid?

    lhs_valid = valid0
    rhs_valid = valid2

    # Branching
    # State 0: LHS & RHS valid
    # State 1: !LHS & RHS valid (l0=l1)
    # State 2: LHS & !RHS valid (l1=l2)
    # State 3: !LHS & !RHS (l0=l1=l2 or A=0)

    state = 0 * (lhs_valid & rhs_valid) + \
            1 * ((~lhs_valid) & rhs_valid) + \
            2 * (lhs_valid & (~rhs_valid)) + \
            3 * ((~lhs_valid) & (~rhs_valid))

    # However the bit math above is wrong logic for 0,1,2,3 exclusive selection.
    # Use jnp.select or nested where

    # Output vectors
    # Case 0 (v0, v2 valid) -> c2
    # Case 1 (v2 valid) -> c3
    # Case 2 (v0 valid) -> c4
    # Case 3 (none) -> c5

    # Note: State 0 covers "All valid" too.

    final_v0 = jnp.where(lhs_valid & rhs_valid, c2_v0,
                jnp.where(rhs_valid, c3_v0,
                 jnp.where(lhs_valid, c4_v0, c5_v0)))

    final_v1 = jnp.where(lhs_valid & rhs_valid, c2_v1,
                jnp.where(rhs_valid, c3_v1,
                 jnp.where(lhs_valid, c4_v1, c5_v1)))

    final_v2 = jnp.where(lhs_valid & rhs_valid, c2_v2,
                jnp.where(rhs_valid, c3_v2,
                 jnp.where(lhs_valid, c4_v2, c5_v2)))

    # Stack column-wise
    vectors = jnp.stack([final_v0, final_v1, final_v2], axis=1)

    return vectors

def eigh_3x3(A):
    lambdas = get_eigenvalues_3x3(A)
    vectors = get_eigenvectors_3x3(A, lambdas)
    return lambdas, vectors

# Verification and Tests
if __name__ == "__main__":
    import numpy as np

    def test_eigh():
        # Case 1: Random Symmetric
        key = jax.random.PRNGKey(0)
        for i in range(5):
            key, subkey = jax.random.split(key)
            A = jax.random.normal(subkey, (3, 3))
            A = A + A.T # Symmetric

            w, v = eigh_3x3(A)

            # Check reconstruction
            recon = v @ jnp.diag(w) @ v.T
            err = jnp.linalg.norm(A - recon)
            print(f"Random {i} Reconstruction Error: {err}")

            # Check orthogonality
            orth = jnp.linalg.norm(v.T @ v - jnp.eye(3))
            print(f"Random {i} Orthogonality Error: {orth}")

            # Compare with jnp.linalg.eigh
            w_jax, v_jax = jnp.linalg.eigh(A)
            print(f"Eigenvalues Diff: {jnp.linalg.norm(w - w_jax)}")

        # Case 2: Degenerate (Diagonal with repeated)
        A_deg = jnp.diag(jnp.array([1.0, 1.0, 2.0]))
        w, v = eigh_3x3(A_deg)
        print(f"Degenerate [1,1,2] Values: {w}")
        recon = v @ jnp.diag(w) @ v.T
        print(f"Degenerate [1,1,2] Error: {jnp.linalg.norm(A_deg - recon)}")

        # Case 3: Identity
        A_id = jnp.eye(3)
        w, v = eigh_3x3(A_id)
        print(f"Identity Values: {w}")
        recon = v @ jnp.diag(w) @ v.T
        print(f"Identity Error: {jnp.linalg.norm(A_id - recon)}")

        # Case 4: Triple Degenerate non-identity? (Scalar)
        A_sc = jnp.eye(3) * 5.0
        w, v = eigh_3x3(A_sc)
        print(f"Scalar [5,5,5] Error: {jnp.linalg.norm(A_sc - v @ jnp.diag(w) @ v.T)}")

        # Case 5: Covariance Matrix
        # Generate random data (N samples, 3 features)
        N = 100
        key, subkey = jax.random.split(key)
        X = jax.random.normal(subkey, (3, N))
        # jnp.cov expects (M, N) where M is variables (features), N is observations
        cov = jnp.cov(X)

        w_cov, v_cov = eigh_3x3(cov)

        # Verify against jnp.linalg.eigh
        w_jax, v_jax = jnp.linalg.eigh(cov)

        print(f"Covariance Matrix Reconstruction Error: {jnp.linalg.norm(cov - v_cov @ jnp.diag(w_cov) @ v_cov.T)}")
        print(f"Covariance Matrix Eigenvalues Diff: {jnp.linalg.norm(w_cov - w_jax)}")


    print("Running Tests...")
    test_eigh()

    # Check Lowering
    print("\nChecking HLO for custom_call...")
    A = jnp.eye(3)
    c = jax.jit(eigh_3x3).lower(A).as_text()
    if "custom_call" in c:
        print("FAIL: custom_call found in HLO!")
    else:
        print("SUCCESS: No custom_call found.")

