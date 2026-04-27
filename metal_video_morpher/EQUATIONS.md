# Spacetime-Consistent Photo Transformations — Equation Set

This document is the math reference for the kernels in `Shaders.metal`.

The pipeline composes three transformations on a video volume `I(x, y, t)`:

1. **Optical-flow advection** for temporal consistency.
2. **Tricubic Bézier free-form deformation (FFD)** for smooth volumetric warps.
3. **Spacetime tetrahedral barycentric warp** for piecewise-linear keypoint control.

All three contribute additive displacement to a single backward-warp sample:

```
sample(x, y, t) = (x, y) − D_FFD(x, y, t) − Δ_tet(x, y, t) − ∫₀ᵗ v ds
out(x, y, t)    = I_0( sample(x, y, t) )
```

---

## 1 · Optical-flow advection

For a dense flow field `v(x, t)` (pixels / frame) the advected image is the
solution to the transport PDE

```
∂I/∂t + v · ∇I = 0
```

With first-order discretization this is the Lagrangian backward warp

```
I_t(x) = I_0( x − t · v(x) )           (1)
```

Path-integrated form (more accurate, recommended for `t > a few frames`):

```
x(t) = x₀ + ∫₀ᵗ v(x(s), s) ds
I_t(x) = I_0( x − ∫₀ᵗ v(·, s) ds )      (2)
```

Discrete recurrence per frame:

```
I_{n+1}(x) = I_n( x − v_n(x) · Δt )    (3)
```

For ComfyUI-style "consistent subject" transfer, the same advection is
applied to the latent code `z` before decoding:

```
z_t(x) = z_0( x − ∫₀ᵗ v ds )           (4)
```

A temporal-consistency loss for fitting:

```
L_flow = Σ_t ‖ I_{t+1}(x + v_t) − I_t(x) ‖²
```

Implemented in `flowAdvectKernel` and as the `dFlow` term inside
`consistentMorphKernel`.

---

## 2 · Tricubic Bézier FFD (volumetric)

Embed the video in a parametric box `(u, v, w) ∈ [0, 1]³` with
`w = t / T`. Place a 4 × 4 × 4 control lattice of *displacements*
`P_{ijk} ∈ ℝ³`. The displacement field is

```
D(u, v, w) = Σ_{i=0..3} Σ_{j=0..3} Σ_{k=0..3}
                B_i³(u) · B_j³(v) · B_k³(w) · P_{ijk}     (5)
```

with cubic Bernstein basis

```
B_i³(t) = C(3, i) · tⁱ · (1 − t)^(3 − i)                   (6)
```

The warp is `(x, y, t) ↦ (x, y, t) + D(u(x), v(y), w(t))`.

Properties:

- **C² continuous** along every axis.
- **Local control**: moving one `P_{ijk}` only ripples a Bernstein-weighted
  neighborhood.
- **Affine-invariant**: warped point lies in the convex hull of the
  control lattice.

### Triquadratic variant

If you'd rather have a 3 × 3 × 3 lattice (cheaper, still smooth), swap the
cubic basis (6) for the quadratic Bernstein

```
B_i²(t) = C(2, i) · tⁱ · (1 − t)^(2 − i)                   (7)
```

`bernstein2()` in `Shaders.metal` is provided for this case.

Implemented in `ffd_tricubic()` and called from `consistentMorphKernel`.

---

## 3 · Spacetime tetrahedral barycentric warp

Treat `(x, y, t)` as a 3-vector. Given keypoint correspondences

```
{ q_α = (x_α, y_α, t_α)  →  q_α* = (x_α*, y_α*, t_α) }
```

build a 3-D Delaunay tetrahedralization (Bowyer–Watson, see `Delaunay3D.swift`).
For any query `q`, find the enclosing tet with vertices `v₁, v₂, v₃, v₄` and
solve for barycentric coordinates:

```
q = λ₁ v₁ + λ₂ v₂ + λ₃ v₃ + λ₄ v₄                          (8)
λ₁ + λ₂ + λ₃ + λ₄ = 1,    λ_i ≥ 0
```

Numerically, given `M = [v₁ − v₄ | v₂ − v₄ | v₃ − v₄]`,

```
[λ₁ λ₂ λ₃]ᵀ = M⁻¹ (q − v₄),   λ₄ = 1 − λ₁ − λ₂ − λ₃
```

(see `barycentric_3d()` in `Shaders.metal`).

The warped position is

```
q* = λ₁ v₁* + λ₂ v₂* + λ₃ v₃* + λ₄ v₄*                     (9)
```

with displacement `Δ_tet(q) = q* − q`.

This is the natural 3-D extension of the 2-D Delaunay morph already sketched
in the project's earlier shader.

---

## 4 · Composition

The composed displacement at a pixel is

```
Δ_total(x, y, t) =   α_FFD · D_FFD(u, v, w)
                   + α_tet · Δ_tet(x, y, t)
                   + α_flow · ∫₀ᵗ v ds                    (10)
```

The morph kernel scales by a global `morphStrength s`:

```
sample(x, y, t) = (x, y) − s · ( α_FFD · D_FFD + α_tet · Δ_tet )
                          − α_flow · ∫₀ᵗ v ds
out(x, y, t)    = bilinear( I_0,  sample )                (11)
```

Equation (11) is exactly what `consistentMorphKernel` computes per pixel.

---

## 5 · Joint fitting objective

Useful for optimizing the lattice and tet vertex displacements end-to-end
(e.g. to match a target style or keypoint trajectory):

```
L = L_recon
  + λ_f · L_flow                          (temporal consistency)
  + λ_s · ‖∇D_FFD‖²                        (FFD smoothness)
  + λ_t · Var( edge-length(tet) )          (avoid sliver tets)
```

`L_recon` is whatever appearance loss you want — pixel L1, perceptual
(LPIPS), or a CLIP-based subject-consistency loss to mimic ComfyUI
"consistent character" workflows.

---

## File map

| File                              | Implements                       |
| --------------------------------- | -------------------------------- |
| `Shaders.metal`                   | (1)–(11) on the GPU              |
| `BezierLattice.swift`             | 4×4×4 control lattice CPU side   |
| `Delaunay3D.swift`                | Bowyer–Watson 3-D tetrahedralizer|
| `ConsistentMorphProcessor.swift`  | Pipeline driver, flow estimation |
| `ConsistentMorphSettingsPanel.swift` | SwiftUI tunables             |
