//! Discrete-adjoint **geometry / shape** sensitivities for the complex
//! frequency-domain driven H(curl)/Nédélec solve (Epic #569, issue #577):
//! `∂(scalar EM observable)/∂(geometry parameter θ)`, finite-difference
//! validated. The hardest, highest-value gradient of the epic — the shape
//! sensitivity of a *real Maxwell observable*.
//!
//! # Where this sits
//!
//! Three prior pieces compose here:
//!
//! * [`crate::adjoint`] (#570) established the discrete-adjoint pattern on the
//!   real SPD scalar operator: factor once, transpose-solve the adjoint reusing
//!   that factorization, contract `−λᵀ(∂A/∂p) x` locally.
//! * [`crate::driven::adjoint`] (#576) carried that algebra to the **complex**
//!   driven Nédélec pencil `A(ε, ω) x = b`, `A = K − ω² M(ε)` — the Wirtinger
//!   real-`g` collapse `dg/dp = −2 Re[λᵀ(∂A/∂p) x]` with `Aᵀλ = ∂g/∂x`, and the
//!   complex-symmetric reuse of the forward LU for the transpose (adjoint)
//!   solve.
//! * [`crate::shape`] (#571) established the **geometry** counterpart on the
//!   scalar P1 operator: an exact `∂K_local/∂X` via a forward-mode `Dual`
//!   through the *same* closed-form element kernel, chained through an analytic
//!   node-motion map `θ ↦ X(θ)`.
//!
//! This module is the intersection: the **geometry** derivative of the
//! **complex driven Nédélec** solve. It mirrors [`crate::shape`]'s Dual-kernel
//! recipe for the edge-element (Whitney 1-form) curl-curl / mass / current-RHS
//! geometry factors, and reuses [`crate::driven::adjoint`]'s complex adjoint
//! λ (one forward + one adjoint solve sharing a single LU).
//!
//! # The shape adjoint identity (with a geometry-dependent RHS)
//!
//! Let the node coordinates be `X` and a geometry parameter be `θ`, with an
//! analytic node-motion map `θ ↦ X(θ)` on a **fixed mesh topology** (fixed edge
//! set, fixed PEC mask). The interior driven system is `A(X) x = b(X)` and the
//! observable is a real scalar `g(x, x̄)` with **no** explicit geometry
//! dependence. Unlike the material case (#576), where `b` is ε-independent, the
//! **current-source RHS** `b = iωμ₀ ∫ N·J dV` depends on geometry through the
//! Whitney basis and the element volume — so the shape derivative carries an
//! extra `∂b/∂X` term:
//!
//! ```text
//!   ∂x/∂X = A⁻¹ ( ∂b/∂X − (∂A/∂X) x ),
//!   dg/dX = 2 Re[ (∂g/∂x)ᵀ ∂x/∂X ]
//!         = 2 Re[ λᵀ ∂b/∂X ] − 2 Re[ λᵀ (∂A/∂X) x ],   with  Aᵀ λ = ∂g/∂x.
//! ```
//!
//! Both terms are **local** contractions, one sweep over the tets, reusing the
//! single forward LU for the adjoint (a transpose back-substitution — never a
//! refactorization). The PEC-eliminated edges carry exact zeros in both `x` and
//! `λ`, so a full-length per-tet contraction automatically restricts to the
//! interior block (the constraint `x_pec ≡ 0` is `X`-independent, so those DOFs
//! do not vary). Chaining through the node-motion Jacobian yields the design
//! gradient
//!
//! ```text
//!   ∂g/∂θ = Σ_{n,d} (∂g/∂X_{n,d}) (∂X_{n,d}/∂θ) = ⟨grad_node, ∂X/∂θ⟩,
//! ```
//!
//! evaluated by [`crate::shape::chain_node_motion`] (shared with the P1 path —
//! the chain rule is geometry-kernel-agnostic).
//!
//! # `∂A/∂X` and `∂b/∂X` are **exact** (forward-mode AD of the element kernel)
//!
//! Rather than hand-derive the (correct but error-prone) analytic Jacobian of
//! the closed-form Nédélec curl-curl / mass / current-RHS entries w.r.t. the
//! twelve tet coordinates, we evaluate the **same closed-form kernels** as
//! [`crate::elements::nedelec::batched_nedelec_local_matrices`] and
//! [`crate::elements::nedelec::batched_nedelec_local_rhs`] in dual-number
//! arithmetic (`Dual`) and read off the directional derivative. This is
//! **analytic** (exact forward-mode automatic differentiation — no
//! finite-difference truncation), so the adjoint-vs-FD test isolates the
//! correctness of the adjoint algebra + geometry chain, not the element
//! derivative. Dedicated unit tests cross-check (a) the dual `.re` against the
//! production Burn kernel and (b) the dual tangent against a central finite
//! difference of the same `f64` kernel.
//!
//! # Scope (v1): lossless real ε_r, per-tet-constant current source
//!
//! Following the issue's honesty clause, the load-bearing demonstration is a
//! **lossless, real-ε_r** driven PEC cavity with a per-tet-constant complex
//! current source (`J` held fixed per element as the mesh morphs — the natural
//! "given source density" convention, so `∂b/∂X` is purely geometric). Complex
//! ε (loss tangent) and a spatially-resampled `J(x)` source are documented
//! follow-ons.

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use faer::c64;

use crate::assembly::hcurl_space::TetOrientation;
use crate::assembly::nedelec::{
    NedelecScatterMap, assemble_global_nedelec_with_complex_epsilon_sparse,
    assemble_global_nedelec_with_full_tensors_sparse,
};
use crate::assembly::surface_p2::{
    p2_trace_dofs, tri_nedelec2_surface_flux_dual, tri_nedelec2_surface_mass_dual,
};
use crate::driven::adjoint::DrivenGradientProblem;
use crate::driven::adjoint::{AssembledPencilHost, P1ForwardAdjoint, p1_forward_adjoint};
use crate::driven::ports::{LumpedPort, assemble_port_flux, assemble_port_surface_mass};
use crate::driven::solve::{
    CurrentSource, DrivenBcs, DrivenError, DrivenMaterials, DrivenOperator,
    validate_driven_surfaces,
};
use crate::elements::ElementOrder;
use crate::elements::nedelec_p2::{TET_NEDELEC2_DOFS, TET_NEDELEC2_FACE_DOF_BASE, tet_quad_deg4};
use crate::mesh::{TET_LOCAL_EDGES, TET_LOCAL_FACES, TetMesh};

use crate::dual::{Dual, dcross3, ddot3, dlc3, dscale3, dsub3};

/// Scale a dual 3-vector by a **dual** scalar (both carry tangents).
#[inline]
fn dvscale(s: Dual, a: [Dual; 3]) -> [Dual; 3] {
    [a[0].mul(s), a[1].mul(s), a[2].mul(s)]
}
/// The first-order Nédélec element-local **curl-curl** `K`, **mass** `M`
/// (both sign-unaware, `6×6`) and **current-RHS moments** `∫ N_i dV`
/// (`6×3`, sign-unaware), all evaluated in dual arithmetic on dual-valued
/// `coords`, so each `.du` is the directional derivative w.r.t. whichever
/// coordinate was seeded with [`Dual::var`].
///
/// Mirrors [`crate::elements::nedelec::batched_nedelec_local_matrices`] and
/// [`crate::elements::nedelec::batched_nedelec_local_rhs`] entry-for-entry
/// (so the `.re` fields reproduce those real `f64` kernels):
///
/// ```text
///   e_k = v_k − v_0,   g_1 = e_2×e_3,  g_2 = e_3×e_1,  g_3 = e_1×e_2,
///   det = e_1·g_1,     g_0 = −(g_1+g_2+g_3),   gg_pq = g_p·g_q,
///
///   K_ij = (2/3)(gg_ac gg_bd − gg_ad gg_bc)/|det|³,           i=(a,b), j=(c,d)
///   M_ij = (1/120)(f_ac gg_bd − f_ad gg_bc − f_bc gg_ad + f_bd gg_ac)/|det|,
///          f_pq = 2 if p==q else 1
///   ∫N_i dV = sign(det)/24 · (g_b − g_a)
/// ```
///
/// `pub(crate)`: reused by [`crate::eigen::sensitivity`] (issue #596) for the
/// geometry half of the Hellmann–Feynman eigenvalue sensitivity — the same
/// exact `∂K_local/∂X`, `∂M_local/∂X` element JVP, contracted `xᵀ(·)x` instead
/// of the adjoint's `λᵀ(·)x`.
#[allow(clippy::type_complexity)]
pub(crate) fn nedelec_local_dual(
    coords: &[[Dual; 3]; 4],
) -> ([[Dual; 6]; 6], [[Dual; 6]; 6], [[Dual; 3]; 6]) {
    let v0 = coords[0];
    let e1 = dsub3(coords[1], v0);
    let e2 = dsub3(coords[2], v0);
    let e3 = dsub3(coords[3], v0);

    let g1 = dcross3(e2, e3);
    let g2 = dcross3(e3, e1);
    let g3 = dcross3(e1, e2);
    let det = ddot3(e1, g1); // signed 6V
    let g0 = [
        g1[0].add(g2[0]).add(g3[0]).neg(),
        g1[1].add(g2[1]).add(g3[1]).neg(),
        g1[2].add(g2[2]).add(g3[2]).neg(),
    ];
    let g = [g0, g1, g2, g3];

    // Cofactor gram gg_pq = g_p · g_q (physical gram G_pq = gg_pq/det²; the
    // det powers are folded into the K/M scale factors below, matching the
    // Burn kernel's `inv_abs_det{,3}` factoring).
    let gg = |p: usize, q: usize| -> Dual { ddot3(g[p], g[q]) };

    let abs_det = det.abs();
    let inv_abs = Dual::cst(1.0).div(abs_det); // 1/|det|
    let inv_abs3 = inv_abs.mul(inv_abs).mul(inv_abs); // 1/|det|³

    let mut k = [[Dual::cst(0.0); 6]; 6];
    let mut m = [[Dual::cst(0.0); 6]; 6];
    for (i, &(a, b)) in TET_LOCAL_EDGES.iter().enumerate() {
        for (j, &(c, d)) in TET_LOCAL_EDGES.iter().enumerate() {
            // K_ij = (2/3)(gg_ac gg_bd − gg_ad gg_bc)/|det|³
            let k_term = gg(a, c).mul(gg(b, d)).sub(gg(a, d).mul(gg(b, c)));
            k[i][j] = k_term.mul(inv_abs3).scale(2.0 / 3.0);

            // M_ij = (1/120)(f_ac gg_bd − f_ad gg_bc − f_bc gg_ad + f_bd gg_ac)/|det|
            let f_ac = if a == c { 2.0 } else { 1.0 };
            let f_ad = if a == d { 2.0 } else { 1.0 };
            let f_bc = if b == c { 2.0 } else { 1.0 };
            let f_bd = if b == d { 2.0 } else { 1.0 };
            let m_term = gg(b, d)
                .scale(f_ac)
                .sub(gg(b, c).scale(f_ad))
                .sub(gg(a, d).scale(f_bc))
                .add(gg(a, c).scale(f_bd));
            m[i][j] = m_term.mul(inv_abs).scale(1.0 / 120.0);
        }
    }

    // ∫ N_i dV = sign(det)/24 · (g_b − g_a). factor = det/|det|/24.
    let factor = det.mul(inv_abs).scale(1.0 / 24.0);
    let mut nint = [[Dual::cst(0.0); 3]; 6];
    for (i, &(a, b)) in TET_LOCAL_EDGES.iter().enumerate() {
        for c in 0..3 {
            nint[i][c] = factor.mul(g[b][c].sub(g[a][c]));
        }
    }

    (k, m, nint)
}

// ─────────────────────────────────────────────────────────────────────────
// Tensor-material (box-UPML) Dual twin of the FIRST-ORDER Nédélec element
// (issue #635, Epic #628 Phase C1).
//
// Forward-mode-AD counterpart of the full-3×3-weight forward kernels
// `crate::elements::nedelec::batched_nedelec_local_stiffness_weighted` and
// `batched_nedelec_local_mass_anisotropic_full` — the ones
// `assemble_global_nedelec_with_full_tensors{,_sparse}` (nedelec.rs:1899/2008)
// use for the matched box-UPML pencil `A = K(ν) − ω² M(ε)` with per-tet
// constitutive tensors `ν = Λ⁻¹` (curl weight) and `ε = ε_r·Λ` (mass weight).
//
// Scope C1 **pins the PML shell**: the node-motion map holds every PML-region
// node fixed, so `∂Λ/∂X = 0` and `Λ` (hence the real 3×3 weight components
// `W_k`, `W_m`) is a **constant** per-tet input here. Only the fixed-Λ geometry
// contraction — the element-Jacobian / cofactor derivatives sandwiched by the
// constant weight — is differentiated; the moving-PML centroid-profile
// derivative `∂s_k/∂centroid_k` is the explicit Phase C2 non-goal.
// ─────────────────────────────────────────────────────────────────────────

/// Contract two dual 3-vectors through a **fixed** real 3×3 weight `W`:
/// `aᵀ W b = Σ_pq a[p] W[p][q] b[q]`. `W` is a lifted constant (zero tangent),
/// so the tangent flows only through the dual vectors `a`, `b` — exactly the
/// "fixed Λ" convention of the box-UPML shape adjoint (issue #635).
#[inline]
fn dweighted3(a: [Dual; 3], w: &[[f64; 3]; 3], b: [Dual; 3]) -> Dual {
    let mut acc = Dual::cst(0.0);
    for p in 0..3 {
        // (row p of W) · b, then times a[p].
        let mut wb = Dual::cst(0.0);
        for q in 0..3 {
            wb = wb.add(b[q].scale(w[p][q]));
        }
        acc = acc.add(a[p].mul(wb));
    }
    acc
}

/// Tensor-material (box-UPML, issue #635) Dual twin of the first-order Nédélec
/// element: local **curl-curl** `K(W_k)` and **mass** `M(W_m)` (both `6×6`,
/// sign-unaware) for a **fixed** real 3×3 per-tet curl weight `W_k` (a real
/// component of `ν = Λ⁻¹`) and mass weight `W_m` (a real component of
/// `ε = ε_r·Λ`), evaluated in dual arithmetic on dual-valued `coords` — so each
/// `.du` is the exact `∂/∂X` at fixed Λ.
///
/// Mirrors [`crate::elements::nedelec::batched_nedelec_local_stiffness_weighted`]
/// and [`crate::elements::nedelec::batched_nedelec_local_mass_anisotropic_full`]
/// entry-for-entry (so the `.re` fields reproduce those real `f64` kernels), with
/// the SAME cofactor construction as [`nedelec_local_dual`]:
///
/// ```text
///   e_k = v_k − v_0,  g_1 = e_2×e_3, g_2 = e_3×e_1, g_3 = e_1×e_2,
///   det = e_1·g_1,    g_0 = −(g_1+g_2+g_3),   cr_i = g_a × g_b  (edge i=(a,b))
///
///   K(W_k)_ij = (2/3) cr_iᵀ W_k cr_j / |det|³,
///   gw_pq     = g_pᵀ W_m g_q,
///   M(W_m)_ij = (1/120)(f_ac gw_bd − f_ad gw_bc − f_bc gw_ad + f_bd gw_ac)/|det|,
///              f_pq = 2 if p==q else 1.
/// ```
///
/// A complex weight runs as two calls (real-part weight, imag-part weight),
/// exactly the Re/Im split the forward assembler uses; the caller recombines
/// `∂K = ∂K_re + i·∂K_im`, `∂M = ∂M_re + i·∂M_im`.
#[allow(clippy::type_complexity)]
pub(crate) fn nedelec_local_dual_tensor(
    coords: &[[Dual; 3]; 4],
    w_k: &[[f64; 3]; 3],
    w_m: &[[f64; 3]; 3],
) -> ([[Dual; 6]; 6], [[Dual; 6]; 6]) {
    let v0 = coords[0];
    let e1 = dsub3(coords[1], v0);
    let e2 = dsub3(coords[2], v0);
    let e3 = dsub3(coords[3], v0);

    let g1 = dcross3(e2, e3);
    let g2 = dcross3(e3, e1);
    let g3 = dcross3(e1, e2);
    let det = ddot3(e1, g1); // signed 6V
    let g0 = [
        g1[0].add(g2[0]).add(g3[0]).neg(),
        g1[1].add(g2[1]).add(g3[1]).neg(),
        g1[2].add(g2[2]).add(g3[2]).neg(),
    ];
    let g = [g0, g1, g2, g3];

    let abs_det = det.abs();
    let inv_abs = Dual::cst(1.0).div(abs_det); // 1/|det|
    let inv_abs3 = inv_abs.mul(inv_abs).mul(inv_abs); // 1/|det|³

    // Constant per-tet (unnormalized) curls cr_i = g_a × g_b; the physical curl
    // `2 cr_i / det²` folds its det powers into the K scale below, matching the
    // forward kernel's `inv_abs_det³ · (2/3)` factoring.
    let mut cr = [[Dual::cst(0.0); 3]; 6];
    for (i, &(a, b)) in TET_LOCAL_EDGES.iter().enumerate() {
        cr[i] = dcross3(g[a], g[b]);
    }

    let mut k = [[Dual::cst(0.0); 6]; 6];
    let mut m = [[Dual::cst(0.0); 6]; 6];

    // K(W_k)_ij = (2/3) cr_iᵀ W_k cr_j / |det|³.
    for i in 0..6 {
        for j in 0..6 {
            k[i][j] = dweighted3(cr[i], w_k, cr[j]).mul(inv_abs3).scale(2.0 / 3.0);
        }
    }

    // Weighted gram gw_pq = g_pᵀ W_m g_q (physical G^W = gw/det²; det² and V/20
    // collapse into 1/(120 |det|) below, matching the forward kernel).
    let gw = |p: usize, q: usize| -> Dual { dweighted3(g[p], w_m, g[q]) };
    for (i, &(a, b)) in TET_LOCAL_EDGES.iter().enumerate() {
        for (j, &(c, d)) in TET_LOCAL_EDGES.iter().enumerate() {
            let f_ac = if a == c { 2.0 } else { 1.0 };
            let f_ad = if a == d { 2.0 } else { 1.0 };
            let f_bc = if b == c { 2.0 } else { 1.0 };
            let f_bd = if b == d { 2.0 } else { 1.0 };
            let m_term = gw(b, d)
                .scale(f_ac)
                .sub(gw(b, c).scale(f_ad))
                .sub(gw(a, d).scale(f_bc))
                .add(gw(a, c).scale(f_bd));
            m[i][j] = m_term.mul(inv_abs).scale(1.0 / 120.0);
        }
    }

    (k, m)
}

// ─────────────────────────────────────────────────────────────────────────
// Surface-Whitney (flat-triangle) Dual twin — the moving-feed lumped-port
// geometry derivative (Epic #628 Phase A2, issue #633).
//
// Forward-mode-AD counterpart of the flat-triangle Whitney face kernels the
// forward port assembly uses (`crate::elements::whitney::face_geometry` +
// `face_mass_block`, and the closed-form flux in
// `crate::driven::ports::assemble_port_flux`), restricted to ONE port face and
// evaluated in `Dual` arithmetic — so each `.du` is the exact partial of the
// face's tangential surface-mass block `S^face_{ij} = ∮_T N_i·N_j dS` and its
// flux moment `f^face_k = ∮_T N_k·ê dS` w.r.t. whichever face-node coordinate
// was seeded with `Dual::var`. This is the `∂S_p/∂X` / `∂f/∂X` analog of the
// volume `nedelec_local_dual`, and is what Phase A1 (pinned feed) dropped.
//
// The 3×3 mass block and the 3-vector flux are returned UNSIGNED (local-edge
// order `TRI_LOCAL_EDGES`); the lower-tag-first orientation signs `s_k` are
// geometry-independent integers folded in by the caller (into the signed local
// λ / x), exactly as `assemble_surface_mass_triplets` / `assemble_port_flux`
// scatter `val·s_i·s_j` and `flux·s_k`.
// ─────────────────────────────────────────────────────────────────────────

/// Dual twin of [`crate::elements::whitney::face_geometry`]'s geometric outputs:
/// the triangle **area** and the three in-plane **barycentric gradients** `∇λ_k`,
/// in dual arithmetic on dual-valued triangle vertices `v`.
///
/// Mirrors the `f64` kernel entry-for-entry using the determinant-free form
/// `∇λ_k = (e₁×e₂) × opp_k / ‖e₁×e₂‖²` (identical to the production
/// `n̂ × opp_k / (2·area)` since `n̂ = (e₁×e₂)/‖e₁×e₂‖`), so the `.re` fields
/// reproduce the production gradients bit-for-bit while the `.du` fields carry
/// the exact geometry tangent.
fn port_face_geometry_dual(v: &[[Dual; 3]; 3]) -> (Dual, [[Dual; 3]; 3]) {
    let e10 = dsub3(v[1], v[0]);
    let e20 = dsub3(v[2], v[0]);
    let cross = dcross3(e10, e20);
    let two_area_sq = ddot3(cross, cross); // ‖e₁×e₂‖²
    let two_area = two_area_sq.sqrt();
    let area = two_area.scale(0.5);

    // opp_k = edge opposite local vertex k (from v_{(k+1)%3} to v_{(k+2)%3}).
    let opp = [dsub3(v[2], v[1]), dsub3(v[0], v[2]), dsub3(v[1], v[0])];
    let inv_two_area_sq = Dual::cst(1.0).div(two_area_sq);
    let grad_lambda = [
        dvscale(inv_two_area_sq, dcross3(cross, opp[0])),
        dvscale(inv_two_area_sq, dcross3(cross, opp[1])),
        dvscale(inv_two_area_sq, dcross3(cross, opp[2])),
    ];
    (area, grad_lambda)
}

/// The UNSIGNED tangential surface-mass block `∮_T N_i·N_j dS` of one face in
/// dual arithmetic, from the face's in-plane barycentric gradients and
/// `weight = area/3` (the 3-point edge-midpoint rule of `face_mass_block`).
fn face_mass_block_dual(grad_lambda: &[[Dual; 3]; 3], weight: Dual) -> [[Dual; 3]; 3] {
    // Tangential surface mass, 3-point edge-midpoint rule on the Whitney trace
    // N_e(λ) = λ_la·∇λ_lb − λ_lb·∇λ_la (BAC-CAB rank reduction on the flat face).
    let mut mass = [[Dual::cst(0.0); 3]; 3];
    for lam in crate::elements::whitney::BARYCENTRIC_MIDPOINTS.iter() {
        let mut basis_q = [[Dual::cst(0.0); 3]; 3];
        for (k, &(la, lb)) in crate::elements::whitney::TRI_LOCAL_EDGES.iter().enumerate() {
            let term_a = dscale3(lam[la], grad_lambda[lb]);
            let term_b = dscale3(lam[lb], grad_lambda[la]);
            basis_q[k] = dsub3(term_a, term_b);
        }
        for i in 0..3 {
            for j in 0..3 {
                mass[i][j] = mass[i][j].add(ddot3(basis_q[i], basis_q[j]).mul(weight));
            }
        }
    }
    mass
}

/// Dual twin of one impedance-wall face (Epic #841 Phase 2b, issue #1328): the
/// UNSIGNED `3×3` surface mass `S_Γ^face = ∮_T N_i·N_j dS` of a `Leontovich`
/// (or any `SurfaceImpedanceBc`) wall triangle in dual arithmetic. This is the
/// SAME block as the mass of [`port_face_dual`] (one shared kernel), so at zero
/// perturbation `.re` reproduces `assemble_surface_mass` and `.du` is
/// `∂S_Γ^face/∂X`. The wall coefficient `c_Γ(ω)` is geometry-independent, so
/// `∂A/∂X|_wall = c_Γ ∂S_Γ/∂X`.
pub(crate) fn wall_face_mass_dual(v: &[[Dual; 3]; 3]) -> [[Dual; 3]; 3] {
    let (area, grad_lambda) = port_face_geometry_dual(v);
    face_mass_block_dual(&grad_lambda, area.scale(1.0 / 3.0))
}

/// Dual twin of a single port face: the UNSIGNED tangential surface-mass block
/// `S^face_{ij} = ∮_T N_i·N_j dS` (`3×3`), the UNSIGNED flux moment
/// `f^face_k = ∮_T N_k·ê dS` (`3`), and the face area, all in dual arithmetic on
/// the dual-valued triangle vertices `v`.
///
/// The mass block uses the SAME 3-point edge-midpoint quadrature
/// ([`crate::elements::whitney::BARYCENTRIC_MIDPOINTS`], degree-2 exact) as
/// `face_mass_block`; the flux uses the SAME closed form
/// `∫_T N_k dA = (area/3)(∇λ_lb − ∇λ_la)` dotted with `ê` as
/// `assemble_port_flux`. At zero perturbation the `.re` fields reproduce those
/// `f64` kernels exactly (unit-tested); the `.du` fields are `∂S^face/∂X` and
/// `∂f^face/∂X`.
fn port_face_dual(v: &[[Dual; 3]; 3], e_hat: [f64; 3]) -> (Dual, [[Dual; 3]; 3], [Dual; 3]) {
    let (area, grad_lambda) = port_face_geometry_dual(v);
    let weight = area.scale(1.0 / 3.0);

    let mass = face_mass_block_dual(&grad_lambda, weight);

    // Flux moment f_k = ∮_T N_k·ê dS with ∫_T N_k dA = (area/3)(∇λ_lb − ∇λ_la).
    let e = [
        Dual::cst(e_hat[0]),
        Dual::cst(e_hat[1]),
        Dual::cst(e_hat[2]),
    ];
    let mut flux = [Dual::cst(0.0); 3];
    for (k, &(la, lb)) in crate::elements::whitney::TRI_LOCAL_EDGES.iter().enumerate() {
        let integral = dvscale(weight, dsub3(grad_lambda[lb], grad_lambda[la]));
        flux[k] = ddot3(integral, e);
    }

    (area, mass, flux)
}

// ─────────────────────────────────────────────────────────────────────────
// Dual twin of the SECOND-ORDER (p=2) 20-DOF Nédélec element (issue #619).
//
// Forward-mode-AD counterpart of the `f64` production kernels in
// `crate::elements::nedelec_p2` (`tet_barycentric_gradients` /
// `tet_nedelec2_shapes` / `tet_nedelec2_local` / `tet_nedelec2_local_rhs`),
// mirroring them entry-for-entry but in `Dual` arithmetic so each `.du` is the
// exact partial w.r.t. whichever tet coordinate was seeded with `Dual::var`.
//
// Deliberately a TWIN, not a genericization of the production kernel: the `f64`
// `tet_nedelec2_local` sits on the hot forward-assembly path
// (`assemble_p2_km` → `tet_p2_local_sorted`), so it is left byte-identical for
// the p=1 AND p=2 forward paths (the AC's "documented equivalent" clause and
// the house style established by `nedelec_local_dual` at p=1). The two element
// self-checks (`p2_dual_local_matrices_reproduce_f64_kernel`,
// `p2_dual_local_derivative_matches_finite_difference`) pin the twin's `.re`
// against the `f64` kernel and its `.du` against a central FD of the same
// kernel.
//
// The fixed reference-barycentric quadrature points `lam` from
// `tet_quad_deg4()` are constants (`f64`): geometry enters the integral ONLY
// through `bary = ∇λ` and the element volume, both differentiable outputs of
// the Dual barycentric-gradient step. There is NO ∂lam/∂X term.
// ─────────────────────────────────────────────────────────────────────────

/// Dual twin of [`crate::elements::nedelec_p2::tet_barycentric_gradients`]:
/// physical barycentric gradients `∇λ_p` and the **signed** volume `det/6`, in
/// dual arithmetic (same cofactor construction as the `f64` kernel).
fn tet_barycentric_gradients_dual(coords: &[[Dual; 3]; 4]) -> ([[Dual; 3]; 4], Dual) {
    let v0 = coords[0];
    let e1 = dsub3(coords[1], v0);
    let e2 = dsub3(coords[2], v0);
    let e3 = dsub3(coords[3], v0);
    let g1 = dcross3(e2, e3);
    let g2 = dcross3(e3, e1);
    let g3 = dcross3(e1, e2);
    let det = ddot3(e1, g1); // = 6V (signed)
    let inv = Dual::cst(1.0).div(det);
    let gl1 = [g1[0].mul(inv), g1[1].mul(inv), g1[2].mul(inv)];
    let gl2 = [g2[0].mul(inv), g2[1].mul(inv), g2[2].mul(inv)];
    let gl3 = [g3[0].mul(inv), g3[1].mul(inv), g3[2].mul(inv)];
    let gl0 = [
        gl1[0].add(gl2[0]).add(gl3[0]).neg(),
        gl1[1].add(gl2[1]).add(gl3[1]).neg(),
        gl1[2].add(gl2[2]).add(gl3[2]).neg(),
    ];
    ([gl0, gl1, gl2, gl3], det.scale(1.0 / 6.0))
}

/// Dual twin of [`crate::elements::nedelec_p2::tet_nedelec2_shapes`]: the 20
/// basis vectors `N_i` and their curls at a **fixed** reference barycentric
/// point `lam` (`f64` constants), given the (now-Dual) physical gradients
/// `bary[p] = ∇λ_p`. Mirrors the `f64` construction entry-for-entry.
#[allow(clippy::type_complexity)]
fn tet_nedelec2_shapes_dual(
    lam: &[f64; 4],
    bary: &[[Dual; 3]; 4],
) -> (
    [[Dual; 3]; TET_NEDELEC2_DOFS],
    [[Dual; 3]; TET_NEDELEC2_DOFS],
) {
    let zero = Dual::cst(0.0);
    let mut n = [[zero; 3]; TET_NEDELEC2_DOFS];
    let mut c = [[zero; 3]; TET_NEDELEC2_DOFS];

    // Edge functions: W (Whitney) and Q (gradient) per edge.
    for (e, &(a, b)) in TET_LOCAL_EDGES.iter().enumerate() {
        // W_ab = λ_a ∇λ_b − λ_b ∇λ_a,  curl = 2 ∇λ_a × ∇λ_b.
        n[2 * e] = dlc3(lam[a], bary[b], -lam[b], bary[a]);
        c[2 * e] = dscale3(2.0, dcross3(bary[a], bary[b]));
        // Q_ab = λ_a ∇λ_b + λ_b ∇λ_a = ∇(λ_a λ_b),  curl = 0.
        n[2 * e + 1] = dlc3(lam[a], bary[b], lam[b], bary[a]);
        c[2 * e + 1] = [zero; 3];
    }

    // Face functions: φ0 = λ_c W_ab, φ1 = λ_a W_bc, with (a,b,c) ascending.
    for (f, tri) in TET_LOCAL_FACES.iter().enumerate() {
        let (a, b, cc) = (tri[0], tri[1], tri[2]);
        let base = TET_NEDELEC2_FACE_DOF_BASE + 2 * f;

        // φ0 = λ_c (λ_a ∇λ_b − λ_b ∇λ_a)
        let w_ab = dlc3(lam[a], bary[b], -lam[b], bary[a]);
        n[base] = dscale3(lam[cc], w_ab);
        // curl φ0 = (λ_c ∇λ_a + λ_a ∇λ_c) × ∇λ_b − (λ_c ∇λ_b + λ_b ∇λ_c) × ∇λ_a
        {
            let g_ca = dlc3(lam[cc], bary[a], lam[a], bary[cc]);
            let g_cb = dlc3(lam[cc], bary[b], lam[b], bary[cc]);
            c[base] = dsub3(dcross3(g_ca, bary[b]), dcross3(g_cb, bary[a]));
        }

        // φ1 = λ_a (λ_b ∇λ_c − λ_c ∇λ_b)
        let w_bc = dlc3(lam[b], bary[cc], -lam[cc], bary[b]);
        n[base + 1] = dscale3(lam[a], w_bc);
        // curl φ1 = (λ_a ∇λ_b + λ_b ∇λ_a) × ∇λ_c − (λ_a ∇λ_c + λ_c ∇λ_a) × ∇λ_b
        {
            let g_ab = dlc3(lam[a], bary[b], lam[b], bary[a]);
            let g_ac = dlc3(lam[a], bary[cc], lam[cc], bary[a]);
            c[base + 1] = dsub3(dcross3(g_ab, bary[cc]), dcross3(g_ac, bary[b]));
        }
    }

    (n, c)
}

/// Dual twin of the second-order Nédélec element: local curl-curl `K` (`20×20`),
/// mass `M` (`20×20`, ε = 1), and current-RHS moments `∫ N_i dV` (`20×3`), all
/// in dual arithmetic on dual-valued `coords` — so each `.du` is the exact
/// directional derivative w.r.t. the seeded coordinate.
///
/// Mirrors [`crate::elements::nedelec_p2::tet_nedelec2_local`] and
/// [`crate::elements::nedelec_p2::tet_nedelec2_local_rhs`]: it accumulates over
/// the SAME fixed [`tet_quad_deg4`] rule with weight `|signed_vol| · frac`. The
/// RHS moment is returned per-DOF as the `[20][3]` tensor `∫ N_i dV` (contract
/// with a fixed `J` to recover the RHS entry `∫ N_i · J dV`), so the caller can
/// use the same tangents for any held-fixed complex `J`.
#[allow(clippy::type_complexity)]
pub(crate) fn nedelec2_local_dual(
    coords: &[[Dual; 3]; 4],
) -> (
    [[Dual; TET_NEDELEC2_DOFS]; TET_NEDELEC2_DOFS],
    [[Dual; TET_NEDELEC2_DOFS]; TET_NEDELEC2_DOFS],
    [[Dual; 3]; TET_NEDELEC2_DOFS],
) {
    let (bary, signed_vol) = tet_barycentric_gradients_dual(coords);
    let vol_abs = signed_vol.abs();

    let zero = Dual::cst(0.0);
    let mut k = [[zero; TET_NEDELEC2_DOFS]; TET_NEDELEC2_DOFS];
    let mut m = [[zero; TET_NEDELEC2_DOFS]; TET_NEDELEC2_DOFS];
    let mut nint = [[zero; 3]; TET_NEDELEC2_DOFS];

    for (lam, frac) in tet_quad_deg4() {
        let w = vol_abs.scale(frac);
        let (n, c) = tet_nedelec2_shapes_dual(&lam, &bary);
        for i in 0..TET_NEDELEC2_DOFS {
            for j in 0..TET_NEDELEC2_DOFS {
                m[i][j] = m[i][j].add(ddot3(n[i], n[j]).mul(w));
                k[i][j] = k[i][j].add(ddot3(c[i], c[j]).mul(w));
            }
            for d in 0..3 {
                nint[i][d] = nint[i][d].add(n[i][d].mul(w));
            }
        }
    }

    (k, m, nint)
}

// ─────────────────────────────────────────────────────────────────────────
// Shape-gradient driver.
// ─────────────────────────────────────────────────────────────────────────

/// Result of a driven-Nédélec **geometry** discrete-adjoint gradient
/// evaluation.
#[derive(Debug, Clone)]
pub struct DrivenShapeGradient {
    /// The scalar objective value `g(x)` at the (unperturbed) forward solution.
    pub objective: f64,
    /// The full **nodal-coordinate** gradient `∂g/∂X_{n,d}`, one `[x,y,z]`
    /// triple per node (length `mesh.n_nodes()`). Chain it through a
    /// node-motion map with [`crate::shape::chain_node_motion`] to obtain
    /// `∂g/∂θ`.
    pub grad_node: Vec<[f64; 3]>,
    /// Full-length `[n_edges]` complex forward edge field `x` (PEC-eliminated
    /// edges carry exact zeros), returned for post-processing / cross-checks.
    pub e_edges: Vec<c64>,
    /// Relative residual `‖A x − b‖₂ / ‖b‖₂` of the interior forward solve —
    /// a numerical health check (round-off floor for a healthy direct solve).
    pub residual_rel: f64,
    /// Number of sparse LU **factorizations** performed. Always `1`: the
    /// forward and adjoint solves share a single factorization (the adjoint is
    /// a transpose back-substitution, not a refactorization).
    pub n_factorizations: usize,
}

/// Real-ε_r convenience wrapper over [`driven_shape_gradient_complex`].
///
/// This is the original **lossless** entry point (issue #577): the per-tet
/// permittivity is a real `ε_r`, promoted to `ε_r − i·0` and forwarded to the
/// complex core. The complex path (issue #629, Epic #628 Phase B) adds
/// substrate loss (`ε = ε′ − i·ε″`, nonzero `tan δ`); the two share **one**
/// implementation, so the real gradient returned here is bit-for-bit the
/// complex core evaluated at zero loss. See [`driven_shape_gradient_complex`]
/// for the full contract, the adjoint identity, and the `∂b/∂X` term.
///
/// # Arguments
///
/// * `eps_r` — per-tet **real** relative permittivity (length `mesh.n_tets()`),
///   the evaluated material at which the gradient is taken (held constant under
///   the geometry perturbation). Every other argument matches
///   [`driven_shape_gradient_complex`].
///
/// # Errors
///
/// Propagates [`DrivenError`] on input-shape mismatches or if the sparse
/// factorization / solve fails (e.g. `ω²` collides with a lossless-pencil
/// eigenvalue, making `A(ω)` singular), and on an objective-cotangent length
/// mismatch.
pub fn driven_shape_gradient<B, G>(
    mesh: &TetMesh,
    eps_r: &[f64],
    bcs: &DrivenBcs<'_>,
    omega: f64,
    source: &CurrentSource,
    objective: G,
    device: &B::Device,
) -> Result<DrivenShapeGradient, DrivenError>
where
    B: Backend,
    G: Fn(&[c64]) -> (f64, Vec<c64>),
{
    let eps_complex: Vec<c64> = eps_r.iter().map(|&e| c64::new(e, 0.0)).collect();
    driven_shape_gradient_complex::<B, G>(mesh, &eps_complex, bcs, omega, source, objective, device)
}

/// Compute the full nodal-coordinate gradient `∂g/∂X_{n,d}` of a **complex-ε
/// (lossy) driven Nédélec** EM observable via the discrete adjoint — **one
/// forward + one adjoint solve**, reusing a single complex sparse LU
/// factorization — then chain through any analytic node-motion map with
/// [`crate::shape::chain_node_motion`].
///
/// This is the shape-side twin of the complex **material** adjoint
/// [`crate::driven::adjoint::driven_material_adjoint_gradient_complex`] (#576):
/// the geometry shape gradient of `A(X) x = b(X)`, `A = K − ω² M(ε)`, with a
/// per-tet **complex** permittivity `ε = ε′ − i·ε″` (nonzero `Im(ε)` models a
/// substrate loss tangent `tan δ`) and a per-tet-constant complex current source
/// held fixed as the mesh morphs. See the module docs for the identity
/// (including the geometry-dependent-RHS `∂b/∂X` term).
///
/// The `A(ω) = K − ω² M(ε)` pencil stays **complex-symmetric** under loss
/// (`M(ε)` scales the real symmetric element mass by the complex per-tet `ε`),
/// so the adjoint `Aᵀ λ = ∂g/∂x` still reuses the forward LU — one
/// factorization, two back-substitutions (`n_factorizations == 1`). The only
/// change from the lossless path is that the volume contraction factor
/// `∂A_ij = ∂K_ij − ω² ε ∂M_ij` now carries the complex `ε` (the field `x`,
/// adjoint `λ`, and `term_a` were already complex).
///
/// # Arguments
///
/// * `mesh` — tetrahedral mesh (fixed topology; the gradient is w.r.t. its node
///   positions).
/// * `eps_r` — per-tet **complex** relative permittivity `ε = ε′ − i·ε″`
///   (length `mesh.n_tets()`), the evaluated material at which the gradient is
///   taken (held constant under the geometry perturbation). Use
///   [`driven_shape_gradient`] for the real-ε convenience path.
/// * `bcs` — PEC interior-edge mask, exactly as
///   [`crate::driven::solve::driven_solve`] takes it.
/// * `omega` — drive frequency `ω = k₀` (natural units). Must sit away from a
///   resonance of the pencil so `A(ω)` is non-singular.
/// * `source` — per-tet-constant complex volumetric current source. Its `j_tet`
///   values are held **fixed per element** under the geometry perturbation, so
///   `∂b/∂X` is purely the geometric variation of `∫ N·J dV`.
/// * `objective` — the scalar figure-of-merit. Given the full-length complex
///   edge field `x` (`[n_edges]`, PEC zeros in place) it returns `(g, ∂g/∂x)`
///   with `∂g/∂x` a full-length `[n_edges]` **Wirtinger** cotangent
///   (`∂g/∂x_i`, un-conjugated; e.g. `x̄_i` for `g = Σ|x_i|²`). Must be real and
///   depend on geometry only through `x`; its cotangent on PEC edges is ignored.
///
/// # Errors
///
/// Propagates [`DrivenError`] on input-shape mismatches or if the sparse
/// factorization / solve fails (e.g. `ω²` collides with a pencil eigenvalue,
/// making `A(ω)` singular), and on an objective-cotangent length mismatch.
pub fn driven_shape_gradient_complex<B, G>(
    mesh: &TetMesh,
    eps_r: &[c64],
    bcs: &DrivenBcs<'_>,
    omega: f64,
    source: &CurrentSource,
    objective: G,
    device: &B::Device,
) -> Result<DrivenShapeGradient, DrivenError>
where
    B: Backend,
    G: Fn(&[c64]) -> (f64, Vec<c64>),
{
    // Port-less delegation: the shared core with an empty port list is the
    // historical (Phase B) complex path, bit-for-bit. See the module docs and
    // [`driven_shape_gradient_ports_complex`].
    driven_shape_gradient_ports_complex::<B, G>(
        mesh,
        eps_r,
        bcs,
        omega,
        source,
        &[],
        objective,
        device,
    )
}

/// Complex-ε (lossy) driven Nédélec **shape** gradient of a **port-terminated**
/// pencil — Epic #628 **Phase A1**, the pinned-feed lumped-port termination in
/// the shape adjoint (issue #631).
///
/// This is [`driven_shape_gradient_complex`] with a slice of **pinned** Palace-
/// style lumped ports ([`LumpedPort`]) threaded into the differentiated system.
/// Each port contributes, exactly as the forward
/// [`crate::driven::solve::driven_solve_with_ports`] does,
///
/// ```text
///   A(ω) += (jω/Z_s) S_p,                    Z_s = R·w/l,  S_p = ∮ N_i·N_j dS
///   b_i  += (2jω/Z_s)(V_inc/l) ∮ N_i·ê dS    (only if V_inc ≠ 0),
/// ```
///
/// so the adjoint now differentiates the **port-loaded** pencil
/// `A = K − ω² M(ε) + (jω/Z_s) S_p` with the port boundary drive folded into
/// the RHS `b`. Because `S_p` is real-symmetric and scaled by the scalar
/// `jω/Z_s`, `A(ω)ᵀ = A(ω)` is preserved — the transpose (adjoint) solve reuses
/// the single forward LU (`n_factorizations == 1`), exactly as in the port-less
/// path.
///
/// # Pinned feed — why A1 is tractable
///
/// The port faces / nodes are **geometry-constant** (the feed is *pinned*), so
/// `∂S_p/∂X = 0` and `∂b_port/∂X = 0`: there is **no** new `∂A/∂X` or `∂b/∂X`
/// port term. The constant port term simply loads the forward + adjoint solves,
/// and the volume-term (K, M) shape gradient is then taken through the loaded
/// system — the returned gradient genuinely differs from the port-less path
/// (the field `x`, adjoint `λ`, and the `∂b/∂X` volume RHS all see the port).
/// **This assumption requires that the node-motion map hold every port-face
/// node fixed** (a moving feed — `∂S_p/∂X ≠ 0`, `∂b_port/∂X ≠ 0` — is Phase A2,
/// a documented non-goal here).
///
/// An empty `ports` slice reproduces [`driven_shape_gradient_complex`]
/// bit-for-bit. See that function for the full argument contract; `ports` is
/// the added parameter.
#[allow(clippy::too_many_arguments)]
pub fn driven_shape_gradient_ports_complex<B, G>(
    mesh: &TetMesh,
    eps_r: &[c64],
    bcs: &DrivenBcs<'_>,
    omega: f64,
    source: &CurrentSource,
    ports: &[LumpedPort<'_>],
    objective: G,
    device: &B::Device,
) -> Result<DrivenShapeGradient, DrivenError>
where
    B: Backend,
    G: Fn(&[c64]) -> (f64, Vec<c64>),
{
    let core = driven_shape_gradient_ports_complex_core::<B, G>(
        mesh, eps_r, bcs, omega, source, ports, objective, device,
    )?;
    Ok(DrivenShapeGradient {
        objective: core.objective,
        grad_node: core.grad_node,
        e_edges: core.e_edges,
        residual_rel: core.residual_rel,
        n_factorizations: core.n_factorizations,
    })
}

/// Internal result of the port-loaded shape-gradient core: the public
/// [`DrivenShapeGradient`] fields **plus** the full-length adjoint field
/// `lambda_full` (zero on PEC edges), which the moving-feed sibling
/// [`driven_shape_gradient_moving_port_s11`] needs to contract the extra
/// `∂S_p/∂X`, `∂b_port/∂X`, `∂Z_s/∂X` port terms — WITHOUT a second
/// factorization (the adjoint `λ` is already computed here off the single LU).
struct PortShapeCore {
    objective: f64,
    grad_node: Vec<[f64; 3]>,
    e_edges: Vec<c64>,
    lambda_full: Vec<c64>,
    residual_rel: f64,
    n_factorizations: usize,
}

/// One seeded coordinate's local element tangents for the shape contraction, as
/// supplied by a material path's own `Dual` kernel call(s).
///
/// Each material path computes `da` with its **own** arithmetic, in its own
/// operation order — that is the whole point of the seam (issue #1596): routing
/// the scalar path through the tensor kernels with `ε·I`, `ν = I` would change
/// rounding and break the documented bit-for-bit properties.
struct LocalShapeTangents {
    /// `∂A_loc[i][j]/∂X_c = ∂K_loc[i][j]/∂X_c − ω² ∂M_loc[i][j]/∂X_c`, already
    /// material-weighted, for the single seeded coordinate.
    da: [[c64; 6]; 6],
    /// The **material-independent** current-RHS moment tangents
    /// `∂(∫N_i)/∂X_c` as dual numbers (the `.du` field carries the tangent).
    dnint: [[Dual; 3]; 6],
}

/// The shared core behind **both** port-loaded shape-adjoint paths — the scalar
/// complex-ε [`driven_shape_gradient_ports_complex_core`] and the matched
/// box-UPML tensor [`matched_upml_ports_core`].
///
/// It validates the masks / ports, assembles the port-loaded complex-symmetric
/// pencil, builds the PEC interior remap, factors `A(ω)` **once**, runs the
/// single forward solve plus a residual health check, evaluates the objective,
/// transpose-solves the adjoint off the *same* LU, scatters `λ`, and
/// accumulates the **volume** (`K`, `M`, volume-`∂b`) per-tet `Dual`
/// contraction into `grad_node` — the pinned-feed Phase A1 computation. It
/// additionally returns the adjoint field `lambda_full`, so the moving-feed
/// Phase A2 entry points can add the port `∂A/∂X`, `∂b_port/∂X` and objective
/// `∂f/∂X` terms on top, reusing the SAME `λ` (no refactorization). The public
/// A1 wrappers drop `lambda_full` and are therefore bit-for-bit unchanged.
///
/// # The parameterization seam (issue #1596)
///
/// This body used to exist as two near-verbatim ~380-line copies, differing in
/// only three places. Those three are now injected, so each path keeps its own
/// floating-point operation order:
///
/// * `validate_materials(n_tets)` — the path's own per-tet material length
///   check(s), invoked at exactly the point in the sequence the two
///   hand-written cores invoked them (after the PEC-mask check, before the
///   source check), so the error **precedence** is unchanged.
/// * `assemble_pencil(nodes, tets, tet_sign, scatter)` — the path's own global
///   Nédélec assembly call, returning the pattern-indexed host blocks of `K`
///   and `M` as an [`AssembledPencilHost`].
/// * `local_tangents(tet, coords)` — the path's own per-tet, per-seeded-
///   coordinate element tangent `∂A_loc` (plus the shared `∂∫N`), as a
///   [`LocalShapeTangents`].
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn ports_shape_core<B, G, V, A, D>(
    mesh: &TetMesh,
    bcs: &DrivenBcs<'_>,
    ports: &[LumpedPort<'_>],
    omega: f64,
    source: &CurrentSource,
    objective: G,
    device: &B::Device,
    validate_materials: V,
    assemble_pencil: A,
    local_tangents: D,
) -> Result<PortShapeCore, DrivenError>
where
    B: Backend,
    G: Fn(&[c64]) -> (f64, Vec<c64>),
    V: FnOnce(usize) -> Result<(), DrivenError>,
    A: FnOnce(
        Tensor<B, 2>,
        Tensor<B, 2, Int>,
        &[[i8; 6]],
        &NedelecScatterMap,
    ) -> AssembledPencilHost,
    D: Fn(usize, &[[Dual; 3]; 4]) -> LocalShapeTangents,
{
    let n_nodes = mesh.n_nodes();
    let n_edges = mesh.edges().len();

    // The shared p=1 forward + adjoint pipeline (validation, assembly, one
    // factorization, forward solve, objective, transpose solve). The seam
    // closures keep each material path's own operation order and error
    // precedence (issue #1596); this path has no impedance walls.
    let fa = p1_forward_adjoint::<B, _, _, _, _>(
        mesh,
        bcs,
        omega,
        source,
        ports,
        &[],
        objective,
        device,
        validate_materials,
        |_| Ok(()),
        assemble_pencil,
    )?;
    let P1ForwardAdjoint {
        objective: objective_value,
        e_edges,
        lambda_int,
        remap,
        tet_idx,
        tet_sign,
        residual_rel,
        n_factorizations,
        ..
    } = fa;

    // λ scattered to full edge length, zero on PEC edges.
    let mut lambda_full = vec![c64::new(0.0, 0.0); n_edges];
    for (full_idx, &ri) in remap.iter().enumerate() {
        if ri >= 0 {
            lambda_full[full_idx] = lambda_int[ri as usize];
        }
    }

    // --- Nodal-coordinate gradient. -----------------------------------------
    // ∂g/∂X_{n,d} = 2 Re[ λᵀ ∂b/∂X_{n,d} ] − 2 Re[ λᵀ (∂A/∂X_{n,d}) x ], a
    // purely local per-tet contraction. For each tet we seed each of its 12
    // local coordinates through the path's dual Nédélec kernel(s) and read the
    // tangents (∂A = ∂K − ω²∂M, and ∂∫N) into the signed complex contraction.
    // Signs `s_i` fold into the local λ / x (matrix path: s_i s_j; RHS path:
    // single s_i), and PEC edges carry zeros so only the interior block
    // survives.
    let mut grad_node = vec![[0.0_f64; 3]; n_nodes];
    let iomega = c64::new(0.0, omega);
    for (t, tet) in mesh.tets.iter().enumerate() {
        let gidx = &tet_idx[t];
        let gsign = &tet_sign[t];

        // Signed complex local adjoint λ and forward x (per-DOF sign s_i).
        let lam_loc: [c64; 6] =
            std::array::from_fn(|i| lambda_full[gidx[i] as usize] * (gsign[i] as f64));
        // Skip a tet whose adjoint vanishes on all six local edges (its local
        // A/b never couples into the objective).
        if lam_loc.iter().all(|z| z.re == 0.0 && z.im == 0.0) {
            continue;
        }
        let x_loc: [c64; 6] =
            std::array::from_fn(|i| e_edges[gidx[i] as usize] * (gsign[i] as f64));

        let base = [
            mesh.nodes[tet[0] as usize],
            mesh.nodes[tet[1] as usize],
            mesh.nodes[tet[2] as usize],
            mesh.nodes[tet[3] as usize],
        ];
        let jt = source.j_tet[t]; // [c64; 3], held fixed as the mesh morphs

        for a in 0..4 {
            let node = tet[a] as usize;
            for c_axis in 0..3 {
                // Seed local vertex a, axis c_axis; all other coords constant.
                let mut dc = base.map(|v| v.map(Dual::cst));
                dc[a][c_axis] = Dual::var(base[a][c_axis]);
                let LocalShapeTangents { da, dnint } = local_tangents(t, &dc);

                // −λᵀ (∂A/∂X) x, with the path's own ∂A_ij already weighted by
                // its material (complex scalar ε, or the ν / ε tensors).
                let mut term_a = c64::new(0.0, 0.0);
                for i in 0..6 {
                    for j in 0..6 {
                        term_a += lam_loc[i] * x_loc[j] * da[i][j];
                    }
                }

                // +λᵀ ∂b/∂X, ∂b_i = iω · (∂(∫N_i)·J), J = jt held fixed.
                let mut term_b = c64::new(0.0, 0.0);
                for i in 0..6 {
                    let mut dnj = c64::new(0.0, 0.0);
                    for c in 0..3 {
                        dnj += jt[c] * dnint[i][c].du;
                    }
                    term_b += lam_loc[i] * (iomega * dnj);
                }

                grad_node[node][c_axis] += 2.0 * (term_b - term_a).re;
            }
        }
    }

    Ok(PortShapeCore {
        objective: objective_value,
        grad_node,
        e_edges,
        lambda_full,
        residual_rel,
        n_factorizations,
    })
}

/// The scalar complex-ε core of [`driven_shape_gradient_ports_complex`] and
/// [`driven_shape_gradient_moving_port_s11`]: a thin adapter that supplies the
/// scalar path's material validation, its `K(real) / M(ε)` assembly call and
/// its per-tet `∂A_loc = ∂K − ω² ε_t ∂M` kernel to the shared
/// [`ports_shape_core`].
///
/// The local arithmetic stays here, in this path's own operation order, so the
/// bit-for-bit properties documented on the public wrappers are preserved.
#[allow(clippy::too_many_arguments)]
fn driven_shape_gradient_ports_complex_core<B, G>(
    mesh: &TetMesh,
    eps_r: &[c64],
    bcs: &DrivenBcs<'_>,
    omega: f64,
    source: &CurrentSource,
    ports: &[LumpedPort<'_>],
    objective: G,
    device: &B::Device,
) -> Result<PortShapeCore, DrivenError>
where
    B: Backend,
    G: Fn(&[c64]) -> (f64, Vec<c64>),
{
    let omega2 = omega * omega;
    ports_shape_core::<B, G, _, _, _>(
        mesh,
        bcs,
        ports,
        omega,
        source,
        objective,
        device,
        // Scalar ε: one per-tet complex permittivity.
        |n_tets| {
            if eps_r.len() != n_tets {
                return Err(DrivenError::MaterialDimMismatch {
                    got: eps_r.len(),
                    want: n_tets,
                });
            }
            Ok(())
        },
        // K and M(ε) with a complex ε = ε′ − i·ε″; K itself is real, so k_im is
        // an exact-zero array and the shared pencil entry reproduces the former
        // hand-written `c64::new(k_re, 0.0)` bit-for-bit.
        |nodes_t, tets_t, tet_sign, scatter| {
            let sys = assemble_global_nedelec_with_complex_epsilon_sparse(
                nodes_t, tets_t, tet_sign, scatter, eps_r,
            );
            let k_re: Vec<f64> = sys.k_vals.into_data().iter::<f64>().collect();
            AssembledPencilHost {
                k_re,
                k_im: None,
                m_re: sys.m_re_vals.into_data().iter::<f64>().collect(),
                m_im: sys.m_im_vals.into_data().iter::<f64>().collect(),
            }
        },
        // ∂A_ij = ∂K_ij − ω² ε_t ∂M_ij. ε_t is the complex per-tet permittivity
        // (geometry-independent); the element tangents ∂K/∂X, ∂M/∂X are real,
        // so the mass factor carries the loss and ∂A is complex.
        |t, dc| {
            let eps_t = eps_r[t];
            let (dk, dm, dnint) = nedelec_local_dual(dc);
            LocalShapeTangents {
                da: std::array::from_fn(|i| {
                    std::array::from_fn(|j| {
                        c64::new(dk[i][j].du, 0.0) - eps_t * (omega2 * dm[i][j].du)
                    })
                }),
                dnint,
            }
        },
    )
}

/// Accumulate the **moving-feed** lumped-port terms of an `|S₁₁|²` objective
/// into a nodal-coordinate gradient (Epic #628 Phase A2, issue #633): the port
/// admittance `∂S_p/∂X` (with `∂Z_s/∂X`), the port drive `∂b_port/∂X`, and the
/// explicit objective geometry term `∂g/∂X|_x` (the port voltage reads the
/// geometry through the flux moments `f` and `1/w`). `e_edges` and
/// `lambda_full` are the forward field and the adjoint of the port-loaded
/// pencil (full edge length, zero on PEC edges); `flux` is
/// [`assemble_port_flux`] of `port` on `mesh`. The terms are a post-solve
/// contraction against the same `λ`, so the caller's single factorization is
/// preserved. Shared by [`driven_shape_gradient_moving_port_s11`] (scalar
/// complex `ε`) and [`driven_shape_gradient_matched_upml_moving_port_s11`]
/// (matched box-UPML tensors, issue #1120): the port terms do not depend on
/// the volume material.
#[allow(clippy::too_many_arguments)]
fn accumulate_moving_port_terms(
    mesh: &TetMesh,
    port: &LumpedPort<'_>,
    z0: f64,
    omega: f64,
    edges: &[[u32; 2]],
    flux: &[f64],
    e_edges: &[c64],
    lambda_full: &[c64],
    grad_node: &mut [[f64; 3]],
) {
    use crate::driven::extraction::s11_sq_and_dg_dv;
    use crate::elements::whitney::{edge_lookup, face_geometry};

    let n_nodes = mesh.n_nodes();
    let inv_width = 1.0 / port.width;

    // --- Base scalars (all expressed through geometric `area`, `l`, `f`, S_p) --
    let l = port.length;
    let w = port.width;
    let area = w * l; // = Σ face areas (w = area/l), consistent with the forward
    let inv_area = 1.0 / area;
    let inv_area2 = inv_area * inv_area;
    let z_s = port.surface_impedance(); // R·w/l = R·area/l²
    let alpha = c64::new(0.0, omega / z_s); // jω/Z_s   (A_port = α S_p)
    let e_inc = port.v_inc * (1.0 / l);
    let beta = c64::new(0.0, 2.0 * omega / z_s) * e_inc; // (2jω/Z_s)(V_inc/l)

    // Port voltage V = (1/w) Σ f_i x_i = (l/area) P, its |S₁₁|² and ∂g/∂V.
    let p_scalar: c64 = flux
        .iter()
        .zip(e_edges.iter())
        .map(|(&f, &x)| x * f)
        .fold(c64::new(0.0, 0.0), |acc, z| acc + z);
    let v_port = p_scalar * inv_width;
    let (_g, dg_dv) = s11_sq_and_dg_dv(v_port, port.v_inc, port.resistance, z0);

    // Geometry-independent scalar functionals contracted against λ / x:
    //   Q_S = λᵀ S_p x,   Q_f = λᵀ f   (S_p triplets already carry the signs).
    let mut q_s = c64::new(0.0, 0.0);
    for (r, c, val) in assemble_port_surface_mass(mesh, port.faces, edges) {
        q_s += lambda_full[r] * val * e_edges[c];
    }
    let mut q_f = c64::new(0.0, 0.0);
    for (&f, &lam) in flux.iter().zip(lambda_full.iter()) {
        q_f += lam * f;
    }

    // --- Per-node/axis geometry tangents from the surface-Whitney Dual twin ----
    // d_area:  ∂(Σ face areas)/∂X
    // d_l:     ∂l/∂X  (symmetric subgradient of hi − lo, filled below)
    // dS_c:    λᵀ (∂S_p/∂X) x        df_l: λᵀ (∂f/∂X)        dP_c: Σ (∂f_i/∂X) x_i
    let mut d_area = vec![[0.0_f64; 3]; n_nodes];
    let mut d_l = vec![[0.0_f64; 3]; n_nodes];
    let mut ds_c = vec![[c64::new(0.0, 0.0); 3]; n_nodes];
    let mut df_l = vec![[c64::new(0.0, 0.0); 3]; n_nodes];
    let mut dp_c = vec![[c64::new(0.0, 0.0); 3]; n_nodes];

    let lookup = edge_lookup(edges);
    for tri in port.faces {
        let v_real: [[f64; 3]; 3] = [
            mesh.nodes[tri[0] as usize],
            mesh.nodes[tri[1] as usize],
            mesh.nodes[tri[2] as usize],
        ];
        // Geometry-independent global edge index + lower-tag-first sign.
        let edge_info = face_geometry(tri, &v_real, &lookup).edge_info;
        let lam_loc: [c64; 3] =
            std::array::from_fn(|k| lambda_full[edge_info[k].0 as usize] * (edge_info[k].1 as f64));
        let x_loc: [c64; 3] =
            std::array::from_fn(|k| e_edges[edge_info[k].0 as usize] * (edge_info[k].1 as f64));

        for a in 0..3 {
            let node = tri[a] as usize;
            for c in 0..3 {
                let mut vd = v_real.map(|p| p.map(Dual::cst));
                vd[a][c] = Dual::var(v_real[a][c]);
                let (area_d, mass_d, flux_d) = port_face_dual(&vd, port.e_hat);

                d_area[node][c] += area_d.du;

                let mut ds = c64::new(0.0, 0.0);
                for i in 0..3 {
                    for j in 0..3 {
                        ds += lam_loc[i] * x_loc[j] * mass_d[i][j].du;
                    }
                }
                ds_c[node][c] += ds;

                let mut dfl = c64::new(0.0, 0.0);
                let mut dp = c64::new(0.0, 0.0);
                for i in 0..3 {
                    dfl += lam_loc[i] * flux_d[i].du;
                    dp += x_loc[i] * flux_d[i].du;
                }
                df_l[node][c] += dfl;
                dp_c[node][c] += dp;
            }
        }
    }

    // ∂l/∂X: l = hi − lo, hi = max_n (p_n·ê), lo = min over the port nodes. Use
    // the symmetric subgradient at ties (±ê split equally over the extremum set).
    let mut port_nodes: Vec<usize> = port
        .faces
        .iter()
        .flat_map(|f| f.iter().map(|&n| n as usize))
        .collect();
    port_nodes.sort_unstable();
    port_nodes.dedup();
    let along = |n: usize| -> f64 {
        let p = mesh.nodes[n];
        p[0] * port.e_hat[0] + p[1] * port.e_hat[1] + p[2] * port.e_hat[2]
    };
    let hi = port_nodes
        .iter()
        .map(|&n| along(n))
        .fold(f64::NEG_INFINITY, f64::max);
    let lo = port_nodes
        .iter()
        .map(|&n| along(n))
        .fold(f64::INFINITY, f64::min);
    let tie = 1e-9 * (hi - lo).abs().max(1.0);
    let argmax: Vec<usize> = port_nodes
        .iter()
        .copied()
        .filter(|&n| (along(n) - hi).abs() <= tie)
        .collect();
    let argmin: Vec<usize> = port_nodes
        .iter()
        .copied()
        .filter(|&n| (along(n) - lo).abs() <= tie)
        .collect();
    let inv_kmax = 1.0 / argmax.len() as f64;
    let inv_kmin = 1.0 / argmin.len() as f64;
    for &n in &argmax {
        for (slot, &eh) in d_l[n].iter_mut().zip(port.e_hat.iter()) {
            *slot += eh * inv_kmax;
        }
    }
    for &n in &argmin {
        for (slot, &eh) in d_l[n].iter_mut().zip(port.e_hat.iter()) {
            *slot -= eh * inv_kmin;
        }
    }

    // --- Accumulate the A2 port + objective terms into grad_node --------------
    // dg/dX += 2Re[ Q_f ∂β + β·(λᵀ∂f) ]        (port ∂b_port/∂X)
    //        − 2Re[ Q_S ∂α + α·(λᵀ∂S_p x) ]    (port ∂A_port/∂X, incl. ∂Z_s/∂X)
    //        + 2Re[ (∂g/∂V)·∂V|_x ]            (objective ∂f/∂X + ∂(1/w)/∂X)
    let jw_over_r = c64::new(0.0, omega / port.resistance); // jω/R
    let two_jw_vinc_over_r = c64::new(0.0, 2.0 * omega / port.resistance) * port.v_inc;
    for node in 0..n_nodes {
        for c in 0..3 {
            let da = d_area[node][c];
            let dl = d_l[node][c];
            if da == 0.0
                && dl == 0.0
                && ds_c[node][c] == c64::new(0.0, 0.0)
                && df_l[node][c] == c64::new(0.0, 0.0)
                && dp_c[node][c] == c64::new(0.0, 0.0)
            {
                continue;
            }
            // ∂(l/area) = ∂l/area − l ∂area/area²  (real).
            let dloa = dl * inv_area - l * da * inv_area2;
            // ∂α = jω/R · ∂(l²/area) = jω/R (2l ∂l/area − l² ∂area/area²).
            let d_alpha = jw_over_r * (2.0 * l * dl * inv_area - l * l * da * inv_area2);
            // ∂β = 2jωV_inc/R · ∂(l/area).
            let d_beta = two_jw_vinc_over_r * dloa;

            let term_b = q_f * d_beta + beta * df_l[node][c];
            let term_a = q_s * d_alpha + alpha * ds_c[node][c];
            // ∂V|_x = ∂(l/area)·P + (l/area)·Σ (∂f_i) x_i.
            let dv = p_scalar * dloa + dp_c[node][c] * (l * inv_area);
            let term_obj = dg_dv * dv;

            grad_node[node][c] += 2.0 * term_b.re - 2.0 * term_a.re + 2.0 * term_obj.re;
        }
    }
}

/// Complex-ε (lossy) driven Nédélec **shape** gradient of a **moving-feed**
/// lumped-port-terminated pencil with a `|S₁₁(f₀)|²` objective — Epic #628
/// **Phase A2** (issue #633), the sibling of the pinned-feed
/// [`driven_shape_gradient_ports_complex`] that additionally differentiates the
/// port through a moving feed.
///
/// Phase A1 assumes the port faces/nodes are geometry-constant, so `∂S_p/∂X = 0`
/// and `∂b_port/∂X = 0`. When a design node **touches a port face** those
/// assumptions break and three port terms plus one objective term appear, all of
/// which this entry point accumulates on top of the A1 volume contraction:
///
/// ```text
///   ∂S_p/∂X   — tangential surface-mass geometry derivative (face area + ∇λ),
///   ∂f/∂X     — port-flux moment geometry derivative,
///   ∂Z_s/∂X   — Z_s = R·w/l with w = area/l, l = extent along ê both geometric,
///   ∂g/∂X|_x  — the |S₁₁|² covector's EXPLICIT dependence on the moving feed
///               (V = (l/area)·Σ f_i x_i reads geometry through f and 1/w).
/// ```
///
/// # Geometry model for the scalar port dims (matches `mesh/patch.rs`)
///
/// The forward solve uses the port's supplied `width` (`w`) and `length` (`l`);
/// this gradient treats them as the **geometry-derived** quantities the patch
/// fixture builds ([`crate::mesh::patch`]): `l = hi − lo` is the extent of the
/// port nodes along `ê` and `w = area/l` with `area = Σ face areas`, so
/// `Z_s = R·w/l = R·area/l²`. The caller must therefore pass a port whose
/// `width`/`length` equal those geometric values at the current mesh (as
/// `PatchFixture::port` does). `∂l/∂X` uses the symmetric subgradient of the
/// non-smooth `max`/`min`: each of the `K` nodes tied at the extremum receives
/// `±ê/K` (the derivative for a coherent extremum motion, e.g. a feed-height
/// design DOF), which is what the FD gate exercises.
///
/// # Single factorization preserved
///
/// The port admittance is still a scalar (`jω/Z_s`) times the real-symmetric
/// `S_p`, so `A(ω)ᵀ = A(ω)` and the forward + adjoint share one LU. The extra
/// A2 terms are a post-solve geometry contraction against the SAME `λ` the core
/// already computed, so `n_factorizations == 1` is preserved.
///
/// # Arguments
///
/// Matches [`driven_shape_gradient_ports_complex`] but takes a **single** moving
/// [`LumpedPort`] and the S-parameter reference impedance `z0` (for the
/// Palace-style uniform port, `z0 = R`). The objective is fixed to `|S₁₁(f₀)|²`.
///
/// # Errors
///
/// Propagates [`DrivenError`] on input-shape mismatches, a bad port spec, empty
/// interior, or factorization / solve failure.
#[allow(clippy::too_many_arguments)]
pub fn driven_shape_gradient_moving_port_s11<B>(
    mesh: &TetMesh,
    eps_r: &[c64],
    bcs: &DrivenBcs<'_>,
    omega: f64,
    source: &CurrentSource,
    port: &LumpedPort<'_>,
    z0: f64,
    device: &B::Device,
) -> Result<DrivenShapeGradient, DrivenError>
where
    B: Backend,
{
    use crate::driven::extraction::s11_sq_objective;

    let edges = mesh.edges();
    let n_edges = edges.len();

    // The moving port's faces feed the Whitney port kernels (and the direct
    // `face_geometry` call of the dual twin below) BEFORE the core's own port
    // validation runs, so check them here first (issue #725): a triangle that
    // is not a tet face is an error, not a panic.
    validate_driven_surfaces(mesh, "lumped port", std::iter::once(port.faces))?;

    // The |S₁₁|² objective closure over the (geometry-fixed-at-base) port-flux
    // covector — identical to what a caller would build for the A1 path. This
    // supplies the field cotangent ∂g/∂x that the core adjoint solve uses; the
    // EXPLICIT geometry term ∂g/∂X|_x is added separately below.
    let flux = assemble_port_flux(mesh, port.faces, port.e_hat, &edges);
    if flux.len() != n_edges {
        return Err(DrivenError::SparseAssembly(format!(
            "port flux length {} != edge count {n_edges}",
            flux.len()
        )));
    }
    let inv_width = 1.0 / port.width;
    let objective = s11_sq_objective(flux.clone(), inv_width, port.v_inc, port.resistance, z0);

    // ONE forward + ONE adjoint solve on the port-loaded pencil; also returns λ.
    let core = driven_shape_gradient_ports_complex_core::<B, _>(
        mesh,
        eps_r,
        bcs,
        omega,
        source,
        std::slice::from_ref(port),
        objective,
        device,
    )?;

    let mut grad_node = core.grad_node;
    accumulate_moving_port_terms(
        mesh,
        port,
        z0,
        omega,
        &edges,
        &flux,
        &core.e_edges,
        &core.lambda_full,
        &mut grad_node,
    );

    Ok(DrivenShapeGradient {
        objective: core.objective,
        grad_node,
        e_edges: core.e_edges,
        residual_rel: core.residual_rel,
        n_factorizations: core.n_factorizations,
    })
}

// ─────────────────────────────────────────────────────────────────────────
// Box-UPML tensor-material shape-gradient driver — Epic #628 Phase C1
// (issue #635). PML shell pinned (∂Λ/∂X = 0); the moving-PML profile
// derivative is the explicit Phase C2 non-goal.
// ─────────────────────────────────────────────────────────────────────────

/// Length-`n_nodes` mask of the mesh nodes touched by any **PML-shell** tet
/// (`pml_tet_mask[t] == true`, e.g. the tets tagged
/// [`crate::mesh::patch::PHYS_UPML`]). These nodes carry a geometry-dependent
/// stretch `Λ(X)` that the **fixed-Λ** box-UPML shape adjoint
/// ([`driven_shape_gradient_matched_upml`], Phase C1) does **not** differentiate
/// (`∂Λ/∂X` is the Phase C2 non-goal), so any valid node-motion map MUST hold
/// them fixed. Pair with [`chain_node_motion_pml_pinned`] to enforce that.
///
/// # Panics
///
/// If `pml_tet_mask.len() != mesh.n_tets()`.
pub fn pml_shell_nodes(mesh: &TetMesh, pml_tet_mask: &[bool]) -> Vec<bool> {
    assert_eq!(
        pml_tet_mask.len(),
        mesh.n_tets(),
        "pml_tet_mask length {} != n_tets {}",
        pml_tet_mask.len(),
        mesh.n_tets()
    );
    let mut pinned = vec![false; mesh.n_nodes()];
    for (t, tet) in mesh.tets.iter().enumerate() {
        if pml_tet_mask[t] {
            for &v in tet.iter() {
                pinned[v as usize] = true;
            }
        }
    }
    pinned
}

/// [`crate::shape::chain_node_motion`] with a **PML-pinned tripwire** (issue
/// #635, Phase C1): asserts every node flagged in `pinned` (see
/// [`pml_shell_nodes`]) carries **zero** design motion before contracting
/// `⟨grad_node, dnode_dtheta⟩= ∂g/∂θ`.
///
/// Because the box-UPML shape adjoint holds `Λ` fixed (`∂Λ/∂X = 0` is unmodeled
/// — Phase C2), a design node that enters a PML tet would silently drop the
/// profile-derivative term and return a wrong gradient; this guard turns that
/// into a loud failure at chain time.
///
/// # Panics
///
/// If the three slices disagree in length, or if any `pinned` node has a
/// nonzero `dnode_dtheta` entry.
pub fn chain_node_motion_pml_pinned(
    grad_node: &[[f64; 3]],
    dnode_dtheta: &[[f64; 3]],
    pinned: &[bool],
) -> f64 {
    assert_eq!(
        pinned.len(),
        dnode_dtheta.len(),
        "pinned length {} != dnode_dtheta length {}",
        pinned.len(),
        dnode_dtheta.len()
    );
    for (n, (&is_pinned, d)) in pinned.iter().zip(dnode_dtheta.iter()).enumerate() {
        if is_pinned {
            let mag2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            assert!(
                mag2 == 0.0,
                "PML-pinned node {n} has nonzero design motion {d:?}; ∂Λ/∂X is \
                 unmodeled in the box-UPML shape adjoint (Phase C1) — a moving PML \
                 shell is the Phase C2 non-goal"
            );
        }
    }
    crate::shape::chain_node_motion(grad_node, dnode_dtheta)
}

/// Compute the full nodal-coordinate gradient `∂g/∂X_{n,d}` of a **matched
/// box-UPML** (tensor-material) driven Nédélec EM observable via the discrete
/// adjoint — **one forward + one adjoint solve**, reusing a single complex
/// sparse LU — with the **PML shell pinned** (Epic #628 Phase C1, issue #635).
///
/// This is the tensor-material twin of [`driven_shape_gradient_complex`]: it
/// differentiates the matched-UPML pencil
///
/// ```text
///   A(ω) = K(ν) − ω² M(ε),   ν = Λ⁻¹ (curl weight),   ε = ε_r·Λ (mass weight),
/// ```
///
/// assembled by [`crate::assembly::nedelec::assemble_global_nedelec_with_full_tensors_sparse`]
/// (the same forward path [`crate::driven::solve::DrivenMaterials::MatchedUpml`]
/// uses), with per-tet full-3×3 **complex** constitutive tensors. The element
/// factors `∂K(ν)/∂X` and `∂M(ε)/∂X` are read from the tensor-material Dual twin
/// `nedelec_local_dual_tensor` (exact forward-mode AD) at **fixed** Λ, and the
/// current-source RHS carries the same geometric `∂b/∂X` term as the scalar path
/// (the RHS is material-independent).
///
/// # Pinned PML shell — why C1 is tractable
///
/// The per-tet `Λ` (hence `ε`, `ν`) is a **constant** input here: the node-motion
/// map must hold every PML-region node fixed, so `∂Λ/∂X = 0` and the only new work
/// is the fixed-Λ geometry contraction. Use [`pml_shell_nodes`] +
/// [`chain_node_motion_pml_pinned`] to enforce (and assert) that pinning at chain
/// time — a design node entering a PML tet would need the unmodeled profile
/// derivative `∂s_k/∂centroid_k` (Phase C2). A diagonal box `Λ` is symmetric, so
/// the complex-symmetric pencil `A(ω)ᵀ = A(ω)` survives and the adjoint reuses the
/// forward LU (`n_factorizations == 1`).
///
/// # Arguments
///
/// * `mesh` — tetrahedral mesh (fixed topology; gradient w.r.t. node positions).
/// * `epsilon_tensor`, `nu_tensor` — per-tet full-3×3 complex mass / curl weights
///   (`ε = ε_r·Λ`, `ν = Λ⁻¹`; length `mesh.n_tets()` each), e.g. from
///   [`crate::mesh::patch::PatchFixture::matched_upml_materials`]. Held **fixed**
///   under the geometry perturbation (the pinned-shell convention).
/// * `bcs`, `omega`, `source`, `objective`, `device` — exactly as
///   [`driven_shape_gradient_complex`].
///
/// # Errors
///
/// Propagates [`DrivenError`] on input-shape mismatches or a failed
/// factorization / solve, and on an objective-cotangent length mismatch.
#[allow(clippy::too_many_arguments)]
pub fn driven_shape_gradient_matched_upml<B, G>(
    mesh: &TetMesh,
    epsilon_tensor: &[[[c64; 3]; 3]],
    nu_tensor: &[[[c64; 3]; 3]],
    bcs: &DrivenBcs<'_>,
    omega: f64,
    source: &CurrentSource,
    objective: G,
    device: &B::Device,
) -> Result<DrivenShapeGradient, DrivenError>
where
    B: Backend,
    G: Fn(&[c64]) -> (f64, Vec<c64>),
{
    // Port-less delegation: the shared core with an empty port list is the
    // historical (Phase C1) box-UPML path, bit-for-bit. See the module docs and
    // [`driven_shape_gradient_matched_upml_ports`].
    driven_shape_gradient_matched_upml_ports::<B, G>(
        mesh,
        epsilon_tensor,
        nu_tensor,
        bcs,
        &[],
        omega,
        source,
        objective,
        device,
    )
}

/// Compute the full nodal-coordinate gradient `∂g/∂X_{n,d}` of the **composed
/// open-radiator** driven Nédélec pencil — a **matched box-UPML** tensor
/// material with a **complex (lossy) ε** AND a slice of **pinned-feed lumped
/// ports** threaded into the differentiated system — via the discrete adjoint
/// (**one forward + one adjoint solve**, reusing a single complex sparse LU),
/// with the **PML shell pinned** (Epic #628 capstone, issue #636).
///
/// This is the full-forward composition of the three epic ingredients:
///
/// * the **box-UPML tensor material** of [`driven_shape_gradient_matched_upml`]
///   (`A(ω) = K(ν) − ω² M(ε)`, `ν = Λ⁻¹`, `ε = ε_r·Λ`, both full 3×3 complex),
/// * the **complex/lossy ε** it already carries (nonzero `Im(ε_r)` for `tan δ`),
///   and
/// * the **pinned-feed lumped-port termination** of
///   [`driven_shape_gradient_ports_complex`]
///   (`A(ω) += (jω/Z_s) S_p`, port boundary drive folded into `b`).
///
/// It differentiates the **exact** system the public open-radiator forward
/// [`crate::driven::solve::driven_solve_with_ports`] builds with
/// [`crate::driven::solve::DrivenMaterials::MatchedUpml`] materials and the same
/// `ports` slice — the full radiating patch model. Radiation loss (the UPML)
/// dissipates power, so the extracted `|S₁₁|` is a **passive** reflection
/// coefficient (`≤ 1`), unlike the #626 closed-cavity pencil.
///
/// # Pinned feed + pinned PML shell — why the capstone is tractable
///
/// Both the port faces/nodes and the PML-region nodes are **geometry-constant**
/// (the feed is pinned, the shell is pinned), so `∂S_p/∂X = 0`,
/// `∂b_port/∂X = 0`, and `∂Λ/∂X = 0`: there is **no** new geometry term beyond
/// the fixed-Λ volume `∂K(ν)/∂X`, `∂M(ε)/∂X` and the material-independent
/// `∂b/∂X` current-RHS contraction. The port admittance and drive simply load
/// the shared forward + adjoint solves, and the returned gradient is taken
/// through the loaded system. Use [`pml_shell_nodes`] +
/// [`chain_node_motion_pml_pinned`] to enforce (and assert) the shell pinning,
/// and hold every port-face node fixed in the node-motion map. Both loads keep
/// `A(ω)` complex-symmetric (`Λ` diagonal-box symmetric; `S_p` real-symmetric
/// scaled by the scalar `jω/Z_s`), so the transpose (adjoint) solve reuses the
/// single forward LU (`n_factorizations == 1`).
///
/// An empty `ports` slice reproduces [`driven_shape_gradient_matched_upml`]
/// bit-for-bit.
///
/// # Arguments
///
/// * `mesh`, `epsilon_tensor`, `nu_tensor`, `bcs`, `omega`, `source`,
///   `objective`, `device` — exactly as [`driven_shape_gradient_matched_upml`].
/// * `ports` — the pinned-feed [`LumpedPort`] slice (added parameter), threaded
///   into the forward + adjoint solves identically to
///   [`driven_shape_gradient_ports_complex`].
///
/// # Errors
///
/// Propagates [`DrivenError`] on input-shape mismatches, a bad port spec, empty
/// interior, a failed factorization / solve, or an objective-cotangent length
/// mismatch.
#[allow(clippy::too_many_arguments)]
pub fn driven_shape_gradient_matched_upml_ports<B, G>(
    mesh: &TetMesh,
    epsilon_tensor: &[[[c64; 3]; 3]],
    nu_tensor: &[[[c64; 3]; 3]],
    bcs: &DrivenBcs<'_>,
    ports: &[LumpedPort<'_>],
    omega: f64,
    source: &CurrentSource,
    objective: G,
    device: &B::Device,
) -> Result<DrivenShapeGradient, DrivenError>
where
    B: Backend,
    G: Fn(&[c64]) -> (f64, Vec<c64>),
{
    let core = matched_upml_ports_core::<B, G>(
        mesh,
        epsilon_tensor,
        nu_tensor,
        bcs,
        ports,
        omega,
        source,
        objective,
        device,
    )?;
    Ok(DrivenShapeGradient {
        objective: core.objective,
        grad_node: core.grad_node,
        e_edges: core.e_edges,
        residual_rel: core.residual_rel,
        n_factorizations: core.n_factorizations,
    })
}

/// Matched box-UPML **moving-feed** `|S₁₁|²` shape gradient (issue #1120): the
/// tensor-material sibling of [`driven_shape_gradient_moving_port_s11`].
///
/// One forward + one adjoint solve on the same port-loaded matched-UPML pencil
/// as [`driven_shape_gradient_matched_upml_ports`] (single LU), then the
/// moving-feed port terms `∂S_p/∂X` (with `∂Z_s/∂X`), `∂b_port/∂X` and the
/// explicit objective term `∂g/∂X|_x` are contracted against the same `λ`
/// (the shared `accumulate_moving_port_terms`). The PML shell must stay pinned
/// (`∂Λ/∂X = 0`, the Phase C2 non-goal), exactly as for the pinned-feed path;
/// the port-face nodes may move.
///
/// As for the scalar sibling, the caller must pass a port whose `width` and
/// `length` are the **geometric** values at `mesh` (`l` = the extent of the
/// port nodes along `ê`, `w` = face area / `l`, as
/// [`crate::mesh::patch::PatchFixture::port`] builds them), and the
/// S-parameter reference impedance `z0` (`= R` for the uniform port).
///
/// # Errors
///
/// As [`driven_shape_gradient_matched_upml_ports`].
#[allow(clippy::too_many_arguments)]
pub fn driven_shape_gradient_matched_upml_moving_port_s11<B>(
    mesh: &TetMesh,
    epsilon_tensor: &[[[c64; 3]; 3]],
    nu_tensor: &[[[c64; 3]; 3]],
    bcs: &DrivenBcs<'_>,
    port: &LumpedPort<'_>,
    omega: f64,
    source: &CurrentSource,
    z0: f64,
    device: &B::Device,
) -> Result<DrivenShapeGradient, DrivenError>
where
    B: Backend,
{
    use crate::driven::extraction::s11_sq_objective;

    let edges = mesh.edges();
    validate_driven_surfaces(mesh, "lumped port", std::iter::once(port.faces))?;
    let flux = assemble_port_flux(mesh, port.faces, port.e_hat, &edges);
    let objective = s11_sq_objective(
        flux.clone(),
        1.0 / port.width,
        port.v_inc,
        port.resistance,
        z0,
    );
    let core = matched_upml_ports_core::<B, _>(
        mesh,
        epsilon_tensor,
        nu_tensor,
        bcs,
        std::slice::from_ref(port),
        omega,
        source,
        objective,
        device,
    )?;
    let mut grad_node = core.grad_node;
    accumulate_moving_port_terms(
        mesh,
        port,
        z0,
        omega,
        &edges,
        &flux,
        &core.e_edges,
        &core.lambda_full,
        &mut grad_node,
    );
    Ok(DrivenShapeGradient {
        objective: core.objective,
        grad_node,
        e_edges: core.e_edges,
        residual_rel: core.residual_rel,
        n_factorizations: core.n_factorizations,
    })
}

/// The matched box-UPML tensor core of
/// [`driven_shape_gradient_matched_upml_ports`] and
/// [`driven_shape_gradient_matched_upml_moving_port_s11`]: a thin adapter that
/// supplies the tensor path's material validation, its `K(ν) / M(ε)` full-tensor
/// assembly call and its per-tet Re/Im `Dual`-twin kernel passes to the shared
/// [`ports_shape_core`].
///
/// The local arithmetic stays here, in this path's own operation order; the
/// scalar path is deliberately NOT routed through these tensor kernels with
/// `ε·I`, `ν = I` (that would change rounding — issue #1596).
#[allow(clippy::too_many_arguments)]
fn matched_upml_ports_core<B, G>(
    mesh: &TetMesh,
    epsilon_tensor: &[[[c64; 3]; 3]],
    nu_tensor: &[[[c64; 3]; 3]],
    bcs: &DrivenBcs<'_>,
    ports: &[LumpedPort<'_>],
    omega: f64,
    source: &CurrentSource,
    objective: G,
    device: &B::Device,
) -> Result<PortShapeCore, DrivenError>
where
    B: Backend,
    G: Fn(&[c64]) -> (f64, Vec<c64>),
{
    let omega2 = omega * omega;
    ports_shape_core::<B, G, _, _, _>(
        mesh,
        bcs,
        ports,
        omega,
        source,
        objective,
        device,
        // Full ε and ν tensors: both are per-tet 3×3 complex matrices.
        |n_tets| {
            if epsilon_tensor.len() != n_tets {
                return Err(DrivenError::MaterialDimMismatch {
                    got: epsilon_tensor.len(),
                    want: n_tets,
                });
            }
            if nu_tensor.len() != n_tets {
                return Err(DrivenError::MaterialDimMismatch {
                    got: nu_tensor.len(),
                    want: n_tets,
                });
            }
            Ok(())
        },
        // K(ν) and M(ε) from the full complex tensors. Unlike the scalar path,
        // the matched-UPML curl-curl K also has an imaginary part (ν = Λ⁻¹ is
        // complex), so k_im is carried alongside k_re.
        |nodes_t, tets_t, tet_sign, scatter| {
            let sys = assemble_global_nedelec_with_full_tensors_sparse(
                nodes_t,
                tets_t,
                tet_sign,
                scatter,
                epsilon_tensor,
                nu_tensor,
            );
            AssembledPencilHost {
                k_re: sys.k_re_vals.into_data().iter::<f64>().collect(),
                k_im: Some(sys.k_im_vals.into_data().iter::<f64>().collect()),
                m_re: sys.m_re_vals.into_data().iter::<f64>().collect(),
                m_im: sys.m_im_vals.into_data().iter::<f64>().collect(),
            }
        },
        // ∂A_ij = ∂K(ν)_ij − ω² ∂M(ε)_ij. Both element tangents are COMPLEX
        // here (the box Λ makes ν, ε complex), so each runs as a Re/Im weight
        // pass of the tensor Dual twin. The RHS moment ∂(∫N_i) is
        // material-independent — reused from the scalar dual kernel.
        |t, dc| {
            // Real / imaginary weight components: ν (curl) and ε (mass).
            let nu_re = tensor_re(&nu_tensor[t]);
            let nu_im = tensor_im(&nu_tensor[t]);
            let eps_re = tensor_re(&epsilon_tensor[t]);
            let eps_im = tensor_im(&epsilon_tensor[t]);

            // Re/Im passes of the tensor twin → complex ∂K, ∂M.
            let (dk_re, dm_re) = nedelec_local_dual_tensor(dc, &nu_re, &eps_re);
            let (dk_im, dm_im) = nedelec_local_dual_tensor(dc, &nu_im, &eps_im);
            // Material-independent RHS moments ∂(∫N_i)/∂X.
            let (_, _, dnint) = nedelec_local_dual(dc);

            LocalShapeTangents {
                da: std::array::from_fn(|i| {
                    std::array::from_fn(|j| {
                        let dk = c64::new(dk_re[i][j].du, dk_im[i][j].du);
                        let dm = c64::new(dm_re[i][j].du, dm_im[i][j].du);
                        dk - dm * omega2
                    })
                }),
                dnint,
            }
        },
    )
}

/// Real part of a complex 3×3 tensor as a plain `f64` matrix (a lifted constant
/// weight for the fixed-Λ tensor Dual twin, issue #635).
#[inline]
fn tensor_re(t: &[[c64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|p| std::array::from_fn(|q| t[p][q].re))
}
/// Imaginary part of a complex 3×3 tensor as a plain `f64` matrix.
#[inline]
fn tensor_im(t: &[[c64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|p| std::array::from_fn(|q| t[p][q].im))
}

// ─────────────────────────────────────────────────────────────────────────
// ORDER-GENERIC shape gradient on a `DrivenOperator` (issue #1264, Epic #836
// Phase 4).
// ─────────────────────────────────────────────────────────────────────────

/// Tensor-material Dual twin of the **second-order** Nédélec element
/// ([`nedelec2_local_dual`] for a fixed real 3×3 weight): the local ν-weighted
/// curl-curl `K(W_k)`, the ε-weighted mass `M(W_m)` (both `20×20`) and the
/// current-RHS moments `∫ N_i dV` (`20×3`), in dual arithmetic — so each `.du`
/// is the exact `∂/∂X` at **fixed** weights (the pinned-UPML-shell
/// convention of the p=1 [`nedelec_local_dual_tensor`]).
///
/// Mirrors [`crate::assembly::nedelec_p2::p2_local_curl_tensor`] /
/// [`crate::assembly::nedelec_p2::p2_local_mass_tensor`] (`Σ_q w_q u_iᵀ W
/// u_j`) over the same fixed [`tet_quad_deg4`] rule. A complex weight runs as
/// two calls (real-part weights, imaginary-part weights); the caller
/// recombines `∂K = ∂K_re + i·∂K_im`.
#[allow(clippy::type_complexity)]
pub(crate) fn nedelec2_local_dual_tensor(
    coords: &[[Dual; 3]; 4],
    w_k: &[[f64; 3]; 3],
    w_m: &[[f64; 3]; 3],
) -> (
    [[Dual; TET_NEDELEC2_DOFS]; TET_NEDELEC2_DOFS],
    [[Dual; TET_NEDELEC2_DOFS]; TET_NEDELEC2_DOFS],
    [[Dual; 3]; TET_NEDELEC2_DOFS],
) {
    // W · a for a dual 3-vector and a fixed real weight.
    let apply = |w: &[[f64; 3]; 3], a: [Dual; 3]| -> [Dual; 3] {
        std::array::from_fn(|p| {
            a[0].scale(w[p][0])
                .add(a[1].scale(w[p][1]))
                .add(a[2].scale(w[p][2]))
        })
    };
    let (bary, signed_vol) = tet_barycentric_gradients_dual(coords);
    let vol_abs = signed_vol.abs();
    let zero = Dual::cst(0.0);
    let mut k = [[zero; TET_NEDELEC2_DOFS]; TET_NEDELEC2_DOFS];
    let mut m = [[zero; TET_NEDELEC2_DOFS]; TET_NEDELEC2_DOFS];
    let mut nint = [[zero; 3]; TET_NEDELEC2_DOFS];
    for (lam, frac) in tet_quad_deg4() {
        let w = vol_abs.scale(frac);
        let (n, c) = tet_nedelec2_shapes_dual(&lam, &bary);
        let wn: [[Dual; 3]; TET_NEDELEC2_DOFS] = std::array::from_fn(|j| apply(w_m, n[j]));
        let wc: [[Dual; 3]; TET_NEDELEC2_DOFS] = std::array::from_fn(|j| apply(w_k, c[j]));
        for i in 0..TET_NEDELEC2_DOFS {
            for j in 0..TET_NEDELEC2_DOFS {
                m[i][j] = m[i][j].add(ddot3(n[i], wn[j]).mul(w));
                k[i][j] = k[i][j].add(ddot3(c[i], wc[j]).mul(w));
            }
            for d in 0..3 {
                nint[i][d] = nint[i][d].add(n[i][d].mul(w));
            }
        }
    }
    (k, m, nint)
}

/// A fault injected into one p=2 surface Dual-twin term of the shape
/// gradient — **test-only** (the mutation gate of issue #1264): every public
/// entry point passes [`SurfaceTermMutation::None`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SurfaceTermMutation {
    /// The real gradient.
    None,
    /// Drop the impedance-wall `∂S_Γ/∂X` contribution.
    DropWallMass,
    /// Drop the lumped-port `∂S_p/∂X` contribution.
    DropPortMass,
    /// Drop the lumped-port `∂f_p/∂X` (drive) contribution.
    DropPortFlux,
}

/// Promote per-tet diagonal permittivities `[ε_x, ε_y, ε_z]` to the full 3×3
/// `(ε, ν)` tensor pair of the matched-UPML pencil, with `ν = μ⁻¹ = I`
/// (`μ = 1`), so a [`DrivenMaterials::DiagTensor`] material can reuse the
/// tensor Dual shape-gradient path.
#[allow(clippy::type_complexity)]
pub(crate) fn diag_tensor_to_full(
    eps_diag: &[[c64; 3]],
) -> (Vec<[[c64; 3]; 3]>, Vec<[[c64; 3]; 3]>) {
    let zero = c64::new(0.0, 0.0);
    let one = c64::new(1.0, 0.0);
    let eps = eps_diag
        .iter()
        .map(|d| [[d[0], zero, zero], [zero, d[1], zero], [zero, zero, d[2]]])
        .collect();
    let nu = vec![[[one, zero, zero], [zero, one, zero], [zero, zero, one]]; eps_diag.len()];
    (eps, nu)
}

/// Nodal-coordinate gradient `∂g/∂X_{n,d}` of the driven solve described by
/// `problem`, at **either element order** — the order-generic successor of
/// the former cube-only `driven_shape_gradient_p2` / `driven_shape_gradient_p2_complex`
/// drivers (issue #1264, Epic #836 Phase 4).
///
/// - **p=1:** delegates bit-for-bit to
///   [`driven_shape_gradient_ports_complex`] (scalar ε, no walls) or
///   [`driven_shape_gradient_matched_upml_ports`] (matched UPML, no walls);
///   anything else (walls, diagonal tensors, `excited = Some(_)`) is
///   [`DrivenError::UnsupportedAtOrder`] — the p=1 shape drivers have no
///   wall term.
/// - **p=2:** assembles the p=2 [`DrivenOperator`] from `problem`, factors
///   `A(ω)` once (`n_factorizations == 1`), and contracts, per tet and per
///   seeded vertex coordinate through exact forward-mode AD,
///
///   ```text
///   ∂g/∂X = 2 Re[ λᵀ ∂b/∂X − λᵀ (∂A/∂X) x ],         Aᵀ λ = ∂g/∂x,
///   ∂A = ∂K(ν) − ω² ∂M(ε) + Σ_Γ (iω/Z_Γ) ∂S_Γ + Σ_p (iω/Z_p) ∂S_p,
///   ∂b = iω ∂∫N·J + Σ_p (2iω/Z_p)(V_inc,p/l_p) ∂f_p.
///   ```
///
///   The volume terms use `nedelec2_local_dual` (scalar ε) or
///   `nedelec2_local_dual_tensor` (diagonal / matched-UPML tensors, held
///   **fixed** under the motion: the node-motion map must pin every UPML-shell
///   node, as in the p=1 matched-UPML driver); the surface terms use the p=2
///   trace Dual twins of [`crate::assembly::surface_p2`]
///   (`∂S/∂X`, `∂f/∂X`). The surface coefficients `Z_Γ(ω)`, `Z_p = R w / l`
///   and the incident drive are **parameters** of the forward operator, so
///   they are held fixed (exactly what the forward operator does when the
///   mesh moves); a pinned-feed port or wall simply sees zero node motion on
///   its faces and contributes nothing. An objective that reads the geometry
///   explicitly (e.g. through the port voltage `(1/w) Σ f_i E_i`) owns that
///   `∂g/∂X|_x` term.
///
/// # Errors
///
/// [`DrivenError::UnsupportedAtOrder`] for `problem.wave_ports` ≠ ∅ (names
/// `#836 P3b`), plus the assembly / factorization errors of the forward
/// operator.
pub fn driven_shape_gradient_on_space<B, G>(
    problem: &DrivenGradientProblem<'_>,
    objective: G,
    device: &B::Device,
) -> Result<DrivenShapeGradient, DrivenError>
where
    B: Backend,
    G: Fn(&[c64]) -> (f64, Vec<c64>),
{
    problem.check_supported()?;
    if problem.space.order() == ElementOrder::P1 {
        let unsupported = |feature: &'static str| DrivenError::UnsupportedAtOrder {
            order: ElementOrder::P1,
            feature,
        };
        if !problem.surfaces.is_empty() {
            return Err(unsupported("the shape gradient of impedance-surface walls"));
        }
        if problem.excited.is_some() {
            return Err(unsupported(
                "the excited-port selector of the shape gradient",
            ));
        }
        return match problem.materials {
            DrivenMaterials::Scalar(eps) => driven_shape_gradient_ports_complex::<B, G>(
                problem.mesh,
                eps,
                problem.bcs,
                problem.omega,
                problem.source,
                problem.ports,
                objective,
                device,
            ),
            DrivenMaterials::MatchedUpml {
                epsilon_tensor,
                nu_tensor,
            } => driven_shape_gradient_matched_upml_ports::<B, G>(
                problem.mesh,
                epsilon_tensor,
                nu_tensor,
                problem.bcs,
                problem.ports,
                problem.omega,
                problem.source,
                objective,
                device,
            ),
            // Epic #841 Phase 2a (#1259): a diagonal anisotropic ε with μ = 1 is
            // exactly the matched-UPML pencil with `ε = diag(ε_x, ε_y, ε_z)` and
            // `ν = I` (same global-Cartesian basis, same K/M assembly), so the
            // FD-validated tensor Dual path (`nedelec_local_dual_tensor`) is
            // reused verbatim by promoting the diagonal to full tensors.
            DrivenMaterials::DiagTensor(eps_diag) => {
                let (epsilon_tensor, nu_tensor) = diag_tensor_to_full(eps_diag);
                driven_shape_gradient_matched_upml_ports::<B, G>(
                    problem.mesh,
                    &epsilon_tensor,
                    &nu_tensor,
                    problem.bcs,
                    problem.ports,
                    problem.omega,
                    problem.source,
                    objective,
                    device,
                )
            }
        };
    }
    let op = problem.assemble_p2_operator()?;
    shape_gradient_p2_core(problem, &op, None, objective, SurfaceTermMutation::None)
}

/// The p=2 shape-gradient core on an assembled operator (see
/// [`driven_shape_gradient_on_space`]). `b_override` replaces the operator's
/// own RHS (the legacy raw-RHS entry points); `mutation` is the test-only
/// fault injection.
pub(crate) fn shape_gradient_p2_core<G>(
    problem: &DrivenGradientProblem<'_>,
    op: &DrivenOperator,
    b_override: Option<&[c64]>,
    objective: G,
    mutation: SurfaceTermMutation,
) -> Result<DrivenShapeGradient, DrivenError>
where
    G: Fn(&[c64]) -> (f64, Vec<c64>),
{
    let mesh = problem.mesh;
    let space = problem.space;
    let omega = problem.omega;
    let omega2 = omega * omega;
    let iomega = c64::new(0.0, omega);
    let zero_c = c64::new(0.0, 0.0);
    let n_tets = mesh.n_tets();
    let n_nodes = mesh.n_nodes();

    // Wall coefficients iω/Z_Γ(ω) (an error on a singular impedance, exactly
    // like the forward operator).
    let wall_coeffs: Vec<c64> = problem
        .surfaces
        .iter()
        .map(|bc| bc.model.weak_coefficient(omega))
        .collect::<Result<_, _>>()?;

    let fa = crate::driven::adjoint::p2_forward_adjoint(problem, op, b_override, objective)?;
    let (x, lam) = (&fa.x_full, &fa.lam_full);
    let mut grad_node = vec![[0.0_f64; 3]; n_nodes];

    // --- Volume terms (K, M, ∂b of the volumetric source). -------------------
    for (t, tet) in mesh.tets.iter().enumerate() {
        let dofs = space.tet_dofs(t);
        let lam_loc: [c64; TET_NEDELEC2_DOFS] = std::array::from_fn(|i| lam[dofs[i] as usize]);
        if lam_loc.iter().all(|z| z.re == 0.0 && z.im == 0.0) {
            continue;
        }
        let x_loc: [c64; TET_NEDELEC2_DOFS] = std::array::from_fn(|i| x[dofs[i] as usize]);
        let TetOrientation::AscendingPerm(perm) = space.tet_orientation(t) else {
            return Err(DrivenError::UnsupportedAtOrder {
                order: space.order(),
                feature: "the p=2 shape gradient on a non-p=2 space",
            });
        };
        let sorted = space.tet_local_coords(mesh, t);
        let jt = problem.source.j_tet[t];
        // Per-tet material: scalar ε, or the fixed complex (ε, ν) tensors.
        #[allow(clippy::type_complexity)]
        let tensors: Option<([[c64; 3]; 3], [[c64; 3]; 3])> = match problem.materials {
            DrivenMaterials::Scalar(_) => None,
            DrivenMaterials::DiagTensor(eps) => {
                let d = eps[t];
                let z = zero_c;
                let one = c64::new(1.0, 0.0);
                Some((
                    [[d[0], z, z], [z, d[1], z], [z, z, d[2]]],
                    [[one, z, z], [z, one, z], [z, z, one]],
                ))
            }
            DrivenMaterials::MatchedUpml {
                epsilon_tensor,
                nu_tensor,
            } => Some((epsilon_tensor[t], nu_tensor[t])),
        };
        let eps_scalar = match problem.materials {
            DrivenMaterials::Scalar(eps) => eps[t],
            _ => zero_c,
        };

        for a in 0..4 {
            let node = tet[perm[a]] as usize;
            for c_axis in 0..3 {
                let mut dc = sorted.map(|v| v.map(Dual::cst));
                dc[a][c_axis] = Dual::var(sorted[a][c_axis]);

                // ∂A_ij (complex) and the source-moment tangents.
                let (d_a, dnint): ([[c64; TET_NEDELEC2_DOFS]; TET_NEDELEC2_DOFS], _) =
                    if let Some((eps_t, nu_t)) = &tensors {
                        let (k_re, m_re, dn) =
                            nedelec2_local_dual_tensor(&dc, &tensor_re(nu_t), &tensor_re(eps_t));
                        let (k_im, m_im, _) =
                            nedelec2_local_dual_tensor(&dc, &tensor_im(nu_t), &tensor_im(eps_t));
                        (
                            std::array::from_fn(|i| {
                                std::array::from_fn(|j| {
                                    c64::new(k_re[i][j].du, k_im[i][j].du)
                                        - c64::new(m_re[i][j].du, m_im[i][j].du) * omega2
                                })
                            }),
                            dn,
                        )
                    } else {
                        let (dk, dm, dn) = nedelec2_local_dual(&dc);
                        (
                            std::array::from_fn(|i| {
                                std::array::from_fn(|j| {
                                    c64::new(dk[i][j].du, 0.0) - eps_scalar * (omega2 * dm[i][j].du)
                                })
                            }),
                            dn,
                        )
                    };

                // −λᵀ (∂A/∂X) x.
                let mut term_a = zero_c;
                for i in 0..TET_NEDELEC2_DOFS {
                    for j in 0..TET_NEDELEC2_DOFS {
                        term_a += lam_loc[i] * x_loc[j] * d_a[i][j];
                    }
                }
                // +λᵀ ∂b/∂X, ∂b_i = iω (∂(∫N_i)·J), J held fixed per element.
                let mut term_b = zero_c;
                for i in 0..TET_NEDELEC2_DOFS {
                    let mut dnj = zero_c;
                    for d in 0..3 {
                        dnj += jt[d] * dnint[i][d].du;
                    }
                    term_b += lam_loc[i] * (iomega * dnj);
                }
                grad_node[node][c_axis] += 2.0 * (term_b - term_a).re;
            }
        }
    }
    let _ = n_tets;

    // --- Surface terms on the p=2 trace Dual twins. ---------------------------
    // For one boundary triangle: ∂(coeff·S)/∂X contracted with λ, x, and (for
    // a driven port) the flux tangent.
    let tri_geometry = |tri: &[u32; 3]| -> Result<([u32; 8], [u32; 3]), DrivenError> {
        p2_trace_dofs(space, tri).ok_or(DrivenError::SurfaceNotOnMesh {
            surface: "gradient surface".to_string(),
            triangle: *tri,
            dangling: 1,
            total: 1,
        })
    };
    let accumulate_mass =
        |tris: &[[u32; 3]], coeff: c64, grad: &mut Vec<[f64; 3]>| -> Result<(), DrivenError> {
            for tri in tris {
                let (tdofs, s) = tri_geometry(tri)?;
                let lam_t: [c64; 8] = std::array::from_fn(|i| lam[tdofs[i] as usize]);
                let x_t: [c64; 8] = std::array::from_fn(|i| x[tdofs[i] as usize]);
                if lam_t.iter().all(|z| z.re == 0.0 && z.im == 0.0) {
                    continue;
                }
                let base: [[f64; 3]; 3] = std::array::from_fn(|i| mesh.nodes[s[i] as usize]);
                for a in 0..3 {
                    for c_axis in 0..3 {
                        let mut dv = base.map(|p| p.map(Dual::cst));
                        dv[a][c_axis] = Dual::var(base[a][c_axis]);
                        let ds = tri_nedelec2_surface_mass_dual(&dv);
                        let mut acc = zero_c;
                        for i in 0..8 {
                            for j in 0..8 {
                                acc += lam_t[i] * x_t[j] * ds[i][j].du;
                            }
                        }
                        // A += coeff·S ⇒ −λᵀ(coeff ∂S)x.
                        grad[s[a] as usize][c_axis] += 2.0 * (-(coeff * acc)).re;
                    }
                }
            }
            Ok(())
        };
    if mutation != SurfaceTermMutation::DropWallMass {
        for (bc, &coeff) in problem.surfaces.iter().zip(&wall_coeffs) {
            accumulate_mass(bc.triangles, coeff, &mut grad_node)?;
        }
    }
    for (p, port) in problem.ports.iter().enumerate() {
        let z_s = port.surface_impedance();
        if mutation != SurfaceTermMutation::DropPortMass {
            accumulate_mass(port.faces, c64::new(0.0, omega / z_s), &mut grad_node)?;
        }
        // Drive b_port = (2iω/Z_s)(V_inc/l) f (restricted to the excited port).
        let driven = port.v_inc != zero_c && problem.excited.is_none_or(|j| j == p);
        if driven && mutation != SurfaceTermMutation::DropPortFlux {
            let drive = c64::new(0.0, 2.0 * omega / z_s) * (port.v_inc * (1.0 / port.length));
            for tri in port.faces {
                let (tdofs, s) = tri_geometry(tri)?;
                let lam_t: [c64; 8] = std::array::from_fn(|i| lam[tdofs[i] as usize]);
                if lam_t.iter().all(|z| z.re == 0.0 && z.im == 0.0) {
                    continue;
                }
                let base: [[f64; 3]; 3] = std::array::from_fn(|i| mesh.nodes[s[i] as usize]);
                for a in 0..3 {
                    for c_axis in 0..3 {
                        let mut dv = base.map(|q| q.map(Dual::cst));
                        dv[a][c_axis] = Dual::var(base[a][c_axis]);
                        let df = tri_nedelec2_surface_flux_dual(&dv, port.e_hat);
                        let mut acc = zero_c;
                        for i in 0..8 {
                            acc += lam_t[i] * df[i].du;
                        }
                        grad_node[s[a] as usize][c_axis] += 2.0 * (drive * acc).re;
                    }
                }
            }
        }
    }

    Ok(DrivenShapeGradient {
        objective: fa.objective,
        grad_node,
        e_edges: fa.x_full,
        residual_rel: fa.residual_rel,
        n_factorizations: 1,
    })
}

#[cfg(test)]
mod tests;
