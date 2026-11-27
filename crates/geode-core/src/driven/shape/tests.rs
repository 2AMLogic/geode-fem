use super::*;
use crate::assembly::hcurl_space::HcurlSpace;
use crate::assembly::nedelec::cube_pec_interior_edges;
use crate::driven::solve::{DrivenMaterials, driven_solve};
use crate::elements::nedelec::batched_nedelec_local_matrices;
use crate::mesh::cube_tet_mesh;
use crate::shape::chain_node_motion;
use crate::testing::TestBackend;
use burn::tensor::TensorData;
use burn::tensor::backend::BackendTypes;

type B = TestBackend;

fn device() -> <B as BackendTypes>::Device {
    <B as BackendTypes>::Device::default()
}

/// Objective `g(x) = Σ_i |x_i|²` and its Wirtinger cotangent
/// `∂g/∂x_i = x̄_i`. Real, no explicit geometry dependence.
fn l2_objective(x: &[c64]) -> (f64, Vec<c64>) {
    let g = x.iter().map(|z| z.re * z.re + z.im * z.im).sum::<f64>();
    let cot = x.iter().map(|z| c64::new(z.re, -z.im)).collect();
    (g, cot)
}

/// Driven PEC cube cavity, uniform lossless ε_r, driven by a genuinely
/// COMPLEX per-tet current source (so the field `x` is fully complex and
/// the Wirtinger conjugation convention is exercised). Returns
/// `(mesh, eps_r, interior_mask, source)`.
fn cavity_fixture(n: usize) -> (TetMesh, Vec<f64>, Vec<bool>, CurrentSource) {
    let mesh = cube_tet_mesh(n, 1.0);
    let eps_r = vec![2.0_f64; mesh.n_tets()];
    let (_, interior) = cube_pec_interior_edges(&mesh, 1.0);
    let pi = std::f64::consts::PI;
    let source = CurrentSource::from_centroids(&mesh, |c| {
        [
            c64::new(0.0, 0.3 * (pi * c[2]).sin()),
            c64::new(0.5 * (pi * c[1]).sin(), 0.2),
            c64::new((pi * c[0]).sin(), 0.4 * c[2]),
        ]
    });
    (mesh, eps_r, interior, source)
}

/// The real `f64` Nédélec local matrices from the **production** Burn
/// kernel for a single tet, used to pin the dual `.re` fields.
fn burn_local_matrices(coords: &[[f64; 3]; 4]) -> ([[f64; 6]; 6], [[f64; 6]; 6]) {
    let flat: Vec<f64> = coords.iter().flat_map(|v| v.iter().copied()).collect();
    let t = burn::tensor::Tensor::<B, 1>::from_data(TensorData::new(flat, [12]), &device())
        .reshape([1, 4, 3]);
    let local = batched_nedelec_local_matrices(t);
    let k: Vec<f64> = local.k_local.into_data().iter::<f64>().collect();
    let m: Vec<f64> = local.m_local.into_data().iter::<f64>().collect();
    let mut kk = [[0.0_f64; 6]; 6];
    let mut mm = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            kk[i][j] = k[i * 6 + j];
            mm[i][j] = m[i * 6 + j];
        }
    }
    (kk, mm)
}

/// **The dual `.re` faithfully lifts the production Nédélec kernel.** The
/// `.re` fields of the dual local K/M must reproduce the real `f64` Burn
/// kernel [`batched_nedelec_local_matrices`] entry-for-entry — proving the
/// dual pass differentiates *the same* closed form the solver assembles.
#[test]
fn dual_local_matrices_reproduce_burn_kernel() {
    // Generic well-shaped (non-axis-aligned) tet.
    let base = [
        [0.10, 0.20, 0.05],
        [1.05, 0.15, 0.20],
        [0.25, 0.95, 0.10],
        [0.20, 0.30, 1.10],
    ];
    let dc = base.map(|v| v.map(Dual::cst));
    let (dk, dm, _) = nedelec_local_dual(&dc);
    let (bk, bm) = burn_local_matrices(&base);
    let mut worst = 0.0_f64;
    for i in 0..6 {
        for j in 0..6 {
            let rk = (dk[i][j].re - bk[i][j]).abs() / bk[i][j].abs().max(1e-12);
            let rm = (dm[i][j].re - bm[i][j]).abs() / bm[i][j].abs().max(1e-12);
            worst = worst.max(rk).max(rm);
        }
    }
    assert!(
        worst < 1e-10,
        "dual .re vs Burn kernel worst rel-err {worst:.3e}"
    );
}

/// **Element-kernel derivative is exact.** The dual tangents of the local
/// K, M and current-RHS moments ∫N_i must match a central finite difference
/// of the same `f64` kernel for every one of the twelve node coordinates —
/// proving `∂A/∂X` and `∂b/∂X` are analytic (forward-mode AD), not FD
/// approximations. A sign flip or dropped term in [`nedelec_local_dual`]
/// fails this immediately.
#[test]
fn dual_local_derivative_matches_finite_difference() {
    let base = [
        [0.10, 0.20, 0.05],
        [1.05, 0.15, 0.20],
        [0.25, 0.95, 0.10],
        [0.20, 0.30, 1.10],
    ];
    // Real f64 kernel = dual `.re` on all-constant coords.
    let eval = |coords: &[[f64; 3]; 4]| {
        let dc = coords.map(|v| v.map(Dual::cst));
        let (k, m, n) = nedelec_local_dual(&dc);
        let kre =
            std::array::from_fn::<_, 6, _>(|i| std::array::from_fn::<_, 6, _>(|j| k[i][j].re));
        let mre =
            std::array::from_fn::<_, 6, _>(|i| std::array::from_fn::<_, 6, _>(|j| m[i][j].re));
        let nre =
            std::array::from_fn::<_, 6, _>(|i| std::array::from_fn::<_, 3, _>(|c| n[i][c].re));
        (kre, mre, nre)
    };

    // Per-kernel normalization: compare the worst absolute dual-vs-FD gap
    // against that kernel's own derivative scale (max |FD| across entries).
    // A per-entry relative denominator would blow up on the small-magnitude
    // entries where the central FD itself loses conditioning (catastrophic
    // cancellation on the 1/|det|³ curl-curl terms) — the dual is exact, so
    // the scale-normalized gap is the honest fidelity measure.
    let h = 1e-6;
    let mut diff_k = 0.0_f64;
    let mut scale_k = 0.0_f64;
    let mut diff_m = 0.0_f64;
    let mut scale_m = 0.0_f64;
    let mut diff_n = 0.0_f64;
    let mut scale_n = 0.0_f64;
    for a in 0..4 {
        for c in 0..3 {
            let mut dc = base.map(|v| v.map(Dual::cst));
            dc[a][c] = Dual::var(base[a][c]);
            let (dk, dm, dn) = nedelec_local_dual(&dc);

            let mut cp = base;
            let mut cm = base;
            cp[a][c] += h;
            cm[a][c] -= h;
            let (kp, mp, np) = eval(&cp);
            let (km, mm, nm) = eval(&cm);

            for i in 0..6 {
                for j in 0..6 {
                    let fdk = (kp[i][j] - km[i][j]) / (2.0 * h);
                    let fdm = (mp[i][j] - mm[i][j]) / (2.0 * h);
                    diff_k = diff_k.max((dk[i][j].du - fdk).abs());
                    scale_k = scale_k.max(fdk.abs());
                    diff_m = diff_m.max((dm[i][j].du - fdm).abs());
                    scale_m = scale_m.max(fdm.abs());
                }
                for c2 in 0..3 {
                    let fdn = (np[i][c2] - nm[i][c2]) / (2.0 * h);
                    diff_n = diff_n.max((dn[i][c2].du - fdn).abs());
                    scale_n = scale_n.max(fdn.abs());
                }
            }
        }
    }
    // Each kernel must exercise a genuinely nonzero geometry sensitivity.
    assert!(
        scale_k > 1e-3 && scale_m > 1e-3 && scale_n > 1e-3,
        "kernel derivative scales too small (K {scale_k:.3e}, M {scale_m:.3e}, N {scale_n:.3e})"
    );
    let rel_k = diff_k / scale_k;
    let rel_m = diff_m / scale_m;
    let rel_n = diff_n / scale_n;
    let worst = rel_k.max(rel_m).max(rel_n);
    assert!(
        worst < 1e-6,
        "dual-vs-FD scale-normalized rel-err too large \
         (K {rel_k:.3e}, M {rel_m:.3e}, N {rel_n:.3e})"
    );
}

/// **The load-bearing test.** The driven-Nédélec discrete-adjoint **shape**
/// gradient `∂g/∂θ` — one forward + one adjoint solve + the geometry
/// Jacobian — must match a full central finite difference of the entire
/// driven pipeline (perturb θ → **move the nodes** → re-assemble the
/// Nédélec `A` and current RHS `b` on the moved mesh → re-solve → recompute
/// g), for two distinct node-motion maps, to a tight relative tolerance.
/// The FD arm drives the **public** [`driven_solve`] path, an independent
/// cross-check. A wrong sign, a wrong `∂A/∂X` / `∂b/∂X`, a dropped RHS-shape
/// term, or a conjugation error fails it.
#[test]
fn driven_shape_gradient_matches_central_finite_difference() {
    let (mesh, eps_r, interior, source) = cavity_fixture(3);
    let omega = 1.5;
    let bcs = DrivenBcs {
        pec_interior_mask: &interior,
    };

    // ONE forward + ONE adjoint solve → full nodal-coordinate gradient.
    let sg =
        driven_shape_gradient::<B, _>(&mesh, &eps_r, &bcs, omega, &source, l2_objective, &device())
            .expect("shape gradient");
    assert_eq!(
        sg.n_factorizations, 1,
        "shape adjoint must reuse the forward factorization (no refactorize)"
    );
    assert!(
        sg.residual_rel < 1e-9,
        "forward solve unhealthy (residual {:.3e}); pick ω off resonance",
        sg.residual_rel
    );

    // Full-pipeline objective as a function of θ under a node-velocity
    // field D: move nodes to X⁰ + θD, re-assemble + re-solve via the public
    // driven path (same fixed per-tet source), recompute g.
    let g_of_theta = |theta: f64, d: &[[f64; 3]]| -> f64 {
        let mut moved = mesh.clone();
        for (node, dn) in moved.nodes.iter_mut().zip(d) {
            node[0] += theta * dn[0];
            node[1] += theta * dn[1];
            node[2] += theta * dn[2];
        }
        let eps_c: Vec<c64> = eps_r.iter().map(|&e| c64::new(e, 0.0)).collect();
        let sol = driven_solve::<B>(
            &moved,
            DrivenMaterials::Scalar(&eps_c),
            &bcs,
            omega,
            &source,
            &device(),
        )
        .expect("driven solve");
        l2_objective(&sol.e_edges).0
    };

    // Objective must agree between the two forward paths (sanity).
    let g0_pub = g_of_theta(0.0, &vec![[0.0; 3]; mesh.n_nodes()]);
    assert!(
        (g0_pub - sg.objective).abs() <= 1e-9 * g0_pub.abs().max(1.0),
        "objective mismatch: shape adjoint {} vs public driven_solve {g0_pub}",
        sg.objective
    );

    // Two analytic node-motion maps, LINEAR in θ so X(θ)=X⁰+θD and the
    // constant velocity field D is exact.
    //   1. Translate ONLY the hi PEC face (x=1) in +x: stretches the last
    //      tet layer so the cavity gap (and the driven field) shifts.
    let tol = 1e-9;
    let d_face: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|p| {
            if (p[0] - 1.0).abs() < tol {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect();
    //   2. Move a single interior control node (nearest the domain centre)
    //      in +x — a localized one-node morph.
    let ctr = [0.5, 0.5, 0.5];
    let ctrl = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, p)| p.iter().all(|&c| c > tol && c < 1.0 - tol))
        .min_by(|(_, a), (_, b)| {
            let da = (a[0] - ctr[0]).powi(2) + (a[1] - ctr[1]).powi(2) + (a[2] - ctr[2]).powi(2);
            let db = (b[0] - ctr[0]).powi(2) + (b[1] - ctr[1]).powi(2) + (b[2] - ctr[2]).powi(2);
            da.partial_cmp(&db).unwrap()
        })
        .map(|(i, _)| i)
        .expect("mesh has an interior node");
    let mut d_node = vec![[0.0_f64; 3]; mesh.n_nodes()];
    d_node[ctrl] = [1.0, 0.0, 0.0];

    let h = 1e-6;
    for (name, d) in [
        ("hi-face-translate", &d_face),
        ("interior-control-node", &d_node),
    ] {
        let ana = chain_node_motion(&sg.grad_node, d);
        let fd = (g_of_theta(h, d) - g_of_theta(-h, d)) / (2.0 * h);
        let rel = (ana - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
        assert!(
            fd.abs() > 1e-6,
            "map {name}: FD gradient {fd} unexpectedly ~0 (fixture degenerate?)"
        );
        // Coords flow in full f64 (NdArray backend) and the source's ε/J are
        // held fixed, so the only residual is the FD's own O(h²) truncation
        // + solver round-off — orders below the 1e-3 issue spec.
        assert!(
            rel < 1e-3,
            "map {name}: adjoint {ana} vs central-FD {fd}, rel-err {rel:.3e} exceeds 1e-3"
        );
    }

    // The two maps must give DISTINCT gradients (they probe different
    // geometry perturbations), else the test could pass on a constant.
    let g_face = chain_node_motion(&sg.grad_node, &d_face);
    let g_node = chain_node_motion(&sg.grad_node, &d_node);
    assert!(
        (g_face - g_node).abs() > 1e-6,
        "the two node-motion maps must yield distinct gradients ({g_face} vs {g_node})"
    );
}

/// Mutation tripwire: the finite-difference check must **reject** a shape
/// gradient built with the classic complex-adjoint conjugation error (the
/// `Aᴴ` mistake, ≡ feeding the wrong Wirtinger cotangent `∂g/∂x̄` instead of
/// `∂g/∂x`). The field is genuinely complex, so conjugation changes the
/// answer and the FD rejects it — proving the load-bearing test's tolerance
/// is biting, not vacuously satisfied.
#[test]
fn conjugation_error_is_detected_by_fd() {
    let (mesh, eps_r, interior, source) = cavity_fixture(3);
    let omega = 1.5;
    let bcs = DrivenBcs {
        pec_interior_mask: &interior,
    };

    // Wrong cotangent: +i·Im instead of −i·Im (∂g/∂x̄ rather than ∂g/∂x).
    let wrong_objective = |x: &[c64]| -> (f64, Vec<c64>) {
        let g = x.iter().map(|z| z.re * z.re + z.im * z.im).sum::<f64>();
        let cot = x.iter().map(|z| c64::new(z.re, z.im)).collect();
        (g, cot)
    };
    let wrong = driven_shape_gradient::<B, _>(
        &mesh,
        &eps_r,
        &bcs,
        omega,
        &source,
        wrong_objective,
        &device(),
    )
    .expect("wrong-conjugation shape gradient");

    // Independent FD reference (public driven path).
    let g_of_theta = |theta: f64, d: &[[f64; 3]]| -> f64 {
        let mut moved = mesh.clone();
        for (node, dn) in moved.nodes.iter_mut().zip(d) {
            node[0] += theta * dn[0];
            node[1] += theta * dn[1];
            node[2] += theta * dn[2];
        }
        let eps_c: Vec<c64> = eps_r.iter().map(|&e| c64::new(e, 0.0)).collect();
        let sol = driven_solve::<B>(
            &moved,
            DrivenMaterials::Scalar(&eps_c),
            &bcs,
            omega,
            &source,
            &device(),
        )
        .expect("driven solve");
        l2_objective(&sol.e_edges).0
    };

    // A morph with a genuinely nonzero, complex-sensitive gradient.
    let tol = 1e-9;
    let d_face: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|p| {
            if (p[0] - 1.0).abs() < tol {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect();
    let h = 1e-6;
    let fd = (g_of_theta(h, &d_face) - g_of_theta(-h, &d_face)) / (2.0 * h);
    let ana_wrong = chain_node_motion(&wrong.grad_node, &d_face);
    let rel = (ana_wrong - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel > 1e-2,
        "conjugation-error gradient {ana_wrong} matched the FD {fd} (rel {rel:.3e}) — \
         the tolerance is not biting"
    );
}

// ─────────────────────────────────────────────────────────────────────
// COMPLEX-ε (lossy substrate) shape-gradient tests (issue #629, #628 B).
// ─────────────────────────────────────────────────────────────────────

/// Uniform **lossy** complex-ε fixture `ε = ε′ − i·ε″` (nonzero `Im(ε)`, a
/// substrate loss tangent `tan δ = ε″/ε′`), driven by the same genuinely
/// complex source as [`cavity_fixture`], for the complex-ε shape-gradient
/// tests. Returns `(mesh, eps_complex, interior_mask, source)`.
fn cavity_fixture_complex(n: usize) -> (TetMesh, Vec<c64>, Vec<bool>, CurrentSource) {
    let (mesh, eps_r, interior, source) = cavity_fixture(n);
    // ε′ from the lossless fixture (2.0), ε″ = 0.3 → tan δ = 0.15: a big,
    // unambiguous loss so the complex-ε contraction path is genuinely
    // exercised (not a near-lossless perturbation the FD could miss).
    let eps_c: Vec<c64> = eps_r.iter().map(|&e| c64::new(e, -0.3)).collect();
    (mesh, eps_c, interior, source)
}

/// **The load-bearing complex-ε test.** The lossy driven-Nédélec
/// discrete-adjoint **shape** gradient `∂g/∂θ` — one forward + one adjoint
/// solve on the complex-symmetric pencil `A = K − ω² M(ε)` with `Im(ε) ≠ 0`
/// — must match a full central finite difference of the entire complex
/// driven pipeline (perturb θ → move nodes → re-assemble the complex `A` and
/// current RHS `b` → re-solve → recompute g), for two distinct node-motion
/// maps, to the ≤1e-3 issue tolerance. The FD arm drives the **public**
/// [`driven_solve`] complex `Scalar(ε)` path, an independent cross-check
/// against the shape adjoint's own forward. A dropped loss term, a wrong
/// sign, or a conjugation error fails it.
#[test]
fn driven_shape_gradient_complex_matches_central_finite_difference() {
    let (mesh, eps_c, interior, source) = cavity_fixture_complex(3);
    let omega = 1.5;
    let bcs = DrivenBcs {
        pec_interior_mask: &interior,
    };

    // ONE forward + ONE adjoint solve → full nodal-coordinate gradient.
    let sg = driven_shape_gradient_complex::<B, _>(
        &mesh,
        &eps_c,
        &bcs,
        omega,
        &source,
        l2_objective,
        &device(),
    )
    .expect("complex shape gradient");
    assert_eq!(
        sg.n_factorizations, 1,
        "complex shape adjoint must reuse the forward factorization (no refactorize)"
    );
    assert!(
        sg.residual_rel < 1e-9,
        "forward solve unhealthy (residual {:.3e}); pick ω off resonance",
        sg.residual_rel
    );

    // Full-pipeline objective under a node-velocity field D, via the public
    // complex driven path (Scalar takes the c64 ε directly).
    let g_of_theta = |theta: f64, d: &[[f64; 3]]| -> f64 {
        let mut moved = mesh.clone();
        for (node, dn) in moved.nodes.iter_mut().zip(d) {
            node[0] += theta * dn[0];
            node[1] += theta * dn[1];
            node[2] += theta * dn[2];
        }
        let sol = driven_solve::<B>(
            &moved,
            DrivenMaterials::Scalar(&eps_c),
            &bcs,
            omega,
            &source,
            &device(),
        )
        .expect("driven solve");
        l2_objective(&sol.e_edges).0
    };

    // Objective must agree between the two forward paths (sanity).
    let g0_pub = g_of_theta(0.0, &vec![[0.0; 3]; mesh.n_nodes()]);
    assert!(
        (g0_pub - sg.objective).abs() <= 1e-9 * g0_pub.abs().max(1.0),
        "objective mismatch: shape adjoint {} vs public driven_solve {g0_pub}",
        sg.objective
    );

    // Two analytic node-motion maps, LINEAR in θ.
    let tol = 1e-9;
    let d_face: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|p| {
            if (p[0] - 1.0).abs() < tol {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect();
    let ctr = [0.5, 0.5, 0.5];
    let ctrl = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, p)| p.iter().all(|&c| c > tol && c < 1.0 - tol))
        .min_by(|(_, a), (_, b)| {
            let da = (a[0] - ctr[0]).powi(2) + (a[1] - ctr[1]).powi(2) + (a[2] - ctr[2]).powi(2);
            let db = (b[0] - ctr[0]).powi(2) + (b[1] - ctr[1]).powi(2) + (b[2] - ctr[2]).powi(2);
            da.partial_cmp(&db).unwrap()
        })
        .map(|(i, _)| i)
        .expect("mesh has an interior node");
    let mut d_node = vec![[0.0_f64; 3]; mesh.n_nodes()];
    d_node[ctrl] = [1.0, 0.0, 0.0];

    let h = 1e-6;
    for (name, d) in [
        ("hi-face-translate", &d_face),
        ("interior-control-node", &d_node),
    ] {
        let ana = chain_node_motion(&sg.grad_node, d);
        let fd = (g_of_theta(h, d) - g_of_theta(-h, d)) / (2.0 * h);
        assert!(
            fd.abs() > 1e-6,
            "map {name}: FD gradient {fd} unexpectedly ~0 (fixture degenerate?)"
        );
        let rel = (ana - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
        assert!(
            rel < 1e-3,
            "map {name}: complex adjoint {ana} vs central-FD {fd}, rel-err {rel:.3e} exceeds 1e-3"
        );
    }

    // The two maps must give DISTINCT gradients.
    let g_face = chain_node_motion(&sg.grad_node, &d_face);
    let g_node = chain_node_motion(&sg.grad_node, &d_node);
    assert!(
        (g_face - g_node).abs() > 1e-6,
        "the two node-motion maps must yield distinct gradients ({g_face} vs {g_node})"
    );
}

/// Mutation tripwire (complex-ε): the FD check must **reject** a lossy shape
/// gradient built with the classic complex-adjoint conjugation error
/// (feeding `∂g/∂x̄` instead of `∂g/∂x`). With `Im(ε) ≠ 0` the field is
/// even more thoroughly complex, so the conjugation error is unmistakable —
/// proving the load-bearing complex test's tolerance is biting.
#[test]
fn complex_conjugation_error_is_detected_by_fd() {
    let (mesh, eps_c, interior, source) = cavity_fixture_complex(3);
    let omega = 1.5;
    let bcs = DrivenBcs {
        pec_interior_mask: &interior,
    };

    // Wrong cotangent: +i·Im instead of −i·Im (∂g/∂x̄ rather than ∂g/∂x).
    let wrong_objective = |x: &[c64]| -> (f64, Vec<c64>) {
        let g = x.iter().map(|z| z.re * z.re + z.im * z.im).sum::<f64>();
        let cot = x.iter().map(|z| c64::new(z.re, z.im)).collect();
        (g, cot)
    };
    let wrong = driven_shape_gradient_complex::<B, _>(
        &mesh,
        &eps_c,
        &bcs,
        omega,
        &source,
        wrong_objective,
        &device(),
    )
    .expect("wrong-conjugation complex shape gradient");

    let g_of_theta = |theta: f64, d: &[[f64; 3]]| -> f64 {
        let mut moved = mesh.clone();
        for (node, dn) in moved.nodes.iter_mut().zip(d) {
            node[0] += theta * dn[0];
            node[1] += theta * dn[1];
            node[2] += theta * dn[2];
        }
        let sol = driven_solve::<B>(
            &moved,
            DrivenMaterials::Scalar(&eps_c),
            &bcs,
            omega,
            &source,
            &device(),
        )
        .expect("driven solve");
        l2_objective(&sol.e_edges).0
    };

    let tol = 1e-9;
    let d_face: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|p| {
            if (p[0] - 1.0).abs() < tol {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect();
    let h = 1e-6;
    let fd = (g_of_theta(h, &d_face) - g_of_theta(-h, &d_face)) / (2.0 * h);
    let ana_wrong = chain_node_motion(&wrong.grad_node, &d_face);
    let rel = (ana_wrong - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel > 1e-2,
        "complex conjugation-error gradient {ana_wrong} matched the FD {fd} (rel {rel:.3e}) — \
         the tolerance is not biting"
    );
}

/// **Backward-compat / delegation guard.** At zero loss the complex entry
/// point must reproduce the real [`driven_shape_gradient`] bit-for-bit (the
/// real path is just this core with `ε″ = 0`). Guards against the real
/// signature silently diverging from the shared implementation.
#[test]
fn real_path_equals_complex_at_zero_loss() {
    let (mesh, eps_r, interior, source) = cavity_fixture(3);
    let omega = 1.5;
    let bcs = DrivenBcs {
        pec_interior_mask: &interior,
    };
    let real =
        driven_shape_gradient::<B, _>(&mesh, &eps_r, &bcs, omega, &source, l2_objective, &device())
            .expect("real shape gradient");
    let eps_c: Vec<c64> = eps_r.iter().map(|&e| c64::new(e, 0.0)).collect();
    let cplx = driven_shape_gradient_complex::<B, _>(
        &mesh,
        &eps_c,
        &bcs,
        omega,
        &source,
        l2_objective,
        &device(),
    )
    .expect("complex shape gradient at zero loss");
    assert_eq!(real.n_factorizations, cplx.n_factorizations);
    let mut worst = 0.0_f64;
    for (r, c) in real.grad_node.iter().zip(cplx.grad_node.iter()) {
        for d in 0..3 {
            worst = worst.max((r[d] - c[d]).abs());
        }
    }
    assert!(
        worst < 1e-12,
        "real vs complex-at-zero-loss shape gradient differ by {worst:.3e}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// PINNED-FEED LUMPED-PORT shape-gradient tests (issue #631, Epic #628 A1).
// ─────────────────────────────────────────────────────────────────────

use crate::driven::extraction::{s11, s11_sq_and_dg_dv, s11_sq_objective};
use crate::driven::ports::{port_input_impedance, port_voltage};
use crate::driven::solve::driven_solve_with_ports;

/// Boundary faces of `mesh` lying entirely in the plane `coord[axis]==value`.
/// (House twin of the `lumped_port.rs` integration-test helper.)
fn plane_faces(mesh: &TetMesh, axis: usize, value: f64) -> Vec<[u32; 3]> {
    mesh.faces()
        .into_iter()
        .filter(|f| {
            f.iter()
                .all(|&n| (mesh.nodes[n as usize][axis] - value).abs() < 1e-12)
        })
        .collect()
}

/// PEC interior-edge mask eliminating every edge whose **both** endpoints lie
/// on the same listed plane `(axis, value)`. The port plane is deliberately
/// left OUT of the eliminated set, so its edges stay interior and the port
/// admittance term is live.
fn pec_mask_for_planes(mesh: &TetMesh, edges: &[[u32; 2]], planes: &[(usize, f64)]) -> Vec<bool> {
    edges
        .iter()
        .map(|e| {
            let a = mesh.nodes[e[0] as usize];
            let b = mesh.nodes[e[1] as usize];
            !planes.iter().any(|&(axis, value)| {
                (a[axis] - value).abs() < 1e-12 && (b[axis] - value).abs() < 1e-12
            })
        })
        .collect()
}

/// Pinned-port fixture: unit-cube parallel-plate line — PEC plates at
/// `y = 0/1`, a PEC short at `z = 1`, natural/PMC side walls at `x = 0/1` —
/// with the lumped feed across the `z = 0` face (`ê = ŷ`). A **lossy**
/// substrate `ε = 2 − 0.3i` (so the terminated `|S11| < 1` strictly) plus a
/// genuinely COMPLEX volume source (so the volume `∂b/∂X` term is exercised
/// alongside the port load). Returns `(mesh, eps_complex, interior_mask,
/// source)`; the caller builds the `LumpedPort` (it must borrow the
/// `plane_faces(z=0)` list).
fn port_line_fixture(n: usize) -> (TetMesh, Vec<c64>, Vec<bool>, CurrentSource) {
    let mesh = cube_tet_mesh(n, 1.0);
    let edges = mesh.edges();
    let mask = pec_mask_for_planes(&mesh, &edges, &[(1, 0.0), (1, 1.0), (2, 1.0)]);
    let eps = vec![c64::new(2.0, -0.3); mesh.n_tets()];
    let pi = std::f64::consts::PI;
    let source = CurrentSource::from_centroids(&mesh, |c| {
        [
            c64::new(0.0, 0.2 * (pi * c[2]).sin()),
            c64::new(0.3 * (pi * c[1]).sin(), 0.15),
            c64::new(0.1 * (pi * c[0]).sin(), 0.0),
        ]
    });
    (mesh, eps, mask, source)
}

/// **The load-bearing A1 test.** The pinned-feed **port-loaded** driven-
/// Nédélec shape gradient `∂g/∂θ` — one forward + one adjoint solve on the
/// port-loaded complex-symmetric pencil `A = K − ω²M(ε) + (jω/Z_s)S_p` with
/// the port boundary drive in `b` — must match a full central finite
/// difference of the entire **port-loaded** driven pipeline. The FD arm
/// independently RE-ASSEMBLES + RE-SOLVES the port forward via the public
/// [`driven_solve_with_ports`] at moved nodes (not the adjoint's own
/// forward). Both node-motion maps hold every `z = 0` port node fixed — the
/// pinned-feed premise (`∂S_p/∂X = ∂b_port/∂X = 0`). A wrong sign, a dropped
/// port term, or a conjugation error fails it.
#[test]
fn driven_shape_gradient_with_pinned_port_matches_central_finite_difference() {
    let (mesh, eps_c, mask, source) = port_line_fixture(4);
    let port_faces = plane_faces(&mesh, 2, 0.0);
    assert!(!port_faces.is_empty(), "port surface must be non-empty");
    let omega = 1.3;
    let bcs = DrivenBcs {
        pec_interior_mask: &mask,
    };
    let port = LumpedPort {
        faces: &port_faces,
        e_hat: [0.0, 1.0, 0.0],
        resistance: 1.0,
        width: 1.0,
        length: 1.0,
        v_inc: c64::new(1.0, 0.5), // genuinely complex drive
    };

    // ONE forward + ONE adjoint solve on the PORT-LOADED pencil.
    let sg = driven_shape_gradient_ports_complex::<B, _>(
        &mesh,
        &eps_c,
        &bcs,
        omega,
        &source,
        std::slice::from_ref(&port),
        l2_objective,
        &device(),
    )
    .expect("port-loaded shape gradient");
    assert_eq!(
        sg.n_factorizations, 1,
        "port-loaded adjoint must reuse the forward LU (complex-symmetric)"
    );
    assert!(
        sg.residual_rel < 1e-9,
        "port-loaded forward unhealthy (residual {:.3e}); pick ω off resonance",
        sg.residual_rel
    );

    // Independent FD reference: RE-ASSEMBLE + RE-SOLVE the PORT-LOADED
    // forward via the public `driven_solve_with_ports` at moved nodes.
    let g_of_theta = |theta: f64, d: &[[f64; 3]]| -> f64 {
        let mut moved = mesh.clone();
        for (node, dn) in moved.nodes.iter_mut().zip(d) {
            node[0] += theta * dn[0];
            node[1] += theta * dn[1];
            node[2] += theta * dn[2];
        }
        let sol = driven_solve_with_ports::<B>(
            &moved,
            DrivenMaterials::Scalar(&eps_c),
            None,
            &bcs,
            std::slice::from_ref(&port),
            omega,
            &source,
            &device(),
        )
        .expect("port forward");
        l2_objective(&sol.e_edges).0
    };

    // The adjoint's own port-loaded forward must equal the public port
    // forward (proves the differentiated pencil IS the public system).
    let g0 = g_of_theta(0.0, &vec![[0.0; 3]; mesh.n_nodes()]);
    assert!(
        (g0 - sg.objective).abs() <= 1e-9 * g0.abs().max(1.0),
        "objective mismatch: adjoint {} vs public port forward {g0}",
        sg.objective
    );

    // Pinned-feed node-motion maps (both hold every z=0 port node fixed):
    //   1. translate the PEC short (z = 1) in +x — the port at z = 0 is
    //      untouched, so ∂S_p/∂X = ∂b_port/∂X = 0 holds exactly.
    //   2. an interior control node near the centre (off the port plane).
    let tol = 1e-9;
    let d_short: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|p| {
            if (p[2] - 1.0).abs() < tol {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect();
    // Guard the pinned-feed premise: no port-face node is moved.
    for f in &port_faces {
        for &nn in f {
            assert_eq!(
                d_short[nn as usize], [0.0; 3],
                "d_short moves a port-face node — pinned-feed premise violated"
            );
        }
    }
    let ctr = [0.5, 0.5, 0.5];
    let ctrl = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, p)| p.iter().all(|&c| c > tol && c < 1.0 - tol))
        .min_by(|(_, a), (_, b)| {
            let da = (a[0] - ctr[0]).powi(2) + (a[1] - ctr[1]).powi(2) + (a[2] - ctr[2]).powi(2);
            let db = (b[0] - ctr[0]).powi(2) + (b[1] - ctr[1]).powi(2) + (b[2] - ctr[2]).powi(2);
            da.partial_cmp(&db).unwrap()
        })
        .map(|(i, _)| i)
        .expect("mesh has an interior node");
    assert!(
        mesh.nodes[ctrl][2].abs() > tol,
        "interior control node must be off the z=0 port plane"
    );
    let mut d_node = vec![[0.0_f64; 3]; mesh.n_nodes()];
    d_node[ctrl] = [1.0, 0.0, 0.0];

    let h = 1e-6;
    for (name, d) in [("short-translate", &d_short), ("interior-node", &d_node)] {
        let ana = chain_node_motion(&sg.grad_node, d);
        let fd = (g_of_theta(h, d) - g_of_theta(-h, d)) / (2.0 * h);
        assert!(
            fd.abs() > 1e-6,
            "map {name}: FD gradient {fd} unexpectedly ~0 (fixture degenerate?)"
        );
        let rel = (ana - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
        println!("port map {name}: adjoint {ana:.6}, central-FD {fd:.6}, rel-err {rel:.3e}");
        assert!(
            rel < 1e-3,
            "map {name}: port adjoint {ana} vs central-FD {fd}, rel-err {rel:.3e} exceeds 1e-3"
        );
    }

    // The two maps must give DISTINCT gradients.
    let g_short = chain_node_motion(&sg.grad_node, &d_short);
    let g_node = chain_node_motion(&sg.grad_node, &d_node);
    assert!(
        (g_short - g_node).abs() > 1e-6,
        "the two node-motion maps must yield distinct gradients ({g_short} vs {g_node})"
    );
}

/// Mutation tripwire (port-loaded): the FD check must **reject** a
/// port-loaded shape gradient built with the classic complex-adjoint
/// conjugation error (`∂g/∂x̄` instead of `∂g/∂x`) — proving the load-bearing
/// A1 test's tolerance is biting on the genuinely complex port-driven field.
#[test]
fn port_conjugation_error_is_detected_by_fd() {
    let (mesh, eps_c, mask, source) = port_line_fixture(4);
    let port_faces = plane_faces(&mesh, 2, 0.0);
    let omega = 1.3;
    let bcs = DrivenBcs {
        pec_interior_mask: &mask,
    };
    let port = LumpedPort {
        faces: &port_faces,
        e_hat: [0.0, 1.0, 0.0],
        resistance: 1.0,
        width: 1.0,
        length: 1.0,
        v_inc: c64::new(1.0, 0.5),
    };

    // Wrong cotangent: +i·Im instead of −i·Im (∂g/∂x̄ rather than ∂g/∂x).
    let wrong_objective = |x: &[c64]| -> (f64, Vec<c64>) {
        let g = x.iter().map(|z| z.re * z.re + z.im * z.im).sum::<f64>();
        let cot = x.iter().map(|z| c64::new(z.re, z.im)).collect();
        (g, cot)
    };
    let wrong = driven_shape_gradient_ports_complex::<B, _>(
        &mesh,
        &eps_c,
        &bcs,
        omega,
        &source,
        std::slice::from_ref(&port),
        wrong_objective,
        &device(),
    )
    .expect("wrong-conjugation port shape gradient");

    let g_of_theta = |theta: f64, d: &[[f64; 3]]| -> f64 {
        let mut moved = mesh.clone();
        for (node, dn) in moved.nodes.iter_mut().zip(d) {
            node[0] += theta * dn[0];
            node[1] += theta * dn[1];
            node[2] += theta * dn[2];
        }
        let sol = driven_solve_with_ports::<B>(
            &moved,
            DrivenMaterials::Scalar(&eps_c),
            None,
            &bcs,
            std::slice::from_ref(&port),
            omega,
            &source,
            &device(),
        )
        .expect("port forward");
        l2_objective(&sol.e_edges).0
    };

    let tol = 1e-9;
    let d_short: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|p| {
            if (p[2] - 1.0).abs() < tol {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect();
    let h = 1e-6;
    let fd = (g_of_theta(h, &d_short) - g_of_theta(-h, &d_short)) / (2.0 * h);
    let ana_wrong = chain_node_motion(&wrong.grad_node, &d_short);
    let rel = (ana_wrong - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel > 1e-2,
        "port conjugation-error gradient {ana_wrong} matched the FD {fd} (rel {rel:.3e}) — \
         the tolerance is not biting"
    );
}

/// The port load must **genuinely change** the shape gradient (the whole
/// point of A1): differentiating the port-loaded pencil is not the same as
/// the port-less pencil. Same mesh / ε / source / ω, port present vs absent.
#[test]
fn pinned_port_changes_the_shape_gradient() {
    let (mesh, eps_c, mask, source) = port_line_fixture(3);
    let port_faces = plane_faces(&mesh, 2, 0.0);
    let omega = 1.3;
    let bcs = DrivenBcs {
        pec_interior_mask: &mask,
    };
    let port = LumpedPort {
        faces: &port_faces,
        e_hat: [0.0, 1.0, 0.0],
        resistance: 1.0,
        width: 1.0,
        length: 1.0,
        v_inc: c64::new(1.0, 0.5),
    };
    let with_port = driven_shape_gradient_ports_complex::<B, _>(
        &mesh,
        &eps_c,
        &bcs,
        omega,
        &source,
        std::slice::from_ref(&port),
        l2_objective,
        &device(),
    )
    .expect("port-loaded gradient");
    let no_port = driven_shape_gradient_complex::<B, _>(
        &mesh,
        &eps_c,
        &bcs,
        omega,
        &source,
        l2_objective,
        &device(),
    )
    .expect("port-less gradient");
    let mut worst = 0.0_f64;
    for (a, b) in with_port.grad_node.iter().zip(no_port.grad_node.iter()) {
        for d in 0..3 {
            worst = worst.max((a[d] - b[d]).abs());
        }
    }
    assert!(
        worst > 1e-6,
        "the pinned-port load must change the shape gradient (max diff {worst:.3e})"
    );
}

/// **Backward-compat / delegation guard.** An empty port slice through the
/// new [`driven_shape_gradient_ports_complex`] must reproduce the port-less
/// [`driven_shape_gradient_complex`] bit-for-bit — the historical Phase B
/// path is exactly this core with no ports.
#[test]
fn empty_ports_equals_portless_shape_gradient() {
    let (mesh, eps_c, interior, source) = cavity_fixture_complex(3);
    let omega = 1.5;
    let bcs = DrivenBcs {
        pec_interior_mask: &interior,
    };
    let portless = driven_shape_gradient_complex::<B, _>(
        &mesh,
        &eps_c,
        &bcs,
        omega,
        &source,
        l2_objective,
        &device(),
    )
    .expect("port-less gradient");
    let empty = driven_shape_gradient_ports_complex::<B, _>(
        &mesh,
        &eps_c,
        &bcs,
        omega,
        &source,
        &[],
        l2_objective,
        &device(),
    )
    .expect("empty-ports gradient");
    assert_eq!(portless.n_factorizations, empty.n_factorizations);
    let mut worst = 0.0_f64;
    for (a, b) in portless.grad_node.iter().zip(empty.grad_node.iter()) {
        for d in 0..3 {
            worst = worst.max((a[d] - b[d]).abs());
        }
    }
    assert!(
        worst == 0.0,
        "empty ports diverged from the port-less path by {worst:.3e}"
    );
}

/// **Physical sanity (issue #631 AC2).** With the port termination present,
/// the reflection coefficient `|S11|` read off the port-terminated forward
/// is a **bounded** reflection (`≤ 1`) on a passive fixture — a marked
/// change from #627's synthetic `|S11|² > 1`, which differentiated a pencil
/// with NO port termination (so its extracted `Z = V/I` was not a passive
/// impedance). The lossy substrate makes it a strict contraction `< 1`.
#[test]
fn port_terminated_s11_is_bounded_on_passive_fixture() {
    let (mesh, eps_c, mask, _src) = port_line_fixture(4);
    let edges = mesh.edges();
    let port_faces = plane_faces(&mesh, 2, 0.0);
    let omega = 1.3;
    let bcs = DrivenBcs {
        pec_interior_mask: &mask,
    };
    // Pure port drive (zero volume source): the reflection is then the
    // structure's genuine S11 seen at the port.
    let zero_src = CurrentSource {
        j_tet: vec![[c64::new(0.0, 0.0); 3]; mesh.n_tets()],
    };
    let r = 1.0;
    let port = LumpedPort {
        faces: &port_faces,
        e_hat: [0.0, 1.0, 0.0],
        resistance: r,
        width: 1.0,
        length: 1.0,
        v_inc: c64::new(1.0, 0.0),
    };
    let sol = driven_solve_with_ports::<B>(
        &mesh,
        DrivenMaterials::Scalar(&eps_c),
        None,
        &bcs,
        std::slice::from_ref(&port),
        omega,
        &zero_src,
        &device(),
    )
    .expect("port-terminated forward");
    assert!(
        sol.residual_rel < 1e-9,
        "forward unhealthy (residual {:.3e})",
        sol.residual_rel
    );
    let z_in = port_input_impedance(&mesh, &port, &edges, &sol.e_edges);
    let mag = s11(z_in, r).norm();
    println!("port-terminated |S11| = {mag:.6}  (Z_in = {z_in})");
    // Passivity: |S11| ≤ 1 for any passive one-port (the provable bound).
    assert!(
        mag <= 1.0 + 1e-9,
        "passive port |S11| = {mag} exceeds 1 — not a bounded reflection coefficient"
    );
    // Dissipative substrate ⇒ strict contraction (not the lossless |S11| = 1
    // marginal case), confirming the termination absorbs real power.
    assert!(
        mag < 1.0,
        "lossy passive fixture should give |S11| < 1, got {mag}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MOVING-FEED lumped-port shape adjoint (Epic #628 Phase A2, issue #633).
// ─────────────────────────────────────────────────────────────────────

/// Geometry-derived port dims `(width, length)` — the SAME construction as
/// [`crate::mesh::patch::PatchFixture::port`]: `length = hi − lo` along `ê`,
/// `width = (Σ face areas) / length`. The moving-feed FD reference recomputes
/// these at each perturbed mesh (independent of the adjoint's own formulas).
fn port_dims(mesh: &TetMesh, faces: &[[u32; 3]], e_hat: [f64; 3]) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut area = 0.0_f64;
    for tri in faces {
        let v: [[f64; 3]; 3] = std::array::from_fn(|k| mesh.nodes[tri[k] as usize]);
        for p in &v {
            let along = p[0] * e_hat[0] + p[1] * e_hat[1] + p[2] * e_hat[2];
            lo = lo.min(along);
            hi = hi.max(along);
        }
        let e1 = [v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]];
        let e2 = [v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]];
        let cx = e1[1] * e2[2] - e1[2] * e2[1];
        let cy = e1[2] * e2[0] - e1[0] * e2[2];
        let cz = e1[0] * e2[1] - e1[1] * e2[0];
        area += 0.5 * (cx * cx + cy * cy + cz * cz).sqrt();
    }
    let l = hi - lo;
    (area / l, l)
}

/// **Surface-Whitney Dual twin `.re` faithfully lifts the real port kernels.**
/// Scattering the UNSIGNED `port_face_dual` `.re` blocks with the lower-tag
/// signs must reproduce the public [`assemble_port_surface_mass`] and
/// [`assemble_port_flux`] at zero perturbation — proving the twin
/// differentiates *the same* closed forms the forward assembly uses.
#[test]
fn port_face_dual_re_reproduces_real_kernels() {
    use crate::elements::whitney::{edge_lookup, face_geometry};
    let mesh = cube_tet_mesh(2, 1.0);
    let edges = mesh.edges();
    let faces = plane_faces(&mesh, 2, 0.0);
    assert!(!faces.is_empty());
    let e_hat = [0.0, 1.0, 0.0];

    let n = edges.len();
    let mut s_real = vec![0.0_f64; n * n];
    for (r, c, v) in assemble_port_surface_mass(&mesh, &faces, &edges) {
        s_real[r * n + c] += v;
    }
    let f_real = assemble_port_flux(&mesh, &faces, e_hat, &edges);

    let lookup = edge_lookup(&edges);
    let mut s_dual = vec![0.0_f64; n * n];
    let mut f_dual = vec![0.0_f64; n];
    for tri in &faces {
        let v_real: [[f64; 3]; 3] = std::array::from_fn(|k| mesh.nodes[tri[k] as usize]);
        let info = face_geometry(tri, &v_real, &lookup).edge_info;
        let vd = v_real.map(|p| p.map(Dual::cst));
        let (_area, mass_d, flux_d) = port_face_dual(&vd, e_hat);
        for i in 0..3 {
            let (gi, si) = info[i];
            f_dual[gi as usize] += flux_d[i].re * (si as f64);
            for j in 0..3 {
                let (gj, sj) = info[j];
                s_dual[gi as usize * n + gj as usize] +=
                    mass_d[i][j].re * (si as f64) * (sj as f64);
            }
        }
    }
    let mut max_s = 0.0_f64;
    let mut max_f = 0.0_f64;
    for k in 0..n {
        max_f = max_f.max((f_dual[k] - f_real[k]).abs());
        for j in 0..n {
            max_s = max_s.max((s_dual[k * n + j] - s_real[k * n + j]).abs());
        }
    }
    assert!(max_s < 1e-13, "dual S_p .re vs real: {max_s}");
    assert!(max_f < 1e-13, "dual flux .re vs real: {max_f}");
}

/// **Surface-Whitney Dual twin `.du` matches central FD.** On a generic
/// (non-axis-aligned) triangle, each `.du` of the face area, surface-mass
/// block, and flux moment must match a central finite difference of the
/// twin's own `.re` — the element-level `∂S_p/∂X`, `∂f/∂X` gate.
#[test]
fn port_face_dual_derivative_matches_finite_difference() {
    let v0 = [[0.10, 0.20, 0.05], [0.90, 0.15, 0.12], [0.25, 0.85, -0.08]];
    let e_hat = {
        let e = [0.3_f64, 0.6, -0.2];
        let n = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
        [e[0] / n, e[1] / n, e[2] / n]
    };
    let h = 1e-6;
    let mut worst = 0.0_f64;
    for a in 0..3 {
        for c in 0..3 {
            let mut vd = v0.map(|p| p.map(Dual::cst));
            vd[a][c] = Dual::var(v0[a][c]);
            let (area_d, mass_d, flux_d) = port_face_dual(&vd, e_hat);

            let eval = |val: f64| -> (f64, [[f64; 3]; 3], [f64; 3]) {
                let mut vv = v0;
                vv[a][c] = val;
                let vc = vv.map(|p| p.map(Dual::cst));
                let (ar, mm, ff) = port_face_dual(&vc, e_hat);
                let mut m = [[0.0; 3]; 3];
                let mut f = [0.0; 3];
                for i in 0..3 {
                    f[i] = ff[i].re;
                    for j in 0..3 {
                        m[i][j] = mm[i][j].re;
                    }
                }
                (ar.re, m, f)
            };
            let (ap, mp, fp) = eval(v0[a][c] + h);
            let (am, mm, fm) = eval(v0[a][c] - h);
            worst = worst.max((area_d.du - (ap - am) / (2.0 * h)).abs());
            for i in 0..3 {
                worst = worst.max((flux_d[i].du - (fp[i] - fm[i]) / (2.0 * h)).abs());
                for j in 0..3 {
                    worst = worst.max((mass_d[i][j].du - (mp[i][j] - mm[i][j]) / (2.0 * h)).abs());
                }
            }
        }
    }
    assert!(
        worst < 1e-7,
        "port_face_dual .du vs central FD: worst abs err {worst:.3e}"
    );
}

/// **The load-bearing A2 test.** The moving-feed port-loaded driven-Nédélec
/// `|S₁₁|²` shape gradient — one forward + one adjoint solve on the
/// complex-symmetric pencil, plus the extra `∂S_p/∂X`, `∂b_port/∂X`,
/// `∂Z_s/∂X` and objective `∂f/∂X` contractions — must match a full central
/// finite difference of the entire moving-feed pipeline. The FD arm
/// RE-ASSEMBLES the geometry-derived port dims (`w = area/l`, `l = hi−lo`)
/// AND re-solves the port forward at moved nodes. The fixture is genuinely
/// complex (ε = 2 − 0.3i, complex `V_inc`, complex source), so a dropped
/// port term, a wrong sign, or a conjugation error in the new covector paths
/// breaks the match.
#[test]
fn moving_feed_s11_shape_gradient_matches_central_finite_difference() {
    let (mesh, eps_c, mask, source) = port_line_fixture(4);
    let port_faces = plane_faces(&mesh, 2, 0.0);
    assert!(!port_faces.is_empty());
    let e_hat = [0.0, 1.0, 0.0];
    let omega = 1.3;
    let r = 1.0;
    let z0 = r;
    let bcs = DrivenBcs {
        pec_interior_mask: &mask,
    };
    let (w0, l0) = port_dims(&mesh, &port_faces, e_hat);
    let port = LumpedPort {
        faces: &port_faces,
        e_hat,
        resistance: r,
        width: w0,
        length: l0,
        v_inc: c64::new(1.0, 0.5),
    };

    let sg = driven_shape_gradient_moving_port_s11::<B>(
        &mesh,
        &eps_c,
        &bcs,
        omega,
        &source,
        &port,
        z0,
        &device(),
    )
    .expect("moving-feed shape gradient");
    assert_eq!(
        sg.n_factorizations, 1,
        "moving-feed adjoint must reuse the single forward LU"
    );
    assert!(
        sg.residual_rel < 1e-9,
        "moving-feed forward unhealthy (residual {:.3e})",
        sg.residual_rel
    );

    let edges = mesh.edges();
    // Independent FD reference: recompute geometry-derived port dims + the
    // |S₁₁|² objective from a re-solved moving-feed forward at moved nodes.
    let g_of_theta = |theta: f64, d: &[[f64; 3]]| -> f64 {
        let mut moved = mesh.clone();
        for (node, dn) in moved.nodes.iter_mut().zip(d) {
            node[0] += theta * dn[0];
            node[1] += theta * dn[1];
            node[2] += theta * dn[2];
        }
        let (w, l) = port_dims(&moved, &port_faces, e_hat);
        let port_m = LumpedPort {
            faces: &port_faces,
            e_hat,
            resistance: r,
            width: w,
            length: l,
            v_inc: c64::new(1.0, 0.5),
        };
        let sol = driven_solve_with_ports::<B>(
            &moved,
            DrivenMaterials::Scalar(&eps_c),
            None,
            &bcs,
            std::slice::from_ref(&port_m),
            omega,
            &source,
            &device(),
        )
        .expect("moving port forward");
        let v = port_voltage(&moved, &port_m, &edges, &sol.e_edges);
        s11_sq_and_dg_dv(v, port_m.v_inc, r, z0).0
    };

    let g0 = g_of_theta(0.0, &vec![[0.0; 3]; mesh.n_nodes()]);
    assert!(
        (g0 - sg.objective).abs() <= 1e-9 * g0.abs().max(1.0),
        "objective mismatch: adjoint {} vs forward {g0}",
        sg.objective
    );

    // Two moving-feed node-motion maps that touch the port face IN-PLANE (an
    // out-of-plane move is even in the displacement, so its ∂S_p = ∂f = 0 at
    // the flat face — only in-plane feed motion exercises the A2 terms):
    //   1. slide a single interior port node in +x (l unchanged; exercises
    //      ∂area→∂w→∂Z_s, ∂S_p, ∂f, objective ∂f — the dl = 0 path);
    //   2. lift every y = 1 port node in +y (coherent extremum motion, so
    //      ∂l ≠ 0 through the symmetric subgradient — the ∂Z_s/∂l path).
    let tol = 1e-9;
    let interior_port = mesh
        .nodes
        .iter()
        .enumerate()
        .find(|(_, p)| {
            p[2].abs() < tol && p[0] > tol && p[0] < 1.0 - tol && p[1] > tol && p[1] < 1.0 - tol
        })
        .map(|(i, _)| i)
        .expect("mesh has an interior z=0 port node");
    let mut d_shear = vec![[0.0_f64; 3]; mesh.n_nodes()];
    d_shear[interior_port] = [1.0, 0.0, 0.0];

    let d_lift: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|p| {
            if p[2].abs() < tol && (p[1] - 1.0).abs() < tol {
                [0.0, 1.0, 0.0]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect();

    let h = 1e-6;
    let mut grads = Vec::new();
    for (name, d) in [("port-slide-x", &d_shear), ("port-lift-y", &d_lift)] {
        let ana = chain_node_motion(&sg.grad_node, d);
        let fd = (g_of_theta(h, d) - g_of_theta(-h, d)) / (2.0 * h);
        assert!(
            fd.abs() > 1e-6,
            "map {name}: FD gradient {fd} unexpectedly ~0 (map does not touch the feed?)"
        );
        let rel = (ana - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
        println!("moving-feed {name}: adjoint {ana:.8}, central-FD {fd:.8}, rel-err {rel:.3e}");
        assert!(
            rel < 5e-3,
            "map {name}: moving-feed adjoint {ana} vs central-FD {fd}, rel {rel:.3e} exceeds 5e-3"
        );
        grads.push(ana);
    }
    assert!(
        (grads[0] - grads[1]).abs() > 1e-6,
        "the two moving-feed maps must yield distinct gradients"
    );
}

/// **Tripwire that bites.** On a port-moving map the pinned-feed A1 gradient
/// (which DROPS `∂S_p/∂X`, `∂b_port/∂X`, `∂Z_s/∂X` and the objective `∂f/∂X`)
/// must FAIL the same central-FD gate the moving-feed A2 gradient passes —
/// proving the four new terms are load-bearing, not cosmetic. On the
/// genuinely complex fixture this also rejects a conjugation error in the new
/// covector paths (which would perturb the A2 match away from FD).
#[test]
fn pinned_a1_gradient_fails_moving_feed_fd() {
    let (mesh, eps_c, mask, source) = port_line_fixture(4);
    let port_faces = plane_faces(&mesh, 2, 0.0);
    let e_hat = [0.0, 1.0, 0.0];
    let omega = 1.3;
    let r = 1.0;
    let z0 = r;
    let bcs = DrivenBcs {
        pec_interior_mask: &mask,
    };
    let (w0, l0) = port_dims(&mesh, &port_faces, e_hat);
    let v_inc = c64::new(1.0, 0.5);
    let port = LumpedPort {
        faces: &port_faces,
        e_hat,
        resistance: r,
        width: w0,
        length: l0,
        v_inc,
    };
    let edges = mesh.edges();

    // Moving-feed A2 gradient (correct).
    let a2 = driven_shape_gradient_moving_port_s11::<B>(
        &mesh,
        &eps_c,
        &bcs,
        omega,
        &source,
        &port,
        z0,
        &device(),
    )
    .expect("A2 gradient");

    // Pinned-feed A1 gradient with the SAME |S₁₁|² objective closure — it
    // omits every port/objective geometry term (∂S_p = ∂f = ∂Z_s = 0).
    let flux = assemble_port_flux(&mesh, &port_faces, e_hat, &edges);
    let obj = s11_sq_objective(flux, 1.0 / w0, v_inc, r, z0);
    let a1 = driven_shape_gradient_ports_complex::<B, _>(
        &mesh,
        &eps_c,
        &bcs,
        omega,
        &source,
        std::slice::from_ref(&port),
        obj,
        &device(),
    )
    .expect("A1 gradient");

    // A port-moving map that touches the feed IN-PLANE: slide a single
    // interior port node in +x (an out-of-plane move would leave ∂S_p =
    // ∂f = 0 at the flat face and A1 would spuriously pass).
    let tol = 1e-9;
    let interior_port = mesh
        .nodes
        .iter()
        .enumerate()
        .find(|(_, p)| {
            p[2].abs() < tol && p[0] > tol && p[0] < 1.0 - tol && p[1] > tol && p[1] < 1.0 - tol
        })
        .map(|(i, _)| i)
        .expect("interior port node");
    let mut d = vec![[0.0_f64; 3]; mesh.n_nodes()];
    d[interior_port] = [1.0, 0.0, 0.0];

    let g_of_theta = |theta: f64| -> f64 {
        let mut moved = mesh.clone();
        moved.nodes[interior_port][0] += theta;
        let (w, l) = port_dims(&moved, &port_faces, e_hat);
        let port_m = LumpedPort {
            faces: &port_faces,
            e_hat,
            resistance: r,
            width: w,
            length: l,
            v_inc,
        };
        let sol = driven_solve_with_ports::<B>(
            &moved,
            DrivenMaterials::Scalar(&eps_c),
            None,
            &bcs,
            std::slice::from_ref(&port_m),
            omega,
            &source,
            &device(),
        )
        .expect("forward");
        let v = port_voltage(&moved, &port_m, &edges, &sol.e_edges);
        s11_sq_and_dg_dv(v, v_inc, r, z0).0
    };
    let h = 1e-6;
    let fd = (g_of_theta(h) - g_of_theta(-h)) / (2.0 * h);
    let rel_a2 =
        (chain_node_motion(&a2.grad_node, &d) - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
    let rel_a1 =
        (chain_node_motion(&a1.grad_node, &d) - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
    println!("moving-feed FD {fd:.8}: A2 rel {rel_a2:.3e}, A1(pinned) rel {rel_a1:.3e}");
    assert!(
        rel_a2 < 5e-3,
        "A2 must pass the moving-feed FD (rel {rel_a2:.3e})"
    );
    assert!(
        rel_a1 > 1e-2,
        "A1(pinned) must FAIL the moving-feed FD (rel {rel_a1:.3e}) — the A2 terms are not biting"
    );
}

// ─────────────────────────────────────────────────────────────────────
// SECOND-ORDER (p=2) tests (issue #619).
// ─────────────────────────────────────────────────────────────────────

use crate::assembly::nedelec_p2::{P2DofMap, assemble_p2_rhs_constant, cube_pec_interior_p2_dofs};
use crate::driven::solve::driven_solve_p2;
use crate::elements::nedelec_p2::{tet_nedelec2_local, tet_nedelec2_local_rhs};

/// A generic well-shaped (non-axis-aligned) tet for the element self-checks.
fn generic_tet() -> [[f64; 3]; 4] {
    [
        [0.10, 0.20, 0.05],
        [1.05, 0.15, 0.20],
        [0.25, 0.95, 0.10],
        [0.20, 0.30, 1.10],
    ]
}

/// Driven PEC cube cavity at `p=2`: uniform lossless ε_r, driven by a
/// genuinely COMPLEX per-tet current source. Returns
/// `(mesh, eps_r, interior_dof_mask, source)`.
fn cavity_fixture_p2(n: usize) -> (TetMesh, Vec<f64>, Vec<bool>, CurrentSource) {
    let mesh = cube_tet_mesh(n, 1.0);
    let eps_r = vec![2.0_f64; mesh.n_tets()];
    let dofs = P2DofMap::build(&mesh);
    let mask = cube_pec_interior_p2_dofs(&mesh, &dofs, 1.0);
    let pi = std::f64::consts::PI;
    let source = CurrentSource::from_centroids(&mesh, |c| {
        [
            c64::new(0.0, 0.3 * (pi * c[2]).sin()),
            c64::new(0.5 * (pi * c[1]).sin(), 0.2),
            c64::new((pi * c[0]).sin(), 0.4 * c[2]),
        ]
    });
    (mesh, eps_r, mask, source)
}

/// **The p=2 dual `.re` faithfully lifts the f64 element kernel.** The `.re`
/// fields of the Dual local K/M must reproduce
/// [`tet_nedelec2_local`] entry-for-entry, and the Dual RHS-moment tensor
/// contracted with a fixed `J` must reproduce
/// [`tet_nedelec2_local_rhs`] — proving the Dual twin differentiates *the
/// same* closed form the `p=2` solver assembles (mirrors the p=1
/// `dual_local_matrices_reproduce_burn_kernel`).
#[test]
fn p2_dual_local_matrices_reproduce_f64_kernel() {
    let base = generic_tet();
    let dc = base.map(|v| v.map(Dual::cst));
    let (dk, dm, dnint) = nedelec2_local_dual(&dc);
    let (fk, fm, _vol) = tet_nedelec2_local(&base);
    let j = [0.7_f64, -0.3, 1.1];
    let frhs = tet_nedelec2_local_rhs(&base, j);

    let mut worst = 0.0_f64;
    for i in 0..TET_NEDELEC2_DOFS {
        for jj in 0..TET_NEDELEC2_DOFS {
            let rk = (dk[i][jj].re - fk[i][jj]).abs() / fk[i][jj].abs().max(1e-12);
            let rm = (dm[i][jj].re - fm[i][jj]).abs() / fm[i][jj].abs().max(1e-12);
            worst = worst.max(rk).max(rm);
        }
        // ∫ N_i · J = Σ_c (∫ N_i,c) · J_c.
        let rhs_i = (0..3).map(|c| dnint[i][c].re * j[c]).sum::<f64>();
        let rr = (rhs_i - frhs[i]).abs() / frhs[i].abs().max(1e-12);
        worst = worst.max(rr);
    }
    assert!(
        worst < 1e-10,
        "p2 dual .re vs f64 kernel worst rel-err {worst:.3e}"
    );
}

/// **The p=2 element-kernel derivative is exact.** The Dual tangents of the
/// local K, M and RHS moments must match a central finite difference of the
/// same `f64` kernel for every one of the twelve node coordinates
/// (scale-normalized, mirroring the p=1
/// `dual_local_derivative_matches_finite_difference`) — proving `∂A/∂X` and
/// `∂b/∂X` at p=2 are analytic forward-mode AD, not FD approximations.
#[test]
fn p2_dual_local_derivative_matches_finite_difference() {
    let base = generic_tet();
    let j = [0.7_f64, -0.3, 1.1];
    let h = 1e-6;

    let mut diff_k = 0.0_f64;
    let mut scale_k = 0.0_f64;
    let mut diff_m = 0.0_f64;
    let mut scale_m = 0.0_f64;
    let mut diff_n = 0.0_f64;
    let mut scale_n = 0.0_f64;

    for a in 0..4 {
        for c in 0..3 {
            let mut dc = base.map(|v| v.map(Dual::cst));
            dc[a][c] = Dual::var(base[a][c]);
            let (dk, dm, dnint) = nedelec2_local_dual(&dc);

            let mut cp = base;
            let mut cm = base;
            cp[a][c] += h;
            cm[a][c] -= h;
            let (kp, mp, _) = tet_nedelec2_local(&cp);
            let (km, mm, _) = tet_nedelec2_local(&cm);
            let rp = tet_nedelec2_local_rhs(&cp, j);
            let rm = tet_nedelec2_local_rhs(&cm, j);

            for i in 0..TET_NEDELEC2_DOFS {
                for jj in 0..TET_NEDELEC2_DOFS {
                    let fdk = (kp[i][jj] - km[i][jj]) / (2.0 * h);
                    let fdm = (mp[i][jj] - mm[i][jj]) / (2.0 * h);
                    diff_k = diff_k.max((dk[i][jj].du - fdk).abs());
                    scale_k = scale_k.max(fdk.abs());
                    diff_m = diff_m.max((dm[i][jj].du - fdm).abs());
                    scale_m = scale_m.max(fdm.abs());
                }
                // RHS moment derivative: Σ_c (∂∫N_i,c) J_c vs FD of ∫N_i·J.
                let dual_rhs = (0..3).map(|c2| dnint[i][c2].du * j[c2]).sum::<f64>();
                let fdn = (rp[i] - rm[i]) / (2.0 * h);
                diff_n = diff_n.max((dual_rhs - fdn).abs());
                scale_n = scale_n.max(fdn.abs());
            }
        }
    }

    assert!(
        scale_k > 1e-3 && scale_m > 1e-3 && scale_n > 1e-3,
        "p2 kernel derivative scales too small (K {scale_k:.3e}, M {scale_m:.3e}, N {scale_n:.3e})"
    );
    let rel_k = diff_k / scale_k;
    let rel_m = diff_m / scale_m;
    let rel_n = diff_n / scale_n;
    let worst = rel_k.max(rel_m).max(rel_n);
    assert!(
        worst < 1e-6,
        "p2 dual-vs-FD scale-normalized rel-err too large \
         (K {rel_k:.3e}, M {rel_m:.3e}, N {rel_n:.3e})"
    );
}

/// Port-less, wall-less, scalar-ε `p=2` shape gradient through the
/// order-generic [`driven_shape_gradient_on_space`] (the entry point that
/// replaced the cube-only `driven_shape_gradient_p2*` wrappers).
fn p2_scalar_shape_gradient<G>(
    mesh: &TetMesh,
    eps: &[c64],
    mask: &[bool],
    omega: f64,
    source: &CurrentSource,
    objective: G,
) -> Result<DrivenShapeGradient, DrivenError>
where
    G: Fn(&[c64]) -> (f64, Vec<c64>),
{
    let space = HcurlSpace::build(mesh, ElementOrder::P2);
    let bcs = DrivenBcs {
        pec_interior_mask: mask,
    };
    let problem = DrivenGradientProblem {
        space: &space,
        mesh,
        materials: DrivenMaterials::Scalar(eps),
        bcs: &bcs,
        source,
        ports: &[],
        surfaces: &[],
        wave_ports: &[],
        omega,
        excited: None,
    };
    driven_shape_gradient_on_space::<TestBackend, _>(&problem, objective, &Default::default())
}

/// **The load-bearing p=2 test.** The `p=2` driven-Nédélec discrete-adjoint
/// **shape** gradient must match a full central finite difference of the
/// entire `p=2` driven pipeline (move nodes → reassemble the `p=2` `A` and
/// current RHS `b` → resolve via the public [`driven_solve_p2`] → recompute
/// g), for two distinct node-motion maps. A wrong sign, a wrong `∂A/∂X` /
/// `∂b/∂X`, a dropped RHS term, the wrong sorted-vertex node mapping, or a
/// conjugation error fails it.
#[test]
fn driven_shape_gradient_p2_matches_central_finite_difference() {
    let (mesh, eps_r, mask, source) = cavity_fixture_p2(2);
    let omega = 1.5;
    let dofs = P2DofMap::build(&mesh);

    // ONE forward + ONE adjoint solve → full nodal-coordinate gradient.
    let eps_c0: Vec<c64> = eps_r.iter().map(|&e| c64::new(e, 0.0)).collect();
    let sg = p2_scalar_shape_gradient(&mesh, &eps_c0, &mask, omega, &source, l2_objective)
        .expect("p2 shape gradient");
    assert_eq!(
        sg.n_factorizations, 1,
        "p2 shape adjoint must reuse the forward factorization"
    );
    assert!(
        sg.residual_rel < 1e-9,
        "p2 forward solve unhealthy (residual {:.3e}); pick ω off resonance",
        sg.residual_rel
    );

    // Full-pipeline objective as a function of θ under a node-velocity field
    // D: move nodes to X⁰ + θD, rebuild the p=2 RHS (same drive factor and
    // fixed per-tet source), re-solve via the public path, recompute g. The
    // PEC mask is topology-derived, so it is held fixed under node motion.
    let g_of_theta = |theta: f64, d: &[[f64; 3]]| -> f64 {
        let mut moved = mesh.clone();
        for (node, dn) in moved.nodes.iter_mut().zip(d) {
            node[0] += theta * dn[0];
            node[1] += theta * dn[1];
            node[2] += theta * dn[2];
        }
        let j_re: Vec<[f64; 3]> = source
            .j_tet
            .iter()
            .map(|jj| [jj[0].re, jj[1].re, jj[2].re])
            .collect();
        let j_im: Vec<[f64; 3]> = source
            .j_tet
            .iter()
            .map(|jj| [jj[0].im, jj[1].im, jj[2].im])
            .collect();
        let b_re = assemble_p2_rhs_constant(&moved, &dofs, &j_re);
        let b_im = assemble_p2_rhs_constant(&moved, &dofs, &j_im);
        let rhs: Vec<c64> = b_re
            .iter()
            .zip(b_im.iter())
            .map(|(&re, &im)| c64::new(-omega * im, omega * re))
            .collect();
        let sol = driven_solve_p2(&moved, &eps_r, &mask, omega, &rhs).expect("driven_solve_p2");
        l2_objective(&sol.x).0
    };

    // Objective must agree between the two forward paths (sanity).
    let g0_pub = g_of_theta(0.0, &vec![[0.0; 3]; mesh.n_nodes()]);
    assert!(
        (g0_pub - sg.objective).abs() <= 1e-9 * g0_pub.abs().max(1.0),
        "objective mismatch: shape adjoint {} vs public driven_solve_p2 {g0_pub}",
        sg.objective
    );

    // Two analytic node-motion maps, LINEAR in θ.
    let tol = 1e-9;
    let d_face: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|p| {
            if (p[0] - 1.0).abs() < tol {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect();
    let ctr = [0.5, 0.5, 0.5];
    let ctrl = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, p)| p.iter().all(|&c| c > tol && c < 1.0 - tol))
        .min_by(|(_, a), (_, b)| {
            let da = (a[0] - ctr[0]).powi(2) + (a[1] - ctr[1]).powi(2) + (a[2] - ctr[2]).powi(2);
            let db = (b[0] - ctr[0]).powi(2) + (b[1] - ctr[1]).powi(2) + (b[2] - ctr[2]).powi(2);
            da.partial_cmp(&db).unwrap()
        })
        .map(|(i, _)| i)
        .expect("mesh has an interior node");
    let mut d_node = vec![[0.0_f64; 3]; mesh.n_nodes()];
    d_node[ctrl] = [1.0, 0.0, 0.0];

    let h = 1e-6;
    for (name, d) in [
        ("hi-face-translate", &d_face),
        ("interior-control-node", &d_node),
    ] {
        let ana = chain_node_motion(&sg.grad_node, d);
        let fd = (g_of_theta(h, d) - g_of_theta(-h, d)) / (2.0 * h);
        assert!(
            fd.abs() > 1e-6,
            "map {name}: FD gradient {fd} unexpectedly ~0 (fixture degenerate?)"
        );
        let rel = (ana - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
        assert!(
            rel < 1e-3,
            "map {name}: p2 adjoint {ana} vs central-FD {fd}, rel-err {rel:.3e} exceeds 1e-3"
        );
    }

    // The two maps must give DISTINCT gradients.
    let g_face = chain_node_motion(&sg.grad_node, &d_face);
    let g_node = chain_node_motion(&sg.grad_node, &d_node);
    assert!(
        (g_face - g_node).abs() > 1e-6,
        "the two node-motion maps must yield distinct gradients ({g_face} vs {g_node})"
    );
}

/// Mutation tripwire (p=2): the finite-difference check must **reject** a
/// shape gradient built with the classic complex-adjoint conjugation error
/// (feeding `∂g/∂x̄` instead of `∂g/∂x`) on the genuinely complex cavity
/// field — proving the load-bearing p=2 test's tolerance is biting.
#[test]
fn p2_conjugation_error_is_detected_by_fd() {
    let (mesh, eps_r, mask, source) = cavity_fixture_p2(2);
    let omega = 1.5;
    let dofs = P2DofMap::build(&mesh);

    // Wrong cotangent: +i·Im instead of −i·Im (∂g/∂x̄ rather than ∂g/∂x).
    let wrong_objective = |x: &[c64]| -> (f64, Vec<c64>) {
        let g = x.iter().map(|z| z.re * z.re + z.im * z.im).sum::<f64>();
        let cot = x.iter().map(|z| c64::new(z.re, z.im)).collect();
        (g, cot)
    };
    let eps_c0: Vec<c64> = eps_r.iter().map(|&e| c64::new(e, 0.0)).collect();
    let wrong = p2_scalar_shape_gradient(&mesh, &eps_c0, &mask, omega, &source, wrong_objective)
        .expect("wrong-conjugation p2 shape gradient");

    let g_of_theta = |theta: f64, d: &[[f64; 3]]| -> f64 {
        let mut moved = mesh.clone();
        for (node, dn) in moved.nodes.iter_mut().zip(d) {
            node[0] += theta * dn[0];
            node[1] += theta * dn[1];
            node[2] += theta * dn[2];
        }
        let j_re: Vec<[f64; 3]> = source
            .j_tet
            .iter()
            .map(|jj| [jj[0].re, jj[1].re, jj[2].re])
            .collect();
        let j_im: Vec<[f64; 3]> = source
            .j_tet
            .iter()
            .map(|jj| [jj[0].im, jj[1].im, jj[2].im])
            .collect();
        let b_re = assemble_p2_rhs_constant(&moved, &dofs, &j_re);
        let b_im = assemble_p2_rhs_constant(&moved, &dofs, &j_im);
        let rhs: Vec<c64> = b_re
            .iter()
            .zip(b_im.iter())
            .map(|(&re, &im)| c64::new(-omega * im, omega * re))
            .collect();
        let sol = driven_solve_p2(&moved, &eps_r, &mask, omega, &rhs).expect("driven_solve_p2");
        l2_objective(&sol.x).0
    };

    let tol = 1e-9;
    let d_face: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|p| {
            if (p[0] - 1.0).abs() < tol {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect();
    let h = 1e-6;
    let fd = (g_of_theta(h, &d_face) - g_of_theta(-h, &d_face)) / (2.0 * h);
    let ana_wrong = chain_node_motion(&wrong.grad_node, &d_face);
    let rel = (ana_wrong - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel > 1e-2,
        "p2 conjugation-error gradient {ana_wrong} matched the FD {fd} (rel {rel:.3e}) — \
         the tolerance is not biting"
    );
}

/// **The load-bearing complex-ε p=2 test.** The lossy second-order
/// driven-Nédélec shape gradient `∂g/∂θ` (complex `ε = ε′ − i·ε″`,
/// `Im(ε) ≠ 0`) must match a full central finite difference of the entire
/// complex p=2 pipeline. The public `driven_solve_p2` is real-ε only, so the
/// FD reference reuses [`p2_scalar_shape_gradient`]'s **own forward
/// objective** at perturbed nodes (it re-assembles the complex `A` + RHS and
/// re-solves for each θ) — the gradient under test comes from the adjoint
/// algebra, the FD reference only reads the forward `g`. A dropped loss term
/// in the p=2 pencil or contraction fails it.
#[test]
fn driven_shape_gradient_p2_complex_matches_central_finite_difference() {
    let (mesh, eps_r, mask, source) = cavity_fixture_p2(2);
    let omega = 1.5;
    // Promote to a lossy substrate: ε = 2.0 − 0.3i (tan δ = 0.15).
    let eps_c: Vec<c64> = eps_r.iter().map(|&e| c64::new(e, -0.3)).collect();

    // ONE forward + ONE adjoint solve → full nodal-coordinate gradient.
    let sg = p2_scalar_shape_gradient(&mesh, &eps_c, &mask, omega, &source, l2_objective)
        .expect("complex p2 shape gradient");
    assert_eq!(
        sg.n_factorizations, 1,
        "complex p2 shape adjoint must reuse the forward factorization"
    );
    assert!(
        sg.residual_rel < 1e-9,
        "complex p2 forward solve unhealthy (residual {:.3e})",
        sg.residual_rel
    );

    // Full-pipeline objective: re-run the complex forward at moved nodes and
    // read its forward `g` (`.objective`), an independent-of-the-adjoint FD.
    let g_of_theta = |theta: f64, d: &[[f64; 3]]| -> f64 {
        let mut moved = mesh.clone();
        for (node, dn) in moved.nodes.iter_mut().zip(d) {
            node[0] += theta * dn[0];
            node[1] += theta * dn[1];
            node[2] += theta * dn[2];
        }
        p2_scalar_shape_gradient(&moved, &eps_c, &mask, omega, &source, l2_objective)
            .expect("complex p2 forward")
            .objective
    };

    // Sanity: θ=0 forward matches the gradient run's own objective.
    let g0 = g_of_theta(0.0, &vec![[0.0; 3]; mesh.n_nodes()]);
    assert!(
        (g0 - sg.objective).abs() <= 1e-9 * g0.abs().max(1.0),
        "objective mismatch: {} vs {g0}",
        sg.objective
    );

    let tol = 1e-9;
    let d_face: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|p| {
            if (p[0] - 1.0).abs() < tol {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect();
    let ctr = [0.5, 0.5, 0.5];
    let ctrl = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, p)| p.iter().all(|&c| c > tol && c < 1.0 - tol))
        .min_by(|(_, a), (_, b)| {
            let da = (a[0] - ctr[0]).powi(2) + (a[1] - ctr[1]).powi(2) + (a[2] - ctr[2]).powi(2);
            let db = (b[0] - ctr[0]).powi(2) + (b[1] - ctr[1]).powi(2) + (b[2] - ctr[2]).powi(2);
            da.partial_cmp(&db).unwrap()
        })
        .map(|(i, _)| i)
        .expect("mesh has an interior node");
    let mut d_node = vec![[0.0_f64; 3]; mesh.n_nodes()];
    d_node[ctrl] = [1.0, 0.0, 0.0];

    let h = 1e-6;
    for (name, d) in [
        ("hi-face-translate", &d_face),
        ("interior-control-node", &d_node),
    ] {
        let ana = chain_node_motion(&sg.grad_node, d);
        let fd = (g_of_theta(h, d) - g_of_theta(-h, d)) / (2.0 * h);
        assert!(
            fd.abs() > 1e-6,
            "map {name}: FD gradient {fd} unexpectedly ~0 (fixture degenerate?)"
        );
        let rel = (ana - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
        assert!(
            rel < 1e-3,
            "map {name}: complex p2 adjoint {ana} vs central-FD {fd}, rel-err {rel:.3e} exceeds 1e-3"
        );
    }

    let g_face = chain_node_motion(&sg.grad_node, &d_face);
    let g_node = chain_node_motion(&sg.grad_node, &d_node);
    assert!(
        (g_face - g_node).abs() > 1e-6,
        "the two node-motion maps must yield distinct gradients ({g_face} vs {g_node})"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Box-UPML tensor-material shape adjoint (issue #635, Epic #628 Phase C1).
// ─────────────────────────────────────────────────────────────────────

/// The real `f64` **weighted** Nédélec local matrices from the production
/// Burn kernels for one tet and a fixed 3×3 weight pair, used to pin the
/// tensor dual `.re` fields.
fn burn_weighted_local(
    coords: &[[f64; 3]; 4],
    w_k: &[[f64; 3]; 3],
    w_m: &[[f64; 3]; 3],
) -> ([[f64; 6]; 6], [[f64; 6]; 6]) {
    use crate::elements::nedelec::{
        batched_nedelec_local_mass_anisotropic_full, batched_nedelec_local_stiffness_weighted,
    };
    let cflat: Vec<f64> = coords.iter().flat_map(|v| v.iter().copied()).collect();
    let ct = burn::tensor::Tensor::<B, 1>::from_data(TensorData::new(cflat, [12]), &device())
        .reshape([1, 4, 3]);
    let wk_flat: Vec<f64> = w_k.iter().flat_map(|r| r.iter().copied()).collect();
    let wm_flat: Vec<f64> = w_m.iter().flat_map(|r| r.iter().copied()).collect();
    let wkt = burn::tensor::Tensor::<B, 1>::from_data(TensorData::new(wk_flat, [9]), &device())
        .reshape([1, 3, 3]);
    let wmt = burn::tensor::Tensor::<B, 1>::from_data(TensorData::new(wm_flat, [9]), &device())
        .reshape([1, 3, 3]);
    let k: Vec<f64> = batched_nedelec_local_stiffness_weighted(ct.clone(), wkt)
        .into_data()
        .iter::<f64>()
        .collect();
    let m: Vec<f64> = batched_nedelec_local_mass_anisotropic_full(ct, wmt)
        .into_data()
        .iter::<f64>()
        .collect();
    let mut kk = [[0.0_f64; 6]; 6];
    let mut mm = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            kk[i][j] = k[i * 6 + j];
            mm[i][j] = m[i * 6 + j];
        }
    }
    (kk, mm)
}

/// Tet vertices plus the curl-weight (ν) and mass-weight (ε) 3×3 tensors that
/// drive the tensor-material twin fixtures.
type TensorTwinInputs = ([[f64; 3]; 4], [[f64; 3]; 3], [[f64; 3]; 3]);

/// A generic (non-axis-aligned) well-shaped tet and a genuinely **full**
/// (off-diagonal, asymmetric) 3×3 weight pair — a strictly harder input than
/// the diagonal box `Λ`, so passing here subsumes the box case.
fn tensor_twin_inputs() -> TensorTwinInputs {
    let base = [
        [0.10, 0.20, 0.05],
        [1.05, 0.15, 0.20],
        [0.25, 0.95, 0.10],
        [0.20, 0.30, 1.10],
    ];
    // ν = Λ⁻¹-like curl weight, ε = ε_r·Λ-like mass weight (both full here).
    let w_k = [[1.30, 0.10, -0.05], [0.07, 0.90, 0.12], [-0.03, 0.08, 1.10]];
    let w_m = [[2.10, 0.15, 0.05], [0.12, 1.80, -0.09], [0.04, 0.11, 2.40]];
    (base, w_k, w_m)
}

/// **The tensor dual `.re` faithfully lifts the production weighted kernels.**
/// Reproduces the forward `assemble_global_nedelec_with_full_tensors` element
/// contribution (`batched_nedelec_local_stiffness_weighted` /
/// `batched_nedelec_local_mass_anisotropic_full`) at **zero perturbation**.
#[test]
fn tensor_dual_local_matrices_reproduce_forward_kernel() {
    let (base, w_k, w_m) = tensor_twin_inputs();
    let dc = base.map(|v| v.map(Dual::cst));
    let (dk, dm) = nedelec_local_dual_tensor(&dc, &w_k, &w_m);
    let (bk, bm) = burn_weighted_local(&base, &w_k, &w_m);
    let mut worst = 0.0_f64;
    for i in 0..6 {
        for j in 0..6 {
            let rk = (dk[i][j].re - bk[i][j]).abs() / bk[i][j].abs().max(1e-12);
            let rm = (dm[i][j].re - bm[i][j]).abs() / bm[i][j].abs().max(1e-12);
            worst = worst.max(rk).max(rm);
        }
    }
    assert!(
        worst < 1e-10,
        "tensor dual .re vs forward weighted kernel worst rel-err {worst:.3e}"
    );
}

/// **The tensor element-kernel derivative is exact.** The tensor dual
/// tangents of `∂K(ν)/∂X`, `∂M(ε)/∂X` must match a central finite difference
/// of the SAME `f64` twin for every one of the twelve node coordinates — the
/// fixed-Λ geometry Jacobian is analytic forward-mode AD, not FD.
#[test]
fn tensor_dual_local_derivative_matches_finite_difference() {
    let (base, w_k, w_m) = tensor_twin_inputs();
    let eval = |coords: &[[f64; 3]; 4]| {
        let dc = coords.map(|v| v.map(Dual::cst));
        let (k, m) = nedelec_local_dual_tensor(&dc, &w_k, &w_m);
        let kre =
            std::array::from_fn::<_, 6, _>(|i| std::array::from_fn::<_, 6, _>(|j| k[i][j].re));
        let mre =
            std::array::from_fn::<_, 6, _>(|i| std::array::from_fn::<_, 6, _>(|j| m[i][j].re));
        (kre, mre)
    };
    let h = 1e-6;
    let (mut diff_k, mut scale_k, mut diff_m, mut scale_m) = (0.0, 0.0, 0.0, 0.0);
    for a in 0..4 {
        for c in 0..3 {
            let mut dc = base.map(|v| v.map(Dual::cst));
            dc[a][c] = Dual::var(base[a][c]);
            let (dk, dm) = nedelec_local_dual_tensor(&dc, &w_k, &w_m);

            let mut cp = base;
            let mut cm = base;
            cp[a][c] += h;
            cm[a][c] -= h;
            let (kp, mp) = eval(&cp);
            let (km, mm) = eval(&cm);
            for i in 0..6 {
                for j in 0..6 {
                    let fdk = (kp[i][j] - km[i][j]) / (2.0 * h);
                    let fdm = (mp[i][j] - mm[i][j]) / (2.0 * h);
                    diff_k = f64::max(diff_k, (dk[i][j].du - fdk).abs());
                    scale_k = f64::max(scale_k, fdk.abs());
                    diff_m = f64::max(diff_m, (dm[i][j].du - fdm).abs());
                    scale_m = f64::max(scale_m, fdm.abs());
                }
            }
        }
    }
    assert!(
        scale_k > 1e-3 && scale_m > 1e-3,
        "kernel derivative scales too small (K {scale_k:.3e}, M {scale_m:.3e})"
    );
    let (rel_k, rel_m) = (diff_k / scale_k, diff_m / scale_m);
    assert!(
        rel_k.max(rel_m) < 1e-6,
        "tensor dual-vs-FD scale-normalized rel-err too large (K {rel_k:.3e}, M {rel_m:.3e})"
    );
}

/// **PML-pinned tripwire bites.** [`chain_node_motion_pml_pinned`] must panic
/// when a design node that belongs to a PML tet carries nonzero motion, since
/// `∂Λ/∂X` is unmodeled in Phase C1 (the moving-PML profile derivative is C2).
#[test]
#[should_panic(expected = "PML-pinned node")]
fn pml_pinned_tripwire_rejects_moving_pml_node() {
    let mesh = cube_tet_mesh(2, 1.0);
    let mut pml = vec![false; mesh.n_tets()];
    pml[0] = true; // tag the first tet as PML → its 4 nodes are pinned.
    let pinned = pml_shell_nodes(&mesh, &pml);
    let node = pinned.iter().position(|&p| p).expect("a pinned node");
    let grad = vec![[0.0_f64; 3]; mesh.n_nodes()];
    let mut d = vec![[0.0_f64; 3]; mesh.n_nodes()];
    d[node] = [1.0, 0.0, 0.0]; // move a pinned node → must panic.
    let _ = chain_node_motion_pml_pinned(&grad, &d, &pinned);
}

/// **The load-bearing box-UPML gate.** The tensor-material box-UPML shape
/// adjoint `∂g/∂θ` on the **real patch UPML shell**
/// ([`crate::mesh::patch::PatchFixture::matched_upml_materials`]) must match a
/// full central finite difference of the entire UPML driven pipeline
/// (perturb θ → move the non-PML nodes → re-assemble + re-solve the
/// matched-UPML forward via the public [`driven_solve`] → recompute g) to
/// `rel_err ≤ 5e-3`. The PML shell is held fixed (∂Λ/∂X = 0) and the FD
/// reference holds the per-tet Λ tensors fixed, matching the C1 convention.
/// A conjugation-error tripwire proves the tolerance bites.
#[test]
fn driven_shape_gradient_matched_upml_matches_central_finite_difference() {
    use crate::mesh::patch::{FR4_MATERIALS, PHYS_UPML, read_patch_smoke_fixture};
    use crate::mesh::pec_interior_mask_from_triangles;

    let fixture = read_patch_smoke_fixture().expect("patch smoke fixture");
    let mesh = fixture.mesh.clone();
    let patch_tris = fixture.patch_triangles();
    let ground_tris = fixture.ground_triangles();
    let outer_tris = fixture.outer_boundary_triangles();
    let edges = mesh.edges();
    let interior = pec_interior_mask_from_triangles(
        &edges,
        &[
            patch_tris.as_slice(),
            ground_tris.as_slice(),
            outer_tris.as_slice(),
        ],
    );
    let bcs = DrivenBcs {
        pec_interior_mask: &interior,
    };

    let pml_thick = 8.0;
    let (air_lo, air_hi) = fixture.air_box(pml_thick);
    let sigma_0 = 1.0;
    let omega = 0.35;
    let (eps_tensor, nu_tensor) =
        fixture.matched_upml_materials(&FR4_MATERIALS, air_lo, air_hi, pml_thick, sigma_0, omega);

    let pml_tet_mask: Vec<bool> = fixture
        .tet_physical_tags
        .iter()
        .map(|&t| t == PHYS_UPML)
        .collect();
    let pinned = pml_shell_nodes(&mesh, &pml_tet_mask);

    // Fully complex volumetric current source across the domain (so the field
    // is genuinely complex and the Wirtinger convention is exercised).
    let source = CurrentSource::from_centroids(&mesh, |c| {
        [
            c64::new(0.0, 0.20 * c[2].cos()),
            c64::new(0.15 * c[0].sin(), 0.10),
            c64::new(0.30 * c[0].cos(), 0.20 * c[1].sin()),
        ]
    });

    let sg = driven_shape_gradient_matched_upml::<B, _>(
        &mesh,
        &eps_tensor,
        &nu_tensor,
        &bcs,
        omega,
        &source,
        l2_objective,
        &device(),
    )
    .expect("box-UPML shape gradient");
    assert_eq!(
        sg.n_factorizations, 1,
        "box-UPML shape adjoint must reuse the forward factorization"
    );
    assert!(
        sg.residual_rel < 1e-8,
        "forward UPML solve unhealthy (residual {:.3e}); pick ω off resonance",
        sg.residual_rel
    );

    // FD reference: hold the per-tet Λ tensors FIXED (the pinned-shell C1
    // convention), move only non-PML nodes, re-assemble + re-solve the UPML
    // forward through the independent public driven_solve path.
    let eps_ref = eps_tensor.clone();
    let nu_ref = nu_tensor.clone();
    let g_of_theta = |theta: f64, d: &[[f64; 3]]| -> f64 {
        let mut moved = mesh.clone();
        for (node, dn) in moved.nodes.iter_mut().zip(d) {
            node[0] += theta * dn[0];
            node[1] += theta * dn[1];
            node[2] += theta * dn[2];
        }
        let sol = driven_solve::<B>(
            &moved,
            DrivenMaterials::MatchedUpml {
                epsilon_tensor: &eps_ref,
                nu_tensor: &nu_ref,
            },
            &bcs,
            omega,
            &source,
            &device(),
        )
        .expect("driven UPML solve");
        l2_objective(&sol.e_edges).0
    };

    let g0 = g_of_theta(0.0, &vec![[0.0; 3]; mesh.n_nodes()]);
    assert!(
        (g0 - sg.objective).abs() <= 1e-8 * g0.abs().max(1.0),
        "objective mismatch: adjoint {} vs public driven_solve {g0}",
        sg.objective
    );

    // Node-motion map: a smooth +z Gaussian bump on the NON-pinned nodes
    // (pinned PML nodes held at zero), linear in θ so X(θ)=X⁰+θD is exact.
    let mut center = [0.0_f64; 3];
    let mut n_free = 0.0_f64;
    for (i, p) in mesh.nodes.iter().enumerate() {
        if !pinned[i] {
            for k in 0..3 {
                center[k] += p[k];
            }
            n_free += 1.0;
        }
    }
    for c in center.iter_mut() {
        *c /= n_free.max(1.0);
    }
    let s2 = 25.0_f64; // bump width² (mesh units)
    let d: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if pinned[i] {
                [0.0; 3]
            } else {
                let r2 = (p[0] - center[0]).powi(2)
                    + (p[1] - center[1]).powi(2)
                    + (p[2] - center[2]).powi(2);
                [0.0, 0.0, (-r2 / (2.0 * s2)).exp()]
            }
        })
        .collect();

    let h = 1e-6;
    let ana = chain_node_motion_pml_pinned(&sg.grad_node, &d, &pinned);
    let fd = (g_of_theta(h, &d) - g_of_theta(-h, &d)) / (2.0 * h);
    assert!(
        fd.abs() > 1e-8,
        "FD gradient {fd} unexpectedly ~0 (fixture/source degenerate?)"
    );
    let rel = (ana - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel < 5e-3,
        "box-UPML adjoint {ana} vs central-FD {fd}, rel-err {rel:.3e} exceeds 5e-3"
    );

    // Conjugation tripwire: the wrong Wirtinger cotangent (+i·Im instead of
    // −i·Im, i.e. ∂g/∂x̄) must be REJECTED by the FD — proving the 5e-3 gate
    // is biting, not vacuously satisfied.
    let wrong_objective = |x: &[c64]| -> (f64, Vec<c64>) {
        let g = x.iter().map(|z| z.re * z.re + z.im * z.im).sum::<f64>();
        let cot = x.iter().map(|z| c64::new(z.re, z.im)).collect();
        (g, cot)
    };
    let wrong = driven_shape_gradient_matched_upml::<B, _>(
        &mesh,
        &eps_tensor,
        &nu_tensor,
        &bcs,
        omega,
        &source,
        wrong_objective,
        &device(),
    )
    .expect("wrong-cotangent box-UPML gradient");
    let ana_wrong = chain_node_motion_pml_pinned(&wrong.grad_node, &d, &pinned);
    let rel_wrong = (ana_wrong - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel_wrong > 1e-2,
        "conjugation error NOT detected by FD: wrong-adjoint {ana_wrong} vs FD {fd}, \
         rel-err {rel_wrong:.3e} (gate not biting)"
    );
}

/// **The load-bearing capstone test (issue #636).** The composed
/// open-radiator shape gradient — matched box-UPML tensor material + lossy
/// FR-4 ε + a pinned-feed lumped port, all in one differentiated pencil via
/// [`driven_shape_gradient_matched_upml_ports`] — must match a full central
/// finite difference of the entire **port-loaded UPML** pipeline (perturb θ →
/// move only the non-PML, non-port nodes → re-assemble + re-solve the forward
/// through the independent public [`driven_solve_with_ports`] with
/// [`DrivenMaterials::MatchedUpml`] → recompute g) to `rel_err ≤ 5e-3`, with
/// `n_factorizations == 1`. The PML shell (`∂Λ/∂X = 0`) and the port feed
/// (`∂S_p/∂X = ∂b_port/∂X = 0`) are both pinned. A conjugation tripwire proves
/// the tolerance bites. This is the composition #626 documented as the
/// open-radiator follow-on.
#[test]
fn driven_shape_gradient_matched_upml_ports_matches_central_finite_difference() {
    use crate::mesh::patch::composed_upml_port_fd_gate;
    use crate::mesh::patch::read_patch_smoke_fixture;

    let fixture = read_patch_smoke_fixture().expect("patch smoke fixture");
    composed_upml_port_fd_gate(&fixture, 8.0, "flat smoke patch");
}

/// Empty `ports` reproduces the port-less box-UPML gradient
/// ([`driven_shape_gradient_matched_upml`]) — the composed capstone entry
/// point is a strict superset (issue #636).
///
/// **Tolerance, not bit identity (issue #1108).** The port-less entry point
/// delegates to the ports entry point with `&[]`, so the two calls below run
/// the same code on the same inputs, and the comparison checks that the
/// delegation and the empty-ports path add nothing. It does not check
/// determinism. Two calls of the same
/// sparse factorization + solve in one process are not bit-reproducible
/// on every target: on x86_64 the CI arpack job saw the objective differ
/// by 11 ulps (≈ 2.4e-15 relative) on PRs that touch no Rust (#1107,
/// #1119), and a rerun passed. The likely cause is floating-point
/// summation order that depends on run-time state (SIMD kernels whose
/// head / tail split follows the buffer alignment, or thread scheduling).
/// So the objective and every gradient component must agree to a
/// relative `1e-12` (the gradient scaled by its largest component, since
/// single components can be near zero). That is about 400× the observed
/// spread and still far below what any real change of the code path
/// (a stray port term, a different material or right-hand side) produces.
#[test]
fn empty_ports_equals_portless_matched_upml_gradient() {
    use crate::mesh::patch::{FR4_MATERIALS, PHYS_UPML, read_patch_smoke_fixture};
    use crate::mesh::pec_interior_mask_from_triangles;

    let fixture = read_patch_smoke_fixture().expect("patch smoke fixture");
    let mesh = fixture.mesh.clone();
    let edges = mesh.edges();
    let interior = pec_interior_mask_from_triangles(
        &edges,
        &[
            fixture.patch_triangles().as_slice(),
            fixture.ground_triangles().as_slice(),
            fixture.outer_boundary_triangles().as_slice(),
        ],
    );
    let bcs = DrivenBcs {
        pec_interior_mask: &interior,
    };
    let pml_thick = 8.0;
    let (air_lo, air_hi) = fixture.air_box(pml_thick);
    let omega = 0.35;
    let (eps_tensor, nu_tensor) =
        fixture.matched_upml_materials(&FR4_MATERIALS, air_lo, air_hi, pml_thick, 1.0, omega);
    let _ = PHYS_UPML;
    let source = CurrentSource::from_centroids(&mesh, |c| {
        [
            c64::new(0.0, 0.20 * c[2].cos()),
            c64::new(0.15 * c[0].sin(), 0.10),
            c64::new(0.30 * c[0].cos(), 0.20 * c[1].sin()),
        ]
    });

    let portless = driven_shape_gradient_matched_upml::<B, _>(
        &mesh,
        &eps_tensor,
        &nu_tensor,
        &bcs,
        omega,
        &source,
        l2_objective,
        &device(),
    )
    .expect("portless box-UPML gradient");
    let empty = driven_shape_gradient_matched_upml_ports::<B, _>(
        &mesh,
        &eps_tensor,
        &nu_tensor,
        &bcs,
        &[],
        omega,
        &source,
        l2_objective,
        &device(),
    )
    .expect("empty-ports composed gradient");

    assert_eq!(portless.n_factorizations, empty.n_factorizations);
    assert_eq!(portless.grad_node.len(), empty.grad_node.len());
    const REL_TOL: f64 = 1e-12;
    let obj_rel = (portless.objective - empty.objective).abs()
        / portless.objective.abs().max(f64::MIN_POSITIVE);
    let scale = portless
        .grad_node
        .iter()
        .flatten()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "zero gradient");
    let grad_rel = portless
        .grad_node
        .iter()
        .zip(empty.grad_node.iter())
        .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
        .fold(0.0_f64, f64::max)
        / scale;
    println!("empty ports vs port-less: objective rel {obj_rel:.3e}, gradient rel {grad_rel:.3e}");
    assert!(
        obj_rel <= REL_TOL,
        "objective differs on empty ports: {} vs {} (rel {obj_rel:.3e})",
        portless.objective,
        empty.objective
    );
    assert!(
        grad_rel <= REL_TOL,
        "grad_node differs on empty ports: max |diff| / max |grad| = {grad_rel:.3e}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// High-DOF freeform boundary parametrization + mesh-morph regularizer
// (Epic #647 Phase 1, issue #648) — validated END-TO-END through the
// composed open-radiator capstone gradient.
// ─────────────────────────────────────────────────────────────────────

/// Assemble the composed-capstone patch fixture (matched box-UPML + lossy
/// FR-4 + a pinned-feed lumped port) shared by the freeform-morph tests. It
/// returns everything needed to (a) compute the capstone nodal gradient and
/// (b) re-solve the port-loaded UPML forward under a node-motion map, plus
/// the pinned-node mask (PML shell ∪ port faces).
#[allow(clippy::type_complexity)]
fn capstone_patch_fixture() -> (
    TetMesh,
    Vec<[[c64; 3]; 3]>,
    Vec<[[c64; 3]; 3]>,
    Vec<bool>,
    crate::mesh::patch::PatchPort,
    CurrentSource,
    Vec<bool>,
    f64,
) {
    use crate::mesh::patch::{FR4_MATERIALS, PHYS_UPML, read_patch_smoke_fixture};
    use crate::mesh::pec_interior_mask_from_triangles;

    let fixture = read_patch_smoke_fixture().expect("patch smoke fixture");
    let mesh = fixture.mesh.clone();
    let patch_tris = fixture.patch_triangles();
    let ground_tris = fixture.ground_triangles();
    let outer_tris = fixture.outer_boundary_triangles();
    let edges = mesh.edges();
    let interior = pec_interior_mask_from_triangles(
        &edges,
        &[
            patch_tris.as_slice(),
            ground_tris.as_slice(),
            outer_tris.as_slice(),
        ],
    );

    let pml_thick = 8.0;
    let (air_lo, air_hi) = fixture.air_box(pml_thick);
    let omega = 0.35;
    let (eps_tensor, nu_tensor) =
        fixture.matched_upml_materials(&FR4_MATERIALS, air_lo, air_hi, pml_thick, 1.0, omega);

    let patch_port = fixture.port();

    let pml_tet_mask: Vec<bool> = fixture
        .tet_physical_tags
        .iter()
        .map(|&t| t == PHYS_UPML)
        .collect();
    let mut pinned = pml_shell_nodes(&mesh, &pml_tet_mask);
    for tri in &patch_port.faces {
        for &nn in tri {
            pinned[nn as usize] = true;
        }
    }

    let source = CurrentSource::from_centroids(&mesh, |c| {
        [
            c64::new(0.0, 0.20 * c[2].cos()),
            c64::new(0.15 * c[0].sin(), 0.10),
            c64::new(0.30 * c[0].cos(), 0.20 * c[1].sin()),
        ]
    });

    (
        mesh, eps_tensor, nu_tensor, interior, patch_port, source, pinned, omega,
    )
}

/// **The multi-DOF freeform-shape gradient is FD-exact on a real mesh.** A
/// high-DOF freeform boundary parametrization ([`FreeformBoundaryMorph`]) of
/// several free patch-metal nodes — each extended into the interior by the
/// Laplace mesh-morph regularizer, with the PML shell and pinned feed held
/// fixed — contracted against the composed-capstone nodal gradient must match
/// a full central finite difference of the entire port-loaded UPML pipeline,
/// per DOF, to `rel_err ≤ 5e-3`, with `n_factorizations == 1`. The columns
/// vanish on every pinned node, so the contraction agrees bit-for-bit with
/// the PML-pinned chain guard.
#[test]
fn driven_freeform_multidof_shape_gradient_matches_central_finite_difference() {
    use crate::constants::ETA_0_OHM as ETA_0;
    use crate::driven::solve::{DrivenMaterials, driven_solve_with_ports};
    use crate::shape::{BoundaryMotionDof, FreeformBoundaryMorph};

    let (mesh, eps_tensor, nu_tensor, interior, patch_port, source, pinned, omega) =
        capstone_patch_fixture();
    let bcs = DrivenBcs {
        pec_interior_mask: &interior,
    };
    let port = patch_port.lumped_port(50.0 / ETA_0, c64::new(1.0, 0.5));

    // Composed-capstone nodal gradient (ONE forward + ONE adjoint solve).
    let sg = driven_shape_gradient_matched_upml_ports::<B, _>(
        &mesh,
        &eps_tensor,
        &nu_tensor,
        &bcs,
        std::slice::from_ref(&port),
        omega,
        &source,
        l2_objective,
        &device(),
    )
    .expect("composed capstone gradient");
    assert_eq!(
        sg.n_factorizations, 1,
        "multi-DOF gradient must inherit the single-factorization capstone adjoint"
    );
    assert!(sg.residual_rel < 1e-8, "forward solve unhealthy");

    // fixed_zero = every pinned node (PML shell ∪ port feed). The morph
    // columns are exactly zero there, so ∂Λ/∂X = 0 and the pinned feed hold.
    let fixed_zero: Vec<u32> = pinned
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| if p { Some(i as u32) } else { None })
        .collect();

    // Freeform boundary DOFs: several FREE patch-metal nodes, each a +z
    // (out-of-plane) normal bump — the curved-metal antenna deformation.
    let mut patch_nodes: Vec<u32> = {
        use crate::mesh::patch::read_patch_smoke_fixture;
        read_patch_smoke_fixture()
            .expect("patch smoke fixture")
            .patch_triangles()
            .iter()
            .flatten()
            .copied()
            .collect()
    };
    patch_nodes.sort_unstable();
    patch_nodes.dedup();
    let free_patch: Vec<u32> = patch_nodes
        .into_iter()
        .filter(|&nn| !pinned[nn as usize])
        .collect();
    assert!(
        free_patch.len() >= 3,
        "need at least 3 free patch nodes for a multi-DOF test, got {}",
        free_patch.len()
    );
    let dofs: Vec<BoundaryMotionDof> = free_patch
        .iter()
        .take(3)
        .map(|&node| BoundaryMotionDof {
            node,
            dir: [0.0, 0.0, 1.0],
        })
        .collect();

    let morph = FreeformBoundaryMorph::harmonic_boundary(&mesh, &dofs, &fixed_zero)
        .expect("harmonic freeform morph");
    assert_eq!(morph.n_dofs(), dofs.len());

    // Columns must vanish on every pinned node (⇒ the PML-pinned chain guard
    // never trips and the pinned-feed premise holds).
    for p in 0..morph.n_dofs() {
        let col = morph.velocity(p);
        for (i, &is_pinned) in pinned.iter().enumerate() {
            if is_pinned {
                assert_eq!(col[i], [0.0; 3], "morph col {p} moves pinned node {i}");
            }
        }
    }

    // Analytic design-space gradient (per-column contraction of grad_node).
    let ana = morph.design_gradient(&sg.grad_node);
    // The PML-pinned chain guard must agree bit-for-bit (columns are 0 on PML).
    for (p, &ana_p) in ana.iter().enumerate() {
        let guarded = chain_node_motion_pml_pinned(&sg.grad_node, morph.velocity(p), &pinned);
        assert_eq!(
            ana_p.to_bits(),
            guarded.to_bits(),
            "design_gradient[{p}] != PML-pinned chain (columns not zero on PML?)"
        );
    }

    // Full-pipeline FD reference: move nodes by θ·D_p, re-solve the
    // port-loaded UPML forward with the per-tet Λ tensors held fixed.
    let eps_ref = eps_tensor.clone();
    let nu_ref = nu_tensor.clone();
    let g_of = |theta: f64, d: &[[f64; 3]]| -> f64 {
        let mut moved = mesh.clone();
        for (node, dn) in moved.nodes.iter_mut().zip(d) {
            node[0] += theta * dn[0];
            node[1] += theta * dn[1];
            node[2] += theta * dn[2];
        }
        let sol = driven_solve_with_ports::<B>(
            &moved,
            DrivenMaterials::MatchedUpml {
                epsilon_tensor: &eps_ref,
                nu_tensor: &nu_ref,
            },
            None,
            &bcs,
            std::slice::from_ref(&port),
            omega,
            &source,
            &device(),
        )
        .expect("port-loaded UPML forward");
        l2_objective(&sol.e_edges).0
    };

    let h = 1e-6;
    let mut worst = 0.0_f64;
    for (p, &ana_p) in ana.iter().enumerate() {
        let d = morph.velocity(p);
        let fd = (g_of(h, d) - g_of(-h, d)) / (2.0 * h);
        assert!(
            fd.abs() > 1e-8,
            "DOF {p}: FD gradient {fd} ~0 (degenerate boundary DOF)"
        );
        let rel = (ana_p - fd).abs() / fd.abs().max(f64::MIN_POSITIVE);
        worst = worst.max(rel);
        assert!(
            rel < 5e-3,
            "DOF {p}: multi-DOF adjoint {ana_p} vs central-FD {fd}, rel-err {rel:.3e} exceeds 5e-3"
        );
    }
    assert!(worst < 5e-3, "worst multi-DOF FD rel-err {worst:.3e}");

    // The DOFs must probe genuinely distinct geometry (not a constant).
    let distinct = ana
        .iter()
        .any(|&g| (g - ana[0]).abs() > 1e-9 * ana[0].abs().max(1.0));
    assert!(
        distinct,
        "all DOF gradients identical — degenerate parametrization"
    );
}

/// **The multi-DOF parametrization reduces to the single-DOF capstone path
/// bit-for-bit.** A one-column [`FreeformBoundaryMorph`] built from the exact
/// capstone Gaussian-bump velocity must return precisely the value the
/// existing single-DOF chain ([`chain_node_motion_pml_pinned`]) returns from
/// the SAME (untouched) capstone `grad_node` — the regression that pins the
/// "reduces to the #636 capstone as a special case" invariant.
#[test]
fn driven_freeform_reduces_to_single_dof_capstone_gradient_bit_for_bit() {
    use crate::constants::ETA_0_OHM as ETA_0;
    use crate::shape::{FreeformBoundaryMorph, chain_node_motion};

    let (mesh, eps_tensor, nu_tensor, interior, patch_port, source, pinned, omega) =
        capstone_patch_fixture();
    let bcs = DrivenBcs {
        pec_interior_mask: &interior,
    };
    let port = patch_port.lumped_port(50.0 / ETA_0, c64::new(1.0, 0.5));

    let sg = driven_shape_gradient_matched_upml_ports::<B, _>(
        &mesh,
        &eps_tensor,
        &nu_tensor,
        &bcs,
        std::slice::from_ref(&port),
        omega,
        &source,
        l2_objective,
        &device(),
    )
    .expect("composed capstone gradient");

    // The capstone single-DOF motion map: a +z Gaussian bump on the free
    // (non-pinned) nodes, identical to the capstone FD-gate test's `d`.
    let mut center = [0.0_f64; 3];
    let mut n_free = 0.0_f64;
    for (i, p) in mesh.nodes.iter().enumerate() {
        if !pinned[i] {
            for k in 0..3 {
                center[k] += p[k];
            }
            n_free += 1.0;
        }
    }
    for c in center.iter_mut() {
        *c /= n_free.max(1.0);
    }
    let s2 = 25.0_f64;
    let d: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if pinned[i] {
                [0.0; 3]
            } else {
                let r2 = (p[0] - center[0]).powi(2)
                    + (p[1] - center[1]).powi(2)
                    + (p[2] - center[2]).powi(2);
                [0.0, 0.0, (-r2 / (2.0 * s2)).exp()]
            }
        })
        .collect();

    let morph = FreeformBoundaryMorph::from_columns(vec![d.clone()]).unwrap();
    let dg = morph.design_gradient(&sg.grad_node);
    assert_eq!(dg.len(), 1);

    // Bit-for-bit against BOTH the raw chain and the PML-pinned chain guard.
    let raw = chain_node_motion(&sg.grad_node, &d);
    let guarded = chain_node_motion_pml_pinned(&sg.grad_node, &d, &pinned);
    assert_eq!(
        dg[0].to_bits(),
        raw.to_bits(),
        "single-column morph != chain_node_motion bit-for-bit"
    );
    assert_eq!(
        dg[0].to_bits(),
        guarded.to_bits(),
        "single-column morph != PML-pinned single-DOF chain bit-for-bit"
    );
    assert!(dg[0].abs() > 0.0, "degenerate zero gradient");
}

/// Mutation gate (issue #1264): dropping any one p=2 surface Dual-twin
/// term (`∂S_Γ/∂X`, `∂S_p/∂X`, `∂f_p/∂X`) makes the central-FD gate FAIL,
/// while the unmutated gradient passes it.
#[test]
fn p2_surface_dual_twin_mutations_fail_the_fd_gate() {
    use crate::assembly::hcurl_space::HcurlSpace;
    use crate::driven::solve::{DrivenSource, SurfaceImpedanceBc, SurfaceImpedanceModel};
    let mesh = cube_tet_mesh(2, 1.0);
    let face_nodes = |axis: usize, v: f64| -> Vec<[u32; 3]> {
        mesh.boundary_faces()
            .into_iter()
            .filter(|f| {
                f.iter()
                    .all(|&n| (mesh.nodes[n as usize][axis] - v).abs() < 1e-12)
            })
            .collect()
    };
    let port_faces = face_nodes(2, 0.0);
    let cap_faces = face_nodes(2, 1.0);
    let plates = [face_nodes(1, 0.0), face_nodes(1, 1.0)];
    let walls: Vec<&[[u32; 3]]> = plates.iter().map(Vec::as_slice).collect();
    let space = HcurlSpace::build(&mesh, ElementOrder::P2);
    let mask = space.pec_interior_mask(&mesh, &walls).expect("mask");
    let eps = vec![c64::new(2.0, -0.1); mesh.n_tets()];
    let source = CurrentSource::from_centroids(&mesh, |c| {
        [
            c64::new(0.0, c[2]),
            c64::new(0.4 * c[1], 0.2),
            c64::new(c[0], 0.0),
        ]
    });
    let port = LumpedPort {
        faces: &port_faces,
        e_hat: [0.0, 1.0, 0.0],
        resistance: 1.3,
        width: 1.0,
        length: 1.0,
        v_inc: c64::new(1.0, 0.2),
    };
    let cap = SurfaceImpedanceBc {
        triangles: &cap_faces,
        model: SurfaceImpedanceModel::GoodConductor { sigma: 40.0 },
    };
    let omega = 1.3;
    let g_of = |m: &TetMesh| -> f64 {
        let sp = HcurlSpace::build(m, ElementOrder::P2);
        let op = DrivenOperator::assemble_p2_host(
            &sp,
            m,
            DrivenMaterials::Scalar(&eps),
            &DrivenBcs {
                pec_interior_mask: &mask,
            },
            std::slice::from_ref(&port),
            std::slice::from_ref(&cap),
            DrivenSource::Constant(&source),
        )
        .unwrap();
        l2_objective(&op.solve_at(omega).unwrap().e_edges).0
    };
    let bcs = DrivenBcs {
        pec_interior_mask: &mask,
    };
    let problem = DrivenGradientProblem {
        space: &space,
        mesh: &mesh,
        materials: DrivenMaterials::Scalar(&eps),
        bcs: &bcs,
        source: &source,
        ports: std::slice::from_ref(&port),
        surfaces: std::slice::from_ref(&cap),
        wave_ports: &[],
        omega,
        excited: None,
    };
    let op = problem.assemble_p2_operator().unwrap();
    // Worst normalised FD mismatch over port-face and cap-face nodes.
    let worst = |mutation: SurfaceTermMutation| -> f64 {
        let grad = shape_gradient_p2_core(&problem, &op, None, l2_objective, mutation)
            .unwrap()
            .grad_node;
        let scale = grad.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut worst = 0.0_f64;
        #[allow(clippy::needless_range_loop)]
        for n in 0..mesh.n_nodes() {
            let z = mesh.nodes[n][2];
            if !(z.abs() < 1e-12 || (z - 1.0).abs() < 1e-12) {
                continue;
            }
            #[allow(clippy::needless_range_loop)]
            for axis in 0..3 {
                let at = |d: f64| {
                    let mut m = mesh.clone();
                    m.nodes[n][axis] += d;
                    g_of(&m)
                };
                let fd = (at(1e-6) - at(-1e-6)) / 2e-6;
                worst = worst.max((grad[n][axis] - fd).abs() / (fd.abs() + 1e-3 * scale));
            }
        }
        worst
    };
    let clean = worst(SurfaceTermMutation::None);
    eprintln!("unmutated: worst rel-err = {clean:.3e}");
    assert!(
        clean < 1e-4,
        "unmutated gradient must pass the FD gate: {clean:e}"
    );
    for m in [
        SurfaceTermMutation::DropWallMass,
        SurfaceTermMutation::DropPortMass,
        SurfaceTermMutation::DropPortFlux,
    ] {
        let w = worst(m);
        eprintln!("{m:?}: worst rel-err = {w:.3e} (FD gate FAILS if > 1e-4)");
        assert!(w > 1e-2, "{m:?} must fail the FD gate, worst rel-err {w:e}");
    }
}
