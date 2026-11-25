//! Golden tests (issue #1804): `geode thermal` and `geode self-heating` on
//! a uniformly heated bar, mirroring `geode-core/tests/thermal_conduction.rs`.
//!
//! The bar `[0,LX]×[0,LY]×[0,LZ]` mesh units (mm) is built in-process from
//! the unit-cube tet mesh and written as a tagged MSH 4.1 file with end
//! surfaces `left` (`x = 0`) and `right` (`x = LX`). Both ends are held at
//! `T0`; the analytic profile is `T0 + q x(L−x)/(2k)` with `x`, `L` in
//! metres.

#[path = "support/msh.rs"]
mod msh;
#[path = "support/scratch.rs"]
mod scratch_support;

use std::path::Path;

use geode_core::mesh::{TetMesh, cube_tet_mesh};
use scratch_support::Scratch;
use serde_json::{Value, json};

const LX: f64 = 2.0;
const LY: f64 = 0.5;
const LZ: f64 = 0.5;
const UNIT_M: f64 = 1e-3;
const T0: f64 = 300.0;

#[path = "support/cli.rs"]
mod cli_support;
use cli_support::geode;

fn bar_mesh() -> TetMesh {
    let mut mesh = cube_tet_mesh(8, 1.0);
    for p in &mut mesh.nodes {
        *p = [p[0] * LX, p[1] * LY, p[2] * LZ];
    }
    mesh
}

/// Write the bar mesh; returns its path.
fn write_bar(dir: &Path, mesh: &TetMesh) -> std::path::PathBuf {
    let face_at = |x: f64| -> Vec<[u32; 3]> {
        mesh.boundary_faces()
            .into_iter()
            .filter(|f| {
                f.iter()
                    .all(|&n| (mesh.nodes[n as usize][0] - x).abs() < 1e-12)
            })
            .collect()
    };
    let (left, right) = (face_at(0.0), face_at(LX));
    assert!(!left.is_empty() && !right.is_empty());
    let text = msh::write_msh_volumes(
        &mesh.nodes,
        &[(1, "bar", mesh.tets.as_slice())],
        &[
            (11, "left", left.as_slice()),
            (12, "right", right.as_slice()),
        ],
    );
    let path = dir.join("bar.msh");
    std::fs::write(&path, text).unwrap();
    path
}

fn spec(mesh: &Path, section: &str, body: Value) -> Value {
    json!({
        "schema_version": 1,
        "mesh": {"path": mesh.to_str().unwrap(), "length_unit_m": UNIT_M},
        section: body
    })
}

fn fixed_ends() -> Value {
    json!([
        {"surface": "left", "temperature_k": T0},
        {"surface": "right", "temperature_k": T0}
    ])
}

/// Run `geode <cmd>`; the report on success, the error message otherwise.
fn run(dir: &Path, cmd: &str, name: &str, spec: &Value) -> Result<Value, String> {
    let path = dir.join(format!("{name}.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(spec).unwrap()).unwrap();
    let out = geode(&[cmd, path.to_str().unwrap()]);
    let v: Value = serde_json::from_slice(&out.stdout).expect("JSON report");
    if out.status.success() {
        Ok(v)
    } else {
        Err(v["error"]["message"].as_str().unwrap_or("").to_string())
    }
}

fn f(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

#[test]
fn heated_bar_matches_parabola_and_energy_balance() {
    let dir = Scratch::new("geode-", "thermal-bar");
    let mesh = bar_mesh();
    let path = write_bar(&dir, &mesh);
    let (k, q) = (40.0, 4.0e11);
    let s = spec(
        &path,
        "thermal",
        json!({
            "regions": [{"physical_group": "bar", "conductivity_w_m_k": k, "source_w_m3": q}],
            "fixed_temperatures": fixed_ends()
        }),
    );
    let r = run(&dir, "thermal", "bar", &s).expect("solve");
    assert_eq!(r["kind"], "thermal");
    let l = LX * UNIT_M;
    let rise = q * (l / 2.0) * (l / 2.0) / (2.0 * k);
    let temps = r["temperature_k"].as_array().unwrap();
    assert_eq!(temps.len(), mesh.n_nodes());
    for (i, p) in mesh.nodes.iter().enumerate() {
        let x = p[0] * UNIT_M;
        let exact = T0 + q * x * (l - x) / (2.0 * k);
        assert!(
            (f(&temps[i]) - exact).abs() / rise < 0.01,
            "node {i}: {} vs {exact}",
            temps[i]
        );
    }
    assert!((f(&r["max_temperature_k"]) - T0 - rise).abs() / rise < 0.01);
    let vol = l * (LY * UNIT_M) * (LZ * UNIT_M);
    assert!((f(&r["source_power_w"]) - q * vol).abs() / (q * vol) < 1e-12);
    assert!(f(&r["energy_rel_mismatch"]) < 1e-8);
    let (src, out) = (f(&r["source_power_w"]), f(&r["boundary_power_w"]));
    assert!((out - src).abs() / src < 1e-8);
}

#[test]
fn missing_fixed_surface_is_a_structured_input_error() {
    let dir = Scratch::new("geode-", "thermal-missing");
    let path = write_bar(&dir, &bar_mesh());
    let s = spec(
        &path,
        "thermal",
        json!({
            "regions": [{"physical_group": "bar", "conductivity_w_m_k": 40.0}],
            "fixed_temperatures": [{"surface": "nowhere", "temperature_k": T0}]
        }),
    );
    let p = dir.join("m.json");
    std::fs::write(&p, serde_json::to_vec(&s).unwrap()).unwrap();
    let out = geode(&["thermal", p.to_str().unwrap()]);
    assert!(!out.status.success());
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["kind"], "error");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("nowhere"),
        "{v}"
    );
}

#[test]
fn unlisted_region_and_wrong_subcommand_are_rejected() {
    let dir = Scratch::new("geode-", "thermal-bad");
    let path = write_bar(&dir, &bar_mesh());
    let s = spec(
        &path,
        "thermal",
        json!({"regions": [], "fixed_temperatures": fixed_ends()}),
    );
    let e = run(&dir, "thermal", "empty", &s).expect_err("no regions");
    assert!(e.contains("regions"), "{e}");
    let s = spec(
        &path,
        "thermal",
        json!({
            "regions": [{"physical_group": "bar", "conductivity_w_m_k": 40.0}],
            "fixed_temperatures": fixed_ends()
        }),
    );
    let e = run(&dir, "self-heating", "wrong", &s).expect_err("wrong subcommand");
    assert!(e.contains("geode thermal"), "{e}");
}

fn self_heating_spec(path: &Path, volts: f64, picard: Value) -> Value {
    spec(
        path,
        "self_heating",
        json!({
            "terminals": [
                {"surface": "left", "voltage_v": volts},
                {"surface": "right", "voltage_v": 0.0}
            ],
            "fixed_temperatures": fixed_ends(),
            "initial_temperature_k": T0,
            "regions": [{
                "physical_group": "bar",
                "sigma_ref_s_m": 5.8e7,
                "alpha_per_k": 4.0e-3,
                "t_ref_k": T0,
                "conductivity_w_m_k": 400.0
            }],
            "picard": picard
        }),
    )
}

#[test]
fn self_heating_distinguishes_the_three_outcomes() {
    let dir = Scratch::new("geode-", "self-heating-bar");
    let path = write_bar(&dir, &bar_mesh());

    // Mild heating: converges, peak above the prescribed end temperature.
    let r = run(
        &dir,
        "self-heating",
        "mild",
        &self_heating_spec(&path, 1.0e-2, json!({})),
    )
    .expect("mild");
    assert_eq!(r["kind"], "self_heating");
    assert_eq!(r["termination"], "converged");
    assert_eq!(r["has_thermal_solution"], true);
    assert!(f(&r["peak_temperature_k"]) > T0 + 0.1, "{r}");
    assert!(f(&r["energy_rel_mismatch"]) < 1e-8);
    assert_eq!(
        r["temperature_k"].as_array().unwrap().len(),
        r["mesh"]["n_nodes"].as_u64().unwrap() as usize
    );

    // One-iteration limit: iteration exhaustion, not an error.
    let r = run(
        &dir,
        "self-heating",
        "limit",
        &self_heating_spec(&path, 1.0e-2, json!({"max_iter": 1})),
    )
    .expect("limit");
    assert_eq!(r["termination"], "not_converged");
    assert_eq!(r["iterations"], 1);
    assert_eq!(r["has_thermal_solution"], true);

    // Heating past the temperature ceiling: runaway, with a last iterate.
    let r = run(
        &dir,
        "self-heating",
        "ceiling",
        &self_heating_spec(&path, 1.0e-1, json!({"t_max_k": 320.0})),
    )
    .expect("ceiling");
    assert_eq!(r["termination"], "runaway");

    // Default ceiling crossed on the very first iteration: no thermal
    // solution is kept, but the report still carries the temperature field.
    let r = run(
        &dir,
        "self-heating",
        "first",
        &self_heating_spec(&path, 10.0, json!({})),
    )
    .expect("first");
    assert_eq!(r["termination"], "runaway");
    assert_eq!(r["has_thermal_solution"], false);
    assert!(r["energy_rel_mismatch"].is_null());
    assert!(!r["temperature_k"].as_array().unwrap().is_empty());
}
