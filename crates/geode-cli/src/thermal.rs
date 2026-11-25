//! `geode thermal` and `geode self-heating`: steady heat conduction and
//! electro-thermal Joule self-heating (issue #1804).
//!
//! Wraps [`geode_core::assembly::thermal`]. Unlike the other static
//! commands, the mesh coordinates are converted to **metres** before the
//! core is called (`mesh.length_unit_m`), so conductivities (W/m/K, S/m),
//! sources (W/m³), voltages and every integrated power are plain SI with
//! no per-quantity length factors.

use std::path::Path;
use std::time::Instant;

use geode_core::assembly::conduction::ConductionTerminal;
use geode_core::assembly::thermal::{
    PicardOptions, PicardStatus, ThermalDirichlet, ThermalOptions, solve_self_heating,
    solve_thermal,
};
use geode_core::mesh::TetMesh;

use crate::check::{conductor_summaries, mesh_summary};
use crate::error::CliError;
use crate::problem::{self, FixedTemperature, HeatRegion, Problem};
use crate::report::{
    FixedTemperatureSummary, HeatRegionSummary, HeatSolverStats, Provenance, SelfHeatingReport,
    SelfHeatingSettingsSummary, ThermalReport, ThermalSettingsSummary,
};
use crate::spec::Analysis;

/// The tet mesh with node coordinates converted from mesh units to metres.
fn mesh_in_metres(p: &Problem) -> TetMesh {
    let scale = p.length_unit_m();
    let mut mesh = p.tagged.mesh.clone();
    for n in &mut mesh.nodes {
        *n = [n[0] * scale, n[1] * scale, n[2] * scale];
    }
    mesh
}

fn dirichlet(fixed: &[FixedTemperature]) -> Vec<ThermalDirichlet> {
    fixed
        .iter()
        .map(|f| ThermalDirichlet {
            nodes: f.conductor.nodes.clone(),
            temperature: f.temperature_k,
        })
        .collect()
}

fn fixed_summaries(fixed: &[FixedTemperature]) -> Vec<FixedTemperatureSummary> {
    fixed
        .iter()
        .map(|f| FixedTemperatureSummary {
            physical_group: f.conductor.surface.name.clone(),
            tag: f.conductor.surface.tag,
            n_triangles: f.conductor.surface.triangles.len(),
            n_nodes: f.conductor.nodes.len(),
            temperature_k: f.temperature_k,
        })
        .collect()
}

fn region_summaries(regions: &[HeatRegion]) -> Vec<HeatRegionSummary> {
    regions
        .iter()
        .map(|r| HeatRegionSummary {
            physical_group: r.name.clone(),
            tag: r.tag,
            n_tets: r.n_tets,
        })
        .collect()
}

/// Distinct nodes pinned by the fixed-temperature surfaces.
fn n_pinned(fixed: &[FixedTemperature]) -> usize {
    let mut nodes: Vec<u32> = fixed
        .iter()
        .flat_map(|f| f.conductor.nodes.iter().copied())
        .collect();
    nodes.sort_unstable();
    nodes.dedup();
    nodes.len()
}

/// `geode thermal`: load, solve and report.
pub fn run(spec_path: &Path, provenance: Provenance) -> Result<ThermalReport, CliError> {
    let p = problem::load(spec_path, Some(Analysis::Thermal))?;
    let target = p
        .thermal
        .as_ref()
        .expect("thermal spec has a resolved thermal section");
    let mesh = mesh_in_metres(&p);
    let n_nodes = mesh.n_nodes();

    let t0 = Instant::now();
    let sol = solve_thermal(
        &mesh,
        &target.k_w_m_k,
        &target.q_w_m3,
        &dirichlet(&target.fixed),
        ThermalOptions::default(),
    )?;
    let wall_time_s = t0.elapsed().as_secs_f64();

    let min = sol
        .temperature
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max = sol
        .temperature
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(ThermalReport {
        provenance,
        kind: "thermal",
        status: "ok",
        mesh: mesh_summary(&p),
        regions: region_summaries(&target.regions),
        thermal: ThermalSettingsSummary {
            fixed_temperatures: fixed_summaries(&target.fixed),
            element: "p1_tet",
            n_dof: n_nodes,
            n_free_dof: n_nodes - n_pinned(&target.fixed),
        },
        min_temperature_k: min,
        max_temperature_k: max,
        source_power_w: sol.power_source,
        boundary_power_w: sol.boundary_heat,
        energy_rel_mismatch: sol.energy_rel_mismatch,
        temperature_k: sol.temperature,
        solver: HeatSolverStats {
            method: "p1_galerkin",
            inner: "direct_lu",
            max_rel_residual: Some(sol.residual),
            wall_time_s,
        },
    })
}

/// `geode self-heating`: load, iterate and report.
pub fn run_self_heating(
    spec_path: &Path,
    provenance: Provenance,
) -> Result<SelfHeatingReport, CliError> {
    let p = problem::load(spec_path, Some(Analysis::SelfHeating))?;
    let target = p
        .self_heating
        .as_ref()
        .expect("self-heating spec has a resolved self_heating section");
    let picard = &p
        .spec
        .self_heating
        .as_ref()
        .expect("self-heating spec has a self_heating section")
        .picard;
    let mesh = mesh_in_metres(&p);
    let n_nodes = mesh.n_nodes();

    let defaults = PicardOptions::default();
    let opts = PicardOptions {
        relaxation: picard.relaxation.unwrap_or(defaults.relaxation),
        tol: picard.tol_k.unwrap_or(defaults.tol),
        max_iter: picard.max_iter.unwrap_or(defaults.max_iter),
        runaway_growth_iters: picard
            .runaway_growth_iters
            .unwrap_or(defaults.runaway_growth_iters),
        t_max: picard.t_max_k.unwrap_or(defaults.t_max),
        ..defaults
    };

    let terminals: Vec<ConductionTerminal> = target
        .terminals
        .iter()
        .map(|t| ConductionTerminal {
            name: t.surface.name.clone(),
            nodes: t.nodes.clone(),
            r_ext: None,
        })
        .collect();
    let laws = &target.sigma_law;
    let kk = &target.k_w_m_k;
    // `σ = σ_ref / (1 + α (T − T_ref))`; a non-positive denominator is a
    // non-physical law, reported as the core's non-finite-σ runaway.
    let sigma_of_t = |t: usize, temp: f64| {
        let l = &laws[t];
        let d = 1.0 + l.alpha_per_k * (temp - l.t_ref_k);
        if d > 0.0 {
            l.sigma_ref_s_m / d
        } else {
            f64::INFINITY
        }
    };
    let k_of_t = |t: usize, _temp: f64, _sigma: f64| kk[t];

    let t0 = Instant::now();
    let res = solve_self_heating(
        &mesh,
        &terminals,
        &target.voltages_v,
        &dirichlet(&target.fixed),
        target.t_initial_k,
        sigma_of_t,
        k_of_t,
        opts,
    )?;
    let wall_time_s = t0.elapsed().as_secs_f64();

    let termination = match res.status {
        PicardStatus::Converged => "converged",
        PicardStatus::NotConverged => "not_converged",
        PicardStatus::Runaway => "runaway",
    };
    let peak = res
        .temperature
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(SelfHeatingReport {
        provenance,
        kind: "self_heating",
        status: "ok",
        mesh: mesh_summary(&p),
        regions: region_summaries(&target.regions),
        self_heating: SelfHeatingSettingsSummary {
            terminals: conductor_summaries(&target.terminals),
            voltages_v: target.voltages_v.clone(),
            fixed_temperatures: fixed_summaries(&target.fixed),
            initial_temperature_k: target.t_initial_k,
            relaxation: opts.relaxation,
            tol_k: opts.tol,
            max_iter: opts.max_iter,
            runaway_growth_iters: opts.runaway_growth_iters,
            t_max_k: opts.t_max,
            element: "p1_tet",
            n_dof: n_nodes,
        },
        termination,
        iterations: res.iterations,
        update_history_k: res.history,
        has_thermal_solution: res.thermal.is_some(),
        peak_temperature_k: peak,
        source_power_w: res.thermal.as_ref().map(|t| t.power_source),
        boundary_power_w: res.thermal.as_ref().map(|t| t.boundary_heat),
        energy_rel_mismatch: res.thermal.as_ref().map(|t| t.energy_rel_mismatch),
        terminal_current_a: res.conduction.as_ref().map(|c| c.terminal_current.clone()),
        temperature_k: res.temperature,
        solver: HeatSolverStats {
            method: "picard_p1_galerkin",
            inner: "direct_lu",
            max_rel_residual: res.thermal.as_ref().map(|t| t.residual),
            wall_time_s,
        },
    })
}
