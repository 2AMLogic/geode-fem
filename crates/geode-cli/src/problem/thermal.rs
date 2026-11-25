//! Thermal and self-heating spec validation and resolution (issue #1804).

use super::*;
use crate::spec::{FixedTemperatureSpec, SelfHeatingSpec, ThermalSpec};

/// Rules shared by `thermal` and `self_heating` specs (scalar, before the
/// mesh is read). Both are static real solves: anything that only has
/// meaning for a frequency-domain, electrostatic or magnetostatic solve is
/// rejected rather than silently ignored.
pub(super) fn validate_heat(spec: &ProblemSpec) -> Result<(), CliError> {
    let name = spec.analysis().name();
    let static_only = |what: &str, why: &str| {
        invalid(format!(
            "a {name} spec cannot have {what}: {why} (`geode {name}` is a static solve)"
        ))
    };
    if !spec.materials.is_empty() {
        return Err(static_only(
            "`materials`",
            "list each volume region with its conductivity (and source / resistivity law) in \
             the analysis section's `regions` instead",
        ));
    }
    if !spec.ports.is_empty() {
        return Err(static_only("`ports`", "lumped ports are frequency-domain"));
    }
    if !spec.wave_ports.is_empty() {
        return Err(static_only(
            "`wave_ports`",
            "wave ports are frequency-domain modal boundaries",
        ));
    }
    if spec.frequencies.is_some() {
        return Err(static_only("`frequencies`", "there is no frequency"));
    }
    if !spec.absorbing_regions.is_empty() {
        return Err(static_only(
            "`absorbing_regions`",
            "UPML is a frequency-domain absorber",
        ));
    }
    let bcs = &spec.boundary_conditions;
    if !bcs.leontovich.is_empty() || !bcs.silver_muller.is_empty() || !bcs.pec.is_empty() {
        return Err(static_only(
            "`boundary_conditions` (pec / leontovich / silver_muller)",
            "the fixed-temperature surfaces are the only Dirichlet surfaces; every other \
             boundary is adiabatic",
        ));
    }
    if let SolverSpec::Iterative { .. } = spec.solver {
        return Err(invalid(format!(
            "a {name} spec supports only `solver.mode = \"direct\"` (sparse LU)"
        )));
    }
    let finite_pos = |what: &str, v: f64| {
        if v.is_finite() && v > 0.0 {
            Ok(())
        } else {
            Err(invalid(format!("{what} must be finite and > 0 (got {v})")))
        }
    };
    let fixed = |section: &str, f: &[FixedTemperatureSpec]| -> Result<(), CliError> {
        if f.is_empty() {
            return Err(invalid(format!(
                "{section}.fixed_temperatures must name at least one fixed-temperature surface \
                 (the heat problem is singular without one)"
            )));
        }
        for (i, e) in f.iter().enumerate() {
            finite_pos(
                &format!("{section}.fixed_temperatures[{}].temperature_k", e.surface),
                e.temperature_k,
            )?;
            if f[..i].iter().any(|o| o.surface == e.surface) {
                return Err(invalid(format!(
                    "{section}.fixed_temperatures lists surface `{}` more than once",
                    e.surface
                )));
            }
        }
        Ok(())
    };
    let distinct = |section: &str, names: Vec<&str>| -> Result<(), CliError> {
        if names.is_empty() {
            return Err(invalid(format!(
                "{section}.regions must list at least one volume region"
            )));
        }
        for (i, n) in names.iter().enumerate() {
            if names[..i].contains(n) {
                return Err(invalid(format!(
                    "{section}.regions lists volume region `{n}` more than once"
                )));
            }
        }
        Ok(())
    };
    if let Some(t) = &spec.thermal {
        fixed("thermal", &t.fixed_temperatures)?;
        distinct(
            "thermal",
            t.regions
                .iter()
                .map(|r| r.physical_group.as_str())
                .collect(),
        )?;
        for r in &t.regions {
            finite_pos(
                &format!("thermal.regions[{}].conductivity_w_m_k", r.physical_group),
                r.conductivity_w_m_k,
            )?;
            if !r.source_w_m3.is_finite() {
                return Err(invalid(format!(
                    "thermal.regions[{}].source_w_m3 must be finite (got {})",
                    r.physical_group, r.source_w_m3
                )));
            }
        }
    }
    if let Some(h) = &spec.self_heating {
        fixed("self_heating", &h.fixed_temperatures)?;
        distinct(
            "self_heating",
            h.regions
                .iter()
                .map(|r| r.physical_group.as_str())
                .collect(),
        )?;
        finite_pos(
            "self_heating.initial_temperature_k",
            h.initial_temperature_k,
        )?;
        if h.terminals.is_empty() {
            return Err(invalid(
                "self_heating.terminals must name at least one voltage-driven terminal surface",
            ));
        }
        for (i, t) in h.terminals.iter().enumerate() {
            if !t.voltage_v.is_finite() {
                return Err(invalid(format!(
                    "self_heating.terminals[{}].voltage_v must be finite (got {})",
                    t.surface, t.voltage_v
                )));
            }
            if h.terminals[..i].iter().any(|o| o.surface == t.surface) {
                return Err(invalid(format!(
                    "self_heating.terminals lists surface `{}` more than once",
                    t.surface
                )));
            }
        }
        for r in &h.regions {
            let g = &r.physical_group;
            finite_pos(
                &format!("self_heating.regions[{g}].conductivity_w_m_k"),
                r.conductivity_w_m_k,
            )?;
            if !(r.sigma_ref_s_m.is_finite() && r.sigma_ref_s_m >= 0.0) {
                return Err(invalid(format!(
                    "self_heating.regions[{g}].sigma_ref_s_m must be finite and >= 0 (got {})",
                    r.sigma_ref_s_m
                )));
            }
            if !(r.alpha_per_k.is_finite() && r.t_ref_k.is_finite()) {
                return Err(invalid(format!(
                    "self_heating.regions[{g}]: alpha_per_k and t_ref_k must be finite"
                )));
            }
        }
        let p = &h.picard;
        if let Some(w) = p.relaxation
            && !(w > 0.0 && w <= 1.0)
        {
            return Err(invalid(format!(
                "self_heating.picard.relaxation must be in (0, 1] (got {w})"
            )));
        }
        for (what, v) in [("tol_k", p.tol_k), ("t_max_k", p.t_max_k)] {
            if let Some(v) = v {
                finite_pos(&format!("self_heating.picard.{what}"), v)?;
            }
        }
        if p.max_iter == Some(0) || p.runaway_growth_iters == Some(0) {
            return Err(invalid(
                "self_heating.picard.max_iter and runaway_growth_iters must be >= 1",
            ));
        }
    }
    Ok(())
}

/// Bind each fixed-temperature entry to its tagged triangles and nodes.
fn resolve_fixed(
    fixed: &[FixedTemperatureSpec],
    tags: &[i32],
    surface: &dyn Fn(&str, i32, &str) -> Result<Surface, CliError>,
) -> Result<Vec<FixedTemperature>, CliError> {
    fixed
        .iter()
        .zip(tags)
        .map(|(f, &tag)| {
            let s = surface(&f.surface, tag, "fixed-temperature")?;
            let mut nodes: Vec<u32> = s.triangles.iter().flatten().copied().collect();
            nodes.sort_unstable();
            nodes.dedup();
            Ok(FixedTemperature {
                conductor: Conductor { surface: s, nodes },
                temperature_k: f.temperature_k,
            })
        })
        .collect()
}

/// Map every tet to the index of the listed region that owns its tag.
/// A tet in an unlisted region is an error (a silent default `k` would be
/// a wrong answer).
fn assign_regions(
    names: &[&str],
    tags: &[i32],
    tagged: &TaggedTetMesh,
) -> Result<(Vec<usize>, Vec<HeatRegion>), CliError> {
    let mut regions: Vec<HeatRegion> = names
        .iter()
        .zip(tags)
        .map(|(n, &tag)| HeatRegion {
            name: (*n).to_string(),
            tag,
            n_tets: 0,
        })
        .collect();
    let mut idx = Vec::with_capacity(tagged.tet_physical_tags.len());
    let mut uncovered: BTreeMap<i32, usize> = BTreeMap::new();
    for &t in &tagged.tet_physical_tags {
        match regions.iter().position(|r| r.tag == t) {
            Some(i) => {
                regions[i].n_tets += 1;
                idx.push(i);
            }
            None => {
                *uncovered.entry(t).or_default() += 1;
                idx.push(usize::MAX);
            }
        }
    }
    if !uncovered.is_empty() {
        let list: Vec<String> = uncovered
            .iter()
            .map(|(tag, n)| {
                let name = tagged
                    .mesh
                    .physical_groups
                    .get(&(3, *tag))
                    .map_or_else(|| format!("tag {tag}"), |n| format!("`{n}`"));
                format!("{name} ({n} tets)")
            })
            .collect();
        return Err(invalid(format!(
            "volume region(s) {} are in the mesh but not listed in `regions`: every tet needs a \
             thermal conductivity",
            list.join(", ")
        )));
    }
    Ok((idx, regions))
}

pub(super) fn resolve_thermal(
    t: &ThermalSpec,
    tags: &GroupTags,
    tagged: &TaggedTetMesh,
    surface: &dyn Fn(&str, i32, &str) -> Result<Surface, CliError>,
) -> Result<ThermalTarget, CliError> {
    let names: Vec<&str> = t
        .regions
        .iter()
        .map(|r| r.physical_group.as_str())
        .collect();
    let (idx, regions) = assign_regions(&names, &tags.heat_region, tagged)?;
    let k_w_m_k = idx
        .iter()
        .map(|&i| t.regions[i].conductivity_w_m_k)
        .collect();
    let q_w_m3 = idx.iter().map(|&i| t.regions[i].source_w_m3).collect();
    Ok(ThermalTarget {
        fixed: resolve_fixed(&t.fixed_temperatures, &tags.heat_fixed, surface)?,
        regions,
        k_w_m_k,
        q_w_m3,
    })
}

pub(super) fn resolve_self_heating(
    h: &SelfHeatingSpec,
    tags: &GroupTags,
    tagged: &TaggedTetMesh,
    surface: &dyn Fn(&str, i32, &str) -> Result<Surface, CliError>,
) -> Result<SelfHeatingTarget, CliError> {
    let names: Vec<&str> = h
        .regions
        .iter()
        .map(|r| r.physical_group.as_str())
        .collect();
    let (idx, regions) = assign_regions(&names, &tags.heat_region, tagged)?;
    let mut terminals = Vec::with_capacity(h.terminals.len());
    for (t, &tag) in h.terminals.iter().zip(&tags.heat_terminal) {
        let s = surface(&t.surface, tag, "self-heating terminal")?;
        let mut nodes: Vec<u32> = s.triangles.iter().flatten().copied().collect();
        nodes.sort_unstable();
        nodes.dedup();
        terminals.push(Conductor { surface: s, nodes });
    }
    for (i, t) in terminals.iter().enumerate() {
        for u in &terminals[i + 1..] {
            if t.nodes.iter().any(|x| u.nodes.binary_search(x).is_ok()) {
                return Err(invalid(format!(
                    "self-heating terminals `{}` and `{}` share mesh nodes: touching terminals \
                     are electrically shorted",
                    t.surface.name, u.surface.name
                )));
            }
        }
    }
    Ok(SelfHeatingTarget {
        voltages_v: h.terminals.iter().map(|t| t.voltage_v).collect(),
        terminals,
        fixed: resolve_fixed(&h.fixed_temperatures, &tags.heat_fixed, surface)?,
        t_initial_k: h.initial_temperature_k,
        regions,
        sigma_law: idx
            .iter()
            .map(|&i| SigmaLaw {
                sigma_ref_s_m: h.regions[i].sigma_ref_s_m,
                alpha_per_k: h.regions[i].alpha_per_k,
                t_ref_k: h.regions[i].t_ref_k,
            })
            .collect(),
        k_w_m_k: idx
            .iter()
            .map(|&i| h.regions[i].conductivity_w_m_k)
            .collect(),
    })
}
