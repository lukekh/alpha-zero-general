//! One whitespace-delimited request per line, one JSON response per line.
use intransitive_rust_teacher::{evaluate_weighted, pressure, Config, Position, Search, Weights};
use std::io::{self, BufRead, Write};

fn bytes(text: &str) -> Result<Vec<u8>, String> {
    if text.len() != 13608 || !text.is_ascii() {
        return Err("Expected state-v2 hex".into());
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).map_err(|_| "Invalid hex".into()))
        .collect()
}
fn number<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.parse().map_err(|_| "Invalid numeric argument".into())
}
fn flag(s: &str) -> Result<bool, String> {
    match s { "0" => Ok(false), "1" => Ok(true), _ => Err("Expected boolean 0 or 1".into()) }
}
/// Material mode digit: 0 flat, 1 variable, 2 variable without the square root.
/// Extending the existing digit keeps every older caller's 0 and 1 valid.
fn material_mode(s: &str) -> Result<(bool, bool), String> {
    match s {
        "0" => Ok((false, false)),
        "1" => Ok((true, false)),
        "2" => Ok((true, true)),
        _ => Err("Expected material mode 0, 1 or 2".into()),
    }
}
fn handle(line: &str, cache: &mut Option<(Config, Search)>) -> Result<String, String> {
    let v: Vec<_> = line.split_whitespace().collect();
    match v.first().copied() {
        Some("search" | "search_reuse") if matches!(v.len(),10|14|15|16|20|21|22|23|24|27) => {
            let has_weights=matches!(v.len(),14|15|20|21|22|23);
            let has_mode=matches!(v.len(),15|21|22|23);
            let has_selective=matches!(v.len(),16|20|21|22|23);
            let (variable,linear)=if has_mode {material_mode(v[13])?} else {(false,false)};
            let weights=if has_weights { Weights {variable_material_enabled:variable,variable_material_linear:linear,material:number(v[9])?,advantage:number(v[10])?,attack:number(v[11])?,defence:number(v[12])?} } else {Weights::default()};
            let offset=9+if has_weights {4} else {0}+if has_mode {1} else {0};
            let mut config=Config {weights,depth:number(v[1])?,milliseconds:number(v[2])?,node_limit:number(v[3])?,
                radius:number(v[4])?,pressure_weight:number(v[5])?,proof_depth:number(v[6])?,
                proof_nodes:number(v[7])?,table_entries:number(v[8])?,..Config::default()};
            if has_selective {
                config.nmp_enabled=number(v[offset])?;
                config.nmp_min_depth=number(v[offset+1])?;
                config.nmp_reduction=number(v[offset+2])?;
                config.futility_enabled=number(v[offset+3])?;
                config.futility_max_depth=number(v[offset+4])?;
                config.futility_margin=number(v[offset+5])?;
            }
            if v.len()>=22 {config.selective_evaluator_enabled=number(v[20])?;}
            if v.len()>=23 {config.mvv_lva_enabled=number(v[21])?;}
            if v.len()>=24 {config.certificate_enabled=number(v[22])?;}
            // The certificate-as-a-bound group travels whole: a cutoff without
            // its depth threshold would silently pick this binary's default.
            if v.len()==27 {
                config.certificate_cutoff_enabled=number(v[23])?;
                config.certificate_cutoff_min_depth=number(v[24])?;
                config.certificate_guard_enabled=number(v[25])?;
            }
            let p=Position::from_bytes(&bytes(v[v.len()-1])?)?;
            if cache.as_ref().map(|x| &x.0)!=Some(&config) {
                *cache=Some((config.clone(),Search::new(config.clone())?));
            }
            let search=&mut cache.as_mut().unwrap().1;
            let r=if v[0]=="search_reuse" {search.analyze_reusing(&p)} else {search.analyze(&p)};
            let stats=search.selective_stats;
            let ordering=format!("{{\"mvv_lva_enabled\":{},\"mvv_lva_nodes\":{},\"mvv_lva_captures\":{}}}",config.mvv_lva_enabled,search.mvv_lva_stats[0],search.mvv_lva_stats[1]);
            let runs=search.certificate_stats;
            let certificate=format!("{{\"leaf_enabled\":{},\"cutoff_enabled\":{},\"cutoff_min_depth\":{},\"guard_enabled\":{},\"plies\":{},\"probes\":{},\"gated\":{},\"certified\":{},\"cutoffs\":{},\"guards\":{}}}",
                config.certificate_enabled,config.certificate_cutoff_enabled,config.certificate_cutoff_min_depth,
                config.certificate_guard_enabled,config.certificate_plies,runs[0],runs[1],runs[2],runs[3],runs[4]);
            let settings=format!("{{\"version\":\"intransitive-selective-v1\",\"selective_evaluator_enabled\":{},\"enabled\":{},\"effective\":{},\"nmp_enabled\":{},\"nmp_min_depth\":{},\"nmp_reduction\":{},\"futility_enabled\":{},\"futility_max_depth\":{},\"futility_margin\":{},\"nmp_attempts\":{},\"nmp_cutoffs\":{},\"nmp_skips\":{},\"verification_searches\":{},\"verification_failures\":{},\"futility_eligible\":{},\"futility_pruned\":{},\"static_evaluations\":{},\"null_nodes\":{},\"verification_nodes\":{}}}",
                config.selective_evaluator_enabled,config.nmp_enabled||config.futility_enabled,(config.nmp_enabled||config.futility_enabled)&&config.selective_supported(),
                config.nmp_enabled,config.nmp_min_depth,config.nmp_reduction,config.futility_enabled,config.futility_max_depth,config.futility_margin,
                stats[0],stats[1],stats[2],stats[3],stats[4],stats[5],stats[6],stats[7],stats[8],stats[9]);
            Ok(format!("{{\"ordering\":{},\"certificate\":{},\"selective\":{},\"action\":{},\"score\":{},\"completed_depth\":{},\"target_depth\":{},\"complete\":{},\"stop_reason\":{:?},\"nodes\":{},\"proof_nodes\":{},\"seconds\":{},\"pv\":{:?},\"tt_hits\":{},\"table_entries\":{},\"table_bytes\":{},\"work\":{}}}",
                ordering,certificate,settings,r.action.map_or("null".into(),|x|x.to_string()),r.score.map_or("null".into(),|x|x.to_string()),
                r.completed_depth,r.target_depth,r.complete,r.stop_reason,r.nodes,r.proof_nodes,r.seconds,r.pv,search.tt_hits,search.table_entries(),search.table_bytes(),r.nodes))
        },
        Some("inspect") if v.len()==4 || v.len()==8 || v.len()==9 => {
            let radius=number(v[1])?;let weight:f64=number(v[2])?;
            if !(3..=4).contains(&radius) || !weight.is_finite() {return Err("Invalid pressure settings".into());}
            let (variable,linear)=if v.len()==9 {material_mode(v[7])?} else {(false,false)};
            let weights=if v.len()>=8 {Weights {variable_material_enabled:variable,variable_material_linear:linear,material:number(v[3])?,advantage:number(v[4])?,attack:number(v[5])?,defence:number(v[6])?}} else {Weights::default()};
            weights.validate()?;
            let p=Position::from_bytes(&bytes(v[v.len()-1])?)?;
            let term=p.terminal(true);let official=p.terminal(false);
            let legal=if term.is_some() {vec![]} else {p.raw_legal(p.side)};
            Ok(format!("{{\"legal\":{:?},\"score\":{},\"pressure\":{:?},\"reason\":{:?},\"official_reason\":{:?}}}",
                legal,evaluate_weighted(&p,radius,weight,&weights),pressure(p.board(),radius),term.map_or("ongoing",|x|x.1),official.map_or("ongoing",|x|x.1)))
        },
        Some("apply") if v.len()==4 => {
            let action=number(v[1])?;let modelling=match v[2] {"0"=>false,"1"=>true,_=>return Err("Invalid rule mode".into())};
            let mut p=Position::from_bytes(&bytes(v[3])?)?;p.apply(action,modelling)?;
            let digits=b"0123456789abcdef";
            let mut hex=String::with_capacity(13608);
            for b in p.to_bytes() {
                hex.push(digits[(b >> 4) as usize] as char);
                hex.push(digits[(b & 15) as usize] as char);
            }
            Ok(format!("{{\"state_hex\":{:?}}}",hex))
        },
        _=>Err("Expected search depth ms nodes radius weight proof_depth proof_nodes table_entries state_hex; inspect radius weight state_hex; apply action modelling state_hex".into()),
    }
}
fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    let mut cache = None;
    for line in stdin.lock().lines() {
        let response = match line {
            Ok(line) => handle(&line, &mut cache),
            Err(e) => Err(e.to_string()),
        };
        let text = response.unwrap_or_else(|e| format!("{{\"error\":{:?}}}", e));
        if writeln!(stdout, "{}", text)
            .and_then(|_| stdout.flush())
            .is_err()
        {
            break;
        }
    }
}
