//! One whitespace-delimited request per line, one JSON response per line.
use intransitive_rust_teacher::{evaluate, pressure, Config, Position, Search};
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
fn handle(line: &str, cache: &mut Option<(Config, Search)>) -> Result<String, String> {
    let v: Vec<_> = line.split_whitespace().collect();
    match v.first().copied() {
        Some("search" | "search_reuse") if v.len()==10 || v.len()==16 => {
            let config=Config {depth:number(v[1])?,milliseconds:number(v[2])?,node_limit:number(v[3])?,
                radius:number(v[4])?,pressure_weight:number(v[5])?,proof_depth:number(v[6])?,
                proof_nodes:number(v[7])?,table_entries:number(v[8])?,
                nmp_enabled:if v.len()==16 {number::<bool>(v[9])?} else {false},
                nmp_min_depth:if v.len()==16 {number(v[10])?} else {3},
                nmp_reduction:if v.len()==16 {number(v[11])?} else {1},
                futility_enabled:if v.len()==16 {number::<bool>(v[12])?} else {false},
                futility_max_depth:if v.len()==16 {number(v[13])?} else {2},
                futility_margin:if v.len()==16 {number(v[14])?} else {1.0}};
            let p=Position::from_bytes(&bytes(v[v.len()-1])?)?;
            if cache.as_ref().map(|x| &x.0)!=Some(&config) {
                *cache=Some((config.clone(),Search::new(config.clone())?));
            }
            let search=&mut cache.as_mut().unwrap().1;
            let r=if v[0]=="search_reuse" {search.analyze_reusing(&p)} else {search.analyze(&p)};
            let stats=search.selective_stats;
            let settings=format!("{{\"version\":\"intransitive-selective-v1\",\"enabled\":{},\"effective\":{},\"nmp_enabled\":{},\"nmp_min_depth\":{},\"nmp_reduction\":{},\"futility_enabled\":{},\"futility_max_depth\":{},\"futility_margin\":{},\"nmp_attempts\":{},\"nmp_cutoffs\":{},\"nmp_skips\":{},\"verification_searches\":{},\"verification_failures\":{},\"futility_eligible\":{},\"futility_pruned\":{},\"static_evaluations\":{},\"null_nodes\":{},\"verification_nodes\":{}}}",
                config.nmp_enabled||config.futility_enabled,(config.nmp_enabled||config.futility_enabled)&&config.pressure_weight<=20.0,
                config.nmp_enabled,config.nmp_min_depth,config.nmp_reduction,config.futility_enabled,config.futility_max_depth,config.futility_margin,
                stats[0],stats[1],stats[2],stats[3],stats[4],stats[5],stats[6],stats[7],stats[8],stats[9]);
            Ok(format!("{{\"selective\":{},\"action\":{},\"score\":{},\"completed_depth\":{},\"target_depth\":{},\"complete\":{},\"stop_reason\":{:?},\"nodes\":{},\"proof_nodes\":{},\"seconds\":{},\"pv\":{:?},\"tt_hits\":{},\"table_entries\":{},\"table_bytes\":{},\"work\":{}}}",
                settings,r.action.map_or("null".into(),|x|x.to_string()),r.score.map_or("null".into(),|x|x.to_string()),
                r.completed_depth,r.target_depth,r.complete,r.stop_reason,r.nodes,r.proof_nodes,r.seconds,r.pv,search.tt_hits,search.table_entries(),search.table_bytes(),r.nodes))
        },
        Some("inspect") if v.len()==4 => {
            let radius=number(v[1])?;let weight:f64=number(v[2])?;
            if !(3..=4).contains(&radius) || !weight.is_finite() || weight<0.0 {return Err("Invalid pressure settings".into());}
            let p=Position::from_bytes(&bytes(v[3])?)?;
            let term=p.terminal(true);let official=p.terminal(false);
            let legal=if term.is_some() {vec![]} else {p.raw_legal(p.side)};
            Ok(format!("{{\"legal\":{:?},\"score\":{},\"pressure\":{:?},\"reason\":{:?},\"official_reason\":{:?}}}",
                legal,evaluate(&p,radius,weight),pressure(p.board(),radius),term.map_or("ongoing",|x|x.1),official.map_or("ongoing",|x|x.1)))
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
