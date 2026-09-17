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
fn handle(line: &str, cache: &mut Option<(Config, Search)>) -> Result<String, String> {
    let v: Vec<_> = line.split_whitespace().collect();
    match v.first().copied() {
        Some("search" | "search_reuse") if v.len()==10 || v.len()==14 || v.len()==15 => {
            let weights=if v.len()>=14 { Weights {variable_material_enabled:if v.len()==15 {flag(v[13])?} else {false},material:number(v[9])?,advantage:number(v[10])?,attack:number(v[11])?,defence:number(v[12])?} } else {Weights::default()};
            let config=Config {weights,depth:number(v[1])?,milliseconds:number(v[2])?,node_limit:number(v[3])?,
                radius:number(v[4])?,pressure_weight:number(v[5])?,proof_depth:number(v[6])?,
                proof_nodes:number(v[7])?,table_entries:number(v[8])?};
            let p=Position::from_bytes(&bytes(v[v.len()-1])?)?;
            if cache.as_ref().map(|x| &x.0)!=Some(&config) {
                *cache=Some((config.clone(),Search::new(config)?));
            }
            let search=&mut cache.as_mut().unwrap().1;
            let r=if v[0]=="search_reuse" {search.analyze_reusing(&p)} else {search.analyze(&p)};
            Ok(format!("{{\"action\":{},\"score\":{},\"completed_depth\":{},\"target_depth\":{},\"complete\":{},\"stop_reason\":{:?},\"nodes\":{},\"proof_nodes\":{},\"seconds\":{},\"pv\":{:?},\"tt_hits\":{},\"table_entries\":{}}}",
                r.action.map_or("null".into(),|x|x.to_string()),r.score.map_or("null".into(),|x|x.to_string()),
                r.completed_depth,r.target_depth,r.complete,r.stop_reason,r.nodes,r.proof_nodes,r.seconds,r.pv,search.tt_hits,search.table_entries()))
        },
        Some("inspect") if v.len()==4 || v.len()==8 || v.len()==9 => {
            let radius=number(v[1])?;let weight:f64=number(v[2])?;
            if !(3..=4).contains(&radius) || !weight.is_finite() {return Err("Invalid pressure settings".into());}
            let weights=if v.len()>=8 {Weights {variable_material_enabled:if v.len()==9 {flag(v[7])?} else {false},material:number(v[3])?,advantage:number(v[4])?,attack:number(v[5])?,defence:number(v[6])?}} else {Weights::default()};
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
