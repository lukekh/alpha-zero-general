//! Experimental exact-depth label search. No dependencies, selective pruning,
//! unsafe code, neural inference, or changes to the live training pipeline.
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
mod pressure_delta;
mod routes;
use pressure_delta::Pressure;

pub const MATE: f64 = 100_000.0;
const MAX_PLY: u64 = (1_u64 << 35) - 1;
const DIR: [(i32, i32); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];
type Frame = [i8; 82];

pub fn destination(action: u16) -> Option<usize> {
    if action >= 648 {
        return None;
    }
    let square = action as usize / 8;
    let (dx, dy) = DIR[action as usize % 8];
    let (x, y) = ((square % 9) as i32 + dx, (square / 9) as i32 + dy);
    if (0..9).contains(&x) && (0..9).contains(&y) {
        Some((9 * y + x) as usize)
    } else {
        None
    }
}
fn owner(code: i8) -> u8 {
    u8::from(code < 0)
}
fn distance(a: usize, b: usize) -> usize {
    (a % 9).abs_diff(b % 9).max((a / 9).abs_diff(b / 9))
}
fn captures(a: i8, b: i8) -> bool {
    a * b < 0 && a.abs() % 3 + 1 == b.abs()
}

pub const BASE: f64 = 100.0;
pub const REG: f64 = 0.25;

/// Per-piece values in ROCK, SCISSORS, PAPER order, using the own army total.
pub fn variable_piece_values(own: [usize; 3], enemy: [usize; 3]) -> [f64; 3] {
    let target = own.iter().sum::<usize>() as f64 / 3.0;
    [0, 1, 2].map(|kind| {
        BASE * (enemy[(kind + 1) % 3] as f64 + REG)
            / (enemy[(kind + 2) % 3] as f64 + REG)
            * ((target + REG) / (own[kind] as f64 + REG)).sqrt()
    })
}

/// Counts and type contributions change only on captures. The masks also track
/// quiet moves so evaluation retains Python's first-occurrence summation order.
#[derive(Clone, Debug, PartialEq)]
struct Material {
    counts: [[usize; 3]; 2],
    squares: [[u128; 3]; 2],
    contributions: [[f64; 3]; 2],
}
impl Material {
    fn new(board: &[i8; 81]) -> Self {
        let mut value = Self {
            counts: [[0; 3]; 2],
            squares: [[0; 3]; 2],
            contributions: [[0.0; 3]; 2],
        };
        for (square, &code) in board.iter().enumerate() {
            if code != 0 {
                let side = owner(code) as usize;
                let kind = code.unsigned_abs() as usize - 1;
                value.counts[side][kind] += 1;
                value.squares[side][kind] |= 1 << square;
            }
        }
        value.refresh();
        value
    }

    fn refresh(&mut self) {
        for side in 0..2 {
            let enemy = self.counts[1 - side];
            for kind in 1..=3 {
                let predator = enemy[(kind + 1) % 3];
                let prey = enemy[kind % 3];
                self.contributions[side][kind - 1] = self.counts[side][kind - 1] as f64
                    * ((if predator == 0 { 1.0 } else { 0.0 })
                        + 0.5 / (1 + predator) as f64
                        + 0.25 * prey as f64 / (1 + prey) as f64);
            }
        }
    }

    fn move_piece(&mut self, code: i8, source: usize, target: usize) {
        self.squares[owner(code) as usize][code.unsigned_abs() as usize - 1] ^=
            (1 << source) | (1 << target);
    }

    fn capture(&mut self, code: i8, square: usize, restore: bool) {
        let side = owner(code) as usize;
        let kind = code.unsigned_abs() as usize - 1;
        self.squares[side][kind] ^= 1 << square;
        if restore {
            self.counts[side][kind] += 1;
        } else {
            self.counts[side][kind] -= 1;
        }
        self.refresh();
    }

    fn advantage(&self, side: usize) -> f64 {
        let mut ordered = [0, 1, 2].map(|kind| {
            (
                self.squares[side][kind].trailing_zeros(),
                self.contributions[side][kind],
            )
        });
        // Three-element sorting network. Missing types sort last and add zero.
        for (a, b) in [(0, 1), (1, 2), (0, 1)] {
            if ordered[a].0 > ordered[b].0 {
                ordered.swap(a, b);
            }
        }
        ((0.0 + ordered[0].1) + ordered[1].1) + ordered[2].1
    }

    fn score(&self, side: usize, weights: &Weights) -> f64 {
        let material = |player: usize| {
            let own = self.counts[player];
            if weights.variable_material_enabled {
                let values = variable_piece_values(own, self.counts[1 - player]);
                ((own[0] as f64 * values[0] + own[1] as f64 * values[1])
                    + own[2] as f64 * values[2]) / BASE
            } else {
                own.iter().sum::<usize>() as f64
            }
        };
        let mut total = weights.material * (material(side) - material(1 - side));
        total += weights.advantage * (self.advantage(side) - self.advantage(1 - side));
        total
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Position {
    board: [i8; 81],
    material: Material,
    pressure: Option<Box<Pressure>>,
    pub side: u8,
    pub a1: u8,
    pub total: u64,
    history: Vec<Frame>,
    start: usize,
}
#[derive(Clone, Copy)]
struct Undo {
    source: usize,
    target: usize,
    captured: i8,
    start: usize,
}

impl Position {
    pub fn from_bytes(raw: &[u8]) -> Result<Self, String> {
        if raw.len() != 6804 {
            return Err("Expected 6804 state-v2 bytes".into());
        }
        let mut meta = [0u8; 162];
        for i in 0..162 {
            meta[i] = raw[(i / 2) * 84 + 82 + i % 2];
        }
        let len = meta[4] as usize;
        if meta[0] != 2
            || meta[1] > 1
            || meta[2] > 1
            || !(1..=81).contains(&len)
            || meta[3] as usize != len - 1
            || meta[5..10].iter().any(|x| *x > 127)
            || meta[91..].iter().any(|x| *x != 0)
        {
            return Err("Invalid state metadata".into());
        }
        let total = (0..5).map(|i| (meta[5 + i] as u64) << (7 * i)).sum();
        if total < (len - 1) as u64 {
            return Err("Invalid total ply".into());
        }
        let mut board = [0i8; 81];
        for s in 0..81 {
            board[s] = raw[s * 84] as i8;
        }
        if board.iter().any(|c| !(-3..=3).contains(c)) {
            return Err("Invalid piece".into());
        }
        let mut history: Vec<Frame> = Vec::with_capacity(160);
        for i in 0..81 {
            let mut frame = [0i8; 82];
            for s in 0..81 {
                frame[s] = raw[s * 84 + i + 1] as i8;
            }
            frame[81] = meta[10 + i] as i8;
            if i < len {
                if frame[..81].iter().any(|c| !(-3..=3).contains(c))
                    || !(0..=1).contains(&frame[81])
                    || (i > 0 && frame[81] == history[i - 1][81])
                {
                    return Err("Invalid history".into());
                }
                history.push(frame);
            } else if frame.iter().any(|x| *x != 0) {
                return Err("Nonzero unused history".into());
            }
        }
        if history[len - 1][..81] != board || history[len - 1][81] != meta[1] as i8 {
            return Err("Latest history differs from board/turn".into());
        }
        let p = Self {
            material: Material::new(&board),
            pressure: None,
            board,
            side: meta[1],
            a1: meta[2],
            total,
            history,
            start: 0,
        };
        if p.board[0] != 0
            && owner(p.board[0]) != p.a1
            && p.board[80] != 0
            && owner(p.board[80]) == p.a1
        {
            return Err("Both corners won".into());
        }
        Ok(p)
    }
    pub fn board(&self) -> &[i8; 81] {
        &self.board
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut raw = vec![0; 6804];
        for s in 0..81 {
            raw[s * 84] = self.board[s] as u8;
        }
        let history = &self.history[self.start..];
        let mut meta = [0u8; 162];
        meta[..5].copy_from_slice(&[
            2,
            self.side,
            self.a1,
            (history.len() - 1) as u8,
            history.len() as u8,
        ]);
        for i in 0..5 {
            meta[5 + i] = ((self.total >> (7 * i)) & 127) as u8;
        }
        for (i, frame) in history.iter().enumerate() {
            for s in 0..81 {
                raw[s * 84 + i + 1] = frame[s] as u8;
            }
            meta[10 + i] = frame[81] as u8;
        }
        for i in 0..162 {
            raw[(i / 2) * 84 + 82 + i % 2] = meta[i];
        }
        raw
    }
    pub fn raw_legal(&self, side: u8) -> Vec<u16> {
        let mut actions = Vec::with_capacity(80);
        for (square, &piece) in self.board.iter().enumerate() {
            if piece == 0 || owner(piece) != side {
                continue;
            }
            for dir in 0..8 {
                let action = (8 * square + dir) as u16;
                if let Some(to) = destination(action) {
                    if self.board[to] == 0 || captures(piece, self.board[to]) {
                        actions.push(action);
                    }
                }
            }
        }
        actions
    }
    fn has_move(&self, side: u8) -> bool {
        for (square, &piece) in self.board.iter().enumerate() {
            if piece == 0 || owner(piece) != side {
                continue;
            }
            for dir in 0..8 {
                if let Some(to) = destination((8 * square + dir) as u16) {
                    if self.board[to] == 0 || captures(piece, self.board[to]) {
                        return true;
                    }
                }
            }
        }
        false
    }
    /// Official victories take precedence over modelling draws.
    pub fn terminal(&self, modelling: bool) -> Option<(Option<u8>, &'static str)> {
        if self.board[0] != 0 && owner(self.board[0]) != self.a1 {
            return Some((Some(1 - self.a1), "corner"));
        }
        if self.board[80] != 0 && owner(self.board[80]) == self.a1 {
            return Some((Some(self.a1), "corner"));
        }
        if !self.has_move(self.side) {
            return Some((Some(1 - self.side), "stalemate"));
        }
        if modelling {
            let history = &self.history[self.start..];
            if history
                .iter()
                .filter(|f| *f == history.last().unwrap())
                .count()
                >= 3
            {
                return Some((None, "repetition"));
            }
            if history.len() >= 81 {
                return Some((None, "no-capture limit"));
            }
        }
        None
    }
    fn terminal_score(&self, ply: usize) -> Option<f64> {
        self.terminal(true).map(|(winner, _)| match winner {
            Some(w) if w == self.side => MATE - ply as f64,
            Some(_) => -MATE + ply as f64,
            None => 0.0,
        })
    }
    fn push(&mut self, action: u16) -> Undo {
        if let Some(pressure) = &mut self.pressure {
            let source = action as usize / 8;
            let target = destination(action).expect("generated legal move");
            pressure.remove(&self.board, source);
            if self.board[target] != 0 {
                pressure.remove(&self.board, target);
            }
        }
        let undo = self.push_board(action);
        if let Some(pressure) = &mut self.pressure {
            pressure.add(&self.board, undo.target);
        }
        self.material
            .move_piece(self.board[undo.target], undo.source, undo.target);
        if undo.captured != 0 {
            self.material.capture(undo.captured, undo.target, false);
        }
        undo
    }

    // Proof search and move ordering use only board/history, never material.
    // Keep the caller's accumulator untouched throughout those temporary moves.
    fn push_board(&mut self, action: u16) -> Undo {
        let source = action as usize / 8;
        let target = destination(action).expect("generated legal move");
        let undo = Undo {
            source,
            target,
            captured: self.board[target],
            start: self.start,
        };
        self.board[target] = self.board[source];
        self.board[source] = 0;
        self.side = 1 - self.side;
        self.total += 1;
        if undo.captured != 0 {
            self.start = self.history.len();
        } else if self.history.len() - self.start >= 81 {
            self.start += 1;
        }
        let mut frame = [0i8; 82];
        frame[..81].copy_from_slice(&self.board);
        frame[81] = self.side as i8;
        self.history.push(frame);
        undo
    }
    fn pop(&mut self, u: Undo) {
        if let Some(pressure) = &mut self.pressure {
            pressure.remove(&self.board, u.target);
        }
        self.material
            .move_piece(self.board[u.target], u.target, u.source);
        if u.captured != 0 {
            self.material.capture(u.captured, u.target, true);
        }
        self.pop_board(u);
        if let Some(pressure) = &mut self.pressure {
            if u.captured != 0 {
                pressure.add(&self.board, u.target);
            }
            pressure.add(&self.board, u.source);
        }
    }

    fn pop_board(&mut self, u: Undo) {
        self.board[u.source] = self.board[u.target];
        self.board[u.target] = u.captured;
        self.side = 1 - self.side;
        self.total -= 1;
        self.history.pop();
        self.start = u.start;
    }
    pub fn apply(&mut self, action: u16, modelling: bool) -> Result<(), String> {
        if self.total == MAX_PLY
            || self.terminal(modelling).is_some()
            || !self.raw_legal(self.side).contains(&action)
        {
            return Err("Illegal move or terminal/overflow state".into());
        }
        self.push(action);
        Ok(())
    }
    // Full active history is deliberately more conservative than merging
    // histories with equal occurrence counts. HashMap also checks exact equality.
    fn key(&self, depth: usize) -> Vec<u8> {
        let mut key = Vec::with_capacity(4 + 82 * (self.history.len() - self.start));
        key.extend([
            depth as u8,
            self.side,
            self.a1,
            (self.history.len() - self.start) as u8,
        ]);
        for frame in &self.history[self.start..] {
            key.extend(frame.iter().map(|c| *c as u8));
        }
        key
    }
}

pub fn pressure(board: &[i8; 81], radius: usize) -> [f64; 2] {
    assert!(radius == 3 || radius == 4);
    let mut squares = [0usize; 81];
    let mut count = 0;
    for (s, c) in board.iter().enumerate() {
        if *c != 0 {
            squares[count] = s;
            count += 1;
        }
    }
    let mut defenders = [[0usize; 5]; 81];
    for i in 0..count {
        let a = board[squares[i]];
        for j in 0..count {
            let b = board[squares[j]];
            if a * b > 0 && b.abs() == a.abs() % 3 + 1 {
                let d = distance(squares[i], squares[j]);
                if (1..=radius).contains(&d) {
                    defenders[i][d] += 1;
                }
            }
        }
        for r in 2..=radius {
            defenders[i][r] += defenders[i][r - 1];
        }
    }
    let mut totals = [0.0; 2];
    let weights = [0.0, 2.0, 1.0, 0.5, 0.25];
    for i in 0..count {
        for j in i + 1..count {
            let (a, b) = (board[squares[i]], board[squares[j]]);
            let d = distance(squares[i], squares[j]);
            if a * b >= 0 || a.abs() == b.abs() || d > radius {
                continue;
            }
            let (attacker, victim) = if captures(a, b) { (i, j) } else { (j, i) };
            // Exact binary powers, without an expensive general pow call.
            let discount = f64::from_bits(((1023 - defenders[victim][d]) as u64) << 52);
            totals[owner(board[squares[attacker]]) as usize] += weights[d] * discount;
        }
    }
    totals
}

pub fn evaluate(p: &Position, radius: usize, weight: f64) -> f64 {
    evaluate_weighted(p, radius, weight, &Weights::default())
}

pub fn evaluate_weighted(p: &Position, radius: usize, weight: f64, weights: &Weights) -> f64 {
    let side = p.side as usize;
    let mut total = p.material.score(side, weights);
    if weights.attack != 0.0 || weights.defence != 0.0 {
        let (attack, defence) = routes::terms(p, weights.attack != 0.0, weights.defence != 0.0);
        total += weights.attack * attack;
        total += weights.defence * defence;
    }
    if weight != 0.0 {
        let ring = match &p.pressure {
            Some(cached) if cached.radius == radius => cached.totals,
            _ => pressure(&p.board, radius),
        };
        total += weight * (ring[side] - ring[1 - side]);
    }
    total.clamp(-10000.0, 10000.0)
}

/// Adopted endgame defaults; all supported coefficients may also be signed.
#[derive(Clone, Debug, PartialEq)]
pub struct Weights {
    pub variable_material_enabled: bool,
    pub material: f64,
    pub advantage: f64,
    pub attack: f64,
    pub defence: f64,
}
impl Default for Weights {
    fn default() -> Self {
        Self {
            material: 100.0,
            variable_material_enabled: false,
            advantage: 23.967050360966205,
            attack: 25.714516982666414,
            defence: 32.5643023919054,
        }
    }
}
impl Weights {
    pub fn validate(&self) -> Result<(), String> {
        if [self.material, self.advantage, self.attack, self.defence]
            .iter()
            .any(|x| !x.is_finite())
        {
            return Err("Weights must be finite".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub weights: Weights,
    pub depth: usize,
    pub milliseconds: u64,
    pub node_limit: u64,
    pub radius: usize,
    pub pressure_weight: f64,
    pub proof_depth: usize,
    pub proof_nodes: u64,
    pub table_entries: usize,
}
impl Config {
    pub fn validate(&self) -> Result<(), String> {
        self.weights.validate()?;
        if !(1..=32).contains(&self.depth)
            || !(3..=4).contains(&self.radius)
            || !self.pressure_weight.is_finite()
            || self.proof_depth > 2
            || self.proof_nodes > 64
            || self.table_entries > 1000000
        {
            return Err("Unsupported configuration: depth 1..32, radius 3/4, proof depth <=2 and nodes <=64".into());
        }
        Ok(())
    }
}
#[derive(Clone)]
struct Entry {
    score: f64,
    bound: i8,
    pv: Vec<u16>,
}
#[derive(Debug)]
pub struct SearchResult {
    pub action: Option<u16>,
    pub score: Option<f64>,
    pub completed_depth: usize,
    pub target_depth: usize,
    pub complete: bool,
    pub stop_reason: &'static str,
    pub nodes: u64,
    pub proof_nodes: u64,
    pub seconds: f64,
    pub pv: Vec<u16>,
}
pub struct Search {
    config: Config,
    start: Instant,
    nodes: u64,
    proof_nodes: u64,
    table: HashMap<Vec<u8>, Entry>,
    fifo: VecDeque<Vec<u8>>,
    killers: [[Option<u16>; 2]; 64],
    history: [[u32; 648]; 2],
    previous: HashMap<u16, f64>,
    root_scores: HashMap<u16, f64>,
    pub tt_hits: u64,
}
type Line = (f64, Vec<u16>);
impl Search {
    pub fn new(config: Config) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            start: Instant::now(),
            nodes: 0,
            proof_nodes: 0,
            table: HashMap::new(),
            fifo: VecDeque::new(),
            killers: [[None; 2]; 64],
            history: [[0; 648]; 2],
            previous: HashMap::new(),
            root_scores: HashMap::new(),
            tt_hits: 0,
        })
    }
    fn visit(&mut self, proof: bool) -> Result<(), &'static str> {
        if self.start.elapsed() >= Duration::from_millis(self.config.milliseconds) {
            return Err("time");
        }
        if self.nodes >= self.config.node_limit {
            return Err("nodes");
        }
        self.nodes += 1;
        if proof {
            self.proof_nodes += 1;
        }
        Ok(())
    }
    fn proof_safe(p: &Position, depth: usize) -> bool {
        let mut counts = [0; 2];
        for (s, &c) in p.board.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let side = owner(c);
            counts[side as usize] += 1;
            let corner = if side == p.a1 { 80 } else { 0 };
            if distance(s, corner) <= (depth + usize::from(side == p.side)) / 2 {
                return false;
            }
        }
        let required = 2 * depth + 1;
        if counts.iter().any(|c| *c < required) {
            return false;
        }
        for side in 0..2 {
            let mut used = [false; 81];
            let mut pairs = 0;
            for (s, &c) in p.board.iter().enumerate() {
                if c == 0 || owner(c) != side {
                    continue;
                }
                for d in 0..8 {
                    if let Some(to) = destination((8 * s + d) as u16) {
                        if p.board[to] == 0 && !used[to] {
                            used[to] = true;
                            pairs += 1;
                            break;
                        }
                    }
                }
            }
            if pairs < required {
                return false;
            }
        }
        true
    }
    fn proof(
        &mut self,
        p: &mut Position,
        depth: usize,
        ply: usize,
        mut alpha: f64,
        beta: f64,
        used: &mut u64,
    ) -> Result<Line, &'static str> {
        if *used >= self.config.proof_nodes {
            return Err("proof budget");
        }
        self.visit(true)?;
        *used += 1;
        if let Some(score) = p.terminal_score(ply) {
            return Ok((score, vec![]));
        }
        if depth == 0 || (ply == 0 && Self::proof_safe(p, depth)) {
            return Ok((0.0, vec![]));
        }
        if p.total == MAX_PLY {
            return Err("ply overflow");
        }
        let goal = if p.side == p.a1 { 80 } else { 0 };
        let mut actions = p.raw_legal(p.side);
        actions.sort_by_key(|a| (destination(*a) != Some(goal), *a));
        let mut best = (-f64::INFINITY, vec![]);
        for action in actions {
            let undo = p.push_board(action);
            let result = self.proof(p, depth - 1, ply + 1, -beta, -alpha, used);
            p.pop_board(undo);
            let (value, line) = result?;
            let value = -value;
            if value > best.0 {
                best = (value, std::iter::once(action).chain(line).collect());
            }
            alpha = alpha.max(value);
            if alpha >= beta || best.0 == MATE - ply as f64 - 1.0 {
                break;
            }
        }
        Ok(best)
    }
    fn exposed(p: &Position, square: usize, code: i8, removed: Option<usize>) -> bool {
        p.board
            .iter()
            .enumerate()
            .any(|(s, &c)| Some(s) != removed && distance(s, square) <= 1 && captures(c, code))
    }
    fn ordered(&self, p: &mut Position, preferred: Option<u16>, ply: usize) -> Vec<u16> {
        let side = p.side;
        let goal = if side == p.a1 { 80 } else { 0 };
        let own_goal = 80 - goal;
        let actions = p.raw_legal(side);
        let mut ranked = Vec::with_capacity(actions.len());
        for a in actions {
            let from = a as usize / 8;
            let to = destination(a).unwrap();
            let mover = p.board[from];
            let occupant = p.board[to];
            let threat = occupant != 0 && owner(occupant) != side && distance(to, own_goal) <= 1;
            let unsafe_move = Self::exposed(p, to, mover, Some(to));
            let escape = Self::exposed(p, from, mover, None) && !unsafe_move;
            // Only board-based legality is queried until these edits are undone.
            p.board[from] = 0;
            p.board[to] = mover;
            let win = to == goal || !p.has_move(1 - side);
            p.board[from] = mover;
            p.board[to] = occupant;
            let rank = [
                f64::from(win),
                f64::from(preferred == Some(a)),
                if ply == 0 {
                    *self.previous.get(&a).unwrap_or(&-f64::INFINITY)
                } else {
                    -f64::INFINITY
                },
                f64::from(threat || to == own_goal),
                f64::from(occupant != 0 && !unsafe_move),
                f64::from(escape),
                f64::from(occupant != 0),
                if self.killers[ply][0] == Some(a) {
                    2.0
                } else if self.killers[ply][1] == Some(a) {
                    1.0
                } else {
                    0.0
                },
                self.history[side as usize][a as usize] as f64,
                -(distance(to, goal) as f64),
                -(a as f64),
            ];
            ranked.push((a, rank));
        }
        ranked.sort_by(|a, b| {
            for i in 0..11 {
                if a.1[i] != b.1[i] {
                    return b.1[i].partial_cmp(&a.1[i]).unwrap();
                }
            }
            std::cmp::Ordering::Equal
        });
        ranked.into_iter().map(|x| x.0).collect()
    }
    fn store(&mut self, key: Vec<u8>, score: f64, pv: Vec<u16>, bound: i8, ply: usize) {
        if self.config.table_entries == 0 {
            return;
        }
        let score = if score > 90000.0 {
            score + ply as f64
        } else if score < -90000.0 {
            score - ply as f64
        } else {
            score
        };
        if !self.table.contains_key(&key) {
            if self.table.len() >= self.config.table_entries {
                if let Some(old) = self.fifo.pop_front() {
                    self.table.remove(&old);
                }
            }
            self.fifo.push_back(key.clone());
        }
        self.table.insert(key, Entry { score, bound, pv });
    }
    fn search(
        &mut self,
        p: &mut Position,
        depth: usize,
        mut alpha: f64,
        mut beta: f64,
        ply: usize,
    ) -> Result<Line, &'static str> {
        self.visit(false)?;
        if let Some(score) = p.terminal_score(ply) {
            return Ok((score, vec![]));
        }
        if depth == 0 {
            if self.config.proof_depth > 0 && self.config.proof_nodes > 0 {
                match self.proof(
                    p,
                    self.config.proof_depth,
                    0,
                    -f64::INFINITY,
                    f64::INFINITY,
                    &mut 0,
                ) {
                    Ok((score, line)) if score.abs() > 90000.0 => {
                        return Ok((
                            if score > 0.0 {
                                score - ply as f64
                            } else {
                                score + ply as f64
                            },
                            line,
                        ));
                    }
                    Err(e) if e != "proof budget" => return Err(e),
                    _ => (),
                }
            }
            return Ok((
                evaluate_weighted(
                    p,
                    self.config.radius,
                    self.config.pressure_weight,
                    &self.config.weights,
                ),
                vec![],
            ));
        }
        if p.total == MAX_PLY {
            return Err("ply overflow");
        }
        let original = (alpha, beta);
        let key = p.key(depth);
        let mut preferred = None;
        if let Some(entry) = self.table.get(&key) {
            self.tt_hits += 1;
            let score = if entry.score > 90000.0 {
                entry.score - ply as f64
            } else if entry.score < -90000.0 {
                entry.score + ply as f64
            } else {
                entry.score
            };
            preferred = entry.pv.first().copied();
            if entry.bound == 0 {
                return Ok((score, entry.pv.clone()));
            }
            if entry.bound == 1 {
                alpha = alpha.max(score);
            } else {
                beta = beta.min(score);
            }
            if alpha >= beta {
                return Ok((score, entry.pv.clone()));
            }
        }
        if preferred.is_none() {
            for d in (1..depth).rev() {
                if let Some(entry) = self.table.get(&p.key(d)) {
                    preferred = entry.pv.first().copied();
                    break;
                }
            }
        }
        let mut best = (-f64::INFINITY, vec![]);
        let side = p.side;
        for (index, action) in self.ordered(p, preferred, ply).into_iter().enumerate() {
            let capture = p.board[destination(action).unwrap()] != 0;
            let undo = p.push(action);
            let probe = next_up(alpha);
            let child = if index > 0 && alpha.is_finite() && probe < beta {
                match self.search(p, depth - 1, -probe, -alpha, ply + 1) {
                    Ok((v, _)) if alpha < -v && -v < beta => {
                        self.search(p, depth - 1, -beta, -alpha, ply + 1)
                    }
                    other => other,
                }
            } else {
                self.search(p, depth - 1, -beta, -alpha, ply + 1)
            };
            p.pop(undo);
            let (value, line) = child?;
            let value = -value;
            if ply == 0 {
                self.root_scores.insert(action, value);
            }
            if value > best.0 {
                best = (value, std::iter::once(action).chain(line).collect());
            }
            alpha = alpha.max(value);
            if alpha >= beta {
                if !capture {
                    if self.killers[ply][0] != Some(action) {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = Some(action);
                    }
                    let h = &mut self.history[side as usize][action as usize];
                    *h = (*h + (depth * depth) as u32).min(32767);
                }
                break;
            }
        }
        let bound = if best.0 <= original.0 {
            -1
        } else if best.0 >= original.1 {
            1
        } else {
            0
        };
        self.store(key, best.0, best.1.clone(), bound, ply);
        Ok(best)
    }
    /// Only a fully completed iteration (or proof) can become a training label.
    pub fn analyze(&mut self, position: &Position) -> SearchResult {
        self.analyze_impl(position, false)
    }
    pub fn analyze_reusing(&mut self, position: &Position) -> SearchResult {
        self.analyze_impl(position, true)
    }
    pub fn table_entries(&self) -> usize {
        self.table.len()
    }
    fn analyze_impl(&mut self, position: &Position, reuse: bool) -> SearchResult {
        self.start = Instant::now();
        self.nodes = 0;
        self.proof_nodes = 0;
        self.tt_hits = 0;
        if !reuse {
            self.table.clear();
            self.fifo.clear();
        }
        self.previous.clear();
        self.root_scores.clear();
        self.killers = [[None; 2]; 64];
        self.history = [[0; 648]; 2];
        let mut result = SearchResult {
            action: None,
            score: None,
            completed_depth: 0,
            target_depth: self.config.depth,
            complete: false,
            stop_reason: "maximum_depth",
            nodes: 0,
            proof_nodes: 0,
            seconds: 0.0,
            pv: vec![],
        };
        let mut p = position.clone();
        if p.terminal(true).is_some() {
            result.stop_reason = "terminal";
            return result;
        }
        p.pressure = if self.config.pressure_weight != 0.0 {
            Pressure::new(&p.board, self.config.radius)
        } else {
            None
        };
        #[cfg(debug_assertions)]
        let root = p.clone();
        for depth in 1..=self.config.depth {
            self.root_scores.clear();
            match self.search(&mut p, depth, -f64::INFINITY, f64::INFINITY, 0) {
                Ok((score, line)) => {
                    result.action = line.first().copied();
                    result.score = Some(score);
                    result.pv = line;
                    result.completed_depth = depth;
                    self.previous = std::mem::take(&mut self.root_scores);
                    if score.abs() > 90000.0 {
                        result.complete = true;
                        result.stop_reason = "proven_result";
                        break;
                    }
                    if depth == self.config.depth {
                        result.complete = true;
                    }
                }
                Err(reason) => {
                    result.stop_reason = reason;
                    break;
                }
            }
        }
        #[cfg(debug_assertions)]
        assert_eq!(&p, &root, "search must unwind on every exit");
        result.nodes = self.nodes;
        result.proof_nodes = self.proof_nodes;
        result.seconds = self.start.elapsed().as_secs_f64();
        result
    }
}
fn next_up(x: f64) -> f64 {
    if x == f64::INFINITY {
        x
    } else if x == 0.0 {
        f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(pieces: &[(usize, i8)]) -> Position {
        let mut board = [0; 81];
        for &(s, c) in pieces {
            board[s] = c;
        }
        let mut frame = [0; 82];
        frame[..81].copy_from_slice(&board);
        Position {
            material: Material::new(&board),
            pressure: None,
            board,
            side: 0,
            a1: 0,
            total: 0,
            history: vec![frame],
            start: 0,
        }
    }
    fn reference_evaluate(p: &Position, radius: usize, weight: f64) -> f64 {
        let mut counts = [[0usize; 3]; 2];
        for &c in &p.board {
            if c != 0 {
                counts[owner(c) as usize][c.unsigned_abs() as usize - 1] += 1;
            }
        }
        let mut seen = [[false; 3]; 2];
        let mut advantages = [0.0; 2];
        // Match Python's first-occurrence piece-type summation order, including
        // floating point grouping; do not fast-math/reassociate these operations.
        for &c in &p.board {
            if c == 0 {
                continue;
            }
            let (side, kind) = (owner(c) as usize, c.unsigned_abs() as usize);
            if seen[side][kind - 1] {
                continue;
            }
            seen[side][kind - 1] = true;
            let enemy = counts[1 - side];
            let predator = enemy[(kind + 1) % 3];
            let prey = enemy[kind % 3];
            advantages[side] += counts[side][kind - 1] as f64
                * ((if predator == 0 { 1.0 } else { 0.0 })
                    + 0.5 / (1 + predator) as f64
                    + 0.25 * prey as f64 / (1 + prey) as f64);
        }
        let side = p.side as usize;
        let mut total = 100.0
            * (counts[side].iter().sum::<usize>() as f64
                - counts[1 - side].iter().sum::<usize>() as f64);
        total += Weights::default().advantage * (advantages[side] - advantages[1 - side]);
        if weight != 0.0 {
            let ring = pressure(&p.board, radius);
            total += weight * (ring[side] - ring[1 - side]);
        }
        total.clamp(-10000.0, 10000.0)
    }

    #[test]
    fn roundtrip_and_undo() {
        let mut p = fixture(&[(40, 1), (50, -2), (70, -3)]);
        let old = p.clone();
        assert_eq!(Position::from_bytes(&p.to_bytes()).unwrap(), p);
        for a in p.raw_legal(0) {
            let undo = p.push(a);
            p.pop(undo);
            assert_eq!(p, old);
        }
    }

    fn assert_material(p: &Position) {
        assert_eq!(p.material, Material::new(&p.board));
        if let Some(cached) = &p.pressure {
            assert_eq!(cached.totals, pressure(&p.board, cached.radius));
            assert_eq!(p.pressure, Pressure::new(&p.board, cached.radius));
        }
        for side in 0..2 {
            let mut view = p.clone();
            view.side = side;
            for radius in [3, 4] {
                for weight in [0.0, 10.0, 1000000.0] {
                    assert_eq!(
                        evaluate_weighted(
                            &view,
                            radius,
                            weight,
                            &Weights {
                                attack: 0.0,
                                defence: 0.0,
                                ..Weights::default()
                            }
                        )
                        .to_bits(),
                        reference_evaluate(&view, radius, weight).to_bits(),
                        "side={side}, radius={radius}, weight={weight}, board={:?}",
                        p.board
                    );
                }
            }
        }
    }

    #[test]
    fn material_trajectories_siblings_and_full_unwind() {
        let mut random = 0x20260916u64;
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        let mut captures_seen = 0;
        let mut quiet_seen = 0;
        for game in 0..12 {
            // Independent dense/sparse boards exercise all types and both
            // halves of the 81-square masks, followed by legal trajectories.
            let pieces: Vec<_> = (1..80)
                .filter_map(|square| {
                    let value = next();
                    (value % 4 == 0)
                        .then_some((square, [1, 2, 3, -1, -2, -3][(value / 4 % 6) as usize]))
                })
                .collect();
            let mut p = fixture(&pieces);
            p.pressure = Pressure::new(&p.board, 3 + game % 2);
            let root = p.clone();
            let mut undos = vec![];
            for _ in 0..60 {
                assert_material(&p);
                if p.terminal(true).is_some() {
                    break;
                }
                let actions = p.raw_legal(p.side);
                let parent = p.clone();
                for &action in actions.iter().step_by(7) {
                    let undo = p.push(action);
                    assert_material(&p);
                    p.pop(undo);
                    assert_eq!(p, parent);
                }
                let undo = p.push(actions[next() as usize % actions.len()]);
                if undo.captured == 0 {
                    quiet_seen += 1;
                    assert_eq!(p.material.counts, parent.material.counts);
                    assert_eq!(p.material.contributions, parent.material.contributions);
                } else {
                    captures_seen += 1;
                }
                undos.push(undo);
            }
            for undo in undos.into_iter().rev() {
                assert_material(&p);
                p.pop(undo);
            }
            assert_eq!(p, root);
        }
        assert!(captures_seen > 10 && quiet_seen > 100);
    }

    #[test]
    fn material_extinction_and_quiet_type_order() {
        for sign in [1, -1] {
            for attacker in 1..=3 {
                let victim = attacker % 3 + 1;
                let mut p = fixture(&[
                    (40, sign * attacker),
                    (41, -sign * victim),
                    (70, -sign * attacker),
                ]);
                p.side = owner(sign * attacker);
                p.pressure = Pressure::new(&p.board, 3);
                let parent = p.clone();
                let undo = p.push(40 * 8 + 2);
                assert_eq!(
                    p.material.counts[1 - parent.side as usize][victim as usize - 1],
                    0
                );
                assert_material(&p);
                p.pop(undo);
                assert_eq!(p, parent);
            }
        }
        let mut p = fixture(&[
            (20, 1),
            (21, 2),
            (22, 3),
            (40, -1),
            (41, -2),
            (42, -3),
            (43, -3),
        ]);
        let parent = p.clone();
        let undo = p.push(20 * 8); // Type 1 moves behind the first type 2 and 3.
        assert_material(&p);
        assert_eq!(p.material.contributions, parent.material.contributions);
        assert!(
            p.material.squares[0][0].trailing_zeros() > p.material.squares[0][2].trailing_zeros()
        );
        p.pop(undo);
        assert_eq!(p, parent);
    }

    #[test]
    fn material_restored_after_search_cutoffs_and_cancellation() {
        let original = fixture(&[(20, 1), (21, 2), (22, 3), (40, -2), (41, -3), (60, -1)]);
        for node_limit in [1, 10, 100, 1000, 1000000] {
            let config = Config {
                weights: Weights::default(),
                depth: 3,
                milliseconds: 60000,
                node_limit,
                radius: 3,
                pressure_weight: 10.0,
                proof_depth: 2,
                proof_nodes: 64,
                table_entries: 100,
            };
            let mut search = Search::new(config.clone()).unwrap();
            let mut p = original.clone();
            let result = search.search(&mut p, 3, -1.0, 1.0, 0);
            if node_limit == 1 {
                assert_eq!(result, Err("nodes"));
            }
            assert_eq!(p, original);
            assert_material(&p);
            search.config.node_limit = 1000000;
            let resumed = search.analyze(&p);
            let mut fresh_config = config;
            fresh_config.node_limit = 1000000;
            let fresh = Search::new(fresh_config).unwrap().analyze(&original);
            assert_eq!(
                (resumed.score, resumed.action, resumed.pv),
                (fresh.score, fresh.action, fresh.pv)
            );
        }
    }

    #[test]
    fn proof_only_moves_preserve_material_on_budget_exit() {
        let mut original = fixture(&[(40, 1), (41, -2), (50, 3), (60, -1)]);
        original.pressure = Pressure::new(&original.board, 3);
        for proof_nodes in [1, 2, 10, 64] {
            let mut p = original.clone();
            let mut search = Search::new(Config {
                weights: Weights::default(),
                depth: 3,
                milliseconds: 60000,
                node_limit: 1000000,
                radius: 3,
                pressure_weight: 10.0,
                proof_depth: 2,
                proof_nodes,
                table_entries: 100,
            })
            .unwrap();
            let mut used = 0;
            let result = search.proof(&mut p, 2, 0, -f64::INFINITY, f64::INFINITY, &mut used);
            assert!(used > 0);
            if proof_nodes == 1 {
                assert_eq!(result, Err("proof budget"));
            }
            assert_eq!(p, original);
            assert_material(&p);
        }
    }
    #[test]
    fn rings_and_defenders() {
        let p = fixture(&[(40, 1), (41, -2)]);
        assert_eq!(pressure(&p.board, 3), [2.0, 0.0]);
        let p = fixture(&[(40, 1), (41, -2), (42, -3)]);
        assert_eq!(pressure(&p.board, 3), [1.0, 1.0]);
        assert_eq!(
            pressure(&fixture(&[(40, 1), (44, -2)]).board, 3),
            [0.0, 0.0]
        );
        assert_eq!(
            pressure(&fixture(&[(40, 1), (44, -2)]).board, 4),
            [0.25, 0.0]
        );
    }

    #[test]
    fn incremental_pressure_updates_stationary_victims_and_dense_fallback() {
        let mut p = fixture(&[(40, 1), (41, -2), (42, -3)]);
        p.side = 1;
        p.pressure = Pressure::new(&p.board, 3);
        let original = p.clone();
        assert_eq!(p.pressure.as_ref().unwrap().totals, [1.0, 1.0]);
        let undo = p.push(42 * 8 + 2);
        assert_eq!(p.pressure.as_ref().unwrap().totals, [2.0, 0.5]);
        assert_material(&p);
        p.pop(undo);
        assert_eq!(p, original);
        for count in [32, 33, 79] {
            let pieces: Vec<_> = (1..=count)
                .map(|square| (square, [1, -2, 3, -1, 2, -3][square % 6]))
                .collect();
            let mut dense = fixture(&pieces);
            for radius in [3, 4] {
                dense.pressure = Pressure::new(&dense.board, radius);
                assert_eq!(dense.pressure.is_some(), count <= 32);
                assert_material(&dense);
            }
        }
    }
    #[test]
    fn cancellation_never_labels_partial() {
        let p = fixture(&[(40, 1), (60, -2)]);
        let mut s = Search::new(Config {
            weights: Weights::default(),
            depth: 6,
            milliseconds: 0,
            node_limit: 100,
            radius: 3,
            pressure_weight: 10.0,
            proof_depth: 2,
            proof_nodes: 64,
            table_entries: 10,
        })
        .unwrap();
        let r = s.analyze(&p);
        assert!(!r.complete);
        assert_eq!(r.completed_depth, 0);
    }
    #[test]
    fn modelling_only_draw_and_win_precedence() {
        let mut p = fixture(&[(40, 1), (60, -2)]);
        let frame = p.history[0];
        p.history = vec![frame; 3];
        assert_eq!(p.terminal(true), Some((None, "repetition")));
        assert_eq!(p.terminal(false), None);
        p.board[80] = 1;
        p.material = Material::new(&p.board);
        assert_eq!(p.terminal(true), Some((Some(0), "corner")));
    }
    #[test]
    fn variable_material_formula_and_extinction() {
        assert_eq!(variable_piece_values([2, 2, 2], [3, 3, 3]), [100.0; 3]);
        let values = variable_piece_values([1, 4, 4], [5, 8, 2]);
        let rock = 100.0 * 8.25 / 2.25 * (3.25_f64 / 1.25).sqrt();
        assert_eq!(values[0], rock);
        let rotated = variable_piece_values([4, 4, 1], [8, 2, 5]);
        assert_eq!(rotated, [values[1], values[2], values[0]]);
        assert!(variable_piece_values([0; 3], [0; 3]).iter().all(|v| v.is_finite()));
    }

}
