//! Incremental legality masks, independent of the evaluation accumulator.
use super::{owner, DIR};
const FULL: u128 = (1 << 81) - 1;
const fn edges() -> [u128; 8] {
    let mut masks = [0; 8];
    let mut d = 0;
    while d < 8 {
        let mut s = 0;
        while s < 81 {
            let x = s % 9 + DIR[d].0;
            let y = s / 9 + DIR[d].1;
            if x >= 0 && x < 9 && y >= 0 && y < 9 {
                masks[d] |= 1 << s;
            }
            s += 1;
        }
        d += 1;
    }
    masks
}
const EDGES: [u128; 8] = edges();
const fn neighbours() -> [u128; 81] {
    let mut masks = [0; 81];
    let mut square = 0;
    while square < 81 {
        let mut other = 0usize;
        while other < 81 {
            let dx = square as i32 % 9 - other as i32 % 9;
            let dy = square as i32 / 9 - other as i32 / 9;
            if dx >= -1 && dx <= 1 && dy >= -1 && dy <= 1 {
                masks[square] |= 1 << other;
            }
            other += 1;
        }
        square += 1;
    }
    masks
}
const NEIGHBOURS: [u128; 81] = neighbours();
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Moves([[u128; 3]; 2]);
impl Moves {
    pub(super) fn new(board: &[i8; 81]) -> Self {
        let mut masks = Self([[0; 3]; 2]);
        for (s, &c) in board.iter().enumerate() {
            if c != 0 {
                masks.0[owner(c) as usize][c.unsigned_abs() as usize - 1] |= 1 << s;
            }
        }
        masks
    }
    // XOR is its own inverse, including captures.
    pub(super) fn toggle(&mut self, mover: i8, from: usize, to: usize, captured: i8) {
        self.0[owner(mover) as usize][mover.unsigned_abs() as usize - 1] ^= (1 << from) | (1 << to);
        if captured != 0 {
            self.0[owner(captured) as usize][captured.unsigned_abs() as usize - 1] ^= 1 << to;
        }
    }
    pub(super) fn exposed(&self, square: usize, code: i8, removed: Option<usize>) -> bool {
        let predators = self.0[1 - owner(code) as usize][(code.unsigned_abs() as usize + 1) % 3];
        let removed = removed.map_or(0, |s| 1_u128 << s);
        predators & NEIGHBOURS[square] & !removed != 0
    }
    fn empty(&self) -> u128 {
        FULL ^ self.0.iter().flatten().fold(0, |a, b| a | b)
    }
    fn sources(&self, side: usize, dir: usize, empty: u128) -> u128 {
        let shift = DIR[dir].0 + 9 * DIR[dir].1;
        let mut sources = 0;
        for kind in 0..3 {
            let targets = empty | self.0[1 - side][(kind + 1) % 3];
            // Shift permitted destinations back to sources: equivalent to
            // shifting pieces forward, but directly retains action IDs.
            let reachable = if shift > 0 {
                targets >> shift
            } else {
                targets << -shift
            };
            sources |= self.0[side][kind] & reachable & EDGES[dir];
        }
        sources
    }
    pub(super) fn has_move(&self, side: u8) -> bool {
        let empty = self.empty();
        (0..8).any(|d| self.sources(side as usize, d, empty) != 0)
    }
    pub(super) fn legal(&self, side: u8) -> Vec<u16> {
        let empty = self.empty();
        let sources = std::array::from_fn::<_, 8, _>(|d| self.sources(side as usize, d, empty));
        let mut occupied = sources.iter().fold(0, |a, b| a | b);
        let mut actions = Vec::with_capacity(80);
        while occupied != 0 {
            let s = occupied.trailing_zeros();
            for (d, mask) in sources.iter().enumerate() {
                if mask & (1 << s) != 0 {
                    actions.push((8 * s) as u16 + d as u16);
                }
            }
            occupied &= occupied - 1;
        }
        actions
    }
}
