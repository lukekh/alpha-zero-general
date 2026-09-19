//! Python-compatible static occupied-board route features.
//!
//! Mirrors `intransitive/heuristics/{flood,geometry,features}.py`, including the
//! four structural changes those carry:
//!
//! * Route maps are a bitboard flood fill over the 81-square `u128` the rest of
//!   the crate already uses, with passability masks built once per position
//!   rather than re-derived for every neighbour of every search.
//! * Safety is one multi-source fill per predator type. `arrival` is monotone in
//!   distance, so the minimum over predators is the arrival of the nearest one,
//!   and every piece of a code shares the map.
//! * Each runner's interception squares and deadlines are built once instead of
//!   once per defender, and its shortest-path set is computed once per position
//!   rather than allocated afresh on every call.
//! * Attack walks the eight neighbours: both of its tests require distance one.
//!   Its capture half is a boolean, since `opportunities` holds one per distinct
//!   square under `min(1, sum)`.
use super::{captures, owner, Position, DIR};

const FULL: u128 = (1 << 81) - 1;
const UNREACHED: i16 = 99;
/// Above any reachable arrival (99 moves arrive by ply 198), so a piece whose
/// predators are all captured stays safe at every deadline.
const NO_PREDATOR: i16 = i16::MAX;

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

const fn offsets() -> [i32; 8] {
    let mut shifts = [0; 8];
    let mut d = 0;
    while d < 8 {
        shifts[d] = DIR[d].0 + 9 * DIR[d].1;
        d += 1;
    }
    shifts
}
const OFFSETS: [i32; 8] = offsets();

/// Every square one king step from the given set, never wrapping a row.
const fn dilate(bits: u128) -> u128 {
    let mut out = 0;
    let mut d = 0;
    while d < 8 {
        let masked = bits & EDGES[d];
        out |= if OFFSETS[d] > 0 {
            masked << OFFSETS[d]
        } else {
            masked >> -OFFSETS[d]
        };
        d += 1;
    }
    out & FULL
}

const fn steps() -> [u128; 81] {
    let mut masks = [0; 81];
    let mut square = 0;
    while square < 81 {
        masks[square] = dilate(1 << square);
        square += 1;
    }
    masks
}
/// STEPS[s] is the eight squares adjacent to s, excluding s itself.
const STEPS: [u128; 81] = steps();

fn slot(code: i8) -> usize {
    3 * owner(code) as usize + code.unsigned_abs() as usize - 1
}

/// The single code that captures `code`: paper<-rock<-scissors<-paper.
fn predator(code: i8) -> i8 {
    let kind = if code.abs() == 1 { 3 } else { code.abs() - 1 };
    if code > 0 {
        -kind
    } else {
        kind
    }
}

fn arrival(moves: i16, side: u8, turn: u8) -> i16 {
    if moves == 0 {
        0
    } else {
        2 * moves - i16::from(side == turn)
    }
}

/// Squares each code may enter: empty, or holding the prey it captures.
fn passable_masks(board: &[i8; 81]) -> [u128; 6] {
    let mut by_slot = [0_u128; 6];
    let mut occupied = 0_u128;
    for (square, &code) in board.iter().enumerate() {
        if code != 0 {
            by_slot[slot(code)] |= 1 << square;
            occupied |= 1 << square;
        }
    }
    let empty = FULL ^ occupied;
    let mut masks = [0_u128; 6];
    for (index, mask) in masks.iter_mut().enumerate() {
        let (side, kind) = (index / 3, index % 3 + 1);
        // This code captures the opposing type one step around the cycle.
        *mask = empty | by_slot[3 * (1 - side) + kind % 3];
    }
    masks
}

/// Ring-by-ring distances from a source set. 99 = unreachable.
fn flood(passable: u128, sources: u128) -> [i16; 81] {
    let mut distance = [UNREACHED; 81];
    let mut frontier = sources;
    let mut seen = sources;
    let mut step: i16 = 0;
    while frontier != 0 {
        let mut bits = frontier;
        while bits != 0 {
            distance[bits.trailing_zeros() as usize] = step;
            bits &= bits - 1;
        }
        frontier = dilate(frontier) & passable & !seen;
        seen |= frontier;
        step += 1;
    }
    distance
}

struct Piece {
    square: usize,
    code: i8,
    side: u8,
    slot: usize,
    goal: usize,
    distances: [i16; 81],
    reverse: [i16; 81],
    goal_distance: i16,
}

/// One runner's interception squares and deadlines, shared by every defender.
struct Profile {
    squares: Vec<usize>,
    deadlines: Vec<i16>,
    block_deadline: i16,
}

struct Geometry {
    board: [i8; 81],
    pieces: Vec<Piece>,
    turn: u8,
    goals: [usize; 2],
    threat: [[i16; 81]; 6],
    profiles: Vec<Option<Profile>>,
}

impl Geometry {
    fn new(p: &Position) -> Self {
        Self::from_board(&p.board, p.side, p.a1)
    }

    fn from_board(board: &[i8; 81], turn: u8, a1: u8) -> Self {
        let goals = if a1 == 0 { [80, 0] } else { [0, 80] };
        let passable = passable_masks(board);
        let pieces: Vec<Piece> = board
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != 0)
            .map(|(square, &code)| {
                let side = owner(code);
                let goal = goals[side as usize];
                let index = slot(code);
                let distances = flood(passable[index], 1 << square);
                // The runner vacates its own square, so its reverse map may
                // pass through it.
                let reverse = flood(passable[index] | (1 << square), 1 << goal);
                let goal_distance = distances[goal];
                Piece {
                    square,
                    code,
                    side,
                    slot: index,
                    goal,
                    distances,
                    reverse,
                    goal_distance,
                }
            })
            .collect();

        // One multi-source fill per predator type present, shared by every
        // piece of the code it protects.
        let mut sources = [0_u128; 6];
        for piece in &pieces {
            sources[piece.slot] |= 1 << piece.square;
        }
        let mut threat = [[NO_PREDATOR; 81]; 6];
        for code in [1_i8, 2, 3, -1, -2, -3] {
            let hunter = predator(code);
            let index = slot(hunter);
            if sources[index] == 0 {
                continue;
            }
            let distances = flood(passable[index], sources[index]);
            let hunter_side = owner(hunter);
            let row = &mut threat[slot(code)];
            for (square, &moves) in distances.iter().enumerate() {
                row[square] = arrival(moves, hunter_side, turn);
            }
        }

        let profiles = pieces
            .iter()
            .map(|runner| {
                if runner.goal_distance == UNREACHED {
                    return None;
                }
                let total = runner.goal_distance;
                let mut squares = vec![runner.square];
                let mut deadlines = vec![arrival(1, runner.side, turn) - 1];
                for square in 0..81 {
                    if square == runner.square
                        || runner.distances[square] + runner.reverse[square] != total
                    {
                        continue;
                    }
                    let runner_ply = arrival(runner.distances[square], runner.side, turn);
                    squares.push(square);
                    // Capture may occur on the reply to arrival, except at the
                    // winning goal.
                    deadlines.push(runner_ply + if square == runner.goal { -1 } else { 1 });
                }
                let block_deadline = if runner.goal == runner.square {
                    arrival(1, runner.side, turn) - 1
                } else {
                    arrival(runner.distances[runner.goal], runner.side, turn) - 1
                };
                Some(Profile {
                    squares,
                    deadlines,
                    block_deadline,
                })
            })
            .collect();

        Self {
            board: *board,
            pieces,
            turn,
            goals,
            threat,
            profiles,
        }
    }

    fn own(&self, side: u8) -> impl Iterator<Item = &Piece> {
        self.pieces.iter().filter(move |p| p.side == side)
    }

    fn safe(&self, piece: &Piece, square: usize, ply: i16) -> bool {
        self.threat[piece.slot][square] > ply
    }

    fn intercept(&self, defender: &Piece, runner: usize) -> Option<i16> {
        let profile = self.profiles[runner].as_ref()?;
        let runner = &self.pieces[runner];
        let capture = captures(defender.code, runner.code);
        if !capture && defender.code.abs() != runner.code.abs() {
            return None;
        }
        let threat = &self.threat[defender.slot];
        if !capture {
            let limit = profile.block_deadline;
            let ply = arrival(defender.distances[runner.goal], defender.side, self.turn);
            return (ply <= limit && threat[runner.goal] > ply.max(limit)).then_some(ply);
        }
        let mut best: Option<i16> = None;
        for (index, &square) in profile.squares.iter().enumerate() {
            let limit = profile.deadlines[index];
            let ply = arrival(defender.distances[square], defender.side, self.turn);
            if ply <= limit && threat[square] > ply.max(limit) {
                best = Some(best.map_or(ply, |current| current.min(ply)));
            }
        }
        best
    }

    fn attack(&self, side: u8) -> f64 {
        let mut progress = 0.0_f64;
        let mut capture = 0.0_f64;
        let deadline = arrival(1, side, self.turn) + 1;
        for piece in self.own(side) {
            let total = piece.goal_distance;
            let mut stepped = false;
            let mut bits = STEPS[piece.square];
            while bits != 0 {
                let target = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if piece.distances[target] != 1 || !self.safe(piece, target, deadline) {
                    continue;
                }
                if total != UNREACHED && piece.reverse[target] == total - 1 {
                    stepped = true;
                }
                // Distance one into an occupied square means a capturable
                // enemy: the route map only enters squares this code may take.
                if self.board[target] != 0 {
                    capture = 1.0;
                }
            }
            if stepped {
                progress += 1.0 / (1.0 + f64::from(total));
            }
        }
        progress.min(2.0) + capture
    }

    fn defence(&self, side: u8) -> f64 {
        let mut value = 0.0_f64;
        for (index, runner) in self.pieces.iter().enumerate() {
            if runner.side == side || runner.goal_distance == UNREACHED {
                continue;
            }
            let coverage: f64 = self
                .own(side)
                .filter_map(|defender| self.intercept(defender, index))
                .map(|ply| 1.0 / (1.0 + f64::from(ply)))
                .sum();
            value += coverage.min(1.0);
        }
        let goal = self.goals[(1 - side) as usize];
        if let Some(blocker) = self.own(side).find(|p| p.square == goal) {
            if self.safe(blocker, goal, 2)
                && self
                    .own(1 - side)
                    .any(|p| p.code.abs() == blocker.code.abs())
            {
                value += 1.0;
            }
        }
        value.min(4.0)
    }
}

pub(super) fn terms(p: &Position, attack: bool, defence: bool) -> (f64, f64) {
    let g = Geometry::new(p);
    let side = p.side;
    (
        if attack {
            g.attack(side) - g.attack(1 - side)
        } else {
            0.0
        },
        if defence {
            g.defence(side) - g.defence(1 - side)
        } else {
            0.0
        },
    )
}


#[cfg(test)]
mod tests {
    use super::*;

    /// The queue search the flood fill replaced, kept as the reference.
    fn reference_distances(board: &[i8; 81], source: usize, code: i8, removed: usize) -> [i16; 81] {
        let mut result = [UNREACHED; 81];
        let mut queue = [0_usize; 81];
        result[source] = 0;
        queue[0] = source;
        let (mut head, mut tail) = (0, 1);
        while head < tail {
            let square = queue[head];
            head += 1;
            for direction in 0..8 {
                if let Some(target) = super::super::destination((square * 8 + direction) as u16) {
                    let occupant = if target == removed { 0 } else { board[target] };
                    if result[target] != UNREACHED || (occupant != 0 && !captures(code, occupant)) {
                        continue;
                    }
                    result[target] = result[square] + 1;
                    queue[tail] = target;
                    tail += 1;
                }
            }
        }
        result
    }

    /// The per-enemy scan the shared threat maps replaced.
    fn reference_safe(g: &Geometry, piece: &Piece, square: usize, ply: i16) -> bool {
        !g.own(1 - piece.side).any(|enemy| {
            captures(enemy.code, piece.code)
                && arrival(enemy.distances[square], enemy.side, g.turn) <= ply
        })
    }

    /// Small LCG, so the corpus needs no dependency and is reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn board(&mut self) -> [i8; 81] {
            let mut board = [0_i8; 81];
            let count = 2 + self.next() % 18;
            for _ in 0..count {
                let square = (self.next() % 81) as usize;
                let kind = 1 + (self.next() % 3) as i8;
                board[square] = if self.next() % 2 == 0 { kind } else { -kind };
            }
            board
        }
    }

    #[test]
    fn predator_is_the_unique_capturing_code() {
        for code in [1_i8, 2, 3, -1, -2, -3] {
            let hunter = predator(code);
            assert!(captures(hunter, code), "{code} is not captured by {hunter}");
            for other in [1_i8, 2, 3, -1, -2, -3] {
                assert_eq!(captures(other, code), other == hunter, "{code} vs {other}");
            }
        }
    }

    #[test]
    fn passable_masks_admit_exactly_empty_and_prey_squares() {
        let mut rng = Rng(7);
        for _ in 0..40 {
            let board = rng.board();
            let masks = passable_masks(&board);
            for code in [1_i8, 2, 3, -1, -2, -3] {
                for square in 0..81 {
                    let allowed = masks[slot(code)] & (1 << square) != 0;
                    let expected = board[square] == 0 || captures(code, board[square]);
                    assert_eq!(allowed, expected, "code {code} square {square}");
                }
            }
        }
    }

    #[test]
    fn flood_matches_the_queue_search_for_every_piece() {
        let mut rng = Rng(11);
        for _ in 0..60 {
            let board = rng.board();
            let masks = passable_masks(&board);
            for goals in [[80_usize, 0_usize], [0, 80]] {
                for (square, &code) in board.iter().enumerate() {
                    if code == 0 {
                        continue;
                    }
                    let index = slot(code);
                    let goal = goals[owner(code) as usize];
                    assert_eq!(
                        flood(masks[index], 1 << square),
                        reference_distances(&board, square, code, 81),
                        "forward map at {square}"
                    );
                    assert_eq!(
                        flood(masks[index] | (1 << square), 1 << goal),
                        reference_distances(&board, goal, code, square),
                        "reverse map at {square}"
                    );
                }
            }
        }
    }

    #[test]
    fn multi_source_flood_is_the_minimum_over_single_sources() {
        let mut rng = Rng(13);
        for _ in 0..40 {
            let board = rng.board();
            let masks = passable_masks(&board);
            for code in [1_i8, 2, 3, -1, -2, -3] {
                let mut sources = 0_u128;
                for (square, &c) in board.iter().enumerate() {
                    if c == code {
                        sources |= 1 << square;
                    }
                }
                if sources == 0 {
                    continue;
                }
                let shared = flood(masks[slot(code)], sources);
                let mut expected = [UNREACHED; 81];
                let mut bits = sources;
                while bits != 0 {
                    let source = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let single = flood(masks[slot(code)], 1 << source);
                    for square in 0..81 {
                        expected[square] = expected[square].min(single[square]);
                    }
                }
                assert_eq!(shared, expected, "code {code}");
            }
        }
    }

    #[test]
    fn safe_matches_the_per_enemy_scan_at_every_square_and_ply() {
        let mut rng = Rng(17);
        for _ in 0..30 {
            let board = rng.board();
            for turn in [0_u8, 1] {
                let g = Geometry::from_board(&board, turn, 0);
                for piece in &g.pieces {
                    for square in (0..81).step_by(3) {
                        for ply in [0_i16, 1, 2, 3, 6, 12, 40, 200] {
                            assert_eq!(
                                g.safe(piece, square, ply),
                                reference_safe(&g, piece, square, ply),
                                "piece {} square {square} ply {ply}",
                                piece.square
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_piece_with_no_predator_left_is_safe_at_any_ply() {
        // Blue scissors with every Red rock gone: nothing can capture it.
        let mut board = [0_i8; 81];
        board[30] = 2;
        board[40] = -2;
        board[50] = -3;
        let g = Geometry::from_board(&board, 0, 0);
        let piece = g.pieces.iter().find(|p| p.square == 30).unwrap();
        assert_eq!(g.threat[piece.slot][0], NO_PREDATOR);
        for ply in [0_i16, 10, 500, 30_000] {
            assert!(g.safe(piece, 40, ply));
            assert!(reference_safe(&g, piece, 40, ply));
        }
    }

    #[test]
    fn profiles_match_the_inline_squares_and_deadlines() {
        let mut rng = Rng(19);
        for _ in 0..30 {
            let board = rng.board();
            let g = Geometry::from_board(&board, 0, 0);
            for (index, runner) in g.pieces.iter().enumerate() {
                let profile = match &g.profiles[index] {
                    Some(profile) => profile,
                    None => {
                        assert_eq!(runner.goal_distance, UNREACHED);
                        continue;
                    }
                };
                let mut expected = vec![runner.square];
                expected.extend((0..81).filter(|&s| {
                    s != runner.square
                        && runner.distances[s] + runner.reverse[s] == runner.goal_distance
                }));
                assert_eq!(profile.squares, expected);
                for (position, &square) in profile.squares.iter().enumerate() {
                    let runner_ply = arrival(runner.distances[square], runner.side, g.turn);
                    let want = if square == runner.square {
                        arrival(1, runner.side, g.turn) - 1
                    } else {
                        runner_ply + if square == runner.goal { -1 } else { 1 }
                    };
                    assert_eq!(profile.deadlines[position], want, "square {square}");
                }
            }
        }
    }
}
