//! Forced corner-run certificate: a sufficient condition, never an estimate.
//!
//! Mirrors `intransitive/heuristics/clear_run.py`. The bounded terminal search
//! sees two plies; a runner that cannot be stopped often wins further out, and
//! saying so is cheap — but the claim must be airtight, because a proof result
//! replaces the weighted sum with a mate score and prunes the lines that would
//! have refuted it.
//!
//! Every bound errs toward silence: enemy arrival uses free-board Chebyshev
//! distance, which occupancy can only worsen; blocking and capturing are
//! treated alike, so any enemy able to stand on a square in time disqualifies
//! it; the runner walks only empty squares; it must arrive strictly before any
//! enemy could reach their own corner; and the draw clock must not expire.
use super::{owner, Position};

pub(super) const NO_RUN: i32 = -1;
pub(super) const NO_SIDE: i8 = -1;
const UNREACHABLE: i32 = 1 << 20;

fn chebyshev(a: usize, b: usize) -> i32 {
    (a % 9).abs_diff(b % 9).max((a / 9).abs_diff(b / 9)) as i32
}

fn arrival(moves: i32, side: u8, turn: u8) -> i32 {
    if moves == 0 {
        0
    } else {
        2 * moves - i32::from(side == turn)
    }
}

/// The only side whose certificate could survive here, or `NO_SIDE`.
///
/// Mirrors `race_gate` in `clear_run.py`. `certify` sweeps the whole board once
/// per enemy piece and then floods once per runner; this reads the board once,
/// and every refusal it makes is one the full argument would have made too,
/// from the same free-board bound. It never hides a certificate, it only
/// declines to look for one that cannot exist.
///
/// Write `d` for the fewest king steps any of this side's pieces needs to reach
/// its corner, ignoring the board. A run of `moves` steps has `moves >= d` and
/// `arrival` is monotone in moves, so `arrival(d)` is the earliest ply any run
/// of this side could finish on. `certify` already refuses a run that needs
/// more than `limit` moves, that does not finish strictly inside the draw
/// clock, that does not arrive strictly before the other side's own fastest
/// free-board arrival at its corner, that steps onto an occupied square — the
/// corner included — or that reaches the corner no earlier than an enemy could
/// stand on it. Each is read here at the most permissive run the side could
/// hold, so passing is necessary, never sufficient.
///
/// The race comparison is strict and symmetric, so at most one side can pass.
pub(super) fn race_gate(p: &Position, clock_left: i32, limit: i32) -> i8 {
    let goals = if p.a1 == 0 { [80_usize, 0] } else { [0, 80] };
    // `nearest`/`steps` skip a piece already standing on its goal, as the
    // runner loop in `certify` does. `race` keeps every piece, as
    // `soonest_enemy` does, and `cover` is that same arrival read at the other
    // side's corner.
    let mut nearest = [UNREACHABLE; 2];
    let mut steps = [UNREACHABLE; 2];
    let mut race = [UNREACHABLE; 2];
    let mut cover = [UNREACHABLE; 2];
    for (square, &code) in p.board.iter().enumerate() {
        if code == 0 {
            continue;
        }
        let side = owner(code) as usize;
        let moves = chebyshev(square, goals[side]);
        let ply = arrival(moves, side as u8, p.side);
        race[side] = race[side].min(ply);
        let reach = arrival(chebyshev(square, goals[1 - side]), side as u8, p.side);
        cover[1 - side] = cover[1 - side].min(reach);
        if square == goals[side] {
            continue;
        }
        steps[side] = steps[side].min(moves);
        nearest[side] = nearest[side].min(ply);
    }
    for index in 0..2 {
        let side = if index == 0 { p.side } else { 1 - p.side } as usize;
        if steps[side] <= limit
            && nearest[side] < clock_left
            && nearest[side] < race[1 - side]
            && p.board[goals[side]] == 0
            && cover[side] > nearest[side]
        {
            return side as i8;
        }
    }
    NO_SIDE
}

/// Earliest ply any enemy could occupy each square, and reach its own corner.
fn soonest_enemy(board: &[i8; 81], side: u8, turn: u8, enemy_goal: usize) -> ([i32; 81], i32) {
    let mut soonest = [UNREACHABLE; 81];
    let mut race = UNREACHABLE;
    let enemy_side = 1 - side;
    for (square, &code) in board.iter().enumerate() {
        if code == 0 || owner(code) != enemy_side {
            continue;
        }
        for (target, slot) in soonest.iter_mut().enumerate() {
            let ply = arrival(chebyshev(square, target), enemy_side, turn);
            if ply < *slot {
                *slot = ply;
            }
        }
        let ply = arrival(chebyshev(square, enemy_goal), enemy_side, turn);
        if ply < race {
            race = ply;
        }
    }
    (soonest, race)
}

/// Plies for one runner to force the goal, or NO_RUN.
fn safe_run(
    board: &[i8; 81],
    source: usize,
    side: u8,
    turn: u8,
    goal: usize,
    soonest: &[i32; 81],
    limit: i32,
) -> i32 {
    // The runner stands on its own square until it first moves. When the
    // opponent moves first they get a ply to capture it where it sits, so the
    // starting square needs the same clearance as every square on the route.
    if soonest[source] <= arrival(1, side, turn) - 1 {
        return NO_RUN;
    }
    let mut step = [-1_i32; 81];
    let mut frontier = Vec::with_capacity(81);
    let mut following = Vec::with_capacity(81);
    step[source] = 0;
    frontier.push(source);
    let mut moves = 0;
    while !frontier.is_empty() && moves < limit {
        moves += 1;
        let ply = arrival(moves, side, turn);
        following.clear();
        for &square in &frontier {
            let (x, y) = ((square % 9) as i32, (square / 9) as i32);
            for dy in -1..=1_i32 {
                for dx in -1..=1_i32 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (x + dx, y + dy);
                    if !(0..9).contains(&nx) || !(0..9).contains(&ny) {
                        continue;
                    }
                    let target = (ny * 9 + nx) as usize;
                    if step[target] >= 0 || board[target] != 0 {
                        continue;
                    }
                    // The runner holds a square until its next move, so an
                    // enemy arriving one ply later could still take or block
                    // it. Reaching the goal ends the game before any reply.
                    let deadline = if target == goal { ply } else { ply + 1 };
                    if soonest[target] <= deadline {
                        continue;
                    }
                    step[target] = moves;
                    following.push(target);
                }
            }
            if step[goal] >= 0 {
                break;
            }
        }
        if step[goal] >= 0 {
            return arrival(step[goal], side, turn);
        }
        std::mem::swap(&mut frontier, &mut following);
    }
    NO_RUN
}

/// Plies until `side` forces a corner win, or NO_RUN.
pub(super) fn certify(p: &Position, side: u8, clock_left: i32, limit: i32) -> i32 {
    let goals = if p.a1 == 0 { [80_usize, 0] } else { [0, 80] };
    let goal = goals[side as usize];
    let enemy_goal = goals[(1 - side) as usize];
    let (soonest, race) = soonest_enemy(&p.board, side, p.side, enemy_goal);
    let mut best = NO_RUN;
    for (square, &code) in p.board.iter().enumerate() {
        if code == 0 || owner(code) != side || square == goal {
            continue;
        }
        let plies = safe_run(&p.board, square, side, p.side, goal, &soonest, limit);
        if plies == NO_RUN || plies >= race || plies >= clock_left {
            continue;
        }
        if best == NO_RUN || plies < best {
            best = plies;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exhaustive search for any terminal result inside `depth` plies.
    fn decided_within(p: &mut Position, depth: usize) -> bool {
        if p.terminal(true).is_some() {
            return true;
        }
        if depth == 0 {
            return false;
        }
        for action in p.raw_legal(p.side) {
            let undo = p.push(action);
            let found = decided_within(p, depth - 1);
            p.pop(undo);
            if found {
                return true;
            }
        }
        false
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    #[test]
    fn a_claimed_run_of_two_plies_is_always_really_forced() {
        // The shortest claims are the ones an exhaustive check can confirm, and
        // the ones that broke first: with the opponent to move, a runner one
        // step out can simply be taken where it stands.
        let mut rng = Rng(101);
        let mut checked = 0;
        for _ in 0..4000 {
            let mut board = [0_i8; 81];
            for _ in 0..2 + rng.next() % 6 {
                let square = (rng.next() % 81) as usize;
                let kind = 1 + (rng.next() % 3) as i8;
                board[square] = if rng.next() % 2 == 0 { kind } else { -kind };
            }
            let mut p = super::super::tests::fixture_board(board);
            if p.terminal(true).is_some() {
                continue;
            }
            let plies = certify(&p, p.side, 80, 20);
            if plies != 2 && plies != 1 {
                continue;
            }
            checked += 1;
            assert!(
                decided_within(&mut p, plies as usize),
                "certificate claimed a forced win in {plies} plies that is not there"
            );
        }
        assert!(checked > 0, "the corpus never exercised the certificate");
    }

    #[test]
    fn the_gate_never_hides_a_certificate() {
        // The gate exists to make an interior probe affordable, so what has to
        // hold is one-sided: it may refuse a position `certify` would also
        // refuse, but never a position that really holds a run.
        let mut rng = Rng(7717);
        let (mut passes, mut certificates) = (0, 0);
        for _ in 0..20000 {
            let mut board = [0_i8; 81];
            for _ in 0..2 + rng.next() % 8 {
                let square = (rng.next() % 81) as usize;
                let kind = 1 + (rng.next() % 3) as i8;
                board[square] = if rng.next() % 2 == 0 { kind } else { -kind };
            }
            let p = super::super::tests::fixture_board(board);
            let clock_left = (rng.next() % 81) as i32;
            let limit = 1 + (rng.next() % 20) as i32;
            let gate = race_gate(&p, clock_left, limit);
            if gate != NO_SIDE {
                passes += 1;
            }
            for side in 0..2_u8 {
                let plies = certify(&p, side, clock_left, limit);
                if plies != NO_RUN {
                    certificates += 1;
                    assert_eq!(gate, side as i8, "the gate refused a real certificate");
                }
            }
        }
        assert!(certificates > 0, "the corpus never produced a certificate");
        assert!(passes > 0, "the gate refused every position");
    }

    #[test]
    fn a_held_corner_and_a_covered_corner_are_refused_without_certifying() {
        // Blue paper is two steps from I9 and would certify, but the gate
        // refuses once the corner itself is occupied, and again once a red
        // piece can stand on it first.
        let mut board = [0_i8; 81];
        board[9 * 6 + 6] = 3;
        board[9 * 4] = -1;
        let p = super::super::tests::fixture_board(board);
        assert_eq!(race_gate(&p, 80, 20), 0);
        assert_ne!(certify(&p, 0, 80, 20), NO_RUN);
        let mut held = board;
        held[80] = -1;
        assert_eq!(race_gate(&super::super::tests::fixture_board(held), 80, 20), NO_SIDE);
        let mut covered = board;
        covered[9 * 7 + 7] = -1;
        assert_eq!(race_gate(&super::super::tests::fixture_board(covered), 80, 20), NO_SIDE);
    }

    #[test]
    fn chebyshev_and_arrival_match_the_python_definitions() {
        assert_eq!(chebyshev(0, 80), 8);
        assert_eq!(chebyshev(40, 40), 0);
        assert_eq!(arrival(0, 0, 0), 0);
        assert_eq!(arrival(3, 0, 0), 5);
        assert_eq!(arrival(3, 1, 0), 6);
    }
}
