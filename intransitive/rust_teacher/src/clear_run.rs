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
    fn chebyshev_and_arrival_match_the_python_definitions() {
        assert_eq!(chebyshev(0, 80), 8);
        assert_eq!(chebyshev(40, 40), 0);
        assert_eq!(arrival(0, 0, 0), 0);
        assert_eq!(arrival(3, 0, 0), 5);
        assert_eq!(arrival(3, 1, 0), 6);
    }
}
