//! Python-compatible static occupied-board route features.
use super::{captures, destination, owner, Position};

struct Piece {
    square: usize,
    code: i8,
    side: u8,
    goal: usize,
    distances: [i16; 81],
    reverse: [i16; 81],
}
impl Piece {
    fn distance(&self) -> i16 {
        self.distances[self.goal]
    }
    fn route(&self) -> Vec<usize> {
        if self.distance() == 99 {
            return vec![];
        }
        (0..81)
            .filter(|&s| s != self.square && self.distances[s] + self.reverse[s] == self.distance())
            .collect()
    }
}
fn distances(board: &[i8; 81], source: usize, code: i8, removed: usize) -> [i16; 81] {
    let mut result = [99; 81];
    let mut queue = [0; 81];
    result[source] = 0;
    queue[0] = source;
    let (mut head, mut tail) = (0, 1);
    while head < tail {
        let square = queue[head];
        head += 1;
        for direction in 0..8 {
            if let Some(target) = destination((square * 8 + direction) as u16) {
                let occupant = if target == removed { 0 } else { board[target] };
                if result[target] != 99 || (occupant != 0 && !captures(code, occupant)) {
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
fn arrival(moves: i16, side: u8, turn: u8) -> i16 {
    if moves == 0 {
        0
    } else {
        2 * moves - i16::from(side == turn)
    }
}
struct Geometry {
    pieces: Vec<Piece>,
    turn: u8,
    goals: [usize; 2],
}
impl Geometry {
    fn new(p: &Position) -> Self {
        let goals = if p.a1 == 0 { [80, 0] } else { [0, 80] };
        let pieces = p
            .board
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != 0)
            .map(|(square, &code)| {
                let side = owner(code);
                let goal = goals[side as usize];
                Piece {
                    square,
                    code,
                    side,
                    goal,
                    distances: distances(&p.board, square, code, 81),
                    reverse: distances(&p.board, goal, code, square),
                }
            })
            .collect();
        Self {
            pieces,
            turn: p.side,
            goals,
        }
    }
    fn own(&self, side: u8) -> impl Iterator<Item = &Piece> {
        self.pieces.iter().filter(move |p| p.side == side)
    }
    fn safe(&self, piece: &Piece, square: usize, ply: i16) -> bool {
        !self.own(1 - piece.side).any(|enemy| {
            captures(enemy.code, piece.code)
                && arrival(enemy.distances[square], enemy.side, self.turn) <= ply
        })
    }
    fn intercept(&self, defender: &Piece, runner: &Piece) -> Option<i16> {
        if runner.distance() == 99 {
            return None;
        }
        let capture = captures(defender.code, runner.code);
        if !capture && defender.code.abs() != runner.code.abs() {
            return None;
        }
        let squares = if capture {
            let mut v = vec![runner.square];
            v.extend(runner.route());
            v
        } else {
            vec![runner.goal]
        };
        squares
            .into_iter()
            .filter_map(|square| {
                let rp = arrival(runner.distances[square], runner.side, self.turn);
                let dp = arrival(defender.distances[square], defender.side, self.turn);
                let deadline = if square == runner.square {
                    arrival(1, runner.side, self.turn) - 1
                } else {
                    rp + if capture && square != runner.goal {
                        1
                    } else {
                        -1
                    }
                };
                if dp <= deadline && self.safe(defender, square, dp.max(deadline)) {
                    Some(dp)
                } else {
                    None
                }
            })
            .min()
    }
    fn attack(&self, side: u8) -> f64 {
        let mut progress = 0.0_f64;
        let mut opportunities = [false; 81];
        for piece in self.own(side) {
            if piece.route().into_iter().any(|s| {
                piece.distances[s] == 1 && self.safe(piece, s, arrival(1, side, self.turn) + 1)
            }) {
                progress += 1.0 / (1.0 + piece.distance() as f64);
            }
            for enemy in self.own(1 - side) {
                if captures(piece.code, enemy.code)
                    && piece.distances[enemy.square] == 1
                    && self.safe(piece, enemy.square, arrival(1, side, self.turn) + 1)
                {
                    opportunities[enemy.square] = true;
                }
            }
        }
        progress.min(2.0)
            + if opportunities.iter().any(|&v| v) {
                1.0
            } else {
                0.0
            }
    }
    fn defence(&self, side: u8) -> f64 {
        let mut value = 0.0_f64;
        for runner in self.own(1 - side).filter(|p| p.distance() < 99) {
            let coverage: f64 = self
                .own(side)
                .filter_map(|d| self.intercept(d, runner))
                .map(|ply| 1.0 / (1.0 + ply as f64))
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
