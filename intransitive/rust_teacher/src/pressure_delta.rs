//! Board-local attacker/defender ring counts for incremental pressure.
use crate::{captures, distance, owner};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Pressure {
    pub radius: usize,
    pub totals: [f64; 2],
    occupied: u128,
    attackers: [[u8; 5]; 81],
    defenders: [[u8; 5]; 81],
    values: [f64; 81],
}

impl Pressure {
    pub fn new(board: &[i8; 81], radius: usize) -> Option<Box<Self>> {
        // With at most 32 pieces, every term and intermediate total fits in
        // binary64 exactly: dyadic denominator <= 2^33, total < 2^10. Larger
        // edited boards retain the reference pair-summation implementation.
        if board.iter().filter(|&&code| code != 0).count() > 32 {
            return None;
        }
        let mut value = Box::new(Self {
            radius,
            totals: [0.0; 2],
            occupied: 0,
            attackers: [[0; 5]; 81],
            defenders: [[0; 5]; 81],
            values: [0.0; 81],
        });
        let mut partial = [0; 81];
        for (square, &code) in board.iter().enumerate() {
            if code != 0 {
                partial[square] = code;
                value.add(&partial, square);
            }
        }
        Some(value)
    }

    fn refresh(&mut self, square: usize, code: i8) {
        let weights = [0.0, 2.0, 1.0, 0.5, 0.25];
        let mut defence = 0;
        let mut value = 0.0;
        for (ring, &weight) in weights.iter().enumerate().take(self.radius + 1).skip(1) {
            defence += self.defenders[square][ring] as u64;
            let discount = f64::from_bits((1023 - defence) << 52);
            value += self.attackers[square][ring] as f64 * weight * discount;
        }
        // A victim's threats are credited to its opponent, exactly once.
        self.totals[1 - owner(code) as usize] += value - self.values[square];
        self.values[square] = value;
    }

    fn defends(friend: i8, victim: i8) -> bool {
        friend * victim > 0 && friend.abs() == victim.abs() % 3 + 1
    }

    pub fn remove(&mut self, board: &[i8; 81], square: usize) {
        let code = board[square];
        self.occupied &= !(1 << square);
        let mut remaining = self.occupied;
        while remaining != 0 {
            let other = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            let victim = board[other];
            let attack = captures(code, victim);
            let defence = Self::defends(code, victim);
            if !attack && !defence {
                continue;
            }
            let ring = distance(square, other);
            if ring > self.radius {
                continue;
            }
            if attack {
                self.attackers[other][ring] -= 1;
                self.refresh(other, victim);
            } else {
                self.defenders[other][ring] -= 1;
                if self.values[other] != 0.0 {
                    self.refresh(other, victim);
                }
            }
        }
        self.totals[1 - owner(code) as usize] -= self.values[square];
        self.values[square] = 0.0;
        self.attackers[square] = [0; 5];
        self.defenders[square] = [0; 5];
    }

    pub fn add(&mut self, board: &[i8; 81], square: usize) {
        let code = board[square];
        let mut remaining = self.occupied;
        while remaining != 0 {
            let other = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            let piece = board[other];
            let ring = distance(square, other);
            if ring > self.radius {
                continue;
            }
            if captures(piece, code) {
                self.attackers[square][ring] += 1;
            } else if Self::defends(piece, code) {
                self.defenders[square][ring] += 1;
            }
            if captures(code, piece) {
                self.attackers[other][ring] += 1;
                self.refresh(other, piece);
            } else if Self::defends(code, piece) {
                self.defenders[other][ring] += 1;
                if self.values[other] != 0.0 {
                    self.refresh(other, piece);
                }
            }
        }
        self.occupied |= 1 << square;
        self.refresh(square, code);
    }
}
