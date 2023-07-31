use std::collections::VecDeque;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct FpsCounter {
    buffer: VecDeque<Instant>,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            buffer: VecDeque::with_capacity(60),
        }
    }

    pub fn tick(&mut self) -> f64 {
        let now = Instant::now();

        // Remove all frame instances that were older than a second.
        let cutoff = now - Duration::from_secs(1);
        while self.buffer.front().map_or(false, |t| *t < cutoff) {
            self.buffer.pop_front();
        }

        self.buffer.push_back(now);
        if self.buffer.len() == 1 {
            return 0.0;
        }

        let (sum, count) = self
            .buffer
            .iter()
            .zip(self.buffer.iter().skip(1))
            .map(|(&prev, &current)| (current - prev).as_secs_f64())
            .fold((0.0, 0), |(sum, count), value| (sum + value, count + 1));
        (count as f64) / sum
    }
}
