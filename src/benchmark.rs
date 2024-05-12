use std::time::{Duration, Instant};



#[derive(Clone, Debug, PartialEq, Default, Eq, Hash)]
pub struct Timer {
    pub time: Duration,
    pub start: Option<Instant>,
}

impl Timer {
    pub const fn new() -> Self {
        Self { time: Duration::new(0, 0), start: None }
    }

    pub fn start(&mut self) {
        let prev = self.start.replace(Instant::now());
        assert!(
            prev.is_none(),
            "failed to start bench capture after another start",
        );
    }

    pub fn end(&mut self) {
        let start = self.start.take()
            .expect("failed to end unstartded bench capture");

        self.time += start.elapsed();
    }
}



#[derive(Clone, Debug, PartialEq, Default, Eq, Hash)]
pub struct Bench {
    pub render: Timer,
    pub copy: Timer,
}

impl Bench {
    pub const fn new() -> Self {
        Self { render: Timer::new(), copy: Timer::new() }
    }

    pub const fn total(self) -> TotalTime {
        assert!(
            self.render.start.is_none(),
            "failed to calculate total time: start time is pending",
        );

        assert!(
            self.copy.start.is_none(),
            "failed to calculate total time: copy time is pending",
        );

        TotalTime { render: self.render.time, copy: self.copy.time }
    }
}



#[derive(Clone, Debug, PartialEq, Default, Eq, Hash)]
pub struct TotalTime {
    pub render: Duration,
    pub copy: Duration,
}

impl std::fmt::Display for TotalTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Time spent on: \n  - rendering: {:?}\n  - copying: {:?}\n\
            Total: {:?}",
            self.render,
            self.copy,
            self.render + self.copy,
        )
    }
}