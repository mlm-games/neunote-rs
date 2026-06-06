/// Simple synthesizer voice for audio preview playback
pub struct SynthVoice {
    phase: f64,
    frequency: f64,
    sample_rate: f64,
    envelope: Envelope,
}

#[derive(Clone, Copy)]
pub struct Envelope {
    pub attack: f64,
    pub decay: f64,
    pub sustain: f32,
    pub release: f64,
    level: f64,
    phase: EnvPhase,
    counter: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum EnvPhase {
    Off,
    Attack,
    Decay,
    Sustain,
    Release,
}

impl Envelope {
    pub fn new(attack: f64, decay: f64, sustain: f32, release: f64) -> Self {
        Self {
            attack,
            decay,
            sustain,
            release,
            level: 0.0,
            phase: EnvPhase::Off,
            counter: 0,
        }
    }

    pub fn note_on(&mut self, _sample_rate: f64) {
        self.phase = EnvPhase::Attack;
        self.counter = 0;
    }

    pub fn note_off(&mut self) {
        if self.phase != EnvPhase::Off {
            self.phase = EnvPhase::Release;
            self.counter = 0;
        }
    }

    pub fn process(&mut self, sample_rate: f64) -> f64 {
        match self.phase {
            EnvPhase::Off => 0.0,
            EnvPhase::Attack => {
                let dur = (self.attack * sample_rate) as usize;
                self.level = (self.counter as f64 / dur as f64).min(1.0);
                self.counter += 1;
                if self.counter >= dur {
                    self.phase = EnvPhase::Decay;
                    self.counter = 0;
                }
                self.level
            }
            EnvPhase::Decay => {
                let dur = (self.decay * sample_rate) as usize;
                let t = (self.counter as f64 / dur as f64).min(1.0);
                self.level = 1.0 - (1.0 - self.sustain as f64) * t;
                self.counter += 1;
                if self.counter >= dur {
                    self.phase = EnvPhase::Sustain;
                    self.counter = 0;
                }
                self.level
            }
            EnvPhase::Sustain => self.sustain as f64,
            EnvPhase::Release => {
                let dur = (self.release * sample_rate) as usize;
                let t = (self.counter as f64 / dur as f64).min(1.0);
                self.level = self.sustain as f64 * (1.0 - t);
                self.counter += 1;
                if self.counter >= dur {
                    self.phase = EnvPhase::Off;
                    self.level = 0.0;
                }
                self.level
            }
        }
    }

    pub fn is_active(&self) -> bool {
        self.phase != EnvPhase::Off
    }
}

impl SynthVoice {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            phase: 0.0,
            frequency: 440.0,
            sample_rate,
            envelope: Envelope::new(0.01, 0.1, 0.7, 0.2),
        }
    }

    pub fn note_on(&mut self, midi_note: u8, _amplitude: f32) {
        self.frequency = 440.0 * 2.0_f64.powf((midi_note as f64 - 69.0) / 12.0);
        self.envelope.note_on(self.sample_rate);
    }

    pub fn note_off(&mut self) {
        self.envelope.note_off();
    }

    /// Generate next sample of audio
    pub fn process(&mut self) -> f32 {
        let env = self.envelope.process(self.sample_rate);
        if env == 0.0 {
            return 0.0;
        }

        // Sine wave with some harmonics for a richer tone
        let mut sample = (self.phase * 2.0 * std::f64::consts::PI).sin() * 0.3;
        sample += (self.phase * 4.0 * std::f64::consts::PI).sin() * 0.15;
        sample += (self.phase * 6.0 * std::f64::consts::PI).sin() * 0.05;

        self.phase += self.frequency / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        (sample * env) as f32
    }

    pub fn is_active(&self) -> bool {
        self.envelope.is_active()
    }
}
