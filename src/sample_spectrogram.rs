use ruststft::{WindowType, STFT};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub struct SampleSpectrogram {
    spectrogram_column: Arc<Mutex<Vec<f64>>>,
    sample: Arc<Mutex<f64>>,
    freqs: Vec<f64>,
}

impl SampleSpectrogram {
    const STFT_WINDOW_SIZE: usize = 300;
    const STFT_STEP_SIZE: usize = 30;
    const SAMPLING_RATE: usize = 30;

    pub fn new() -> Self {
        let window_type = WindowType::Hanning;
        let window_size: usize = Self::STFT_WINDOW_SIZE;
        let step_size: usize = Self::STFT_STEP_SIZE;
        let stft = STFT::new(window_type, window_size, step_size);
        let freqs = stft.freqs(Self::SAMPLING_RATE as _);

        let sample = Arc::new(Mutex::new(0.0));

        let spectrogram_column: Vec<f64> = std::iter::repeat(0.).take(stft.output_size()).collect();

        let spectrogram_column = Arc::new(Mutex::new(spectrogram_column));

        thread::spawn({
            let sample = sample.clone();
            let spectrogram_column = spectrogram_column.clone();
            move || Self::run_thread(stft, sample, spectrogram_column)
        });

        Self {
            spectrogram_column,
            sample,
            freqs,
        }
    }

    pub fn sample_and_hold(&self, sample: f64) {
        let mut value = self.sample.lock().unwrap();
        *value = sample;
    }

    pub fn copy_spectrogram_into(&self, column: &mut Vec<f64>) {
        let spectrogram = self.spectrogram_column.lock().unwrap();
        if column.len() != spectrogram.len() {
            column.resize(spectrogram.len(), 0.0);
        }

        column.copy_from_slice(&spectrogram);
    }

    /// Determines the corresponding frequencies of a column of the spectrogram.
    pub fn frequencies(&self) -> &Vec<f64> {
        &self.freqs
    }

    fn run_thread(
        mut stft: STFT<f64>,
        sample: Arc<Mutex<f64>>,
        spectrogram_column: Arc<Mutex<Vec<f64>>>,
    ) {
        let mut buffer: Vec<f64> = std::iter::repeat(0.).take(stft.output_size()).collect();

        // pre-fill
        stft.append_samples(&vec![0.0; Self::STFT_WINDOW_SIZE]);

        let mut samples = [0.0];
        let sample_delay = Duration::from_secs_f64(1.0 / (Self::SAMPLING_RATE as f64));

        let mut last_tick = Instant::now();
        loop {
            // Sample every ~30 Hz
            let now = Instant::now();
            let delay = now - last_tick;
            if delay < sample_delay {
                let remaining = sample_delay - delay;
                thread::sleep(remaining);
            }

            last_tick = now;

            let sample = sample.lock().unwrap();
            samples[0] = *sample;
            stft.append_samples(&samples);

            while stft.contains_enough_to_compute() {
                stft.compute_column(&mut buffer[..]);
                stft.move_to_next_column();

                /*
                // Normalize the spectrum.
                let length_adjustment = 1.0 / (buffer.len() as f64);
                let amplitude_adjustment = 1.0 / 1.0;
                for value in &mut buffer {
                    *value = (10.0_f64.powf(*value) * length_adjustment * amplitude_adjustment);
                }
                */

                // Copy the data over.
                let mut spectrogram_column = spectrogram_column.lock().unwrap();
                spectrogram_column.copy_from_slice(&buffer);
            }
        }
    }
}
