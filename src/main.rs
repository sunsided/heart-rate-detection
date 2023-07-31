mod camera;
mod face_detector;
mod fps_counter;
mod sample_spectrogram;

use crate::camera::{CameraCapture, FrameNotification};
use crate::face_detector::FaceDetector;
use crate::sample_spectrogram::SampleSpectrogram;
use colorgrad::Gradient;
use nokhwa::utils::CameraIndex;
use opencv::core::{mean, no_array, Rect, Scalar, Vec3b, CV_8UC3};
use opencv::highgui::{imshow, wait_key};
use opencv::imgproc::{rectangle, FILLED, LINE_4};
use opencv::prelude::*;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync;
use std::time::{Duration, Instant};

const KEY_CODE_ESCAPE: i32 = 27;
const CASCADE_XML_FILE: &str = "haarcascade_frontalface_alt.xml";

const HISTOGRAM_WIDTH: i32 = 640;
const HISTOGRAM_HEIGHT: i32 = 100;

const MIN_IBI_FREQ: f64 = 0.5;
const MAX_IBI_FREQ: f64 = 1.5;

// TODO: Impractical but fun: Use optical flow to adjust the detected face until the next guess comes in.

// TODO: Heart Rate Variability (~55 to ~105 ms variability for teenagers, ~25 to ~45 for older people)
// TODO: Interbeat Interval (IBI)

fn main() {
    let palette = colorgrad::turbo();
    let face_detector = FaceDetector::new(&PathBuf::from(CASCADE_XML_FILE));

    let sample_spectrogram = SampleSpectrogram::new();
    let frequencies = sample_spectrogram.frequencies();

    let min_idx = frequencies.iter().position(|&v| v >= MIN_IBI_FREQ).unwrap();
    let max_idx = frequencies.iter().position(|&v| v >= MAX_IBI_FREQ).unwrap();

    let (notify_sender, new_frame_notification) = sync::mpsc::sync_channel(1);

    let index = CameraIndex::Index(0);
    let camera = CameraCapture::try_new_from_index(index, notify_sender).unwrap();

    let histogram_background = palette.at(0.0).to_rgba8();

    let mut spectrogram_buffer = Mat::new_rows_cols_with_default(
        HISTOGRAM_HEIGHT,
        HISTOGRAM_WIDTH,
        CV_8UC3,
        Scalar::from_array([
            histogram_background[2] as _,
            histogram_background[1] as _,
            histogram_background[0] as _,
            0.0,
        ]),
    )
    .unwrap();

    let mut display_delay = Instant::now();
    let mut sample_delay = Instant::now();

    let mut min_hist = 0.0;
    let mut max_hist = 0.0;

    let mut last_detection = None;

    let color_red = Scalar::new(0.0, 0.0, 255.0, -1.0);
    let color_green = Scalar::new(0.0, 255.0, 0.0, -1.0);

    let sample_history_length = 90;
    let mut sample_history = VecDeque::with_capacity(sample_history_length);
    let mut sample_count = 0;

    while let Ok(FrameNotification::NewFrame { fps: _fps }) = new_frame_notification.recv() {
        let mut bgr_buffer = match camera.frame() {
            None => {
                eprintln!("Got an empty buffer");
                continue;
            }
            Some(frame) => frame,
        };

        face_detector.provide_image(&bgr_buffer);

        let color = match face_detector.face() {
            None => color_red,
            Some((true, region)) => {
                last_detection = Some(region);
                color_green
            }
            Some((false, region)) => {
                last_detection = Some(region);
                color_red
            }
        };

        if let Some(region) = last_detection {
            let roi = Mat::roi(&bgr_buffer, region).unwrap();
            let mean = mean(&roi, &no_array()).unwrap();

            // Sample and hold.
            let sample = mean[1] / 255.0;

            // Keep a moving average range.
            if sample_history.len() == sample_history_length {
                sample_history.pop_front();
            }
            sample_history.push_back(sample);
            let min = sample_history
                .iter()
                .min_by(|&x, &y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                .unwrap();
            let max = sample_history
                .iter()
                .max_by(|&x, &y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                .unwrap();

            sample_count += 1;
            let sample_adjusted = (sample - min) / (max - min);

            sample_spectrogram.sample_and_hold(sample_adjusted);
            println!("Sample #{sample_count:5}: {sample_adjusted:.4} ({sample:.6} in {min:.6} .. {max:.6})");

            draw_box_around_face(&mut bgr_buffer, region, &color).unwrap();

            rectangle(
                &mut bgr_buffer,
                Rect::new(
                    0,
                    (CameraCapture::CAMERA_CAPTURE_HEIGHT as f64
                        - CameraCapture::CAMERA_CAPTURE_HEIGHT as f64 * sample_adjusted)
                        as _,
                    5,
                    (CameraCapture::CAMERA_CAPTURE_HEIGHT as i32 - HISTOGRAM_HEIGHT) as _,
                ),
                Scalar::from_array([0.0, 0.0, 255.0, 0.0]),
                FILLED,
                LINE_4,
                0,
            )
            .unwrap();
        }

        update_spectrogram_display(
            &palette,
            &sample_spectrogram,
            &mut spectrogram_buffer,
            &mut sample_delay,
            &mut min_hist,
            &mut max_hist,
            min_idx,
            max_idx,
        );

        let mut spectrogram_region = Mat::roi(
            &bgr_buffer,
            Rect::new(
                0,
                bgr_buffer.rows() - HISTOGRAM_HEIGHT,
                HISTOGRAM_WIDTH,
                HISTOGRAM_HEIGHT,
            ),
        )
        .unwrap();

        spectrogram_buffer.copy_to(&mut spectrogram_region).unwrap();

        if Instant::now() - display_delay > Duration::from_millis(30) {
            display_delay = Instant::now();

            imshow("Image", &bgr_buffer).unwrap();
            if let Ok(KEY_CODE_ESCAPE) = wait_key(1) {
                break;
            }
        }
    }
}

fn update_spectrogram_display(
    palette: &Gradient,
    sample_spectrogram: &SampleSpectrogram,
    mut spectrogram_buffer: &mut Mat,
    sample_delay: &mut Instant,
    min_hist: &mut f64,
    max_hist: &mut f64,
    min_idx: usize,
    max_idx: usize,
) {
    let tbf = Instant::now() - *sample_delay;
    if tbf.as_millis() >= 30 {
        *sample_delay = Instant::now();

        let mut spectrogram_column = Vec::new();
        sample_spectrogram.copy_spectrogram_into(&mut spectrogram_column);

        let mut min = f64::MAX;
        let mut max = f64::MIN;

        let skip = min_idx;
        let take_n = max_idx - min_idx;
        for &value in spectrogram_column.iter().skip(skip).take(take_n) {
            min = min.min(value);
            max = max.max(value);
        }

        if max <= 1e-4 {
            max = 1.0;
        }

        min = min * 0.9 + *min_hist * 0.1;
        max = max * 0.9 + *max_hist * 0.1;

        *min_hist = min;
        *max_hist = max;

        let row = spectrogram_buffer
            .at_row_mut::<Vec3b>(HISTOGRAM_HEIGHT - 1)
            .unwrap();

        let bin_width = (1.0 / (take_n as f64) * (HISTOGRAM_WIDTH as f64)) as usize;

        for (i, &value) in spectrogram_column
            .iter()
            .skip(skip)
            .take(take_n)
            .enumerate()
        {
            let sample = (value - min) / (max - min);
            // let sample = value;
            let color = palette.at(sample);
            let rgb = color.to_rgba8();

            let j = ((i as f64) / (take_n as f64) * (HISTOGRAM_WIDTH as f64)) as usize;
            for j in j..(j + bin_width) {
                row[j] = Vec3b::from_array([rgb[2], rgb[1], rgb[0]]);
            }
        }

        shift_spectrogram(&mut spectrogram_buffer);
    }
}

fn shift_spectrogram(histogram_buffer: &mut Mat) {
    let src = Mat::roi(
        &histogram_buffer,
        Rect::new(0, 1, HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT - 1),
    )
    .unwrap();
    let mut dst = Mat::roi(
        &histogram_buffer,
        Rect::new(0, 0, HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT - 1),
    )
    .unwrap();
    src.copy_to(&mut dst).unwrap();
}

fn draw_box_around_face(frame: &mut Mat, face: Rect, color: &Scalar) -> Result<(), opencv::Error> {
    const THICKNESS: i32 = 2;
    const LINE_TYPE: i32 = 8;
    const SHIFT: i32 = 0;

    rectangle(frame, face, color.clone(), THICKNESS, LINE_TYPE, SHIFT)?;
    Ok(())
}
