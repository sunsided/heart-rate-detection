mod camera;
mod fps_counter;

use crate::camera::{CameraCapture, FrameNotification};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
};
use nokhwa::{Buffer, Camera, NokhwaError};
use opencv::core::{mean, no_array, Rect, Scalar, Size, Vec3b, CV_8UC3};
use opencv::highgui::{imshow, wait_key};
use opencv::imgproc::{
    cvt_color, equalize_hist, rectangle, resize, COLOR_BGR2GRAY, COLOR_RGB2BGR, INTER_LINEAR,
};
use opencv::objdetect::CascadeClassifier;
use opencv::prelude::*;
use opencv::types::VectorOfRect;
use ruststft::{WindowType, STFT};
use std::sync;
use std::time::{Duration, Instant};

const KEY_CODE_ESCAPE: i32 = 27;
const CASCADE_XML_FILE: &str = "haarcascade_frontalface_alt.xml";

const CAMERA_CAPTURE_WIDTH: u32 = 640;
const CAMERA_CAPTURE_HEIGHT: u32 = 480;
const CAMERA_CAPTURE_IDEAL_FPS: u32 = 30;

// Reduced-size image dimensions for face detection
const SCALE_FACTOR: f64 = 0.5_f64;
const SCALE_FACTOR_INV: i32 = (1f64 / SCALE_FACTOR) as i32;

const HISTOGRAM_WIDTH: i32 = 640;
const HISTOGRAM_HEIGHT: i32 = 100;

const STFT_WINDOW_SIZE: usize = 4 * (HISTOGRAM_WIDTH as usize);
const STFT_STEP_SIZE: usize = 15;

const IOU_THRESHOLD: f32 = 0.8;

// TODO: Start with a rectangle in the center of the image as the initial guess for the face.
// TODO: Run haar cascades on another thread every once in a while. Update the region of interest.
// TODO: Sample-and-hold for the FFT.
// TODO: Run the FFT in a separate thread.

// TODO: Impractical but fun: Use optical flow to adjust the detected face until the next guess comes in.

// TODO: Heart Rate Variability (~55 to ~105 ms variability for teenagers, ~25 to ~45 for older people)
// TODO: Interbeat Interval (IBI)

fn main() {
    let palette = colorgrad::turbo();

    let mut classifier = CascadeClassifier::new(CASCADE_XML_FILE).unwrap();

    let window_type = WindowType::Hanning;
    let window_size: usize = STFT_WINDOW_SIZE;
    let step_size: usize = STFT_STEP_SIZE;
    let mut stft = STFT::new(window_type, window_size, step_size);

    let mut spectrogram_column: Vec<f64> = std::iter::repeat(0.).take(stft.output_size()).collect();
    // assert_eq!(spectrogram_column.len(), HISTOGRAM_WIDTH as _);

    // pre-fill
    stft.append_samples(&vec![0.0; STFT_WINDOW_SIZE]);

    let (notify_sender, new_frame_notification) = sync::mpsc::sync_channel(1);

    let index = CameraIndex::Index(0);
    let camera = CameraCapture::try_new_from_index(index, notify_sender).unwrap();

    let mut histogram_buffer = Mat::new_rows_cols_with_default(
        HISTOGRAM_HEIGHT,
        HISTOGRAM_WIDTH,
        CV_8UC3,
        Scalar::all(0.0),
    )
    .unwrap();

    let mut display_delay = Instant::now();
    let mut sample_delay = Instant::now();
    let mut sample = 0.0;
    let mut last_region = Rect::default();

    let mut min_hist = 0.0;
    let mut max_hist = 0.0;

    let mut last_detection = None;

    let color_red = Scalar::new(0.0, 0.0, 255.0, -1.0);
    let color_green = Scalar::new(0.0, 255.0, 0.0, -1.0);

    while let Ok(FrameNotification::NewFrame { fps }) = new_frame_notification.recv() {
        println!("Camera FPS: {fps}");

        let mut bgr_buffer = match camera.frame() {
            None => {
                eprintln!("Got an empty buffer");
                continue;
            }
            Some(frame) => frame,
        };

        let preprocessed = preprocess_image(&bgr_buffer).unwrap();
        let faces = detect_faces(&mut classifier, preprocessed).unwrap();

        let mut color = color_red;
        if let Some(region) = faces.into_iter().next() {
            last_detection = Some(region);
            color = color_green;
        }

        if let Some(region) = last_detection {
            let iou = calculate_iou(&region, &last_region);
            if iou < IOU_THRESHOLD {
                last_region = region.clone();
            }

            let rect = draw_box_around_face(&mut bgr_buffer, &last_region, &color).unwrap();
            let roi = Mat::roi(&bgr_buffer, rect).unwrap();
            let mean = mean(&roi, &no_array()).unwrap();

            // Sample and hold.
            sample = mean[1] / 255.0;
        }

        let tbf = Instant::now() - sample_delay;
        if tbf.as_millis() >= 60 {
            let fps = 1.0 / tbf.as_secs_f64();
            sample_delay = Instant::now();
            println!("x FPS: {} ({:?})", fps, tbf);

            stft.append_samples(&[sample]);

            while stft.contains_enough_to_compute() {
                stft.compute_column(&mut spectrogram_column[..]);
                stft.move_to_next_column();

                let row = histogram_buffer
                    .at_row_mut::<Vec3b>(HISTOGRAM_HEIGHT - 1)
                    .unwrap();

                let mut min = f64::MAX;
                let mut max = f64::MIN;

                for &value in &spectrogram_column {
                    min = min.min(value);
                    max = max.max(value);
                }

                if max <= 1e-4 {
                    max = 1.0;
                }

                min = min * 0.9 + min_hist * 0.1;
                max = max * 0.9 + max_hist * 0.1;

                min_hist = min;
                max_hist = max;

                println!("min={min}, max={max}");

                const SKIP: usize = 10;
                const N: usize = HISTOGRAM_WIDTH as usize - SKIP;
                for (i, &value) in spectrogram_column.iter().skip(SKIP).take(N).enumerate() {
                    let sample = ((value - min) / max + min).min(1.0);
                    let color = palette.at(sample);
                    let rgb = color.to_rgba8();

                    let j = ((i as f64) / (N as f64) * (HISTOGRAM_WIDTH as f64)) as usize;
                    row[j] = Vec3b::from_array([rgb[2], rgb[1], rgb[0]]);
                }
            }

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

        let mut histogram_region = Mat::roi(
            &bgr_buffer,
            Rect::new(
                0,
                bgr_buffer.rows() - HISTOGRAM_HEIGHT,
                HISTOGRAM_WIDTH,
                HISTOGRAM_HEIGHT,
            ),
        )
        .unwrap();

        histogram_buffer.copy_to(&mut histogram_region).unwrap();

        if Instant::now() - display_delay > Duration::from_millis(30) {
            display_delay = Instant::now();

            imshow("Image", &bgr_buffer).unwrap();
            if let Ok(KEY_CODE_ESCAPE) = wait_key(1) {
                break;
            }
        }
    }
}

fn calculate_iou(r1: &Rect, r2: &Rect) -> f32 {
    let x_overlap = f32::max(
        0.0,
        f32::min((r1.x + r1.width) as f32, (r2.x + r2.width) as f32)
            - f32::max(r1.x as f32, r2.x as f32),
    );
    let y_overlap = f32::max(
        0.0,
        f32::min((r1.y + r1.height) as f32, (r2.y + r2.height) as f32)
            - f32::max(r1.y as f32, r2.y as f32),
    );
    let intersection_area = x_overlap * y_overlap;
    let union_area =
        r1.width as f32 * r1.height as f32 + r2.width as f32 * r2.height as f32 - intersection_area;

    intersection_area / union_area
}

fn get_camera(index: CameraIndex) -> Result<Camera, NokhwaError> {
    // TODO: Use threaded camera
    let format = CameraFormat::new(
        Resolution::new(CAMERA_CAPTURE_WIDTH, CAMERA_CAPTURE_HEIGHT),
        FrameFormat::MJPEG,
        CAMERA_CAPTURE_IDEAL_FPS,
    );
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(format));
    let mut camera = Camera::new(index, requested)?;
    camera.open_stream()?;
    Ok(camera)
}

fn prepare_buffers(camera: &mut Camera) -> (Mat, Mat) {
    let resolution = camera.resolution();
    let capture_buffer =
        unsafe { Mat::new_rows_cols(resolution.height() as _, resolution.width() as _, CV_8UC3) }
            .unwrap();

    let bgr_buffer =
        unsafe { Mat::new_rows_cols(resolution.height() as _, resolution.width() as _, CV_8UC3) }
            .unwrap();
    (capture_buffer, bgr_buffer)
}

fn decode_to_bgr(frame: Buffer, capture_buffer: &mut Mat, dst: &mut Mat) {
    let mut decoded = frame.decode_image::<RgbFormat>().unwrap();

    unsafe {
        capture_buffer.set_data(decoded.as_mut_ptr());
    }

    cvt_color(capture_buffer, dst, COLOR_RGB2BGR, 3).unwrap();
}

fn preprocess_image(frame: &Mat) -> Result<Mat, opencv::Error> {
    let gray = convert_to_grayscale(frame)?;
    let reduced = reduce_image_size(&gray, SCALE_FACTOR)?;
    equalize_image(&reduced)
}

fn convert_to_grayscale(frame: &Mat) -> Result<Mat, opencv::Error> {
    let mut gray = Mat::default();
    cvt_color(frame, &mut gray, COLOR_BGR2GRAY, 0)?;
    Ok(gray)
}

fn reduce_image_size(image: &Mat, factor: f64) -> Result<Mat, opencv::Error> {
    // Destination size is determined by scaling `factor`, not by target size.
    const SIZE_AUTO: Size = Size {
        width: 0,
        height: 0,
    };
    let mut reduced = Mat::default();
    resize(
        image,
        &mut reduced,
        SIZE_AUTO,
        factor, // fx
        factor, // fy
        INTER_LINEAR,
    )?;
    Ok(reduced)
}

fn equalize_image(reduced: &Mat) -> Result<Mat, opencv::Error> {
    let mut equalized = Mat::default();
    equalize_hist(reduced, &mut equalized)?;
    Ok(equalized)
}

fn detect_faces(
    classifier: &mut CascadeClassifier,
    image: Mat,
) -> Result<VectorOfRect, opencv::Error> {
    const SCALE_FACTOR: f64 = 1.1;
    const MIN_NEIGHBORS: i32 = 2;
    const FLAGS: i32 = 0;
    const MIN_FACE_SIZE: Size = Size {
        width: 30,
        height: 30,
    };
    const MAX_FACE_SIZE: Size = Size {
        width: 0,
        height: 0,
    };

    let mut faces = VectorOfRect::new();
    classifier.detect_multi_scale(
        &image,
        &mut faces,
        SCALE_FACTOR,
        MIN_NEIGHBORS,
        FLAGS,
        MIN_FACE_SIZE,
        MAX_FACE_SIZE,
    )?;
    Ok(faces)
}

fn draw_box_around_face(
    frame: &mut Mat,
    face: &Rect,
    color: &Scalar,
) -> Result<Rect, opencv::Error> {
    // Trim the face width down a bit.
    let scaled_width = ((face.width * SCALE_FACTOR_INV) as f32 * 0.8) as _;
    let width_delta = (face.width * SCALE_FACTOR_INV) - scaled_width;
    let half_width_delta = width_delta / 2;

    let scaled_face = Rect {
        x: face.x * SCALE_FACTOR_INV + half_width_delta,
        y: face.y * SCALE_FACTOR_INV,
        width: scaled_width,
        height: face.height * SCALE_FACTOR_INV,
    };

    const THICKNESS: i32 = 2;
    const LINE_TYPE: i32 = 8;
    const SHIFT: i32 = 0;

    rectangle(
        frame,
        scaled_face,
        color.clone(),
        THICKNESS,
        LINE_TYPE,
        SHIFT,
    )?;
    Ok(scaled_face)
}
