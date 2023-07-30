use nokhwa::pixel_format::{RgbAFormat, RgbFormat};
use nokhwa::utils::{
    ApiBackend, CameraFormat, ControlValueSetter, FrameFormat, KnownCameraControl, RequestedFormat,
    RequestedFormatType, Resolution,
};
use nokhwa::{nokhwa_initialize, Buffer, CallbackCamera, Camera, NokhwaError};
use opencv::core::{mean, no_array, Rect, Scalar, Size, Vec3b, CV_8UC3};
use opencv::highgui::{imshow, wait_key};
use opencv::imgproc::{
    cvt_color, equalize_hist, rectangle, resize, COLOR_BGR2GRAY, COLOR_RGB2BGR, INTER_LINEAR,
};
use opencv::objdetect::CascadeClassifier;
use opencv::prelude::*;
use opencv::types::VectorOfRect;
use std::ffi::c_void;
use std::time::{Duration, Instant};

const KEY_CODE_ESCAPE: i32 = 27;
const CASCADE_XML_FILE: &str = "haarcascade_frontalface_alt.xml";

const CAMERA_CAPTURE_WIDTH: u32 = 640;
const CAMERA_CAPTURE_HEIGHT: u32 = 480;
const CAMERA_CAPTURE_IDEAL_FPS: u32 = 30;

// Reduced-size image dimensions for face detection
const HAAR_SCALE_FACTOR: f64 = 0.5_f64;
const HAAR_SCALE_FACTOR_INV: i32 = (1f64 / HAAR_SCALE_FACTOR) as i32;

const HISTOGRAM_WIDTH: i32 = 640;
const HISTOGRAM_HEIGHT: i32 = 100;

fn main() {
    // Only needs to be run on OSX.
    nokhwa_initialize(|granted| {
        println!("Camera initialization succeeded: {}", granted);
    });

    let cameras = nokhwa::query(ApiBackend::Auto).unwrap();
    cameras.iter().for_each(|cam| println!("{:?}", cam));

    let mut classifier = CascadeClassifier::new(CASCADE_XML_FILE).unwrap();
    let camera = cameras.last().unwrap();

    let format = CameraFormat::new(
        Resolution::new(CAMERA_CAPTURE_WIDTH, CAMERA_CAPTURE_HEIGHT),
        FrameFormat::YUYV,
        CAMERA_CAPTURE_IDEAL_FPS,
    );

    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(format));

    let mut threaded = CallbackCamera::new(camera.index().clone(), format, |buffer| {
        // TODO: Use RGBA? Could be faster.
        /*let mut image = buffer.decode_image::<RgbFormat>().unwrap();
        let capture_buffer = unsafe {
            Mat::new_rows_cols_with_data(
                image.height() as _,
                image.width() as _,
                CV_8UC3,
                image.as_mut_ptr() as *mut c_void,
                0,
            )
        }
        .unwrap();
        */

        println!("Cough");

        // decode_to_bgr(buffer, &mut capture_buffer, &mut bgr_buffer);
    })
    .unwrap();

    threaded.open_stream().unwrap();

    let resolution = threaded.resolution().unwrap();
    let (mut capture_buffer, mut bgr_buffer) = prepare_buffers(&resolution).unwrap();

    let mut histogram_buffer = Mat::new_rows_cols_with_default(
        HISTOGRAM_HEIGHT,
        HISTOGRAM_WIDTH,
        CV_8UC3,
        Scalar::all(255.0),
    )
    .unwrap();

    let mut display_delay = Instant::now();
    let mut detect_delay = Instant::now();

    let mut face: Option<Rect> = None;

    loop {
        if face.is_none() {
            let preprocessed = preprocess_image(&bgr_buffer).unwrap();
            let faces = detect_faces(&mut classifier, preprocessed).unwrap();
            face = faces.into_iter().next();
        }

        if let Some(region) = face.clone() {
            let rect = draw_box_around_face(&mut bgr_buffer, region).unwrap();
            let roi = Mat::roi(&bgr_buffer, rect).unwrap();
            let mean = mean(&roi, &no_array()).unwrap();

            let fps = 1.0 / (Instant::now() - detect_delay).as_secs_f64();
            detect_delay = Instant::now();
            println!("FPS: {}", fps);

            let b = (mean[0] * (HISTOGRAM_HEIGHT as f64 / 255.0)) as i32;
            let g = (mean[1] * (HISTOGRAM_HEIGHT as f64 / 255.0)) as i32;
            let r = (mean[2] * (HISTOGRAM_HEIGHT as f64 / 255.0)) as i32;

            if let Ok(pixel) = histogram_buffer.at_2d_mut::<Vec3b>(HISTOGRAM_HEIGHT - b, 10) {
                pixel[0] = 255;
                pixel[1] = 0;
                pixel[2] = 0;
            }

            if let Ok(pixel) = histogram_buffer.at_2d_mut::<Vec3b>(HISTOGRAM_HEIGHT - g, 10) {
                pixel[0] = 0;
                pixel[1] = 255;
                pixel[2] = 0;
            }

            if let Ok(pixel) = histogram_buffer.at_2d_mut::<Vec3b>(HISTOGRAM_HEIGHT - r, 10) {
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 255;
            }
        }

        let src = Mat::roi(
            &histogram_buffer,
            Rect::new(0, 0, HISTOGRAM_WIDTH - 1, HISTOGRAM_HEIGHT),
        )
        .unwrap();
        let mut dst = Mat::roi(
            &histogram_buffer,
            Rect::new(1, 0, HISTOGRAM_WIDTH - 1, HISTOGRAM_HEIGHT),
        )
        .unwrap();
        src.copy_to(&mut dst).unwrap();

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

    println!("Stopping camera stream ...");
    threaded.stop_stream().unwrap();
    drop(threaded);

    println!("Bye.");
}

fn prepare_buffers(resolution: &Resolution) -> Result<(Mat, Mat), opencv::Error> {
    let capture_buffer = Mat::new_rows_cols_with_default(
        resolution.height() as _,
        resolution.width() as _,
        CV_8UC3,
        Scalar::default(),
    )?;
    let bgr_buffer = Mat::new_rows_cols_with_default(
        resolution.height() as _,
        resolution.width() as _,
        CV_8UC3,
        Scalar::default(),
    )
    .unwrap();
    Ok((capture_buffer, bgr_buffer))
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
    let reduced = reduce_image_size(&gray, HAAR_SCALE_FACTOR)?;
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

fn draw_box_around_face(frame: &mut Mat, face: Rect) -> Result<Rect, opencv::Error> {
    // Trim the face width down a bit.
    let scaled_width = ((face.width * HAAR_SCALE_FACTOR_INV) as f32 * 0.8) as _;
    let width_delta = (face.width * HAAR_SCALE_FACTOR_INV) - scaled_width;
    let half_width_delta = width_delta / 2;

    let scaled_face = Rect {
        x: face.x * HAAR_SCALE_FACTOR_INV + half_width_delta,
        y: face.y * HAAR_SCALE_FACTOR_INV,
        width: scaled_width,
        height: face.height * HAAR_SCALE_FACTOR_INV,
    };

    const THICKNESS: i32 = 2;
    const LINE_TYPE: i32 = 8;
    const SHIFT: i32 = 0;
    let color_red = Scalar::new(0f64, 0f64, 255f64, -1f64);

    rectangle(frame, scaled_face, color_red, THICKNESS, LINE_TYPE, SHIFT)?;
    Ok(scaled_face)
}
