use opencv::core::{Rect, Size};
use opencv::imgproc::{cvt_color, equalize_hist, resize, COLOR_BGR2GRAY, INTER_LINEAR};
use opencv::objdetect::CascadeClassifier;
use opencv::prelude::*;
use opencv::types::VectorOfRect;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

pub struct FaceDetector {
    image: Arc<Mutex<Option<Mat>>>,
    need_new: Arc<AtomicBool>,
    face: Arc<Mutex<Option<(bool, Rect)>>>,
}

impl FaceDetector {
    // Reduced-size image dimensions for face detection
    const SCALE_FACTOR: f64 = 0.5_f64;
    const SCALE_FACTOR_INV: i32 = (1f64 / Self::SCALE_FACTOR) as i32;

    const IOU_THRESHOLD: f32 = 0.8;

    // TODO: Refactor into function returning Result
    pub fn new(xml_file: &PathBuf) -> Self {
        let str = xml_file.to_str().unwrap();
        let classifier = CascadeClassifier::new(str).unwrap();
        let image = Arc::new(Mutex::new(None));
        let need_new = Arc::new(AtomicBool::new(true));
        let face = Arc::new(Mutex::new(None));

        thread::spawn({
            let image = image.clone();
            let need_new = need_new.clone();
            let face = face.clone();
            move || Self::run_thread(need_new, image, classifier, face)
        });

        Self {
            image,
            need_new,
            face,
        }
    }

    /// Provides a new frame.
    ///
    /// This will discard the current frame if the system has not calculated
    /// a new face yet.
    pub fn provide_image(&self, mat: &Mat) {
        let need_new = self
            .need_new
            .compare_exchange(true, false, Ordering::Acquire, Ordering::Relaxed)
            .map_or_else(|e| e, |v| v);

        if !need_new {
            return;
        }

        let mut image = self.image.lock().unwrap();
        *image = Some(mat.clone())
    }

    /// Obtains the current face prediction.
    pub fn face(&self) -> Option<(bool, Rect)> {
        let face = self.face.lock().unwrap();
        *face
    }

    fn run_thread(
        need_new: Arc<AtomicBool>,
        image: Arc<Mutex<Option<Mat>>>,
        mut classifier: CascadeClassifier,
        detected_face: Arc<Mutex<Option<(bool, Rect)>>>,
    ) {
        let mut last_detection = None;

        loop {
            let image = {
                let mut image_guard = image.lock().unwrap();
                image_guard.take()
            };
            need_new.store(true, Ordering::Release);

            if let Some(image) = image {
                let preprocessed = Self::preprocess_image(&image).unwrap();
                let face = Self::detect_face(&mut classifier, preprocessed).unwrap();

                if let Some(mut region) = face {
                    // Debounce detections.
                    let previous = match last_detection {
                        None => {
                            last_detection = Some(region);
                            region
                        }
                        Some(previous) => previous,
                    };

                    let iou = Self::calculate_iou(&region, &previous);
                    if iou < Self::IOU_THRESHOLD {
                        last_detection = Some(region);
                    } else {
                        // Ensure wee use the debounced value.
                        region = previous;
                    }

                    // Update detected face.
                    let mut detected_face = detected_face.lock().unwrap();
                    *detected_face = Some((true, region));
                } else {
                    // Mark detected face as stale.
                    let mut detected_face = detected_face.lock().unwrap();
                    *detected_face = last_detection.map(|face| (false, face));
                }
            }

            // TODO: Sleep until a new image has arrived.
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn preprocess_image(frame: &Mat) -> Result<Mat, opencv::Error> {
        let gray = Self::convert_to_grayscale(frame)?;
        let reduced = Self::reduce_image_size(&gray, Self::SCALE_FACTOR)?;
        Self::equalize_image(&reduced)
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

    fn detect_face(
        classifier: &mut CascadeClassifier,
        image: Mat,
    ) -> Result<Option<Rect>, opencv::Error> {
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

        if let Some(face) = faces.into_iter().next() {
            // Trim the face width down a bit.
            let scaled_width = ((face.width * Self::SCALE_FACTOR_INV) as f32 * 0.8) as _;
            let width_delta = (face.width * Self::SCALE_FACTOR_INV) - scaled_width;
            let half_width_delta = width_delta / 2;

            let scaled_face = Rect {
                x: face.x * Self::SCALE_FACTOR_INV + half_width_delta,
                y: face.y * Self::SCALE_FACTOR_INV,
                width: scaled_width,
                height: face.height * Self::SCALE_FACTOR_INV,
            };

            Ok(Some(scaled_face))
        } else {
            Ok(None)
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
        let union_area = r1.width as f32 * r1.height as f32 + r2.width as f32 * r2.height as f32
            - intersection_area;

        intersection_area / union_area
    }
}
