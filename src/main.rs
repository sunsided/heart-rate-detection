use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution};
use nokhwa::Camera;
use opencv::core::{Scalar, CV_8UC3};
use opencv::highgui::{imshow, wait_key};
use opencv::imgproc::{cvt_color, COLOR_RGB2BGR};
use opencv::prelude::*;

fn main() {
    let index = CameraIndex::Index(0);

    let requested =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

    let mut camera = Camera::new(index, requested).unwrap();

    camera.set_resolution(Resolution::new(640, 480)).unwrap();

    let resolution = camera.resolution();
    let mut capture_buffer =
        unsafe { Mat::new_rows_cols(resolution.height() as _, resolution.width() as _, CV_8UC3) }
            .unwrap();

    camera.open_stream().unwrap();

    loop {
        let frame = camera.frame().unwrap();
        let mut decoded = frame.decode_image::<RgbFormat>().unwrap();

        unsafe {
            capture_buffer.set_data(decoded.as_mut_ptr());
        }

        let mut converted = Mat::default();
        cvt_color(&capture_buffer, &mut converted, COLOR_RGB2BGR, 3).unwrap();

        imshow("Image", &converted).unwrap();
        let key = wait_key(1).unwrap();
        if key == 27 {
            break;
        }
    }
}
