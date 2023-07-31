use crate::fps_counter::FpsCounter;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
};
use nokhwa::{Camera, NokhwaError};
use opencv::core::{Scalar, CV_8UC3};
use opencv::imgproc::{cvt_color, COLOR_RGB2BGR};
use opencv::prelude::*;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{RecvError, SendError, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::{sync, thread};

pub struct CameraCapture {
    pub frame_rate: u32,
    pub resolution: Resolution,
    cancellation: Arc<AtomicBool>,
    handle: Option<JoinHandle<Result<(), ThreadError>>>,
    pub frame: Arc<Mutex<Cell<Option<Mat>>>>,
}

impl CameraCapture {
    pub const CAMERA_CAPTURE_WIDTH: u32 = 640;
    pub const CAMERA_CAPTURE_HEIGHT: u32 = 480;
    pub const CAMERA_CAPTURE_IDEAL_FPS: u32 = 30;

    pub fn try_new_from_index(
        index: CameraIndex,
        notify: SyncSender<FrameNotification>,
    ) -> Result<Self, CameraError> {
        nokhwa::nokhwa_initialize(|successful| {
            if !successful {
                panic!("Initialization failed.")
            }
        });

        let cancellation = Arc::new(AtomicBool::new(false));

        let frame = Arc::new(Mutex::new(Cell::new(None)));

        let (init_tx, init_rx) = sync::mpsc::sync_channel(0);
        let handle = thread::spawn({
            let cancellation = cancellation.clone();
            let frame = frame.clone();
            move || Self::run_thread(index, init_tx, cancellation, frame, notify)
        });

        let (frame_rate, resolution) = init_rx
            .recv()
            .map_err(|e| CameraError::InitChannelError(InitializationError::ReceiveFailed(e)))?;

        Ok(Self {
            frame_rate,
            resolution,
            cancellation,
            handle: Some(handle),
            frame,
        })
    }

    pub fn frame(&self) -> Option<Mat> {
        let lock = self.frame.lock().unwrap();
        lock.replace(None)
    }

    fn run_thread(
        index: CameraIndex,
        tx: SyncSender<(u32, Resolution)>,
        cancellation: Arc<AtomicBool>,
        target_buffer: Arc<Mutex<Cell<Option<Mat>>>>,
        notify: SyncSender<FrameNotification>,
    ) -> Result<(), ThreadError> {
        let format = CameraFormat::new(
            Resolution::new(Self::CAMERA_CAPTURE_WIDTH, Self::CAMERA_CAPTURE_HEIGHT),
            FrameFormat::MJPEG,
            Self::CAMERA_CAPTURE_IDEAL_FPS,
        );

        // TODO: Try to use RGBA format
        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(format));
        let mut camera = Camera::new(index, requested)
            .map_err(|e| ThreadError::CameraInitializationFailed(e))?;
        camera
            .open_stream()
            .map_err(|e| ThreadError::CameraInitializationFailed(e))?;

        let frame_rate = camera.frame_rate();
        let resolution = camera.resolution();

        if let Err(e) = tx.send((frame_rate, resolution)) {
            return Err(ThreadError::InitChannelError(e));
        }

        // The buffer used to wrap the camera frame in. Simply refers to the last data pointer.
        // TODO: Use CV_8UC4 with RGBA capture?
        let mut capture_buffer = Mat::new_rows_cols_with_default(
            resolution.height() as _,
            resolution.width() as _,
            CV_8UC3,
            Scalar::default(),
        )
        .map_err(|e| ThreadError::OpenCvError(e))?;

        let mut fps = FpsCounter::new();
        while !cancellation.load(Ordering::Acquire) {
            let frame = camera
                .frame()
                .map_err(|e| ThreadError::CameraCaptureFailed(e))?;

            // Decode into the capture buffer.
            let mut dst = capture_buffer
                .data_bytes_mut()
                .map_err(|e| ThreadError::OpenCvError(e))?;
            frame
                .decode_image_to_buffer::<RgbFormat>(&mut dst)
                .map_err(|e| ThreadError::CameraDecodeFailed(e))?;

            // Swap color planes.
            let mut bgr_buffer = Mat::new_rows_cols_with_default(
                resolution.height() as _,
                resolution.width() as _,
                CV_8UC3,
                Scalar::default(),
            )
            .map_err(|e| ThreadError::OpenCvError(e))?;
            // TODO: Account for number of channels (RGBA).
            cvt_color(&capture_buffer, &mut bgr_buffer, COLOR_RGB2BGR, 3)
                .map_err(|e| ThreadError::OpenCvError(e))?;

            let target_buffer = target_buffer.lock().map_err(|_e| ThreadError::LockError)?;

            // Swap the buffer and notify.
            target_buffer.set(Some(bgr_buffer));

            let fps = fps.tick();
            notify
                .try_send(FrameNotification::NewFrame { fps: fps as _ })
                .ok();
        }

        notify.try_send(FrameNotification::Stopped).ok();
        Ok(())
    }
}

impl Drop for CameraCapture {
    fn drop(&mut self) {
        self.cancellation.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            handle.join().expect("join failed").ok();
        }
    }
}

pub enum FrameNotification {
    NewFrame { fps: f32 },
    Stopped,
}

#[derive(Debug, thiserror::Error)]
pub enum CameraError {
    #[error(transparent)]
    CameraInitializationFailed(#[from] NokhwaError),
    #[error(transparent)]
    InitChannelError(#[from] InitializationError),
}

#[derive(Debug, thiserror::Error)]
pub enum InitializationError {
    #[error(transparent)]
    ReceiveFailed(#[from] RecvError),
    #[error(transparent)]
    SendFailed(#[from] SendError<(u32, Resolution)>),
}

#[derive(Debug, thiserror::Error)]
enum ThreadError {
    #[error(transparent)]
    CameraInitializationFailed(NokhwaError),
    #[error(transparent)]
    CameraCaptureFailed(NokhwaError),
    #[error(transparent)]
    CameraDecodeFailed(NokhwaError),
    #[error(transparent)]
    InitChannelError(#[from] SendError<(u32, Resolution)>),
    #[error(transparent)]
    OpenCvError(#[from] opencv::Error),
    #[error("Lock Guard poisoned")]
    LockError,
}
