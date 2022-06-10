use std::error::Error;
use std::time::{Duration, Instant, SystemTime};

use opencv::core::Mat;
use opencv::imgproc::{COLOR_RGBA2RGB, cvt_color};
use simple_moving_average::SMA;

mod ml;
mod zoneminder;

fn main() -> Result<(), Box<dyn Error>> {
    let mut yolo = ml::YoloV4Tiny::new(0.5, 256)?;

    /*
    // run on raw image
    let mut image_data = fs::read("imago_with_human.rgba")?;

    let image = unsafe {
        //let image_data = monitor.shared_images.add(image_size as usize * last_write_index as usize);
        let image_data = image_data.as_mut_ptr();
        let image_row_size = 1280 * 4;

        Mat::new_rows_cols_with_data(1280, 720, CV_8UC4, image_data as *mut c_void, image_row_size)?
    };

    let mut rgb_image = Mat::default();
    cvt_color(&image, &mut rgb_image, COLOR_RGBA2RGB, 0)?;

    //println!("Shape: {:?}", rgb_image);

    let t0 = Instant::now();
    let detections = yolo.infer(&rgb_image)?;
    let td = Instant::now() - t0;

    println!("Inference completed in {:?}:\n{:#?}",
             td, detections);

    */
    /*
    // run on pngs
    for file in vec![
        "19399-video-0001.png",
        "19399-video-0002.png",
        "19399-video-0003.png",
    ] {
        println!("Processing {}", file);
        let image = opencv::imgcodecs::imread(file, IMREAD_UNCHANGED)?;
        println!("{:#?}", image);

        let t0 = Instant::now();
        let detections = yolo.infer(&image)?;
        let td = Instant::now() - t0;
        println!("Inference completed in {:?}:\n{:#?}",
                 td, detections);
    }*/

    /* image should look like
    Mat {
        type: "CV_8UC3",
        flags: 1124024336,
        channels: 3,
        depth: "CV_8U",
        dims: 2,
        size: Size_ {
            width: 1280,
            height: 720,
        },
        rows: 720,
        cols: 1280,
        elem_size: 3,
        elem_size1: 1,
        total: 921600,
        is_continuous: true,
        is_submatrix: false,
    }
     */

    //run for real
    let mid = 5;
    let max_fps = 2.0;
    let monitor = zoneminder::Monitor::connect(mid)?;
    let mut last_read_index = monitor.image_buffer_count;

    //let image_size = unsafe { (*monitor.shared_data).imagesize };
    //println!("Image size is claimed to be: {}", image_size);

    let mut last_frame_completed: Option<Instant> = None;

    let mut delay_sma = simple_moving_average::NoSumSMA::<_, f32, 10>::new();

    loop {
        //println!("Last write time: {:?}", monitor.last_write_time());
        //sleep(Duration::from_millis(500));

        let state = monitor.read()?;

        let last_write_index = state.last_write_index();
        if last_write_index != last_read_index && last_write_index != monitor.image_buffer_count {
            //let timestamp = unsafe { *monitor.shared_timestamps.offset(last_write_index as isize) };
            //let timestamp = UNIX_EPOCH + Duration::from_secs(timestamp.tv_sec as u64) + Duration::from_micros(timestamp.tv_usec as u64);
            //println!("New image available at index {}, timestamp {:?}", last_write_index, timestamp);
            last_read_index = last_write_index;

            let image = monitor.read_image(state.last_image_token())?;

            let mut rgb_image = Mat::default();
            cvt_color(&image, &mut rgb_image, COLOR_RGBA2RGB, 0)?;

            let t0 = Instant::now();
            let detections = yolo.infer(&rgb_image)?;
            let t1 = Instant::now();
            let td = t1 - t0;
            println!("{:?}: Inference completed in {:?}: {:#?}", SystemTime::now(), td, detections);

            if let Some(last_frame_completed) = last_frame_completed {
                let real_interval = (t1 - last_frame_completed).as_secs_f32();
                let target_interval = 1.0f32 / max_fps;
                let delta = target_interval - real_interval;
                delay_sma.add_sample(delta);

                let sleep_time = delay_sma.get_average();
                if sleep_time > 0.0 {
                    std::thread::sleep(Duration::from_secs_f32(sleep_time));
                } else {
                    eprintln!("Cannot keep up with max-analysis-fps (inference taking {:?})!", td);
                }
            }
            last_frame_completed = Some(Instant::now());

            //std::fs::write(format!("/tmp/imago-{:?}", timestamp), image_data)?;
        }
    }
    Ok(())
}
