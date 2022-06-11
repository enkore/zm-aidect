use std::collections::HashMap;
use std::error::Error;
use std::time::{Duration, Instant};

use opencv::core::Mat;
use opencv::imgproc::{cvt_color, COLOR_RGBA2RGB};
use simple_moving_average::SMA;

mod ml;
mod zoneminder;

use ml::Detection;
use zoneminder::Bounding;

fn main() -> Result<(), Box<dyn Error>> {
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

    //opencv::core::set_num_threads(1);

    // run on pngs
    /* {
        let mut yolo = ml::YoloV4Tiny::new(0.5, 256)?;
        for file in vec![
            "19399-video-0001.png",
            "19399-video-0002.png",
            "19399-video-0003.png",
        ] {
            println!("Processing {}", file);
            let image = opencv::imgcodecs::imread(file, opencv::imgcodecs::IMREAD_UNCHANGED)?;
            //println!("{:#?}", image);
            let mut detections = HashSet::new();

            // run once to ignore CUDA compilation
            yolo.infer(&image)?;

            let t0 = Instant::now();
            let n = 80;
            for _ in 0..n {
                detections.extend(yolo.infer(&image)?);
            }
            let td = Instant::now() - t0;
            println!("Inference completed in {:?}: {:#?}",
                     td / n, detections);
        }
        return Ok(());
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

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: zm-aidect MONITOR_ID");
        std::process::exit(1);
    }
    let monitor_id = args[1].trim().parse()?;
    let zm_conf = zoneminder::ZoneMinderConf::parse_default()?;

    let monitor = zoneminder::Monitor::connect(&zm_conf, monitor_id)?;

    eprintln!(
        "{}: Picked up zone configuration: {:?}",
        monitor_id, monitor.zone
    );

    let bounding_box = monitor.zone.shape.bounding_box();
    eprintln!("{}: Picked up zone bounds {:?}", monitor_id, bounding_box);

    let mut yolo = ml::YoloV4Tiny::new(
        monitor.zone.threshold.unwrap_or(0.5),
        monitor.zone.size.unwrap_or(256),
        false,
    )?;

    let max_fps = monitor.zone.fps
        .map(|v| v as f32)
        .unwrap_or(monitor.max_fps);
    let mut pacemaker = Pacemaker::new(max_fps);

    let classes: HashMap<i32, &str> = [
        (1, "Human"), // person
        (3, "Car"),   // car
        (15, "Bird"), // bird
        (16, "Cat"),  // cat
        (17, "Dog"),  // dog
    ].into();

    for image in monitor.stream_images() {
        let image = image?;
        // TODO: blank remaining area outside zone polygon
        let image = Mat::roi(&image, bounding_box)?;

        let mut rgb_image = Mat::default();
        cvt_color(&image, &mut rgb_image, COLOR_RGBA2RGB, 0)?;

        let t0 = Instant::now();
        let detections = yolo.infer(&rgb_image)?;
        let t1 = Instant::now();
        let td = t1 - t0;

        let detections: Vec<&Detection> = detections
            .iter()
            .filter(|d| classes.contains_key(&d.class_id))
            .filter(|d| {
                (d.bounding_box.width * d.bounding_box.height) as u32
                    > monitor.zone.min_area.unwrap_or(0)
            })
            .collect();

        if detections.len() > 0 {
            println!("{}: Inference result (took {:?}): {:?}", monitor_id, td, detections);

            let d = detections.iter().max_by_key(|d| (d.confidence * 1000.0) as u32).unwrap();  // generally there will only be one anyway
            let description = format!(
                "{} ({:.1}%) {}x{} (={}) at {}x{}",
                classes[&d.class_id],
                d.confidence * 100.0,
                d.bounding_box.width,
                d.bounding_box.height,
                d.bounding_box.width * d.bounding_box.height,
                d.bounding_box.x + bounding_box.x,
                d.bounding_box.y + bounding_box.y,
            );
            let trigger_id = monitor.zone.trigger.unwrap_or(monitor_id);
            if let Err(e) = zoneminder::zmtrigger::trigger_autocancel(trigger_id, "aidect", &description, 1) {
                eprintln!("{}: Failed to trigger zm: {}", monitor_id, e);
            }
        }

        if td.as_secs_f32() > pacemaker.target_interval {
            eprintln!(
                "{}: Cannot keep up with max-analysis-fps (inference taking {:?})!",
                monitor_id, td,
            );
        }

        pacemaker.tick();
    }
    Ok(())
}

struct Pacemaker {
    target_interval: f32,
    last_tick: Option<Instant>,
    avg: simple_moving_average::NoSumSMA<f32, f32, 10>,
}

impl Pacemaker {
    fn new(frequency: f32) -> Pacemaker {
        Pacemaker {
            target_interval: 1.0f32 / frequency,
            last_tick: None,
            avg: simple_moving_average::NoSumSMA::new(),
        }
    }

    fn tick(&mut self) {
        let now = Instant::now();
        if let Some(last_iteration) = self.last_tick {
            let real_interval = (now - last_iteration).as_secs_f32();
            let delta = self.target_interval - real_interval;
            self.avg.add_sample(delta);

            let sleep_time = self.avg.get_average();
            if sleep_time > 0.0 {
                std::thread::sleep(Duration::from_secs_f32(sleep_time));
            }
        }
        self.last_tick = Some(now);
    }
}
