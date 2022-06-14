use std::collections::HashMap;
use std::error::Error;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use opencv::core::{Mat, Rect};
use simple_moving_average::SMA;

mod instrumentation;
mod ml;
mod zoneminder;

use ml::Detection;
use zoneminder::{Bounding, MonitorTrait};

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
    let zone_config = zoneminder::ZoneConfig::get_zone_config(&zm_conf, monitor_id)?;

    instrumentation::spawn_prometheus_client(9000 + monitor_id as u16);

    eprintln!(
        "{}: Picked up zone configuration: {:?}",
        monitor_id, zone_config
    );

    let bounding_box = zone_config.shape.bounding_box();
    eprintln!("{}: Picked up zone bounds {:?}", monitor_id, bounding_box);

    let mut yolo = ml::YoloV4Tiny::new(
        zone_config.threshold.unwrap_or(0.5),
        zone_config.size.unwrap_or(256),
        false,
    )?;

    let max_fps = monitor.get_max_analysis_fps()?;
    let max_fps = zone_config.fps.map(|v| v as f32).unwrap_or(max_fps);
    let mut pacemaker = Pacemaker::new(max_fps);

    // watchdog is set to 20x max_fps frame interval
    let watchdog = Watchdog::new(Duration::from_secs_f32(20.0 / max_fps));

    let classes: HashMap<i32, &str> = [
        (1, "Human"), // person
        (3, "Car"),   // car
        (15, "Bird"), // bird
        (16, "Cat"),  // cat
        (17, "Dog"),  // dog
    ]
    .into();

    let trigger_id = zone_config.trigger.unwrap_or(monitor_id);
    let mut event_tracker = coalescing::EventTracker::new();

    let process_update_event = |update: Option<coalescing::UpdateEvent>| {
        if let Some(update) = update {
            let description = describe(&classes, &bounding_box, &update.detection);
            if let Err(e) = zoneminder::update_event_notes(&zm_conf, update.event_id, &description)
            {
                eprintln!(
                    "{}: Failed to update event {} notes: {}",
                    trigger_id, update.event_id, e
                );
            }
        }
    };

    for image in monitor.stream_images()? {
        let image = image?;
        // TODO: blank remaining area outside zone polygon
        let image = Mat::roi(&image, bounding_box)?;

        let inference_start = Instant::now();
        let detections = yolo.infer(&image)?;
        let inference_duration = inference_start.elapsed();

        let detections: Vec<&Detection> = detections
            .iter()
            .filter(|d| classes.contains_key(&d.class_id))
            .filter(|d| {
                (d.bounding_box.width * d.bounding_box.height) as u32
                    > zone_config.min_area.unwrap_or(0)
            })
            .collect();

        if detections.len() > 0 {
            println!(
                "{}: Inference result (took {:?}): {:?}",
                monitor_id, inference_duration, detections
            );

            let &d = detections
                .iter()
                .max_by_key(|d| (d.confidence * 1000.0) as u32)
                .unwrap(); // generally there will only be one anyway
            let score = (d.confidence * 100.0) as u32;
            let description = describe(&classes, &bounding_box, &d);

            let event_id = if trigger_id != monitor_id {
                let trigger_monitor = zoneminder::Monitor::connect(&zm_conf, trigger_id)?;
                trigger_monitor.trigger("aidect", &description, score)?
            } else {
                monitor.trigger("aidect", &description, score)?
            };
            let update = event_tracker.push_detection(d.clone(), event_id);
            process_update_event(update);
        }

        if monitor.is_idle()? {
            // Not recording any more, flush current event description if any
            let update = event_tracker.clear();
            process_update_event(update);
        }

        if inference_duration.as_secs_f32() > pacemaker.target_interval {
            eprintln!(
                "{}: Cannot keep up with max-analysis-fps (inference taking {:?})!",
                monitor_id, inference_duration,
            );
        }

        instrumentation::INFERENCE_DURATION.observe(inference_duration.as_secs_f64());
        instrumentation::INFERENCES.inc();

        pacemaker.tick();
        watchdog.reset();
        instrumentation::FPS.set(pacemaker.current_frequency() as f64);
    }
    Ok(())
}

fn describe(classes: &HashMap<i32, &str>, bounding_box: &Rect, d: &Detection) -> String {
    format!(
        "{} ({:.1}%) {}x{} (={}) at {}x{}",
        classes[&d.class_id],
        d.confidence * 100.0,
        d.bounding_box.width,
        d.bounding_box.height,
        d.bounding_box.width * d.bounding_box.height,
        d.bounding_box.x + bounding_box.x,
        d.bounding_box.y + bounding_box.y,
    )
}

mod coalescing {
    use crate::ml::Detection;

    struct TrackedEvent {
        event_id: u64,
        detections: Vec<Detection>,
    }

    pub struct UpdateEvent {
        pub event_id: u64,
        pub detection: Detection,
    }

    pub struct EventTracker {
        current_event: Option<TrackedEvent>,
    }

    impl EventTracker {
        pub fn new() -> EventTracker {
            EventTracker {
                current_event: None,
            }
        }

        pub fn push_detection(&mut self, d: Detection, event_id: u64) -> Option<UpdateEvent> {
            let mut update = None;
            if let Some(current_event) = self.current_event.as_mut() {
                if current_event.event_id != event_id {
                    update = self.clear();
                } else {
                    current_event.detections.push(d);
                    return None;
                }
            }
            self.current_event = Some(TrackedEvent {
                event_id,
                detections: vec![d],
            });
            update
        }

        pub fn clear(&mut self) -> Option<UpdateEvent> {
            if self.current_event.is_none() {
                return None;
            }
            let current_event = self.current_event.take().unwrap();
            let detection = current_event
                .detections
                .iter()
                .max_by_key(|d| (d.confidence * 1000.0) as u32)
                .unwrap();
            // TODO: aggregate by classes, annotate counts.
            Some(UpdateEvent {
                event_id: current_event.event_id,
                detection: detection.clone(),
            })
        }
    }
}

struct Pacemaker {
    target_interval: f32,
    last_tick: Option<Instant>,
    avg: simple_moving_average::NoSumSMA<f32, f32, 10>,
    current_frequency: f32,
}

impl Pacemaker {
    fn new(frequency: f32) -> Pacemaker {
        Pacemaker {
            target_interval: 1.0f32 / frequency,
            last_tick: None,
            avg: simple_moving_average::NoSumSMA::new(),
            current_frequency: 0.0,
        }
    }

    fn tick(&mut self) {
        let now = Instant::now();
        if let Some(last_iteration) = self.last_tick {
            let real_interval = (now - last_iteration).as_secs_f32();
            let delta = self.target_interval - real_interval;
            self.avg.add_sample(delta);
            self.current_frequency = 1.0f32 / real_interval;

            let sleep_time = self.avg.get_average();
            if sleep_time > 0.0 {
                std::thread::sleep(Duration::from_secs_f32(sleep_time));
            }
        }
        self.last_tick = Some(now);
    }

    fn current_frequency(&self) -> f32 {
        self.current_frequency
    }
}

struct Watchdog {
    tx: mpsc::Sender<()>
}

impl Watchdog {
    fn new(timeout: Duration) -> Watchdog {
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            loop {
                if let Err(mpsc::RecvTimeoutError::Timeout) = rx.recv_timeout(timeout) {
                    eprintln!("Watchdog expired, terminating.");
                    std::process::exit(1);
                }
            }
        });

        Watchdog { tx }
    }

    fn reset(&self) -> () {
        self.tx.send(()).unwrap()
    }
}
