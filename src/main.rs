use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use lazy_static::lazy_static;
use log::{debug, error, info, warn};
use opencv::core::{Mat, Rect};
use simple_moving_average::SMA;

use crate::ml::Detection;
use crate::zoneminder::db::Bounding;
use crate::zoneminder::MonitorTrait;

mod instrumentation;
mod ml;
mod zoneminder;

// TODO: Heed analysis images setting in ZM and generate those from within zm-aidect (sparsely, only for frames actually analyzed, not sure if the DB schema allows for that)

#[derive(Parser, Debug)]
#[clap(disable_help_subcommand = true)]
struct Args {
    #[clap(
        long,
        short = 'v',
        parse(from_occurrences),
        global = true,
        help = "Increase log verbosity (stacks up to -vvv)"
    )]
    verbose: usize,

    #[clap(subcommand)]
    mode: Mode,
}

#[derive(Subcommand, Debug)]
enum Mode {
    Run {
        /// Zoneminder monitor ID
        #[clap(value_parser)]
        monitor_id: u32,
        // TODO: instrumentation listen address, base port, some way to disable it (=> no default?)
    },
    Test {
        /// Zoneminder monitor ID
        #[clap(value_parser)]
        monitor_id: u32,
    },
    Event {
        /// Zoneminder event ID to check for detections
        #[clap(value_parser)]
        event: u64,
    },
    Bench {
        /// Zoneminder monitor ID
        #[clap(value_parser)]
        monitor_id: i32,

        /// Image files to use
        #[clap(value_parser, required = true)]
        images: Vec<PathBuf>,
    },
}

fn main() -> Result<()> {
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

    let args: Args = Args::parse();
    stderrlog::new()
        .module(module_path!())
        .verbosity(args.verbose + 1)
        .timestamp(stderrlog::Timestamp::Off)
        .init()
        .unwrap();

    match args.mode {
        Mode::Run { monitor_id } => run(monitor_id),
        Mode::Test { monitor_id } => test(monitor_id),
        _ => panic!("Not implemented"),
    }
}

struct MonitorContext<'zm_conf> {
    zm_conf: &'zm_conf zoneminder::ZoneMinderConf,
    monitor: zoneminder::Monitor<'zm_conf>,
    zone_config: zoneminder::db::ZoneConfig,
    monitor_settings: zoneminder::db::MonitorSettings,
    bounding_box: Rect,
    yolo: ml::YoloV4Tiny,
    max_fps: f32,
    monitor_id: u32,
    trigger_id: u32,
}

fn connect_zm(monitor_id: u32, zm_conf: &zoneminder::ZoneMinderConf) -> Result<MonitorContext> {
    let monitor = zoneminder::Monitor::connect(zm_conf, monitor_id)?;
    let zone_config = zoneminder::db::ZoneConfig::get_zone_config(zm_conf, monitor_id)?;
    let monitor_settings = zoneminder::db::MonitorSettings::query(zm_conf, monitor_id)?;

    info!(
        "{}: Picked up zone configuration: {:?}",
        monitor_id, zone_config
    );

    let bounding_box = zone_config.shape.bounding_box();
    info!("{}: Picked up zone bounds {:?}", monitor_id, bounding_box);

    let max_fps = monitor_settings.analysis_fps_limit;
    let max_fps = zone_config.fps.map(|v| v as f32).unwrap_or(max_fps);

    let trigger_id = zone_config.trigger.unwrap_or(monitor_id);

    let yolo = ml::YoloV4Tiny::new(
        zone_config.threshold.unwrap_or(0.5),
        zone_config.size.unwrap_or(256),
        false,
    )?;

    Ok(MonitorContext {
        zm_conf, monitor, zone_config, monitor_settings, bounding_box, yolo, max_fps, monitor_id, trigger_id
    })
}

struct Inferred {
    duration: Duration,
    detections: Vec<Detection>,
}

fn infer(image: zoneminder::Image, bounding_box: Rect, zone_config: &zoneminder::db::ZoneConfig, yolo: &mut ml::YoloV4Tiny) -> Result<Inferred> {
    let image = image.convert_to_rgb24()?;
    // TODO: blank remaining area outside zone polygon
    let image = Mat::roi(&image, bounding_box)?;

    let start = Instant::now();
    let detections = yolo.infer(&image)?;
    let duration = start.elapsed();

    let detections: Vec<Detection> = detections
        .iter()
        .filter(|d| CLASSES.contains_key(&d.class_id))
        .filter(|d| {
            (d.bounding_box.width * d.bounding_box.height) as u32
                > zone_config.min_area.unwrap_or(0)
        })
        .cloned()
        .collect();

    Ok(Inferred { duration, detections })
}

fn trigger(ctx: &MonitorContext, description: &str, score: u32) -> Result<u64> {
    Ok(if ctx.trigger_id != ctx.monitor_id {
        let trigger_monitor = zoneminder::Monitor::connect(&ctx.zm_conf, ctx.trigger_id)?;
        trigger_monitor.trigger("aidect", description, score).with_context(|| format!("Failed to trigger monitor ID {}", ctx.trigger_id))?
    } else {
        ctx.monitor.trigger("aidect", description, score).with_context(|| "Failed to trigger event")?
    })
}

fn test(monitor_id: u32) -> Result<()> {
    let zm_conf = zoneminder::ZoneMinderConf::parse_default()?;
    let mut ctx = connect_zm(monitor_id, &zm_conf)?;

    println!("Connected to monitor ID {}: {}", monitor_id, ctx.monitor_settings.name);

    let num_images = 3;
    println!("Grabbing {} images and running detection", num_images);
    for image in ctx.monitor.stream_images()?.take(num_images) {
        let result = infer(image?, ctx.bounding_box, &ctx.zone_config, &mut ctx.yolo)?;
        println!("Inference took {:?}, detections: {:#?}", result.duration, result.detections);
    }

    println!("Triggering an event on monitor {}", ctx.trigger_id);
    let event_id = trigger(&ctx, "zm-aidect test", 1)?;
    println!("Success, event ID is {}", event_id);

    Ok(())
}

lazy_static! {
    static ref CLASSES: HashMap<i32, &'static str> = [  // TODO this should be loaded at runtime from the model definition
        (1, "Human"),
        (3, "Car"),
        (15, "Bird"),
        (16, "Cat"),
        (17, "Dog"),
    ].into();
}

fn run(monitor_id: u32) -> Result<()> {
    let zm_conf = zoneminder::ZoneMinderConf::parse_default()?;
    let mut ctx = connect_zm(monitor_id, &zm_conf)?;

    instrumentation::spawn_prometheus_client(9000 + monitor_id as u16);


    let mut pacemaker = RealtimePacemaker::new(ctx.max_fps);
    let mut event_tracker = coalescing::EventTracker::new();

    // watchdog is set to 20x max_fps frame interval
    let watchdog = ThreadedWatchdog::new(Duration::from_secs_f32(20.0 / ctx.max_fps));

    let process_update_event = |update: Option<coalescing::UpdateEvent>| {
        if let Some(update) = update {
            let description = describe(&CLASSES, &ctx.bounding_box, &update.detection);
            if let Err(e) =
                zoneminder::db::update_event_notes(&ctx.zm_conf, update.event_id, &description)
            {
                error!(
                    "{}: Failed to update event {} notes: {}",
                    ctx.trigger_id, update.event_id, e
                );
            }
        }
    };

    for image in ctx.monitor.stream_images()? {
        let Inferred { duration: inference_duration, detections } = infer(image?, ctx.bounding_box, &ctx.zone_config, &mut ctx.yolo)?;

        if detections.len() > 0 {
            debug!(
                "{}: Inference result (took {:?}): {:?}",
                monitor_id, inference_duration, detections
            );

            let d = detections
                .iter()
                .max_by_key(|d| (d.confidence * 1000.0) as u32)
                .unwrap(); // generally there will only be one anyway
            let score = (d.confidence * 100.0) as u32;
            let description = describe(&CLASSES, &ctx.bounding_box, &d);

            let event_id = trigger(&ctx, &description, score)?;
            let update = event_tracker.push_detection(d.clone(), event_id);
            process_update_event(update);
        }

        if ctx.monitor.is_idle()? {
            // Not recording any more, flush current event description if any
            let update = event_tracker.clear();
            if update.is_some() {
                debug!("Flushing event because idle");
            }
            process_update_event(update);
        }

        if inference_duration.as_secs_f32() > pacemaker.target_interval {
            warn!(
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
    use log::trace;

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
                    trace!("Flushing event {} -> {}", current_event.event_id, event_id);
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
            let current_event = self.current_event.take()?;
            let detection = current_event
                .detections
                .iter()
                .max_by_key(|d| (d.confidence * 1000.0) as u32)
                .unwrap();
            // TODO: aggregate by classes, annotate counts.
            trace!(
                "Coalesce {} with {:?} to {:?}",
                current_event.event_id,
                current_event.detections,
                detection
            );
            Some(UpdateEvent {
                event_id: current_event.event_id,
                detection: detection.clone(),
            })
        }
    }
}

trait Pacemaker {
    fn tick(&mut self);
    fn current_frequency(&self) -> f32;
}

struct RealtimePacemaker {
    target_interval: f32,
    last_tick: Option<Instant>,
    avg: simple_moving_average::NoSumSMA<f32, f32, 10>,
    current_frequency: f32,
}

impl RealtimePacemaker {
    fn new(frequency: f32) -> RealtimePacemaker {
        RealtimePacemaker {
            target_interval: 1.0f32 / frequency,
            last_tick: None,
            avg: simple_moving_average::NoSumSMA::new(),
            current_frequency: 0.0,
        }
    }
}

impl Pacemaker for RealtimePacemaker {
    fn tick(&mut self) {
        if let Some(last_iteration) = self.last_tick {
            let now = Instant::now();
            let frame_duration = (now - last_iteration).as_secs_f32(); // how long the paced workload ran
                                                                       // smoothing using moving average
            self.avg.add_sample(frame_duration);
            let average_duration = self.avg.get_average();

            let sleep_duration = self.target_interval - average_duration;
            if sleep_duration > 0.0 {
                std::thread::sleep(Duration::from_secs_f32(sleep_duration));
            }

            // calculate current frequency from the tick interval (workload + sleeping)
            let tick_interval = Instant::now() - last_iteration;
            self.current_frequency = 1.0f32 / tick_interval.as_secs_f32();
        }
        self.last_tick = Some(Instant::now());
    }

    fn current_frequency(&self) -> f32 {
        self.current_frequency
    }
}

trait Watchdog {
    fn reset(&self) -> ();
}

struct ThreadedWatchdog {
    tx: mpsc::Sender<()>,
}

impl ThreadedWatchdog {
    fn new(timeout: Duration) -> ThreadedWatchdog {
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || loop {
            if let Err(mpsc::RecvTimeoutError::Timeout) = rx.recv_timeout(timeout) {
                error!("Watchdog expired, terminating.");
                std::process::exit(1);
            }
        });

        ThreadedWatchdog { tx }
    }
}

impl Watchdog for ThreadedWatchdog {
    fn reset(&self) -> () {
        self.tx.send(()).unwrap()
    }
}
