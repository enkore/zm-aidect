use std::collections::HashMap;
use std::env;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use lazy_static::lazy_static;
use log::{debug, error, info, warn};
use opencv::core::{Mat, MatTraitConst, Rect};
use simple_moving_average::SMA;

use crate::ml::Detection;
use crate::zoneminder::db::Bounding;
use crate::zoneminder::{MonitorTrait};

mod instrumentation;
mod ml;
mod vio;
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
        #[clap(long)]
        instrumentation_address: Option<String>,
        #[clap(long, default_value_t = 9000)]
        instrumentation_port: u16,
    },
    Test {
        /// Zoneminder monitor ID
        #[clap(value_parser)]
        monitor_id: u32,
    },
    Event {
        /// Zoneminder event ID to check for detections
        #[clap(value_parser)]
        event_id: u64,

        /// Zoneminder monitor ID for the zone configuration
        #[clap(long, short = 'm')]
        monitor_id: Option<u32>,
    },
}

fn main() -> Result<()> {
    env::set_current_dir(env::current_exe()?.parent().unwrap())?;

    let args: Args = Args::parse();
    stderrlog::new()
        .module(module_path!())
        .verbosity(args.verbose + 1)
        .timestamp(stderrlog::Timestamp::Off)
        .init()
        .unwrap();

    match args.mode {
        Mode::Run { monitor_id, instrumentation_address, instrumentation_port } => run(monitor_id, instrumentation_address, instrumentation_port),
        Mode::Test { monitor_id } => test(monitor_id),
        Mode::Event {
            event_id,
            monitor_id,
        } => event(event_id, monitor_id),
    }
}

fn event(event_id: u64, monitor_id: Option<u32>) -> Result<()> {
    let zm_conf = zoneminder::ZoneMinderConf::parse_default()?;
    let event = zoneminder::db::Event::query(&zm_conf, event_id)?;
    let monitor_id = monitor_id.unwrap_or(event.monitor_id);
    let mut ctx = connect_zm(monitor_id, &zm_conf)?; // TODO: If this errors on "Error: No aidect zone found for monitor 6", suggest --monitor-id

    let video_path = event.video_path()?;
    println!("Analyzing video file {}", video_path.display());
    let props = vio::properties(&video_path)?;

    if props.width != ctx.monitor_settings.width || props.height != ctx.monitor_settings.height {
        println!("Note: Recording is from a different (higher?) resolution, so performance is not indicative due to rescaling");
    }

    println!("Note: Timestamps [mm:ss:ts] are at best a rough approximation.");
    println!("Note: Because analysis start frames aren't aligned between what zm-aidect might have originally done,");
    println!("      and this run, results can and will differ."); // TODO: This can be a good thing of course, but maybe add a way to analyse the logged alarm frames only or something like that

    let mut inference_durations = vec![];
    let mut videotime = Duration::default(); // EXTREMELY approximate
    let timestep = Duration::from_secs_f32(1f32 / ctx.max_fps); // video people are crying at this
    for image in vio::stream_file(
        &video_path,
        ctx.monitor_settings.width,
        ctx.monitor_settings.height,
        ctx.max_fps,
    )? {
        let result = infer(image, ctx.bounding_box, &ctx.zone_config, &mut ctx.yolo)?;
        if result.detections.len() > 0 {
            // TODO: How could we get the actual frame number or timestamp here?

            let ts = videotime.as_secs_f32();
            let frac = (ts.fract() * 1000f32) as u32;
            let seconds = ts.trunc() as u32;
            let secs = seconds % 60;
            let mins = seconds / 60;

            let description: Vec<String> = result
                .detections
                .iter()
                .map(|d| describe(&CLASSES, &d))
                .collect();
            println!(
                "[{:02}:{:02}:{:03}] Inference took {:?}: {}",
                mins,
                secs,
                frac,
                result.duration,
                description.join(", ")
            );
        }
        inference_durations.push(result.duration);
        videotime += timestep;
    }

    let total_duration = inference_durations.iter().sum::<Duration>();
    println!(
        "Processed {} frames, total ML time {:?}, average time {:?}",
        inference_durations.len(),
        total_duration,
        total_duration / inference_durations.len() as u32
    );

    Ok(())
}

struct MonitorContext<'zm_conf> {
    zm_conf: &'zm_conf zoneminder::ZoneMinderConf,
    monitor: zoneminder::Monitor<'zm_conf>,
    trigger_monitor: zoneminder::Monitor<'zm_conf>,
    zone_config: zoneminder::db::ZoneConfig,
    monitor_settings: zoneminder::db::MonitorSettings,
    bounding_box: Rect,
    yolo: ml::YoloV4Tiny,
    max_fps: f32,
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
    let max_fps = zone_config.fps.or(max_fps);
    let max_fps = max_fps.ok_or(anyhow!("No analysis FPS limit set - set either \"Analysis FPS\" in the Zoneminder web console, or set the FPS key in the aidect zone."))?;
    info!("{}: Setting maximum fps to {}", monitor_id, max_fps);

    let trigger_id = zone_config.trigger.unwrap_or(monitor_id);
    info!("{}: Connecting to trigger monitor {}", monitor_id, trigger_id);
    let trigger_monitor = zoneminder::Monitor::connect(zm_conf, trigger_id)?;

    let size = zone_config.size.unwrap_or(256);
    let threshold = zone_config.threshold.unwrap_or(0.5);
    let yolo = ml::YoloV4Tiny::new(
        threshold,
        size,
        false,
    )?;

    instrumentation::SIZE.set(size as f64);

    Ok(MonitorContext {
        zm_conf,
        monitor,
        trigger_monitor,
        zone_config,
        monitor_settings,
        bounding_box,
        yolo,
        max_fps,
    })
}

struct Inferred {
    duration: Duration,
    detections: Vec<Detection>,
}

fn infer(
    image: Mat,
    bounding_box: Rect,
    zone_config: &zoneminder::db::ZoneConfig,
    yolo: &mut ml::YoloV4Tiny,
) -> Result<Inferred> {
    assert_eq!(image.typ(), opencv::core::CV_8UC3);
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
        .map(|d| Detection {
            // Adjust bounding box to zone bounding box (RoI)
            bounding_box: Rect {
                x: d.bounding_box.x + bounding_box.x,
                y: d.bounding_box.y + bounding_box.y,
                ..d.bounding_box
            },
            ..*d
        })
        .collect();

    Ok(Inferred {
        duration,
        detections,
    })
}

fn trigger(ctx: &MonitorContext, description: &str, score: u32) -> Result<u64> {
    ctx.trigger_monitor
        .trigger("aidect", description, score)
        .with_context(|| format!("Failed to trigger monitor ID {}", ctx.trigger_monitor.id()))
}

fn test(monitor_id: u32) -> Result<()> {
    let zm_conf = zoneminder::ZoneMinderConf::parse_default()?;
    let mut ctx = connect_zm(monitor_id, &zm_conf)?;

    println!(
        "Connected to monitor ID {}: {}",
        monitor_id, ctx.monitor_settings.name
    );

    let num_images = 3;
    println!("Grabbing {} images and running detection", num_images);
    for image in ctx.monitor.stream_images()?.take(num_images) {
        let image = image?.convert_to_rgb24()?;
        let result = infer(image, ctx.bounding_box, &ctx.zone_config, &mut ctx.yolo)?;
        let description: Vec<String> = result
            .detections
            .iter()
            .map(|d| describe(&CLASSES, &d))
            .collect();
        println!(
            "Inference took {:?}: {}",
            result.duration,
            description.join(", ")
        );
    }

    println!("Triggering an event on monitor {}", ctx.trigger_monitor.id());
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

fn run(monitor_id: u32, instrumentation_address: Option<String>, instrumentation_port: u16) -> Result<()> {
    let zm_conf = zoneminder::ZoneMinderConf::parse_default()?;
    let mut ctx = connect_zm(monitor_id, &zm_conf)?;

    if let Some(address) = instrumentation_address {
        instrumentation::spawn_prometheus_client(address, instrumentation_port + monitor_id as u16);
    }

    let mut pacemaker = RealtimePacemaker::new(ctx.max_fps);
    let mut event_tracker = coalescing::EventTracker::new();

    // watchdog is set to 20x max_fps frame interval
    let watchdog = ThreadedWatchdog::new(Duration::from_secs_f32(20.0 / ctx.max_fps));

    fn process_update_event(ctx: &MonitorContext, update: Option<coalescing::UpdateEvent>) {
        if let Some(update) = update {
            let description = describe(&CLASSES, &update.detection);
            if let Err(e) =
                zoneminder::db::update_event_notes(&ctx.zm_conf, update.event_id, &description)
            {
                error!(
                    "{}: Failed to update event {} notes: {}",
                    ctx.trigger_monitor.id(), update.event_id, e
                );
            }
        }
    }

    // For yolov4-tiny and moderate input sizes, multithreading does speed things up, but at the expense
    // of higher overall CPU usage. As you would usually have multiple zm-aidect processes running, as
    // well as zmc, there is no particular need for a single zm-aidect process to scale to multiple cores,
    // especially when that comes with an efficiency hit. Large inputs and/or high framerates aren't
    // sensible on a CPU anyway.
    opencv::core::set_num_threads(1)?;

    for image in ctx.monitor.stream_images()? {
        let image = image?.convert_to_rgb24()?;
        let Inferred {
            duration: inference_duration,
            detections,
        } = infer(image, ctx.bounding_box, &ctx.zone_config, &mut ctx.yolo)?;

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
            let description = describe(&CLASSES, &d);

            let event_id =  trigger(&ctx, &description, score)?;
            let update = event_tracker.push_detection(d.clone(), event_id);
            process_update_event(&ctx, update);
        }

        if ctx.trigger_monitor.is_idle()? {
            // Not recording any more, flush current event description if any
            let update = event_tracker.clear();
            if update.is_some() {
                debug!("Flushing event because idle");
            }
            process_update_event(&ctx, update);
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
        let current_fps = pacemaker.current_frequency() as f64;
        instrumentation::FPS.set(current_fps);
        instrumentation::FPS_DEVIATION.set(current_fps - ctx.max_fps as f64);
    }
    Ok(())
}

fn describe(classes: &HashMap<i32, &str>, d: &Detection) -> String {
    format!(
        "{} ({:.1}%) {}x{} (={}) at {}x{}",
        classes[&d.class_id],
        d.confidence * 100.0,
        d.bounding_box.width,
        d.bounding_box.height,
        d.bounding_box.width * d.bounding_box.height,
        d.bounding_box.x,
        d.bounding_box.y,
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
