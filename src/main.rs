use libc::{c_char, c_double, c_void, time_t, timeval};
use memmap2::MmapRaw;
use opencv::core::{Mat, MatTraitConst, MatTraitConstManual, Point2f, Rect, Rect2f, Scalar, Vector, CV_8U, MatTrait, CV_8UC4};
use opencv::dnn::{blob_from_image, nms_boxes, read_net, LayerTraitConst, NetTrait, NetTraitConst, Net};
use opencv::types::{VectorOfMat, VectorOfRect, VectorOfString};
use std::collections::HashMap;
use std::error::Error;
use std::fs::OpenOptions;
use std::mem::size_of;
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{io, slice};
use opencv::imgproc::{COLOR_RGBA2RGB, cvt_color};

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct MonitorSharedData {
    // size and offsets in ZM code is all wrong
    size: u32,
    last_write_index: i32,
    last_read_index: i32,
    state: u32,
    capture_fps: c_double,
    analysis_fps: c_double,
    last_event_id: u64,
    action: u32,
    brightness: i32,
    hue: i32,
    colour: i32,
    contrast: i32,
    alarm_x: i32,
    alarm_y: i32,
    valid: u8,
    active: u8,
    signal: u8,
    format: SubpixelOrder,
    imagesize: u32,
    last_frame_score: u32,
    audio_frequency: u32,
    audio_channels: u32,

    startup_time: time_t,
    zmc_heartbeat_time: time_t,
    last_write_time: time_t,
    last_read_time: time_t,

    control_state: [u8; 256],

    alarm_cause: [c_char; 256],
    video_fifo_path: [c_char; 64],
    audio_fifo_path: [c_char; 64],
}

const _: [u8; 760] = [0; size_of::<MonitorSharedData>()];

// zm_rgb.h

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum ColourType {
    GRAY8 = 1,
    RGB24 = 3,
    RGB32 = 4,
}

#[derive(Copy, Clone, Debug)]
#[repr(u8)]
enum SubpixelOrder {
    NONE = 2, // grayscale
    RGB = 6,
    BGR = 5,
    BGRA = 7,
    RGBA = 8,
    ABGR = 9,
    ARGB = 10,
}

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum TriggerState {
    TriggerCancel,
    TriggerOn,
    TriggerOff,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct MonitorTriggerData {
    size: u32,
    trigger_state: TriggerState,
    trigger_score: u32,
    padding: u32,
    trigger_cause: [c_char; 32],
    trigger_text: [c_char; 256],
    trigger_showtext: [c_char; 256],
}

const _: [u8; 560] = [0; size_of::<MonitorTriggerData>()];

#[derive(Copy, Clone)]
#[repr(C)]
struct MonitorVideoStoreData {
    // size in ZM is wrong
    size: u32,
    padding: u32, // padding here, not in ZM which is wrong
    current_event: u64,
    event_file: [c_char; 4096],
    recording: timeval,
}

const _: [u8; 4128] = [0; size_of::<MonitorVideoStoreData>()];

#[repr(C)]
struct MonitorMmapLayout {
    shared_data: MonitorSharedData,
    trigger_data: MonitorTriggerData,
    videostore_data: MonitorVideoStoreData,
    // ...
}

struct Monitor {
    mmap: MmapRaw,

    shared_data: *const MonitorSharedData,
    trigger_data: *const MonitorTriggerData,
    videostore_data: *const MonitorVideoStoreData,
    shared_timestamps: *const timeval,
    shared_images: *const u8,
}

impl Monitor {
    fn connect(monitor_id: u32) -> io::Result<Monitor> {
        let mmap_path = format!("/dev/shm/zm.mmap.{}", monitor_id);
        let file = OpenOptions::new().read(true).write(true).open(&mmap_path)?;
        let mmap = MmapRaw::map_raw(&file)?; // we don't actually have to mmap this at all. we can just pread from the file. its just a file, bro.

        let shared_data = mmap.as_ptr(); // as *const MonitorSharedData;

        let image_buffer_count = 3; // needs to be retrieved from the database

        let monitor = unsafe {
            let trigger_data = shared_data.add(size_of::<MonitorSharedData>());
            let videostore_data = trigger_data.add(size_of::<MonitorTriggerData>());
            let shared_timestamps = videostore_data.add(size_of::<MonitorVideoStoreData>());
            let shared_images = shared_timestamps.add(image_buffer_count * size_of::<timeval>());
            let image_alignment_adj = 64 - (shared_images as usize % 64);
            let shared_images = shared_images.add(image_alignment_adj);

            let shared_data = shared_data as *const MonitorSharedData;
            let trigger_data = trigger_data as *const MonitorTriggerData;
            let videostore_data = videostore_data as *const MonitorVideoStoreData;
            let shared_timestamps = shared_timestamps as *const timeval;
            //let shared_images = shared_images as *const u8;

            assert_eq!((*shared_data).size, size_of::<MonitorSharedData>() as u32);
            assert_eq!((*trigger_data).size, size_of::<MonitorTriggerData>() as u32);
            assert_eq!(
                (*videostore_data).size,
                size_of::<MonitorVideoStoreData>() as u32
            );

            Monitor {
                mmap,
                shared_data,
                trigger_data,
                videostore_data,
                shared_timestamps,
                shared_images,
            }
        };

        Ok(monitor)
    }

    fn valid(&self) -> bool {
        // should use self.read_volatile().valid or have entirely separate read() method which gives a pubstruct
        unsafe { (*self.shared_data).valid > 0 }
    }

    fn last_write_time(&self) -> SystemTime {
        let value = unsafe { (*self.shared_data).last_write_time };
        UNIX_EPOCH + Duration::from_secs(value as u64)
    }
}

#[derive(Clone, Debug)]
struct Detection {
    confidence: f32,
    class_id: i32,
    bounding_box: Rect,
}

struct YoloV4Tiny {
    net: Net,
    confidence_threshold: f32,
    nms_threshold: f32,
    size: i32,

    out_names: Vector<String>,
    out_layers: Vector<i32>,
}

impl YoloV4Tiny {
    fn new(confidence_threshold: f32, size: i32) -> opencv::Result<YoloV4Tiny> {
        let mut net = read_net("yolov4-tiny.weights", "yolov4-tiny.cfg", "")?;
        net.set_preferable_target(0)?;

        let out_names = net.get_unconnected_out_layers_names()?;
        let out_layers = net.get_unconnected_out_layers()?;
        let out_layer_type = net.get_layer(out_layers.get(0).unwrap()).unwrap().typ();
        assert_eq!(out_layer_type, "Region");

        Ok(YoloV4Tiny {
            net,
            out_names, out_layers,
            confidence_threshold: confidence_threshold,
            nms_threshold: 0.4,
            size: size,
        })
    }

    fn infer(&mut self, image: &Mat) -> opencv::Result<Vec<Detection>> {
        let size = (self.size, self.size);
        let mean = (0.0, 0.0, 0.0);
        let blob = blob_from_image(&image, 1.0, size.into(), mean.into(), false, false, CV_8U)?;
        let scale = 1.0 / 255.0;
        self.net.set_input(&blob, "", scale, mean.into())?;

        let outs = {
            let mut outs = VectorOfMat::new();
            self.net.forward(&mut outs, &self.out_names)?;
            outs
        };

        let image_width = image.cols() as f32;
        let image_height = image.rows() as f32;

        let detections: Vec<Detection> = outs
            .iter()
            .map(|out| {
                // Network produces output blob with a shape NxC where N is a number of
                // detected objects and C is a number of classes + 4 where the first 4
                // numbers are [center_x, center_y, width, height]

                (0..out.rows())
                    .map(move |i| {
                        let row = out.at_row::<f32>(i).unwrap();

                        let get_bounding_box = |row: &[f32]| -> Rect {
                            let (center_x, center_y) = (row[0], row[1]);
                            let (width, height) = (row[2], row[3]);

                            let center_x = (center_x * image_width) as i32;
                            let center_y = (center_y * image_height) as i32;
                            let width = (width * image_width) as i32;
                            let height = (height * image_height) as i32;

                            let left_edge = center_x - width / 2;
                            let top_edge = center_y - height / 2;

                            Rect::new(left_edge, top_edge, width, height)
                        };

                        let get_class = |row: &[f32]| {
                            let class = row[4..]
                                .iter()
                                //.cloned()
                                .zip(1..) // 1.. for 1-based class index, 0.. for 0-based
                                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            let (&confidence, class_id) = class.unwrap();
                            (confidence, class_id)
                        };

                        let (confidence, class_id) = get_class(row);
                        let bounding_box = get_bounding_box(row);

                        Detection {
                            confidence,
                            class_id,
                            bounding_box,
                        }
                    })
                    .filter(|detection| detection.confidence >= self.confidence_threshold)
            })
            .flatten()
            .collect();

        // Perform NMS filtering
        let mut class2detections: HashMap<i32, Vec<&Detection>> = HashMap::new();
        for detection in &detections {
            let dets = class2detections
                .entry(detection.class_id)
                .or_insert_with(Vec::new);
            dets.push(&detection);
        }

        let mut nms_detections = vec![];

        for (&class_id, detections) in &class2detections {
            let bounding_boxes: VectorOfRect =
                detections.iter().map(|det| det.bounding_box).collect();
            let confidences: Vector<f32> = detections.iter().map(|det| det.confidence).collect();
            let mut chosen_indices = Vector::new();
            nms_boxes(
                &bounding_boxes,
                &confidences,
                self.confidence_threshold,
                self.nms_threshold,
                &mut chosen_indices,
                1.0,
                0,
            )?;

            for index in chosen_indices {
                nms_detections.push(detections[index as usize].clone());
            }
        }

        Ok(nms_detections)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mid = 5;

    let mut yolo = YoloV4Tiny::new(0.5, 256)?;

    /*for file in vec![
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

    let monitor = Monitor::connect(mid)?;
    println!("Monitor shm valid: {}", monitor.valid());

    let image_buffer_count = 3;  // needs to be retrieved from the database
    let mut last_read_index = image_buffer_count;

    let image_size = unsafe { (*monitor.shared_data).imagesize };

    println!("Image size is claimed to be: {}", image_size);

    loop {
        //println!("Last write time: {:?}", monitor.last_write_time());
        //sleep(Duration::from_millis(500));

        let last_write_index = unsafe { (*monitor.shared_data).last_write_index };
        if last_write_index != last_read_index && last_write_index != image_buffer_count {
            let timestamp = unsafe { *monitor.shared_timestamps.offset(last_write_index as isize) };
            let timestamp = UNIX_EPOCH + Duration::from_secs(timestamp.tv_sec as u64) + Duration::from_micros(timestamp.tv_usec as u64);
            //println!("New image available at index {}, timestamp {:?}", last_write_index, timestamp);
            last_read_index = last_write_index;

            //let image_data = unsafe { slice::from_raw_parts(monitor.shared_images, image_size as usize) };
            //let image_data = image_data.to_vec();

            let image = unsafe {
                let image_data = monitor.shared_images.add(image_size as usize * last_write_index as usize);
                let image_row_size = 1280 * 4;

                Mat::new_rows_cols_with_data(1280, 720, CV_8UC4, image_data as *mut c_void, image_row_size)?
            };

            let mut rgb_image = Mat::default();
            cvt_color(&image, &mut rgb_image, COLOR_RGBA2RGB, 0)?;

            //println!("Shape: {:?}", rgb_image);

            let t0 = Instant::now();
            let detections = yolo.infer(&rgb_image)?;
            let td = Instant::now() - t0;

            if detections.len() > 1 {
                println!("Inference completed in {:?}:\n{:#?}",
                         td, detections);
            }

            //std::fs::write("/tmp/imago", image_data)?;
        }
    }
    Ok(())
}
