use libc::{c_char, c_double, time_t, timeval};
use memmap2::MmapRaw;
use std::fs::OpenOptions;
use std::{io, slice};
use std::error::Error;
use std::mem::size_of;
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use opencv::core::{CV_8U, Mat, MatTraitConst, MatTraitConstManual, Scalar};
use opencv::dnn::{blob_from_image, LayerTraitConst, NetTrait, NetTraitConst, read_net};
use opencv::imgcodecs::IMREAD_UNCHANGED;
use opencv::types::VectorOfMat;

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
    NONE = 2,  // grayscale
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
        let mmap = MmapRaw::map_raw(&file)?;  // we don't actually have to mmap this at all. we can just pread from the file. its just a file, bro.

        let shared_data = mmap.as_ptr(); // as *const MonitorSharedData;

        let image_buffer_count = 3;  // needs to be retrieved from the database

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
            assert_eq!((*videostore_data).size, size_of::<MonitorVideoStoreData>() as u32);

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

fn main() -> Result<(), Box<dyn Error>> {
    let mid = 5;

    let confidence_threshold = 0.5;

    let mut net = read_net("yolov4-tiny.weights", "yolov4-tiny.cfg", "")?;
    net.set_preferable_target(0)?;

    let out_names = net.get_unconnected_out_layers_names()?;

    for file in vec!("19399-video-0001.png", "19399-video-0002.png", "19399-video-0003.png") {
        println!("Processing {}", file);
        let image = opencv::imgcodecs::imread(file, IMREAD_UNCHANGED)?;

        // preprocess
        let scale = 1.0 /*/ 255.0*/;
        let size = (192, 192);
        let mean = (0.0, 0.0, 0.0);

        let blob = blob_from_image(&image, scale, size.into(), mean.into(), false, false, CV_8U)?;

        net.set_input(&blob, "", scale, mean.into())?;

        // run
        let mut outs = VectorOfMat::new();
        net.forward(&mut outs, &out_names)?;

        // postprocess

        let out_layers = net.get_unconnected_out_layers()?;
        let out_layer_type = net.get_layer(out_layers.get(0).unwrap()).unwrap().typ();

        assert_eq!(out_layer_type, "Region");
        for out in outs {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]

            println!("out: {:#?}", out);

            for row_idx in 0..out.rows() {
                let row = out.at_row::<f32>(row_idx).unwrap();

                let (center_x, center_y) = (row[0], row[1]);
                let (width, heigt) = (row[2], row[3]);

                let class = row[5..]
                    .iter()
                    .cloned()
                    .zip(2..)
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                if let None = class {
                    continue
                }
                let (confidence, class_id) = class.unwrap();

                if confidence < confidence_threshold {
                    continue
                }

                println!("   {}: {:?} @ {}", row_idx, confidence, class_id);
            }
        }

        /*
                for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
         */

    }


    /*let monitor = Monitor::connect(mid)?;
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
            println!("New image available at index {}, timestamp {:?}", last_write_index, timestamp);
            last_read_index = last_write_index;

            let image_data = unsafe { slice::from_raw_parts(monitor.shared_images, image_size as usize) };
            let image_data = image_data.to_vec();

            std::fs::write("/tmp/imago", image_data)?;
        }
    }*/
    Ok(())
}
