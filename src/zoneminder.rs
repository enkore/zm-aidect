use libc::timeval;
use mysql::params;
use mysql::prelude::Queryable;
use opencv::core::{Mat, MatTrait, MatTraitConst, Rect};
use std::collections::HashMap;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::mem::size_of;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::time::Duration;
use std::{fs, io, slice};

mod shm;

pub fn update_event_notes(
    zm_conf: &ZoneMinderConf,
    event_id: u64,
    notes: &str,
) -> mysql::Result<()> {
    let mut db = zm_conf.connect_db()?;
    db.exec_drop(
        "UPDATE Events SET Notes = :notes WHERE Id = :id",
        params! {
            "id" => event_id,
            "notes" => notes,
        },
    )
}

pub struct Monitor<'zmconf> {
    monitor_id: u32,
    zm_conf: &'zmconf ZoneMinderConf,

    mmap_path: String,
    file: File,
    ino: u64,

    trigger_data_offset: usize,
    videostore_data_offset: usize,
}

impl Monitor<'_> {
    pub fn connect(zm_conf: &ZoneMinderConf, monitor_id: u32) -> Result<Monitor, Box<dyn Error>> {
        let mmap_path = format!("{}/zm.mmap.{}", zm_conf.mmap_path, monitor_id);
        let file = OpenOptions::new().read(true).write(true).open(&mmap_path)?;

        let trigger_data_offset = size_of::<shm::MonitorSharedData>();
        let videostore_data_offset = trigger_data_offset + size_of::<shm::MonitorTriggerData>();

        Ok(Monitor {
            monitor_id,
            zm_conf,
            mmap_path,
            ino: file.metadata()?.ino(),
            file,

            trigger_data_offset,
            videostore_data_offset,
        })
    }

    pub fn get_max_analysis_fps(&self) -> Result<f32, Box<dyn Error>> {
        Ok(self.query_monitor_config()?.analysis_fps_limit)
    }

    pub fn stream_images(&self) -> Result<ImageStream, Box<dyn Error>> {
        let state = self.read()?;
        let config = self.query_monitor_config()?;
        let image_buffer_count = config.image_buffer_count;

        // now that we have the image buffer size we can figure the dynamic offsets out
        let shared_timestamps_offset =
            self.videostore_data_offset + size_of::<shm::MonitorVideoStoreData>();
        let shared_images_offset =
            shared_timestamps_offset + image_buffer_count as usize * size_of::<timeval>();
        let shared_images_offset = shared_images_offset + 64 - (shared_images_offset % 64);

        Ok(ImageStream {
            width: config.width,
            height: config.height,
            image_buffer_count,
            monitor: self,
            last_read_index: image_buffer_count,
            image_size: state.shared_data.imagesize,
            format: state.shared_data.format,
            shared_images_offset: shared_images_offset as u64,
        })
    }

    pub fn is_idle(&self) -> io::Result<bool> {
        Ok(self.read()?.shared_data.state == shm::MonitorState::Idle)
    }

    /// Mark at least one frame as an alarm frame with the given score. Wait for event to be created,
    /// then return event ID. Does not necessarily cause creation of a new event.
    pub fn trigger(&self, cause: &str, description: &str, score: u32) -> io::Result<u64> {
        let poll_interval = 10;
        self.set_trigger(cause, description, score)?;
        for n in 0.. {
            let state = self.read()?.shared_data.state;
            // Alarm sorta implies that we just triggered an alarm frame, while
            // Alert sorta implies there's an on-going event.
            // In any case last_event_id ought to usually be "our" event ID for this alarmation.
            if state == shm::MonitorState::Alarm || state == shm::MonitorState::Alert {
                break;
            }
            std::thread::sleep(Duration::from_millis(poll_interval));
            if n > 500 {
                eprintln!("Waited {} ms for zoneminder to notice our bulgy wulgy, giving up and canceling it :c", n * poll_interval);
            }
        }
        self.reset_trigger()?;
        Ok(self.read()?.shared_data.last_event_id)
    }

    fn set_trigger(&self, cause: &str, description: &str, score: u32) -> io::Result<()> {
        let cause = cause.as_bytes();
        let description = description.as_bytes();

        let mut trigger_data = self.read()?.trigger_data;
        trigger_data.trigger_cause[..cause.len()].copy_from_slice(cause);
        trigger_data.trigger_text[..description.len()].copy_from_slice(description);
        trigger_data.trigger_showtext.fill(0);
        trigger_data.trigger_score = score;
        // all of this is terribly racy but pwritin' the data before the state change should reduce the odds of problems
        self.pwrite(self.trigger_data_offset, &trigger_data)?;
        trigger_data.trigger_state = shm::TriggerState::TriggerOn;
        self.pwrite(self.trigger_data_offset, &trigger_data)
    }

    fn reset_trigger(&self) -> io::Result<()> {
        let mut trigger_data = self.read()?.trigger_data;
        trigger_data.trigger_cause.fill(0);
        trigger_data.trigger_text.fill(0);
        trigger_data.trigger_showtext.fill(0);
        trigger_data.trigger_score = 0;
        self.pwrite(self.trigger_data_offset, &trigger_data)?;
        trigger_data.trigger_state = shm::TriggerState::TriggerCancel;
        self.pwrite(self.trigger_data_offset, &trigger_data)
    }

    fn read(&self) -> io::Result<MonitorState> {
        let shared_data: shm::MonitorSharedData = self.pread(0)?;
        let trigger_data: shm::MonitorTriggerData = self.pread(self.trigger_data_offset)?;
        if shared_data.valid == 0 {
            return Err(io::Error::new(ErrorKind::Other, "Monitor shm is not valid"));
        }
        self.check_file_stale()?;
        assert_eq!(
            shared_data.size as usize,
            size_of::<shm::MonitorSharedData>(),
            "Invalid SHM shared_data size, incompatible ZoneMinder version"
        );
        assert_eq!(
            trigger_data.size as usize,
            size_of::<shm::MonitorTriggerData>(),
            "Invalid SHM trigger_data size, incompatible ZoneMinder version"
        );
        Ok(MonitorState {
            shared_data,
            trigger_data,
        })
    }

    fn pread<T>(&self, offset: usize) -> io::Result<T> {
        let mut buf = Vec::new();
        buf.resize(size_of::<T>(), 0);
        self.file.read_exact_at(&mut buf, offset as u64)?;
        unsafe { Ok(std::ptr::read(buf.as_ptr() as *const _)) }
    }

    fn pwrite<T>(&self, offset: usize, data: &T) -> io::Result<()> {
        let data = unsafe { slice::from_raw_parts(data as *const T as *const u8, size_of::<T>()) };
        self.file.write_all_at(data, offset as u64)
    }

    fn check_file_stale(&self) -> io::Result<()> {
        // Additional sanity check, if the file-on-tmpfs is now a different file, we're definitely listening to a stranger.
        // ZM seems to be quite good about ensuring shared_data.valid gets flipped to 0 even when zmc crashes though.
        if fs::metadata(&self.mmap_path)?.ino() != self.ino {
            return Err(io::Error::new(
                ErrorKind::Other,
                "Monitor shm fd is stale, must reconnect",
            ));
        }
        Ok(())
    }

    fn query_monitor_config(&self) -> mysql::Result<MonitorDatabaseConfig> {
        let mut db = self.zm_conf.connect_db()?;
        Ok(db.exec_map("SELECT Name, StorageId, Enabled, Width, Height, Colours, ImageBufferCount, AnalysisFPSLimit FROM Monitors WHERE Id = :id",
                                  params! { "id" => self.monitor_id },
            |(name, storage_id, enabled, width, height, colours, image_buffer_count, analysis_fps_limit)| {
                MonitorDatabaseConfig {
                    name,
                    storage_id,
                    enabled,
                    width,
                    height,
                    colours,
                    image_buffer_count,
                    analysis_fps_limit,
                }
            }
        )?.remove(0))
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct MonitorDatabaseConfig {
    name: String,
    storage_id: u32,
    enabled: bool,
    width: u32,
    height: u32,
    colours: u32,
    image_buffer_count: u32,
    analysis_fps_limit: f32,
}

fn convert_to_rgb(format: shm::SubpixelOrder, image: Mat) -> opencv::Result<Mat> {
    let mut rgb_image = Mat::default();
    let conversion = match format {
        shm::SubpixelOrder::NONE => opencv::imgproc::COLOR_GRAY2RGB,
        shm::SubpixelOrder::RGB => return Ok(image),
        shm::SubpixelOrder::BGR => opencv::imgproc::COLOR_BGR2RGB,
        shm::SubpixelOrder::BGRA => opencv::imgproc::COLOR_BGRA2RGB,
        shm::SubpixelOrder::RGBA => opencv::imgproc::COLOR_RGBA2RGB,
        _ => panic!("Unsupported pixel format: {:?}", format),
    };
    opencv::imgproc::cvt_color(&image, &mut rgb_image, conversion, 0)?;
    Ok(rgb_image)
}

fn zm_format_to_cv_format(format: shm::SubpixelOrder) -> i32 {
    match format {
        shm::SubpixelOrder::NONE => opencv::core::CV_8UC1,
        shm::SubpixelOrder::RGB => opencv::core::CV_8UC3,
        shm::SubpixelOrder::BGR => opencv::core::CV_8UC3,
        shm::SubpixelOrder::BGRA => opencv::core::CV_8UC4,
        shm::SubpixelOrder::RGBA => opencv::core::CV_8UC4,
        shm::SubpixelOrder::ABGR => opencv::core::CV_8UC4,
        shm::SubpixelOrder::ARGB => opencv::core::CV_8UC4,
    }
}

pub struct ImageStream<'mon> {
    monitor: &'mon Monitor<'mon>,
    last_read_index: u32,
    width: u32,
    height: u32,
    image_size: u32,
    format: shm::SubpixelOrder,
    image_buffer_count: u32,
    shared_images_offset: u64,
}

impl ImageStream<'_> {
    fn wait_for_image(&mut self) -> Result<Mat, Box<dyn Error>> {
        loop {
            let state = self.monitor.read()?;
            let last_write_index = state.shared_data.last_write_index as u32;
            if last_write_index != self.last_read_index
                && last_write_index != self.image_buffer_count
            {
                self.last_read_index = last_write_index;
                let image = self.read_image(last_write_index)?;
                return Ok(convert_to_rgb(self.format, image)?);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn read_image(&self, index: u32) -> Result<Mat, Box<dyn Error>> {
        assert_eq!(self.width * self.height * 4, self.image_size);
        let mut mat = Mat::new_size_with_default(
            (self.width as i32, self.height as i32).into(),
            zm_format_to_cv_format(self.format),
            0.into(),
        )?;
        self.read_image_into(index, &mut mat)?;
        Ok(mat)
    }

    fn read_image_into(&self, index: u32, mat: &mut Mat) -> Result<(), Box<dyn Error>> {
        assert_eq!(self.width * self.height, mat.total() as u32);
        assert_eq!(mat.typ(), zm_format_to_cv_format(self.format));
        self.monitor.check_file_stale()?;
        let mut slice =
            unsafe { slice::from_raw_parts_mut(mat.ptr_mut(0)?, self.image_size as usize) };
        let image_offset = self.shared_images_offset as u64 + self.image_size as u64 * index as u64;
        self.monitor.file.read_exact_at(&mut slice, image_offset)?;
        Ok(())
    }
}

impl Iterator for ImageStream<'_> {
    type Item = Result<Mat, Box<dyn Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.wait_for_image())
    }
}

struct MonitorState {
    shared_data: shm::MonitorSharedData,
    trigger_data: shm::MonitorTriggerData,
}

pub type ZoneShape = Vec<(i32, i32)>;

pub trait Bounding {
    fn bounding_box(&self) -> Rect;
}

impl Bounding for ZoneShape {
    fn bounding_box(&self) -> Rect {
        let min_x = self.iter().map(|xy| xy.0).min().unwrap();
        let min_y = self.iter().map(|xy| xy.1).min().unwrap();
        let max_x = self.iter().map(|xy| xy.0).max().unwrap();
        let max_y = self.iter().map(|xy| xy.1).max().unwrap();

        let width = max_x - min_x;
        let height = max_y - min_y;
        Rect {
            x: min_x,
            y: min_y,
            width,
            height,
        }
    }
}

#[derive(Debug)]
pub struct ZoneConfig {
    pub size: Option<u32>,
    pub threshold: Option<f32>,
    pub shape: ZoneShape,
    pub trigger: Option<u32>,
    pub fps: Option<u32>,
    pub min_area: Option<u32>,
}

impl ZoneConfig {
    pub fn get_zone_config(
        zm_conf: &ZoneMinderConf,
        monitor_id: u32,
    ) -> Result<ZoneConfig, Box<dyn Error>> {
        let mut db = zm_conf.connect_db()?;
        let dbzone = db.exec_first(
            "SELECT Name, Type, Coords FROM Zones WHERE MonitorId = :id AND Name LIKE \"aidect%\"",
            params! { "id" => monitor_id },
        )?;
        let dbzone: mysql::Row = dbzone.unwrap();

        Ok(ZoneConfig::parse(
            &dbzone.get::<String, &str>("Name").unwrap(),
            &dbzone.get::<String, &str>("Coords").unwrap(),
        ))
    }

    fn parse(name: &str, coords: &str) -> ZoneConfig {
        let mut config = Self::parse_zone_name(name);
        config.shape = Self::parse_zone_coords(coords);
        config
    }

    fn parse_zone_name(zone_name: &str) -> ZoneConfig {
        let keys: HashMap<&str, &str> = zone_name
            .split_ascii_whitespace()
            .skip(1)
            .map(|item| item.split_once('='))
            .filter_map(|x| x)
            .collect();

        let get_int = |key| keys.get(key).and_then(|v| v.trim().parse::<u32>().ok());

        ZoneConfig {
            shape: Vec::new(),
            threshold: keys
                .get("Threshold")
                .and_then(|v| v.trim().parse::<f32>().ok())
                .map(|v| v / 100.0),
            size: get_int("Size"),
            trigger: get_int("Trigger"),
            fps: get_int("FPS"),
            min_area: get_int("MinArea"),
        }
    }

    fn parse_zone_coords(coords: &str) -> ZoneShape {
        let parse = |v: &str| v.trim().parse::<i32>().unwrap();
        coords
            .split_ascii_whitespace()
            .map(|point| point.split_once(','))
            .filter_map(|v| v)
            .map(|(x, y)| (parse(x), parse(y)))
            .collect()
    }
}

#[derive(Debug)]
pub struct ZoneMinderConf {
    db_host: String,
    db_name: String,
    db_user: String,
    db_password: String,
    mmap_path: String,
}

impl ZoneMinderConf {
    fn parse_zm_conf(zm_conf_contents: &str) -> ZoneMinderConf {
        let keys: HashMap<&str, &str> = zm_conf_contents
            .lines()
            .map(|line| line.trim())
            .filter(|line| line.starts_with("ZM_"))
            .filter_map(|line| line.split_once('='))
            .collect();

        ZoneMinderConf {
            db_host: keys["ZM_DB_HOST"].to_string(),
            db_name: keys["ZM_DB_NAME"].to_string(),
            db_user: keys["ZM_DB_USER"].to_string(),
            db_password: keys["ZM_DB_PASS"].to_string(),
            mmap_path: keys["ZM_PATH_MAP"].to_string(),
        }
    }

    pub fn parse_default() -> io::Result<ZoneMinderConf> {
        let path = "/etc/zm/zm.conf";
        let contents = fs::read_to_string(path)?;
        let contents = contents
            + "\n"
            + &fs::read_dir("/etc/zm/conf.d")?
                .filter_map(Result::ok)
                .map(|entry| fs::read_to_string(entry.path()))
                .filter_map(Result::ok)
                .fold(String::new(), |a, b| a + "\n" + &b); // O(n**2)

        Ok(Self::parse_zm_conf(&contents))
    }
}

impl ZoneMinderConf {
    fn connect_db(&self) -> mysql::Result<mysql::Conn> {
        let builder = mysql::OptsBuilder::new()
            .ip_or_hostname(Some(&self.db_host))
            .db_name(Some(&self.db_name))
            .user(Some(&self.db_user))
            .pass(Some(&self.db_password));
        mysql::Conn::new(builder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_zm_conf() {
        let conf = "# ZoneMinder database hostname or ip address and optionally port or unix socket
# Acceptable formats include hostname[:port], ip_address[:port], or
# localhost:/path/to/unix_socket
ZM_DB_HOST=localhost

# ZoneMinder database name
ZM_DB_NAME=zm

# ZoneMinder database user
ZM_DB_USER=zmuser

# ZoneMinder database password
ZM_DB_PASS=zmpass

ZM_PATH_MAP=/dev/shm
";

        let parsed = ZoneMinderConf::parse_zm_conf(conf);
        assert_eq!(parsed.db_host, "localhost");
        assert_eq!(parsed.db_name, "zm");
        assert_eq!(parsed.db_user, "zmuser");
        assert_eq!(parsed.db_password, "zmpass");
        assert_eq!(parsed.mmap_path, "/dev/shm");
    }

    #[test]
    fn test_parse_zone_name_basic() {
        let zone_name = "aidect";
        let parsed = ZoneConfig::parse_zone_name(zone_name);
        assert_eq!(parsed.shape.len(), 0);
        assert_eq!(parsed.threshold, None);
        assert_eq!(parsed.size, None);
    }

    #[test]
    fn test_parse_zone_name() {
        let zone_name = "aidect Size=128 Threshold=50";
        let parsed = ZoneConfig::parse_zone_name(zone_name);
        assert_eq!(parsed.shape.len(), 0);
        assert_eq!(parsed.threshold, Some(0.5));
        assert_eq!(parsed.size, Some(128));
    }

    #[test]
    fn test_parse_zone_coords() {
        let coords = "123,56 899,41 687,425";
        let parsed = ZoneConfig::parse_zone_coords(coords);
        assert_eq!(parsed, vec![(123, 56), (899, 41), (687, 425)]);
    }
}
