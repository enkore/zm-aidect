use std::error::Error;
use std::fs::{File, OpenOptions};
use std::{fs, io, slice};
use std::collections::HashMap;
use std::io::ErrorKind;
use std::mem::size_of;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use libc::timeval;
use mysql::params;
use mysql::prelude::Queryable;
use opencv::core::{CV_8UC4, Mat, MatTrait, MatTraitConst};

mod shm;
pub mod zmtrigger;

#[allow(dead_code)]
pub struct Monitor {
    mmap_path: String,
    file: File,
    ino: u64,

    width: u32,
    height: u32,

    pub zone: ZoneConfig,

    pub max_fps: f32,  // XXX
    pub image_buffer_count: u32,  // XXX

    trigger_data_offset: usize,
    videostore_data_offset: usize,
    shared_timestamps_offset: usize,
    shared_images_offset: usize,
}

impl Monitor {
    pub fn connect(zm_conf: &ZoneMinderConf, monitor_id: u32) -> Result<Monitor, Box<dyn Error>> {
        let mut db = zm_conf.connect_db()?;
        let dbmon = db.exec_first("SELECT Name, StorageId, Enabled, Width, Height, Colours, ImageBufferCount, AnalysisFPSLimit FROM Monitors WHERE Id = :id",
                                  params! { "id" => monitor_id }
        )?;
        let dbmon: mysql::Row = dbmon.unwrap();

        let dbzone = db.exec_first("SELECT Name, Type, Coords FROM Zones WHERE MonitorId = :id AND Name LIKE \"aidect%\"",
        params! { "id" => monitor_id }
        )?;
        let dbzone: mysql::Row = dbzone.unwrap();

        let zone = ZoneConfig::parse(
            &dbzone.get::<String, &str>("Name").unwrap(),
            &dbzone.get::<String, &str>("Coords").unwrap()
        );

        // TODOs here:
        // 4. crop image to bounding box of zone polygon
        // 5. blank remaining area outside zone polygon

        let image_buffer_count: usize = dbmon.get("ImageBufferCount").unwrap();
        let width: u32 = dbmon.get("Width").unwrap();
        let height: u32 = dbmon.get("Height").unwrap();
        let max_fps: f32 = dbmon.get("AnalysisFPSLimit").unwrap();


        let mmap_path = format!("{}/zm.mmap.{}", zm_conf.mmap_path, monitor_id);
        let file = OpenOptions::new().read(true).write(true).open(&mmap_path)?;

        let trigger_data_offset = size_of::<shm::MonitorSharedData>();
        let videostore_data_offset = trigger_data_offset + size_of::<shm::MonitorTriggerData>();
        let shared_timestamps_offset = videostore_data_offset + size_of::<shm::MonitorVideoStoreData>();
        let shared_images_offset = shared_timestamps_offset + image_buffer_count * size_of::<timeval>();
        let shared_images_offset = shared_images_offset + 64 - (shared_images_offset % 64);

        Ok(Monitor {
            mmap_path,
            ino: file.metadata()?.ino(),
            file,

            zone,
            width, height,
            image_buffer_count: image_buffer_count as u32,
            max_fps,

            trigger_data_offset,
            videostore_data_offset,
            shared_timestamps_offset,
            shared_images_offset,
        })
    }

    fn pread<T>(&self, offset: usize) -> io::Result<T> {
        let mut buf = Vec::new();
        buf.resize(size_of::<T>(), 0);
        self.file.read_exact_at(&mut buf, offset as u64)?;
        unsafe {
            Ok(std::ptr::read(buf.as_ptr() as *const _))
        }
    }

    pub fn read(&self) -> io::Result<MonitorState> {
        let shared_data: shm::MonitorSharedData = self.pread(0)?;
        let trigger_data: shm::MonitorTriggerData = self.pread(self.trigger_data_offset)?;
        if shared_data.valid == 0 {
            return Err(io::Error::new(ErrorKind::Other, "Monitor shm is not valid"));
        }
        self.check_file_stale()?;
        Ok(MonitorState {
            shared_data, trigger_data
        })
    }

    fn check_file_stale(&self) -> io::Result<()> {
        // Additional sanity check, if the file-on-tmpfs is now a different file, we're definitely listening to a stranger.
        // ZM seems to be quite good about ensuring shared_data.valid gets flipped to 0 even when zmc crashes though.
        if fs::metadata(&self.mmap_path)?.ino() != self.ino {
            return Err(io::Error::new(ErrorKind::Other, "Monitor shm fd is stale, must reconnect"));
        }
        Ok(())
    }

    pub fn read_image(&self, token: ImageToken) -> Result<Mat, Box<dyn Error>> {
        assert_eq!(self.width * self.height * 4, token.size);
        self.check_file_stale()?;
        let mut mat = Mat::new_size_with_default((self.width as i32, self.height as i32).into(), CV_8UC4, 0.into())?;
        self.read_image_into(token, &mut mat)?;
        Ok(mat)
    }

    pub fn read_image_into(&self, token: ImageToken, mat: &mut Mat) -> Result<(), Box<dyn Error>> {
        assert_eq!(self.width * self.height, mat.total() as u32);
        assert_eq!(mat.typ(), CV_8UC4);
        self.check_file_stale()?;
        let mut slice = unsafe { slice::from_raw_parts_mut(mat.ptr_mut(0)?, token.size as usize) };
        let image_offset = self.shared_images_offset as u64 + token.size as u64 * token.index as u64;
        self.file.read_exact_at(&mut slice, image_offset)?;
        Ok(())
    }
}

pub struct MonitorState {
    shared_data: shm::MonitorSharedData,
    trigger_data: shm::MonitorTriggerData,
}

impl MonitorState {
    pub fn last_write_time(&self) -> SystemTime {
        let value = self.shared_data.last_write_time;
        UNIX_EPOCH + Duration::from_secs(value as u64)
    }

    pub fn last_write_index(&self) -> u32 {
        self.shared_data.last_write_index as u32
    }

    pub fn last_image_token(&self) -> ImageToken {
        ImageToken {
            index: self.last_write_index(),
            size: self.shared_data.imagesize,
        }
    }
}

pub struct ImageToken {
    index: u32,
    size: u32,
}

#[derive(Debug)]
pub struct ZoneConfig {
    pub size: Option<u32>,
    pub threshold: Option<f32>,
    pub shape: Vec<(u32, u32)>,
    pub trigger: Option<u32>,
}

impl ZoneConfig {
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

        ZoneConfig {
            shape: Vec::new(),
            threshold: keys.get("Threshold")
                .and_then(|v| v.trim().parse::<f32>().ok())
                .map(|v| v / 100.0),
            size: keys.get("Size").and_then(|v| v.trim().parse::<u32>().ok()),
            trigger: keys.get("Trigger").and_then(|v| v.trim().parse::<u32>().ok()),
        }
    }

    fn parse_zone_coords(coords: &str) -> Vec<(u32, u32)> {
        let parse = |v: &str| v.trim().parse::<u32>().unwrap();
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
        let contents = contents + "\n" + &fs::read_dir("/etc/zm/conf.d")?
            .filter_map(Result::ok)
            .map(|entry| fs::read_to_string(entry.path()))
            .filter_map(Result::ok)
            .fold(String::new(), |a, b| a + "\n" + &b);  // O(n**2)

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
