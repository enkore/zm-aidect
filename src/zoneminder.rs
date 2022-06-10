use std::error::Error;
use std::fs::{File, OpenOptions};
use std::{fs, io, slice};
use std::collections::HashMap;
use std::io::ErrorKind;
use std::mem::size_of;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use libc::timeval;
use opencv::core::{CV_8UC4, Mat, MatTrait, MatTraitConst};

mod shm;

#[allow(dead_code)]
pub struct Monitor {
    mmap_path: String,
    file: File,
    ino: u64,

    pub image_buffer_count: u32,  // XXX

    trigger_data_offset: usize,
    videostore_data_offset: usize,
    shared_timestamps_offset: usize,
    shared_images_offset: usize,
}

impl Monitor {
    pub fn connect(monitor_id: u32) -> io::Result<Monitor> {
        let mmap_path = format!("/dev/shm/zm.mmap.{}", monitor_id);
        let file = OpenOptions::new().read(true).write(true).open(&mmap_path)?;

        let image_buffer_count = 3;  // needs to be retrieved from the database

        let trigger_data_offset = size_of::<shm::MonitorSharedData>();
        let videostore_data_offset = trigger_data_offset + size_of::<shm::MonitorTriggerData>();
        let shared_timestamps_offset = videostore_data_offset + size_of::<shm::MonitorVideoStoreData>();
        let shared_images_offset = shared_timestamps_offset + image_buffer_count * size_of::<timeval>();
        let shared_images_offset = shared_images_offset + 64 - (shared_images_offset % 64);

        Ok(Monitor {
            mmap_path,
            ino: file.metadata()?.ino(),
            file,
            image_buffer_count: image_buffer_count as u32,
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
        assert_eq!(1280 * 720 * 4, token.size);
        self.check_file_stale()?;
        let mut mat = Mat::new_size_with_default((1280, 720).into(), CV_8UC4, 0.into())?;
        self.read_image_into(token, &mut mat)?;
        Ok(mat)
    }

    pub fn read_image_into(&self, token: ImageToken, mat: &mut Mat) -> Result<(), Box<dyn Error>> {
        assert_eq!(1280 * 720, mat.total());
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
struct ZoneMinderConf {
    db_host: String,
    db_name: String,
    db_user: String,
    db_password: String,
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
        }
    }

    fn parse_default() -> io::Result<ZoneMinderConf> {
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
";

        let parsed = ZoneMinderConf::parse_zm_conf(conf);
        assert_eq!(parsed.db_host, "localhost");
        assert_eq!(parsed.db_name, "zm");
        assert_eq!(parsed.db_user, "zmuser");
        assert_eq!(parsed.db_password, "zmpass");
    }
}
