use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::mem::size_of;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use libc::timeval;
use log::error;
use opencv::core::{Mat, MatTraitConst, MatTraitManual};

use crate::zoneminder::db::MonitorSettings;

pub mod db;
mod shm;

pub trait MonitorTrait<'this> {
    // for lack of a better term
    type ImageIterator: Iterator<Item = Result<Image>>;

    fn stream_images(&'this self) -> Result<Self::ImageIterator>;

    fn is_idle(&self) -> Result<bool>;

    fn trigger(&self, cause: &str, description: &str, score: u32) -> Result<u64>;

    fn id(&self) -> u32;
}

pub struct Monitor<'zmconf> {
    monitor_id: u32,
    zm_conf: &'zmconf ZoneMinderConf,

    mmap_path: String,
    ino: u64,
    shm: shm::MonitorShm<File>,
}

impl<'this> MonitorTrait<'this> for Monitor<'this> {
    type ImageIterator = ImageStream<'this>;

    fn stream_images(&'this self) -> Result<Self::ImageIterator> {
        let state = self.read()?;
        let settings = MonitorSettings::query(self.zm_conf, self.monitor_id)?;
        let image_buffer_count = settings.image_buffer_count;

        // now that we have the image buffer size we can figure the dynamic offsets out
        let shared_timestamps_offset = self.shm.read_field::<u32>(shm::ShmField::SHARED_SIZE)?
            + self.shm.read_field::<u32>(shm::ShmField::TRIGGER_SIZE)?
            + self.shm.read_field::<u32>(shm::ShmField::VIDEOSTORE_SIZE)?;
        let shared_images_offset =
            shared_timestamps_offset as usize + image_buffer_count as usize * size_of::<timeval>();
        let shared_images_offset = shared_images_offset + 64 - (shared_images_offset % 64);

        Ok(ImageStream {
            width: settings.width,
            height: settings.height,
            image_buffer_count,
            monitor: self,
            last_read_index: image_buffer_count,
            image_size: state.imagesize,
            format: state.format,
            shared_images_offset: shared_images_offset as u64,
        })
    }

    fn is_idle(&self) -> Result<bool> {
        Ok(self.read()?.state == shm::MonitorState::Idle)
    }

    /// Mark at least one frame as an alarm frame with the given score. Wait for event to be created,
    /// then return event ID. Does not necessarily cause creation of a new event.
    fn trigger(&self, cause: &str, description: &str, score: u32) -> Result<u64> {
        let poll_interval = 10;
        self.set_trigger(cause, description, score)?;
        for n in 0.. {
            let state = self.read()?.state;
            // Alarm sorta implies that we just triggered an alarm frame, while
            // Alert sorta implies there's an on-going event.
            // Wait for Alarm state to become active so that the frame is marked.
            if state == shm::MonitorState::Alarm {
                break;
            }
            std::thread::sleep(Duration::from_millis(poll_interval));
            if n > 500 {
                error!("Waited {} ms for zoneminder to notice our bulgy wulgy, giving up and canceling it :c", n * poll_interval);
            }
        }
        self.reset_trigger()?;
        Ok(self.read()?.last_event_id)
    }

    fn id(&self) -> u32 {
        self.monitor_id
    }
}

impl Monitor<'_> {
    pub fn connect(zm_conf: &ZoneMinderConf, monitor_id: u32) -> Result<Monitor> {
        let mmap_path = format!("{}/zm.mmap.{}", zm_conf.mmap_path, monitor_id);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&mmap_path)
            .with_context(|| {
                format!(
                    "Failed to open mmap file {} for monitor {}",
                    mmap_path, monitor_id
                )
            })?;

        Ok(Monitor {
            monitor_id,
            zm_conf,
            mmap_path,
            ino: file.metadata()?.ino(),
            shm: shm::MonitorShm::new(file)?,
        })
    }

    fn set_trigger(&self, cause: &str, description: &str, score: u32) -> Result<()> {
        self.shm.write_string(shm::ShmField::TRIGGER_CAUSE, cause)?;
        self.shm
            .write_string(shm::ShmField::TRIGGER_TEXT, description)?;
        self.shm.write_string(shm::ShmField::TRIGGER_SHOWTEXT, "")?;
        self.shm.write_field(shm::ShmField::TRIGGER_SCORE, &score)?;
        // all of this is terribly racy but pwritin' the data before the state change should reduce the odds of problems
        self.shm
            .write_field(shm::ShmField::TRIGGER_STATE, &shm::TriggerState::TriggerOn)
    }

    fn reset_trigger(&self) -> Result<()> {
        self.shm.write_string(shm::ShmField::TRIGGER_CAUSE, "")?;
        self.shm.write_string(shm::ShmField::TRIGGER_TEXT, "")?;
        self.shm.write_string(shm::ShmField::TRIGGER_SHOWTEXT, "")?;
        self.shm.write_field(shm::ShmField::TRIGGER_SCORE, &0)?;
        self.shm.write_field(
            shm::ShmField::TRIGGER_STATE,
            &shm::TriggerState::TriggerCancel,
        )
    }

    fn read(&self) -> Result<MonitorState> {
        if self.shm.read_field::<u8>(shm::ShmField::VALID)? == 0 {
            return Err(anyhow!("Monitor shm is not valid"));
        }
        self.check_file_stale()?;

        Ok(MonitorState {
            last_write_index: self.shm.read_field(shm::ShmField::LAST_WRITE_INDEX)?,
            state: self.shm.read_field(shm::ShmField::STATE)?,
            last_event_id: self.shm.read_field(shm::ShmField::LAST_EVENT_ID)?,
            format: self.shm.read_field(shm::ShmField::FORMAT)?,
            imagesize: self.shm.read_field(shm::ShmField::IMAGESIZE)?,
        })
    }

    fn check_file_stale(&self) -> Result<()> {
        // Additional sanity check, if the file-on-tmpfs is now a different file, we're definitely listening to a stranger.
        // ZM seems to be quite good about ensuring shared_data.valid gets flipped to 0 even when zmc crashes though.
        if fs::metadata(&self.mmap_path)
            .with_context(|| format!("Monitor mmap file {} does not exist", self.mmap_path))?
            .ino()
            != self.ino
        {
            return Err(anyhow!("Monitor shm fd is stale, must reconnect"));
        }
        Ok(())
    }
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

pub struct Image {
    image: Mat,
    format: shm::SubpixelOrder,
}

impl Image {
    pub fn convert_to_rgb24(self) -> Result<Mat> {
        let conversion = match self.format {
            shm::SubpixelOrder::NONE => Some(opencv::imgproc::COLOR_GRAY2RGB),
            shm::SubpixelOrder::RGB => None,
            shm::SubpixelOrder::BGR => Some(opencv::imgproc::COLOR_BGR2RGB),
            shm::SubpixelOrder::BGRA => Some(opencv::imgproc::COLOR_BGRA2RGB),
            shm::SubpixelOrder::RGBA => Some(opencv::imgproc::COLOR_RGBA2RGB),
            _ => panic!("Unsupported pixel format: {:?}", self.format),
        };
        self.convert(conversion)
    }

    #[allow(dead_code)]
    pub fn convert_to_rgb32(self) -> Result<Mat> {
        let conversion = match self.format {
            shm::SubpixelOrder::NONE => Some(opencv::imgproc::COLOR_GRAY2RGBA),
            shm::SubpixelOrder::RGB => Some(opencv::imgproc::COLOR_RGB2RGBA),
            shm::SubpixelOrder::BGR => Some(opencv::imgproc::COLOR_BGR2RGBA),
            shm::SubpixelOrder::BGRA => Some(opencv::imgproc::COLOR_BGRA2RGBA),
            shm::SubpixelOrder::RGBA => None,
            _ => panic!("Unsupported pixel format: {:?}", self.format),
        };
        self.convert(conversion)
    }

    #[allow(dead_code)]
    pub fn convert_to_gray(self) -> Result<Mat> {
        let conversion = match self.format {
            shm::SubpixelOrder::NONE => None,
            shm::SubpixelOrder::RGB => Some(opencv::imgproc::COLOR_RGB2GRAY),
            shm::SubpixelOrder::BGR => Some(opencv::imgproc::COLOR_BGR2GRAY),
            shm::SubpixelOrder::BGRA => Some(opencv::imgproc::COLOR_BGRA2GRAY),
            shm::SubpixelOrder::RGBA => Some(opencv::imgproc::COLOR_RGBA2GRAY),
            _ => panic!("Unsupported pixel format: {:?}", self.format),
        };
        self.convert(conversion)
    }

    fn convert(self, conversion: Option<i32>) -> Result<Mat> {
        if let Some(conversion) = conversion {
            let mut rgb_image = Mat::default();
            // You could do this in-place as well, though it's probably not worth it
            opencv::imgproc::cvt_color(&self.image, &mut rgb_image, conversion, 0)?;
            return Ok(rgb_image);
        }
        Ok(self.image)
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
    fn wait_for_image(&mut self) -> Result<Image> {
        loop {
            let state = self.monitor.read()?;
            let last_write_index = state.last_write_index as u32;
            if last_write_index != self.last_read_index
                && last_write_index != self.image_buffer_count
            {
                self.last_read_index = last_write_index;
                let image = self.read_image(last_write_index)?;
                return Ok(Image {
                    image,
                    format: self.format,
                });
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn read_image(&self, index: u32) -> Result<Mat> {
        assert_eq!(self.width * self.height * 4, self.image_size);
        let mut mat = Mat::new_size_with_default(
            (self.width as i32, self.height as i32).into(),
            zm_format_to_cv_format(self.format),
            0.into(),
        )?;
        self.read_image_into(index, &mut mat)?;
        Ok(mat)
    }

    fn read_image_into(&self, index: u32, mat: &mut Mat) -> Result<()> {
        assert_eq!(self.width * self.height, mat.total() as u32);
        assert_eq!(mat.typ(), zm_format_to_cv_format(self.format));
        self.monitor.check_file_stale()?;
        let mut slice = mat.data_bytes_mut()?;
        let image_offset = self.shared_images_offset as u64 + self.image_size as u64 * index as u64;
        self.monitor
            .shm
            .file
            .read_exact_at(&mut slice, image_offset)
            .with_context(|| "Failed to read image")?;
        Ok(())
    }
}

impl Iterator for ImageStream<'_> {
    type Item = Result<Image>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.wait_for_image())
    }
}

struct MonitorState {
    pub last_write_index: i32,
    pub state: shm::MonitorState,
    pub last_event_id: u64,
    pub format: shm::SubpixelOrder,
    pub imagesize: u32,
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

    pub fn parse_default() -> Result<ZoneMinderConf> {
        let zm_conf = "/etc/zm/zm.conf";
        let zm_conf_d = "/etc/zm/conf.d";
        let contents = fs::read_to_string(zm_conf).with_context(|| {
            format!("Failed to parse Zoneminder configuration file {}", zm_conf)
        })?;
        let contents = contents
            + "\n"
            + &fs::read_dir(zm_conf_d)
                .with_context(|| format!("Failed to read Zoneminder overrides from {}", zm_conf_d))?
                .filter_map(Result::ok)
                .map(|entry| fs::read_to_string(entry.path()))
                .filter_map(Result::ok)
                .fold(String::new(), |a, b| a + "\n" + &b); // O(n**2)

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

ZM_PATH_MAP=/dev/shm
";

        let parsed = ZoneMinderConf::parse_zm_conf(conf);
        assert_eq!(parsed.db_host, "localhost");
        assert_eq!(parsed.db_name, "zm");
        assert_eq!(parsed.db_user, "zmuser");
        assert_eq!(parsed.db_password, "zmpass");
        assert_eq!(parsed.mmap_path, "/dev/shm");
    }
}
