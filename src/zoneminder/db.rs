use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use mysql::params;
use mysql::prelude::Queryable;
use opencv::core::Rect;

use crate::zoneminder::ZoneMinderConf;

trait ZoneMinderDB {
    fn connect_db(&self) -> mysql::Result<mysql::Conn>;
}

impl ZoneMinderDB for ZoneMinderConf {
    fn connect_db(&self) -> mysql::Result<mysql::Conn> {
        let builder = mysql::OptsBuilder::new()
            .ip_or_hostname(Some(&self.db_host))
            .db_name(Some(&self.db_name))
            .user(Some(&self.db_user))
            .pass(Some(&self.db_password));
        mysql::Conn::new(builder)
    }
}

pub fn update_event_notes(
    zm_conf: &ZoneMinderConf,
    event_id: u64,
    notes: &str,
) -> Result<()> {
    let mut db = zm_conf.connect_db()?;
    Ok(db.exec_drop(
        "UPDATE Events SET Notes = :notes WHERE Id = :id",
        params! {
            "id" => event_id,
            "notes" => notes,
        },
    )?)
}

#[derive(Debug)]
pub struct MonitorSettings {
    pub name: String,
    pub storage_id: u32,
    pub enabled: bool,
    pub width: u32,
    pub height: u32,
    pub colours: u32,
    pub image_buffer_count: u32,
    pub analysis_fps_limit: Option<f32>,
}

impl MonitorSettings {
    pub fn query(
        zm_conf: &ZoneMinderConf,
        monitor_id: u32,
    ) -> Result<MonitorSettings> {
        let mut db = zm_conf.connect_db()?;
        Ok(db.exec_map("SELECT Name, StorageId, Enabled, Width, Height, Colours, ImageBufferCount, AnalysisFPSLimit FROM Monitors WHERE Id = :id",
                       params! { "id" => monitor_id },
                       |(name, storage_id, enabled, width, height, colours, image_buffer_count, analysis_fps_limit)| {
                           MonitorSettings {
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
pub struct Event {
    pub id: u64,
    pub monitor_id: u32,
    pub name: String,
    pub max_score: u32,
    pub avg_score: u32,
    pub total_score: u32,
    default_video: String,
    start_datetime: String,  // local time, 2022-01-27 18:45:59

    storage: Storage,
}

impl Event {
    pub fn query(
        zm_conf: &ZoneMinderConf,
        event_id: u64,
    ) -> Result<Event> {
        let mut db = zm_conf.connect_db()?;

        let storage_id = db.exec_first("SELECT StorageId FROM Events WHERE Id = :id", params!{ "id" => event_id })?;
        let storage = get_storage_by_id(&mut db, storage_id.unwrap())?;

        // the "date time" handling here is janky af but sufficient for what's needed (only used to derive the file name)
        Ok(db.exec_map("SELECT Name, MonitorId, MaxScore, AvgScore, TotScore, DefaultVideo, CAST(StartDateTime AS CHAR) FROM Events WHERE Id = :id",
                       params! { "id" => event_id },
                       |(name, monitor_id, max_score, avg_score, total_score, default_video, start_datetime)| {
                           Event {
                               id: event_id,
                               name,
                               monitor_id,
                               max_score,
                               avg_score,
                               total_score,
                               default_video,
                               start_datetime,
                               storage: storage.clone(),
                           }
                       }
        )?.remove(0))
    }

    pub fn video_path(&self) -> Result<PathBuf> {
        if self.storage.storage_type != "local" {
            return Err(anyhow!("Unsupported storage type {} for event {}", self.storage.storage_type, self.id));
        }

        let event_path = match self.storage.scheme {
            StorageScheme::Deep => {
                let re = regex::Regex::new("[-: ]").unwrap();
                format!("{}/{}", re.replace_all(&self.start_datetime, "/"), self.id)
            },
            StorageScheme::Medium => format!("{}/{}", self.start_datetime.split_once(" ").unwrap().0, self.id),
            StorageScheme::Shallow => format!("{}", self.id)
        };

        let monitor_path = self.monitor_id.to_string();

        let path: PathBuf = [&self.storage.path, &monitor_path, &event_path, &self.default_video].iter().collect();
        Ok(path)
    }
}

#[derive(Debug, Copy, Clone)]
enum StorageScheme {
    Deep,
    Medium,
    Shallow,
}

impl TryFrom<&str> for StorageScheme {
    type Error = anyhow::Error;

    fn try_from(input: &str) -> std::result::Result<StorageScheme, Self::Error> {
        Ok(match input {
            "Deep" => StorageScheme::Deep,
            "Medium" => StorageScheme::Medium,
            "Shallow" => StorageScheme::Shallow,
            _ => return Err(anyhow!("Invalid/unknown storage scheme {}", input)),
        })
    }
}

#[derive(Debug, Clone)]
struct Storage {
    id: u64,
    name: String,
    path: String,
    storage_type: String,
    scheme: StorageScheme,
}

fn get_storage_by_id(db: &mut mysql::Conn, storage_id: u64) -> Result<Storage> {
    //let mut db = zm_conf.connect_db()?;
    Ok(db.exec_map("SELECT Name, Path, Type, Scheme FROM Storage WHERE Id = :id",
                   params! { "id" => storage_id },
                   |(name, path, storage_type, scheme)| -> Result<Storage> {
                       let scheme: String = scheme;
                       Ok(Storage {
                           id: storage_id,
                           name,
                           path,
                           storage_type,
                           scheme: StorageScheme::try_from(scheme.as_str())?,
                       })
                   }
    )?.remove(0)?)
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
    ) -> Result<ZoneConfig> {
        let mut db = zm_conf.connect_db()?;
        let dbzone = db.exec_first(
            "SELECT Name, Type, Coords FROM Zones WHERE MonitorId = :id AND Name LIKE \"aidect%\"",
            params! { "id" => monitor_id },
        )?;
        let dbzone: mysql::Row = dbzone.ok_or(anyhow!("No aidect zone found for monitor {}", monitor_id))?;

        Ok(ZoneConfig::parse(
            &dbzone.get::<String, &str>("Name").unwrap(),
            &dbzone.get::<String, &str>("Coords").unwrap(),
        ))
    }

    fn parse(name: &str, coords: &str) -> ZoneConfig {
        ZoneConfig {
            shape: Self::parse_zone_coords(coords),
            ..Self::parse_zone_name(name)
        }
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

#[cfg(test)]
mod tests {
    use super::*;

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
