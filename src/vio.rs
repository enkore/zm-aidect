use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};

use anyhow::{anyhow, Result};
use opencv::core::{Mat, MatTraitManual};
use serde::Deserialize;

#[derive(Debug, Deserialize, Eq, PartialEq)]
struct ProbeOutput {
    streams: Vec<VideoProperties>,
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
pub struct VideoProperties {
    codec_name: String,
    avg_frame_rate: String,
    pub width: u32,
    pub height: u32,
}

impl VideoProperties {
    pub fn get_fps(&self) -> f32 {
        let (a, b) = self.avg_frame_rate.split_once('/').unwrap();
        let (a, b) = (a.parse::<f32>().unwrap(), (b.parse::<f32>().unwrap()));
        a / b
    }
}

impl ToString for VideoProperties {
    fn to_string(&self) -> String {
        format!(
            "{}x{} {:.1} fps ({})",
            self.width,
            self.height,
            self.get_fps(),
            self.codec_name
        )
    }
}

pub fn properties(path: &Path) -> Result<VideoProperties> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-select_streams",
            "v:0",
            "-show_streams",
        ])
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err(anyhow!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let output = String::from_utf8(output.stdout)?;
    let mut output: ProbeOutput = serde_json::from_str(&output)?;
    Ok(output.streams.remove(0))
}

pub struct ImageStream {
    width: u32,
    height: u32,
    ffmpeg: Child,
}

impl Iterator for ImageStream {
    type Item = Mat;

    fn next(&mut self) -> Option<Self::Item> {
        let mut mat = Mat::new_size_with_default(
            (self.width as i32, self.height as i32).into(),
            opencv::core::CV_8UC3,
            0.into(),
        )
        .ok()?;
        let mut slice = mat.data_bytes_mut().expect("Got an non-continuous Mat for some reason?");
        let stdout = self.ffmpeg.stdout.as_mut()?;
        stdout.read_exact(&mut slice).ok()?;
        return Some(mat);
    }
}

pub fn stream_file(path: &Path, width: u32, height: u32, framerate: f32) -> Result<ImageStream> {
    let video_size = format!("{}x{}", width, height);
    let framerate = framerate.to_string();
    let ffmpeg = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            &video_size,
            "-sws_flags",
            "neighbor",
            "-r",
            &framerate,
            "-",
        ])
        .stdout(Stdio::piped())
        .spawn()?;

    Ok(ImageStream {
        width,
        height,
        ffmpeg,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn test_parse_ffprobe() -> Result<()> {
        let ffprobe = r#"
        {
            "streams": [
                {
                    "index": 0,
                    "codec_name": "h264",
                    "codec_long_name": "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
                    "profile": "High",
                    "codec_type": "video",
                    "codec_time_base": "7491/449600",
                    "codec_tag_string": "avc1",
                    "codec_tag": "0x31637661",
                    "width": 1920,
                    "height": 1080,
                    "coded_width": 1920,
                    "coded_height": 1088,
                    "closed_captions": 0,
                    "has_b_frames": 0,
                    "pix_fmt": "yuvj420p",
                    "level": 42,
                    "color_range": "pc",
                    "color_space": "bt470bg",
                    "color_transfer": "bt709",
                    "color_primaries": "bt470bg",
                    "chroma_location": "left",
                    "refs": 1,
                    "is_avc": "true",
                    "nal_length_size": "4",
                    "r_frame_rate": "100/1",
                    "avg_frame_rate": "2248/74",
                    "time_base": "1/90000",
                    "start_pts": 0,
                    "start_time": "0.000000",
                    "duration_ts": 6741900,
                    "duration": "74.910000",
                    "bit_rate": "3046762",
                    "bits_per_raw_sample": "8",
                    "disposition": {
                        "default": 1,
                        "dub": 0,
                        "original": 0,
                        "comment": 0,
                        "lyrics": 0,
                        "karaoke": 0,
                        "forced": 0,
                        "hearing_impaired": 0,
                        "visual_impaired": 0,
                        "clean_effects": 0,
                        "attached_pic": 0,
                        "timed_thumbnails": 0
                    },
                    "tags": {
                        "language": "und",
                        "handler_name": "VideoHandler"
                    }
                }
            ]
        }"#;

        assert_eq!(
            serde_json::from_str::<ProbeOutput>(&ffprobe)?,
            ProbeOutput {
                streams: vec![VideoProperties {
                    codec_name: "h264".to_string(),
                    avg_frame_rate: "2248/74".to_string(),
                    width: 1920,
                    height: 1080,
                }]
            }
        );
        Ok(())
    }
}
