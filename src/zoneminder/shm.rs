use std::mem::size_of;
use libc::{c_char, c_double, time_t, timeval};

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[allow(dead_code)]
pub(super) struct MonitorSharedData {
    // size and offsets in ZM code is all wrong
    pub size: u32,
    pub last_write_index: i32,
    pub last_read_index: i32,
    state: u32,
    capture_fps: c_double,
    analysis_fps: c_double,
    pub last_event_id: u64,
    action: u32,
    brightness: i32,
    hue: i32,
    colour: i32,
    contrast: i32,
    alarm_x: i32,
    alarm_y: i32,
    pub valid: u8,
    active: u8,
    signal: u8,
    pub format: SubpixelOrder,
    pub imagesize: u32,
    last_frame_score: u32,
    audio_frequency: u32,
    audio_channels: u32,

    startup_time: time_t,
    zmc_heartbeat_time: time_t,
    pub last_write_time: time_t,
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
#[allow(dead_code)]
pub(super) enum ColourType {
    GRAY8 = 1,
    RGB24 = 3,
    RGB32 = 4,
}

#[derive(Copy, Clone, Debug)]
#[repr(u8)]
#[allow(dead_code)]
pub(super) enum SubpixelOrder {
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
#[allow(dead_code)]
pub(super) enum TriggerState {
    TriggerCancel,
    TriggerOn,
    TriggerOff,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[allow(dead_code)]
pub(super) struct MonitorTriggerData {
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
#[allow(dead_code)]
pub(super) struct MonitorVideoStoreData {
    // size in ZM is wrong
    size: u32,
    padding: u32, // padding here, not in ZM which is wrong
    current_event: u64,
    event_file: [c_char; 4096],
    recording: timeval,
}

const _: [u8; 4128] = [0; size_of::<MonitorVideoStoreData>()];
