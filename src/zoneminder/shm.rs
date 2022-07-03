use std::io::Read;
use std::mem::{align_of, size_of};
use std::os::unix::fs::FileExt;
use std::slice;

use anyhow::{anyhow, Context, Result};
use lazy_static::lazy_static;
use libc::time_t;
use regex::Regex;

// TODO: panic! wrapper which adds a bit that this requires maintainer attention

#[derive(Debug, Eq, PartialEq)]
struct Type {
    size: usize,
    alignment: usize,
}

impl Type {
    fn new<T>() -> Type {
        Type {
            size: size_of::<T>(),
            alignment: align_of::<T>(),
        }
    }

    fn array_of(&self, num_elements: usize) -> Type {
        Type {
            size: self.size * num_elements,
            alignment: self.alignment,
        }
    }
}

fn parse_basic_typename(typename: &str) -> Type {
    match typename {
        "uint8" => Type::new::<u8>(),
        "int8" => Type::new::<i8>(),
        "uint32" => Type::new::<u32>(),
        "int32" => Type::new::<i32>(),
        "uint64" => Type::new::<u64>(),
        "int64" => Type::new::<i64>(),
        "float" => Type::new::<f32>(),
        "double" => Type::new::<f64>(),
        "time_t64" => Type::new::<time_t>(),
        _ => panic!(
            "Unhandled ABI type in Memory.pm shm definition: {}",
            typename
        ),
    }
}

fn parse_typename(typename: &str) -> Type {
    match typename.split_once('[') {
        None => parse_basic_typename(typename),
        Some((basic_typename, array_size)) => {
            let t = parse_basic_typename(basic_typename);
            assert!(array_size.ends_with(']'));
            let array_size = &array_size[0..array_size.len() - 1];
            let elements = array_size
                .parse::<usize>()
                .with_context(|| {
                    format!(
                        "Could not parse array size in Memory.pm shm definition: {}",
                        typename
                    )
                })
                .unwrap();
            t.array_of(elements)
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct ParsedField {
    name: String,
    typ: Type,
}

fn parse_field_definition(line: &str) -> ParsedField {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"(\w+)\s+=> \{ type=>'([a-z0-9_\[\]]+)'").unwrap();
    }
    let m = RE
        .captures(line)
        .ok_or(anyhow!(
            "Could not parse field definition in Memory.pm shm definition: {:?}",
            line
        ))
        .unwrap();
    ParsedField {
        name: m[1].to_string(),
        typ: parse_typename(&m[2]),
    }
}

#[derive(Debug, Eq, PartialEq)]
struct ParsedStruct {
    name: String,
    fields: Vec<ParsedField>,
}

#[derive(Debug, Eq, PartialEq)]
struct Field {
    name: String,
    offset: usize,
    typ: Type,
}

#[derive(Debug, Eq, PartialEq)]
struct Struct {
    name: String,
    size: usize,
    fields: Vec<Field>,
}

impl ParsedStruct {
    fn calculate_offsets(self) -> Struct {
        let mut offset = 0;
        let mut fields = vec![];

        for ParsedField { name, typ } in self.fields {
            let field = Field {
                offset: align_to(offset, typ.alignment),
                name,
                typ,
            };
            offset = field.offset + field.typ.size;
            fields.push(field);
        }

        Struct {
            name: self.name,
            size: offset,
            fields,
        }
    }
}

fn align_to(offset: usize, alignment: usize) -> usize {
    if offset % alignment == 0 {
        offset
    } else {
        offset + alignment - (offset % alignment)
    }
}

fn parse_struct_definition(input: &mut std::str::Lines) -> Option<ParsedStruct> {
    lazy_static! {
        static ref RE: Regex =
            Regex::new(r"\w+\s+=> \{ type=>'(\w+)', seq=>\$mem_seq\+\+, '?contents'?").unwrap();
    }
    let struct_def = input.next().expect("Empty struct definition in Memory.pm");
    if struct_def.trim_start().starts_with("end =>") {
        return None;
    }

    let m = RE
        .captures(struct_def)
        .ok_or(anyhow!(
            "Could not parse struct definition in Memory.pm shm definition: {:?}",
            struct_def
        ))
        .unwrap();

    let mut fields = vec![];
    loop {
        let line = input.next().expect("Unexpected EOR in Memory.pm");
        let line = line.trim_start();
        if line == "}" {
            continue;
        }
        if line == "}," {
            break;
        }
        fields.push(parse_field_definition(line));
    }

    Some(ParsedStruct {
        name: m[1].to_string(),
        fields,
    })
}

fn parse_memory_pm(input: &str) -> ParsedStruct {
    let re = Regex::new(r"(?ms)our \$mem_data = \{\n(.*?)};").unwrap();
    let m = re
        .captures(input)
        .expect("No shm definitions found in Memory.pm");

    let mut lines = m[1].lines();
    let mut fields = vec![];
    loop {
        if let Some(s) = parse_struct_definition(&mut lines) {
            fields.extend(s.fields.into_iter().map(|f| ParsedField {
                name: format!("{}::{}", s.name, f.name),
                ..f
            }));
        } else {
            break;
        }
    }

    // Memory.pm does not define this struct, but we need to read this field to calculate
    // the offset of the timestamps and the shared image buffer.
    fields.push(ParsedField {
        name: "VideoStoreData::size".into(),
        typ: Type::new::<u32>(),
    });

    ParsedStruct {
        name: "memory".into(),
        fields,
    }
}

fn read_memory_pm<T: Read>(mut input: T) -> Result<Struct> {
    let input = {
        let mut contents = String::new();
        input.read_to_string(&mut contents)?;
        contents
    };
    Ok(parse_memory_pm(&input).calculate_offsets())
}

lazy_static! {
    static ref LAYOUT: Struct = {
        let file = std::fs::File::open("/usr/share/perl5/ZoneMinder/Memory.pm").expect("Failed to open ZoneMinder Memory.pm - ZM not installed or installed in unknown location.");
        read_memory_pm(file).unwrap()
    };
}

#[non_exhaustive]
pub struct MonitorShm<T: Read> {
    pub file: T,
    pub videostore_size: u32,
}

impl<File: FileExt + Read> MonitorShm<File> {
    pub fn new(file: File) -> Result<MonitorShm<File>> {
        let mut mshm = MonitorShm {
            file,
            videostore_size: 0,
        };
        mshm.videostore_size = mshm.read_field("VideoStoreData::size")?;
        Ok(mshm)
    }

    fn lookup_field(&self, name: &str) -> &Field {
        for field in LAYOUT.fields.iter() {
            if field.name == name {
                return field;
            }
        }
        panic!("Field not found in Memory.pm: {name}");
    }

    fn typecheck<T>(&self, field: &Field) {
        let typ = Type::new::<T>();
        if field.typ != typ {
            panic!(
                "Mismatched field type for {} (wanted: {typ:?}, got: {:?}",
                field.name, field.typ
            );
        }
    }

    pub fn read_field<T>(&self, name: &str) -> Result<T> {
        let field = self.lookup_field(name);
        self.typecheck::<T>(field);
        self.pread(field.offset)
    }

    pub fn write_field<T>(&self, name: &str, value: &T) -> Result<()> {
        let field = self.lookup_field(name);
        self.typecheck::<T>(field);
        self.pwrite(field.offset, value)
    }

    pub fn write_string(&self, name: &str, value: &str) -> Result<()> {
        let field = self.lookup_field(name);
        let terminated_len = value.len() + 1;
        assert!(field.typ.size >= terminated_len);
        let mut s = String::with_capacity(terminated_len);
        s.push_str(value);
        s.push('\0');
        self.file.write_all_at(s.as_bytes(), field.offset as u64)?;
        Ok(())
    }

    fn pread<T>(&self, offset: usize) -> Result<T> {
        let mut buf = Vec::new();
        buf.resize(size_of::<T>(), 0);
        self.file.read_exact_at(&mut buf, offset as u64)?;
        unsafe { Ok(std::ptr::read(buf.as_ptr() as *const _)) }
    }

    fn pwrite<T>(&self, offset: usize, data: &T) -> Result<()> {
        let data = unsafe { slice::from_raw_parts(data as *const T as *const u8, size_of::<T>()) };
        self.file.write_all_at(data, offset as u64)?;
        Ok(())
    }
}

#[non_exhaustive]
pub(super) struct ShmField;

// TODO: This should be an enum and we should associate the name and expected type internally,
// TODO: so that all fields we may can be validated after parsing Memory.pm
impl ShmField {
    pub const LAST_WRITE_INDEX: &'static str = "SharedData::last_write_index";
    pub const STATE: &'static str = "SharedData::state";
    pub const LAST_EVENT_ID: &'static str = "SharedData::last_event";
    pub const VALID: &'static str = "SharedData::valid";
    pub const FORMAT: &'static str = "SharedData::format";
    pub const IMAGESIZE: &'static str = "SharedData::imagesize";

    pub const TRIGGER_STATE: &'static str = "TriggerData::trigger_state";
    pub const TRIGGER_SCORE: &'static str = "TriggerData::trigger_score";
    pub const TRIGGER_CAUSE: &'static str = "TriggerData::trigger_cause";
    pub const TRIGGER_TEXT: &'static str = "TriggerData::trigger_text";
    pub const TRIGGER_SHOWTEXT: &'static str = "TriggerData::trigger_showtext";

    pub const SHARED_SIZE: &'static str = "SharedData::size";
    pub const TRIGGER_SIZE: &'static str = "TriggerData::size";
    pub const VIDEOSTORE_SIZE: &'static str = "VideoStoreData::size";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_typename() {
        assert_eq!(parse_typename("int32"), Type::new::<i32>());
        assert_eq!(parse_typename("int32[44]"), Type::new::<i32>().array_of(44));
    }

    #[test]
    #[should_panic]
    fn test_parse_typename_panic() {
        parse_typename("int32[44x]");
    }

    #[test]
    #[should_panic]
    fn test_parse_typename_panic2() {
        parse_typename("int32[44");
    }

    #[test]
    fn test_parse_field_definition() {
        assert_eq!(
            parse_field_definition("  size             => { type=>'uint32', seq=>$mem_seq++ },"),
            ParsedField {
                name: "size".into(),
                typ: Type::new::<u32>(),
            }
        );
        assert_eq!(
            parse_field_definition("  size             => { type=>'uint32[5]', seq=>$mem_seq++ },"),
            ParsedField {
                name: "size".into(),
                typ: Type::new::<u32>().array_of(5),
            }
        );
    }

    #[test]
    fn test_parse_struct_definition() {
        assert_eq!(
            parse_struct_definition(
                &mut r#"trigger_data => { type=>'TriggerData', seq=>$mem_seq++, 'contents'=> {
    size             => { type=>'uint32', seq=>$mem_seq++ },
    trigger_cause    => { type=>'int8[32]', seq=>$mem_seq++ },
  }
  },"#
                .lines()
            )
            .unwrap(),
            ParsedStruct {
                name: "TriggerData".into(),
                fields: vec![
                    ParsedField {
                        name: "size".into(),
                        typ: Type::new::<u32>()
                    },
                    ParsedField {
                        name: "trigger_cause".into(),
                        typ: Type::new::<i8>().array_of(32)
                    },
                ],
            }
        );
    }

    const INPUT: &str = "our $mem_seq = 0;

our $mem_data = {
  shared_data => { type=>'SharedData', seq=>$mem_seq++, contents=> {
    size             => { type=>'uint32', seq=>$mem_seq++ },
    startup_time     => { type=>'time_t64', seq=>$mem_seq++ },
    audio_fifo       => { type=>'int8[64]', seq=>$mem_seq++ },
  }
  },
  trigger_data => { type=>'TriggerData', seq=>$mem_seq++, 'contents'=> {
    size             => { type=>'uint32', seq=>$mem_seq++ },
    trigger_cause    => { type=>'int8[32]', seq=>$mem_seq++ },
  }
  },
  end => { seq=>$mem_seq++, size=>0 }
};

our $mem_size = 0;

sub zmMemInit {
";

    #[test]
    fn test_parse_memory_pm() {
        assert_eq!(
            parse_memory_pm(INPUT),
            vec![
                ParsedStruct {
                    name: "SharedData".into(),
                    fields: vec![
                        ParsedField {
                            name: "size".into(),
                            typ: Type::new::<u32>()
                        },
                        ParsedField {
                            name: "startup_time".into(),
                            typ: Type::new::<time_t>()
                        },
                        ParsedField {
                            name: "audio_fifo".into(),
                            typ: Type::new::<i8>().array_of(64)
                        },
                    ],
                },
                ParsedStruct {
                    name: "TriggerData".into(),
                    fields: vec![
                        ParsedField {
                            name: "size".into(),
                            typ: Type::new::<u32>()
                        },
                        ParsedField {
                            name: "trigger_cause".into(),
                            typ: Type::new::<i8>().array_of(32)
                        },
                    ],
                },
            ]
        );
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(6, 8), 8);
        assert_eq!(align_to(8, 8), 8);
        assert_eq!(align_to(0, 8), 0);
        assert_eq!(align_to(7, 1), 7);
    }

    #[test]
    fn test_read_memory_pm() {
        assert_eq!(
            read_memory_pm(INPUT.as_bytes()).unwrap(),
            vec![
                Struct {
                    name: "SharedData".into(),
                    size: 4 + 4 + 8 + 64,
                    fields: vec![
                        Field {
                            name: "size".into(),
                            typ: Type::new::<u32>(),
                            offset: 0,
                        },
                        Field {
                            name: "startup_time".into(),
                            typ: Type::new::<time_t>(),
                            offset: align_of::<time_t>(),
                        },
                        Field {
                            name: "audio_fifo".into(),
                            typ: Type::new::<i8>().array_of(64),
                            offset: align_of::<time_t>() + 8,
                        },
                    ],
                },
                Struct {
                    name: "TriggerData".into(),
                    size: 4 + 32,
                    fields: vec![
                        Field {
                            name: "size".into(),
                            typ: Type::new::<u32>(),
                            offset: 0,
                        },
                        Field {
                            name: "trigger_cause".into(),
                            typ: Type::new::<i8>().array_of(32),
                            offset: 4,
                        },
                    ],
                },
            ]
        );
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
#[allow(dead_code)]
pub(super) enum MonitorState {
    Unknown = 0,
    Idle,
    Prealarm, // Likely when there are alarm frames but not enough to trigger an event
    Alarm,    // I believe "current" frame is an alarm frame
    Alert,    // Current frame is not an alarm frame, but we're still in an alarmed state
    Tape,     // I think this is the idle state of Mocord and Record
}

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
