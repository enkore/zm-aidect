use std::error::Error;
use std::io;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;

pub fn trigger_autocancel(
    monitor_id: u32,
    cause: &str,
    text: &str,
    duration: u32,
) -> Result<u32, Box<dyn Error>> {
    assert!(cause.len() < 32);
    assert!(text.len() < 255);
    let response = send_zm_trigger_command(
        &format!("{}|on+{}|1|{}|{}|", monitor_id, duration, cause, text),
        true,
    )?;
    let error = format!("No/invalid reply: {}", response).to_string();
    let event_id = response.rsplit('|').next().ok_or(error.clone())?;
    let event_id = event_id.trim_end().parse().map_err(|_| error)?;
    Ok(event_id)
}

pub fn trigger_suspend(monitor_id: u32) -> io::Result<()> {
    send_zm_trigger_command(&format!("{}|disable|0|||", monitor_id), false)?;
    Ok(())
}

pub fn trigger_resume(monitor_id: u32) -> io::Result<()> {
    send_zm_trigger_command(&format!("{}|enable|0|||", monitor_id), false)?;
    Ok(())
}

/*fn zm_trigger_reset(monitor_id: u32) -> std::io::Result<()> {
    send_zm_trigger_command(&format!("{}|cancel|0|||", monitor_id))

}*/

fn send_zm_trigger_command(command: &str, response: bool) -> Result<String, io::Error> {
    let mut stream = UnixStream::connect("/run/zm/zmtrigger.sock")?;
    stream.write_all(command.as_bytes())?;

    if response {
        let mut stream = BufReader::new(stream);
        let mut response = String::new();
        stream.read_line(&mut response)?;
        return Ok(response);
    }
    Ok("".into())
}
