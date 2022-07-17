use lazy_static::lazy_static;
use prometheus::{
    register_counter, register_gauge, register_histogram, Counter, Encoder, Gauge, Histogram,
    TextEncoder, DEFAULT_BUCKETS,
};

lazy_static! {
    // DEFAULT_BUCKETS are a good fit here actually.
    pub static ref INFERENCE_DURATION: Histogram = register_histogram!("inference_duration", "Duration of ML inference in ms", DEFAULT_BUCKETS[0..].into()).unwrap();
    pub static ref INFERENCES: Counter = register_counter!("inferences", "Number of ML inferences").unwrap();
    pub static ref FPS: Gauge = register_gauge!("fps", "Current fps").unwrap();
    pub static ref FPS_DEVIATION: Gauge = register_gauge!("fps_deviation", "Current deviation from configured fps (positive=faster, negative=slower)").unwrap();
    pub static ref SIZE: Gauge = register_gauge!("size", "ML network input size").unwrap();
}

fn collect() -> String {
    let mut buffer = Vec::new();
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer.clone()).unwrap()
}

pub fn spawn_prometheus_client(address: String, port: u16) {
    std::thread::spawn(move || {
        let server = tiny_http::Server::http((address, port)).unwrap();
        for request in server.incoming_requests() {
            let response = tiny_http::Response::from_string(collect());
            let _ = request.respond(response);
        }
    });
}
