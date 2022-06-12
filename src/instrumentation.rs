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
}

fn collect() -> String {
    let mut buffer = Vec::new();
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer.clone()).unwrap()
}

pub fn spawn_prometheus_client(port: u16) {
    std::thread::spawn(move || {
        rouille::start_server_with_pool(("0.0.0.0", port), Some(1), move |request| {
            rouille::router!(request,
                (GET) (/stats) => {
                    rouille::Response::text(collect())
                },
                _ => rouille::Response::empty_404(),
            )
        })
    });
}
