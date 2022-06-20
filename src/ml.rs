use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use opencv::core::{Mat, MatTraitConst, MatTraitConstManual, Rect, Vector, CV_8U};
use opencv::dnn::{
    blob_from_image, nms_boxes, read_net, LayerTraitConst, Net, NetTrait, NetTraitConst,
};
use opencv::types::{VectorOfMat, VectorOfRect};

#[derive(Clone, Debug)]
pub struct Detection {
    pub confidence: f32,
    pub class_id: i32,
    pub bounding_box: Rect,
}

impl Detection {
    fn confidence_pct(&self) -> u8 {
        (self.confidence * 100.0) as u8
    }
}

impl Hash for Detection {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u8(self.confidence_pct());
        state.write_i32(self.class_id);
        state.write_i32(self.bounding_box.width);
        state.write_i32(self.bounding_box.height);
        state.write_i32(self.bounding_box.x);
        state.write_i32(self.bounding_box.y);
    }
}

impl PartialEq<Self> for Detection {
    fn eq(&self, other: &Self) -> bool {
        self.confidence_pct() == other.confidence_pct()
            && self.class_id == other.class_id
            && self.bounding_box == other.bounding_box
    }
}

impl Eq for Detection {}

pub struct YoloV4Tiny {
    net: Net,
    confidence_threshold: f32,
    nms_threshold: f32,
    size: u32,

    out_names: Vector<String>,
}

impl YoloV4Tiny {
    pub fn new(confidence_threshold: f32, size: u32, use_cuda: bool) -> opencv::Result<YoloV4Tiny> {
        let mut net = read_net("yolov4-tiny.weights", "yolov4-tiny.cfg", "")?;
        if use_cuda {
            net.set_preferable_target(opencv::dnn::DNN_TARGET_CUDA)?;
            net.set_preferable_backend(opencv::dnn::DNN_BACKEND_CUDA)?;
        } else {
            net.set_preferable_target(opencv::dnn::DNN_TARGET_CPU)?;
            net.set_preferable_backend(opencv::dnn::DNN_BACKEND_OPENCV)?;
        }

        let out_names = net.get_unconnected_out_layers_names()?;
        let out_layers = net.get_unconnected_out_layers()?;
        let out_layer_type = net.get_layer(out_layers.get(0).unwrap()).unwrap().typ();
        assert_eq!(out_layer_type, "Region");

        Ok(YoloV4Tiny {
            net,
            size,
            out_names,
            confidence_threshold,
            nms_threshold: 0.4,
        })
    }

    pub fn infer(&mut self, image: &Mat) -> opencv::Result<Vec<Detection>> {
        let size = self.size as i32;
        let size = (size, size);
        let mean = (0.0, 0.0, 0.0);
        let blob = blob_from_image(&image, 1.0, size.into(), mean.into(), false, false, CV_8U)?;
        let scale = 1.0 / 255.0;
        self.net.set_input(&blob, "", scale, mean.into())?;

        let outs = {
            let mut outs = VectorOfMat::new();
            self.net.forward(&mut outs, &self.out_names)?;
            outs
        };

        let image_width = image.cols() as f32;
        let image_height = image.rows() as f32;

        let detections: Vec<Detection> = outs
            .iter()
            .map(|out| {
                // Network produces output blob with a shape NxC where N is a number of
                // detected objects and C is a number of classes + 4 where the first 4
                // numbers are [center_x, center_y, width, height]

                (0..out.rows())
                    .map(move |i| {
                        let row = out.at_row::<f32>(i).unwrap();

                        let get_bounding_box = |row: &[f32]| -> Rect {
                            let (center_x, center_y) = (row[0], row[1]);
                            let (width, height) = (row[2], row[3]);

                            let center_x = (center_x * image_width) as i32;
                            let center_y = (center_y * image_height) as i32;
                            let width = (width * image_width) as i32;
                            let height = (height * image_height) as i32;

                            let left_edge = (center_x - width / 2).max(0);
                            let top_edge = (center_y - height / 2).max(0);

                            Rect::new(left_edge, top_edge, width, height)
                        };

                        let get_class = |row: &[f32]| {
                            let class = row[4..]
                                .iter()
                                //.cloned()
                                .zip(1..) // 1.. for 1-based class index, 0.. for 0-based
                                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            let (&confidence, class_id) = class.unwrap();
                            (confidence, class_id)
                        };

                        let (confidence, class_id) = get_class(row);
                        let bounding_box = get_bounding_box(row);

                        Detection {
                            confidence,
                            class_id,
                            bounding_box,
                        }
                    })
                    .filter(|detection| detection.confidence >= self.confidence_threshold)
            })
            .flatten()
            .collect();

        // Perform NMS filtering
        let mut class2detections: HashMap<i32, Vec<&Detection>> = HashMap::new();
        for detection in &detections {
            let dets = class2detections
                .entry(detection.class_id)
                .or_insert_with(Vec::new);
            dets.push(&detection);
        }

        let mut nms_detections = vec![];

        for (_, detections) in &class2detections {
            let bounding_boxes: VectorOfRect =
                detections.iter().map(|det| det.bounding_box).collect();
            let confidences: Vector<f32> = detections.iter().map(|det| det.confidence).collect();
            let mut chosen_indices = Vector::new();
            nms_boxes(
                &bounding_boxes,
                &confidences,
                self.confidence_threshold,
                self.nms_threshold,
                &mut chosen_indices,
                1.0,
                0,
            )?;

            for index in chosen_indices {
                nms_detections.push(detections[index as usize].clone());
            }
        }

        Ok(nms_detections)
    }
}
