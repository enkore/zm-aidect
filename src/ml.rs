use std::collections::HashMap;
use opencv::core::{Mat, Vector, Rect, CV_8U, MatTraitConst, MatTraitConstManual};
use opencv::dnn::{blob_from_image, LayerTraitConst, Net, NetTrait, NetTraitConst, nms_boxes, read_net};
use opencv::types::{VectorOfMat, VectorOfRect};

#[derive(Clone, Debug)]
pub struct Detection {
    confidence: f32,
    class_id: i32,
    bounding_box: Rect,
}

pub struct YoloV4Tiny {
    net: Net,
    confidence_threshold: f32,
    nms_threshold: f32,
    size: i32,

    out_names: Vector<String>,
}

impl YoloV4Tiny {
    pub fn new(confidence_threshold: f32, size: i32) -> opencv::Result<YoloV4Tiny> {
        let mut net = read_net("yolov4-tiny.weights", "yolov4-tiny.cfg", "")?;
        net.set_preferable_target(0)?;

        let out_names = net.get_unconnected_out_layers_names()?;
        let out_layers = net.get_unconnected_out_layers()?;
        let out_layer_type = net.get_layer(out_layers.get(0).unwrap()).unwrap().typ();
        assert_eq!(out_layer_type, "Region");

        Ok(YoloV4Tiny {
            net,
            out_names,
            confidence_threshold: confidence_threshold,
            nms_threshold: 0.4,
            size: size,
        })
    }

    pub fn infer(&mut self, image: &Mat) -> opencv::Result<Vec<Detection>> {
        let size = (self.size, self.size);
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

                            let left_edge = center_x - width / 2;
                            let top_edge = center_y - height / 2;

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
