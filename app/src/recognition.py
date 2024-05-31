import os
import warnings
import json
import numpy as np
import time

warnings.filterwarnings("ignore")


class Numbers:
    """detect and classify on video or classify on cut bboxes players' numbers
    TODO:
    1) add embedding outputs (target file - all_boxes.json) after detection (dino or pretrained resnet) -- done
    2) for classifier - work mode after tracker -- done
    3)
    a) arcface or triplet
    b) repair correction method"""

    def __init__(self, input_dir: str, output_dir: str, weights: str, emb_weights: str, yolo_model: str, team=None):
        self.team = team
        self.weights = os.path.join(os.path.dirname(__file__), weights)
        self.emb_weights = os.path.join(os.path.dirname(__file__), emb_weights)
        self.num_classes = 100
        self.root = os.path.join(os.path.dirname(__file__), input_dir)
        self.output_dir = os.path.join(os.path.dirname(__file__), output_dir)
        self.yolo = None
        self.device = None
        self.pic_transform = None

    def detect(self, in_folder: str, frames: list, iou: float) -> str:
        """load video and detect people on chosen frames, prepare 2 dictionaries: boxes more than 64x64 px for
        classifier and all boxes at frame - for tracker"""
        time.sleep(1)
        all_boxes = {}
        vname = 'test-detection'
        for i in range(len(frames)):
            # if i not in frames
            all_boxes[f'{i}_0'] = {'bbox': list(map(int, np.random.randint(100, 500, 4))),
                                   'embedding': list(map(int, np.random.randint(50, 100, 8)))}
            with open(os.path.join(self.output_dir, f"{vname}_all_boxes.json"), "w+") as fid:
                json.dump(all_boxes, fid)
        return os.path.join(self.output_dir, f"{vname}_all_boxes.json")

    def predict_after(self, class_threshold: float, ann_path: str, video_path: str, test=False) -> tuple:
        """this method allows usage of the classifier after tracking
        it loads frames of video, crops them according to tracking results and recognizes numbers"""
        time.sleep(1)
        with open(ann_path) as fid:
            track_results = json.load(fid)
        numbers = []
        confs = []
        for pl in track_results['tracking']:
            pred = 42
            cfs = np.random.random()
            if cfs < class_threshold:
                pred = 'unknown'
            pl['predicted_number'] = pred
            numbers.append(pred)
            confs.append(cfs)
        with open(ann_path.split('.')[0] + '_num.json', "w+") as fid_r:
            json.dump(track_results, fid_r)
        return track_results
