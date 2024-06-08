import os
import torch
import sys
import gc
import tqdm
import json
import cv2
from datetime import date
import tempfile
import collections
import supervision as sv
import numpy as np
import pandas as pd
from utils import Numbers
from clear_game import write_new_file

from argparse import ArgumentParser
from deva.model.network import DEVA
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.ext.grounding_dino import get_grounding_dino_model, segment_with_text
from deva.ext.with_text_processor import process_frame_with_text


torch.autograd.set_grad_enabled(False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class TrackingPlayer():
    def __init__(self, clear_dir: str, json_data: str, final_dir: str, player_number=0, start_frame = 500, stop_frame = None, step_frames = 5):

        self.clear = os.path.join(os.path.dirname(__file__), clear_dir)
        self.recognized = os.path.join(os.path.dirname(__file__), json_data)
        self.final = os.path.join(os.path.dirname(__file__), final_dir)
        
        self.step_frames = step_frames
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.player_number  = player_number
        self.parser = ArgumentParser()
        add_common_eval_args(self.parser)
        add_ext_eval_args(self.parser)
        add_text_default_args(self.parser)

        self.args = self.parser.parse_args([])
        self.cfg = vars(self.args)
        self.cfg['enable_long_term'] = True

        self.deva_model = DEVA(self.cfg).to(device).eval()
        self.model_weights = torch.load(self.args.model)
        self.deva_model.load_weights(self.model_weights)
    
        self.gd_model, self.sam_model = get_grounding_dino_model(self.cfg, 'cuda')
        self.cfg['enable_long_term_count_usage'] = True
        self.cfg['max_num_objects'] = 120
        self.cfg['size'] = 480
        self.cfg['DINO_THRESHOLD'] = 0.35
        self.cfg['amp'] = True
        self.cfg['chunk_size'] = 15
        self.cfg['suppress_small_objects'] = True
        self.cfg['detection_every'] = 10
        self.cfg['max_missed_detection_count'] = 40
        self.cfg['sam_variant'] = 'original'
        self.cfg['temporal_setting'] = 'online' 
        self.cfg['pluralize'] = True
        self.cfg['prompt'] = '.'.join(['person'])
        self.cfg['model'] = ['D17']

        self.deva = DEVAInferenceCore(self.deva_model, config=self.cfg)
        self.deva.next_voting_frame = self.cfg['num_voting_frames'] - 1
        self.deva.enabled_long_id()

    def get_bbox_track(self):
        videos = [v for v in os.listdir(os.path.join(self.clear)) if v.endswith('.mp4')]
        for video in videos:
            ti = 0
            result_saver = ResultSaver(os.path.join(self.final), None, dataset='demo', object_manager=self.deva.object_manager)
            cap, _ = write_new_file(os.path.join(self.clear, video), None)
            with torch.cuda.amp.autocast(enabled=True):
                if self.stop_frame is None:
                    self.stop_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for i in tqdm.tqdm(range(self.start_frame, self.stop_frame, self.step_frames)): 
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    _, frame = cap.read()
                    process_frame_with_text(self.deva, self.gd_model, self.sam_model, f'{i}.png',result_saver, ti, image_np=frame)
                    ti += 1
                    torch.cuda.empty_cache()
                    gc.collect()
            flush_buffer(self.deva, result_saver)
            file_json = video.split('.')[0]+'.json'
            with open(os.path.join(self.final, file_json), 'w') as f:
                json.dump(result_saver.video_json, f, indent=4)
            cap.release()
            self.deva.clear_buffer() 
                  
    def get_iou(self, box_yolo, box_deva):
    	xA = max(box_yolo[0], box_deva[0])
    	yA = max(box_yolo[1], box_deva[1])
    	xB = min(box_yolo[2], box_deva[2])
    	yB = min(box_yolo[3], box_deva[3])
    	inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    	box_yolo_area = (box_yolo[2] - box_yolo[0] + 1) * (box_yolo[3] - box_yolo[1] + 1)
    	box_deva_area = (box_deva[2] - box_deva[0] + 1) * (box_deva[3] - box_deva[1] + 1)
    	iou = inter_area / float(box_yolo_area + box_deva_area - inter_area)
    	return iou

    def get_id_player(self, file_detection, file_deva, file_rcg, iou=0.6):
        with open(file_deva) as f:
            templates = json.load(f)
        resul_deva = pd.json_normalize(templates['annotations'], 'segmentations', ['file_name'])
        resul_deva['file_name'] = resul_deva['file_name'].apply(lambda x: int(x.split('.')[0])).astype('int32')

        with open(file_detection) as f:
            templates_ = json.load(f)
        resul_rec = pd.json_normalize(templates_).T.reset_index().rename(columns = {'index': 'frame', 0: 'xyxy'})
        resul_rec['file_name'] = resul_rec['frame'].apply(lambda x: int(x.split('_')[0]))

        iou_list = []
        for frame in resul_deva.file_name.unique():
            for id_player in resul_deva[resul_deva['file_name']==frame]['id'].values:
                for box in resul_rec[resul_rec['file_name']==frame]['xyxy'].values:
                    bbox = resul_deva[(resul_deva['file_name']==frame)&(resul_deva['id']==id_player)]['xyxy'].values[0]
                    iou_list.append([frame, box, bbox[0], self.get_iou(box, bbox[0]), id_player])
        true_bbox = pd.DataFrame(iou_list, columns = ['frame', 'bbox_yolo', 'bbox_deva', 'iou', 'id'])
        true_bbox = true_bbox[true_bbox.iou>=iou][['frame', 'bbox_yolo', 'id']]
        with open(file_rcg) as f:
             res = json.load(f)
        df = pd.json_normalize(res)
        df = df[df.number==self.player_number]

        ids = []
        for frame_ in df.frame.unique():
            iou_res, id_ = [],[]
            r = true_bbox[true_bbox.frame==frame_]
            for i, bbox in enumerate(r.bbox_yolo.values):
                for box_coord in df[df.frame==frame_]['box_coord'].values:
                    iou_res.append(self.get_iou(bbox, box_coord))
                    id_.append(r.iloc[i]['id'])
            try:
                index = np.argmax(np.array(iou_res))
                ids.append(id_[index])
            except ValueError:
                continue
        if len(ids)>0:
            c = collections.Counter(map(int, ids))
            item, count = max(c.items(), key=lambda x: x[::-1])
            return true_bbox[true_bbox['id']==item].frame.values, true_bbox[true_bbox['id']==item].bbox_yolo.values
        else:
            return None, None
        
    def save_video_result(self, fr_=10):
        result = {}
        videos = [v for v in os.listdir(os.path.join(self.clear)) if v.endswith('.mp4')]
        for i, video in enumerate(videos):
            file_json = video.split('.')[0]+'.json'
            file_json_det = video.split('.')[0]+'_all_boxes.json'
            frames, bboxs = self.get_id_player(os.path.join(self.recognized, file_json_det), \
                                               os.path.join(self.final, file_json), os.path.join(self.recognized, file_json))
            if frames is None:
                message = f'Player {self.player_number} not found'
                result[video] = {self.player_number: [None]}
                final_path = None
            else:
                result[video] = {self.player_number: frames.tolist()}
                final_path = os.path.join(self.final,  f'{date.today()}_{i}_{self.player_number}.mp4')
                cap, output = write_new_file(os.path.join(self.clear, video), os.path.join(self.final,  f'{date.today()}_{i}_{self.player_number}.mp4'), fr_)
                for fr in tqdm.tqdm(range(len(frames))):
                   cap.set(cv2.CAP_PROP_POS_FRAMES, frames[fr])
                   _, frame = cap.read()
                   detections = sv.Detections(np.array([bboxs[fr]]), class_id=np.array([self.player_number,]))
                   annotator = sv.BoxAnnotator()
                   labels = [f'player {detections.class_id[0]}']
                   blend = annotator.annotate(scene=frame, detections=detections, labels=labels)
                   output.write(frame)
                   output.write(frame)
                   output.write(frame)
                output.release()
                cap.release()
                message = f'Tracking for {self.player_number} is ready'
            print(message)
            return result, final_path, message