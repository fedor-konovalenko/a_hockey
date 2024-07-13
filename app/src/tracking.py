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


class TrackingPlayer:
    def __init__(self, convert_dir: str, clear_dir: str, final_dir: str):

        self.convert = os.path.join(os.path.dirname(__file__), convert_dir)
        self.clear = os.path.join(os.path.dirname(__file__), clear_dir)
        self.final = os.path.join(os.path.dirname(__file__), final_dir)

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

    def get_bbox_track(self, video_name: str, start_frame: int = 0, stop_frame: int = None, step_frames: int = 5):
        video = os.path.join(self.convert, video_name)
        ti = 0
        result_saver = ResultSaver(os.path.join(self.final), None, dataset='demo',
                                   object_manager=self.deva.object_manager)
        cap = cv2.VideoCapture(os.path.join(self.convert, video))

        file_name = video.split('.')[0].split('/')[-1]

        with torch.cuda.amp.autocast(enabled=True):
            if stop_frame is None:
                stop_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            with open(os.path.join(self.clear, f'{file_name}.json')) as f:
                list_frames = json.load(f)
            stop_frame_list = list_frames[file_name]
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for i in tqdm.tqdm(range(start_frame, stop_frame, step_frames)):
                _, frame = cap.read()
                if i in stop_frame_list:
                    if i + step_frames not in stop_frame_list:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i + step_frames)
                    continue
                process_frame_with_text(self.deva, self.gd_model, self.sam_model, f'{i}.png', result_saver, ti,
                                        image_np=frame)
                ti += 1
                torch.cuda.empty_cache()
                gc.collect()
        flush_buffer(self.deva, result_saver)

        with open(os.path.join(self.final, f'{file_name}.json'), 'w') as f:
            json.dump(result_saver.video_json, f, indent=4)
        cap.release()
        self.deva.clear_buffer()

        with open(os.path.join(self.final, f'{file_name}.json'), 'r') as f:
            templates = json.load(f)
        resul_deva = pd.json_normalize(templates['annotations'], 'segmentations', ['file_name'])
        resul_deva['frame'] = resul_deva['file_name'].apply(lambda x: int(x.split('.')[0]))
        resul_deva = resul_deva.merge(resul_deva.groupby(by='id')['file_name'] \
                                      .count().to_frame().reset_index().rename(columns={'file_name': 'count_id'}),
                                      how='left', on='id')
        resul_deva = resul_deva.loc[resul_deva['count_id'] > 1000][['id', 'frame', 'xyxy']]
        resul_deva.to_json(os.path.join(self.final, f'{file_name}_result.json'))
        return os.path.join(self.final, f'{file_name}_result.json')
