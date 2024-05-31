import os
import json
import numpy as np
import time


class TrackingPlayer:
    def __init__(self, clear_dir: str, json_data: str, final_dir: str, player_number=0, start_frame=None,
                 stop_frame=None, step_frames=1):

        self.clear = os.path.join(os.path.dirname(__file__), clear_dir)
        self.recognized = os.path.join(os.path.dirname(__file__), json_data)
        self.final = os.path.join(os.path.dirname(__file__), final_dir)
        self.step_frames = step_frames
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.player_number = player_number
        self.boxes = json_data

    def get_bbox_track(self, link):
        print(f'I am tracking players in game...')
        time.sleep(1)
        res = {'link2video': link,
               'annotations': []}
        len_res = 5
        for i in range(len_res):
            res['annotations'].append({
                'filename': '100.jpg',
                'segmentations': [
                    {'id': 7007,
                     'area': 42,
                     'xyxy': list(map(int, np.random.randint(100, 500, 4)))},
                    {'id': 7007,
                     'area': 42,
                     'xyxy': list(map(int, np.random.randint(100, 500, 4)))},
                    {'id': 7007,
                     'area': 42,
                     'xyxy': list(map(int, np.random.randint(100, 500, 4)))},
                    {'id': 7007,
                     'area': 42,
                     'xyxy': list(map(int, np.random.randint(100, 500, 4)))}
                ]
            })

            with open(os.path.join(self.final, 'tracking_result.json'), 'w') as f:
                json.dump(res, f)
        return os.path.join(self.final, 'tracking_result.json')

    def save_video_result(self, player_num, tracking_result):
        print(f'I am preparing video for player {player_num}...')
        time.sleep(1)
        result = f'your video for player {player_num}'
        with open(os.path.join(self.final, 'result_video.txt'), 'w') as fid:
            fid.write(f'{result}\n')
        return os.path.join(self.final, 'result_video.txt')
