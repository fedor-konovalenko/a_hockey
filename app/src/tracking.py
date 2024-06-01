import os
import json
import numpy as np
import time


class TrackingPlayer:
    def __init__(self, clear_dir: str, final_dir: str):

        self.clear = os.path.join(os.path.dirname(__file__), clear_dir)
        self.final = os.path.join(os.path.dirname(__file__), final_dir)

    def get_bbox_track(self, json_data: str, numbers: list, ids: list, game: int, teams: list) -> str:
        """prepares frames and boxes for each player"""
        time.sleep(1)
        res = {'game_id': game,
               'tracking': []}
        len_res = 5
        for i in range(len_res):
            res['tracking'].append(
                    {'player_id': ids[0],
                     'team_id': teams[0],
                     'frames': list(map(int, np.random.randint(1000, 5000, 2))),
                     'boxes': [list(map(int, np.random.randint(100, 500, 4))),
                               list(map(int, np.random.randint(100, 500, 4)))],
                     'time': ['00:15:00-00:20:00', '00:21:00-00:22:02']}
                )

            with open(os.path.join(self.final, 'tracking_result.json'), 'w') as f:
                json.dump(res, f)
        return os.path.join(self.final, 'tracking_result.json')

    def save_video_result(self, video_path: str, player_num: int, frames: list, boxes: list) -> str:
        """prepares video with selected player"""
        time.sleep(1)
        result = f'your video for player {player_num}'
        with open(os.path.join(self.final, 'result_video.txt'), 'w') as fid:
            fid.write(f'{result}\n')
        return os.path.join(self.final, 'result_video.txt')
