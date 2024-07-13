import os
import torch
import gc
import numpy as np
import json
import yadisk
import cv2
from datetime import date
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

HOCKEY_LIST = ['hockey', 'ice', 'stick', 'puck', 'goal', 'goalie', 'net', 'skate', 'rink', 'team', 'player', 'referee',
               'penalty', 'power play', 'faceoff', 'slapshot', 'wrist shot', 'body check', 'hat trick', 'overtime',
               'shootout', ]


class Helper:
    """Class with methods for downloading and converting video"""

    def __init__(self, input_dir: str, convert_dir: str):
        self.raw = os.path.join(os.path.dirname(__file__), input_dir)
        self.convert = os.path.join(os.path.dirname(__file__), convert_dir)

    def convert_file(self, video_name: str) -> str:
        """function for downloaded video converting (to save memory)"""
        video = os.path.join(self.raw, video_name)
        output = os.path.join(self.convert, f'{date.today()}_converted.mp4')
        os.system(f'ffmpeg -i {video} -crf 20 -vf scale=720:-2 -y {output}')
        return f'{date.today()}_converted.mp4'

    def download_file(self, link: str, token: str, path=None) -> str:
        """function for downloading video from yandex disk"""
        y = yadisk.YaDisk(token=token)
        # y = yadisk.YaDisk()
        try:
            url = y.get_public_download_link(link, path=path)
        except yadisk.exceptions.PathNotFoundError:
            return 'FAIL'
        y.download_by_link(url, os.path.join(self.raw, f'{date.today()}_raw.mp4'))
        return f'{date.today()}_raw.mp4'


class ClearGame:
    """Class with methods for advertisement search in videos"""

    def __init__(self, convert_dir: str, clear_dir: str):
        self.clear = os.path.join(os.path.dirname(__file__), clear_dir)
        self.convert = os.path.join(os.path.dirname(__file__), convert_dir)

        self.processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        self.model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)

    def __cap_video__(self, video: str) -> tuple:
        """loads converted video"""
        cap = cv2.VideoCapture(video)
        filename = os.path.basename(video).split('.')[0]
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cap, filename, fps, count_frames

    def __get_label__(self, text: str) -> int:
        """checks the frame description: is it hockey or not"""
        if ('hockey' in text) or (len([word for word in text.split() if word in HOCKEY_LIST]) > 0):
            return 1
        return 0

    def __get_frame__(self, cap: cv2.VideoCapture, i: int) -> np.array:
        """get single frame"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, i - 1)
        _, frame = cap.read()
        return frame

    def __get_text__(self, frame: np.array) -> str:
        """get text description of the frame"""
        inputs = self.processor(frame, text='', return_tensors='pt')
        inputs = inputs.to(device)
        out = self.model.generate(**inputs)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        torch.cuda.empty_cache()
        gc.collect()
        return text

    def __get_result__(self, cap: cv2.VideoCapture, item: int, filename: str) -> tuple:
        """prepares info about frame"""
        frame = self.__get_frame__(cap, item)
        text = self.__get_text__(frame)
        label = self.__get_label__(text)
        return [f'{filename}_{item - 1}.jpg', item - 1, text, label], label

    def __get_info_about_game__(self, cap: cv2.VideoCapture, fps: int, count_frames: int, filename: str) -> list:
        """prepares values of video description and time labels for frames"""
        info_list = []
        last_value = 0
        time = 120
        for i in range(fps * time, count_frames, fps * time):
            result, label = self.__get_result__(cap, i, filename)
            if label != last_value:
                y = i - 250
                while y > i - fps * time:
                    result, lbl = self.__get_result__(cap, y, filename)
                    y -= 250
                    info_list.append(result)
            else:
                info_list.append(result)
            last_value = label
        info_list = sorted(info_list, key=lambda x: x[1])
        return info_list

    def __get_index_for_game__(self, info_list: list, advertising=1) -> list:
        """prepares indexes for video description"""
        last_value = info_list[0][-1]
        x = info_list[0][1]
        coord = []
        for i in range(len(info_list)):
            if info_list[i][-1] != last_value or i == len(info_list) - 1:
                y = info_list[i][1]
                coord.append([last_value, x, y])
                x, last_value = y, info_list[i][-1]
        game_coord = list(filter(lambda x_: x_[0] == advertising, coord))
        return game_coord

    def get_advertising_frames(self, video_name: str) -> str:
        """returns dictionary with list of frames with advertisement"""
        file = os.path.join(self.convert, video_name)
        cap_, filename, fps, count_frames = self.__cap_video__(file)
        game_info = self.__get_info_about_game__(cap_, fps, count_frames, filename)
        advertising_frames = self.__get_index_for_game__(game_info, 0)
        list_frames = []
        if advertising_frames[0][1] < 5000:
            list_frames = [y for y in range(0, advertising_frames[0][1])]
        for values in advertising_frames:
            list_frames.extend([y for y in range(values[1], values[-1] + 1)])
        with open(os.path.join(self.clear, f'{filename}.json'), 'w') as f:
            json.dump({filename: list_frames}, f, indent=4, sort_keys=True)
        return f'{filename}.json'

