import os
import torch
import sys
import gc
import tqdm
import json
import yadisk
import cv2
from datetime import date
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

HOCKEY_LIST = ['hockey', 'ice', 'stick', 'puck', 'goal', 'goalie', 'net', 'skate', 'rink', 'team', 'player', 'referee',
'penalty', 'power play', 'faceoff', 'slapshot', 'wrist shot', 'body check', 'hat trick', 'overtime', 'shootout', ]


def write_new_file(file, output_file, fps = 25):
    cap = cv2.VideoCapture(file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    output = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    return cap, output  

class Helper:

    def __init__(self, input_dir: str, convert_dir: str):
        self.raw = os.path.join(os.path.dirname(__file__), input_dir)
        self.convert = os.path.join(os.path.dirname(__file__), convert_dir)
    
    def convert_file(self):
        videos = [v for v in os.listdir(self.raw) if v.endswith(('.mp4', '.avi'))]
        for i, video in enumerate(videos):
            input = os.path.join(self.raw, video)
            output = os.path.join(self.convert, f'{date.today()}_{i}.mp4')
            os.system(f'ffmpeg -i {input} -crf 20 -vf scale=720:-2 -y {output}')

    def download_file(self, link, path=None, i=99):
        y = yadisk.YaDisk(token='y0_AgAAAAA8cbR4AAtjkQAAAAD9CA6QAAA7HIEFePpA7qKzGdujIAwVc_JH9w')
        url = y.get_public_download_link(link, path=path)
        y.download_by_link(url, os.path.join(self.raw, f'{date.today()}_{i}.mp4'))


class ClearGame:
    def __init__(self, convert_dir: str, clear_dir: str, descr_dir=None):
        self.clear = os.path.join(os.path.dirname(__file__), clear_dir)
        self.convert = os.path.join(os.path.dirname(__file__), convert_dir)
        self.descr_dir = descr_dir
        self.processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        self.model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)
    
    def cap_video(self, video):
        cap = cv2.VideoCapture(video)
        filename = os.path.basename(video).split('.')[0]
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cap, filename, fps, count_frames
        
    def get_label(self, text):
        if ('hockey' in text) or (len([word for word in text.split() if word in HOCKEY_LIST])>0):
            return 1
        return 0

    def get_frame(self, cap, i):
        cap.set(cv2.CAP_PROP_POS_FRAMES,i-1)
        _, frame = cap.read()
        return frame

    def get_text(self, frame):
        inputs = self.processor(frame, text='', return_tensors='pt')
        inputs = inputs.to(device)
        out = self.model.generate(**inputs)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        torch.cuda.empty_cache()
        gc.collect()
        return text

    def get_result(self, cap, item, filename):
        frame = self.get_frame(cap, item)
        text = self.get_text(frame)
        label = self.get_label(text)
        if self.descr_dir is not None:
            cv2.imwrite(dir+f'{filename}_{item-1}.jpg', frame)
        return [f'{filename}_{item-1}.jpg', item-1, text, label], label

    def get_info_about_game(self, cap, fps, count_frames, filename):
        info_list = []
        last_value = 0
        time = 50
        for i in range(fps*time, count_frames, fps*time):
            result, label = self.get_result(cap, i, filename)
            if label!=last_value:
                y=i-250
                while y>i-fps*time:
                    result, lbl = self.get_result(cap, y, filename)
                    y-=250
                    info_list.append(result)
            else:
                info_list.append(result)
            last_value = label
        info_list  = sorted(info_list, key = lambda x:x[1])
        return info_list

    def get_index_for_game(self, info_list):
        last_value=info_list[0][-1]
        x = info_list[0][1]
        coord = []
        for i in range(len(info_list)):
            if info_list[i][-1]!=last_value or i==len(info_list)-1:
                y=info_list[i][1]
                coord.append([last_value, x,y])
                x,last_value = y,info_list[i][-1]
        game_coord = list(filter(lambda x: x[0]==1, coord))
        return game_coord

    def clear_game(self):
        videos = [v for v in os.listdir(self.convert) if v.endswith('.mp4')]
        for num, video in tqdm.tqdm(enumerate(videos)):
            file = os.path.join(self.convert, video)
            cap_, filename, fps, count_frames = self.cap_video(file)
            game_info = self.get_info_about_game(cap_, fps, count_frames, filename)
            game_coord = self.get_index_for_game(game_info)
            list_frame = []
            for i in range(len(game_coord)):
                list_frame.extend([y for y in range(game_coord[i][1], game_coord[i][-1]+1)])
            output_file = os.path.join(self.clear, f'{date.today()}_{num}.mp4')
            cap, output = write_new_file(file, output_file)
            for i, fr in enumerate(list_frame):
                if fr-1!=list_frame[i-1] or i==0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                _, frame = cap.read()
                output.write(frame)
            output.release()
            cap.release()
            del list_frame
            gc.collect()