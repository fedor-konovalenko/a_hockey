import os
import yadisk
import numpy as np
import time
# from transformers import BlipProcessor, BlipForConditionalGeneration

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

HOCKEY_LIST = ['hockey', 'ice', 'stick', 'puck', 'goal', 'goalie', 'net', 'skate', 'rink', 'team', 'player', 'referee',
               'penalty', 'power play', 'faceoff', 'slapshot', 'wrist shot', 'body check', 'hat trick', 'overtime',
               'shootout', ]


def write_new_file(file, output_file, fps=25):
    print(f'writing new video file...')
    time.sleep(1)


class Helper:

    def __init__(self, input_dir: str, convert_dir: str):
        self.raw = os.path.join(os.path.dirname(__file__), input_dir)
        self.convert = os.path.join(os.path.dirname(__file__), convert_dir)

    def convert_file(self):
        print(f'I am converting video.....')
        time.sleep(1)

    def download_file(self, link, token, path=None, i=99):
        print(f'I am downloading video.....')
        time.sleep(1)
        y = yadisk.YaDisk(token=token)
        url = y.get_public_download_link(link, path=path)
        # y.download_by_link(url, os.path.join(self.raw, f'{date.today()}_{i}.mp4'))
        return url


class ClearGame:
    def __init__(self, convert_dir: str, clear_dir: str, descr_dir=None):
        self.clear = os.path.join(os.path.dirname(__file__), clear_dir)
        self.convert = os.path.join(os.path.dirname(__file__), convert_dir)
        self.descr_dir = descr_dir
        # self.processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        # self.model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)

    def clear_game(self):
        print(f'I am clearing game...')
        time.sleep(1)
        return list(np.random.randint(100, 500, 4))
