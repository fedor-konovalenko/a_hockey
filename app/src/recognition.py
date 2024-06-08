import os
import warnings
import json
import numpy as np
import time
import ultralytics
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import json
import cv2
from torch.utils.data import Dataset, DataLoader
import math
import torchvision.transforms as T
from torchvision.models import resnet50, resnet18
import gc
import shutil

from transformers import AutoImageProcessor, AutoModel

warnings.filterwarnings("ignore")


class Players(torch.utils.data.Dataset):
    """class for loading cropped frames to numbers classifier"""

    def __init__(self, root, boxes):
        self.boxes = boxes
        self.root = root
        self.transforms = T.Compose([
            T.ToTensor(),
            T.CenterCrop((128, 128))
        ])
        self.imgs = list(self.boxes.keys())

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path)
        frame = int(self.imgs[idx].split('_')[0])
        box = self.boxes[os.path.basename(img_path)]

        return self.transforms(img), torch.tensor(frame), torch.tensor(box), img_path

    def __len__(self):
        return len(self.imgs)


class Numbers:
    """detects people on frames, recognize players numbers and prepare embeddings for tracking or for same numbers
    diverse"""

    def __init__(self, input_dir: str, clear_dir: str, output_dir: str, emb_mode: str):
        self.root = os.path.join(os.path.dirname(__file__), clear_dir)
        self.video_dir = os.path.join(os.path.dirname(__file__), input_dir)
        self.output_dir = os.path.join(os.path.dirname(__file__), output_dir)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cls_weights = os.path.join(os.path.dirname(__file__), 'weights/resnet_new.pth')
        self.emb_weights = os.path.join(os.path.dirname(__file__), 'weights/resnet_18.pth')
        self.yolo = YOLO(os.path.join(os.path.dirname(__file__), 'weights/yolov8n.pt'))
        self.num_classes = 100
        self.emb_mode = emb_mode
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.pic_transform = T.Compose([T.ToTensor(),
                                        T.CenterCrop((128, 128))])

        shutil.rmtree(os.path.join(self.root, 'tmp'), ignore_errors=True)
        os.mkdir(os.path.join(self.root, 'tmp'))

    def __create_classifier__(self, outputs, weights):
        """returns resnet50 model with pretrained weights"""
        resnet = resnet50(weights=None)
        in_features = 2048
        resnet.fc = nn.Linear(in_features, outputs)
        resnet.load_state_dict(torch.load(weights, map_location=self.device))
        return resnet.to(self.device)

    def __create_embedder__(self, outputs, backbone):
        """returns embedder maker: resnet18, finetuned on hockey players dataset or
        pretrained dino"""
        if backbone == 'resnet':
            model = resnet18(weights=None)
            in_features = 512
            model.fc = nn.Linear(in_features, outputs)
            model.load_state_dict(torch.load(self.emb_weights, map_location=self.device))
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
        else:
            model = AutoModel.from_pretrained('facebook/dinov2-small')
        return model.to(self.device)

    def detect(self, in_folder: str, video_name: str, adv_name: str, fstep: int, iou: float) -> dict:
        """load video and detect people on chosen frames, prepare 2 dictionaries: boxes more than 64x64 px for
        classifier and all boxes with their embeddings at frame - for tracker"""
        emb_maker = self.__create_embedder__(self.num_classes, backbone=self.emb_mode)
        emb_maker.eval()
        cap = cv2.VideoCapture(os.path.join(in_folder, video_name))
        stop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cls_dict = {}
        track_dict = {}
        with open(os.path.join(self.root, adv_name)) as fid:
            adv_frames = json.load(fid)[adv_name.split('.')[0]]
        u_frames = [f for f in list(range(0, stop, fstep)) if f not in adv_frames]
        for fr in tqdm(u_frames, desc=f'detecting people and computing embeddings in {video_name} video'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            _, frame = cap.read()
            result = self.yolo(frame, verbose=False, iou=iou)
            boxes = result[0].boxes
            i = 0
            for bboxes in boxes:
                if bboxes.cls == 0:
                    for b in bboxes.xyxy:
                        b = b.cpu().tolist()
                        b = list(map(int, b))
                        if self.emb_mode == 'resnet':
                            pic = self.pic_transform(frame[b[1]:b[3], b[0]:b[2]]).to(self.device)
                            with torch.no_grad():
                                emb = emb_maker(pic.unsqueeze(0)).squeeze().half().cpu().tolist()
                        elif self.emb_mode == 'dino':
                            pic = self.processor(images=frame[b[1]:b[3], b[0]:b[2]], return_tensors="pt")
                            with torch.no_grad():
                                emb = emb_maker(**pic.to(self.device)).pooler_output.squeeze(0).half().cpu().tolist()
                        track_dict[f'{fr}_{i}'] = {'bbox': b,
                                                   'embedding': emb}
                        if b[3] - b[1] > 64 and b[2] - b[0] > 64:
                            person = frame[b[1]:b[3], b[0]:b[2]]
                            name = os.path.join(self.root, 'tmp', f'{fr}_{i}_{video_name.split(".")[0]}.png')
                            i += 1
                            cv2.imwrite(name, person)
                            cls_dict[os.path.basename(name)] = b
                        torch.cuda.empty_cache()
        if len(track_dict) < 1000:
            status = 'FAIL'
            result = ''
            return {'status': status, 'result': result}
        with open(os.path.join(self.output_dir, f"all_boxes_{self.emb_mode}.json"), "w+") as fid:
            json.dump(track_dict, fid)
        with open(os.path.join(self.output_dir, f"big_boxes_{self.emb_mode}.json"), "w+") as fid:
            json.dump(cls_dict, fid)
        cap.release()
        del cap
        result = os.path.join(self.output_dir, f"all_boxes_{self.emb_mode}.json")
        result_1 = os.path.join(self.output_dir, f"big_boxes_{self.emb_mode}.json")
        status = 'OK'
        return {'status': status, 'result_for_tracking': result, 'result_for_classifier': result_1}

    def predict_before(self, description: str, class_threshold: float) -> str:
        """load classifier model, predict numbers on cropped frames and save results to json"""
        with open(description) as fid:
            box_descriptions = json.load(fid)
        dataset = Players(os.path.join(self.root, 'tmp'), box_descriptions)
        dataloader = DataLoader(dataset, batch_size=8, num_workers=1, pin_memory=True, shuffle=False)
        classifier = self.__create_classifier__(self.num_classes, self.cls_weights)
        classifier.eval()
        preds = []
        confs = []
        frames = []
        boxes = []
        with torch.no_grad():
            for data, f, b, path in tqdm(dataloader, desc=f'recognizing numbers in cropped frames'):
                data = data.to(self.device)
                output = classifier(data)
                pred = torch.argmax(output, 1).cpu().tolist()
                pred = list(map(int, pred))
                cfs = nn.functional.softmax(output).cpu().tolist()
                cfs = list(map(max, cfs))
                preds.extend(pred)
                confs.extend(cfs)
                f = f.cpu().tolist()
                f = list(map(int, f))
                b = b.cpu().tolist()
                boxes.extend(b)
                frames.extend(f)
            result = []
            for f, p, c, b in zip(frames, preds, confs, boxes):
                if c < class_threshold:
                    p = 'unknown'
                d = {'frame': f,
                     'number': p,
                     'confidence': round(c, 2),
                     'box_coord': b}
                result.append(d)

        with open(os.path.join(self.output_dir, f"rec_numbers.json"), "w+") as fid:
            json.dump(result, fid)
        del (data, dataset, dataloader)
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return os.path.join(self.output_dir, f"rec_numbers.json")

