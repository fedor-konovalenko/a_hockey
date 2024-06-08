import ultralytics
from ultralytics import YOLO
import os
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import json
import cv2
from torch.utils.data import Dataset, DataLoader
import math
import torchvision.transforms as T
from torchvision.models import resnet50
import warnings
import gc
import shutil
from itertools import permutations

warnings.filterwarnings("ignore")


class Players(torch.utils.data.Dataset):
    def __init__(self, root, boxes):
        self.boxes = boxes
        self.root = root
        self.transforms = T.Compose([
            T.ToTensor(),
            T.CenterCrop((128, 128))
        ])
        # self.imgs = [im for im in os.listdir(self.root) if im.endswith('.png')]
        self.imgs = list(self.boxes.keys())
        # to avoid jpynb checkpoints

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path)
        frame = int(self.imgs[idx].split('_')[0])
        box = self.boxes[os.path.basename(img_path)]

        return self.transforms(img), torch.tensor(frame), torch.tensor(box), img_path

    def __len__(self):
        return len(self.imgs)


class Numbers:
    """detect and classify on video or classify on cut bboxes players' numbers
    TODO: add list of numbers checking for prediction correction"""
    def __init__(self, input_dir: str, output_dir: str, weights: str, yolo_model: str, team=None):
        self.team = team
        self.weights = os.path.join(os.path.dirname(__file__), weights)
        # if self.team is None:
            # self.num_classes = 100
        self.num_classes = 100
        self.root = os.path.join(os.path.dirname(__file__), input_dir)
        self.output_dir = os.path.join(os.path.dirname(__file__), output_dir)
        self.yolo = YOLO(os.path.join(os.path.dirname(__file__), yolo_model))
        shutil.rmtree(os.path.join(self.root, 'tmp'), ignore_errors=True)
        os.mkdir(os.path.join(self.root, 'tmp'))

    def __create_model__(self, outputs, weights):
        """returns resnet50 model with pretrained weights"""
        resnet = resnet50(weights=None)
        in_features = 2048
        resnet.fc = nn.Linear(in_features, outputs)
        if torch.cuda.is_available():
            device = 'cuda'
            resnet.load_state_dict(torch.load(weights))
        else:
            device = 'cpu'
            resnet.load_state_dict(torch.load(weights, map_location='cpu'))
        return resnet.to(device)

    def __detect__(self, in_folder: str, start: int, fstep: int, iou: float) -> dict:
        """load video and detect people on chosen frames, prepare 2 dictionaries: boxes more than 64x64 px for 
        classifier and all boxes at frame - for tracker"""
        videos = [v for v in os.listdir(in_folder) if v.endswith('.mp4')]
        # to avoid jpynb checkpoints
        dicts = {}

        for v in videos:
            cap = cv2.VideoCapture(os.path.join(in_folder, v))
            stop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            d = {}
            all_d = {}
            for fr in tqdm(range(start, stop, fstep), desc=f'detecting people in {v} video'):
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
                            all_d[f'{fr}_{i}'] = b
                            if b[3] - b[1] > 64 and b[2] - b[0] > 64:
                                person = frame[b[1]:b[3], b[0]:b[2]]
                                name = os.path.join(in_folder, 'tmp', f'{fr}_{i}_{v.split(".")[0]}.png')
                                i += 1
                                cv2.imwrite(name, person)
                                d[os.path.basename(name)] = b
            vname = v.split('.')[0]
            dicts[vname] = d
            with open(os.path.join(self.output_dir, f"{vname}_all_boxes.json"), "w+") as fid:
                json.dump(all_d, fid)
            cap.release()
            del cap
        return dicts

    def __mark__(self, marked_images: dict):
        """sort marked images to folders to be used further for classifier training"""
        os.mkdir(os.path.join(self.root, 'mark'))
        for f in list(set(list(marked_images.values()))):
            os.mkdir(os.path.join(self.root, 'mark', str(f)))
        for p in tqdm(list(marked_images.keys()), desc='sorting marked images'):
            dest_p = os.path.join(self.root, 'mark', str(marked_images[p]), os.path.basename(p).split('.')[0] + f'_{str(marked_images[p])}.png')
            os.system(f'cp {p} {dest_p}')
        os.system(f'zip -r -q {os.path.join(self.root, "mark")}.zip {os.path.join(self.root, "mark")}')
        shutil.rmtree(os.path.join(self.root, 'mark'))

    def __correct__(self, num: int, predictions: torch.tensor, numbers: list):
        """try to correct mistake in one number with hypothesis of similar numbers"""
        num = str(num)
        numbers = list(map(str, numbers))
        sim_numbs = [['1', '4', '7'], ['3', '5', '8'], ['8', '9', '0']]
        variants = []
        if num in numbers:
            return int(num)
        else:
            num_str = list(str(num))
            for i in range(len(sim_numbs)):
                for sym in num_str:
                    if sym in sim_numbs[i]:
                        perms = list(permutations(sim_numbs[i], 2))
                        for j in range(len(perms)):
                            if sym in perms[j]:
                                for sym_p in perms[j]:
                                    for k in range(len(num_str)):
                                        num_str[k] = sym_p
                                        v = ''.join(num_str)
                                        if v in numbers:
                                            variants.append(v)
        if len(variants) != 0:
            variants = list(map(int, list(set(variants))))
            pred_max = -1e6
            idx = 0
            for i in range(len(variants)):
                if predictions[variants[i]] >= pred_max:
                    pred_max = predictions[variants[i]]
                    idx = variants[i]
                i += 1
            return idx
        else:
            return 'unknown'        
            
    def predict(self, class_threshold: float, detect_iou: float, start: int, fstep: int, mark=False, corr=False) -> tuple:
        """load model, predict numbers and save results to json"""
        box_descriptions = self.__detect__(self.root, start, fstep, detect_iou)
        results = {}
        # return box_descriptions.keys(), box_descriptions
        for name in box_descriptions.keys():
            dataset = Players(os.path.join(self.root, 'tmp'), box_descriptions[name])
            dataloader = DataLoader(dataset, batch_size=8, num_workers=1, pin_memory=True, shuffle=False)
            classifier = self.__create_model__(self.num_classes, self.weights)
            classifier.eval()
            preds = []
            confs = []
            frames = []
            boxes = []
            path_for_m = []
            corrected = []
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
            with torch.no_grad():
                for data, f, b, path in tqdm(dataloader, desc=f'recognizing numbers in {name} video'):
                    data = data.to(device)
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
                    path_for_m.extend(path)
                    if self.team is not None and corr:
                        cr = []
                        for p, o in zip(pred, output):
                            o = o.cpu()
                            cr.append(self.__correct__(p, o, self.team))
                        corrected.extend(cr)
                result = []
                mark_result = {}
                if len(corrected) == 0:
                    for f, p, c, b, mp in zip(frames, preds, confs, boxes, path_for_m):
                        if c < class_threshold:
                            p = 'unknown'
                        d = {'frame': f,
                             'number': p,
                             'confidence': round(c, 2),
                             'box_coord': b}
                        result.append(d)
                        mark_result[mp] = p
                else:
                    for f, p, c, b, mp, pc in zip(frames, preds, confs, boxes, path_for_m, corrected):
                        if c < class_threshold:
                            p = 'unknown'
                        d = {'frame': f,
                             'number': p,
                             'confidence': round(c, 2),
                             'box_coord': b, 
                             'corrected_number': pc}
                        result.append(d)
                        mark_result[mp] = p
                with open(os.path.join(self.output_dir, f"{name}.json"), "w+") as fid:
                    json.dump(result, fid)
                del(data, dataset, dataloader)
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
            results[name] = result
        if mark:
            self.__mark__(mark_result)
        return results, box_descriptions
