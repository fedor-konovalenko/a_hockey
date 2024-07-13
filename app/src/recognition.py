import os
import warnings
import pandas as pd
import json
import numpy as np
from numpy.linalg import norm
from numpy import dot
import time
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import json
import cv2
import math
import torchvision.transforms as T
from torchvision.models import resnet50, resnet18
import gc
import shutil

from transformers import AutoImageProcessor, AutoModel

warnings.filterwarnings("ignore")


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
        self.num_classes = 100
        self.emb_mode = emb_mode
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.pic_transform = T.Compose([T.ToTensor(),
                                        T.CenterCrop((128, 128))])

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

    def predict_after(self, class_threshold: float, ann_path: str, video_path: str, tms: list, box_min_size=0) -> list:
        """this method allows usage of the classifier after tracking
        """
        sim_players = False
        sim_players_list = []
        ctms = []
        ctms.extend(tms[0])
        ctms.extend(tms[1])
        if len(set(ctms)) != len(ctms):
            sim_players = True
            for pl in list(set(ctms)):
                if pl in tms[0] and pl in tms[1]:
                    sim_players_list.append(pl)
        dicts = []
        cap = cv2.VideoCapture(os.path.join(self.video_dir, video_path))
        classifier = self.__create_classifier__(self.num_classes, self.cls_weights)
        emb_maker = self.__create_embedder__(self.num_classes, backbone=self.emb_mode)
        classifier.eval()
        emb_maker.eval()
        with open(ann_path) as fid:
            data = json.load(fid)
        df = pd.DataFrame(data)
        deva_ids = df['id'].unique()
        for did in deva_ids:
            res_dict = {}
            df_p = df.loc[df['id'] == did].reset_index(drop=True)
            predictions = []
            confs = []
            embs = []
            for frame_num in tqdm(df_p['frame'].values, desc=f'recognizing numbers after tracking for deva_id {did}'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                _, frame = cap.read()
                box = df_p.loc[df_p['frame'] == frame_num]['xyxy'].item()
                box = list(map(int, box[0]))
                if min((box[3] - box[1]), (box[2] - box[0])) >= box_min_size:
                    pic = self.pic_transform(frame[box[1]:box[3], box[0]:box[2]]).to(self.device)
                    with torch.no_grad():
                        output = classifier(pic.unsqueeze(0)).squeeze(0)
                        if sim_players:
                            embedding = torch.squeeze(emb_maker(pic.unsqueeze(0))).cpu().numpy()
                        else:
                            embedding = 0
                else:
                    continue
                pred = int(torch.argmax(output, -1).cpu().item())
                cfs = nn.functional.softmax(output).cpu()
                cfs = torch.max(cfs).item()
                if cfs < class_threshold:
                    pred = 'unknown'
                predictions.append(pred)
                confs.append(cfs)
                embs.append(embedding)
            df_p['predicted_number'] = pd.Series(predictions)
            df_p['predicted_cfs'] = pd.Series(confs)

            mvn_series = \
                df_p.query('predicted_number != "unknown" and predicted_number in @ctms').groupby(by='predicted_number',
                                                                                                  as_index=False).agg(
                    {'predicted_cfs': 'mean'}).sort_values(by='predicted_cfs',
                                                           ascending=False)['predicted_number']
            if len(mvn_series) != 0:
                mvn = mvn_series.iloc[0]
            else:
                mvn = 'unknown'

            df_p['final_number'] = mvn
            mvt = 'unknown'
            if mvn not in sim_players_list:
                for i, team in enumerate(tms):
                    if mvn in team:
                        mvt = i
            res_dict['track_id'] = did
            res_dict['number'] = mvn
            res_dict['mean_confidence'] = sum(confs) / len(confs)
            res_dict['team'] = mvt
            res_dict['embedding'] = sum(embs) / len(embs)
            res_dict['frames'] = df_p['frame'].values
            dicts.append(res_dict)
            torch.cuda.empty_cache()
            del df_p
        result = pd.DataFrame(dicts)
        for i in range(len(result)):
            if result.iloc[i]['team'] == 'unknown':
                unk_emb = result.iloc[i]['embedding']
                team_0_sims = []
                team_1_sims = []
                for pl_0 in result.query('team == 0')['number'].unique():
                    e0 = np.mean(result.query('number == @pl_0')['embedding'].values)
                    team_0_sims.append(dot(unk_emb, e0) / (norm(unk_emb) * norm(e0)))
                for pl_1 in result.query('team == 1')['number'].unique():
                    e1 = np.mean(result.query('number == @pl_1')['embedding'].values)
                    team_1_sims.append(dot(unk_emb, e1) / (norm(unk_emb) * norm(e1)))
                result.loc[i, 'team'] = np.argmax(
                    [sum(team_0_sims) / len(team_0_sims), sum(team_1_sims) / len(team_1_sims)])
        final_result = []
        for num in result['number'].unique():
            dfn = result.loc[result['number'] == num]
            fr = np.concatenate(dfn['frames'].values, axis=0)
            cnt = tms[dfn['team'].values[0]].index(num)
            final_result.append({'number': int(num),
                                 'team': int(dfn['team'].values[0]),
                                 'counter': int(cnt),
                                 'frames': fr.tolist()})
        with open(os.path.join(self.output_dir, f'{video_path.split(".")[0]}_recognized.json'), "w+") as fid_r:
            json.dump(final_result, fid_r)
        cap.release()
        del cap
        return final_result
