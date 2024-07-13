from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
from pydantic import BaseModel
import os
import shutil
import json
import logging
from typing import Optional
from utils import setup_logging
from clear_game import Helper, ClearGame
from recognition import Numbers
from tracking import TrackingPlayer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = '0'

app = FastAPI()
_logger = logging.getLogger(__name__)


class GlobalState:
    """
    Class to store global variables
    """
    team_list = None
    selected_player = None
    video_file_path = os.path.join(os.path.dirname(__file__), 'download/')
    convert_file_path = os.path.join(os.path.dirname(__file__), 'convert/')
    clear_file_path = os.path.join(os.path.dirname(__file__), 'clear/')
    track_file_path = os.path.join(os.path.dirname(__file__), 'tracking/')
    result_file_path = os.path.join(os.path.dirname(__file__), 'recognition/')
    frame_step = 5
    deva_path = os.path.join(os.path.dirname(__file__), '../Tracking-Anything-with-DEVA')


os.chdir(GlobalState.deva_path)

helper = Helper(input_dir=GlobalState.video_file_path, convert_dir=GlobalState.convert_file_path)
clear = ClearGame(convert_dir=GlobalState.convert_file_path, clear_dir=GlobalState.clear_file_path)
detector = Numbers(input_dir=GlobalState.convert_file_path, clear_dir=GlobalState.clear_file_path,
                   output_dir=GlobalState.result_file_path, emb_mode='resnet')

tracker = TrackingPlayer(convert_dir=GlobalState.convert_file_path, clear_dir=GlobalState.clear_file_path,
                         final_dir=GlobalState.track_file_path)


@app.get("/")
def main():
    """Start service, create temporary folders"""
    page = "<hml><body><h1>Hockey Game Video Processing</h1></body></html>"
    if not os.path.exists(GlobalState.video_file_path):
        os.makedirs(GlobalState.video_file_path)
    if not os.path.exists(GlobalState.convert_file_path):
        os.makedirs(GlobalState.convert_file_path)
    if not os.path.exists(GlobalState.clear_file_path):
        os.makedirs(GlobalState.clear_file_path)
    if not os.path.exists(GlobalState.result_file_path):
        os.makedirs(GlobalState.result_file_path)
    if not os.path.exists(GlobalState.track_file_path):
        os.makedirs(GlobalState.track_file_path)
    return HTMLResponse(page)


@app.get("/status")
def health():
    pass
    return JSONResponse(content={"status": "OK", "version": 1.0})


@app.get("/version")
def version():
    return {"version": 1.0}


class GameFeatures(BaseModel):
    game_id: Optional[int] = None
    game_link: Optional[str] = None
    token: Optional[str] = None
    player_ids: Optional[list] = None
    player_numbers: Optional[list] = None
    team_ids: Optional[list] = None


@app.post("/process")
def prediction(game_features: GameFeatures):
    """full processing of the new video"""
    v = game_features.model_dump()
    _logger.info(f'Downloading video...')
    raw_name = helper.download_file(link=v['game_link'], token=v['token'], path=None)
    if raw_name == 'FAIL':
        msg = f'Cannot download the video'
        _logger.error(msg)
        return JSONResponse(content={'error': msg})
    _logger.info(f'Converting video...')
    converted_name = helper.convert_file(video_name=raw_name)
    _logger.info(f'Searching frames with advertisement in video...')
    ad_frames = clear.get_advertising_frames(video_name=converted_name)
    _logger.info(f'Tracking people...')
    full_track_pth = tracker.get_bbox_track(video_name=converted_name, start_frame=0, stop_frame=None,
                                            step_frames=GlobalState.frame_step)
    _logger.info(f'Recognizing numbers...')
    result = detector.predict_after(class_threshold=.8, ann_path=full_track_pth, video_path=converted_name,
                                    tms=v['player_numbers'], box_min_size=0)
    _logger.info(f'Collecting results...')
    answer = []
    for res in result:
        res['player_id'] = v['player_ids'][res['team']][res['counter']]
        res['team_id'] = v['team_ids'][res['team']]
        del res['counter']
        answer.append(res)
    with open(os.path.join(GlobalState.result_file_path, f'{converted_name.split(".")[0]}_final.json'), "w+") as fid:
        json.dump(answer, fid)

    return JSONResponse(content={'game_link': v['game_link'],
                                 'token': v['token'],
                                 'players': answer}
                        )


@app.post("/clean")
def remove_content():
    folders_for_clean = [GlobalState.video_file_path,
                         GlobalState.convert_file_path,
                         GlobalState.clear_file_path,
                         GlobalState.track_file_path,
                         GlobalState.result_file_path]
    vol = 0
    count = 0
    for dirpath in folders_for_clean:
        for filename in os.listdir(dirpath):
            filepath = os.path.join(dirpath, filename)
            try:
                vol += sum(d.stat().st_size for d in os.scandir(filepath) if d.is_file()) / 1e6
                shutil.rmtree(filepath)
            except OSError:
                vol += os.path.getsize(filepath) / 1e6
                os.remove(filepath)
            count += 1
    return JSONResponse(content={"Removed": "OK",
                                 "Objects": count,
                                 "Size": f'{vol:.2} Mb'})


if __name__ == "__main__":
    setup_logging(loglevel="INFO")
    uvicorn.run(app, host="0.0.0.0", port=8000)
