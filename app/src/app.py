from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import uvicorn
from pydantic import BaseModel
import os
import shutil
import logging
from utils import setup_logging
from clear_game import Helper, ClearGame
from recognition import Numbers
from tracking import TrackingPlayer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    result_file_path = os.path.join(os.path.dirname(__file__), 'tracking/')
    detection_file_path = os.path.join(os.path.dirname(__file__), 'recognition/')
    frame_step = 5
    deva_path = os.path.join(os.path.dirname(__file__), '../Tracking-Anything-with-DEVA')


helper = Helper(input_dir=GlobalState.video_file_path, convert_dir=GlobalState.convert_file_path)
clear = ClearGame(convert_dir=GlobalState.convert_file_path, clear_dir=GlobalState.clear_file_path)
detector = Numbers(input_dir=GlobalState.convert_file_path, clear_dir=GlobalState.clear_file_path,
                   output_dir=GlobalState.detection_file_path, emb_mode='resnet')


# tracker = TrackingPlayer(clear_dir=GlobalState.clear_file_path, final_dir=GlobalState.result_file_path)


@app.get("/")
def main():
    """TODO: may be, don't overwrite folders...
    add requirements"""
    page = "<hml><body><h1>Hockey Game Video Processing</h1></body></html>"
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'convert/'), ignore_errors=True)
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'download/'), ignore_errors=True)
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'clear/'), ignore_errors=True)
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'recognition/'), ignore_errors=True)
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'tracking/'), ignore_errors=True)
    os.mkdir(os.path.join(os.path.dirname(__file__), 'convert/'))
    os.mkdir(os.path.join(os.path.dirname(__file__), 'clear/'))
    os.mkdir(os.path.join(os.path.dirname(__file__), 'download/'))
    os.mkdir(os.path.join(os.path.dirname(__file__), 'recognition/'))
    os.mkdir(os.path.join(os.path.dirname(__file__), 'tracking/'))
    return HTMLResponse(page)


@app.get("/status")
def health():
    pass
    return JSONResponse(content={"status": "OK", "version": 1.0})


@app.get("/version")
def version():
    return {"version": 1.0}


class GameFeatures(BaseModel):
    game_id: int | None = None
    game_link: str | None = None
    token: str | None = None
    player_ids: list | None = None
    player_numbers: list | None = None
    team_ids: list | None = None


class PlayerFeatures(BaseModel):
    game_link: str | None = None
    token: str | None = None
    player_number: int | None = None
    frames: list | None = None
    boxes: list | None = None


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
    _logger.info(f'Detecting people...')
    all_boxes = detector.detect(in_folder=GlobalState.convert_file_path, video_name=converted_name,
                                adv_name=ad_frames, fstep=GlobalState.frame_step, iou=.5)
    if all_boxes['status'] == 'FAIL':
        msg = f'too less detections'
        _logger.error(msg)
        return JSONResponse(content={'error': msg})

    # full_track_pth = tracker.get_bbox_track(all_boxes_pth, v['player_numbers'], v['player_ids'],
    # v['game_id'], v['team_ids'])
    # _logger.info(f'Tracking in process...')
    # correct_full_track = detector.predict_after(.5, full_track_pth, GlobalState.clear_file_path)
    # _logger.info(f'Preparing final tracking data...')
    ooo = detector.predict_before(all_boxes['result_for_classifier'], .7)
    return JSONResponse(content={'res': ooo})



@app.post("/search")
def track_player(player_features: PlayerFeatures):
    v = player_features.model_dump()
    raw_name = helper.download_file(link=v['game_link'], token=v['token'], path=None)
    _logger.info(f'Downloading video...')
    video_pth = 'ppp'  # tracker.save_video_result(GlobalState.video_file_path, v['player_number'], v['frames'], v['boxes'])
    _logger.info(f'Preparing video for selected player...')
    return FileResponse(video_pth)


if __name__ == "__main__":
    setup_logging(loglevel="INFO")
    uvicorn.run(app, host="0.0.0.0", port=8000)

