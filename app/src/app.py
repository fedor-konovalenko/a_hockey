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

app = FastAPI()
_logger = logging.getLogger(__name__)


class GlobalState:
    """
    Class to store global variables
    """

    team_list = None
    selected_player = None
    video_file_path = os.path.join(os.path.dirname(__file__), 'download/')
    csv_file_path = os.path.join(os.path.dirname(__file__), 'team.csv')
    convert_file_path = os.path.join(os.path.dirname(__file__), 'convert/')
    clear_file_path = os.path.join(os.path.dirname(__file__), 'clear/videos/')
    result_file_path = os.path.join(os.path.dirname(__file__), 'tracking/')
    detection_file_path = os.path.join(os.path.dirname(__file__), 'recognition/')
    weights = os.path.join(os.path.dirname(__file__), 'weights/resnet.pth')
    yolo_path = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')
    deva_path = os.path.join(os.path.dirname(__file__), '../Tracking-Anything-with-DEVA')


helper = Helper(input_dir=GlobalState.video_file_path, convert_dir=GlobalState.convert_file_path)
clear = ClearGame(convert_dir=GlobalState.convert_file_path, clear_dir=GlobalState.clear_file_path)
detector = Numbers(input_dir=GlobalState.clear_file_path, output_dir=GlobalState.detection_file_path,
                   weights=GlobalState.weights, emb_weights=GlobalState.weights, yolo_model=GlobalState.yolo_path)
tracker = TrackingPlayer(clear_dir=GlobalState.clear_file_path, final_dir=GlobalState.result_file_path)


@app.get("/")
def main():
    page = "<hml><body><h1>Hockey Game Video Processing</h1></body></html>"
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'convert/'), ignore_errors=True)
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'download/'), ignore_errors=True)
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'clear/'), ignore_errors=True)
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'recognition/'), ignore_errors=True)
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'tracking/'), ignore_errors=True)
    os.mkdir(os.path.join(os.path.dirname(__file__), 'convert/'))
    os.mkdir(os.path.join(os.path.dirname(__file__), 'clear/'))
    os.mkdir(os.path.join(os.path.dirname(__file__), 'clear/videos/'))
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
    """TODO add check of video (too less detections for example)"""
    v = game_features.model_dump()
    helper.download_file(link=v['game_link'], token=v['token'], path=None, i=99)
    _logger.info(f'Downloading video...')
    helper.convert_file()
    _logger.info(f'Converting video...')
    ad_frames = clear.clear_game()
    _logger.info(f'Clearing video...')
    all_boxes_pth = detector.detect(GlobalState.clear_file_path, ad_frames, iou=.25)
    _logger.info(f'Detecting people...')
    full_track_pth = tracker.get_bbox_track(all_boxes_pth, v['player_numbers'], v['player_ids'],
                                            v['game_id'], v['team_ids'])
    _logger.info(f'Tracking in process...')
    correct_full_track = detector.predict_after(.5, full_track_pth, GlobalState.clear_file_path)
    _logger.info(f'Preparing final tracking data...')
    return JSONResponse(content=correct_full_track)


@app.post("/search")
def track_player(player_features: PlayerFeatures):
    v = player_features.model_dump()
    helper.download_file(link=v['game_link'], token=v['token'], path=None, i=99)
    _logger.info(f'Downloading video...')
    video_pth = tracker.save_video_result(GlobalState.video_file_path, v['player_number'], v['frames'], v['boxes'])
    _logger.info(f'Preparing video for selected player...')
    return FileResponse(video_pth)


if __name__ == "__main__":
    setup_logging(loglevel="INFO")
    uvicorn.run(app, host="0.0.0.0", port=8000)
