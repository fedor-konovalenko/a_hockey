from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import uvicorn
from pydantic import BaseModel
import os
import shutil

from utils import setup_logging
from clear_game import Helper, ClearGame
from recognition import Numbers
from tracking import TrackingPlayer

app = FastAPI()


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
    logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')


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
    player_id: list | None = None
    player_number: list | None = None
    team_id: int | None = None


class PlayerFeatures(BaseModel):
    game_id: int | None = None
    player_id: int | None = None
    frames: list | None = None
    boxes: list | None = None


@app.post("/process")
def prediction(game_features: GameFeatures):
    v = game_features.model_dump()
    helper = Helper(input_dir=GlobalState.video_file_path, convert_dir=GlobalState.convert_file_path)
    clear = ClearGame(convert_dir=GlobalState.convert_file_path, clear_dir=GlobalState.clear_file_path)
    detector = Numbers(GlobalState.clear_file_path, GlobalState.detection_file_path, GlobalState.weights,
                       GlobalState.weights, GlobalState.yolo_path)
    helper.download_file(link=v['game_link'], token=v['token'], path=None, i=99)
    helper.convert_file()
    ad_frames = clear.clear_game()
    all_boxes_pth = detector.detect(GlobalState.clear_file_path, ad_frames, iou=.25)
    tracker = TrackingPlayer(GlobalState.clear_file_path, all_boxes_pth, GlobalState.result_file_path, v['player_number'][0])
    full_track_pth = tracker.get_bbox_track(v['game_link'])
    correct_full_track = detector.predict_after(.8, full_track_pth, GlobalState.clear_file_path)
    return JSONResponse(content=correct_full_track)


@app.post("/search")
def track_player(player_features: PlayerFeatures):
    v = player_features.model_dump()
    all_boxes = ''
    tracker = TrackingPlayer(GlobalState.clear_file_path, all_boxes, GlobalState.result_file_path, 99)
    video_pth = tracker.save_video_result(v['player_id'], None)
    return FileResponse(video_pth)


if __name__ == "__main__":
    setup_logging(loglevel="DEBUG")
    uvicorn.run(app, host="0.0.0.0", port=8000)
