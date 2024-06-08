import gradio as gr
import os
import shutil
import random
import pandas as pd
from tqdm import tqdm

from utils import Numbers
from clear_game import Helper, ClearGame
from tracking_new import TrackingPlayer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil


class GlobalState:
    """
    Class to store global variables
    """

    team_list = None
    selected_player = None
    start_frame = 0
    stop_frame = 10000
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


def update_player(dd):
    GlobalState.selected_player = dd
    return gr.update(visible=True)


def prepare_link(dd):
    link = os.path.join(GlobalState.result_file_path, dd)
    print(link)
    return gr.update(link="/file=" + link, visible=True)


def upload_video_file(fid):
    """
    uploads and save video to workdir
    """
    raw_path = os.path.join(GlobalState.video_file_path, os.path.basename(fid.name))
    # if not os.path.exists(os.path.join(os.path.dirname(__file__), 'download/')):
    # os.makedirs(os.path.join(os.path.dirname(__file__), 'download/'))
    shutil.move(fid.name, raw_path)
    gr.Info("Video uploaded")


def receive_video(text):
    """
    receive video from YaDisk
    """
    helper = Helper(input_dir=GlobalState.video_file_path, convert_dir=GlobalState.convert_file_path)
    helper.download_file(link=text, path=None, i=99)
    gr.Info("Video uploaded")


def upload_csv_file(fid):
    """
    uploads and save csv with players to workdir
    """
    shutil.move(fid.name, GlobalState.csv_file_path)
    new_list = sorted(list(map(int, list(pd.read_csv(GlobalState.csv_file_path)['1'].dropna().unique()))))
    GlobalState.team_list = new_list
    ch = GlobalState.team_list
    gr.Info("CSV file loaded")
    return gr.update(choices=ch, value=None)


def preprocessing(txt_1, txt_2):
    """
    converts, clears video
    detects players and recognizes numbers,
    track s  players
    """

    os.chdir(GlobalState.deva_path)
    GlobalState.start_frame = int(txt_1)
    GlobalState.stop_frame = int(txt_2) + GlobalState.start_frame

    helper = Helper(input_dir=GlobalState.video_file_path, convert_dir=GlobalState.convert_file_path)
    clear = ClearGame(convert_dir=GlobalState.convert_file_path, clear_dir=GlobalState.clear_file_path)
    players = Numbers(GlobalState.clear_file_path, GlobalState.detection_file_path, GlobalState.weights,
                      GlobalState.yolo_path, team=GlobalState.team_list)
    tr = TrackingPlayer(clear_dir=GlobalState.clear_file_path, json_data=GlobalState.detection_file_path,
                        final_dir=GlobalState.result_file_path, player_number=GlobalState.selected_player,
                        start_frame=GlobalState.start_frame, stop_frame=GlobalState.stop_frame, step_frames=5)
    helper.convert_file()
    gr.Info("Convertation Completed")
    clear.clear_game()
    gr.Info("Advertisement Removement Completed")
    detections, _ = players.predict(class_threshold=.7, detect_iou=.25, start=0, fstep=5, mark=False, corr=False)
    gr.Info("Detection Completed")
    tr.get_bbox_track()
    gr.Info("Preprocessing complete!")
    return gr.update(visible=True)


def track_player(btn):
    tr = TrackingPlayer(clear_dir=GlobalState.clear_file_path, json_data=GlobalState.detection_file_path,
                        final_dir=GlobalState.result_file_path, player_number=GlobalState.selected_player,
                        start_frame=GlobalState.start_frame, stop_frame=GlobalState.stop_frame, step_frames=5)
    _, _, message = tr.save_video_result()
    gr.Info(f"Tracking complete! {message}")
    ch = [fid for fid in os.listdir(GlobalState.result_file_path) if fid.endswith('.mp4')]
    return gr.update(choices=ch, value=None, visible=True)


def main():
    '''
    TODO: Fix progress bars
    '''
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
    with gr.Blocks() as demo:
        with gr.Tab("Load"):
            with gr.Row():
                gr.Markdown(
                    """
                    # Load video file and team list in .csv format
                    # Choose the player number
                    # Press **Prepare** (appearing after player selection)
                    # Then press **Run!** (appearing after video processing is completed)
                    # Select the ready video (appearing after tracking is completed)
                    # Have fun:)
                    """)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        video_link = gr.Textbox(label="Insert link to video",
                                                value="https://disk.yandex.ru/i/JLh__1IbAfmK-Q")
                    with gr.Row():
                        video_submit = gr.Button("Receive Video")
                    with gr.Row():
                        video_upload = gr.UploadButton(label="Or if you have several free hours - use this button",
                                                       file_types=["video"], file_count="single")
                    with gr.Row():
                        gr.Image(GlobalState.logo_path)

                with gr.Column():
                    with gr.Row():
                        csv_upload = gr.UploadButton(label="Load CSV", file_types=[".csv"], file_count="single")
                    with gr.Row():
                        players = gr.Dropdown(choices=[], label='Player number to track', interactive=True)
                    with gr.Row():
                        start_frame = gr.Textbox(label="Frame to start", value=0)
                    with gr.Row():
                        number_of_frames = gr.Textbox(label="Number of frames", value=10000, info='maximum 20000')
                    with gr.Row():
                        prepare_button = gr.Button("Prepare video", visible=False)
                    with gr.Row():
                        process_button = gr.Button("Run!", visible=False)
                with gr.Column():
                    with gr.Row():
                        download_selector = gr.Dropdown(choices=[], label='Choose the trimmed video to download',
                                                        interactive=True, visible=False)
                    with gr.Row():
                        download_button = gr.Button("Here is your video!", visible=False)

        video_upload.upload(upload_video_file, video_upload, show_progress='full')
        video_submit.click(receive_video, inputs=[video_link], outputs=None, show_progress='full')
        csv_upload.upload(upload_csv_file, csv_upload, show_progress='full', outputs=[players])
        players.select(update_player, inputs=[players], outputs=[prepare_button])
        prepare_button.click(preprocessing, inputs=[start_frame, number_of_frames], outputs=[process_button],
                             show_progress=True)
        process_button.click(track_player, inputs=[prepare_button], outputs=[download_selector], show_progress='full')
        download_selector.select(prepare_link, inputs=[download_selector], outputs=[download_button])
        download_button.click(prepare_link)

    demo.launch(share=True, allowed_paths=[GlobalState.result_file_path])


if __name__ == "__main__":
    main()
