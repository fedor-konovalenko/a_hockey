# a_hockey
______
## The Service Description

It is the video-processing service for tracking the hockey players on videos of hockey games.
It was developed for [Adaptive Hockey (hockey for people with disabilities) Federation](https://paraicehockey.ru/).


The main parts of the service:
- receiving the query from the backend
- downloading the video
- converting the video with ffmpeg
- selecting frames without advertisement in order not to use them
- tracking players
- recognizing players' numbers
- collecting data and preparing JSON response to the backend

*Fig. 1 - The service scheme*
___

## The Repo Structure

This repo contains branches:
- main - here are necessary scripts and instructions for running the service
- gradio_app - scripts for demo application based on Gradio
- notebooks - experimental notebooks and researches, include:
  - experiments with tracking
  - experiments with video processing
  - training the number recognizing model
___

## Running the service

### local

**Clone the repo**

```bash
git clone git@github.com:fedor-konovalenko/a_hockey.git -b main
cd app
pip install -r requirements.txt
```
**Save pretrained weights and useful script for Deva Tracker**

download with these links:
- [resnet50 weights](https://drive.google.com/file/d/1R-55YD6UPiNi3HXkjYtCrY1HQYOT50Lj/view?usp=sharing)
- [resnet18 weights](https://drive.google.com/file/d/1x6uqQ_jllDkZkAE0JKJ0PxbMIZE3bRhg/view?usp=sharing)

to app/src/weights folder

download with this link:
- [result_utils.py script](https://drive.google.com/file/d/1RIdNrVznsJ5sXVMzomYzLXtMHrtaBZHm/view?usp=sharing)

to app/ folder

**Clone the Deva Repo and replace utils.py script**

```bash
cd app
git clone git@github.com:hkchengrex/Tracking-Anything-with-DEVA.git
pip install -e .
mv result_utils.py /Tracking-Anything-with-DEVA/deva/inference/result_utils.py
```

**Run the FastApi app**

```bash
cd src
python3 app.py
```

then the FastApi application will be available at http://localhost:8000/

Test scripts for simulate the requests are available in app/src/test folder

Two post-requests are available:

**Processing Request**

Post request for download, clean, process video and prepare .json file with tracking results. The tracking result in .json format will be saved in temporary directory /app/src/recognition and will be returned as JsonResponse

The request structure:

```python
{"game_id": int,
 "game_link":  str,
 "token": str,
 "player_ids": [[int, int, ...], [int, int, ...]],
 "player_numbers": [[int, int, ...], [int, int, ...]],
 "team_ids": [int, int]}
```

The test script example

```python
import requests
import json

def main():
    with open("test_query_process.json", "r") as fid:
        data = json.load(fid)
    r = requests.post("http://localhost:8000/process", json=data)
    if r.status_code != 200:
        print(r.status_code)
    print(r.json())

if __name__ == "__main__":
    main()
```

And after processing the video the response is returned:

```python
{"game_link": str,
 "token": str,
 players: [{"player_id": int,
            "team_id": int,
            "number": int,
            "frames": [int, int, ...]}
           ]}
 "player_numbers": [[int, int, ...], [int, int, ...]],
 "team_ids": [int, int]}
```

**Clean Request**

Strongly recommended after each service usage. Removes all content in temporary service directories.

The test script example

```python
import requests
import json

def main():
    r = requests.post("http://localhost:8000/clean")
    if r.status_code != 200:
        print(r.status_code)
    print(r.json())

if __name__ == "__main__":
    main()

```

The response example:

```python
{"Removed": str,
 "Objects": int,
 "Size": str})
```

  ____
  ### Docker
  ```bash
 some code...
  ```
___
## Classes and Public Methods Description

|**Class**.method|**Parameters**|**Returns**|**Comments**|
|--|--|--|--|
|**Helper**|input_dir: str, <br /> convert_dir: str||Class for preparing video. <br /> Required parameters - <br /> path to directory for downloading video <br />and for converting video|
|Helper.download_file|link: str, <br /> token: str|str|Downloads video from Yandex Disk, <br /> returns the raw video name|
|Helper.convert_file|video_name: str|str|Converts video with ffmeg, returns the <br /> name of the converted video|
|**ClearGame**|convert_dir: str, <br /> clear_dir: str||Class for searching frames without a hockey game. <br /> Required parameters - <br /> path to directory with converted <br /> video and to directory for save results|
|ClearGame.get_advertising_frames|video_name: str|str|With image2text model searches frames without hockey game,<br />  prepares the .json file with frames and returns it's name|
|**Tracking**|convert_dir: str, <br /> clear_dir: str, <br /> final_dir: str||Class for tracking players with DEVA. <br /> Required parameters - <br /> path to directory with converted video, <br /> with frames without game and for tracking results|
|Tracking.get_bbox_track|video_name: str|str|Tracks players, prepare json file with tracked objects <br /> and its frames, returns the file name|
|**Numbers**|input_dir: str, <br />clear_dir: str, <br />output_dir: str, emb_mode: str||Class for recognizing numbers. <br /> Required parameters- <br /> path to directory with converted video, with frames without game and for recognizing results  <br /> and the embedding model mode (ResNet or DinoV2)|
|Numbers.predict_after|class_threshold: float,<br /> ann_path: str, <br />video_path: str, <br />tms: list, <br />box_min_size: int|list|Recognizes numbers on tracked objects, <br />compare numbers with team lists, writes the results to .json file, returns list if dictionaries|


  
