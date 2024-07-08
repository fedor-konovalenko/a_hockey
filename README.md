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
- recognixing players' numbers
- collecting data and preparing JSON response to the backend

*Fig. 1 - The service scheme*
___

## The Repo Structure

This repo contains branches:
- main - here are necessary scripts and instructions for running the service
- gradio_app - scripts for demo application based on Gradio
- notebooks - experimental noteboks and researches, include:
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

Two post-requests are available:

**Processing Request**

Post request for download, clean, process video and prepare .json file with tracking results. The tracking result in .json format will be saved in temporary directory /app/src/recognition and will be returned as JsonResponse
  ```bash
  cd test
  python3 test.py
  ```
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
  git clone git@github.com:fedor-konovalenko/a_hockey.git
  cd app
  docker build --tag hockey .
  docker run --rm -p 8010:8000 --name video hockey
  ```
  then the FastApi application will be available at http://localhost:8010/
  ```bash
  docker exec -it [CONTAINER_ID] sh
  cd test
  python3 test.py
  ```
___
## Classes and Methods Description
  
  
