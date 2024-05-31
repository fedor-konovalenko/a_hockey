# a_hockey
____
## how to

Examples of queries are located in app/src/test/ folder
___
### local

```bash
git clone git@github.com:fedor-konovalenko/a_hockey.git
cd app
pip install -r requirements.txt
cd src
python3 app.py
```
then the FastApi application will be available at http://localhost:8000/

Two types of requests (except service ones):
- post request for download, clean, process video and prepare .json file with tracking results
  ```bash
  cd test
  python3 test.py
  ```
- post request for preparing the video with selected player according to previously saved to database tracking results
  ```bash
  cd test
  python3 test_search.py
  ```
  ____
  ### Docker
  ```bash
  git clone git@github.com:fedor-konovalenko/a_hockey.git
  cd app
  docker build --tag hockey .
  docker run --rm -p 8010:8000 --name video hockey
  docker exec -it [CONTAINER_ID] sh
  cd test
  python3 test.py
  ```
  
  then the FastApi application will be available at http://localhost:8010/
