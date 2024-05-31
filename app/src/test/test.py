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
