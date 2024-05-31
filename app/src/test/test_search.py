import requests
import json


def main():
    with open("test_search.json", "r") as fid:
        data = json.load(fid)

    r = requests.post("http://localhost:8000/search", json=data)
    if r.status_code != 200:
        print(r.status_code)


if __name__ == "__main__":
    main()
