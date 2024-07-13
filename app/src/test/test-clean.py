import requests
import json


def main():
    r = requests.post("http://localhost:8000/clean")
    if r.status_code != 200:
        print(r.status_code)
    else:
        print(r.json())


if __name__ == "__main__":
    main()
