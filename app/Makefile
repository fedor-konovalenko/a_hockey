build:
	docker build --tag hockey .

run:
	docker run --rm -it  -p 8010:8000 --gpus all --name hockey hockey

deploy: build run