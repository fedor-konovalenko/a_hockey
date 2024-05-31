build:
	docker build --tag hockey .

run:
	docker run --rm -p 8010:8000 --name video hockey

deploy: build run