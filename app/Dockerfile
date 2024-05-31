FROM python:3.10.2

COPY requirements.txt /app/requirements.txt
COPY src /app

WORKDIR /app

RUN pip install -r requirements.txt


CMD ["python", "app.py"]