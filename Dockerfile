FROM python:3.9-slim-buster

WORKDIR /DeZero

RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "main.py" ]