FROM python:3.10

WORKDIR /DeZero

COPY . /DeZero/

RUN pip3 install -r /DeZero/requirements.txt

CMD [ "python3", "main.py" ]
