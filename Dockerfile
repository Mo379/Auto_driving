FROM python:3.8-alpine
workdir /usr/src/app
COPY ./extras/includes/requirements.txt requirements.txt
RUN /usr/local/bin/python  -m pip install --upgrade pip
RUN pip install setuptools
RUN pip install -r requirements.txt
COPY ./APP APP
WORKDIR /usr/src/app/APP
CMD ["python","main.py"]

