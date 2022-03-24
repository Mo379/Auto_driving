FROM python:3.9
workdir /usr/src/app
COPY ./extras/includes/requirements.txt requirements.txt
RUN /usr/local/bin/python  -m pip install --upgrade pip
RUN pip install setuptools
RUN pip install -r requirements.txt

