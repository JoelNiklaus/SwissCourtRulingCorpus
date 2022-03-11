FROM python:3.7

COPY requirements.txt /app/
WORKDIR /app

RUN apt-get update && apt-get install vim -y

RUN pip install --upgrade pip \
    && pip install --trusted-host pypi.python.org -r requirements.txt

COPY wheel/* ./wheel/

RUN pip install prodigy -f ./wheel/
RUN rm -rf ./wheel/*

RUN python -m spacy download de_core_news_sm

COPY prodigy.json .
COPY recipes ./recipes/

ENV PRODIGY_HOME /app
ENV PRODIGY_LOGGING "verbose"
ENV PRODIGY_BASIC_AUTH_USER "admin"
ENV PRODIGY_BASIC_AUTH_PASS "password"

EXPOSE 8080/tcp

CMD tail -f /dev/null