FROM python:3.7

COPY requirements.txt /app/
WORKDIR /app

RUN apt-get update && apt-get install vim -y

RUN pip install --upgrade pip \
    && pip install --trusted-host pypi.python.org -r requirements.txt

# Install prodigy
COPY wheel/* ./wheel/
RUN pip install prodigy -f ./wheel/
RUN rm -rf ./wheel/*

# Spacy language models for tokenization
RUN python -m spacy download de_core_news_sm
RUN python -m spacy download fr_core_news_sm
RUN python -m spacy download it_core_news_sm

# Copy relevant directories
COPY prodigy.json .
COPY recipes ./recipes/
COPY datasets ./datasets/
COPY annotations ./annotations/

# Credentials from .env file
ENV PRODIGY_HOME /app
ENV PRODIGY_LOGGING "verbose"
ARG user
ENV PRODIGY_BASIC_AUTH_USER=$user
ARG password
ENV PRODIGY_BASIC_AUTH_PASS=$password
EXPOSE 8080/tcp

CMD tail -f /dev/null