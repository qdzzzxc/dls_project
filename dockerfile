FROM postgres:latest

EXPOSE 5432

ENV POSTGRES_PASSWORD megapassword

WORKDIR /postgresql/data

VOLUME ["/postgresql/data"]
