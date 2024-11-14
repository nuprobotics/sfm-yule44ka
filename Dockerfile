FROM hdgigante/python-opencv:4.10.0-ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y python3.12-venv

COPY . /app

WORKDIR /app

RUN python3 -m venv venv --system-site-packages
RUN /app/venv/bin/pip3 install PyYaml matplotlib

ENTRYPOINT ["./auto_test.sh", "/app/venv/bin/python3"]