FROM python:3.10

RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY . .

RUN pip3 install -r requirement.txt

EXPOSE 8000

CMD ["python", "api.py"]