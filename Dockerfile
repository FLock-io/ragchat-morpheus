FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip3 install -r requirement.txt

EXPOSE 8000

CMD ["python", "api.py"]