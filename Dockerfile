FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip3 install --no-cache-dir -r requirement.txt

EXPOSE 8000

CMD ["python", "api.py"]