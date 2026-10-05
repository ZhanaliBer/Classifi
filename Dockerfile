FROM python:3.12-slim

WORKDIR /app

COPY driver.py driver.html ./
COPY materials/*.json ./materials/

CMD ["python", "driver.py"]
