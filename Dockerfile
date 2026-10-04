FROM python:3.12-slim

WORKDIR /app

COPY driver.py driver.html ./

CMD ["python", "driver.py"]
