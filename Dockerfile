FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY driver.py driver.html tutor_graph.py ./
COPY materials/*.json ./materials/

CMD ["python", "driver.py"]
