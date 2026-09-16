# Hosted demo of The Crossover Point: website + results + recorded replay.
# No GPU and no ML stack: CROSSOVER_DEMO_MODE=1 skips every model import.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    CROSSOVER_DEMO_MODE=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user src/__init__.py src/server.py ./src/
COPY --chown=user web ./web
COPY --chown=user charts ./charts
COPY --chown=user results ./results

EXPOSE 7860
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7860"]
