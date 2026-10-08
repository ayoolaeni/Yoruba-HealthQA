# Runs the Phase 8 Gradio prototype (app/app.py) in a container.
#
# This image ships the LIGHTWEIGHT demo stack (requirements-app.txt) so it
# builds fast and stays small. By default (YHQA_ANSWER_MODE=retrieval, set
# in docker-compose.yml) it answers in-scope questions by looking up a
# real, human-checked answer from data/final_demo/ -- no model download,
# no Hugging Face login, no GPU, works fully offline after the image is
# built. The G1/G2/G3 safety guardrails also always work regardless of mode.
#
# To generate answers live from a trained model instead, set
# YHQA_ANSWER_MODE=model plus YHQA_BASE_MODEL (and optionally
# YHQA_ADAPTER_PATH) as environment variables and rebuild with the full
# stack instead:
#   docker build --build-arg REQUIREMENTS_FILE=requirements.txt -t yoruba-healthqa:full .
# (that pulls in torch etc. -- a multi-GB image, a real GPU/lots of RAM to
# run at a usable speed, and internet access + a gated-model licence
# acceptance the first time it downloads the base model.)

FROM python:3.11-slim

WORKDIR /app

ARG REQUIREMENTS_FILE=requirements-app.txt
COPY ${REQUIREMENTS_FILE} ./requirements-to-install.txt
RUN pip install --no-cache-dir -r requirements-to-install.txt

COPY app/ ./app/
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/final_demo/ ./data/final_demo/

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["python", "app/app.py"]
