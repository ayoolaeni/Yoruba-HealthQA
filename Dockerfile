# Runs the Phase 8 Gradio prototype (app/app.py) in a container.
#
# This image ships the LIGHTWEIGHT demo stack (requirements-app.txt) so it
# builds fast and stays small -- enough to show the UI and the G1/G2/G3
# safety guardrails working. It does NOT include torch/transformers/peft, so
# out of the box a question that passes the guardrails will show a "model
# not configured" placeholder instead of a generated answer (see README2.md).
#
# To actually generate answers, set YHQA_BASE_MODEL (and optionally
# YHQA_ADAPTER_PATH) as environment variables and rebuild with the full
# stack instead:
#   docker build --build-arg REQUIREMENTS_FILE=requirements.txt -t yoruba-healthqa:full .
# (that pulls in torch etc. -- a multi-GB image, and a GPU-enabled base image
# / --gpus flag at `docker run` time is needed to actually use a GPU).

FROM python:3.11-slim

WORKDIR /app

ARG REQUIREMENTS_FILE=requirements-app.txt
COPY ${REQUIREMENTS_FILE} ./requirements-to-install.txt
RUN pip install --no-cache-dir -r requirements-to-install.txt

COPY app/ ./app/
COPY src/ ./src/
COPY configs/ ./configs/

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["python", "app/app.py"]
