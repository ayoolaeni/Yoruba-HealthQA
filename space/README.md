---
title: Yoruba HealthQA
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# Yoruba HealthQA (demo)

A Yoruba-language health question-answering prototype. Safety guardrails
(off-topic refusal, dosage/diagnosis redirection, referral disclaimer on
every answer) are always active. See `cached_answers.json` for how a small
set of pre-agreed demo questions get an instant, genuinely model-generated
answer without needing to load a large model on this Space's free CPU
hardware -- everything else falls back to a "model not configured" message
unless `YHQA_BASE_MODEL` is set as a Space secret.
