# Yorùbá HealthQA — Plain-Language Guide (for your defence)

This document explains the project in everyday language: what it is, how it
works step by step, and how to install and run it with Docker Desktop so you
(or your supervisor/examiners) can see it working on any laptop with a few
clicks. Keep `README.md` for the technical/academic reproduction steps; use
this one for explaining and demonstrating the project.

---

## 1. What is this project, in one paragraph?

It's a **safety-first pipeline for building a Yorùbá-language health
question-answering assistant** — think "a chatbot that answers everyday
health questions in Yorùbá" — plus all the scaffolding a dissertation needs
to prove the work was done properly: real source data, human review steps,
measurable scores, and a working demo app. The whole thing is built as a
sequence of **9 phases**, each one a set of small, testable programs, so
every step can be checked and re-run independently instead of being one
giant black box.

---

## 2. How the project works, phase by phase

Think of it like an assembly line. Raw health information goes in one end;
a safe, tested Yorùbá health assistant comes out the other end. Each phase
below is a station on that line.

| # | Phase (plain English) | What actually happens | Status in this build |
|---|---|---|---|
| 1 | **Collect and prepare the data** | Pull real health articles (WHO, MedlinePlus), turn them into question-and-answer pairs, translate them to Yorùbá, and have real humans (a Yorùbá speaker + a health worker) check every pair is correct and safe. | ✅ Tooling built and tested. 714 real draft QA pairs collected so far from real WHO/MedlinePlus pages. ⏳ Translation, human proof-reading, and clinical sign-off still need to happen — that's real human work, not something software can fake. |
| 2 | **Pick the "brain" (base AI model)** | Test a few candidate open-source AI models to see which one understands Yorùbá most efficiently, and pick one. | ✅ Testing tool built and proven on real text. ⏳ Final model choice is a decision for the researcher once enough Yorùbá sample text is ready. |
| 3 | **(Optional) Extra Yorùbá language practice** | Before teaching the AI to answer health questions, optionally give it extra reading practice in plain Yorùbá text, to make it more fluent first. | ✅ Code built. ⏳ Needs a GPU computer to actually run. |
| 4 | **Teach the AI health Q&A (training)** | Using the reviewed Yorùbá question/answer pairs, fine-tune the chosen AI model so it learns to answer health questions the way the dissertation intends. | ✅ Code built (a standard, efficient technique called QLoRA). ⏳ Needs a GPU computer and the finished, human-reviewed dataset. |
| 5 | **Safety guardrails** | Three hard-coded safety rules that run *before and after* the AI, so the AI itself is never the last line of defence: <br>1️⃣ **Off-topic question?** → politely refuse. <br>2️⃣ **Asking for a diagnosis or drug dosage?** → redirect to a real doctor/nurse. <br>3️⃣ **Every answer** → always ends with a safety disclaimer. | ✅ **Fully built, tested, and working right now** — no AI model needed to see these work. |
| 6 | **Score the AI automatically** | Compare the trained AI's answers against 4 other approaches (a plain AI with no training, a "few examples shown" AI, a "translate-then-answer-then-translate-back" approach, and a big commercial AI) using standard scoring methods, plus statistics to prove any difference is real and not luck. | ✅ Code built and proven correct with test data. ⏳ Needs the trained model to score for real. |
| 7 | **Human judges the AI's answers** | A panel of raters scores a sample of answers on correctness, fluency, cultural appropriateness, and — critically — *any potential for harm*, without knowing which answer came from which system (a "blind taste test"). | ✅ Tooling built and proven with test data (blinding, scoring, agreement statistics all verified working). ⏳ Needs a real rating panel. |
| 8 | **The prototype app** | A simple chat webpage where anyone can type a Yorùbá health question and get an answer, with the safety rules always active. | ✅ **Built and working — this is the part you can demo today.** |
| 9 | **Error analysis** | Look at the worst-scoring answers and work out *why* they went wrong (wrong fact? left something out? wrong dialect word? unsafe?), so the dissertation can honestly discuss weaknesses. | ✅ Tooling built and proven. ⏳ Needs a trained model's real answers and a human reviewer's judgement calls. |

**In one sentence for your defence:** *"The engineering pipeline — data
collection, safety system, scoring, and the demo app — is built and proven
working; the remaining steps are the human-in-the-loop work (translation
review, clinical sign-off, rating panel) and the GPU training run, which the
dissertation methodology deliberately keeps as human/researcher
responsibilities rather than something the software should fake."*

---

## 3. How to install and run it with Docker Desktop

This is the easy way to show the app to anyone — your supervisor, examiners,
or a client — **without** them installing Python, or any code libraries, on
their computer. Docker packages the whole app into one self-contained box
("container") that runs the same way on any Windows, Mac, or Linux laptop.

### Step 1 — Install Docker Desktop (one-time, ~10 minutes)

1. Go to **docker.com/products/docker-desktop** and download Docker Desktop
   for your operating system (Windows or Mac).
2. Run the installer and restart your computer if it asks you to.
3. Open **Docker Desktop** from the Start Menu (Windows) or Applications
   folder (Mac) and wait until it says it's running (the whale icon in the
   system tray/menu bar stops animating).

You only need to do this once.

### Step 2 — Get the project folder onto that laptop

Copy the whole project folder (the one with `Dockerfile`, `app/`, `src/`,
etc. in it) onto the laptop — for example via a USB drive, or by cloning it
if it's on GitHub.

### Step 3 — Open a terminal in the project folder

- **Windows:** open the project folder in File Explorer, click the address
  bar, type `powershell`, and press Enter.
- **Mac:** open Terminal, type `cd `, drag the project folder into the
  Terminal window, and press Enter.

### Step 4 — Start the app with one command

```
docker compose up --build
```

The first time, this takes a minute or two while it downloads what it needs.
Wait until you see a line like:

```
* Running on local URL:  http://0.0.0.0:7860
```

### Step 5 — Open it in a browser

Open a web browser and go to:

```
http://localhost:7860
```

You'll see the **Yorùbá HealthQA** page with a text box. Type a question in
Yorùbá and press the button.

### Step 6 — Stop it when you're done

Go back to the terminal window and press `Ctrl + C`, then run:

```
docker compose down
```

That's it — nothing was permanently installed on the computer beyond Docker
Desktop itself; deleting the project folder removes everything else.

### If something goes wrong

- **"docker: command not found"** → Docker Desktop isn't running yet — open
  it and wait for the whale icon to settle, then try again.
- **Port 7860 already in use** → something else on that laptop is using that
  port; edit `docker-compose.yml`, change `"7860:7860"` to `"8080:7860"`,
  save, run `docker compose up --build` again, and open
  `http://localhost:8080` instead.

---

## 4. What you'll actually see when you demo it (and why it's honest, not broken)

Out of the box (without a trained AI model plugged in), the app will:

- ✅ **Correctly refuse** a question that has nothing to do with health.
- ✅ **Correctly redirect** a question asking "what dosage should I take."
- ⚠️ Show a polite **"no model loaded yet"** message for a genuine health
  question, instead of a generated answer.

**This is expected and is actually a good thing to point out in your
defence**: it proves the *safety system* is completely independent of the
AI model — the guardrails work whether or not an AI is even plugged in,
exactly as the methodology requires (the safety rules must never depend on
the AI "deciding" to be safe).

If you *do* have a trained model ready later, you switch it on by setting
two settings before starting the app — no code changes needed:

```
YHQA_BASE_MODEL=<the model's name>
YHQA_ADAPTER_PATH=<path to the trained fine-tuning file>
docker compose up --build
```

---

## 5. A simple way to describe the "how it works" flow in your defence

Draw or describe this flow when asked "how does a question get answered?":

```
Yorùbá question typed in
        │
        ▼
 [Guardrail 1] Is this even about health?
        │ No → polite refusal, stop here
        ▼ Yes
 [Guardrail 2] Is this asking for a diagnosis or drug dosage?
        │ Yes → redirect to a real doctor/nurse, stop here
        ▼ No
   The trained AI model generates an answer
        │
        ▼
 [Guardrail 3] Always attach the safety disclaimer
        │
        ▼
     Answer shown to the user
```

The key point examiners often probe: **the AI is never the last word on
safety** — the three checks around it are plain, predictable rules (not
"learned" behaviour that could fail unpredictably), which is why they were
built this way instead of just trusting the AI to be careful.

---

## 6. Likely defence questions and short honest answers

**Q: Is the dataset finished?**
No — 714 draft English pairs exist from real WHO/MedlinePlus sources, out of
a target of 1,500–3,000 fully Yorùbá, human-reviewed pairs. Translation and
human review are the next steps.

**Q: Has the AI model actually been trained yet?**
No — training needs the finished, human-reviewed Yorùbá dataset and a GPU
computer, neither of which was available at this stage. The training code
itself is written and ready to run once those two things exist.

**Q: How do you know the safety rules actually work, if there's no AI yet?**
Because the safety rules don't depend on the AI at all — they're plain
keyword/rule checks that run before the AI is even asked anything. They've
been tested with dozens of example questions and all pass.

**Q: Why build the app before the AI is trained?**
So the whole system — data flow, safety checks, the web interface — can be
proven to work end-to-end early, rather than discovering a plumbing problem
only after weeks of expensive GPU training.

**Q: Is any part of this made up or faked?**
No. Every number and file described here was actually produced by running
the real programs against real websites and real test cases — nothing was
typed in by hand pretending to be a result. Anywhere a human judgement call
is required (translation quality, clinical safety, rating an answer), the
software stops and asks for one instead of guessing.
