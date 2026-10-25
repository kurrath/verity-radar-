# VERITY//RADAR — offline AI-hallucination auditor

Paste any AI-written draft. VERITY//RADAR splits it into claims, looks for evidence for
each one in **your own documents**, flags what is wrong, and rewrites only the faulty parts.
Everything runs on your machine: **no internet, no API keys, no LLM, no GPU, no pip install.**

## Run (Python 3.8+ only)

- Windows: double-click `run.bat`
- Mac / Linux: `./run.sh`  (or `python3 server.py`)

It opens http://localhost:8765. Then:
1. Click **Demo evidence** (loads facts from the three RAG papers) — or drop in your own .txt/.md/.csv/.json files.
2. Click **Load sample draft** (it has planted errors) and **Scan**.
3. Watch the radar: centre = grounded, edge = hallucinated. Click a blip to jump to that claim.

## What it detects
| Verdict | Meaning |
|---|---|
| SUPPORTED | claim is grounded in the evidence |
| PARTIAL | only partly grounded |
| CONFLICT | evidence disagrees: wrong **number**, opposite **direction** (rose/dropped) or **negation** |
| UNGROUNDED | nothing in the vault supports it |

The **Grounded Rewrite** keeps verified claims untouched, replaces partial/conflicting ones with the
exact evidence sentence, and removes ungrounded ones (selective regeneration, as in the B5G paper).

## How it works (server.py, ~400 lines, stdlib only)
decompose into sentences → index evidence in 3-sentence windows → BM25 + hashed-vector search fused with
Reciprocal Rank Fusion (k=60, as in Fusion-Seek) → IDF-weighted key-term coverage + number/direction/negation
tests → trust score = (supported + ½ partial) / claims.

## API
`GET /api/state` · `POST /api/docs {name,text}` · `POST /api/docs/delete {id}` · `POST /api/demo` ·
`POST /api/reset` · `POST /api/audit {text}`

## Honest limits
It checks claims lexically against your documents; it does not "understand" meaning, so a paraphrase with
very different words can be rated PARTIAL. Treat it as a fast first-pass auditor, not a final judge.
Set `PORT=9000` to change the port. Evidence is stored locally in `data/vault.json`.
