#!/usr/bin/env python3
"""
VERITY//RADAR  -  offline AI-hallucination auditor.

Paste any AI-written draft. The engine:
  1. DECOMPOSES it into individual claims (segment-level decomposition)
  2. RETRIEVES evidence for every claim from YOUR documents
        (BM25 + hashed-vector search fused with Reciprocal Rank Fusion, k=60)
  3. VERIFIES each claim: coverage of key terms, plus number / direction /
        negation conflict checks against the best evidence sentence
  4. REPAIRS only the faulty segments (selective regeneration), keeping
        verified claims untouched

Pure Python standard library: no pip install, no internet, no GPU, no LLM.
Run:  python server.py        (opens http://localhost:8765)
"""
import json
import math
import os
import re
import sys
import threading
import time
import uuid
import webbrowser
import zlib
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
STORE = DATA / "vault.json"
MAX_BODY = 6 * 1024 * 1024
RRF_K = 60

# ------------------------------------------------------------- text helpers
STOP = set("""a an the and or of to in on for with is are was were be been being it its this that these those as at by
from what which who whom how why when where do does did can could should would will shall may might not no yes i you
he she we they them his her our your their about into than then so if but also very just such each both all any some
there here over under up down out per via has have had""".split())
TOK = re.compile(r"\d+(?:[.,]\d+)*|\w+", re.UNICODE)
NUM = re.compile(r"\d+(?:\.\d+)?$")
UP = set("""rose rise rises rising increase increased increases increasing higher grew grows growth longer more improved
improves gain gains""".split())
DOWN = set("""dropped drop drops fell falls fall decrease decreased decreases decreasing lower reduced reduces reduce
reduction shorter less declined decline worse""".split())
NEG = {"not", "no", "never", "cannot", "none", "neither", "nor"}


def stem(w):
    for suf in ("ingly", "edly", "ing", "ied", "ies", "ed", "es", "ly", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf in ("ied", "ies") else "")
    return w


def is_num(t):
    return bool(NUM.match(t))


def norm_tok(w):
    """Normalised term for a raw lowercase token, or None for stop words."""
    if w[0].isdigit():
        return w.replace(",", "")
    if w in STOP or len(w) < 2:
        return None
    return stem(w)


def terms(text):
    out = []
    for w in TOK.findall(text.lower()):
        n = norm_tok(w)
        if n:
            out.append(n)
    return out


def sentences(text):
    out = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if not para:
            continue
        for s in re.split(r"(?<=[.!?।])\s+(?=[A-Z0-9\"'(\[])", para):
            s = s.strip()
            if s:
                out.append(s)
    return out


def direction(raw_words):
    up, down = bool(raw_words & UP), bool(raw_words & DOWN)
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    return None


def has_neg(text):
    w = set(re.findall(r"\w+", text.lower()))
    return bool(w & NEG) or "n't" in text.lower() or "n’t" in text.lower()


# ---------------------------------------------------------- retrieval pieces
def hvec(tlist, dim=2048):
    v = {}
    for t in tlist:
        grams = [t] + [t[i:i + 3] for i in range(max(0, len(t) - 2))]
        for g in grams:
            h = zlib.crc32(g.encode("utf-8")) % dim
            v[h] = v.get(h, 0.0) + (1.0 if g == t else 0.5)
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {k: x / n for k, x in v.items()}


def dot(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(k, 0.0) for k, x in a.items())


def rrf(lists, k=RRF_K):
    fused = {}
    for lst in lists:
        for rank, (idx, _) in enumerate(lst, 1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.docs = {}
        self.chunks = []
        self.df = Counter()
        self.tfs = []
        self.dl = []
        self.avgdl = 0.0
        self.vecs = []
        self._load()

    # --- persistence
    def _load(self):
        if STORE.exists():
            try:
                self.docs = json.loads(STORE.read_text(encoding="utf-8"))["docs"]
            except Exception:
                self.docs = {}
        self.reindex()

    def _save(self):
        STORE.write_text(json.dumps({"docs": self.docs}, ensure_ascii=False), encoding="utf-8")

    # --- indexing: 3-sentence windows, stride 2
    def reindex(self):
        self.chunks = []
        for did, d in self.docs.items():
            sents = sentences(d["text"])
            for i in range(0, len(sents), 2):
                win = sents[i:i + 3]
                self.chunks.append({
                    "doc": d["name"], "did": did, "sents": win,
                    "stsets": [set(terms(s)) for s in win],
                })
        self.tfs, self.df, self.vecs = [], Counter(), []
        for c in self.chunks:
            tl = terms(" ".join(c["sents"]))
            c["tset"] = set(tl)
            self.tfs.append(Counter(tl))
            self.df.update(c["tset"])
            self.vecs.append(hvec(tl))
        self.dl = [sum(t.values()) for t in self.tfs]
        self.avgdl = (sum(self.dl) / len(self.dl)) if self.dl else 0.0

    def idf(self, t):
        n, df = len(self.chunks), self.df.get(t, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def cov(self, cset, tset):
        tot = sum(self.idf(t) for t in cset)
        if tot <= 0:
            return 0.0
        return sum(self.idf(t) for t in cset if t in tset) / tot

    def bm25(self, q, k=20, k1=1.5, b=0.75):
        res = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in set(q):
                f = tf.get(t)
                if f:
                    s += self.idf(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * self.dl[i] / (self.avgdl or 1)))
            if s > 0:
                res.append((i, s))
        res.sort(key=lambda x: -x[1])
        return res[:k]

    def candidates(self, q, k=3):
        bm = self.bm25(q)
        qv = hvec(q)
        de = [(i, dot(qv, v)) for i, v in enumerate(self.vecs)]
        de = sorted([x for x in de if x[1] > 0], key=lambda x: -x[1])[:20]
        bmr = {i: r for r, (i, _) in enumerate(bm, 1)}
        der = {i: r for r, (i, _) in enumerate(de, 1)}
        return [(i, bmr.get(i), der.get(i)) for i, _ in rrf([bm, de])[:k]]

    # --- vault operations
    def add(self, name, text):
        text = text.replace("\x00", " ").strip()
        if len(text.split()) < 5:
            raise ValueError("Evidence is too short to index")
        did = uuid.uuid4().hex[:8]
        with self.lock:
            self.docs[did] = {"id": did, "name": name[:80] or "untitled.txt", "text": text,
                              "words": len(text.split()), "sentences": len(sentences(text)),
                              "added": time.strftime("%H:%M · %d %b")}
            self.reindex()
            self._save()
        return did

    def remove(self, did):
        with self.lock:
            self.docs.pop(did, None)
            self.reindex()
            self._save()

    def reset(self):
        with self.lock:
            self.docs = {}
            self.reindex()
            self._save()

    def listing(self):
        return [{k: d[k] for k in ("id", "name", "words", "sentences", "added")} for d in self.docs.values()]

    # --- the audit
    def audit(self, text):
        t0 = time.perf_counter()
        if not self.chunks:
            raise ValueError("The evidence vault is empty. Add documents (or load the demo evidence) first.")
        claims = []
        for s in sentences(text):
            ct = terms(s)
            if len(set(ct)) >= 2 and len(s.split()) >= 3:
                claims.append((s, ct))
        if not claims:
            raise ValueError("No checkable claims found in the draft.")
        claims = claims[:60]
        out = [self._check(i, s, ct) for i, (s, ct) in enumerate(claims, 1)]
        return self._summarise(out, time.perf_counter() - t0)

    def _check(self, i, text, ct):
        cset = set(ct)
        non_num = {t for t in cset if not is_num(t)}
        cand = self.candidates(ct)
        best = None  # (coverage, chunk index, bm rank, dense rank)
        for ci, br, dr in cand:
            cv = self.cov(cset, self.chunks[ci]["tset"])
            if best is None or cv > best[0]:
                best = (cv, ci, br, dr)
        base = {"id": i, "text": text, "evidence": None, "doc": None, "ev_segs": [], "bm_rank": None, "de_rank": None}
        if best is None or best[0] <= 0:
            return {**base, "verdict": "UNSUPPORTED", "score": 0.0,
                    "reason": "No matching evidence found anywhere in the vault.",
                    "claim_segs": self._claim_segs(text, set())}
        cv, ci, br, dr = best
        ch = self.chunks[ci]
        # best sentence inside the chunk
        si = max(range(len(ch["sents"])), key=lambda j: self.cov(cset, ch["stsets"][j]))
        ev = ch["sents"][si]
        ev_terms = ch["stsets"][si]
        ev_raw = set(re.findall(r"\w+", ev.lower()))
        cl_raw = set(re.findall(r"\w+", text.lower()))
        ctx_cov = self.cov(non_num, ch["tset"]) if non_num else cv
        claim_nums = {t for t in cset if is_num(t)}
        chunk_nums = {t for t in ch["tset"] if is_num(t)}
        sent_nums = sorted({t for t in ev_terms if is_num(t)})
        missing = sorted(claim_nums - chunk_nums)

        verdict, reason = None, None
        if missing and sent_nums and ctx_cov >= 0.5:
            verdict = "CONFLICT"
            pool = [x for x in sent_nums if x not in claim_nums] or sent_nums
            expected = [min(pool, key=lambda x: abs(float(x) - float(m))) for m in missing]
            reason = "Number mismatch: the claim says %s, but the evidence says %s." % (
                ", ".join(missing), ", ".join(dict.fromkeys(expected)))
        elif direction(cl_raw) and direction(ev_raw) and direction(cl_raw) != direction(ev_raw) and ctx_cov >= 0.4:
            verdict = "CONFLICT"
            reason = "Direction mismatch: the claim implies %s, but the evidence describes %s." % (
                "a decrease" if direction(cl_raw) == "down" else "an increase",
                "a decrease" if direction(ev_raw) == "down" else "an increase")
        elif has_neg(text) != has_neg(ev) and ctx_cov >= 0.6:
            verdict = "CONFLICT"
            reason = "Negation mismatch: the claim and the evidence disagree on whether this is true."
        elif cv >= 0.72:
            verdict = "SUPPORTED"
            reason = "Grounded in the evidence (%d%% of key terms matched)." % round(cv * 100)
        elif cv >= 0.40:
            verdict = "PARTIAL"
            reason = "Only partly grounded (%d%% of key terms matched)." % round(cv * 100)
        else:
            verdict = "UNSUPPORTED"
            reason = "No real evidence for this claim (%d%% of key terms matched)." % round(cv * 100)

        novel = []
        for w in TOK.findall(text.lower()):
            n = norm_tok(w)
            if n and n not in ch["tset"] and w not in novel:
                novel.append(w)
        if verdict in ("PARTIAL", "UNSUPPORTED") and novel:
            reason += " Ungrounded: " + ", ".join(novel[:5]) + "."

        score = min(cv, 0.35) if verdict == "CONFLICT" else cv
        if verdict == "UNSUPPORTED":
            score = min(score, 0.3)
        return {**base, "verdict": verdict, "score": round(score, 3), "reason": reason,
                "evidence": ev, "doc": ch["doc"], "bm_rank": br, "de_rank": dr, "weak": cv < 0.25,
                "ev_segs": self._ev_segs(ev, cset),
                "claim_segs": self._claim_segs(text, ch["tset"])}

    @staticmethod
    def _ev_segs(ev, cset):
        segs = []
        for part in re.split(r"(\d+(?:[.,]\d+)*|\w+)", ev):
            if not part:
                continue
            n = norm_tok(part.lower()) if TOK.fullmatch(part) else None
            segs.append([part, 1 if (n and n in cset) else 0])
        return segs

    @staticmethod
    def _claim_segs(text, tset):
        segs = []
        for part in re.split(r"(\d+(?:[.,]\d+)*|\w+)", text):
            if not part:
                continue
            state = 0
            if TOK.fullmatch(part):
                n = norm_tok(part.lower())
                if n:
                    state = 1 if n in tset else 2
            segs.append([part, state])
        return segs

    def _summarise(self, res, secs):
        counts = Counter(r["verdict"] for r in res)
        n = len(res)
        trust = round(100 * (counts["SUPPORTED"] + 0.5 * counts["PARTIAL"]) / n)
        label = "TRUSTWORTHY" if trust >= 85 else "NEEDS REVIEW" if trust >= 60 else "HIGH RISK"
        repaired, used = [], set()
        for r in res:
            if r["verdict"] == "SUPPORTED":
                repaired.append({"kind": "keep", "text": r["text"]})
            elif r["verdict"] in ("PARTIAL", "CONFLICT") and r["evidence"]:
                if r["evidence"] in used:
                    repaired.append({"kind": "remove", "text": r["text"], "why": "duplicate of an earlier fix"})
                else:
                    used.add(r["evidence"])
                    repaired.append({"kind": "replace", "text": r["evidence"], "orig": r["text"], "doc": r["doc"]})
            else:
                repaired.append({"kind": "remove", "text": r["text"], "why": "no supporting evidence"})
        fixed = " ".join(s["text"] for s in repaired if s["kind"] != "remove")
        return {"claims": res, "trust": trust, "label": label, "counts": {k: counts.get(k, 0) for k in
                ("SUPPORTED", "PARTIAL", "CONFLICT", "UNSUPPORTED")},
                "repaired": repaired, "repaired_text": fixed, "ms": round(secs * 1000, 1), "n": n}


ENGINE = Engine()

# ------------------------------------------------------------------- demo
DEMO_DOCS = {
    "fusion-seek-results.txt": (
        "Fusion-Seek hybrid retrieval reached a Recall@10 of 87.4 percent, a Precision@10 of 84.9 percent and an MRR of "
        "0.82. BM25 keyword search alone reached a Recall@10 of 62.3 percent. Dense semantic retrieval alone reached a "
        "Recall@10 of 75.1 percent. A cloud RAG baseline using GPT-3.5 reached 89.1 percent Recall@10 but transmits "
        "document content to external servers. The best chunk size was 400 tokens with 100 tokens of overlap. "
        "End-to-end query latency was 1.12 seconds on a laptop CPU without a GPU. The system answers questions with "
        "Mistral-7B-Instruct running through Ollama. Reciprocal Rank Fusion uses a smoothing constant k of 60. "
        "The evaluation corpus contained 250 documents and 11,340 chunks."),
    "b5g-config-results.txt": (
        "The decomposed generation with decomposed verification approach reached a 94.4 percent success rate in network "
        "configuration. This is 22.7 percent better than monolithic methods. The framework was tested on the "
        "OpenAirInterface emulator with the GPT-OSS-120B model. Mean total inference time rose from 45.886 seconds for "
        "the monolithic pipeline to 79.961 seconds for the decomposed pipeline. The verifier sends back only the faulty "
        "segments for regeneration. Deployment commands require human approval before execution. The knowledge base "
        "held 37 gNB and 5 UE configuration files."),
    "offline-rag-local.txt": (
        "The offline RAG system uses ChromaDB as the vector database and the nomic-embed-text model for embeddings. "
        "Language models such as DeepSeek and Gemma run locally through the Ollama API. The interface is built with "
        "Streamlit and the pipeline is orchestrated with LangChain. The system was tested on a machine with 16GB of RAM "
        "and a mid-range CPU without GPU acceleration. Embedding time scales linearly with file length. A noted "
        "limitation is reduced OCR accuracy for handwritten content."),
}

SAMPLE = (
    "Fusion-Seek hybrid retrieval reached a Recall@10 of 91.2 percent. "
    "BM25 keyword search alone reached a Recall@10 of 62.3 percent. "
    "The best chunk size was 400 tokens with 100 tokens of overlap. "
    "The system requires a cluster of high-end GPUs to run at all. "
    "The decomposed framework reached a 94.4 percent success rate on the OpenAirInterface emulator. "
    "Total inference time dropped from 79.961 seconds to 45.886 seconds with decomposition. "
    "The offline RAG system uses ChromaDB and the nomic-embed-text model. "
    "The team won the IEEE best paper award in 2026."
)


def load_demo():
    have = {d["name"] for d in ENGINE.docs.values()}
    for name, text in DEMO_DOCS.items():
        if name not in have:
            ENGINE.add(name, text)


# -------------------------------------------------------------------- HTTP
MIME = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}


class Handler(BaseHTTPRequestHandler):
    server_version = "VerityRadar/1.0"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            return self._send(200, {"docs": ENGINE.listing(), "sample": SAMPLE, "chunks": len(ENGINE.chunks)})
        if path == "/":
            path = "/index.html"
        target = (WEB / path.lstrip("/")).resolve()
        if WEB.resolve() not in target.parents or not target.is_file():
            return self._send(404, {"error": "not found"})
        self._send(200, target.read_bytes(), MIME.get(target.suffix, "application/octet-stream"))

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._send(413, {"error": "Payload too large (max 6 MB)"})
            data = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, {"error": "Invalid JSON"})
        try:
            p = self.path.split("?", 1)[0]
            if p == "/api/audit":
                return self._send(200, ENGINE.audit(str(data.get("text", ""))))
            if p == "/api/docs":
                ENGINE.add(str(data.get("name", "pasted-evidence.txt")), str(data.get("text", "")))
            elif p == "/api/docs/delete":
                ENGINE.remove(str(data.get("id", "")))
            elif p == "/api/demo":
                load_demo()
            elif p == "/api/reset":
                ENGINE.reset()
            else:
                return self._send(404, {"error": "not found"})
            return self._send(200, {"docs": ENGINE.listing(), "chunks": len(ENGINE.chunks)})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:  # never crash the server
            return self._send(500, {"error": "Internal error: %s" % e})


def main():
    base = int(os.getenv("PORT", "8765"))
    server = None
    for port in range(base, base + 15):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        sys.exit("Could not find a free port.")
    url = "http://localhost:%d" % server.server_address[1]
    print("\n  VERITY//RADAR is live at %s   (Ctrl+C to stop)\n" % url)
    if not os.getenv("NO_BROWSER"):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Radar offline. Bye.")


if __name__ == "__main__":
    main()
