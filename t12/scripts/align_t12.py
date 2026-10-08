#!/usr/bin/env python
"""Phoneme-level CTC forced alignment for the T12 speech BCI dataset.

Runs the authors' pretrained baseline RNN (Willett et al. 2023) on CPU, greedy-decodes
for a sanity PER, then does Viterbi CTC forced alignment of the ground-truth phoneme
sequence against the RNN logits and maps each phoneme back to 20 ms neural bins.

Two input routes, both covering all 24 sessions and verified to agree to 1.9e-6:
  --source tfrecords   derived/tfRecords/<session>/<partition>/chunk_0.tfrecord
  --source mat         competitionData/<partition>/<session>.mat   (recomputes features)

Only 19 sessions have a trained day-specific input layer; the other 5 (the 4 nonvocal
days plus the vocal t12.2022.07.29) borrow one, and their alignments are approximate.

Usage examples
  python align_t12.py --sessions t12.2022.08.13 --partitions test
  python align_t12.py --sessions all-vocal --partitions test train
  python align_t12.py --source mat --sessions t12.2022.06.23 --partitions test train \
                      --input-layer-from t12.2022.06.21
  python align_t12.py --combine
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import OmegaConf

import neuralDecoder.models as models
from neuralDecoder.datasets.speechDataset import PHONE_DEF_SIL, SIL_DEF
from neuralDecoder.neuralSequenceDecoder import gaussSmooth

T12 = Path(__file__).resolve().parent.parent
CKPT_DIR = T12 / "data" / "derived" / "rnns" / "baselineRelease"
TFREC_DIR = T12 / "data" / "derived" / "tfRecords"
MAT_DIR = T12 / "data" / "competitionData"
OUT_DIR = T12 / "results" / "alignments"

N_PHONES = len(PHONE_DEF_SIL)          # 40 (index 39 == 'SIL')
BLANK = N_PHONES                        # 40 -> blank is the LAST logit column
SIL_ID = PHONE_DEF_SIL.index("SIL")     # 39
NEG = -1e30

# all 24 recorded sessions and their modality, from AnalysisExamples/getSpeechSessionBlocks.py
ALL_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]
NONVOCAL = {"t12.2022.06.23", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25"}

# Sessions with no trained day-specific input layer (not among args.yaml's 19 training
# sessions) borrow the nearest preceding *trained* session's layer. Note t12.2022.07.29
# is VOCAL but was still left out of training, so it needs a donor too.
DEFAULT_DONOR = {
    "t12.2022.06.23": "t12.2022.06.21",
    "t12.2022.07.29": "t12.2022.07.27",
    "t12.2022.08.18": "t12.2022.08.13",
    "t12.2022.08.23": "t12.2022.08.13",
    "t12.2022.08.25": "t12.2022.08.13",
}


def log(msg, logfile=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------- model


class BaselineRNN:
    """Rebuilds exactly the trainable objects NeuralSequenceDecoder checkpoints
    (net + inputLayer_i + normLayer_i) and restores them, without touching the
    tf.data pipeline (which would require every session's tfRecords to exist)."""

    def __init__(self, ckpt_dir=CKPT_DIR, ckpt_idx=None):
        args = OmegaConf.load(os.path.join(ckpt_dir, "args.yaml"))
        self.args = args
        self.sessions = list(args["dataset"]["sessions"])
        self.layer_map = list(args["dataset"]["datasetToLayerMap"])
        self.n_input_layers = int(np.max(self.layer_map)) + 1
        self.n_features = int(args["dataset"]["nInputFeatures"])
        self.n_classes = int(args["dataset"]["nClasses"])          # 40
        self.kernel = int(args["model"]["stack_kwargs"]["kernel_size"])
        self.stride = int(args["model"]["stack_kwargs"]["strides"])
        self.smooth = bool(args["smoothInputs"])
        self.smooth_sd = float(args["smoothKernelSD"])

        self.model = models.GRU(
            args["model"]["nUnits"], args["model"]["weightReg"], args["model"]["actReg"],
            args["model"]["subsampleFactor"], self.n_classes + 1,
            args["model"]["bidirectional"], args["model"]["dropout"],
            args["model"].get("nLayers", 2),
            conv_kwargs=args["model"].get("conv_kwargs", None),
            stack_kwargs=args["model"].get("stack_kwargs", None),
        )
        self.model(tf.keras.Input(shape=(None, args["model"]["inputNetwork"]["inputLayerSizes"][-1])))

        # day-specific input networks, identical construction to _buildInputNetworks
        self.input_layers, self.norm_layers = [], []
        innet = args["model"]["inputNetwork"]
        for _ in range(self.n_input_layers):
            norm = tf.keras.layers.experimental.preprocessing.Normalization(
                input_shape=[self.n_features])
            m = tf.keras.Sequential()
            m.add(tf.keras.Input(shape=(None, self.n_features)))
            for i in range(innet["nInputLayers"]):
                m.add(tf.keras.layers.Dense(innet["inputLayerSizes"][i],
                                            activation=innet["activation"]))
                m.add(tf.keras.layers.Dropout(rate=innet["dropout"]))
            m(tf.zeros([1, 2, self.n_features]))          # force build
            norm(tf.zeros([1, 2, self.n_features]))
            self.input_layers.append(m)
            self.norm_layers.append(norm)

        ckpt_vars = {"net": self.model}
        for i in range(self.n_input_layers):
            ckpt_vars[f"normLayer_{i}"] = self.norm_layers[i]
            ckpt_vars[f"inputLayer_{i}"] = self.input_layers[i]
        ckpt = tf.train.Checkpoint(**ckpt_vars)
        path = (os.path.join(ckpt_dir, f"ckpt-{ckpt_idx}") if ckpt_idx is not None
                else tf.train.latest_checkpoint(str(ckpt_dir)))
        if path is None:
            raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")
        self.ckpt_path = path
        status = ckpt.restore(path)
        # every object we built must have received a value; the checkpoint additionally
        # holds optimizer slots / step / bestValCer which inference does not need
        status.assert_existing_objects_matched()
        status.expect_partial()
        self._status = status

        self._fwd = {}        # one traced graph per day-specific input layer, built lazily

    def _graph_for(self, i):
        if i not in self._fwd:
            def fwd(feats, i=i):
                x = self.norm_layers[i](feats)
                if self.smooth:
                    x = gaussSmooth(x, kernelSD=self.smooth_sd)
                x = self.input_layers[i](x, training=False)
                return self.model(x, training=False)
            self._fwd[i] = tf.function(
                fwd,
                input_signature=[tf.TensorSpec([None, None, self.n_features], tf.float32)],
                reduce_retracing=True)
        return self._fwd[i]

    def layer_for_session(self, session):
        if session not in self.sessions:
            raise KeyError(f"{session} has no trained input layer; use --input-layer-from")
        return self.layer_map[self.sessions.index(session)]

    def n_steps(self, n_bins):
        """Mirror models.GRU.getSubsampledTimeSteps (subsampleFactor == 1)."""
        return int((n_bins - self.kernel) // self.stride + 1)

    def logits(self, feats_batch, layer_idx):
        return self._graph_for(int(layer_idx))(tf.constant(feats_batch, tf.float32)).numpy()


# ---------------------------------------------------------------- CTC


def greedy_decode(logits, n_steps):
    """argmax -> collapse repeats -> drop blank. Matches tf.nn.ctc_greedy_decoder."""
    best = np.argmax(logits[:n_steps], axis=-1)
    keep = np.concatenate(([True], best[1:] != best[:-1]))
    seq = best[keep]
    return seq[seq != BLANK]


def edit_distance(a, b):
    if len(a) == 0:
        return len(b)
    prev = np.arange(len(a) + 1)
    for j, bj in enumerate(b, 1):
        cur = np.empty_like(prev)
        cur[0] = j
        for i in range(1, len(a) + 1):
            cur[i] = min(prev[i] + 1, cur[i - 1] + 1,
                         prev[i - 1] + (a[i - 1] != bj))
        prev = cur
    return int(prev[-1])


def ctc_forced_align(log_probs, labels):
    """Standard CTC Viterbi forced alignment over the extended state sequence
    [blank, l0, blank, l1, ..., l_{L-1}, blank].

    Returns (states, score) where states[t] is the extended-state index occupied at
    RNN step t; label k lives at extended state 2k+1. A label may occupy several
    consecutive steps, so phoneme durations come out of the alignment rather than
    being fixed at one step (which is what the authors' pytorch-tutorial helper does).
    """
    S = log_probs.shape[0]
    L = len(labels)
    M = 2 * L + 1
    ext = np.full(M, BLANK, dtype=np.int64)
    ext[1::2] = labels
    emit = log_probs[:, ext]                       # (S, M)

    # a skip (i-2 -> i) is legal only into a label state whose label differs from the
    # previous label, otherwise CTC needs the separating blank
    can_skip = np.zeros(M, dtype=bool)
    if L >= 2:
        can_skip[3::2] = labels[1:] != labels[:-1]
    min_steps = L + int(np.sum(labels[1:] == labels[:-1]))
    if S < min_steps:
        return None, -np.inf

    dp = np.full((S, M), NEG)
    bp = np.zeros((S, M), np.int8)
    dp[0, 0] = emit[0, 0]
    if M > 1:
        dp[0, 1] = emit[0, 1]
    for t in range(1, S):
        p = dp[t - 1]
        stay = p
        adv = np.concatenate(([NEG], p[:-1]))
        skip = np.where(can_skip, np.concatenate(([NEG, NEG], p[:-2])), NEG)
        stacked = np.stack([stay, adv, skip])
        choice = np.argmax(stacked, axis=0)
        dp[t] = stacked[choice, np.arange(M)] + emit[t]
        bp[t] = choice

    ends = [M - 1, M - 2] if M >= 2 else [0]
    j = max(ends, key=lambda k: dp[S - 1, k])
    score = dp[S - 1, j]
    if not np.isfinite(score) or score <= NEG / 2:
        return None, -np.inf
    states = np.empty(S, np.int64)
    for t in range(S - 1, -1, -1):
        states[t] = j
        if t:
            j -= int(bp[t][j])
    return states, float(score)


def spans_from_states(states, L):
    """Core (label-occupancy) step span per phoneme + a gap-free partition that
    splits each intervening blank run at its midpoint."""
    core = []
    for k in range(L):
        fr = np.flatnonzero(states == 2 * k + 1)
        if fr.size == 0:
            return None, None
        core.append((int(fr[0]), int(fr[-1])))
    S = len(states)
    seg = []
    for k in range(L):
        s0 = 0 if k == 0 else (core[k - 1][1] + 1 + core[k][0] + 1) // 2
        s1 = S - 1 if k == L - 1 else (core[k][1] + 1 + core[k + 1][0] + 1) // 2 - 1
        seg.append((s0, max(s0, s1)))
    return core, seg


# ---------------------------------------------------------------- labels / words


def phonemes_from_text(sentence, g2p):
    """Exactly the recipe in AnalysisExamples/makeTFRecordsFromSession.py."""
    import re
    t = re.sub(r"[^a-zA-Z\- \']", "", sentence)
    t = t.replace("--", "").lower()
    if len(t) == 0:
        return list(SIL_DEF), t
    out = []
    for p in g2p(t):
        if p == " ":
            out.append("SIL")
        p = re.sub(r"[0-9]", "", p)
        if re.match(r"[A-Z]+", p):
            out.append(p)
    out.append("SIL")
    return out, t


def word_structure(labels, sentence):
    """word_idx + position_in_word for each phoneme. SIL closes a word."""
    words = sentence.split()
    word_idx = np.full(len(labels), -1, np.int64)
    pos = ["" for _ in labels]
    wtext = ["" for _ in labels]
    w, buf = 0, []
    for i, lab in enumerate(labels):
        if lab == SIL_ID:
            for n, j in enumerate(buf):
                word_idx[j] = w
                pos[j] = ("single" if len(buf) == 1 else
                          "initial" if n == 0 else
                          "final" if n == len(buf) - 1 else "medial")
            word_idx[i] = w
            pos[i] = "sil"
            buf = []
            w += 1
        else:
            buf.append(i)
    for n, j in enumerate(buf):                       # trailing word with no SIL
        word_idx[j] = w
        pos[j] = ("single" if len(buf) == 1 else
                  "initial" if n == 0 else
                  "final" if n == len(buf) - 1 else "medial")
    n_words = w + (1 if buf else 0)
    if len(words) == n_words:
        for i in range(len(labels)):
            if 0 <= word_idx[i] < len(words):
                wtext[i] = words[word_idx[i]]
    return word_idx, pos, wtext


# ---------------------------------------------------------------- trial sources


def ascii_to_text(arr):
    arr = np.asarray(arr).ravel()
    end = np.flatnonzero(arr == 0)
    n = end[0] if end.size else len(arr)
    return "".join(chr(int(c)) for c in arr[:n])


def trials_from_tfrecord(session, partition, n_features=256, max_seq=500):
    path = TFREC_DIR / session / partition / "chunk_0.tfrecord"
    if not path.exists():
        return
    spec = {
        "inputFeatures": tf.io.FixedLenSequenceFeature([n_features], tf.float32, allow_missing=True),
        "seqClassIDs": tf.io.FixedLenFeature((max_seq,), tf.int64),
        "nTimeSteps": tf.io.FixedLenFeature((), tf.int64),
        "nSeqElements": tf.io.FixedLenFeature((), tf.int64),
        "transcription": tf.io.FixedLenFeature((max_seq,), tf.int64),
    }
    for i, rec in enumerate(tf.data.TFRecordDataset([str(path)])):
        d = tf.io.parse_single_example(rec, spec)
        n_t = int(d["nTimeSteps"].numpy())
        n_s = int(d["nSeqElements"].numpy())
        # seqClassIDs are stored as PHONE_DEF_SIL index + 1; CTC labels are index
        labels = d["seqClassIDs"].numpy()[:n_s].astype(np.int64) - 1
        yield {
            "trial_idx": i,
            "features": d["inputFeatures"].numpy()[:n_t],
            "labels": labels,
            "sentence": ascii_to_text(d["transcription"].numpy()),
        }


def _mat_sentence(st, i):
    """sentenceText comes back either as an (S,) array of str or an (S, C) char matrix."""
    st = np.asarray(st)
    v = st[i]
    if isinstance(v, bytes):
        v = v.decode("utf-8", "ignore")
    if isinstance(v, np.ndarray):
        v = "".join(x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                    for x in v.ravel().tolist())
    return str(v).strip()


def trials_from_mat(session, partition, g2p):
    import scipy.io
    path = MAT_DIR / partition / f"{session}.mat"
    if not path.exists():
        return
    dat = scipy.io.loadmat(str(path))
    n = dat["tx1"].shape[1]
    feats = [np.concatenate([np.asarray(dat["tx1"][0, i][:, :128], np.float64),
                             np.asarray(dat["spikePow"][0, i][:, :128], np.float64)], axis=1)
             for i in range(n)]
    # blockwise z-score, exactly as makeTFRecordsFromSession.py
    blocks = np.squeeze(np.asarray(dat["blockIdx"]))
    for b in np.unique(blocks):
        idx = np.flatnonzero(blocks == b)
        pool = np.concatenate(feats[idx[0]:idx[-1] + 1], axis=0)
        mu, sd = pool.mean(0, keepdims=True), pool.std(0, keepdims=True)
        for i in idx:
            feats[i] = (feats[i] - mu) / (sd + 1e-8)
    for i in range(n):
        phones, clean = phonemes_from_text(_mat_sentence(dat["sentenceText"], i), g2p)
        yield {
            "trial_idx": i,
            "features": feats[i].astype(np.float32),
            "labels": np.array([PHONE_DEF_SIL.index(p) for p in phones], np.int64),
            "sentence": clean,
        }


# ---------------------------------------------------------------- driver


def _assert_partition(df):
    """seg_[start,end)_bin must be a gap-free, non-overlapping, monotonic cover of the
    span from the first to the last phoneme of each trial, and the tight centre window
    must sit inside it."""
    for tid, g in df.groupby("trial_idx"):
        s = g.seg_start_bin.to_numpy()
        e = g.seg_end_bin.to_numpy()
        if not np.all(e > s):
            raise AssertionError(f"trial {tid}: non-positive seg width")
        if not np.array_equal(s[1:], e[:-1]):
            bad = int(np.flatnonzero(s[1:] != e[:-1])[0])
            raise AssertionError(
                f"trial {tid}: seg not contiguous at phone {bad} "
                f"(end={e[bad]} next start={s[bad + 1]})")
        if not np.all(g.rnn_center_end.to_numpy() > g.rnn_center_start.to_numpy()):
            raise AssertionError(f"trial {tid}: non-positive centre window")
        if not (np.all(g.rnn_center_start.to_numpy() >= s)
                and np.all(g.rnn_center_end.to_numpy() <= e)):
            raise AssertionError(f"trial {tid}: centre window outside seg span")


def run_session(rnn, session, partition, source, g2p, layer_idx, modality, donor,
                batch_size=16, logfile=None, save_logits=True):
    gen = (trials_from_tfrecord(session, partition) if source == "tfrecords"
           else trials_from_mat(session, partition, g2p))
    trials = [t for t in gen]
    if not trials:
        log(f"  {session}/{partition}: no data, skipped", logfile)
        return None

    order = np.argsort([len(t["features"]) for t in trials])   # minimise padding
    rows, summary, logit_store = [], [], {}
    t0 = time.time()

    for b0 in range(0, len(order), batch_size):
        chunk = [trials[i] for i in order[b0:b0 + batch_size]]
        tmax = max(len(t["features"]) for t in chunk)
        batch = np.zeros((len(chunk), tmax, rnn.n_features), np.float32)
        for i, t in enumerate(chunk):
            batch[i, :len(t["features"])] = t["features"]
        out = rnn.logits(batch, layer_idx)

        for i, t in enumerate(chunk):
            S = rnn.n_steps(len(t["features"]))
            lg = out[i, :S].astype(np.float64)
            labels = t["labels"]
            dec = greedy_decode(lg, S)
            ed = edit_distance(list(dec), list(labels))
            per = ed / max(1, len(labels))

            lp = lg - np.max(lg, axis=1, keepdims=True)
            lp = lp - np.log(np.sum(np.exp(lp), axis=1, keepdims=True))
            states, score = ctc_forced_align(lp, labels)
            ok = states is not None
            core = seg = None
            if ok:
                core, seg = spans_from_states(states, len(labels))
                ok = core is not None

            if ok:
                widx, pos, wtext = word_structure(labels, t["sentence"])
                probs = np.exp(lp)
                T = len(t["features"])
                half = rnn.kernel // 2          # centre offset of the stacked window
                clip = lambda b: int(min(max(b, 0), T))
                # centre mapping: step s -> bin 4s + 16, half-open on the right. Adjacent
                # seg spans are then exactly contiguous because seg_start[k+1] == seg_end[k]+1
                # in the step grid. (Using the receptive field 4s..4s+32 here would widen
                # every interval by 28 bins and make all neighbours overlap.)
                ctr0 = lambda s: clip(rnn.stride * s + half)
                ctr1 = lambda s: clip(rnn.stride * (s + 1) + half)
                for k, (s0, s1) in enumerate(core):
                    g0, g1 = seg[k]
                    rows.append((
                        session, partition, modality, donor, t["trial_idx"], t["sentence"],
                        k, PHONE_DEF_SIL[labels[k]], int(widx[k]), pos[k], wtext[k],
                        # receptive field of the occupied steps (what the RNN actually saw)
                        clip(rnn.stride * s0), clip(rnn.stride * s1 + rnn.kernel),
                        clip(rnn.stride * s1 + rnn.kernel) - clip(rnn.stride * s0),
                        s0, s1,
                        # tight centre window of the occupied core steps
                        ctr0(s0), ctr1(s1),
                        # gap-free, non-overlapping partition (blank runs split at midpoint)
                        ctr0(g0), ctr1(g1),
                        float(np.mean(probs[s0:s1 + 1, labels[k]])),
                    ))
            summary.append((session, partition, modality, donor, t["trial_idx"],
                            t["sentence"], len(t["features"]), S, len(labels), len(dec),
                            ed, per, bool(ok), score if ok else np.nan))
            if save_logits:
                logit_store[t["trial_idx"]] = out[i, :S].astype(np.float16)

    cols = ["session", "partition", "modality", "input_layer_from", "trial_idx",
            "sentence", "phone_idx", "phoneme", "word_idx", "position_in_word", "word",
            "start_bin", "end_bin", "n_bins", "rnn_start_step", "rnn_end_step",
            "rnn_center_start", "rnn_center_end", "seg_start_bin", "seg_end_bin", "score"]
    df = pd.DataFrame(rows, columns=cols).sort_values(["trial_idx", "phone_idx"])
    _assert_partition(df)
    scols = ["session", "partition", "modality", "input_layer_from", "trial_idx",
             "sentence", "n_bins_trial", "n_rnn_steps", "n_phonemes_true",
             "n_phonemes_decoded", "edit_distance", "per", "aligned", "align_logprob"]
    sdf = pd.DataFrame(summary, columns=scols).sort_values("trial_idx")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{session}_{partition}.csv", index=False)
    sdf.to_csv(OUT_DIR / f"{session}_{partition}_trials.csv", index=False)
    if save_logits:
        (OUT_DIR / "logits").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT_DIR / "logits" / f"{session}_{partition}_logits.npz",
                            **{f"trial_{k:04d}": v for k, v in logit_store.items()})

    per = sdf.edit_distance.sum() / max(1, sdf.n_phonemes_true.sum())
    dt = time.time() - t0
    log(f"  {session}/{partition} [{modality}] n={len(sdf)} PER={per:.4f} "
        f"aligned={int(sdf.aligned.sum())}/{len(sdf)} phones={len(df)} {dt:.1f}s", logfile)
    return {"session": session, "partition": partition, "modality": modality,
            "layer_idx": layer_idx, "n_trials": int(len(sdf)), "per": float(per),
            "n_aligned": int(sdf.aligned.sum()), "n_phones": int(len(df)),
            "seconds": round(dt, 1)}


def combine():
    tri = sorted(OUT_DIR.glob("*_trials.csv"))
    if not tri:
        print("nothing to combine")
        return
    sdf = pd.concat([pd.read_csv(f) for f in tri], ignore_index=True)
    sdf.to_csv(OUT_DIR / "trial_summary.csv", index=False)
    g = sdf.groupby(["session", "partition", "modality", "input_layer_from"],
                    as_index=False).agg(
        n_trials=("trial_idx", "count"), n_aligned=("aligned", "sum"),
        edit_distance=("edit_distance", "sum"), n_phonemes_true=("n_phonemes_true", "sum"))
    g["per"] = g.edit_distance / g.n_phonemes_true
    g["own_input_layer"] = g.session == g.input_layer_from
    # Data the RNN never saw in training, and therefore the only data valid for accuracy
    # or confusion analysis. The 19 training sessions' `train` partitions were memorised
    # (PER ~0.1%), but the 5 sessions absent from args.yaml were not trained on at all, so
    # BOTH of their partitions are held out.
    g["held_out"] = (g.partition == "test") | (~g.own_input_layer)
    g = g.sort_values(["partition", "session"])
    g.to_csv(OUT_DIR / "session_per.csv", index=False)
    print(g.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["tfrecords", "mat"], default="tfrecords")
    ap.add_argument("--sessions", nargs="+", default=["all-vocal"],
                    help="session ids, or all-vocal / all-nonvocal / all")
    ap.add_argument("--partitions", nargs="+", default=["test"])
    ap.add_argument("--input-layer-from", default=None,
                    help="borrow this vocal session's day-specific input layer")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-logits", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="resume: skip session+partition pairs whose CSVs already exist")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--logfile", default=str(OUT_DIR / "progress.log"))
    a = ap.parse_args()

    if a.combine:
        combine()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rnn = BaselineRNN()
    log(f"restored {rnn.ckpt_path} | input layers={rnn.n_input_layers} "
        f"| stack k={rnn.kernel} s={rnn.stride} | smooth={rnn.smooth} sd={rnn.smooth_sd}",
        a.logfile)

    if a.sessions == ["all-vocal"]:
        sessions = [s for s in ALL_SESSIONS if s not in NONVOCAL]
    elif a.sessions == ["all-trained"]:
        sessions = list(rnn.sessions)
    elif a.sessions == ["all-nonvocal"]:
        sessions = [s for s in ALL_SESSIONS if s in NONVOCAL]
    elif a.sessions == ["all"]:
        sessions = list(ALL_SESSIONS)
    else:
        sessions = a.sessions

    g2p = None
    if a.source == "mat":
        from g2p_en import G2p
        g2p = G2p()

    results = []
    for s in sessions:
        modality = "nonvocal" if s in NONVOCAL else "vocal"
        # sessions outside the 19 training sessions have no input layer of their own
        borrowed = a.input_layer_from or (None if s in rnn.sessions else DEFAULT_DONOR.get(s))
        donor = borrowed or s
        try:
            layer_idx = rnn.layer_for_session(donor)
        except KeyError as e:
            log(f"  !! {s}: {e}", a.logfile)
            continue
        if borrowed:
            log(f"{s} [{modality}]: borrowing input layer from {borrowed} "
                f"(layer {layer_idx}) -> alignments are APPROXIMATE", a.logfile)
        for p in a.partitions:
            if a.skip_existing and (OUT_DIR / f"{s}_{p}.csv").exists() \
                    and (OUT_DIR / f"{s}_{p}_trials.csv").exists():
                log(f"  {s}/{p}: already present, skipped", a.logfile)
                continue
            try:
                r = run_session(rnn, s, p, a.source, g2p, layer_idx, modality, donor,
                                batch_size=a.batch_size, logfile=a.logfile,
                                save_logits=not a.no_logits)
                if r:
                    r["source"] = a.source
                    results.append(r)
            except Exception as e:
                log(f"  !! {s}/{p} FAILED: {type(e).__name__}: {e}", a.logfile)
    if results:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
