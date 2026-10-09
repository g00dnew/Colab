"""t12_pipeline.py — T12 (Willett et al. 2023) 후두 자질(laryngeal feature) 분석 파이프라인.

연구 질문(core-question-v3.md): 영어의 후두 대립은 운동 계획에서 [voice]로 표상되는가,
[spread glottis]로 표상되는가.

  T1 위치   : 어두/어중/어말별 p/b 신경 분리도      (Stage 2, 정렬 필요)
  T2 유표성 : 어느 계열이 무표 기준선에서 더 먼가    (Stage 1 + Stage 2)
  T3 양식   : 무성 발화에서 후두 정보가 남는가      (Stage 2, vocal vs nonvocal)
  보조      : 유성성 쌍 vs 조음위치 쌍 분리도 비교, 어레이별(6v / 44) 분해

구조는 VocalMind 노트북(colab/notebook.ipynb)을 그대로 따른다:
경로 설정 → 데이터 확인 → 인벤토리 → QC → 음성 주석 → 분석 → 요약.
모든 단계는 RESULT_DIR에 캐시하고 재실행 시 재사용한다.

노트북(t12/colab/T12_laryngeal.ipynb)은 이 모듈을 import해서 쓴다.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "get_paths", "PHONE_DEF_SIL", "TUNING_CUE_ARPABET", "TUNING_CUE_TEXT",
    "SESSION_MODALITY", "NONVOCAL_SESSIONS", "ARRAYS", "array_columns",
    "LARYNGEAL_PAIRS", "PLACE_CONTROL_PAIRS",
    "check_archives", "extract_archive", "load_mat",
    "load_tuning_session", "load_competition_session",
    "tuning_inventory", "competition_inventory", "competition_qc",
    "build_phonetic_feature_table",
    "stage1_trial_features", "balanced_pair_decode", "multiclass_decode",
    "estimate_audio_onsets", "ONSET_WINDOWS", "ALL_STAGE1_WINDOWS",
    "pair_manner", "manner_aggregate",
    "stage1_run", "cv_markedness", "stage1_markedness",
    "make_synthetic_alignments", "load_alignments", "apply_score_filter",
    "WINDOW_MODES", "session_per_table", "annotate_context",
    "position_contrast_interaction", "contrast_gap_ci", "VOWELS", "FLAP_PAIR",
    "tuning_qc", "tuning_cue_counts", "competition_session_files",
    "feature_map", "session_modality", "blockwise_zscore",
    "stage1_phoneme39", "markedness_contrast_test",
    "fig_markedness", "fig_feature_transmission", "STAGE2_ARRAYS",
    "stage2_segment_features", "stage2_synthetic_features",
    "stage2_position_decode", "stage2_modality_decode",
    "segment_group_ids", "cap_tokens_per_word", "select_pair_segments",
    "t3_pair_drops", "t3_natural_class_test",
    "stage2_markedness_by_position", "relative_transmitted_information",
    "stage2_run", "write_summary",
    "fig_stage1_pairs", "fig_stage2_position",
]

# ===========================================================================
# 1. 경로
# ===========================================================================

LOCAL_ROOT = Path("/Users/hojaechoi/Documents/research/speech-bci-confusion/t12")
COLAB_ROOT = Path("/content/drive/MyDrive/T12_Project")


def get_paths(local: bool = True, root=None, make_dirs: bool = True) -> dict:
    """LOCAL=True면 t12/ 하위 경로, False면 Drive의 T12_Project 하위 경로."""
    if root is not None:
        root = Path(root)
    else:
        root = LOCAL_ROOT if local else COLAB_ROOT

    if local:
        p = {
            "ROOT": root,
            "ARCHIVE_DIR": root / "raw",
            "DATA_DIR": root / "data",
            "ANNOT_DIR": root / "annotations",
            "RESULT_DIR": root / "results",
            "CODE_DIR": root / "code",
        }
    else:
        p = {
            "ROOT": root,
            "ARCHIVE_DIR": root / "01_data",
            "DATA_DIR": root / "01_data",
            "ANNOT_DIR": root / "02_annotations",
            "RESULT_DIR": root / "04_results",
            "CODE_DIR": root / "03_code",
        }

    p["LOCAL"] = local
    p["FIG_DIR"] = p["RESULT_DIR"] / "figures"
    p["TUNING_DIR"] = p["DATA_DIR"] / "tuningTasks"
    p["COMP_DIR"] = p["DATA_DIR"] / "competitionData"
    p["ALIGN_DIR"] = p["RESULT_DIR"] / "alignments"
    p["CACHE_DIR"] = p["RESULT_DIR"] / "cache"

    if make_dirs:
        for key in ("DATA_DIR", "ANNOT_DIR", "RESULT_DIR", "FIG_DIR",
                    "ALIGN_DIR", "CACHE_DIR"):
            p[key].mkdir(parents=True, exist_ok=True)
    return p


# ===========================================================================
# 2. 상수 — 모두 저장소 코드/readme에서 실측 확인한 값
# ===========================================================================

# NeuralDecoder/neuralDecoder/datasets/speechDataset.py
PHONE_DEF = [
    "AA", "AE", "AH", "AO", "AW",
    "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G",
    "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V",
    "W", "Y", "Z", "ZH",
]
PHONE_DEF_SIL = PHONE_DEF + ["SIL"]

# tuningTasks_readme.txt: phonemes 실험의 trialCues 순서(1-indexed).
TUNING_CUE_ARPABET = [
    "B", "CH", "NOTHING", "D", "F", "G", "HH", "JH", "K", "L",
    "ER", "M", "N", "NG", "P", "R", "S", "SH", "DH", "T",
    "TH", "V", "W", "Y", "Z", "ZH", "OY", "EH", "EY", "UH",
    "IY", "OW", "UW", "IH", "AA", "AW", "AY", "AH", "AO", "AE",
]

# .mat의 cueList에 실제로 들어 있는 철자(2026-10-09 실측, 04.21/04.26 동일).
TUNING_CUE_TEXT = [
    "Bah", "Chah", "DO_NOTHING", "Dah", "Fah", "Gah", "Hah", "Jah", "Kah", "Lah",
    "LettER", "Mah", "Nah", "Ngah", "Pah", "Rah", "Sah", "Shah", "THe", "Tah",
    "Thah", "Vah", "Wah", "Yah", "Zah", "Zhah", "chOIce", "drEss", "fAce", "fOOt",
    "flEEce", "gOAt", "gOOse", "kIt", "lOt", "mOUth", "prIce", "strUt", "thOUGHt", "trAp",
]

TUNING_PHONEME_SESSIONS = ["t12.2022.04.21_phonemes", "t12.2022.04.26_phonemes"]

# AnalysisExamples/getSpeechSessionBlocks.py
NONVOCAL_SESSIONS = {
    "t12.2022.06.23", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
}
_ALL_SPEECH_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]
SESSION_MODALITY = {
    s: ("nonvocal" if s in NONVOCAL_SESSIONS else "vocal")
    for s in _ALL_SPEECH_SESSIONS
}
# tuningTasks 날짜는 getSpeechSessionBlocks(문장 세션 목록)에 없다.
# 근거: tuningTasks_readme가 04.26/05.03에 마이크 녹음이 있었다고 적고 있고,
# 저장소에 이 날짜의 nonvocal 조건이 없다. 즉 고립 음소 과제는 발성 조건뿐이며
# 양식 비교(T3)는 문장 데이터로만 가능하다.
_TUNING_DATES = ["t12.2022.04.21", "t12.2022.04.26", "t12.2022.05.03"]
for _d in _TUNING_DATES:
    SESSION_MODALITY.setdefault(_d, "vocal")


def session_modality(session: str) -> str:
    """세션 이름(접미사 무관)에서 vocal/nonvocal 라벨을 찾는다."""
    key = str(session)
    m = re.match(r"(t12\.\d{4}\.\d{2}\.\d{2})", key)
    if m:
        key = m.group(1)
    return SESSION_MODALITY.get(key, "unknown")


# 저자 readme의 어레이 배치: 0~127 = area 6v, 128~255 = area 44.
N_CHAN = 256
FEATURE_SETS = ("tx1", "spikePow")

# readme의 어레이 지도(8x8 블록 4개)를 그대로 옮긴 채널 집합.
#   Area 6v Superior = 032~095, Area 6v Inferior = 000~031 + 096~127
#   Area 44 Superior = 160~223, Area 44 Inferior = 128~159 + 224~255
# H_sampling(후두 정보가 dorsal 쪽에 있다)의 실제 시험은 6v_sup vs 6v_inf 이다.
_C6V_SUP = np.arange(32, 96)
_C6V_INF = np.concatenate([np.arange(0, 32), np.arange(96, 128)])
_C44_SUP = np.arange(160, 224)
_C44_INF = np.concatenate([np.arange(128, 160), np.arange(224, 256)])
ARRAYS = {
    "6v": np.arange(0, 128),
    "44": np.arange(128, 256),
    "all": np.arange(0, 256),
    "6v_sup": _C6V_SUP,
    "6v_inf": _C6V_INF,
    "44_sup": _C44_SUP,
    "44_inf": _C44_INF,
}
for _name, _ch in ARRAYS.items():
    assert np.unique(_ch).size == _ch.size, f"{_name}: 채널 중복"
    assert _ch.min() >= 0 and _ch.max() < N_CHAN, f"{_name}: 채널 범위 오류"
assert np.array_equal(np.sort(np.concatenate([_C6V_SUP, _C6V_INF])), np.arange(0, 128))
assert np.array_equal(np.sort(np.concatenate([_C44_SUP, _C44_INF])), np.arange(128, 256))


def array_columns(array: str, feature_sets=FEATURE_SETS, n_chan: int = N_CHAN) -> np.ndarray:
    """concat(tx1[:, :256], spikePow[:, :256]) 특징벡터에서 어레이에 해당하는 열 index."""
    chans = ARRAYS[array]
    return np.concatenate([chans + i * n_chan for i, _ in enumerate(feature_sets)])


# 후두(유성성) 최소대립쌍: (무성, 유성)
LARYNGEAL_PAIRS = [
    ("P", "B"), ("T", "D"), ("K", "G"),
    ("F", "V"), ("S", "Z"), ("SH", "ZH"),
    ("CH", "JH"), ("TH", "DH"),
]
# 조음위치 통제쌍(유성성은 같고 위치만 다름) = 기준 눈금
PLACE_CONTROL_PAIRS = [
    ("P", "T"), ("T", "K"), ("B", "D"), ("D", "G"),
    ("F", "S"), ("F", "TH"), ("S", "SH"), ("V", "Z"), ("M", "N"),
]

ARCHIVES = {
    "tuningTasks.tar.gz": {"approx_gb": 0.9, "member": "tuningTasks"},
    "competitionData.tar.gz": {"approx_gb": 3.7, "member": "competitionData"},
    "derived.tar.gz": {"approx_gb": 7.5, "member": "derived"},
}
DRYAD_URL = "https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq"


# ===========================================================================
# 3. 데이터 확인 / 압축 해제
# ===========================================================================

def check_archives(paths: dict, names=None, verify_gzip: bool = True) -> pd.DataFrame:
    """tar.gz 존재 여부와 gzip 무결성을 확인한다.

    Dryad는 봇을 브라우저 챌린지로 막으므로 이 단계에서 내려받지 않는다.
    없으면 브라우저 수동 다운로드 안내를 출력한다.
    """
    import gzip as _gzip

    names = list(names or ARCHIVES)
    rows = []
    for name in names:
        path = paths["ARCHIVE_DIR"] / name
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        gzip_ok = None
        if exists and verify_gzip:
            try:
                with _gzip.open(path, "rb") as fh:
                    while fh.read(8 * 1024 * 1024):
                        pass
                gzip_ok = True
            except Exception as exc:  # noqa: BLE001
                gzip_ok = False
                print(f"  gzip 손상: {name} -> {exc}")
        rows.append({
            "archive": name,
            "exists": exists,
            "size_gb": round(size / 1e9, 3),
            "expected_gb": ARCHIVES.get(name, {}).get("approx_gb"),
            "gzip_ok": gzip_ok,
        })

    df = pd.DataFrame(rows)
    missing = df.loc[~df["exists"], "archive"].tolist()
    broken = df.loc[df["gzip_ok"] == False, "archive"].tolist()  # noqa: E712

    if missing or broken:
        print("\n[수동 다운로드 필요]")
        print(f"  1) 브라우저로 {DRYAD_URL} 접속")
        print(f"  2) 아래 파일을 {paths['ARCHIVE_DIR']} 에 저장")
        for name in missing + broken:
            print(f"       - {name}")
        print("  (Dryad는 자동 다운로드를 차단하므로 requests/curl로는 받을 수 없다)")
    else:
        print(f"아카이브 {len(df)}개 확인 완료 (gzip 무결성 통과)")
    return df


def extract_archive(paths: dict, name: str, force: bool = False) -> Path:
    """tar.gz를 DATA_DIR에 풀고 .EXTRACTED 마커를 남긴다. 멤버 이름을 필터링한다."""
    archive = paths["ARCHIVE_DIR"] / name
    member = ARCHIVES[name]["member"]
    target = paths["DATA_DIR"] / member
    marker = target / ".EXTRACTED"

    if marker.exists() and not force:
        n = len(list(target.rglob("*.mat")))
        print(f"{name}: 기존 압축 해제 결과 재사용 ({n}개 .mat)")
        return target
    if not archive.exists():
        raise FileNotFoundError(f"{archive} 없음. check_archives 안내를 따르세요.")

    print(f"{name} 압축 해제 중 -> {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        def _keep(ti: tarfile.TarInfo):
            clean = ti.name.lstrip("./")
            if not clean.startswith(member):
                return False
            return ti.isdir() or clean.endswith((".mat", ".txt", ".csv", ".json"))
        members = [ti for ti in tf if _keep(ti)]
        if not members:
            raise RuntimeError(f"{name}: '{member}' 로 시작하는 멤버가 없습니다.")
        try:
            tf.extractall(paths["DATA_DIR"], members=members, filter="data")
        except TypeError:  # python < 3.12
            tf.extractall(paths["DATA_DIR"], members=members)  # noqa: S202

    marker.write_text("ok\n", encoding="utf-8")
    print(f"  완료: {len(list(target.rglob('*.mat')))}개 .mat")
    return target


def load_mat(path, variable_names=None) -> dict:
    """MATLAB v5는 scipy.io.loadmat, v7.3이면 h5py로 대체한다."""
    import scipy.io

    path = Path(path)
    try:
        return scipy.io.loadmat(str(path), variable_names=variable_names,
                                squeeze_me=False, struct_as_record=False)
    except NotImplementedError:
        import h5py

        out = {"__loader__": "h5py"}
        with h5py.File(path, "r") as fh:
            for key in fh:
                if key.startswith("#"):
                    continue
                if variable_names and key not in variable_names:
                    continue
                out[key] = np.array(fh[key]).T
        return out


def _mat_strings(obj) -> list:
    """cueList 같은 object 배열을 문자열 리스트로 바꾼다."""
    out = []
    for item in np.ravel(np.asarray(obj, dtype=object)):
        arr = np.squeeze(np.asarray(item))
        out.append(str(arr.item()) if arr.ndim == 0 else "".join(map(str, arr.ravel())))
    return out


def _char_matrix_to_list(obj) -> list:
    """competitionData의 sentenceText (S x C char matrix) -> 문자열 리스트."""
    arr = np.asarray(obj)
    if arr.dtype == object:
        return [s.strip() for s in _mat_strings(arr)]
    if arr.dtype.kind in "US":
        if arr.ndim == 1:
            return [str(x).strip() for x in arr]
        return ["".join(row.astype(str)).strip() for row in arr]
    return ["".join(map(chr, row[row > 0])).strip() for row in arr.astype(int)]


# ===========================================================================
# 4. 세션 로딩
# ===========================================================================

_TUNING_VARS = ["tx1", "spikePow", "trialState", "blockNum", "trialCues",
                "cueList", "goTrialEpochs", "delayTrialEpochs", "blockList"]


def load_tuning_session(path, variable_names=_TUNING_VARS) -> dict:
    """tuningTasks 세션 하나를 읽고 cue 라벨을 ARPABET으로 변환한다.

    실측(2026-10-09): trialCues는 readme의 'N x 2'가 아니라 N x 1.
    trialState는 {0=delay, 1=go}만 등장(phonemes 과제에 return 구간 없음).
    """
    path = Path(path)
    mat = load_mat(path, variable_names=variable_names)
    out = {"session": path.stem, "path": str(path)}

    out["tx1"] = np.asarray(mat["tx1"])
    out["spikePow"] = np.asarray(mat["spikePow"])
    out["trialState"] = np.asarray(mat["trialState"]).ravel()
    out["blockNum"] = np.asarray(mat["blockNum"]).ravel()
    out["goTrialEpochs"] = np.asarray(mat["goTrialEpochs"]).astype(np.int64)
    out["delayTrialEpochs"] = np.asarray(mat["delayTrialEpochs"]).astype(np.int64)

    cues = np.asarray(mat["trialCues"]).astype(np.int64)
    out["trialCues"] = cues[:, 0] if cues.ndim == 2 else cues.ravel()
    out["cueList"] = _mat_strings(mat["cueList"])

    n_trials = out["goTrialEpochs"].shape[0]
    assert out["trialCues"].shape[0] == n_trials, "trialCues / goTrialEpochs 길이 불일치"
    assert out["tx1"].shape == out["spikePow"].shape, "tx1 / spikePow 모양 불일치"
    assert out["tx1"].shape[1] == N_CHAN, f"채널 수 {out['tx1'].shape[1]} != 256"

    # 추가로 요청한 선택 필드(audioEnvelope 등)를 그대로 넘긴다.
    for extra in ("audioEnvelope", "trialDelayTimes", "blockList"):
        if extra in mat:
            out[extra] = np.asarray(mat[extra])

    out["is_phoneme_task"] = "_phonemes" in path.stem
    if out["is_phoneme_task"]:
        cue_list = out["cueList"]
        if cue_list != TUNING_CUE_TEXT:
            warnings.warn(
                f"{path.stem}: cueList가 실측값과 다릅니다. 위치 기반 매핑을 그대로 쓰되 "
                f"반드시 확인하세요. 관측: {cue_list[:6]}...", stacklevel=2)
        assert len(cue_list) == len(TUNING_CUE_ARPABET), (
            f"cueList 길이 {len(cue_list)} != 40. ARPABET 매핑을 재확인하세요.")
        labels = np.array([TUNING_CUE_ARPABET[i - 1] for i in out["trialCues"]], dtype="<U8")
        out["phoneme"] = labels
        out["cue_text"] = np.array([cue_list[i - 1] for i in out["trialCues"]], dtype=object)
    return out


_COMP_VARS = ["sentenceText", "tx1", "spikePow", "blockIdx"]


def load_competition_session(path, variable_names=_COMP_VARS) -> dict:
    """competitionData 세션 하나 (.mat). tx1/spikePow는 S x 1 cell of T x 256."""
    path = Path(path)
    mat = load_mat(path, variable_names=variable_names)
    out = {"file": path.stem, "path": str(path)}

    def _cells(key):
        raw = np.asarray(mat[key], dtype=object).ravel()
        return [np.asarray(x) for x in raw]

    out["tx1"] = _cells("tx1")
    out["spikePow"] = _cells("spikePow")
    out["blockIdx"] = np.asarray(mat["blockIdx"]).astype(np.int64).ravel()
    out["sentenceText"] = (_char_matrix_to_list(mat["sentenceText"])
                           if "sentenceText" in mat else [])

    n = len(out["tx1"])
    assert len(out["spikePow"]) == n, "tx1 / spikePow 시행 수 불일치"
    assert out["blockIdx"].shape[0] == n, "blockIdx 길이 불일치"
    for a, b in zip(out["tx1"][:5], out["spikePow"][:5]):
        assert a.shape == b.shape, f"시행 내 모양 불일치 {a.shape} vs {b.shape}"
        assert a.shape[1] == N_CHAN, f"채널 수 {a.shape[1]} != 256"
    out["n_trials"] = n
    out["session"] = path.stem
    out["modality"] = session_modality(path.stem)
    return out


def competition_session_files(paths: dict) -> pd.DataFrame:
    """competitionData/{train,test,competitionHoldOut}/*.mat 목록."""
    rows = []
    for part in ("train", "test", "competitionHoldOut"):
        for f in sorted((paths["COMP_DIR"] / part).glob("*.mat")):
            rows.append({"partition": part, "session": f.stem, "path": str(f)})
    return pd.DataFrame(rows, columns=["partition", "session", "path"])


# ===========================================================================
# 5. 인벤토리 (원 노트북 cell 4에 대응)
# ===========================================================================

def competition_inventory(paths: dict, force: bool = False) -> pd.DataFrame:
    out_path = paths["RESULT_DIR"] / "competition_inventory.csv"
    if out_path.exists() and not force:
        print("competition 인벤토리: 기존 결과 재사용")
        return pd.read_csv(out_path)

    files = competition_session_files(paths)
    if files.empty:
        print(f"competitionData가 없습니다 ({paths['COMP_DIR']}). 인벤토리 생략.")
        return pd.DataFrame()

    rows = []
    for rec in files.to_dict("records"):
        d = load_competition_session(rec["path"])
        lens = np.array([x.shape[0] for x in d["tx1"]], dtype=np.int64)
        rows.append({
            "session": d["session"],
            "partition": rec["partition"],
            "modality": d["modality"],
            "n_trials": d["n_trials"],
            "n_blocks": int(np.unique(d["blockIdx"]).size),
            "blocks": ";".join(map(str, np.unique(d["blockIdx"]).tolist())),
            "total_bins": int(lens.sum()),
            "total_sec": round(float(lens.sum()) * 0.02, 1),
            "bins_mean": round(float(lens.mean()), 1),
            "bins_min": int(lens.min()),
            "bins_max": int(lens.max()),
            "n_sentences_text": len(d["sentenceText"]),
        })
    df = pd.DataFrame(rows).sort_values(["partition", "session"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"competition 인벤토리 저장: {out_path} ({len(df)}행)")
    return df


def tuning_inventory(paths: dict, force: bool = False) -> pd.DataFrame:
    out_path = paths["RESULT_DIR"] / "tuning_inventory.csv"
    if out_path.exists() and not force:
        print("tuningTasks 인벤토리: 기존 결과 재사용")
        return pd.read_csv(out_path)

    files = sorted(paths["TUNING_DIR"].glob("*.mat"))
    if not files:
        print(f"tuningTasks가 없습니다 ({paths['TUNING_DIR']}).")
        return pd.DataFrame()

    rows = []
    for f in files:
        d = load_tuning_session(f)
        go = d["goTrialEpochs"]
        dur = go[:, 1] - go[:, 0]
        task = f.stem.split("_", 1)[1] if "_" in f.stem else "unknown"
        base = {
            "file": f.stem,
            "task": task,
            "modality": session_modality(f.stem),
            "n_time_bins": int(d["tx1"].shape[0]),
            "total_sec": round(d["tx1"].shape[0] * 0.02, 1),
            "n_trials": int(go.shape[0]),
            "n_blocks": int(np.unique(d["blockNum"]).size),
            "n_cues": len(d["cueList"]),
            "go_bins_mean": round(float(dur.mean()), 1),
            "go_bins_min": int(dur.min()),
            "go_bins_max": int(dur.max()),
            "trial_states": ";".join(map(str, np.unique(d["trialState"]).tolist())),
        }
        cue_idx, counts = np.unique(d["trialCues"], return_counts=True)
        base["trials_per_cue_min"] = int(counts.min())
        base["trials_per_cue_max"] = int(counts.max())
        base["cues"] = ";".join(d["cueList"])
        if d["is_phoneme_task"]:
            base["phonemes"] = ";".join(sorted(set(d["phoneme"].tolist())))
        rows.append(base)
        del d
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"tuningTasks 인벤토리 저장: {out_path} ({len(df)}행)")
    return df


def tuning_cue_counts(paths: dict, force: bool = False) -> pd.DataFrame:
    """phonemes 세션의 cue별 시행 수 (ARPABET 라벨 포함)."""
    out_path = paths["RESULT_DIR"] / "tuning_cue_counts.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)
    rows = []
    for name in TUNING_PHONEME_SESSIONS:
        f = paths["TUNING_DIR"] / f"{name}.mat"
        if not f.exists():
            continue
        d = load_tuning_session(f, variable_names=["trialCues", "cueList",
                                                   "goTrialEpochs", "blockNum",
                                                   "tx1", "spikePow", "trialState",
                                                   "delayTrialEpochs"])
        idx, counts = np.unique(d["trialCues"], return_counts=True)
        for i, c in zip(idx, counts):
            rows.append({"session": name, "cue_index": int(i),
                         "cue_text": d["cueList"][i - 1],
                         "phoneme": TUNING_CUE_ARPABET[i - 1],
                         "n_trials": int(c)})
        del d
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_path, index=False)
    return df


# ===========================================================================
# 6. QC (원 노트북 cell 5에 대응)
# ===========================================================================

def blockwise_zscore_stats(X: np.ndarray, block_ids: np.ndarray, eps: float = 1e-8):
    """블록별 채널 평균/표준편차. {block: (mu, sd)}."""
    stats = {}
    for b in np.unique(block_ids):
        sub = X[block_ids == b].astype(np.float64)
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0)
        sd[sd < eps] = 1.0
        stats[int(b)] = (mu, sd)
    return stats


def blockwise_zscore(X: np.ndarray, block_ids: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """저자 권고(competitionData readme Hints)대로 블록 내 z-score."""
    Z = np.empty(X.shape, dtype=np.float32)
    for b, (mu, sd) in blockwise_zscore_stats(X, block_ids, eps).items():
        m = block_ids == b
        Z[m] = ((X[m].astype(np.float64) - mu) / sd).astype(np.float32)
    return Z


def competition_qc(paths: dict, force: bool = False, max_sessions=None) -> pd.DataFrame:
    """세션별 모양, NaN/Inf, z-score 전후 채널 평균/표준편차, 시행 길이 분포."""
    out_path = paths["RESULT_DIR"] / "competition_qc.csv"
    if out_path.exists() and not force:
        print("competition QC: 기존 결과 재사용")
        return pd.read_csv(out_path)

    files = competition_session_files(paths)
    if files.empty:
        print(f"competitionData가 없습니다 ({paths['COMP_DIR']}). QC 생략.")
        return pd.DataFrame()
    recs = files.to_dict("records")
    if max_sessions:
        recs = recs[:max_sessions]

    rows = []
    for rec in recs:
        d = load_competition_session(rec["path"])
        lens = np.array([x.shape[0] for x in d["tx1"]], dtype=np.int64)
        for fs in FEATURE_SETS:
            big = np.concatenate([a.astype(np.float64) for a in d[fs]], axis=0)
            bins_block = np.repeat(d["blockIdx"], lens)
            n_nan = int(np.isnan(big).sum())
            n_inf = int(np.isinf(big).sum())
            Z = blockwise_zscore(big, bins_block)
            rows.append({
                "session": d["session"], "partition": rec["partition"],
                "modality": d["modality"], "feature": fs,
                "n_trials": d["n_trials"], "n_bins": int(big.shape[0]),
                "n_chan": int(big.shape[1]),
                "n_nan": n_nan, "n_inf": n_inf,
                "raw_chan_mean_med": round(float(np.median(big.mean(0))), 4),
                "raw_chan_std_med": round(float(np.median(big.std(0))), 4),
                "raw_chan_std_min": round(float(big.std(0).min()), 6),
                "raw_chan_std_max": round(float(big.std(0).max()), 4),
                "z_chan_mean_absmax": round(float(np.abs(Z.mean(0)).max()), 4),
                "z_chan_std_med": round(float(np.median(Z.std(0))), 4),
                "n_dead_chan": int((big.std(0) < 1e-8).sum()),
                "trial_bins_mean": round(float(lens.mean()), 1),
                "trial_bins_p05": int(np.percentile(lens, 5)),
                "trial_bins_p95": int(np.percentile(lens, 95)),
            })
            del big, Z
        del d
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"competition QC 저장: {out_path} ({len(df)}행)")
    return df


def tuning_qc(paths: dict, force: bool = False) -> pd.DataFrame:
    """tuningTasks phonemes 세션 QC."""
    out_path = paths["RESULT_DIR"] / "tuning_qc.csv"
    if out_path.exists() and not force:
        print("tuningTasks QC: 기존 결과 재사용")
        return pd.read_csv(out_path)

    rows = []
    for name in TUNING_PHONEME_SESSIONS:
        f = paths["TUNING_DIR"] / f"{name}.mat"
        if not f.exists():
            continue
        d = load_tuning_session(f)
        go = d["goTrialEpochs"]
        dur = go[:, 1] - go[:, 0]
        for fs in FEATURE_SETS:
            X = d[fs]
            Xf = X.astype(np.float64)
            Z = blockwise_zscore(X, d["blockNum"])
            rows.append({
                "session": name, "feature": fs,
                "n_bins": int(X.shape[0]), "n_chan": int(X.shape[1]),
                "dtype": str(X.dtype),
                "n_nan": int(np.isnan(Xf).sum()), "n_inf": int(np.isinf(Xf).sum()),
                "raw_chan_mean_med": round(float(np.median(Xf.mean(0))), 4),
                "raw_chan_std_med": round(float(np.median(Xf.std(0))), 4),
                "raw_chan_std_min": round(float(Xf.std(0).min()), 6),
                "raw_chan_std_max": round(float(Xf.std(0).max()), 4),
                "z_chan_mean_absmax": round(float(np.abs(Z.mean(0)).max()), 4),
                "z_chan_std_med": round(float(np.median(Z.std(0))), 4),
                "n_dead_chan": int((Xf.std(0) < 1e-8).sum()),
                "n_trials": int(go.shape[0]), "n_blocks": int(np.unique(d["blockNum"]).size),
                "go_bins_mean": round(float(dur.mean()), 1),
                "go_bins_min": int(dur.min()), "go_bins_max": int(dur.max()),
            })
            del Xf, Z
        del d
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_path, index=False)
        print(f"tuningTasks QC 저장: {out_path} ({len(df)}행)")
    return df


# ===========================================================================
# 7. 음성 자질 주석 (원 노트북 cell 11/12에 대응)
# ===========================================================================
# verified=False, 사전등록 전 확정 금지.
# voicing/place/manner/nasal은 표준 영어 ARPABET 기술이고,
# spread_glottis(기식 계열)는 laryngeal realism 쪽 가설 라벨이므로
# 여자친구(음성학 담당)의 사전등록 확정이 반드시 필요하다.

_CONSONANT_SPEC = [
    # arpabet, ipa, voicing, place, manner, nasal, sonorant, continuant, spread_glottis
    ("P",  "p",  0, "bilabial",     "stop",        0, 0, 0, 1),
    ("B",  "b",  1, "bilabial",     "stop",        0, 0, 0, 0),
    ("T",  "t",  0, "alveolar",     "stop",        0, 0, 0, 1),
    ("D",  "d",  1, "alveolar",     "stop",        0, 0, 0, 0),
    ("K",  "k",  0, "velar",        "stop",        0, 0, 0, 1),
    ("G",  "g",  1, "velar",        "stop",        0, 0, 0, 0),
    ("CH", "tS", 0, "postalveolar", "affricate",   0, 0, 0, 1),
    ("JH", "dZ", 1, "postalveolar", "affricate",   0, 0, 0, 0),
    ("F",  "f",  0, "labiodental",  "fricative",   0, 0, 1, 0),
    ("V",  "v",  1, "labiodental",  "fricative",   0, 0, 1, 0),
    ("TH", "T",  0, "dental",       "fricative",   0, 0, 1, 0),
    ("DH", "D",  1, "dental",       "fricative",   0, 0, 1, 0),
    ("S",  "s",  0, "alveolar",     "fricative",   0, 0, 1, 0),
    ("Z",  "z",  1, "alveolar",     "fricative",   0, 0, 1, 0),
    ("SH", "S",  0, "postalveolar", "fricative",   0, 0, 1, 0),
    ("ZH", "Z",  1, "postalveolar", "fricative",   0, 0, 1, 0),
    ("HH", "h",  0, "glottal",      "fricative",   0, 0, 1, 1),
    ("M",  "m",  1, "bilabial",     "nasal",       1, 1, 0, 0),
    ("N",  "n",  1, "alveolar",     "nasal",       1, 1, 0, 0),
    ("NG", "N",  1, "velar",        "nasal",       1, 1, 0, 0),
    ("L",  "l",  1, "alveolar",     "lateral",     0, 1, 1, 0),
    ("R",  "r",  1, "postalveolar", "approximant", 0, 1, 1, 0),
    ("W",  "w",  1, "labiovelar",   "glide",       0, 1, 1, 0),
    ("Y",  "j",  1, "palatal",      "glide",       0, 1, 1, 0),
]

CONSONANTS = [row[0] for row in _CONSONANT_SPEC]
# speechDataset.py의 VOWEL_DEF
VOWELS = ["EY", "AE", "AY", "EH", "AA", "AW", "IY", "IH", "OY", "OW",
          "AO", "UH", "AH", "UW", "ER"]
# 미국영어 설탄음화(flapping): 모음 사이의 T/D는 둘 다 설탄음으로 실현되므로
# 어중 T/D의 분리도가 낮은 것은 후두 이론과 무관하다. T1에서 반드시 통제한다.
FLAP_PAIR = ("T", "D")
# 후두 대립을 갖는 자음(최소대립쌍 구성원)만 모은 집합 = 유표성 기준선 후보
PAIRED_OBSTRUENTS = sorted({p for pair in LARYNGEAL_PAIRS for p in pair})


def build_phonetic_feature_table(paths: dict, force: bool = False) -> pd.DataFrame:
    """ARPABET 자음 x 자질표 + 최소대립쌍 라벨을 02_annotations에 저장한다."""
    out_path = paths["ANNOT_DIR"] / "phonetic_features_v1.csv"
    pair_path = paths["ANNOT_DIR"] / "minimal_pairs_v1.csv"
    if out_path.exists() and pair_path.exists() and not force:
        print("음성 자질표: 기존 결과 재사용")
        return pd.read_csv(out_path)

    cols = ["arpabet", "ipa", "voicing", "place", "manner", "nasal",
            "sonorant", "continuant", "spread_glottis"]
    df = pd.DataFrame(_CONSONANT_SPEC, columns=cols)

    lar_map = {}
    for voiceless, voiced in LARYNGEAL_PAIRS:
        lar_map[voiceless] = f"{voiceless}/{voiced}"
        lar_map[voiced] = f"{voiceless}/{voiced}"
    df["laryngeal_pair"] = df["arpabet"].map(lar_map).fillna("")
    df["in_laryngeal_pair"] = df["laryngeal_pair"] != ""
    df["series"] = np.where(df["in_laryngeal_pair"],
                            np.where(df["voicing"] == 1, "voiced", "voiceless"), "")
    df["verified"] = False
    df["note"] = ""
    df.loc[df["arpabet"] == "HH", "note"] = (
        "glottal fricative, 최소대립쌍 없음. spread_glottis=1은 가설 라벨")
    df.loc[df["arpabet"].isin(["P", "T", "K", "CH"]), "note"] = (
        "영어 무성 계열. [spread glottis] 이론의 유표항 후보")
    df.loc[df["arpabet"].isin(["B", "D", "G", "JH"]), "note"] = (
        "영어 유성 계열. [voice] 이론의 유표항 후보")
    df.to_csv(out_path, index=False)

    pair_rows = []
    for a, b in LARYNGEAL_PAIRS:
        pair_rows.append({"pair": f"{a}/{b}", "contrast": "laryngeal",
                          "member_a": a, "member_b": b,
                          "a_series": "voiceless", "b_series": "voiced",
                          "verified": False})
    for a, b in PLACE_CONTROL_PAIRS:
        pair_rows.append({"pair": f"{a}/{b}", "contrast": "place",
                          "member_a": a, "member_b": b,
                          "a_series": "", "b_series": "", "verified": False})
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(pair_path, index=False)

    print(f"음성 자질표 저장: {out_path} ({len(df)}자음)")
    print(f"최소대립쌍 저장: {pair_path} "
          f"(후두 {len(LARYNGEAL_PAIRS)}쌍, 조음위치 통제 {len(PLACE_CONTROL_PAIRS)}쌍)")
    print("  주의: verified=False. 사전등록 전 확정 금지.")
    return df


def pair_manner(a: str, b: str) -> str:
    """쌍의 조음방법. 두 구성원이 같으면 그 값, 다르면 'mixed'."""
    m = feature_map("manner")
    ma, mb = m.get(a), m.get(b)
    if ma is None or mb is None:
        return "unknown"
    return ma if ma == mb else "mixed"


def manner_aggregate(df: pd.DataFrame, group_cols=("window", "array"),
                     value="acc") -> pd.DataFrame:
    """조음방법별(stop/affricate/fricative) 평균 — 지속시간 교란 점검용.

    session_set / partitions 같은 구분 열이 있으면 자동으로 묶는 축에 넣는다
    (서로 다른 세션 집합의 행이 한 평균에 섞이지 않게).
    """
    if df.empty or "manner" not in df.columns:
        return pd.DataFrame()
    sub = df[df["contrast"] == "laryngeal"]
    extra = [c for c in ("session_set", "partitions", "position")
             if c in sub.columns and c not in group_cols]
    cols = [c for c in list(group_cols) + extra if c in sub.columns]
    out = (sub.groupby(cols + ["manner"])[value]
           .agg(mean="mean", sd="std", n_pairs="size").reset_index())
    allrow = (sub.groupby(cols)[value]
              .agg(mean="mean", sd="std", n_pairs="size").reset_index())
    allrow["manner"] = "ALL_laryngeal"
    return pd.concat([out, allrow], ignore_index=True)


def feature_map(feature: str) -> dict:
    """자질 이름 -> {arpabet: 값} (Miller-Nicely 전달량 계산용)."""
    df = pd.DataFrame(_CONSONANT_SPEC,
                      columns=["arpabet", "ipa", "voicing", "place", "manner",
                               "nasal", "sonorant", "continuant", "spread_glottis"])
    return dict(zip(df["arpabet"], df[feature]))


# ===========================================================================
# 8. 분류기 (균형 + 교차검증 + 순열검정)
# ===========================================================================

def _make_clf(kind: str):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "lda":
        return make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l2", C=1.0, max_iter=2000))
    raise ValueError(kind)


def _dprime(tp, fn, fp, tn) -> float:
    """pooled confusion -> d'. 0/1 비율은 loglinear 보정."""
    from scipy.stats import norm

    n_pos, n_neg = tp + fn, fp + tn
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    hit = (tp + 0.5) / (n_pos + 1.0)
    fa = (fp + 0.5) / (n_neg + 1.0)
    return float(norm.ppf(hit) - norm.ppf(fa))


def _balance(y: np.ndarray, rng: np.random.Generator, max_per_class=None):
    """클래스별 시행 수를 최소 클래스에 맞춰 무작위 하향 표집.

    max_per_class를 주면 그 값으로 상한도 둔다. 조건 간 n을 비교 가능하게
    유지하고 실행 시간을 예측 가능하게 만든다.
    """
    classes = np.unique(y)
    n = min(int((y == c).sum()) for c in classes)
    if max_per_class:
        n = min(n, int(max_per_class))
    keep = np.concatenate([rng.choice(np.flatnonzero(y == c), n, replace=False)
                           for c in classes])
    keep.sort()
    return keep, n


def _global_pca(X: np.ndarray, n_comp):
    """라벨과 무관한 전역 PCA. 순열검정의 귀무가설(라벨 교환)에 영향을 주지 않는다."""
    if not n_comp:
        return X
    from sklearn.decomposition import PCA

    n_comp = int(min(n_comp, X.shape[0] - 1, X.shape[1]))
    if n_comp < 2:
        return X
    return PCA(n_components=n_comp, svd_solver="full",
               random_state=0).fit_transform(X).astype(np.float64)


def _fold_model(clf_kind, n_pca, n_train, n_feat):
    """학습 폴드 안에서만 적합되는 PCA -> 스케일러 -> 분류기 파이프라인.

    (2026-10-09 수정) 이전에는 PCA를 CV 분할 전에 전체 데이터로 적합했다
    (라벨 비사용이라 순열검정은 유효하지만 정확도에 약한 transduction).
    이제 PCA도 폴드별로 적합해 시험 폴드 정보가 전혀 새지 않는다.
    """
    base = _make_clf(clf_kind)
    if n_pca:
        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline

        n_comp = int(min(n_pca, n_train - 1, n_feat))
        if n_comp >= 2:
            return make_pipeline(PCA(n_components=n_comp, svd_solver="full",
                                     random_state=0), base)
    return base


def _cv_splits(X, y, n_splits, n_repeats, seed, groups=None):
    """CV 분할 생성기. groups가 있으면 같은 그룹은 한 폴드에만 들어간다.

    (2026-10-09 외부 검토 ①) 같은 문장(시행)에서 나온 음소 구간은 시간적으로
    인접하고 기록 환경을 공유하므로, 무작위 분할은 정확도를 부풀릴 수 있다.
    groups = 시행 / 블록 / 세션 / 단어 ID. StratifiedGroupKFold를 반복마다
    다른 random_state로 섞는다. 그룹 수가 n_splits보다 적으면 폴드 수를 줄인다.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

    if groups is None:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=seed)
        yield from cv.split(X, y)
        return
    groups = np.asarray(groups)
    k = int(min(n_splits, np.unique(groups).size))
    if k < 2:
        return
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + r)
        for tr, te in cv.split(X, y, groups):
            if te.size == 0 or np.unique(y[tr]).size < 2:
                continue
            # 학습 폴드에 클래스당 2개 미만이면 LDA가 적합 불가 -> 폴드 건너뜀(극소 n 보호)
            if min(int((y[tr] == c).sum()) for c in np.unique(y[tr])) < 2:
                continue
            # 무결성: 같은 그룹이 학습·시험에 동시에 있으면 안 된다
            assert not (set(groups[tr].tolist()) & set(groups[te].tolist())), \
                "그룹 CV 무결성 위반: 학습·시험 폴드에 같은 그룹"
            yield tr, te


def _cv_accuracy(clf_kind, X, y, n_splits, n_repeats, seed, want_conf=False, n_pca=None,
                 groups=None, stats=None):
    """반복 CV 정확도. stats(dict)를 주면 폴드별 균형정확도·시험 n·폴드 수를 채운다."""
    accs, conf = [], np.zeros((2, 2), dtype=np.int64)
    baccs, n_tests = [], []
    classes = np.unique(y)
    for tr, te in _cv_splits(X, y, n_splits, n_repeats, seed, groups):
        clf = _fold_model(clf_kind, n_pca, tr.size, X.shape[1]).fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(float((pred == y[te]).mean()))
        n_tests.append(int(te.size))
        if classes.size == 2:
            rec = [float((pred[y[te] == c] == c).mean()) for c in classes
                   if (y[te] == c).any()]
            baccs.append(float(np.mean(rec)))
        if want_conf and classes.size == 2:
            for i, ci in enumerate(classes):
                for j, cj in enumerate(classes):
                    conf[i, j] += int(((y[te] == ci) & (pred == cj)).sum())
    if stats is not None:
        stats["bacc"] = float(np.mean(baccs)) if baccs else np.nan
        stats["n_folds_used"] = len(accs)
        stats["n_test_min"] = int(min(n_tests)) if n_tests else 0
        stats["n_test_mean"] = float(np.mean(n_tests)) if n_tests else np.nan
    if not accs:
        return np.asarray([np.nan]), conf
    return np.asarray(accs), conf


def balanced_pair_decode(X, y, n_pca=30, clf="lda", n_splits=5, n_repeats=10,
                         n_perm=500, perm_repeats=2, seed=0, min_per_class=8,
                         max_per_class=None, groups=None) -> dict:
    """두 클래스 균형 표집 + 반복 층화 CV + 라벨 순열검정.

    groups(구간별 그룹 ID)를 주면 그룹 단위로 폴드를 나눈다(문장·블록·세션·단어).
    순열 null도 같은 그룹 분할을 쓴다.
    반환: n_per_class, acc, acc_sd, dprime, acc_perm_mean, p_perm, ...
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    assert classes.size == 2, f"이진 분류만 지원 (관측 클래스 {classes})"

    rng = np.random.default_rng(seed)
    keep, n_per = _balance(y, rng, max_per_class)
    gb = np.asarray(groups)[keep] if groups is not None else None
    out = {"class_a": str(classes[0]), "class_b": str(classes[1]),
           "n_per_class": int(n_per), "n_features_in": int(X.shape[1]),
           "n_available_min_class": int(min(int((y == c).sum()) for c in classes)),
           "clf": clf, "n_splits": n_splits, "n_repeats": n_repeats,
           "n_perm": int(n_perm), "chance": 0.5,
           "n_cv_groups": int(np.unique(gb).size) if gb is not None else 0}
    if n_per < min_per_class:
        out.update({"acc": np.nan, "acc_sd": np.nan, "dprime": np.nan,
                    "p_perm": np.nan, "skipped": "n_per_class 부족"})
        return out

    Xb, yb = X[keep], y[keep]
    Xr = Xb  # PCA는 _fold_model 안에서 폴드별로 적합
    n_train_min = Xb.shape[0] - Xb.shape[0] // n_splits
    out["n_features_used"] = int(min(n_pca, n_train_min - 1, Xb.shape[1])) if n_pca else int(Xb.shape[1])

    fold_stats = {}
    accs, conf = _cv_accuracy(clf, Xr, yb, n_splits, n_repeats, seed, want_conf=True,
                              n_pca=n_pca, groups=gb, stats=fold_stats)
    out["acc"] = float(accs.mean())
    out["acc_sd"] = float(accs.std())
    out.update(fold_stats)  # bacc, n_folds_used, n_test_min, n_test_mean
    out["dprime"] = _dprime(conf[1, 1], conf[1, 0], conf[0, 1], conf[0, 0])

    if n_perm and n_perm > 0:
        obs, _ = _cv_accuracy(clf, Xr, yb, n_splits, perm_repeats, seed, n_pca=n_pca,
                              groups=gb)
        obs_mean = float(obs.mean())
        null = np.empty(n_perm, dtype=np.float64)
        prng = np.random.default_rng(seed + 991)
        for k in range(n_perm):
            yp = prng.permutation(yb)
            a, _ = _cv_accuracy(clf, Xr, yp, n_splits, perm_repeats, seed + k + 1,
                                n_pca=n_pca, groups=gb)
            null[k] = a.mean()
        out["acc_for_perm"] = obs_mean
        out["acc_perm_mean"] = float(null.mean())
        out["acc_perm_p95"] = float(np.percentile(null, 95))
        out["p_perm"] = float((1 + np.sum(null >= obs_mean)) / (n_perm + 1))
    else:
        out["p_perm"] = np.nan
    out["skipped"] = ""
    return out


def multiclass_decode(X, y, n_pca=100, clf="lda", n_splits=5, n_repeats=3,
                      seed=0, balance=True, max_per_class=None) -> dict:
    """N-way 분류 (39음소 전건전성 점검, Stage 2 혼동행렬 생성)."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    if balance:
        keep, n_per = _balance(y, rng, max_per_class)
        X, y = X[keep], y[keep]
    else:
        n_per = int(min(np.bincount(np.unique(y, return_inverse=True)[1])))

    Xr = X  # PCA는 폴드별 적합(_fold_model)
    classes = np.unique(y)
    conf = np.zeros((classes.size, classes.size), dtype=np.int64)
    c_index = {c: i for i, c in enumerate(classes)}
    accs = []
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    n_used = None
    for tr, te in cv.split(Xr, y):
        model = _fold_model(clf, n_pca, tr.size, Xr.shape[1]).fit(Xr[tr], y[tr])
        if n_used is None:
            n_used = int(min(n_pca, tr.size - 1, Xr.shape[1])) if n_pca else int(Xr.shape[1])
        pred = model.predict(Xr[te])
        accs.append(float((pred == y[te]).mean()))
        for t, p in zip(y[te], pred):
            conf[c_index[t], c_index[p]] += 1
    accs = np.asarray(accs)
    return {
        "n_classes": int(classes.size), "classes": classes.tolist(),
        "n_per_class": int(n_per), "acc": float(accs.mean()),
        "acc_sd": float(accs.std()), "chance": 1.0 / classes.size,
        "n_features_used": int(n_used),
        "confusion": pd.DataFrame(conf, index=classes, columns=classes),
    }


# ===========================================================================
# 9. Stage 1 — 정렬 불필요. tuningTasks 고립 음소 세션의 Gate A 유사체
# ===========================================================================

BIN_SEC = 0.02
# go 신호 이후 창(단위: 20 ms bin). None,None = go 구간 전체
STAGE1_WINDOWS = {
    "go_0_500ms": (0, 25),
    "go_500_1000ms": (25, 50),
    "go_all": (None, None),
}
# 음성 onset 기준 고정 길이 창. 모든 쌍에 같은 길이를 쓰므로 조음방법(manner)에
# 따른 지속시간 차이가 신호량 차이로 새는 것을 막는다(마찰음이 길어서 유리한 문제).
ONSET_WINDOWS = {
    "onset_0_300ms": (0, 15),
    "onset_0_500ms": (0, 25),
}
ALL_STAGE1_WINDOWS = {**STAGE1_WINDOWS, **ONSET_WINDOWS}


def estimate_audio_onsets(session: dict, rel_thr: float = 0.25, skip_bins: int = 5,
                          run_bins: int = 3, k_mad: float = 6.0):
    """audioEnvelope로 시행별 발화 onset(go 시작 기준 bin offset)을 추정한다.

    go 전이 직후 수 bin에 과제 쪽 클릭음이 들어가므로 skip_bins 만큼 건너뛴다.
    임계값 = max(go 구간 최대의 rel_thr, 지연 구간 기저 + k_mad*MAD).
    반환 (onset_rel, flags) — flags는 지연 구간에 이미 발화가 있던 시행(선행 발화).
    """
    if "audioEnvelope" not in session:
        n = session["goTrialEpochs"].shape[0]
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)

    env = np.asarray(session["audioEnvelope"]).ravel().astype(np.float64)
    go, de = session["goTrialEpochs"], session["delayTrialEpochs"]
    n = go.shape[0]
    onset = np.zeros(n, dtype=np.int64)
    flags = np.zeros(n, dtype=bool)
    for t in range(n):
        gs, ge = go[t, 0] - 1, go[t, 1]
        ds, dee = de[t, 0] - 1, de[t, 1]
        seg = env[gs:ge]
        base = env[ds:dee]
        med = float(np.median(base))
        mad = float(np.median(np.abs(base - med))) + 1e-9
        thr = max(rel_thr * float(seg.max()), med + k_mad * 1.4826 * mad)
        flags[t] = bool(base[-min(10, base.size):].max() > thr)
        hit = (seg > thr).astype(np.int64)
        hit[:skip_bins] = 0
        if run_bins > 1:
            c = np.convolve(hit, np.ones(run_bins, dtype=np.int64), "valid")
            idx = np.flatnonzero(c == run_bins)
        else:
            idx = np.flatnonzero(hit)
        if idx.size:
            onset[t] = int(idx[0])
    med_onset = int(np.median(onset[onset > 0])) if (onset > 0).any() else 0
    onset[onset == 0] = med_onset
    return onset, flags


def stage1_trial_features(session: dict, windows=None, feature_sets=FEATURE_SETS,
                          zscore: bool = True, anchor_bins=None) -> dict:
    """시행별 특징 = 창 내 시간평균. 블록별 z-score.

    (x̄ - mu)/sd == mean((x - mu)/sd) 이므로 시간평균 후 표준화해도 동일하다.
    열 순서 = concat(tx1[0..255], spikePow[0..255]).
    anchor_bins를 주면 창의 기준점이 go 시작 + anchor_bins[t] 가 된다
    (음성 onset 기준 고정 길이 창을 쓸 때).
    """
    windows = windows or STAGE1_WINDOWS
    go = session["goTrialEpochs"]
    n_trials = go.shape[0]
    n_bins = session[feature_sets[0]].shape[0]

    # 각 시행이 속한 블록 = go 시작 bin의 blockNum
    starts0 = go[:, 0] - 1  # MATLAB 1-indexed -> 0-indexed
    stops = go[:, 1]        # inclusive end -> python stop
    assert starts0.min() >= 0 and stops.max() <= n_bins, "go epoch가 데이터 범위를 벗어남"
    trial_block = session["blockNum"][starts0].astype(np.int64)

    if anchor_bins is None:
        anchors = np.zeros(n_trials, dtype=np.int64)
    else:
        anchors = np.asarray(anchor_bins, dtype=np.int64)
        assert anchors.shape == (n_trials,), (anchors.shape, n_trials)

    stats = {}
    if zscore:
        for fs in feature_sets:
            stats[fs] = blockwise_zscore_stats(session[fs], session["blockNum"])

    out = {"trial_idx": np.arange(n_trials), "block": trial_block,
           "windows": list(windows), "feature_sets": list(feature_sets),
           "n_chan": N_CHAN, "anchor_bins": anchors}
    if "phoneme" in session:
        out["phoneme"] = session["phoneme"].copy()
    out["session"] = session["session"]

    for wname, (w0, w1) in windows.items():
        X = np.empty((n_trials, N_CHAN * len(feature_sets)), dtype=np.float32)
        for t in range(n_trials):
            s0, s1 = starts0[t], stops[t]
            a0 = s0 + int(anchors[t])
            lo = s0 if w0 is None else min(a0 + w0, s1 - 1)
            hi = s1 if w1 is None else min(a0 + w1, s1)
            hi = max(hi, lo + 1)
            for i, fs in enumerate(feature_sets):
                seg = session[fs][lo:hi].astype(np.float64).mean(axis=0)
                if zscore:
                    mu, sd = stats[fs][int(trial_block[t])]
                    seg = (seg - mu) / sd
                X[t, i * N_CHAN:(i + 1) * N_CHAN] = seg
        assert X.shape == (n_trials, N_CHAN * len(feature_sets)), X.shape
        assert np.isfinite(X).all(), f"{wname}: 비유한 값 발생"
        out[wname] = X
    return out


def load_stage1_features(paths: dict, sessions=None, force: bool = False,
                         zscore: bool = True, windows=None,
                         use_audio_onset: bool = True) -> dict:
    """phonemes 세션들의 시행 특징을 합치고 npz로 캐시한다.

    use_audio_onset=True면 ONSET_WINDOWS는 audioEnvelope로 추정한 발화 onset을
    기준으로, STAGE1_WINDOWS는 go 시작을 기준으로 각각 뽑는다
    (audioEnvelope가 없는 세션은 onset=0, 즉 go 기준으로 되돌아간다).
    """
    sessions = list(sessions or TUNING_PHONEME_SESSIONS)
    windows = windows or ALL_STAGE1_WINDOWS
    tag = "_".join(s.split(".")[-1] for s in sessions)
    suffix = ("_z" if zscore else "_raw") + ("_onset" if use_audio_onset else "")
    cache = paths["CACHE_DIR"] / f"stage1_features_{tag}{suffix}.npz"
    if cache.exists() and not force:
        print(f"Stage 1 특징: 기존 캐시 재사용 ({cache.name})")
        with np.load(cache, allow_pickle=False) as f:
            out = {k: f[k] for k in f.files}
        out["phoneme"] = out["phoneme"].astype("<U8")
        out["session_name"] = out["session_name"].astype("<U32")
        out["windows"] = [w for w in windows if w in out]
        return out

    parts = []
    for name in sessions:
        f = paths["TUNING_DIR"] / f"{name}.mat"
        if not f.exists():
            print(f"  건너뜀(파일 없음): {f}")
            continue
        print(f"  특징 추출: {name}")
        sess = load_tuning_session(
            f, variable_names=_TUNING_VARS + ["audioEnvelope"])
        go_w = {k: v for k, v in windows.items() if k in STAGE1_WINDOWS}
        on_w = {k: v for k, v in windows.items() if k in ONSET_WINDOWS}
        feats = stage1_trial_features(sess, windows=go_w, zscore=zscore)
        if on_w:
            if use_audio_onset and "audioEnvelope" in sess:
                onset, flags = estimate_audio_onsets(sess)
                print(f"    음성 onset: 중앙 {int(np.median(onset))} bin "
                      f"({int(np.median(onset)) * 20} ms), "
                      f"선행 발화 의심 {int(flags.sum())}/{onset.size} 시행")
            else:
                onset = np.zeros(sess["goTrialEpochs"].shape[0], dtype=np.int64)
                flags = np.zeros_like(onset, dtype=bool)
                print("    audioEnvelope 없음 -> onset 창도 go 기준으로 뽑는다")
            on_feats = stage1_trial_features(sess, windows=on_w, zscore=zscore,
                                             anchor_bins=onset)
            for k in on_w:
                feats[k] = on_feats[k]
            feats["onset_bins"] = onset
            feats["onset_flag"] = flags.astype(np.int8)
        feats["session_name"] = np.array([name] * len(feats["trial_idx"]), dtype="<U32")
        parts.append(feats)
        del sess
    if not parts:
        raise FileNotFoundError(f"phonemes 세션이 없습니다: {paths['TUNING_DIR']}")

    out = {"windows": list(windows)}
    for w in windows:
        out[w] = np.concatenate([p[w] for p in parts], axis=0)
    for key in ("phoneme", "session_name"):
        out[key] = np.concatenate([p[key] for p in parts])
    for key in ("block", "trial_idx", "onset_bins", "onset_flag"):
        if all(key in p for p in parts):
            out[key] = np.concatenate([p[key] for p in parts])

    n = out[list(windows)[0]].shape[0]
    for w in windows:
        assert out[w].shape == (n, 512), f"{w} 모양 {out[w].shape} != ({n}, 512)"
    np.savez_compressed(cache, **{k: v for k, v in out.items() if k != "windows"})
    print(f"Stage 1 특징 저장: {cache.name}  시행 {n}개 x 512 특징")
    return out


def stage1_run(paths: dict, feats=None, windows=("go_all",), arrays=("6v", "44", "all"),
               n_perm=500, perm_repeats=2, n_repeats=10, n_pca=30, clf="lda",
               seed=20261009, force: bool = False, tag="") -> pd.DataFrame:
    """최소대립쌍별 균형 분류기 (후두쌍 + 조음위치 통제쌍) x 어레이 x 창."""
    out_path = paths["RESULT_DIR"] / f"stage1_pair_decoding{tag}.csv"
    if out_path.exists() and not force:
        print(f"Stage 1 쌍 분류: 기존 결과 재사용 ({out_path.name})")
        return pd.read_csv(out_path)

    feats = feats if feats is not None else load_stage1_features(paths)
    phon = feats["phoneme"]
    pair_specs = ([(a, b, "laryngeal") for a, b in LARYNGEAL_PAIRS]
                  + [(a, b, "place") for a, b in PLACE_CONTROL_PAIRS])

    rows = []
    for wname in windows:
        X_all = feats[wname]
        for array in arrays:
            cols = array_columns(array)
            Xa = X_all[:, cols]
            for a, b, contrast in pair_specs:
                m = (phon == a) | (phon == b)
                if m.sum() == 0:
                    continue
                res = balanced_pair_decode(
                    Xa[m], phon[m], n_pca=n_pca, clf=clf, n_repeats=n_repeats,
                    n_perm=n_perm, perm_repeats=perm_repeats, seed=seed)
                res.update({"window": wname, "array": array, "pair": f"{a}/{b}",
                            "contrast": contrast, "stage": "stage1",
                            "manner": pair_manner(a, b)})
                rows.append(res)
                print(f"  [{wname}|{array}] {a}/{b:<3} {contrast:<9} "
                      f"n={res['n_per_class']:<3} acc={res['acc']:.3f} "
                      f"d'={res['dprime']:.2f} p={res.get('p_perm', float('nan')):.4f}")
    df = pd.DataFrame(rows)
    front = ["stage", "window", "array", "pair", "contrast", "manner",
             "n_per_class", "acc", "acc_sd", "dprime", "p_perm"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(out_path, index=False)
    print(f"Stage 1 쌍 분류 저장: {out_path} ({len(df)}행)")
    return df


def stage1_phoneme39(paths: dict, feats=None, window="go_all", arrays=("6v", "44", "all"),
                     n_pca=100, n_repeats=3, seed=20261009, force: bool = False) -> pd.DataFrame:
    """39음소 분류 = 전체 전건전성 점검. 저자 보고대로 우연수준을 크게 넘어야 한다."""
    out_path = paths["RESULT_DIR"] / "stage1_phoneme39.csv"
    conf_path = paths["RESULT_DIR"] / "stage1_phoneme39_confusion.csv"
    if out_path.exists() and not force:
        print("Stage 1 39음소 분류: 기존 결과 재사용")
        return pd.read_csv(out_path)

    feats = feats if feats is not None else load_stage1_features(paths)
    phon = feats["phoneme"]
    keep = phon != "NOTHING"
    rows = []
    for array in arrays:
        X = feats[window][:, array_columns(array)][keep]
        res = multiclass_decode(X, phon[keep], n_pca=n_pca, n_repeats=n_repeats, seed=seed)
        if array == "all":
            res["confusion"].to_csv(conf_path)
        rows.append({"stage": "stage1", "window": window, "array": array,
                     "n_classes": res["n_classes"], "n_per_class": res["n_per_class"],
                     "acc": res["acc"], "acc_sd": res["acc_sd"], "chance": res["chance"],
                     "n_features_used": res["n_features_used"]})
        print(f"  [{window}|{array}] 39-way acc={res['acc']:.4f} "
              f"(chance={res['chance']:.4f}, n/class={res['n_per_class']})")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Stage 1 39음소 분류 저장: {out_path}")
    return df


# ===========================================================================
# 10. T2 유표성 — 교차검증 제곱거리(불편추정)
# ===========================================================================

def cv_markedness(X, labels, targets, baseline, n_reps=200, seed=0,
                  leave_target_out=True) -> pd.DataFrame:
    """각 target 음소의 평균 패턴이 '무표 기준선'에서 얼마나 먼가.

    절반 분할 교차검증으로 제곱거리를 불편추정한다:
        d2 = <m_p(half1) - b(half1), m_p(half2) - b(half2)> / n_features
    노이즈가 거리를 부풀리지 않으므로 유성/무성 계열을 직접 비교할 수 있다.
    기준선 b는 baseline 음소들의 '음소별 평균'의 평균(시행 수 불균형에 둔감),
    leave_target_out=True면 자기 자신은 기준선에서 제외한다.

    주의: 과제 명세의 "쌍 내부 평균으로부터의 거리"는 두 클래스가 정의상
    등거리가 되어 정보가 없다. 그래서 기준선은 baseline 집합(기본: 후두 대립을
    갖는 자음 전체)의 평균으로 둔다.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    baseline = list(baseline)
    targets = list(targets)
    pool = sorted(set(baseline) | set(targets))
    idx = {p: np.flatnonzero(labels == p) for p in pool}
    for p in targets:
        if idx[p].size < 4:
            warnings.warn(f"{p}: 시행 {idx[p].size}개로 분할 거리 추정 불안정", stacklevel=2)

    acc = {p: [] for p in targets}
    for _ in range(n_reps):
        means = {}
        for p in pool:
            ii = rng.permutation(idx[p])
            h = ii.size // 2
            if h < 1:
                continue
            means[p] = (X[ii[:h]].mean(0), X[ii[h:2 * h]].mean(0))
        for p in targets:
            if p not in means:
                continue
            base = [q for q in baseline if q in means and not (leave_target_out and q == p)]
            if not base:
                continue
            b1 = np.mean([means[q][0] for q in base], axis=0)
            b2 = np.mean([means[q][1] for q in base], axis=0)
            acc[p].append(float(np.dot(means[p][0] - b1, means[p][1] - b2) / d))

    rows = []
    vmap = feature_map("voicing")
    for p in targets:
        v = np.asarray(acc[p])
        rows.append({"phoneme": p,
                     "series": ("voiced" if vmap.get(p, 0) == 1 else "voiceless"),
                     "d2_cv": float(v.mean()) if v.size else np.nan,
                     "d2_cv_sd": float(v.std()) if v.size else np.nan,
                     "dist_cv": float(np.sign(v.mean()) * np.sqrt(abs(v.mean()))) if v.size else np.nan,
                     "n_reps": int(v.size), "n_trials": int(idx[p].size)})
    return pd.DataFrame(rows)


def stage1_markedness(paths: dict, feats=None, window="go_all",
                      arrays=("6v", "44", "all"), n_reps=200, seed=20261009,
                      force: bool = False) -> pd.DataFrame:
    """T2: 고립 음소에서 유성 계열 vs 무성 계열의 기준선 거리."""
    out_path = paths["RESULT_DIR"] / "stage1_markedness.csv"
    if out_path.exists() and not force:
        print("Stage 1 유표성: 기존 결과 재사용")
        return pd.read_csv(out_path)

    feats = feats if feats is not None else load_stage1_features(paths)
    phon = feats["phoneme"]
    targets = PAIRED_OBSTRUENTS
    rows = []
    for array in arrays:
        X = feats[window][:, array_columns(array)]
        tab = cv_markedness(X, phon, targets, baseline=targets,
                            n_reps=n_reps, seed=seed)
        tab["array"] = array
        tab["window"] = window
        tab["baseline"] = "paired_obstruents_LOO"
        rows.append(tab)
        vo = tab.loc[tab["series"] == "voiced", "d2_cv"].mean()
        vl = tab.loc[tab["series"] == "voiceless", "d2_cv"].mean()
        print(f"  [{array}] 유성 d2={vo:.4f}  무성 d2={vl:.4f}  "
              f"차이(유성-무성)={vo - vl:+.4f}")
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"Stage 1 유표성 저장: {out_path} ({len(df)}행)")
    return df


def markedness_contrast_test(mark_df: pd.DataFrame, array="all") -> dict:
    """후두쌍 8개에 대해 (유성 d2 - 무성 d2) 부호검정 + Wilcoxon."""
    from scipy.stats import wilcoxon

    sub = mark_df[mark_df["array"] == array].set_index("phoneme")
    diffs, pairs = [], []
    for voiceless, voiced in LARYNGEAL_PAIRS:
        if voiceless in sub.index and voiced in sub.index:
            diffs.append(float(sub.loc[voiced, "d2_cv"] - sub.loc[voiceless, "d2_cv"]))
            pairs.append(f"{voiceless}/{voiced}")
    diffs = np.asarray(diffs)
    out = {"array": array, "n_pairs": int(diffs.size), "pairs": ";".join(pairs),
           "mean_diff_voiced_minus_voiceless": float(diffs.mean()) if diffs.size else np.nan,
           "n_voiced_farther": int((diffs > 0).sum())}
    if diffs.size >= 5:
        try:
            stat, p = wilcoxon(diffs)
            out["wilcoxon_stat"], out["wilcoxon_p"] = float(stat), float(p)
        except ValueError:
            out["wilcoxon_stat"], out["wilcoxon_p"] = np.nan, np.nan
    p = out.get("wilcoxon_p", np.nan)
    if not np.isfinite(p) or p >= 0.05:
        out["verdict"] = "판정 불가(유성-무성 차이 비유의)"
    elif out["mean_diff_voiced_minus_voiceless"] > 0:
        out["verdict"] = "[voice] 쪽(유성이 더 유표)"
    else:
        out["verdict"] = "[spread glottis] 쪽(무성이 더 유표)"
    return out


# ===========================================================================
# 11. Stage 2 — 강제 정렬 결과(음소 구간)를 쓰는 분석
# ===========================================================================
# 정렬 CSV 계약 (t12/results/alignments/<session>_<partition>.csv):
#   session, partition, modality, trial_idx, sentence, phone_idx, phoneme,
#   word_idx, position_in_word(initial|medial|final), start_bin, end_bin, n_bins
#   (start_bin/end_bin은 해당 시행의 tx1/spikePow에 대한 20 ms bin index)

ALIGN_COLUMNS = ["session", "partition", "modality", "trial_idx", "sentence",
                 "phone_idx", "phoneme", "word_idx", "position_in_word",
                 "start_bin", "end_bin", "n_bins"]
# 정렬 에이전트가 추가로 주는 열(있으면 쓴다)
ALIGN_OPTIONAL = ["seg_start_bin", "seg_end_bin", "score", "word",
                  "rnn_start_step", "rnn_end_step",
                  "rnn_center_start", "rnn_center_end", "input_layer_from"]
POSITIONS = ["initial", "medial", "final"]
# position_in_word에는 'sil'(어말 표시용 SIL)과 'single'(1음소 단어)도 등장한다.
# 자음 분석에서는 자동으로 빠지지만, 분포는 Stage 2 실행 때 출력한다.
PRE_BINS = 5  # 100 ms
SCORE_MIN = 0.5  # 기본 CTC 사후확률 하한

# 창 선택 규칙:
#   seg_start_bin/seg_end_bin = 시행을 빈틈 없이 분할한 구간(공백 구간은 중점에서 분할).
#   start_bin/end_bin = RNN 수용영역(kernel 32, stride 4) 이라서 32 bin 폭으로 겹친다.
#   따라서 seg_* 가 있으면 반드시 그것을 창으로 쓴다.
WINDOW_COLS_PREFERRED = ("seg_start_bin", "seg_end_bin")
WINDOW_COLS_CENTER = ("rnn_center_start", "rnn_center_end")
WINDOW_COLS_FALLBACK = ("start_bin", "end_bin")

# 사전학습 RNN의 입력 적층 (released args.yaml): kernel 32 bin, stride 4 bin.
# 따라서 RNN step k 는 bin [4k, 4k+32) 을 보고, 그 중심은 bin 4k+16 이다.
RNN_KERNEL_BINS = 32
RNN_STRIDE_BINS = 4

WINDOW_MODES = ("seg", "rnn_center", "legacy")

# Stage 2 기본 특징 = area 6v 만 (tx1[:, :128] + spikePow[:, :128]) + 상하 분할.
# area 44는 Stage 1에서 우연수준이었으므로 기본에서 뺀다.
STAGE2_ARRAYS = ("6v", "6v_sup", "6v_inf")


def load_alignments(align_dir, sessions=None, partitions=None,
                    window_mode: str = "seg", verbose: bool = True) -> pd.DataFrame:
    """있는 정렬 CSV만 모아 하나로 합친다(다른 에이전트가 하나씩 추가하는 구조)."""
    align_dir = Path(align_dir)
    files = sorted(align_dir.glob("*.csv"))
    if not files:
        if verbose:
            print(f"정렬 CSV가 아직 없습니다 ({align_dir}).")
        return pd.DataFrame(columns=ALIGN_COLUMNS)

    parts = []
    for f in files:
        df = pd.read_csv(f)
        missing = [c for c in ALIGN_COLUMNS if c not in df.columns]
        if missing:
            if verbose:
                print(f"  건너뜀 {f.name}: 계약 열 누락 {missing}")
            continue
        keep = ALIGN_COLUMNS + [c for c in ALIGN_OPTIONAL if c in df.columns]
        parts.append(df[keep])
    if not parts:
        return pd.DataFrame(columns=ALIGN_COLUMNS)

    out = pd.concat(parts, ignore_index=True)
    out["session"] = out["session"].astype(str)
    out["partition"] = out["partition"].astype(str)
    out["phoneme"] = out["phoneme"].astype(str).str.upper()
    out["position_in_word"] = out["position_in_word"].astype(str).str.lower()
    # modality는 세션 라벨을 정본으로 다시 채운다(정렬 파일 오기 방지)
    derived = out["session"].map(session_modality)
    bad = (derived != "unknown") & (derived != out["modality"])
    if bad.any():
        print(f"  주의: modality 불일치 {int(bad.sum())}행 -> 세션 라벨로 교정")
        out.loc[bad, "modality"] = derived[bad]
    if sessions:
        out = out[out["session"].isin(list(sessions))]
    if partitions:
        out = out[out["partition"].isin(list(partitions))]
    out = out.reset_index(drop=True)

    # 분석에 쓸 창(win_start, win_end)을 확정한다.
    assert window_mode in WINDOW_MODES, f"window_mode는 {WINDOW_MODES} 중 하나"
    have_seg = all(c in out.columns for c in WINDOW_COLS_PREFERRED)
    have_rnn = all(c in out.columns for c in ("rnn_start_step", "rnn_end_step"))
    have_center = all(c in out.columns for c in WINDOW_COLS_CENTER)

    if window_mode == "rnn_center" and have_center:
        # 정렬 에이전트가 제공하는 네이티브 좁은 창(점유 core step만).
        ws, we = out["rnn_center_start"], out["rnn_center_end"]
        src = "rnn_center(native)"
    elif window_mode == "rnn_center" and have_rnn:
        # 네이티브 열이 없을 때만 수용영역 확장분을 되돌려 직접 계산한다.
        ws = RNN_STRIDE_BINS * out["rnn_start_step"].astype(int) + RNN_KERNEL_BINS // 2
        we = (RNN_STRIDE_BINS * (out["rnn_end_step"].astype(int) + 1)
              + RNN_KERNEL_BINS // 2)
        src = f"rnn_center(derived k{RNN_KERNEL_BINS} s{RNN_STRIDE_BINS})"
    elif window_mode in ("seg", "rnn_center") and have_seg:
        if window_mode == "rnn_center":
            warnings.warn("rnn_*_step 열이 없어 seg_bins로 되돌립니다.", stacklevel=2)
        ws, we, src = out[WINDOW_COLS_PREFERRED[0]], out[WINDOW_COLS_PREFERRED[1]], "seg_bins"
    else:
        ws, we = out[WINDOW_COLS_FALLBACK[0]], out[WINDOW_COLS_FALLBACK[1]]
        src = "rnn_receptive_field"
        warnings.warn(
            "seg_start_bin/seg_end_bin이 없어 start_bin/end_bin(RNN 수용영역, "
            f"{RNN_KERNEL_BINS} bin 폭으로 겹침)을 창으로 씁니다.", stacklevel=2)
    out["win_start"] = np.asarray(ws).astype(int)
    out["win_end"] = np.asarray(we).astype(int)
    out["win_bins"] = out["win_end"] - out["win_start"]
    out["window_source"] = src
    if "score" not in out.columns:
        out["score"] = np.nan

    # held_out: RNN 학습에 쓰이지 않은 구간.
    #   trained 세션 -> test 파티션만 held out (train은 암기돼 PER~0.1%)
    #   학습 설정에 없던 세션(입력층 차용) -> 두 파티션 모두 held out
    if "input_layer_from" in out.columns:
        borrowed = out["input_layer_from"].astype(str) != out["session"].astype(str)
    else:
        borrowed = pd.Series(False, index=out.index)
    out["held_out"] = (out["partition"] == "test") | borrowed
    out = annotate_context(out)

    if verbose:
        print(f"정렬 CSV {len(parts)}개 / {len(out)}개 음소 구간 "
              f"(세션 {out['session'].nunique()}개, "
              f"양식 {sorted(out['modality'].unique())})")
        print(f"  창 출처: {src} (중앙 폭 {out['win_bins'].median():.0f} bin "
              f"= {out['win_bins'].median() * 20:.0f} ms)")
        ov = _window_overlap_report(out)
        print(f"  인접 구간 겹침 {ov['n_overlapping']}/{ov['n_consecutive']} "
              f"({100 * ov['frac_overlapping']:.1f}%), 빈틈 {ov['n_gaps']}")
        if ov["frac_overlapping"] > 0.2:
            print("  주의: 창이 서로 크게 겹칩니다. 인접 음소 신호가 섞입니다. "
                  "window_mode='rnn_center'로 좁은 창 민감도 분석을 함께 보세요.")
        print(f"  위치 라벨: {out['position_in_word'].value_counts().to_dict()}")
        if out["score"].notna().any():
            q = out["score"].quantile([0.05, 0.5]).round(4).to_dict()
            print(f"  score 5%={q[0.05]} 중앙={q[0.5]}")
        if "input_layer_from" in out.columns:
            bs = out.loc[out["input_layer_from"].astype(str)
                         != out["session"], "session"].unique()
            if bs.size:
                print(f"  차용 입력층(= 미학습, 두 파티션 모두 held out): "
                      f"{sorted(bs.tolist())}")
        print(f"  held_out 구간 {int(out['held_out'].sum())}/{len(out)} "
              f"({100 * out['held_out'].mean():.1f}%)")
    return out


def session_per_table(align_dir, verbose: bool = True) -> pd.DataFrame:
    """<session>_<partition>_trials.csv 들을 모아 세션별 PER과 입력층 출처를 낸다."""
    align_dir = Path(align_dir)
    files = sorted(align_dir.glob("*_trials.csv"))
    if not files:
        return pd.DataFrame(columns=["session", "partition", "modality", "per",
                                     "n_trials", "input_layer_from", "borrowed"])
    parts = [pd.read_csv(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    need = {"session", "partition", "per"}
    if not need <= set(df.columns):
        warnings.warn(f"trials CSV에 {need - set(df.columns)} 열이 없습니다", stacklevel=2)
        return pd.DataFrame()
    w = "n_phonemes_true" if "n_phonemes_true" in df.columns else None
    rows = []
    for (sess, part), g in df.groupby(["session", "partition"]):
        per = (float(np.average(g["per"], weights=g[w])) if w and g[w].sum() > 0
               else float(g["per"].mean()))
        layer = (str(g["input_layer_from"].iloc[0])
                 if "input_layer_from" in g.columns else "")
        rows.append({"session": sess, "partition": part,
                     "modality": (str(g["modality"].iloc[0])
                                  if "modality" in g.columns else session_modality(sess)),
                     "per": round(per, 4), "n_trials": int(len(g)),
                     "input_layer_from": layer,
                     "borrowed": bool(layer and layer != sess)})
    out = pd.DataFrame(rows).sort_values(["modality", "session"]).reset_index(drop=True)
    if verbose:
        print("세션별 PER(음소 수 가중):")
        print(out.to_string(index=False))
    return out


def _window_overlap_report(df: pd.DataFrame) -> dict:
    """인접 음소 창이 얼마나 겹치는지(= 신호 혼입 위험) 집계."""
    n_ov = n_gap = n_cons = 0
    for _, g in df.groupby(["session", "partition", "trial_idx"], sort=False):
        g = g.sort_values("phone_idx")
        a = g["win_start"].to_numpy()
        b = g["win_end"].to_numpy()
        if a.size < 2:
            continue
        n_cons += a.size - 1
        n_ov += int((a[1:] < b[:-1]).sum())
        n_gap += int((a[1:] > b[:-1]).sum())
    return {"n_consecutive": n_cons, "n_overlapping": n_ov, "n_gaps": n_gap,
            "frac_overlapping": (n_ov / n_cons if n_cons else 0.0)}


def annotate_context(align_df: pd.DataFrame) -> pd.DataFrame:
    """각 음소 구간에 앞뒤 음소와 모음 간(intervocalic) 여부를 붙인다.

    설탄음화 통제용: 미국영어에서 모음 사이 T/D는 둘 다 설탄음이 되므로,
    어중 T/D는 비모음간 맥락으로 제한한 행을 따로 봐야 한다.
    """
    if align_df.empty:
        for c in ("prev_phoneme", "next_phoneme", "intervocalic"):
            align_df[c] = None
        return align_df
    df = align_df.sort_values(["session", "partition", "trial_idx", "phone_idx"]).copy()
    g = df.groupby(["session", "partition", "trial_idx"], sort=False)["phoneme"]
    df["prev_phoneme"] = g.shift(1).fillna("SIL")
    df["next_phoneme"] = g.shift(-1).fillna("SIL")
    vs = set(VOWELS)
    df["prev_is_vowel"] = df["prev_phoneme"].isin(vs)
    df["next_is_vowel"] = df["next_phoneme"].isin(vs)
    df["intervocalic"] = df["prev_is_vowel"] & df["next_is_vowel"]
    return df.sort_index()


def contrast_gap_ci(df: pd.DataFrame, group_col="position", levels=None,
                    array="6v", n_boot=10000, exclude_pairs=(), seed=20261009,
                    dod_pair=None) -> pd.DataFrame:
    """그룹별 (조음위치 - 후두) 격차와 두 그룹 격차의 차이(DoD)에 부트스트랩 CI.

    판정이 걸려 있는 양은 '그룹 x 대립유형 상호작용'이므로, 쌍을 재표집해
    격차의 불확실성을 직접 낸다. group_col='position'이면 T1, 'modality'면 T3.
    """
    rng = np.random.default_rng(seed)
    sub = df[df["array"] == array]
    if exclude_pairs:
        sub = sub[~sub["pair"].isin(list(exclude_pairs))]
    levels = list(levels) if levels is not None else sorted(sub[group_col].dropna().unique())
    rows = []
    store = {}
    for position in levels:
        s = sub[sub[group_col] == position]
        lar = s.loc[s["contrast"] == "laryngeal", "acc"].to_numpy()
        pla = s.loc[s["contrast"] == "place", "acc"].to_numpy()
        lar, pla = lar[np.isfinite(lar)], pla[np.isfinite(pla)]
        if lar.size == 0 or pla.size == 0:
            continue
        store[position] = (lar, pla)
        boot = np.empty(n_boot)
        for i in range(n_boot):
            boot[i] = (rng.choice(pla, pla.size).mean()
                       - rng.choice(lar, lar.size).mean())
        rows.append({
            "array": array, group_col: position,
            "n_laryngeal_pairs": int(lar.size), "n_place_pairs": int(pla.size),
            "laryngeal_acc": round(float(lar.mean()), 4),
            "place_acc": round(float(pla.mean()), 4),
            "gap_place_minus_laryngeal": round(float(pla.mean() - lar.mean()), 4),
            "gap_ci_lo": round(float(np.percentile(boot, 2.5)), 4),
            "gap_ci_hi": round(float(np.percentile(boot, 97.5)), 4),
            "excluded_pairs": ";".join(exclude_pairs),
        })
    # 차이의 차이. T1 기본은 (어말 격차 - 어두 격차), 양수면 후두 정보가 어두에 쏠림.
    a_lvl, b_lvl = dod_pair or ("initial", "final")
    if a_lvl in store and b_lvl in store:
        li, pi = store[a_lvl]
        lf, pf = store[b_lvl]
        boot = np.empty(n_boot)
        for i in range(n_boot):
            d_i = rng.choice(pi, pi.size).mean() - rng.choice(li, li.size).mean()
            d_f = rng.choice(pf, pf.size).mean() - rng.choice(lf, lf.size).mean()
            boot[i] = d_f - d_i
        obs = ((pf.mean() - lf.mean()) - (pi.mean() - li.mean()))
        rows.append({
            "array": array, group_col: f"DoD({b_lvl} - {a_lvl})",
            "n_laryngeal_pairs": int(min(li.size, lf.size)),
            "n_place_pairs": int(min(pi.size, pf.size)),
            "laryngeal_acc": np.nan, "place_acc": np.nan,
            "gap_place_minus_laryngeal": round(float(obs), 4),
            "gap_ci_lo": round(float(np.percentile(boot, 2.5)), 4),
            "gap_ci_hi": round(float(np.percentile(boot, 97.5)), 4),
            "excluded_pairs": ";".join(exclude_pairs),
        })
    return pd.DataFrame(rows)


def position_contrast_interaction(df: pd.DataFrame, array="6v", n_boot=10000,
                                  exclude_pairs=(), seed=20261009) -> pd.DataFrame:
    """T1용 얇은 래퍼 (group_col='position')."""
    return contrast_gap_ci(df, group_col="position", levels=POSITIONS, array=array,
                           n_boot=n_boot, exclude_pairs=exclude_pairs, seed=seed,
                           dod_pair=("initial", "final"))


def apply_score_filter(align_df: pd.DataFrame, score_min=SCORE_MIN,
                       verbose: bool = True):
    """CTC 사후확률 하한으로 구간을 걸러내고 조건별 탈락 수를 보고한다."""
    if align_df.empty or score_min is None or "score" not in align_df.columns \
            or align_df["score"].isna().all():
        report = pd.DataFrame(columns=["modality", "position_in_word", "n_total",
                                       "n_kept", "n_dropped", "frac_dropped"])
        return align_df, report

    keep = align_df["score"].fillna(-np.inf) >= float(score_min)
    g = align_df.assign(_kept=keep).groupby(["modality", "position_in_word"],
                                            dropna=False)
    report = g["_kept"].agg(n_total="size", n_kept="sum").reset_index()
    report["n_dropped"] = report["n_total"] - report["n_kept"]
    report["frac_dropped"] = (report["n_dropped"] / report["n_total"]).round(4)
    report["score_min"] = float(score_min)
    if verbose:
        print(f"score >= {score_min} 필터: {int(keep.sum())}/{len(align_df)} 유지 "
              f"({int((~keep).sum())} 탈락, {100 * (~keep).mean():.1f}%)")
        print(report.to_string(index=False))
    return align_df.loc[keep].reset_index(drop=True), report


def make_synthetic_alignments(out_path, sessions=(("t12.2022.08.13", "test"),
                                                 ("t12.2022.08.23", "test")),
                              n_trials=30, n_words=6, bins_per_trial=160,
                              seed=20261009) -> pd.DataFrame:
    """계약을 따르는 합성 정렬 CSV. 실제 정렬이 오기 전 Stage 2 배관 점검용."""
    rng = np.random.default_rng(seed)
    # 후두쌍 + 조음위치 통제쌍의 구성원 전체(M, N 포함 -> nasal 자질도 2값이 된다)
    pool = {p for pair in LARYNGEAL_PAIRS + PLACE_CONTROL_PAIRS for p in pair}
    cons = [p for p in CONSONANTS if p in pool]
    rows = []
    for session, partition in sessions:
        for trial in range(n_trials):
            cursor = rng.integers(3, 8)
            phone_idx = 0
            words = []
            for w in range(n_words):
                n_ph = 3
                onset = rng.choice(cons)
                coda = rng.choice(cons)
                seq = [(onset, "initial"), ("AA", "medial"), (coda, "final")]
                words.append(" ".join(s[0] for s in seq))
                for ph, pos in seq:
                    dur = int(rng.integers(3, 9))
                    if cursor + dur >= bins_per_trial:
                        break
                    rows.append({
                        "session": session, "partition": partition,
                        "modality": session_modality(session),
                        "trial_idx": trial, "sentence": None,
                        "phone_idx": phone_idx, "phoneme": ph,
                        "word_idx": w, "position_in_word": pos,
                        "start_bin": cursor, "end_bin": cursor + dur,
                        "n_bins": dur,
                    })
                    cursor += dur + int(rng.integers(0, 2))
                    phone_idx += 1
                del n_ph
            sent = " ".join(words)
            for r in rows:
                if r["session"] == session and r["trial_idx"] == trial and r["sentence"] is None:
                    r["sentence"] = sent
    df = pd.DataFrame(rows, columns=ALIGN_COLUMNS)
    df["seg_start_bin"] = df["start_bin"]
    df["seg_end_bin"] = df["end_bin"]
    df["score"] = 1.0
    df["win_start"] = df["start_bin"]
    df["win_end"] = df["end_bin"]
    df["win_bins"] = df["win_end"] - df["win_start"]
    df["window_source"] = "synthetic"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"합성 정렬 저장: {out_path} ({len(df)}구간, "
          f"세션 {df['session'].nunique()}개)")
    return df


def _competition_block_stats(d: dict, feature_sets=FEATURE_SETS, eps: float = 1e-8) -> dict:
    """블록별 채널 평균/표준편차(러닝 합). {feature_set: {block: (mu, sd)}}."""
    out = {}
    for fs in feature_sets:
        acc = {}
        for i, A in enumerate(d[fs]):
            b = int(d["blockIdx"][i])
            Af = A.astype(np.float64)
            s, ss, n = acc.get(b, (0.0, 0.0, 0))
            acc[b] = (s + Af.sum(0), ss + (Af ** 2).sum(0), n + Af.shape[0])
        st = {}
        for b, (s, ss, n) in acc.items():
            mu = s / n
            sd = np.sqrt(np.maximum(ss / n - mu ** 2, 0.0))
            sd[sd < eps] = 1.0
            st[b] = (mu, sd)
        out[fs] = st
    return out


def stage2_segment_features(paths: dict, align_df: pd.DataFrame, pre_bins: int = PRE_BINS,
                            zscore: bool = True, feature_sets=FEATURE_SETS,
                            verbose: bool = True):
    """정렬 구간별 특징. 반환 (X_seg, X_pre, meta).

    X_seg[i] = concat_fs( mean_t feature[start_bin:end_bin, :] ) 블록 z-score 적용
    X_pre[i] = 같은 방식으로 onset 직전 pre_bins (기본 5 bin = 100 ms)
    """
    if align_df.empty:
        return (np.zeros((0, N_CHAN * len(feature_sets)), np.float32),
                np.zeros((0, N_CHAN * len(feature_sets)), np.float32),
                align_df.copy())

    n_feat = N_CHAN * len(feature_sets)
    X_seg = np.full((len(align_df), n_feat), np.nan, dtype=np.float32)
    X_pre = np.full((len(align_df), n_feat), np.nan, dtype=np.float32)
    ok = np.zeros(len(align_df), dtype=bool)
    blk = np.full(len(align_df), -1, dtype=np.int64)  # 블록 단위 CV용
    pre_ok = np.zeros(len(align_df), dtype=bool)      # 직전 창이 온전히 확보됐는가
    clipped = np.zeros(len(align_df), dtype=bool)     # 요청 창이 시행 길이에 잘렸는가
    used, skipped = [], []

    for (session, partition), grp in align_df.groupby(["session", "partition"], sort=True):
        f = paths["COMP_DIR"] / partition / f"{session}.mat"
        if not f.exists():
            skipped.append(f"{session}/{partition} (파일 없음)")
            continue
        d = load_competition_session(f)
        stats = _competition_block_stats(d, feature_sets) if zscore else None
        n_trials = d["n_trials"]
        n_used = 0
        for pos, row in zip(grp.index.to_numpy(), grp.itertuples(index=False)):
            t = int(row.trial_idx)
            if not (0 <= t < n_trials):
                continue
            T = d[feature_sets[0]][t].shape[0]
            s0, e0 = int(row.win_start), int(row.win_end)
            s, e = max(0, min(s0, T - 1)), max(1, min(e0, T))
            if e <= s:
                continue
            clipped[pos] = (s != s0) or (e != e0)
            ps, pe = max(0, s - pre_bins), s
            # (2026-10-09 검토) 직전 창이 짧으면 seg로 대체되던 것을 플래그로 기록.
            # 계획(pre) 분석은 pre_valid 행만 써야 한다.
            pre_ok[pos] = (pe - ps) == pre_bins
            block = int(d["blockIdx"][t])
            blk[pos] = block
            for i, fs in enumerate(feature_sets):
                A = d[fs][t]
                seg = A[s:e].astype(np.float64).mean(0)
                pre = (A[ps:pe].astype(np.float64).mean(0) if pe > ps else seg)
                if zscore:
                    mu, sd = stats[fs][block]
                    seg = (seg - mu) / sd
                    pre = (pre - mu) / sd
                X_seg[pos, i * N_CHAN:(i + 1) * N_CHAN] = seg
                X_pre[pos, i * N_CHAN:(i + 1) * N_CHAN] = pre
            ok[pos] = True
            n_used += 1
        used.append(f"{session}/{partition}:{n_used}")
        del d

    meta = align_df.loc[ok].reset_index(drop=True)
    meta["block_idx"] = blk[ok]
    meta["pre_valid"] = pre_ok[ok]
    meta["win_clipped"] = clipped[ok]
    X_seg, X_pre = X_seg[ok], X_pre[ok]
    assert X_seg.shape == (len(meta), n_feat), (X_seg.shape, len(meta))
    assert X_pre.shape == X_seg.shape
    assert np.isfinite(X_seg).all() and np.isfinite(X_pre).all(), "비유한 값 발생"
    if verbose:
        print(f"Stage 2 구간 특징: {X_seg.shape[0]}구간 x {n_feat}특징")
        print(f"  사용: {', '.join(used) if used else '없음'}")
        print(f"  직전 {pre_bins}bin 창 온전: {int(meta['pre_valid'].sum())}/{len(meta)}, "
              f"요청 창이 시행 길이에 잘림: {int(meta['win_clipped'].sum())}")
        if skipped:
            print(f"  건너뜀: {', '.join(skipped)}")
    return X_seg, X_pre, meta


def stage2_synthetic_features(align_df: pd.DataFrame, feature_sets=FEATURE_SETS,
                              noise=1.0, voicing_effect=0.45, place_effect=0.9,
                              initial_boost=1.6, nonvocal_scale=0.5,
                              seed=20261009):
    """competitionData가 없을 때 Stage 2 배관을 끝까지 돌리기 위한 전 합성 특징.

    기대 구조를 일부러 심는다: 조음위치 효과 > 유성성 효과, 유성성은 어두에서
    강하고(= [spread glottis] 예측), nonvocal에서 약해진다. area 6v(0~127)에만
    신호를 넣고 area 44에는 넣지 않는다.
    """
    rng = np.random.default_rng(seed)
    n = len(align_df)
    n_feat = N_CHAN * len(feature_sets)
    X = rng.normal(0, noise, size=(n, n_feat))

    vmap = feature_map("voicing")
    places = sorted({p for p in feature_map("place").values()})
    place_axis = {pl: rng.normal(0, 1, n_feat) for pl in places}
    voice_axis = rng.normal(0, 1, n_feat)
    phone_axis = {}
    mask6v = np.zeros(n_feat, dtype=bool)
    mask6v[array_columns("6v")] = True
    for ax in list(place_axis.values()) + [voice_axis]:
        ax[~mask6v] = 0.0

    pmap = feature_map("place")
    for i, row in enumerate(align_df.itertuples(index=False)):
        ph = str(row.phoneme)
        if ph not in phone_axis:
            a = rng.normal(0, 1, n_feat)
            a[~mask6v] = 0.0
            phone_axis[ph] = a
        X[i] += 0.5 * phone_axis[ph]
        if ph in pmap:
            X[i] += place_effect * place_axis[pmap[ph]]
        if ph in vmap:
            w = voicing_effect
            if str(row.position_in_word) == "initial":
                w *= initial_boost
            if str(row.modality) == "nonvocal":
                w *= nonvocal_scale
            X[i] += w * (1.0 if vmap[ph] == 1 else -1.0) * voice_axis
    X_pre = X * 0.3 + rng.normal(0, noise, size=X.shape)
    assert X.shape == (n, n_feat)
    print(f"합성 Stage 2 특징: {X.shape[0]}구간 x {n_feat}특징 (area 6v에만 신호)")
    return X.astype(np.float32), X_pre.astype(np.float32), align_df.reset_index(drop=True)


# --- T1: 위치별 최소대립쌍 분리도 -----------------------------------------

def _partition_mask(meta: pd.DataFrame, partitions=None):
    """파티션 필터.

    None        전부(pooled)
    'held_out'  RNN 학습에 안 쓰인 구간만 (test 파티션 + 미학습 세션의 전 파티션)
    그 외       해당 partition 값만
    """
    if partitions is None:
        return np.ones(len(meta), dtype=bool), "pooled"
    if partitions == "held_out":
        if "held_out" not in meta.columns:
            warnings.warn("held_out 열이 없어 partition=='test'로 대체", stacklevel=2)
            return meta["partition"].eq("test").to_numpy(), "test"
        return meta["held_out"].fillna(False).to_numpy(), "held_out"
    parts = [partitions] if isinstance(partitions, str) else list(partitions)
    return meta["partition"].isin(parts).to_numpy(), "+".join(parts)


def segment_group_ids(meta: pd.DataFrame, group_col: str) -> np.ndarray:
    """구간별 CV 그룹 ID. 'trial'(문장 시행) / 'block' / 'session' / 'word' 또는 meta 열 이름."""
    if group_col == "trial":
        key = (meta["session"].astype(str) + "|" + meta["partition"].astype(str)
               + "|" + meta["trial_idx"].astype(str))
    elif group_col == "block":
        if "block_idx" not in meta.columns:
            warnings.warn("block_idx 열이 없어 trial 그룹으로 대체", stacklevel=2)
            return segment_group_ids(meta, "trial")
        key = (meta["session"].astype(str) + "|" + meta["partition"].astype(str)
               + "|" + meta["block_idx"].astype(str))
    elif group_col == "session":
        key = meta["session"].astype(str)
    elif group_col == "word":
        key = meta["word"].astype(str).str.lower().where(meta["word"].notna(), "__NA__")
    else:
        key = meta[group_col].astype(str)
    return pd.factorize(key.to_numpy())[0]


def cap_tokens_per_word(meta: pd.DataFrame, mask: np.ndarray, cap: int,
                        rng: np.random.Generator) -> np.ndarray:
    """(음소, 위치, 단어)별 구간 수를 cap으로 무작위 하향. 단어 정체 교란 통제.

    예: 어두 T의 64%가 'to', 어말 T/D는 it/that/and가 다수. 분류기가 '단어'를
    학습하는 것을 막기 위해 단어당 토큰 수를 제한한다.
    """
    keep = np.zeros(len(meta), dtype=bool)
    key = (meta["phoneme"].astype(str) + "|" + meta["position_in_word"].astype(str)
           + "|" + meta["word"].astype(str).str.lower()).to_numpy()
    idx = np.flatnonzero(mask & meta["word"].notna().to_numpy())
    for _, g in pd.Series(idx).groupby(key[idx]):
        v = g.to_numpy()
        if v.size > cap:
            v = rng.choice(v, cap, replace=False)
        keep[v] = True
    return keep


def select_pair_segments(phon, pos_all, a, b, position, cap=None, sub_all=None, rng=None):
    """쌍 (a,b)·위치의 구간 인덱스를 고른다. 반환 (idx, mix, n_per_class_available).

    sub_all(각 구간의 하위 위치, 예: 어중/어말)을 주면 두 클래스가 같은 하위 위치
    구성을 갖도록 하위 위치별 min(n_a, n_b)만큼 뽑는다(cap을 넘으면 비례 축소).
    (2026-10-09 검토 ③) 어중·어말을 합친 '비어두'에서 B는 93% 어중, D는 73% 어말이라
    구성 차이가 분류 단서가 될 수 있으므로 구성을 맞춘다.
    sub_all이 None이거나 하위 위치가 하나뿐이면 전체를 돌려주고(정렬된 인덱스),
    균형은 balanced_pair_decode가 맡는다(기존 동작과 동일).
    """
    pm = pos_all == position
    ia, ib = np.flatnonzero(pm & (phon == a)), np.flatnonzero(pm & (phon == b))
    n_plain = int(min(ia.size, ib.size))
    if sub_all is None or n_plain == 0:
        return np.sort(np.concatenate([ia, ib])), "", n_plain
    subs = np.unique(np.concatenate([sub_all[ia], sub_all[ib]]))
    if subs.size < 2:
        return np.sort(np.concatenate([ia, ib])), str(subs[0]), n_plain
    quota = {str(s_): int(min((sub_all[ia] == s_).sum(), (sub_all[ib] == s_).sum()))
             for s_ in subs}
    total = sum(quota.values())
    if cap and total > cap:
        quota = {s_: int(np.floor(q * cap / total)) for s_, q in quota.items()}
    rng = rng if rng is not None else np.random.default_rng(0)
    out = []
    for s_, q in quota.items():
        if q <= 0:
            continue
        for ii in (ia, ib):
            cand = ii[sub_all[ii] == s_]
            out.append(rng.choice(cand, q, replace=False))
    idx = np.sort(np.concatenate(out)) if out else np.array([], dtype=int)
    mix = ";".join(f"{s_}:{q}" for s_, q in quota.items())
    return idx, mix, int(sum(quota.values()))


def stage2_position_decode(paths: dict, X, meta, arrays=STAGE2_ARRAYS,
                           positions=POSITIONS, n_perm=200, perm_repeats=2,
                           n_repeats=10, n_pca=30, clf="lda", seed=20261009,
                           min_per_class=12, force: bool = False, tag="",
                           partitions=None, max_per_class=300,
                           match_n_across_positions: bool = False,
                           drop_intervocalic_flap: bool = False,
                           group_col=None, max_per_word=None,
                           match_subposition_col=None) -> pd.DataFrame:
    """T1. 어두/어중/어말마다 쌍별 균형 분류기. 어레이별로 분해.

    partitions로 파티션을 제한할 수 있다. 사전학습 RNN이 train 파티션으로
    학습됐으므로 train 기반 정렬은 디코더와 독립이 아니다. 주 분석은 pooled로
    하고 partitions='test' 행을 반드시 함께 보고한다.
    group_col: 'trial'/'block'/'session'/'word' -> 그룹 단위 CV(누출 통제).
    max_per_word: (음소, 위치, 단어)별 토큰 상한(단어 정체 교란 통제).
    match_subposition_col: 합친 위치(예: 비어두) 안에서 두 클래스의 하위 위치
        (어중/어말) 구성을 같게 맞추는 데 쓸 meta 열 이름.
    """
    out_path = paths["RESULT_DIR"] / f"stage2_position_decoding{tag}.csv"
    if out_path.exists() and not force:
        print("Stage 2 위치 분류: 기존 결과 재사용")
        return pd.read_csv(out_path)

    pmask, partitions_label = _partition_mask(meta, partitions)
    if drop_intervocalic_flap and "intervocalic" in meta.columns:
        flap = (meta["phoneme"].isin(FLAP_PAIR).to_numpy()
                & meta["intervocalic"].fillna(False).to_numpy())
        n_drop = int((pmask & flap).sum())
        pmask = pmask & ~flap
        print(f"  설탄음화 통제: 모음간 T/D {n_drop}구간 제외")
    if max_per_word:
        n_before = int(pmask.sum())
        pmask = pmask & cap_tokens_per_word(meta, pmask, int(max_per_word),
                                            np.random.default_rng(seed + 7))
        print(f"  단어 상한 {max_per_word}/(음소·위치·단어): {n_before} -> {int(pmask.sum())}구간")
    groups_all = None
    if group_col:
        if group_col == "word":
            pmask = pmask & meta["word"].notna().to_numpy()
        groups_all = segment_group_ids(meta, group_col)
        print(f"  그룹 CV: {group_col} ({np.unique(groups_all[pmask]).size}개 그룹)")
    phon = np.where(pmask, meta["phoneme"].to_numpy(), "__OUT__")
    pos_all = meta["position_in_word"].to_numpy()
    sub_all = (meta[match_subposition_col].astype(str).to_numpy()
               if match_subposition_col else None)
    specs = ([(a, b, "laryngeal") for a, b in LARYNGEAL_PAIRS]
             + [(a, b, "place") for a, b in PLACE_CONTROL_PAIRS])
    # 위치 간 n을 맞춘다: 각 쌍에 대해 세 위치의 최소 클래스 수의 최소값을 상한으로
    # 쓰면, 같은 쌍은 어느 위치에서도 동일한 시행 수로 학습된다. 위치 효과가
    # 단순히 시행 수 차이에서 오는 것이 아님을 보이는 데 필요하다.
    pair_cap = {}
    if match_n_across_positions:
        for a, b, _ in specs:
            per_pos = []
            for position in positions:
                _, _, n_av = select_pair_segments(phon, pos_all, a, b, position,
                                                  None, sub_all)
                per_pos.append(n_av)
            cap = min(per_pos)
            if max_per_class:
                cap = min(cap, int(max_per_class))
            pair_cap[(a, b)] = cap

    rows = []
    for array in arrays:
        Xa = X[:, array_columns(array)]
        for ip, position in enumerate(positions):
            for js, (a, b, contrast) in enumerate(specs):
                cap = pair_cap.get((a, b), max_per_class) if match_n_across_positions \
                    else max_per_class
                if match_n_across_positions and cap < min_per_class:
                    continue
                sel_rng = np.random.default_rng([seed, ip, js])
                m, mix, n_av = select_pair_segments(phon, pos_all, a, b, position,
                                                    cap, sub_all, sel_rng)
                if n_av < min_per_class or m.size < 2 * min_per_class \
                        or np.unique(phon[m]).size < 2:
                    continue
                res = balanced_pair_decode(Xa[m], phon[m], n_pca=n_pca, clf=clf,
                                           n_repeats=n_repeats, n_perm=n_perm,
                                           perm_repeats=perm_repeats, seed=seed,
                                           min_per_class=min_per_class,
                                           max_per_class=cap,
                                           groups=(groups_all[m] if groups_all is not None
                                                   else None))
                res["subpos_mix"] = mix
                res.update({"stage": "stage2", "test": "T1_position",
                            "n_matched_across_positions": bool(match_n_across_positions), "array": array,
                            "position": position, "pair": f"{a}/{b}",
                            "contrast": contrast, "manner": pair_manner(a, b),
                            "partitions": partitions_label,
                            "cv_group_col": group_col or "none",
                            "max_per_word": int(max_per_word or 0)})
                rows.append(res)
                print(f"  [T1|{array}|{position}] {a}/{b:<3} {contrast:<9} "
                      f"n={res['n_per_class']:<4} acc={res['acc']:.3f} "
                      f"d'={res['dprime']:.2f}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("Stage 2 위치 분류: 조건을 만족하는 쌍이 없습니다.")
        return df
    front = ["stage", "test", "array", "position", "pair", "contrast", "manner",
             "partitions", "n_per_class", "acc", "acc_sd", "dprime", "p_perm"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(out_path, index=False)
    print(f"Stage 2 위치 분류 저장: {out_path} ({len(df)}행)")
    return df


# --- T3: 양식(vocal vs nonvocal) ------------------------------------------

def stage2_modality_decode(paths: dict, X, meta, arrays=STAGE2_ARRAYS,
                           n_perm=200, perm_repeats=2, n_repeats=10, n_pca=30,
                           clf="lda", seed=20261009, min_per_class=12,
                           force: bool = False, tag="", session_filter=None,
                           per_table=None, max_per_class=300, partitions=None,
                           positions=None) -> pd.DataFrame:
    """T3. vocal vs nonvocal에서 같은 쌍을 같은 시행 수로 분류한다.

    session_filter: {'vocal': [세션...], 'nonvocal': [...]} 로 비교 세션을 고정한다.
      정렬 품질을 맞추려면 vocal 쪽도 RNN 학습에 쓰이지 않은 날(예: t12.2022.07.29,
      차용 입력층)을 써야 한다. 8월 발성일은 2차 비교군(품질 주의).
    per_table: session_per_table() 결과. 각 행에 세션별 PER을 붙인다.
    """
    out_path = paths["RESULT_DIR"] / f"stage2_modality_decoding{tag}.csv"
    if out_path.exists() and not force:
        print("Stage 2 양식 분류: 기존 결과 재사용")
        return pd.read_csv(out_path)

    sess_all = meta["session"].to_numpy()
    allowed, partitions_label = _partition_mask(meta, partitions)
    if positions is not None:
        plist = [positions] if isinstance(positions, str) else list(positions)
        allowed = allowed & meta["position_in_word"].isin(plist).to_numpy()
    if session_filter:
        keep = np.zeros(len(meta), dtype=bool)
        for m, ss in session_filter.items():
            sel = meta["modality"].to_numpy() == m
            if ss is not None:
                sel &= np.isin(sess_all, list(ss))
            keep |= sel
        allowed = allowed & keep
        print(f"  세션/파티션 제한 적용: {int(allowed.sum())}/{len(meta)} 구간")
    phon = np.where(allowed, meta["phoneme"].to_numpy(), "__OUT__")
    mod = meta["modality"].to_numpy()
    # PER은 (세션, 파티션) 단위다. 세션만으로 키를 잡으면 학습된 세션의 train 행
    # (PER~0.1%, 암기)이 test 행을 덮어써서 정렬 품질을 심하게 과대평가한다.
    # 실제로 쓰인 (세션, 파티션) 조합에 구간 수로 가중해 평균한다.
    part_all = (meta["partition"].to_numpy() if "partition" in meta.columns
                else np.array(["?"] * len(meta)))
    per_lookup = {}
    layer_lookup = {}
    if per_table is not None and len(per_table):
        for r in per_table.itertuples(index=False):
            per_lookup[(r.session, r.partition)] = float(r.per)
            layer_lookup[r.session] = getattr(r, "input_layer_from", "")
    mods = [m for m in ("vocal", "nonvocal") if (mod == m).sum() > 0]
    if len(mods) < 2:
        print(f"Stage 2 양식 분류: 양식이 {mods} 하나뿐이라 비교 불가.")
        return pd.DataFrame()

    specs = ([(a, b, "laryngeal") for a, b in LARYNGEAL_PAIRS]
             + [(a, b, "place") for a, b in PLACE_CONTROL_PAIRS])
    rng = np.random.default_rng(seed)
    rows = []
    for array in arrays:
        Xa = X[:, array_columns(array)]
        for a, b, contrast in specs:
            # 양식 x 클래스 4칸의 최소값으로 시행 수를 맞춘다
            cells = {(m, c): np.flatnonzero((mod == m) & (phon == c))
                     for m in mods for c in (a, b)}
            n_match = min(v.size for v in cells.values())
            if max_per_class:
                n_match = min(n_match, int(max_per_class))
            if n_match < min_per_class:
                continue
            for m in mods:
                idx = np.concatenate([rng.choice(cells[(m, c)], n_match, replace=False)
                                      for c in (a, b)])
                idx.sort()
                res = balanced_pair_decode(Xa[idx], phon[idx], n_pca=n_pca, clf=clf,
                                           n_repeats=n_repeats, n_perm=n_perm,
                                           perm_repeats=perm_repeats, seed=seed,
                                           min_per_class=min_per_class)
                used_sessions = sorted(set(sess_all[idx].tolist()))
                used_sp = pd.Series(list(zip(sess_all[idx], part_all[idx]))).value_counts()
                num = sum(per_lookup[k] * v for k, v in used_sp.items()
                          if k in per_lookup)
                den = sum(v for k, v in used_sp.items() if k in per_lookup)
                weighted_per = (num / den) if den else np.nan
                res.update({"stage": "stage2", "test": "T3_modality", "array": array,
                            "modality": m, "pair": f"{a}/{b}", "contrast": contrast,
                            "manner": pair_manner(a, b),
                            "n_matched_per_class": int(n_match),
                            "partitions": partitions_label,
                            "positions": ("all" if positions is None
                                          else ";".join(plist)),
                            "sessions": ";".join(used_sessions),
                            "mean_per": weighted_per,
                            "partitions_used": ";".join(
                                f"{a_}/{b_}:{int(c_)}"
                                for (a_, b_), c_ in sorted(used_sp.items())),
                            "input_layer_from": ";".join(
                                sorted({str(layer_lookup.get(s, "?"))
                                        for s in used_sessions}))})
                rows.append(res)
                print(f"  [T3|{array}|{m:<8}] {a}/{b:<3} {contrast:<9} "
                      f"n={n_match:<4} acc={res['acc']:.3f} d'={res['dprime']:.2f}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("Stage 2 양식 분류: 시행 수를 맞출 수 있는 쌍이 없습니다.")
        return df
    front = ["stage", "test", "array", "modality", "pair", "contrast", "manner",
             "n_matched_per_class", "acc", "acc_sd", "dprime", "p_perm",
             "mean_per", "sessions", "partitions_used", "input_layer_from"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(out_path, index=False)
    print(f"Stage 2 양식 분류 저장: {out_path} ({len(df)}행)")
    return df


# --- T2: 위치별 유표성 거리 -----------------------------------------------

def t3_pair_drops(X, meta, array="6v", n_rep=20, n_splits=5, n_repeats=2,
                  n_pca=30, clf="lda", max_per_class=300, min_per_class=12,
                  partitions="held_out", session_filter=None, positions=None,
                  seed=20261009) -> pd.DataFrame:
    """쌍별 발성 vs 무성 정확도와 그 낙폭(drop)에 재표집 CI.

    자연부류(natural class) 검정용: 후두 마디가 하나의 부류로 약해지는가
    (파열/마찰/파찰에 걸쳐 균일한 낙폭, 조음위치·비음은 낙폭 없음),
    아니면 조음 차이에 비례해 약해지는가.

    CI는 '균형 부분표집 재표집' 분포에서 낸다: 매 반복마다 (양식 x 클래스) 네 칸에서
    같은 수의 구간을 비복원 추출하고 교차검증을 다시 돌려 낙폭을 하나 얻는다.
    복원추출이 아니므로 같은 구간이 학습/시험 폴드에 동시에 들어가는 누수는 없다.
    따라서 이 구간은 '시행 표집 + 교차검증 난수'에 대한 불확실성이지
    참가자/세션 모집단에 대한 구간이 아니다.
    """
    cols = array_columns(array)
    Xa = np.asarray(X[:, cols], dtype=np.float64)
    allowed, partitions_label = _partition_mask(meta, partitions)
    if positions is not None:
        plist = [positions] if isinstance(positions, str) else list(positions)
        allowed = allowed & meta["position_in_word"].isin(plist).to_numpy()
    if session_filter:
        keep = np.zeros(len(meta), dtype=bool)
        for m, ss in session_filter.items():
            sel = meta["modality"].to_numpy() == m
            if ss is not None:
                sel &= np.isin(meta["session"].to_numpy(), list(ss))
            keep |= sel
        allowed = allowed & keep

    phon = np.where(allowed, meta["phoneme"].to_numpy(), "__OUT__")
    mod = meta["modality"].to_numpy()
    specs = ([(a, b, "laryngeal") for a, b in LARYNGEAL_PAIRS]
             + [(a, b, "place") for a, b in PLACE_CONTROL_PAIRS])

    rows = []
    for a, b, contrast in specs:
        cells = {(m, c): np.flatnonzero((mod == m) & (phon == c))
                 for m in ("vocal", "nonvocal") for c in (a, b)}
        if any(v.size == 0 for v in cells.values()):
            continue
        n_match = min(v.size for v in cells.values())
        if max_per_class:
            n_match = min(n_match, int(max_per_class))
        if n_match < min_per_class:
            continue

        accs = {"vocal": [], "nonvocal": []}
        for r in range(n_rep):
            rng = np.random.default_rng(seed + 1000 * r)
            for m in ("vocal", "nonvocal"):
                idx = np.concatenate([rng.choice(cells[(m, c)], n_match, replace=False)
                                      for c in (a, b)])
                idx.sort()
                Xr = Xa[idx]  # PCA는 폴드별 적합
                fold_acc, _ = _cv_accuracy(clf, Xr, phon[idx], n_splits, n_repeats,
                                           seed + r, n_pca=n_pca)
                accs[m].append(float(fold_acc.mean()))
        av = np.asarray(accs["vocal"])
        an = np.asarray(accs["nonvocal"])
        drop = av - an
        rows.append({
            "array": array, "pair": f"{a}/{b}", "contrast": contrast,
            "manner": ("nasal" if pair_manner(a, b) == "nasal"
                       else pair_manner(a, b)),
            "n_per_class": int(n_match), "n_rep": int(n_rep),
            "acc_vocal": float(av.mean()), "acc_nonvocal": float(an.mean()),
            "drop": float(drop.mean()),
            "drop_ci_lo": float(np.percentile(drop, 2.5)),
            "drop_ci_hi": float(np.percentile(drop, 97.5)),
            "drop_sd_across_reps": float(drop.std()),
            "partitions": partitions_label,
        })
        print(f"  [{array}] {a}/{b:<4} {contrast:<9} n={n_match:<4} "
              f"vocal={av.mean():.3f} nonvocal={an.mean():.3f} "
              f"drop={drop.mean():+.3f} "
              f"[{np.percentile(drop, 2.5):+.3f}, {np.percentile(drop, 97.5):+.3f}]",
              flush=True)
    return pd.DataFrame(rows)


def t3_natural_class_test(df: pd.DataFrame, n_perm=10000, seed=20261009) -> pd.DataFrame:
    """후두쌍 낙폭이 '하나의 자연부류'처럼 균일한지 검정.

    (a) 대립유형/조음방법별 평균 낙폭.
    (b) 후두쌍 낙폭의 SD가, 전체 쌍에서 같은 개수를 무작위로 뽑았을 때의 SD보다
        작은가(= 더 균일한가). p = 무작위 SD가 관측 SD 이하일 비율.
        proposal-v4.md 2.3 검정 1.
    """
    rng = np.random.default_rng(seed)
    out = []
    lar = df.loc[df["contrast"] == "laryngeal", "drop"].to_numpy()
    nasal = df.loc[df["manner"] == "nasal", "drop"].to_numpy()
    place = df.loc[(df["contrast"] == "place") & (df["manner"] != "nasal"),
                   "drop"].to_numpy()
    for name, v in [("laryngeal", lar), ("place (non-nasal)", place), ("nasal", nasal)]:
        out.append({"group": name, "n_pairs": int(v.size),
                    "mean_drop": (float(v.mean()) if v.size else np.nan),
                    "sd_drop": (float(v.std(ddof=1)) if v.size > 1 else np.nan)})
    # 후두쌍 내부에서 조음방법별 낙폭 (자연부류라면 균일해야 한다)
    for mn in ("stop", "fricative", "affricate"):
        v = df.loc[(df["contrast"] == "laryngeal") & (df["manner"] == mn),
                   "drop"].to_numpy()
        if v.size:
            out.append({"group": f"laryngeal:{mn}", "n_pairs": int(v.size),
                        "mean_drop": float(v.mean()),
                        "sd_drop": (float(v.std(ddof=1)) if v.size > 1 else np.nan)})
    summary = pd.DataFrame(out)

    allv = df["drop"].to_numpy()
    k = lar.size
    if k >= 2 and allv.size > k:
        obs_sd = float(lar.std(ddof=1))
        null = np.empty(n_perm)
        for i in range(n_perm):
            null[i] = rng.choice(allv, k, replace=False).std(ddof=1)
        p_uniform = float((1 + np.sum(null <= obs_sd)) / (n_perm + 1))
        summary.loc[len(summary)] = {
            "group": f"UNIFORMITY TEST (laryngeal SD vs random {k} of {allv.size})",
            "n_pairs": k, "mean_drop": obs_sd, "sd_drop": float(null.mean())}
        summary["uniformity_p"] = np.nan
        summary.loc[summary.index[-1], "uniformity_p"] = p_uniform
        summary.attrs["p_uniform"] = p_uniform
        summary.attrs["obs_sd"] = obs_sd
        summary.attrs["null_sd_mean"] = float(null.mean())
    return summary


def stage2_markedness_by_position(paths: dict, X, meta, arrays=STAGE2_ARRAYS,
                                  positions=POSITIONS, n_reps=300, seed=20261009,
                                  min_trials=8, force: bool = False, tag="",
                                  partitions=None) -> pd.DataFrame:
    """T2. 기준선 = 같은 위치의 후두쌍 자음 평균(자기 자신 제외)."""
    out_path = paths["RESULT_DIR"] / f"stage2_markedness{tag}.csv"
    if out_path.exists() and not force:
        print("Stage 2 유표성: 기존 결과 재사용")
        return pd.read_csv(out_path)

    pmask, partitions_label = _partition_mask(meta, partitions)
    phon = np.where(pmask, meta["phoneme"].to_numpy(), "__OUT__")
    pos_all = meta["position_in_word"].to_numpy()
    rows = []
    for array in arrays:
        Xa = X[:, array_columns(array)]
        for position in positions:
            pm = pos_all == position
            if pm.sum() == 0:
                continue
            sub_phon = phon[pm]
            targets = [p for p in PAIRED_OBSTRUENTS if (sub_phon == p).sum() >= min_trials]
            if len(targets) < 4:
                continue
            tab = cv_markedness(Xa[pm], sub_phon, targets, baseline=targets,
                                n_reps=n_reps, seed=seed)
            tab["array"] = array
            tab["position"] = position
            tab["baseline"] = "position_mean_LOO"
            tab["partitions"] = partitions_label
            rows.append(tab)
            vo = tab.loc[tab["series"] == "voiced", "d2_cv"].mean()
            vl = tab.loc[tab["series"] == "voiceless", "d2_cv"].mean()
            print(f"  [T2|{array}|{position}] 유성 d2={vo:.4f} 무성 d2={vl:.4f} "
                  f"차이={vo - vl:+.4f} (음소 {len(targets)}개)")
    if not rows:
        print("Stage 2 유표성: 음소 수가 부족합니다.")
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"Stage 2 유표성 저장: {out_path} ({len(df)}행)")
    return df


# --- Miller-Nicely 자질 전달량 --------------------------------------------

def relative_transmitted_information(conf: pd.DataFrame, values: dict) -> dict:
    """혼동행렬 -> 자질의 상대 전달정보 T = I(자극;반응) / H(자극).

    Miller & Nicely (1955) 방식: 음소를 자질 값으로 접어 넣은 뒤 상호정보량.
    """
    labels = [str(c) for c in conf.index]
    keys = sorted({values[p] for p in labels if p in values}, key=str)
    if len(keys) < 2:
        return {"n_feature_values": len(keys), "T_rel": np.nan,
                "H_stim": np.nan, "MI": np.nan, "n_trials": 0}
    ki = {k: i for i, k in enumerate(keys)}
    M = np.zeros((len(keys), len(keys)), dtype=np.float64)
    for i, s in enumerate(labels):
        if s not in values:
            continue
        for j, r in enumerate(labels):
            if r not in values:
                continue
            M[ki[values[s]], ki[values[r]]] += float(conf.iloc[i, j])
    total = M.sum()
    if total <= 0:
        return {"n_feature_values": len(keys), "T_rel": np.nan,
                "H_stim": np.nan, "MI": np.nan, "n_trials": 0}
    P = M / total
    px, py = P.sum(1), P.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = float(np.nansum(np.where(P > 0, P * np.log2(P / np.outer(px, py)), 0.0)))
        hx = float(-np.nansum(np.where(px > 0, px * np.log2(px), 0.0)))
    return {"n_feature_values": len(keys), "T_rel": (mi / hx if hx > 0 else np.nan),
            "H_stim": hx, "MI": mi, "n_trials": int(total)}


def stage2_feature_transmission(paths: dict, X, meta, arrays=STAGE2_ARRAYS,
                                features=("voicing", "place", "manner", "nasal"),
                                positions=("all",), n_pca=60, n_repeats=3,
                                seed=20261009, min_trials=10,
                                force: bool = False, tag="",
                                max_per_class=300) -> pd.DataFrame:
    """자음 다중분류 혼동행렬 -> 자질별 상대 전달정보 (Miller-Nicely 양식)."""
    out_path = paths["RESULT_DIR"] / f"stage2_feature_transmission{tag}.csv"
    conf_path = paths["RESULT_DIR"] / f"stage2_consonant_confusion{tag}.csv"
    if out_path.exists() and not force:
        print("Stage 2 자질 전달량: 기존 결과 재사용")
        return pd.read_csv(out_path)

    phon = meta["phoneme"].to_numpy()
    pos_all = meta["position_in_word"].to_numpy()
    maps = {f: feature_map(f) for f in features}
    rows, saved_conf = [], False
    for array in arrays:
        Xa = X[:, array_columns(array)]
        for position in positions:
            pm = np.ones(len(meta), bool) if position == "all" else (pos_all == position)
            counts = pd.Series(phon[pm]).value_counts()
            keep_ph = [p for p in CONSONANTS if counts.get(p, 0) >= min_trials]
            if len(keep_ph) < 4:
                continue
            m = pm & np.isin(phon, keep_ph)
            res = multiclass_decode(Xa[m], phon[m], n_pca=n_pca,
                                    n_repeats=n_repeats, seed=seed,
                                    max_per_class=max_per_class)
            conf = res["confusion"]
            if array == "all" and position == "all" and not saved_conf:
                conf.to_csv(conf_path)
                saved_conf = True
            for f in features:
                t = relative_transmitted_information(conf, maps[f])
                t.update({"stage": "stage2", "test": "feature_transmission",
                          "array": array, "position": position, "feature": f,
                          "n_phonemes": len(keep_ph), "n_per_class": res["n_per_class"],
                          "multiclass_acc": res["acc"], "chance": res["chance"]})
                rows.append(t)
                print(f"  [MN|{array}|{position}] {f:<8} T_rel={t['T_rel']:.3f} "
                      f"(acc={res['acc']:.3f}, {len(keep_ph)}음소)")
    df = pd.DataFrame(rows)
    if df.empty:
        print("Stage 2 자질 전달량: 자음 수가 부족합니다.")
        return df
    front = ["stage", "test", "array", "position", "feature", "T_rel", "MI",
             "H_stim", "n_phonemes", "n_per_class", "multiclass_acc"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(out_path, index=False)
    print(f"Stage 2 자질 전달량 저장: {out_path} ({len(df)}행)")
    return df


def stage2_run(paths: dict, align_df=None, synthetic_ok: bool = True,  # noqa: PLR0913
               n_perm=200, perm_repeats=2, n_repeats=10, tag="",
               force: bool = False, pre_bins: int = PRE_BINS,
               features_from: str = "auto", score_min=SCORE_MIN,
               arrays=STAGE2_ARRAYS, session_filter=None) -> dict:
    """Stage 2 전체. 정렬 CSV가 있으면 실제 특징, 없으면 합성 특징으로 끝까지 돈다."""
    if align_df is None:
        align_df = load_alignments(paths["ALIGN_DIR"])
    source = "real_alignments"
    if align_df.empty:
        if not synthetic_ok:
            raise FileNotFoundError("정렬 CSV가 없습니다.")
        print("실제 정렬이 없으므로 합성 정렬로 배관을 점검합니다.")
        align_df = make_synthetic_alignments(paths["CACHE_DIR"] / "synthetic_alignments.csv")
        source = "synthetic_alignments"

    n_before = len(align_df)
    align_df, score_report = apply_score_filter(align_df, score_min)
    if not score_report.empty:
        score_report.to_csv(paths["RESULT_DIR"] / f"stage2_score_filter{tag}.csv",
                            index=False)

    have_comp = (features_from != "synthetic") and any(
        (paths["COMP_DIR"] / p).exists()
        for p in ("train", "test", "competitionHoldOut"))
    if features_from == "real" and not have_comp:
        raise FileNotFoundError(f"competitionData가 없습니다: {paths['COMP_DIR']}")
    if have_comp:
        X, X_pre, meta = stage2_segment_features(paths, align_df, pre_bins=pre_bins)
        feature_source = "competitionData"
        if X.shape[0] == 0:
            print("실제 특징을 못 만들었습니다 -> 전 합성 특징으로 대체")
            X, X_pre, meta = stage2_synthetic_features(align_df)
            feature_source = "synthetic_features"
    elif features_from == "synthetic":
        print("features_from='synthetic' -> 효과를 심은 합성 특징(분석 논리 점검)")
        X, X_pre, meta = stage2_synthetic_features(align_df)
        feature_source = "synthetic_features"
    else:
        print(f"competitionData 없음 ({paths['COMP_DIR']}) -> 전 합성 특징")
        X, X_pre, meta = stage2_synthetic_features(align_df)
        feature_source = "synthetic_features"

    assert X.shape[1] == N_CHAN * len(FEATURE_SETS), X.shape
    assert X.shape[0] == len(meta), (X.shape, len(meta))
    sessions = sorted(meta["session"].unique().tolist())
    print(f"\nStage 2 입력: {X.shape[0]}구간, 정렬={source}, 특징={feature_source}")
    print(f"  포함 세션({len(sessions)}): {', '.join(sessions)}")
    print(f"  양식: {meta['modality'].value_counts().to_dict()}")
    print(f"  위치: {meta['position_in_word'].value_counts().to_dict()}")

    out = {"source": source, "feature_source": feature_source,
           "sessions": sessions, "n_segments": int(X.shape[0]),
           "meta": meta, "X": X, "X_pre": X_pre}
    out["score_report"] = score_report
    per_table = session_per_table(paths["ALIGN_DIR"], verbose=False)
    out["per_table"] = per_table
    out["position"] = stage2_position_decode(
        paths, X, meta, arrays=arrays, n_perm=n_perm, perm_repeats=perm_repeats,
        n_repeats=n_repeats, tag=tag, force=force, partitions=None)
    # test 전용 행(디코더 학습에 쓰이지 않은 파티션)을 항상 같이 낸다
    if "test" in set(meta["partition"]) and set(meta["partition"]) != {"test"}:
        out["position_test_only"] = stage2_position_decode(
            paths, X, meta, arrays=arrays, n_perm=0, n_repeats=n_repeats,
            tag=f"{tag}_testonly", force=force, partitions="test")
    else:
        out["position_test_only"] = out["position"]
    out["modality"] = stage2_modality_decode(
        paths, X, meta, arrays=arrays, n_perm=n_perm, perm_repeats=perm_repeats,
        n_repeats=n_repeats, tag=tag, force=force, session_filter=session_filter,
        per_table=per_table)
    out["markedness"] = stage2_markedness_by_position(paths, X, meta, arrays=arrays,
                                                      tag=tag, force=force)
    out["transmission"] = stage2_feature_transmission(paths, X, meta, arrays=arrays,
                                                      tag=tag, force=force)
    out["manner_stage2"] = manner_aggregate(out["position"],
                                            group_cols=("array", "position"))

    manifest = {
        "alignment_source": source, "feature_source": feature_source,
        "sessions_included": sessions, "n_segments": int(X.shape[0]),
        "modality_counts": {k: int(v) for k, v in meta["modality"].value_counts().items()},
        "position_counts": {k: int(v) for k, v in meta["position_in_word"].value_counts().items()},
        "n_phonemes": int(meta["phoneme"].nunique()),
        "pre_bins": int(pre_bins),
        "arrays": list(arrays),
        "score_min": (None if score_min is None else float(score_min)),
        "n_segments_before_score_filter": int(n_before),
        "n_segments_after_score_filter": int(len(align_df)),
        "window_source": (str(meta["window_source"].iloc[0])
                          if "window_source" in meta.columns and len(meta) else None),
        "median_window_bins": (float(meta["win_bins"].median())
                               if "win_bins" in meta.columns and len(meta) else None),
        "caveat": ("CTC는 peaky하므로 음소 경계는 ~80 ms 수준에서 근사다. "
                   "음성 자질표는 verified=False, 사전등록 전 확정 금지."),
    }
    (paths["RESULT_DIR"] / f"stage2_manifest{tag}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ===========================================================================
# 12. 그림
# ===========================================================================
# 그림 라벨은 영어로 쓴다(Colab/로컬 matplotlib에 한글 폰트가 없는 경우가 많다).

def _save_fig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"그림 저장: {path}")
    return path


def fig_stage1_pairs(df: pd.DataFrame, out_path, window="go_all"):
    """쌍별 정확도 막대 (후두쌍 vs 조음위치쌍) x 어레이."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    sub = df[df["window"] == window].copy()
    if sub.empty:
        sub = df.copy()
    arrays = [a for a in ("6v", "44", "all") if a in set(sub["array"])]
    order = ([f"{a}/{b}" for a, b in LARYNGEAL_PAIRS]
             + [f"{a}/{b}" for a, b in PLACE_CONTROL_PAIRS])
    order = [p for p in order if p in set(sub["pair"])]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2),
                             gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    w = 0.8 / max(len(arrays), 1)
    x = np.arange(len(order))
    colors = {"6v": "#2b6cb0", "44": "#d69e2e", "all": "#718096"}
    for i, array in enumerate(arrays):
        s = (sub[sub["array"] == array].groupby("pair")[["acc", "acc_sd"]]
             .mean().reindex(order))
        ax.bar(x + i * w - 0.4 + w / 2, s["acc"].to_numpy(), w,
               yerr=s["acc_sd"].to_numpy(), capsize=1.5,
               label=f"area {array}", color=colors.get(array, None), alpha=0.9)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    n_lar = sum(1 for p in order if p in {f"{a}/{b}" for a, b in LARYNGEAL_PAIRS})
    if 0 < n_lar < len(order):
        ax.axvline(n_lar - 0.5, color="k", lw=1.2)
        ax.text(n_lar / 2 - 0.5, 1.005, "laryngeal (voicing)", ha="center", fontsize=9)
        ax.text((n_lar + len(order)) / 2 - 0.5, 1.005, "place control",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("balanced CV accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(f"T12 isolated phonemes — minimal-pair decoding ({window})", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    agg = sub.groupby(["array", "contrast"])["acc"].mean().unstack()
    agg = agg.reindex(arrays)
    xs = np.arange(len(agg.index))
    for i, contrast in enumerate([c for c in ("laryngeal", "place") if c in agg.columns]):
        ax.bar(xs + i * 0.35 - 0.175, agg[contrast].to_numpy(), 0.35,
               label=contrast, color=("#c53030" if contrast == "laryngeal" else "#2f855a"))
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"area {a}" for a in agg.index], fontsize=9)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("mean accuracy")
    ax.set_title("voicing vs place, by array", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_fig(fig, out_path)


def fig_stage2_position(df: pd.DataFrame, out_path, array="6v"):
    """T1: 위치별 유성성/조음위치 분리도 + 두 이론의 예측 방향."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    sub = df[df["array"] == array]
    if sub.empty:
        sub = df
    agg = (sub.groupby(["position", "contrast"])["acc"]
           .agg(["mean", "sem", "count"]).reset_index())
    positions = [p for p in POSITIONS if p in set(agg["position"])]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(positions))
    for i, contrast in enumerate([c for c in ("laryngeal", "place")
                                  if c in set(agg["contrast"])]):
        s = agg[agg["contrast"] == contrast].set_index("position").reindex(positions)
        ax.errorbar(x + i * 0.04, s["mean"].to_numpy(), yerr=s["sem"].to_numpy(),
                    marker="o", capsize=3, lw=2,
                    label=f"{contrast} (n pairs={int(s['count'].max())})",
                    color=("#c53030" if contrast == "laryngeal" else "#2f855a"))
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_xlabel("position in word")
    ax.set_ylabel("balanced CV accuracy")
    ax.set_title(f"T1 — laryngeal separability by position (area {array})", fontsize=10)
    ax.text(0.02, 0.02,
            "[voice] predicts peak at MEDIAL\n[spread glottis] predicts peak at INITIAL",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox={"facecolor": "#f7fafc", "edgecolor": "#cbd5e0"})
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return _save_fig(fig, out_path)


def fig_markedness(df: pd.DataFrame, out_path, array="6v", group_col="array"):
    """T2: 음소별 기준선 거리, 유성/무성 색 구분."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    sub = df[df[group_col] == array] if group_col in df.columns else df
    if "position" in sub.columns and sub["position"].nunique() > 1:
        panels = [p for p in POSITIONS if p in set(sub["position"])]
    else:
        panels = [None]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.0),
                             squeeze=False)
    for ax, panel in zip(axes[0], panels):
        s = sub if panel is None else sub[sub["position"] == panel]
        s = s.sort_values("d2_cv", ascending=False)
        colors = ["#c53030" if x == "voiced" else "#2b6cb0" for x in s["series"]]
        ax.bar(np.arange(len(s)), s["d2_cv"].to_numpy(), color=colors)
        ax.set_xticks(np.arange(len(s)))
        ax.set_xticklabels(s["phoneme"].tolist(), rotation=45, fontsize=8)
        ax.axhline(0, color="k", lw=1)
        ax.set_ylabel("cross-validated squared distance\nfrom unmarked baseline")
        ax.set_title(f"T2 markedness — area {array}"
                     + (f", {panel}" if panel else ""), fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#c53030"),
               plt.Rectangle((0, 0), 1, 1, color="#2b6cb0")]
    axes[0][0].legend(handles, ["voiced", "voiceless"], fontsize=8)
    fig.tight_layout()
    return _save_fig(fig, out_path)


def fig_feature_transmission(df: pd.DataFrame, out_path):
    """Miller-Nicely 양식의 자질별 상대 전달정보."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    sub = df[df["position"] == "all"] if "position" in df.columns else df
    piv = sub.pivot_table(index="feature", columns="array", values="T_rel")
    order = [f for f in ("place", "manner", "nasal", "voicing") if f in piv.index]
    piv = piv.reindex(order)
    arrays = [a for a in ("6v", "44", "all") if a in piv.columns]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(piv.index))
    w = 0.8 / max(len(arrays), 1)
    for i, a in enumerate(arrays):
        ax.bar(x + i * w - 0.4 + w / 2, piv[a].to_numpy(), w, label=f"area {a}")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index.tolist())
    ax.set_ylabel("relative transmitted information T")
    ax.set_title("Miller–Nicely style feature transmission", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_fig(fig, out_path)


# ===========================================================================
# 13. 요약
# ===========================================================================

def _md(df) -> str:
    """표를 마크다운으로. tabulate가 없으면 고정폭 코드블록으로 되돌린다."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def write_summary(paths: dict, extra: dict | None = None) -> Path:
    """results/의 CSV를 읽어 압축 표를 출력하고 SUMMARY.md를 쓴다."""
    R = paths["RESULT_DIR"]

    def _read(name):
        f = R / name
        return pd.read_csv(f) if f.exists() else None

    lines = ["# T12 후두 자질 분석 요약", "",
             f"생성 시각: {pd.Timestamp.now():%Y-%m-%d %H:%M}", "",
             "경로: " + str(R), "",
             "## 반드시 함께 읽을 유보 조항", "",
             "1. 음성 자질표는 `verified=False`. 사전등록 전 확정 금지.",
             "   `spread_glottis` 열은 표준 기술이 아니라 laryngeal realism 쪽 가설 라벨이다.",
             "2. 강제 정렬은 CTC 기반이고 CTC는 peaky하므로 음소 경계는 ~80 ms 수준의",
             "   근사다. 창 선택(`seg` 160 ms vs `rnn_center` 80 ms)에 따라 수치가 바뀐다.",
             "3. 사전학습 RNN은 competitionData의 train 파티션으로 학습됐다. 그 세션의",
             "   정렬 PER은 0.1% 수준이고 test는 15~34% 수준이다. 주 보고값은 test 파티션.",
             "4. `input_layer_from != session` 인 세션은 입력층을 차용했으므로 정렬이 근사다.",
             "   T3(양식)는 양쪽의 정렬 품질을 맞춰야 하며, 각 행에 `mean_per`을 붙였다.",
             "5. 분류기 전처리의 PCA·스케일러는 (2026-10-09 수정 후) 교차검증 학습 폴드 안에서만 "
             "적합한다. 이전 버전의 전역 PCA 수치는 results_pre_pcafix/ 에 보존.",
             "   (순열검정의 귀무가설은 라벨 교환이므로 유효하지만, 정확도 추정에는",
             "   약한 transduction이 들어 있다).",
             "6. tuningTasks 고립 음소는 모두 어두 'C + AA' 발성 조건이므로 T1(위치)과",
             "   T3(양식)에 대해서는 말할 수 없다.", ""]
    printed = []

    inv = _read("tuning_inventory.csv")
    if inv is not None:
        lines += ["## tuningTasks 인벤토리", "",
                  _md(inv[["file", "task", "modality", "n_trials", "n_blocks",
                           "n_cues", "go_bins_mean"]]), ""]
    cinv = _read("competition_inventory.csv")
    if cinv is not None and not cinv.empty:
        agg = (cinv.groupby(["partition", "modality"])
               .agg(n_sessions=("session", "nunique"), n_trials=("n_trials", "sum"),
                    total_sec=("total_sec", "sum")).reset_index())
        lines += ["## competitionData 인벤토리", "",
                  _md(agg), ""]

    p39 = _read("stage1_phoneme39.csv")
    if p39 is not None:
        tbl = p39[["array", "acc", "chance", "n_per_class"]].round(4)
        lines += ["## Stage 1 — 39음소 분류 (전건전성)", "",
                  _md(tbl), ""]
        printed.append(("39-way (area all)",
                        f"{float(p39.loc[p39['array'] == 'all', 'acc'].iloc[0]):.4f}"
                        if (p39["array"] == "all").any() else "n/a"))

    s1 = _read("stage1_pair_decoding.csv")
    if s1 is None:
        s1 = _read("stage1_window_scan.csv")
    if s1 is not None and not s1.empty:
        prim = s1[s1["window"] == "go_all"] if "window" in s1.columns else s1
        agg = (prim.groupby(["array", "contrast"])["acc"].mean()
               .unstack().round(4).reset_index())
        lines += ["## Stage 1 — 최소대립쌍 분리도 (go 구간 전체)", "",
                  "어레이 x 대립 유형 평균 정확도:", "",
                  _md(agg), "",
                  "쌍별:", "",
                  _md(prim[[c for c in ("array", "pair", "contrast", "manner",
                                        "n_per_class", "acc", "dprime", "p_perm")
                            if c in prim.columns]].round(4)), ""]
        for arr in ("6v", "44", "all"):
            a = prim[prim["array"] == arr]
            if a.empty:
                continue
            printed.append((f"area {arr}: voicing / place",
                            f"{a.loc[a['contrast'] == 'laryngeal', 'acc'].mean():.3f}"
                            f" / {a.loc[a['contrast'] == 'place', 'acc'].mean():.3f}"))

    onset = _read("stage1_onset_window_scan.csv")
    if onset is not None and not onset.empty:
        key = (["session_set"] if "session_set" in onset.columns else [])
        agg = (onset[onset["array"] == "6v"]
               .pivot_table(index=key + ["window"], columns="contrast",
                            values="acc").round(4).reset_index())
        lines += ["## Stage 1 — 창 비교 (area 6v, 지속시간 교란 점검)", "",
                  "`onset_*` 창은 audioEnvelope로 추정한 발화 onset 기준 고정 길이이므로",
                  "모든 쌍에 같은 길이가 적용된다(조음방법별 지속시간 차이 통제).", "",
                  _md(agg), ""]
    man = _read("stage1_manner_aggregate.csv")
    if man is not None and not man.empty:
        sub = man[man["array"] == "6v"] if "array" in man.columns else man
        lines += ["## Stage 1 — 후두쌍 조음방법 집계 (area 6v)", "",
                  _md(sub.round(4)), ""]

    mk = _read("stage1_markedness_test.csv")
    if mk is not None:
        lines += ["## Stage 1 — T2 유표성", "",
                  _md(mk.round(5)), ""]

    nbp = R / "stage2_n_by_position.csv"
    if nbp.exists():
        lines += ["## Stage 2 — 자음 x 위치 구간 수", "",
                  "S/Z, SH/ZH의 어두 칸이 거의 비어 있다. 영어에서 Z, ZH는 어두에",
                  "거의 나타나지 않으므로 이 두 쌍은 **구조적으로** T1 어두 조건에",
                  "기여할 수 없다. T1은 P/B, T/D, K/G, F/V, TH/DH, CH/JH 6쌍에 의존한다.",
                  "", _md(pd.read_csv(nbp)), ""]

    for name, title in [("stage2_position_decoding_all_pairs_pooled.csv",
                         "Stage 2 — T1 위치 (n 맞춤, pooled)"),
                        ("stage2_position_decoding_all_pairs_heldout.csv",
                         "Stage 2 — T1 위치 (n 맞춤, held-out)"),
                        ("stage2_position_decoding_all_pairs_testonly.csv",
                         "Stage 2 — T1 위치 (n 맞춤, test 전용)"),
                        ("stage2_position_decoding_flapctrl_pooled.csv",
                         "Stage 2 — T1 위치 (설탄음화 통제: 모음간 T/D 제외)"),
                        ("stage2_position_decoding_arraysplit.csv",
                         "Stage 2 — T1 위치 (6v_sup vs 6v_inf)"),
                        ("stage2_T1_interaction.csv",
                         "Stage 2 — T1 위치 x 대립유형 상호작용 (부트스트랩 95% CI)"),
                        ("stage2_t3_comparators.csv",
                         "Stage 2 — T3 양식 (비교군별, 정렬 품질·날짜 맞춤)"),
                        ("stage2_t3_gaps.csv",
                         "Stage 2 — T3 격차와 DoD (부트스트랩 95% CI)"),
                        ("stage2_t3_pairs.csv",
                         "Stage 2 — T3 쌍별 낙폭 (발성 - 무성, 재표집 95% CI)"),
                        ("stage2_t3_natural_class.csv",
                         "Stage 2 — T3 자연부류 검정 (후두 마디가 한 부류로 약해지는가)"),
                        ("stage2_markedness.csv", "Stage 2 — T2 위치별 유표성"),
                        ("stage2_feature_transmission.csv",
                         "Stage 2 — Miller-Nicely 자질 전달량"),
                        ("stage2_arm_comparison.csv",
                         "Stage 2 — 팔 비교 (창/필터/합성 점검)"),
                        ("stage2_score_filter.csv",
                         "Stage 2 — score 필터 탈락 집계")]:
        df = _read(name)
        if df is None or df.empty:
            continue
        if ("arm" in df.columns or "frac_dropped" in df.columns
                or "gap_place_minus_laryngeal" in df.columns
                or "drop_ci_lo" in df.columns or "uniformity_p" in df.columns):
            cols = [c for c in df.columns
                    if c not in ("stage", "test", "n_rep", "partitions",
                                 "drop_sd_across_reps")]
            body = _md(df[cols].round(4))
            if "uniformity_p" in df.columns:
                body += ("\n\n균일성 검정: 후두쌍 낙폭의 표준편차가 전체 쌍에서 같은 수를 "
                         "무작위로 뽑았을 때의 표준편차보다 작은가(= 하나의 자연부류처럼 "
                         "균일하게 약해지는가). p는 무작위 SD가 관측 SD 이하일 비율이므로 "
                         "작을수록 '균일하다'는 증거다.")
        elif "position" in df.columns and "contrast" in df.columns:
            body = _md(df.groupby(["array", "position", "contrast"])["acc"]
                       .mean().unstack().round(4).reset_index())
        elif "modality" in df.columns and "contrast" in df.columns:
            idx = (["comparator"] if "comparator" in df.columns else []) \
                + ["array", "modality"]
            keep = ["acc"] + (["mean_per"] if "mean_per" in df.columns else [])
            body = _md(df.pivot_table(index=idx, columns="contrast", values=keep)
                       .round(4).reset_index())
        elif "feature" in df.columns:
            body = _md(df.pivot_table(index="feature", columns="array",
                                      values="T_rel").round(4).reset_index())
        else:
            body = _md(df.groupby(["array", "position", "series"])["d2_cv"]
                       .mean().unstack().round(5).reset_index())
        lines += [f"## {title}", "", body, ""]

    perf = R / "alignment_per.csv"
    if perf.exists():
        lines += ["## 정렬 품질 (세션별 PER, 음소 수 가중)", "",
                  _md(pd.read_csv(perf)), ""]

    man = R / "stage2_manifest.json"
    if man.exists():
        lines += ["## Stage 2 실행 정보", "", "```json",
                  man.read_text(encoding="utf-8"), "```", ""]
    if extra:
        lines += ["## 추가 메모", "", "```json",
                  json.dumps(extra, indent=2, ensure_ascii=False), "```", ""]

    figs = sorted(paths["FIG_DIR"].glob("*.png"))
    if figs:
        lines += ["## 그림", ""] + [f"- `{f.name}`" for f in figs] + [""]

    out = R / "SUMMARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 62)
    print("핵심 결과")
    print("=" * 62)
    for k, v in printed:
        print(f"  {k:<28} {v}")
    print("=" * 62)
    print(f"요약 저장: {out}")
    return out
