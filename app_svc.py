import re, os, hashlib
import requests, json, torch, shutil, argparse, base64, http.client
import threading
import soundfile
import numpy as np
from functools import wraps
from difflib import SequenceMatcher
from urllib.parse import urlparse

progress_local = threading.local()

import tqdm
class GradioTqdm(tqdm.tqdm):
    def update(self, n=1):
        super().update(n)
        if hasattr(progress_local, 'progress') and progress_local.progress is not None:
            if self.total and self.total > 0:
                pct = int(self.n / self.total * 100)
                if pct != getattr(self, '_last_pct', -1) and pct % 5 == 0:
                    self._last_pct = pct
                    progress_local.progress(0.4 + (pct / 100.0) * 0.2, desc=f"分离人声 {pct}%")

tqdm.tqdm = GradioTqdm
import tqdm.auto
tqdm.auto.tqdm = GradioTqdm

SVC_API_BASE = "http://127.0.0.1:7777"
TIMEOUT = 600
SVC_FUSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SVC-Fusion", "amd"))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_VERSION = "rvcsvc-svc-v2"
OUTPUT_BITRATE = os.environ.get("RVCSVC_OUTPUT_BITRATE", "320k")
CACHE_MAX_FILES = max(10, int(os.environ.get("RVCSVC_CACHE_MAX_FILES", "200")))
CONVERT_LOCK = threading.RLock()
_SOURCE_HASH_CACHE = {}

def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _serialized(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with CONVERT_LOCK:
            return fn(*args, **kwargs)
    return wrapped

def _sha256_file(path):
    resolved = os.path.abspath(path)
    stat = os.stat(resolved)
    key = (resolved, stat.st_size, stat.st_mtime_ns)
    cached = _SOURCE_HASH_CACHE.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with open(resolved, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _SOURCE_HASH_CACHE.clear()
    _SOURCE_HASH_CACHE[key] = value
    return value

def _svc_model_fingerprint(model):
    requested = str(model or "")
    match = re.search(r'\(([^)]+)\)\s*$', requested)
    basename = match.group(1) if match else os.path.basename(requested)
    assets = []
    if os.path.isdir(SVC_FUSION_ROOT):
        for current, _, files in os.walk(SVC_FUSION_ROOT):
            if basename.lower() not in current.lower() and basename.lower() not in " ".join(files).lower():
                continue
            for filename in files:
                if filename.lower().endswith((".pt", ".pth", ".ckpt", ".onnx", ".yaml", ".json")):
                    path = os.path.join(current, filename)
                    stat = os.stat(path)
                    assets.append((os.path.relpath(path, SVC_FUSION_ROOT), stat.st_size, stat.st_mtime_ns))
    payload = json.dumps(sorted(assets), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16] if assets else "missing"

def _cache_areas():
    return {"results": os.path.join(SCRIPT_DIR, "temp"), "separation": os.path.join(SCRIPT_DIR, "output")}

def _cache_files(root):
    if not os.path.isdir(root):
        return []
    return [os.path.join(current, name) for current, _, names in os.walk(root) for name in names]

def cache_info():
    areas = {}
    for name, root in _cache_areas().items():
        files = [path for path in _cache_files(root) if os.path.isfile(path)]
        areas[name] = {"files": len(files), "bytes": sum(os.path.getsize(path) for path in files)}
    return {"service": "RVCSVC-API SVC", "areas": areas,
            "total_files": sum(v["files"] for v in areas.values()),
            "total_bytes": sum(v["bytes"] for v in areas.values())}

@_serialized
def clear_cache(scope="all"):
    requested = str(scope).lower()
    selected = _cache_areas() if requested == "all" else {k: v for k, v in _cache_areas().items() if k == requested}
    deleted = freed = 0
    for root in selected.values():
        for path in _cache_files(root):
            try:
                size = os.path.getsize(path)
                os.remove(path)
                deleted += 1
                freed += size
            except OSError:
                pass
        if os.path.isdir(root):
            for current, dirs, _ in os.walk(root, topdown=False):
                for directory in dirs:
                    try:
                        os.rmdir(os.path.join(current, directory))
                    except OSError:
                        pass
    return {"deleted_files": deleted, "freed_bytes": freed, "scope": requested}

def _trim_cache():
    areas = _cache_areas()
    results = sorted((p for p in _cache_files(areas["results"]) if os.path.isfile(p)), key=os.path.getmtime, reverse=True)
    for path in results[CACHE_MAX_FILES:]:
        try:
            os.remove(path)
        except OSError:
            pass
    root = areas["separation"]
    jobs = [os.path.join(group.path, child.name) for group in os.scandir(root) if group.is_dir() for child in os.scandir(group.path) if child.is_dir()] if os.path.isdir(root) else []
    jobs.sort(key=lambda path: max((os.path.getmtime(p) for p in _cache_files(path)), default=0), reverse=True)
    for path in jobs[CACHE_MAX_FILES:]:
        shutil.rmtree(path, ignore_errors=True)

available_models = []
available_configs = []
available_diffusion_models = []
available_diffusion_configs = []
current_speaker_id = "speaker0"
current_model_type_index = -1
loaded_speakers = []

parser = argparse.ArgumentParser()
parser.add_argument('--is_nohalf', action='store_true')
parser.add_argument('--dml', action='store_true', help='(已废弃) DirectML 已被原生 AMD ROCm 取代')
a = parser.parse_args()

if a.dml:
    print("[ROCm] --dml 参数已废弃，本版本使用原生 AMD ROCm，忽略该参数")

is_half = not a.is_nohalf
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f"[ROCm] PyTorch {torch.__version__}, HIP {torch.version.hip}, GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[ROCm] 未检测到 AMD ROCm GPU，回退到 CPU（速度会很慢）")

import time as _time

_gradio_info_cache = None

def _call_gradio_sse(endpoint, data=None, timeout=TIMEOUT, progress_callback=None):
    if data is None:
        data = []
    parsed = urlparse(SVC_API_BASE)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 7777
    try:
        conn = http.client.HTTPConnection(host, port, timeout=30)
        conn.request("POST", f"/call/{endpoint}", json.dumps({"data": data}), {"Content-Type": "application/json"})
        r = conn.getresponse()
        body = r.read().decode()
        if r.status != 200:
            return None, f"HTTP {r.status}: {body[:200]}"
        submit = json.loads(body)
        event_id = submit.get("event_id")
        if not event_id:
            return None, f"no event_id: {body[:200]}"

        deadline = _time.time() + timeout
        svc_started = _time.time()
        svc_last_update = _time.time()
        while _time.time() < deadline:
            _time.sleep(1)
            conn2 = http.client.HTTPConnection(host, port, timeout=30)
            conn2.request("GET", f"/call/{endpoint}/{event_id}")
            r2 = conn2.getresponse()
            raw = r2.read().decode()
            evt = ""
            for line in raw.split("\n"):
                line = line.strip()
                if line.startswith("event:"):
                    evt = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    payload = line.split(":", 1)[1].strip()
                    if evt == "complete":
                        try:
                            return json.loads(payload), None
                        except json.JSONDecodeError:
                            return payload, None
                    elif evt == "error":
                        return None, f"Gradio error: {payload}"
                    elif evt == "heartbeat":
                        if progress_callback and _time.time() - svc_last_update >= 5:
                            elapsed = _time.time() - svc_started
                            progress_callback(elapsed)
                            svc_last_update = _time.time()
            if "event: complete" in raw or "event: error" in raw:
                break
            if progress_callback and _time.time() - svc_last_update >= 5:
                elapsed = _time.time() - svc_started
                progress_callback(elapsed)
                svc_last_update = _time.time()
        return None, "Gradio call timeout"
    except ConnectionRefusedError:
        return None, f"Cannot connect to SVC-Fusion ({SVC_API_BASE})"
    except Exception as e:
        return None, str(e)

def _upload_file_to_gradio(local_path):
    with open(local_path, "rb") as f:
        fname = os.path.basename(local_path)
        resp = requests.post(
            f"{SVC_API_BASE}/upload",
            files={"files": (fname, f, "application/octet-stream")},
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list) and result:
            return result[0]
        if isinstance(result, dict):
            return result.get("path") or result.get("name")
        return result

def _get_info():
    global _gradio_info_cache
    if _gradio_info_cache:
        return _gradio_info_cache
    try:
        r = requests.get(f"{SVC_API_BASE}/info", timeout=10)
        r.raise_for_status()
        _gradio_info_cache = r.json()
        return _gradio_info_cache
    except Exception as e:
        print(f"Warning: cannot get /info: {e}")
        return {}

def _get_speakers():
    result, err = _call_gradio_sse("get_spk_md", [], timeout=15)
    if err:
        return []
    if isinstance(result, list) and result:
        md_text = result[0] if result[0] else ""
        speakers = []
        for line in md_text.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                speakers.append(line[2:].strip())
        return speakers
    return []

def _scan_models_from_fs():
    global available_models, available_configs, available_diffusion_models, available_diffusion_configs
    available_models = []
    available_configs = []
    available_diffusion_models = []
    available_diffusion_configs = []

    scan_dirs = []

    models_dir = os.path.join(SVC_FUSION_ROOT, "models")
    if os.path.isdir(models_dir):
        for name in os.listdir(models_dir):
            sub = os.path.join(models_dir, name)
            if os.path.isdir(sub):
                scan_dirs.append(sub)

    archive_dir = os.path.join(SVC_FUSION_ROOT, "archive")
    if os.path.isdir(archive_dir):
        for name in os.listdir(archive_dir):
            sub = os.path.join(archive_dir, name)
            if os.path.isdir(sub):
                scan_dirs.append(sub)

    for d in scan_dirs:
        config_yaml = os.path.join(d, "config.yaml")
        config_json = os.path.join(d, "config.json")
        has_config = os.path.isfile(config_yaml) or os.path.isfile(config_json)
        if not has_config:
            continue

        pt_files = []
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if os.path.isfile(fp) and f.endswith(".pt") and f != "model_0.pt":
                pt_files.append(f)

        if not pt_files:
            continue

        available_models.append(d)

        def _sort_key(fname):
            num = fname.replace("model_", "").replace(".pt", "")
            return int(num) if num.isdigit() else 0

        best = sorted(pt_files, key=_sort_key)[-1]
        available_configs.append(best)

        diff_dir = os.path.join(d, "diffusion")
        diff_model = ""
        diff_config = ""
        if os.path.isdir(diff_dir):
            for f in os.listdir(diff_dir):
                if f.endswith(".pt") and f != "model_0.pt":
                    diff_model = os.path.join("diffusion", f)
                if f.endswith(".yaml") or f.endswith(".yml"):
                    diff_config = os.path.join("diffusion", f)
        available_diffusion_models.append(diff_model)
        available_diffusion_configs.append(diff_config)

    print(f"FS scan found {len(available_models)} models")
    for i, m in enumerate(available_models):
        print(f"  [{i+1}] {os.path.basename(m)}")
    return available_models

def optimize_pitch_shift(key_shift):
    if key_shift > 6:
        return key_shift - 12
    elif key_shift < -6:
        return key_shift + 12
    else:
        return key_shift

def find_best_fuzzy_match(source_basename, candidate_list, threshold=0.4, default_value="not_found"):
    best_score = threshold
    best_match = default_value
    for candidate_path in candidate_list:
        candidate_basename = os.path.splitext(os.path.basename(candidate_path))[0]
        score = SequenceMatcher(None, source_basename, candidate_basename).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate_path
    return best_match, best_score

def _model_display_name(model_path):
    basename = os.path.basename(model_path) if os.path.isdir(model_path) else model_path
    config = _read_model_config(model_path)
    spks = config.get("spks", [])
    if spks:
        spk_str = ", ".join(spks)
        return f"{spk_str} ({basename})"
    return basename

def get_models_list_api():
    models_list = refresh_models_svc()
    display_list = []
    for m in models_list:
        display_list.append(_model_display_name(m))
    return display_list

def refresh_models_svc():
    global available_models, available_configs, available_diffusion_models, available_diffusion_configs
    print("Refreshing model list from SVC-Fusion...")
    _scan_models_from_fs()
    return available_models

def _read_model_config(model_path):
    config_yaml = os.path.join(model_path, "config.yaml")
    config_json = os.path.join(model_path, "config.json")
    if os.path.isfile(config_yaml):
        try:
            import yaml
            with open(config_yaml, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    if os.path.isfile(config_json):
        try:
            with open(config_json, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            pass
    return {}


def _is_valid_svc_model_dir(model_path):
    if not model_path or not os.path.isdir(model_path):
        return False
    has_config = os.path.isfile(os.path.join(model_path, "config.yaml")) or os.path.isfile(os.path.join(model_path, "config.json"))
    has_checkpoint = any(
        name.endswith(".pt") and name != "model_0.pt"
        for name in os.listdir(model_path)
        if os.path.isfile(os.path.join(model_path, name))
    )
    return has_config and has_checkpoint


def _model_config_signature(model_path):
    for name in ("config.yaml", "config.json"):
        config_path = os.path.join(model_path, name)
        if os.path.isfile(config_path):
            with open(config_path, "rb") as stream:
                return hashlib.sha256(stream.read()).hexdigest()
    return ""


def _resolve_svc_model_path(model_name):
    requested_display = str(model_name or "").strip()
    if not requested_display:
        raise ValueError("SVC 模型为空，请先刷新并选择一个有效模型")

    import re as _re
    requested = requested_display
    match = _re.search(r"\(([^)]+)\)\s*$", requested)
    if match:
        requested = match.group(1).strip()
    if not requested:
        raise ValueError("SVC 模型为空，请先刷新并选择一个有效模型")

    if not available_models:
        refresh_models_svc()

    root_norm = os.path.normcase(os.path.normpath(SVC_FUSION_ROOT))
    direct_path = os.path.abspath(requested)
    if os.path.normcase(os.path.normpath(direct_path)) == root_norm:
        raise ValueError("SVC-Fusion 根目录不是模型目录，请选择具体模型")
    if _is_valid_svc_model_dir(direct_path):
        return direct_path

    # Friendly models/* folders may only contain config.yaml. Resolve them to
    # the timestamped archive folder with the same configuration and weights.
    if os.path.isdir(direct_path):
        direct_signature = _model_config_signature(direct_path)
        if direct_signature:
            for candidate in available_models:
                if _model_config_signature(candidate) == direct_signature:
                    return os.path.abspath(candidate)

    requested_lower = requested.lower()
    exact_matches = []
    speaker_matches = []
    fuzzy_matches = []
    for candidate in available_models:
        if not _is_valid_svc_model_dir(candidate):
            continue
        basename = os.path.basename(candidate)
        if basename.lower() == requested_lower or _model_display_name(candidate) == requested_display:
            exact_matches.append(candidate)
            continue
        speakers = [str(value).lower() for value in _read_model_config(candidate).get("spks", []) if value]
        if requested_lower in speakers:
            speaker_matches.append(candidate)
        elif requested_lower and requested_lower in basename.lower():
            fuzzy_matches.append(candidate)

    matches = exact_matches or speaker_matches or fuzzy_matches
    if len(matches) == 1:
        return os.path.abspath(matches[0])
    if len(matches) > 1:
        raise ValueError(f"SVC 模型名称不唯一: {requested_display}")

    choices = ", ".join(_model_display_name(path) for path in available_models) or "无"
    raise ValueError(f"找不到 SVC 模型: {requested_display}；可用模型: {choices}")

def _find_gradio_endpoint(keyword=None, min_returns=None, max_params=None):
    info = _get_info()
    for source in ["named_endpoints", "unnamed_endpoints"]:
        endpoints = info.get(source, {})
        for ep_name, ep_info in endpoints.items():
            if keyword and keyword.lower() not in ep_name.lower():
                continue
            rets = ep_info.get("returns", [])
            params = ep_info.get("parameters", [])
            if min_returns is not None and len(rets) < min_returns:
                continue
            if max_params is not None and len(params) > max_params:
                continue
            clean = ep_name.lstrip("/")
            return clean
    return None

def _get_search_path_index(model_path):
    model_basename = os.path.basename(model_path)
    svc_dir = SVC_FUSION_ROOT
    workdir = os.path.join(svc_dir, "exp", "workdir")
    if os.path.normcase(os.path.normpath(model_path)) == os.path.normcase(os.path.normpath(workdir)):
        return 0
    archive_dir = os.path.join(svc_dir, "archive")
    if os.path.isdir(archive_dir):
        archive_items = sorted([p for p in os.listdir(archive_dir) if os.path.isdir(os.path.join(archive_dir, p))])
        for i, name in enumerate(archive_items):
            if name == model_basename:
                return i + 1
    models_dir = os.path.join(svc_dir, "models")
    if os.path.isdir(models_dir):
        models_items = sorted([p for p in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, p))])
        archive_count = 0
        if os.path.isdir(archive_dir):
            archive_count = len([p for p in os.listdir(archive_dir) if os.path.isdir(os.path.join(archive_dir, p))])
        for i, name in enumerate(models_items):
            if name == model_basename:
                return 1 + archive_count + i
    return 0

def load_svc_model(model_name: str):
    global current_speaker_id, current_model_type_index, loaded_speakers, _gradio_info_cache
    print(f"Requesting SVC-Fusion to load model: {model_name}")
    resolved_path = _resolve_svc_model_path(model_name)
    print(f"Resolved SVC model path: {resolved_path}")

    config = _read_model_config(resolved_path)
    model_type_index = config.get("model_type_index", -1)
    config_speakers = config.get("spks", [])

    model_type_names = ["DDSP-SVC 6.0", "Reflow-VAE-SVC", "So-VITS-SVC", "DDSP-SVC 6.1", "DDSP-SVC 6.3", "未知模型"]
    model_type_name = model_type_names[model_type_index] if 0 <= model_type_index < len(model_type_names) else "未知模型"
    current_model_type_index = model_type_index

    if config_speakers and isinstance(config_speakers, list):
        current_speaker_id = str(config_speakers[0])
        speakers = list(config_speakers)
        print(f"Config speakers (highest priority): {speakers}, selected: {current_speaker_id}")
    else:
        current_speaker_id = "speaker0"
        speakers = [current_speaker_id]
        print(f"No speakers in config, using default: {current_speaker_id}")

    api_loaded = False
    try:
        info = _get_info()
        ne = info.get("named_endpoints", {})

        if "/api_load_model_by_path" in ne:
            print(f"api_load_model_by_path: loading directly from path: {resolved_path}")
            r_load, e_load = _call_gradio_sse("api_load_model_by_path", [resolved_path], timeout=120)
            if e_load:
                print(f"api_load_model_by_path error: {e_load}")
            elif isinstance(r_load, list) and len(r_load) >= 2:
                msg = r_load[0]
                api_spks = r_load[1] if isinstance(r_load[1], list) else []
                print(f"api_load_model_by_path result: {msg}, speakers={api_spks}")
                if isinstance(msg, str) and msg.startswith("OK"):
                    if api_spks:
                        no_spk_vals = ["无说话人", "no_speaker", "No Speaker", None]
                        valid_api = [s for s in api_spks if s not in no_spk_vals]
                        if valid_api:
                            current_speaker_id = str(valid_api[0])
                            speakers = list(valid_api)
                            print(f"API loaded speakers: {speakers}, selected: {current_speaker_id}")
                            api_loaded = True
                        else:
                            print(f"API returned only no-speaker values: {api_spks}, using config speakers")
                            api_loaded = True
                    else:
                        print(f"API returned OK but empty speakers, using config speakers: {speakers}")
                        api_loaded = True
                elif isinstance(msg, str) and msg.startswith("ERROR"):
                    print(f"api_load_model_by_path failed: {msg}")
            else:
                print(f"api_load_model_by_path unexpected result: {r_load}")
        else:
            print("api_load_model_by_path not available, trying legacy chain")

            if "/on_refresh" in ne:
                sp_index = _get_search_path_index(resolved_path)
                print(f"on_refresh: using search_path index={sp_index} for model={os.path.basename(resolved_path)}")
                r2, e2 = _call_gradio_sse("on_refresh", [sp_index], timeout=30)
                if e2:
                    print(f"on_refresh({sp_index}) warning: {e2}")
                else:
                    print(f"on_refresh({sp_index}) OK")

            if "/change_model_type" in ne and model_type_name != "未知模型":
                r3, e3 = _call_gradio_sse("change_model_type", [model_type_name], timeout=15)
                if e3:
                    print(f"change_model_type warning: {e3}")
                else:
                    print(f"Changed model type to: {model_type_name}")
                    api_loaded = True
            elif "/change_model" in ne and model_type_name != "未知模型":
                r3, e3 = _call_gradio_sse("change_model", [model_type_name], timeout=15)
                if e3:
                    print(f"change_model warning: {e3}")
                else:
                    print(f"Changed model to: {model_type_name}")
                    api_loaded = True

            if "/on_submit" in ne and model_type_name != "未知模型":
                try:
                    r4, e4 = _call_gradio_sse("on_submit", [], timeout=120)
                    if not e4 and isinstance(r4, list) and len(r4) > 0:
                        sid_update = r4[0]
                        if isinstance(sid_update, dict):
                            choices = sid_update.get("choices", [])
                            if choices:
                                api_spk_list = [c[0] if isinstance(c, (list, tuple)) else c for c in choices]
                                no_spk_vals = ["无说话人", "no_speaker", "No Speaker", None]
                                valid_api = [s for s in api_spk_list if s not in no_spk_vals]
                                if valid_api:
                                    print(f"on_submit returned speakers (info only): {valid_api} (keeping config speaker: {current_speaker_id})")
                            val = sid_update.get("value", "")
                            if val and val not in ["无说话人", "no_speaker", "No Speaker"]:
                                print(f"on_submit returned value (info only): {val} (keeping config speaker: {current_speaker_id})")
                        api_loaded = True
                        print(f"on_submit OK")
                    elif e4:
                        print(f"on_submit warning: {e4}")
                except Exception as e_sub:
                    print(f"on_submit error (non-fatal): {e_sub}")

            speakers_api = _get_speakers()
            if speakers_api:
                no_spk_vals = ["无说话人", "no_speaker", "No Speaker", None]
                valid_api = [s for s in speakers_api if s not in no_spk_vals]
                if valid_api:
                    print(f"SVC-Fusion current state speakers (info only): {valid_api} (keeping config speaker: {current_speaker_id})")
                    api_loaded = True
    except Exception as e:
        print(f"API model loading failed: {e}")

    if not api_loaded:
        raise RuntimeError(f"SVC-Fusion 未确认加载模型成功: {model_name}")

    _gradio_info_cache = None

    loaded_speakers = list(speakers)
    print(f"Model loaded: path={os.path.basename(resolved_path)}, type={model_type_name}, speaker={current_speaker_id}, all_speakers={speakers}, loaded_speakers={loaded_speakers}")
    return f"Model loaded: {model_type_name}, speakers: {', '.join(speakers)}", current_speaker_id

def unload_svc_model():
    print("Requesting SVC-Fusion to unload model...")
    try:
        result, err = _call_gradio_sse("release_memory", [], timeout=15)
        if err:
            raise Exception(err)
        print(f"Model unloaded")
        return "Model unloaded"
    except Exception as e:
        error_msg = f"Failed to unload model: {e}"
        print(error_msg)
        return error_msg

def _find_infer_endpoint():
    info = _get_info()
    model_type_names = ["DDSP-SVC 6.0", "Reflow-VAE-SVC", "So-VITS-SVC", "DDSP-SVC 6.1", "DDSP-SVC 6.3"]
    if 0 <= current_model_type_index < len(model_type_names):
        target = "infer_" + "".join(c if c.isalnum() else "_" for c in model_type_names[current_model_type_index]).strip("_")
    else:
        target = "infer"
    candidates = []
    for source in ["named_endpoints", "unnamed_endpoints"]:
        endpoints = info.get(source, {})
        for ep_name, ep_info in endpoints.items():
            clean = ep_name.lstrip("/")
            if not clean.startswith(target):
                continue
            params = ep_info.get("parameters", [])
            has_audio = any(p.get("component", "") == "Audio" for p in params)
            has_speaker = any("说话人" in p.get("label", "") for p in params)
            if has_audio:
                score = 10
                if has_speaker:
                    score += 5
                if clean == target:
                    score += 20
                suffix = clean[len(target):]
                if suffix == "" or suffix == "_1":
                    score += 15
                elif suffix.startswith("_") and suffix[1:].isdigit():
                    score += 5
                candidates.append((score, clean, len(params)))
    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[2]))
        best = candidates[0][1]
        print(f"Found infer endpoint: {best} (score={candidates[0][0]}, params={candidates[0][2]})")
        return best
    for source in ["named_endpoints", "unnamed_endpoints"]:
        endpoints = info.get(source, {})
        for ep_name, ep_info in endpoints.items():
            clean = ep_name.lstrip("/")
            if clean.startswith("infer"):
                params = ep_info.get("parameters", [])
                has_audio = any(p.get("component", "") == "Audio" for p in params)
                if has_audio:
                    return clean
    return None

def _build_infer_params(endpoint_name, audio_data, speaker_id, key_shift, f0_method="fcpe"):
    info = _get_info()
    for source in ["named_endpoints", "unnamed_endpoints"]:
        endpoints = info.get(source, {})
        ep_info = endpoints.get("/" + endpoint_name, endpoints.get(endpoint_name))
        if ep_info:
            break
    else:
        ep_info = None

    if not ep_info:
        return [audio_data, None, False, False, False, False, False, False,
                f0_method, int(key_shift), 0, -60, "euler", 50, 0.7, 0,
                speaker_id, "DML"]

    params_info = ep_info.get("parameters", [])
    data = []
    for p in params_info:
        comp = p.get("component", "")
        label = p.get("label", "")
        ptype = p.get("type", {})
        default = p.get("parameter_default", None)

        if comp == "Audio":
            data.append(audio_data)
        elif comp == "Files" or comp == "File":
            data.append(None)
        elif comp == "Slider" and ("key" in label.lower() or "变调" in label or "升降调" in label):
            data.append(int(key_shift))
        elif "说话人" in label or "speaker" in label.lower() or "spk" in label.lower():
            if isinstance(default, dict) and "value" in default:
                spk_val = default["value"]
                if spk_val and spk_val != speaker_id:
                    data.append(spk_val if not speaker_id else speaker_id)
                else:
                    data.append(speaker_id)
            else:
                data.append(speaker_id)
        elif "设备" in label or "device" in label.lower():
            device_val = "DML"
            if isinstance(default, dict) and "value" in default:
                device_val = default["value"]
            elif isinstance(default, str):
                device_val = default
            data.append(device_val)
        elif comp == "Checkbox":
            if isinstance(default, bool):
                data.append(default)
            else:
                data.append(False)
        elif comp == "Slider":
            if isinstance(default, (int, float)):
                data.append(default)
            elif isinstance(ptype, dict) and "default" in ptype:
                data.append(ptype["default"])
            else:
                data.append(0)
        elif comp == "Dropdown":
            if "f0" in label.lower() or "音高" in label or "pitch" in label.lower():
                data.append(f0_method)
            elif isinstance(default, str):
                data.append(default)
            elif isinstance(default, dict) and "value" in default:
                data.append(default["value"])
            elif "enum" in (ptype if isinstance(ptype, dict) else {}):
                choices = ptype["enum"]
                data.append(choices[0] if choices else "")
            else:
                data.append("")
        else:
            if isinstance(default, (str, int, float, bool)):
                data.append(default)
            elif isinstance(default, dict) and "value" in default:
                data.append(default["value"])
            else:
                data.append(None)

    return data

def _audio_probe(path):
    info = soundfile.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"audio has no decodable samples: {path}")

    peak = 0.0
    with soundfile.SoundFile(path) as stream:
        while True:
            block = stream.read(262144, dtype="float32", always_2d=True)
            if not block.size:
                break
            peak = max(peak, float(np.max(np.abs(block))))

    return info.frames / float(info.samplerate), peak


def _validate_svc_output(output_path, input_audio_path):
    file_size = os.path.getsize(output_path)
    if file_size < 4096:
        raise ValueError(f"SVC output is too small ({file_size} bytes): {output_path}")

    output_duration, output_peak = _audio_probe(output_path)
    input_duration, _ = _audio_probe(input_audio_path)
    min_duration = max(1.0, input_duration * 0.60)
    max_duration = max(input_duration * 1.60, input_duration + 20.0)
    if not min_duration <= output_duration <= max_duration:
        raise ValueError(
            "SVC output duration is invalid: "
            f"output={output_duration:.2f}s input={input_duration:.2f}s"
        )
    if output_peak <= 1e-7:
        raise ValueError(f"SVC output is silent: {output_path}")

    print(
        "SVC output validated: "
        f"duration={output_duration:.2f}s, input={input_duration:.2f}s, "
        f"peak={output_peak:.6f}, size={file_size}"
    )
    return file_size


def _cached_mix_is_valid(cache_path):
    try:
        if os.path.getsize(cache_path) < 4096:
            return False, "file is smaller than 4096 bytes"
        duration, peak = _audio_probe(cache_path)
        if duration < 2.0:
            return False, f"duration is only {duration:.2f}s"
        if peak <= 1e-7:
            return False, "audio is silent"
        return True, f"duration={duration:.2f}s peak={peak:.6f}"
    except Exception as exc:
        return False, f"audio probe failed: {exc}"


def convert_svc(input_audio_path: str, speaker_id: str, key_shift: int, progress_callback=None, f0_method="fcpe"):
    global current_model_type_index, loaded_speakers
    print(f"SVC-Fusion inference: requested_speaker={speaker_id}, key_shift={key_shift}")
    print(f"SVC-Fusion inference: loaded_speakers={loaded_speakers}, current_speaker_id={current_speaker_id}")

    no_spk_vals = ["无说话人", "no_speaker", "No Speaker", None, "speaker0", ""]

    actual_speakers = loaded_speakers if loaded_speakers else []
    if not actual_speakers:
        svc_speakers = _get_speakers() or []
        actual_speakers = [s for s in svc_speakers if s not in no_spk_vals]
        print(f"No loaded_speakers, fell back to _get_speakers(): {actual_speakers}")

    if speaker_id in no_spk_vals:
        if actual_speakers:
            speaker_id = str(actual_speakers[0])
        else:
            raise Exception("SVC model not loaded - no valid speakers found. Please load model in SVC-Fusion UI first.")
        print(f"Using auto-detected speaker: {speaker_id}")
    elif actual_speakers and speaker_id not in actual_speakers:
        print(f"WARNING: Requested speaker '{speaker_id}' not in loaded model speakers {actual_speakers}")
        print(f"Falling back to first valid speaker: {actual_speakers[0]}")
        speaker_id = str(actual_speakers[0])

    print(f"Final speaker for inference: {speaker_id}")

    try:
        audio_upload = _upload_file_to_gradio(input_audio_path)
        if not audio_upload:
            raise Exception("Audio upload failed")

        if isinstance(audio_upload, str) and not audio_upload.startswith("/"):
            audio_data = {"path": audio_upload, "meta": {"_type": "gradio.FileData"}}
        elif isinstance(audio_upload, dict):
            audio_data = audio_upload
        else:
            audio_data = {"path": str(audio_upload), "meta": {"_type": "gradio.FileData"}}

        endpoint = _find_infer_endpoint()
        if endpoint:
            data = _build_infer_params(endpoint, audio_data, speaker_id, key_shift, f0_method)
        else:
            print("No infer endpoint found, using fallback parameter set")
            is_sovits = current_model_type_index == 2
            if is_sovits:
                data = [
                    audio_data, None, False, False, False, False, False, False,
                    f0_method, int(key_shift), 0, -60, 0, 0, 100, 0, 0.05,
                    False, False, 0, speaker_id,
                    "DML",
                ]
            else:
                data = [
                    audio_data, None, False, False, False, False, False, False,
                    f0_method, int(key_shift), 0, -60, "euler", 50, 0.7, 0,
                    speaker_id, "DML",
                ]

        print(f"Calling {endpoint} with {len(data)} params, key_shift={key_shift}, speaker={speaker_id}")
        result, err = _call_gradio_sse(endpoint, data, timeout=TIMEOUT, progress_callback=progress_callback)
        if err:
            raise Exception(err)

        output_audio = None
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    val = item.get("value", item)
                    if isinstance(val, dict) and ("path" in val or "url" in val):
                        output_audio = val
                        break
                    elif "path" in item or "url" in item:
                        output_audio = item
                        break
                elif isinstance(item, str) and (item.endswith(".wav") or item.endswith(".flac")):
                    output_audio = item
                    break
            if output_audio is None and len(result) >= 1:
                output_audio = result[0]
                if isinstance(output_audio, dict):
                    v = output_audio.get("value")
                    if isinstance(v, dict):
                        output_audio = v

        if output_audio is None:
            raise Exception(f"No audio in response: {result}")

        file_path = None
        if isinstance(output_audio, dict):
            file_path = output_audio.get("path") or output_audio.get("url")
        elif isinstance(output_audio, str):
            file_path = output_audio

        if file_path and os.path.exists(file_path):
            file_size = _validate_svc_output(file_path, input_audio_path)
            print(f"Inference result saved: {file_path} ({file_size} bytes)")
            return file_path
        elif file_path:
            if file_path.startswith("http"):
                download_url = file_path
            elif file_path.startswith("/"):
                download_url = f"{SVC_API_BASE}/file={file_path}"
            else:
                download_url = f"{SVC_API_BASE}/file={file_path}"
            print(f"Downloading result: {download_url}")
            response = requests.get(download_url, timeout=TIMEOUT)
            response.raise_for_status()
            audio_content = response.content
            os.makedirs("./temp", exist_ok=True)
            local_temp_path = f"./temp/svc_output_{int(_time.time())}.wav"
            with open(local_temp_path, "wb") as f:
                f.write(audio_content)
            file_size = _validate_svc_output(local_temp_path, input_audio_path)
            print(f"Inference result saved: {local_temp_path} ({file_size} bytes)")
            return local_temp_path
        else:
            raise Exception(f"Cannot parse output audio: {type(output_audio)} = {output_audio}")
    except Exception as e:
        print(f"SVC inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# =================================================================
#               原有功能的函数（大部分保持不变）
# =================================================================
from uvr5.vr import AudioPre
from pydub import AudioSegment
from pydub.effects import normalize
from pedalboard import Pedalboard, Compressor, Reverb, HighpassFilter, PeakFilter, LowpassFilter, PitchShift, Delay
import librosa, soundfile, gradio as gr, numpy as np
import scipy.signal
if not hasattr(scipy.signal, 'hann'):
    scipy.signal.hann = np.hanning

weight_uvr5_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uvr5", "uvr_model")
pre_fun_hp5 = AudioPre(agg=10, model_path=os.path.join(weight_uvr5_root, "5_HP-Karaoke-UVR.pth"), device=device, is_half=is_half)

def create_uvr5_pre_fun(agg=10, tta=False, postprocess=False, window_size=512, high_end_process="mirroring"):
    _pre = AudioPre(
      agg=int(agg),
      model_path=os.path.join(weight_uvr5_root, "5_HP-Karaoke-UVR.pth"),
      device=device,
      is_half=is_half,
      tta=tta,
    )
    _pre.data["postprocess"] = postprocess
    _pre.data["window_size"] = window_size
    _pre.data["high_end_process"] = high_end_process
    return _pre

split_model = "UVR-HP5"

print("[SVC] Using UVR5 (DirectML) for vocal separation")

headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"}

def get_response(song_id):
    print("开始下载歌曲")
    try:
        response = requests.get(f"https://biliplayer.91vrchat.com/player/?url=https://music.163.com/song?id={song_id}",allow_redirects=True, timeout=30)
        if response.status_code == 200: return response
    except Exception as e: print(f"主源下载失败: {e}")
    print("使用备用源下载歌曲")
    try:
        response1 = requests.get(f"https://api.vkeys.cn/v2/music/netease?id={song_id}", timeout=30).json()["data"]["url"]
        return requests.get(response1, timeout=30)
    except Exception as e: raise Exception(f"所有下载源均失败: {e}")

# 替换这个函数
def wwy_downloader(filename, split_model="UVR-HP5", cache_name=None, uvr5_pre_fun=None):
    cache_dir = cache_name if cache_name else filename
    audio_content = get_response(filename).content
    temp_prefixed_path = f"svc_{cache_dir}.wav"
    with open(temp_prefixed_path, mode="wb") as f:
        f.write(audio_content)

    audio_orig = AudioSegment.from_file(temp_prefixed_path)
    duration_minutes = len(audio_orig) / 60000
    print(f"Duration: {duration_minutes:.2f} min")
    if duration_minutes > 5:
        print("Audio > 5min, trimming...")
        audio_orig = audio_orig[:300000]

    uvr_input_path = f"{cache_dir}.wav"
    audio_orig.export(uvr_input_path, format="wav")

    if os.path.isfile(temp_prefixed_path):
        os.remove(temp_prefixed_path)

    os.makedirs(f"./output/{split_model}/{cache_dir}/", exist_ok=True)
    print("Separating vocals...")
    _pre = uvr5_pre_fun if uvr5_pre_fun else pre_fun_hp5
    _pre._path_audio_(uvr_input_path, f"./output/{split_model}/{cache_dir}/", f"./output/{split_model}/{cache_dir}/", "wav")

    if os.path.isfile(uvr_input_path):
        os.remove(uvr_input_path)

    return f"./output/{split_model}/{cache_dir}/vocal_{cache_dir}.wav_10.wav", f"./output/{split_model}/{cache_dir}/instrument_{cache_dir}.wav_10.wav"


def _get_cache_key(song_name, model, key_shift, vocal_vol, inst_vol, reverb_intensity, delay_intensity, svc_f0_method, uvr5_agg, uvr5_tta, uvr5_postprocess, uvr5_window_size, uvr5_high_end_process, msst_batch_size, msst_num_overlap, msst_normalize, vocal_postprocess, shift_accompaniment):
    params = {
        "pipeline": PIPELINE_VERSION,
        "song": str(song_name),
        "model": str(model),
        "model_assets": _svc_model_fingerprint(model),
        "key_shift": float(key_shift),
        "vocal_vol": float(vocal_vol),
        "inst_vol": float(inst_vol),
        "reverb": float(reverb_intensity),
        "delay": float(delay_intensity),
        "f0": str(svc_f0_method),
        "uvr5_agg": int(uvr5_agg),
        "uvr5_tta": bool(uvr5_tta),
        "uvr5_postprocess": bool(uvr5_postprocess),
        "uvr5_window_size": int(uvr5_window_size),
        "uvr5_high_end": str(uvr5_high_end_process),
        "msst_batch": float(msst_batch_size),
        "msst_overlap": float(msst_num_overlap),
        "msst_norm": bool(msst_normalize),
        "vocal_postprocess": bool(vocal_postprocess),
        "shift_inst": bool(shift_accompaniment),
    }
    return hashlib.md5(json.dumps(params, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:12]

def sanitize_filename(filename):
    return re.sub(r'[\\/:*?"<>|]', '', filename)

# =================================================================
#               核心转换流程 & Gradio UI
# =================================================================
@_serialized
def convert(song_name_src, key_shift, vocal_vol, inst_vol, model_dropdown, reverb_intensity = 4, delay_intensity = 0, svc_f0_method = "fcpe", uvr5_agg = 10, uvr5_tta = False, uvr5_postprocess = False, uvr5_window_size = 512, uvr5_high_end_process = "mirroring", msst_batch_size = 2, msst_num_overlap = 4, msst_normalize = True, vocal_postprocess=False, shift_accompaniment=True, progress=gr.Progress()):
    """进行翻唱推理合成"""
    print(f"🎵 [任务开始] SVC模型: {model_dropdown} | 算法: {svc_f0_method} | 升降调: {key_shift} | 混响: {reverb_intensity} | 延迟: {delay_intensity}")
    print(f"🔧 [UVR5 参数] Agg: {uvr5_agg} | TTA: {uvr5_tta} | PostProcess: {uvr5_postprocess} | WindowSize: {uvr5_window_size} | HighEnd: {uvr5_high_end_process}")
    print(f"🔧 [MSST 参数(忽略，amd后端不使用)] BatchSize: {msst_batch_size} | Overlap: {msst_num_overlap} | Normalize: {msst_normalize}")
    progress_local.progress = progress
    progress(0.1, desc="正在准备处理歌曲...")
    if not song_name_src: raise gr.Error("请输入歌曲ID或链接！")
    uvr5_pre = create_uvr5_pre_fun(agg=uvr5_agg, tta=uvr5_tta, postprocess=uvr5_postprocess, window_size=uvr5_window_size, high_end_process=uvr5_high_end_process)

    is_local_file = os.path.isfile(song_name_src) and not song_name_src.startswith("http")

    if song_name_src.startswith("http"):
        try: song_name_src = song_name_src.split('id=')[1].split('&')[0]
        except IndexError: raise gr.Error("无效的网易云链接格式！")
    song_name_src = song_name_src.strip()
    print(f"处理歌曲ID: {song_name_src}")

    if is_local_file:
        original_song_name = os.path.abspath(song_name_src)
        safe_name = f"local_{_sha256_file(original_song_name)[:16]}"
        song_name_src = safe_name
        source_cache_name = safe_name
    else:
        netease_safe_name = f"netease_{song_name_src}"
        source_cache_name = netease_safe_name

    cache_key = _get_cache_key(
        source_cache_name, model_dropdown, key_shift, vocal_vol, inst_vol,
        reverb_intensity, delay_intensity, svc_f0_method,
        uvr5_agg, uvr5_tta, uvr5_postprocess, uvr5_window_size,
        uvr5_high_end_process, msst_batch_size, msst_num_overlap,
        msst_normalize, vocal_postprocess, shift_accompaniment
    )
    cache_path = f"temp/{sanitize_filename(source_cache_name)}_{cache_key}_SVC.mp3"
    if os.path.isfile(cache_path):
        cache_valid, cache_detail = _cached_mix_is_valid(cache_path)
        if cache_valid:
            os.utime(cache_path, None)
            print(f"Cache hit, returning: {cache_path} ({cache_detail})")
            progress(1.0, desc="Cache hit, returning directly!")
            progress_local.progress = None
            return cache_path, "true"
        print(f"Invalid cache removed: {cache_path} ({cache_detail})")
        os.remove(cache_path)

    if is_local_file:
        vocal_cache_path = f"./output/{split_model}/{safe_name}/vocal_{safe_name}.wav_10.wav"

        if os.path.isfile(vocal_cache_path):
            print("Cached, skipping")
            vocal_path = vocal_cache_path
        else:
            print(f"Loading local file: {os.path.basename(original_song_name)}")
            progress(0.2, desc="加载本地音频文件...")
            audio_orig = AudioSegment.from_file(original_song_name)
            duration_minutes = len(audio_orig) / 60000
            print(f"Duration: {duration_minutes:.2f} min")
            if duration_minutes > 5:
                print("Audio > 5min, trimming...")
                audio_orig = audio_orig[:300000]

            uvr_input_path = safe_name + ".wav"
            audio_orig.export(uvr_input_path, format="wav")

            output_dir = f"./output/{split_model}/{safe_name}/"
            os.makedirs(output_dir, exist_ok=True)
            print("Separating vocals...")
            progress(0.4, desc="分离人声中(UVR5)...")
            uvr5_pre._path_audio_(uvr_input_path, output_dir, output_dir, "wav")

            if os.path.isfile(uvr_input_path):
                os.remove(uvr_input_path)

            vocal_path = f"./output/{split_model}/{safe_name}/vocal_{safe_name}.wav_10.wav"
            if not os.path.isfile(vocal_path):
                raise gr.Error(f"UVR5 separation failed: {vocal_path}")
    else:
        vocal_path = f"./output/{split_model}/{netease_safe_name}/vocal_{netease_safe_name}.wav_10.wav"
        if not os.path.exists(vocal_path):
            progress(0.4, desc="网易云下载并分离人声...")
            vocal_path, _ = wwy_downloader(song_name_src, split_model, cache_name=netease_safe_name, uvr5_pre_fun=uvr5_pre)
        else:
            print("✅ 网易云歌曲已缓存，跳过下载和分离")

    status_msg, speaker_id = load_model_ui(model_dropdown)
    progress(0.55, desc="SVC模型加载中...")
    
    progress(0.58, desc="SVC模型推理中(提交任务)...")
    inferred_audio_path = convert_svc(vocal_path, speaker_id, key_shift, f0_method=svc_f0_method)
        
    if not inferred_audio_path: raise gr.Error("SVC 推理失败，请检查 SVC 服务控制台输出。")
    print("开始处理音频")
    progress(0.75, desc="SVC推理完成，加载音频...")
    audio_data, sr = librosa.load(inferred_audio_path, sr=None, mono=False)
    if audio_data.ndim == 1: audio_data = audio_data.reshape(1, -1)
    progress(0.80, desc="应用音频效果(均衡/压缩/混响)...")
    # ========== 修正后的智能混响参数计算 ==========
    room_size_map =  (0.15,          0.40,          0.90)
    wet_level_map =  (0.10,          0.25,          0.45)

    if reverb_intensity <= 4:
        percent = reverb_intensity / 4.0
        room_size_val = room_size_map[0] + (room_size_map[1] - room_size_map[0]) * percent
        wet_level_val = wet_level_map[0] + (wet_level_map[1] - wet_level_map[0]) * percent
    else:
        percent = (reverb_intensity - 4) / 6.0
        room_size_val = room_size_map[1] + (room_size_map[2] - room_size_map[1]) * percent
        wet_level_val = wet_level_map[1] + (wet_level_map[2] - wet_level_map[1]) * percent

    dry_level_val = 1.0 - wet_level_val

    print(f"🎤 混响设置: 强度 {reverb_intensity}/10 => 房间大小={room_size_val:.2f}, 湿润度={wet_level_val:.2f}")

    # 根据来源类型使用正确的缓存名称构建伴奏路径
    if is_local_file:
        inst_path = f"output/{split_model}/{song_name_src}/instrument_{song_name_src}.wav_10.wav"
    else:
        inst_path = f"output/{split_model}/{netease_safe_name}/instrument_{netease_safe_name}.wav_10.wav"

    effects = [
        HighpassFilter(cutoff_frequency_hz=80),
        PeakFilter(cutoff_frequency_hz=200, gain_db=1.5, q=0.7),
        PeakFilter(cutoff_frequency_hz=3000, gain_db=2.0, q=1.0),
        PeakFilter(cutoff_frequency_hz=7000, gain_db=-3.0, q=2.0),
        LowpassFilter(cutoff_frequency_hz=16000),
        Compressor(threshold_db=-18.0, ratio=4.0, attack_ms=5.0, release_ms=150.0),
    ] if vocal_postprocess else []

    # ========== 只有当用户开启延迟时，才执行所有相关计算 ==========
    if vocal_postprocess and delay_intensity > 0:
        print("🎤 启用回声效果，开始准备参数...")
        try:
            print("🎵 正在检测歌曲BPM...")
            y_inst, sr_inst = librosa.load(inst_path, sr=None)
            tempo, _ = librosa.beat.beat_track(y=y_inst, sr=sr_inst)
            if isinstance(tempo, np.ndarray):
                actual_tempo = tempo[0]
            else:
                actual_tempo = tempo
            if actual_tempo > 0:
                print(f"✅ 检测到歌曲BPM约为: {actual_tempo:.1f}")
                delay_seconds_val = (60.0 / actual_tempo) * 0.5
            else:
                print("⚠️ 未能检测到有效的BPM，将使用默认值。")
                delay_seconds_val = 0.5
        except Exception as e:
            print(f"⚠️ BPM检测失败: {type(e).__name__}: {e}，将使用默认值。")
            delay_seconds_val = 0.5

        delay_mix_val = (delay_intensity / 10.0) * 0.35
        print(f"🎤 回声设置: 强度 {delay_intensity}/10 => 混合度={delay_mix_val:.2f}, 延迟时间={delay_seconds_val:.3f}s (BPM同步)")
        effects.append(Delay(delay_seconds=delay_seconds_val, feedback=0.25, mix=delay_mix_val))
    # ==========================================================

    if vocal_postprocess and reverb_intensity > 0:
        effects.append(Reverb(room_size=room_size_val, damping=0.4, wet_level=wet_level_val, dry_level=dry_level_val, width=0.8))

    board = Pedalboard(effects)
    processed = board(audio_data, sr) if effects else audio_data
    processed = np.clip(processed, -1.0, 1.0 - (1.0 / 32768.0))
    processed_int16 = np.rint(processed.T * 32768.0).astype(np.int16)
    processed_audio = AudioSegment(processed_int16.tobytes(), frame_rate=sr, sample_width=2, channels=processed.shape[0])
    normalized_audio = normalize(processed_audio + vocal_vol, headroom=-1.0)

    progress(0.88, desc="处理伴奏并混音...")
    # ========== 处理伴奏音高 ==========
    print("🎵 准备伴奏...")
    
    # 确保 temp 目录存在
    os.makedirs("temp", exist_ok=True)
    
    inst_shift = key_shift
    if shift_accompaniment and inst_shift != 0 and abs(inst_shift) != 12:
        print(f"🎹 正在将伴奏音高调整 {inst_shift:+d} 半音以匹配人声...")
        try:
            y_inst, sr_inst = librosa.load(inst_path, sr=None)
            pitch_board = Pedalboard([PitchShift(semitones=inst_shift)])
            y_shifted = pitch_board(y_inst, sr_inst)
            shifted_inst_path = f"temp/shifted_{song_name_src}_inst.wav"
            soundfile.write(shifted_inst_path, y_shifted, sr_inst)
            audio_inst = AudioSegment.from_file(shifted_inst_path, format="wav")
            try:
                os.remove(shifted_inst_path)
            except OSError:
                pass
            print(f"✅ 伴奏音高调整完成")
        except Exception as e:
            print(f"⚠️ 伴奏音高调整失败，使用原始伴奏: {e}")
            audio_inst = AudioSegment.from_file(inst_path, format="wav")
    else:
        if not shift_accompaniment:
            print("🎹 已关闭伴奏升调，保持原伴奏")
        else:
            print("🎹 不调整伴奏音高")
        audio_inst = AudioSegment.from_file(inst_path, format="wav")

    audio_inst = audio_inst + inst_vol
    combined_audio = normalized_audio.overlay(audio_inst)
    if combined_audio.max_dBFS > -1.0:
        combined_audio = combined_audio.apply_gain(-1.0 - combined_audio.max_dBFS)

    output_filename = cache_path
    combined_audio.export(output_filename, format="MP3", bitrate=OUTPUT_BITRATE)
    _trim_cache()
    if os.path.isfile(inferred_audio_path): os.remove(inferred_audio_path)
    print(f"✅ 已导出: {output_filename}")
    progress(0.95, desc="导出最终音频文件...")
    progress(1.0, desc="处理完成！")
    progress_local.progress = None
    return output_filename, "false"

# --- Gradio UI 定义 ---
app = gr.Blocks()
with app:
    gr.Markdown("# <center>SVC一键翻唱、重磅更新！</center>")
    gr.Markdown("## <center>自动分离人声翻唱并合并，自动混音！</center>")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                model_dropdown = gr.Dropdown(label="选择AI模型", choices=[], value=None, info="请先点击刷新加载模型列表")
                refresh_btn = gr.Button("🔄 刷新模型")
                load_btn = gr.Button("✅ 加载模型", variant="primary")
            with gr.Row():
                model_status = gr.Textbox(label="模型状态", value="请先加载模型", interactive=False)
                speaker_id_state = gr.Textbox(label="Speaker ID", value="speaker0", visible=False)
            with gr.Row():
                inp1 = gr.Textbox(label="请填写想要AI翻唱的网易云id或链接", placeholder="114514", info="直接填写网易云id或链接")
            with gr.Row():
                inp5 = gr.Slider(-12, 12, value=0, step=1, label="歌曲人声升降调", info="默认为0，+2为升高2个key，以此类推")
                inp6 = gr.Slider(-3, 3, value=0, step=0.5, label="调节人声音量(dB)")
                inp7 = gr.Slider(-3, 3, value=0, step=0.5, label="调节伴奏音量(dB)")
            with gr.Row():
                inp_reverb = gr.Slider(
                    minimum=0, maximum=10, value=4, step=0.5,
                    label="混响强度",
                    info="0为干声，4为默认值，10为宏大混响"
                )
                inp_delay = gr.Slider(
                    minimum=0, maximum=10, value=0, step=0.5,
                    label="回声(延迟)效果",
                    info="0为关闭，数值越大回声越明显"
                )
            btn = gr.Button("一键开启AI翻唱之旅吧💕", variant="primary")
        with gr.Column():
            out = gr.Audio(label="AI歌手为您倾情演唱的歌曲🎶", type="filepath", interactive=False, streaming=True)
            cache_flag = gr.Textbox(visible=False)
    def refresh_models_ui():
        models_list = refresh_models_svc()
        display_list = [_model_display_name(m) for m in models_list]
        return gr.Dropdown(choices=display_list, value=display_list[0] if display_list else "No models available")
    def load_model_ui(model_name):
        global current_speaker_id
        if not model_name or model_name == "No models available": return "❌ Please select a valid model", "speaker0"
        status_msg, speaker_id = load_svc_model(model_name)
        return status_msg, speaker_id
    refresh_btn.click(refresh_models_ui, outputs=model_dropdown,api_name=False)
    load_btn.click(load_model_ui, inputs=model_dropdown, outputs=[model_status, speaker_id_state],api_name=False)
    btn.click(convert, [inp1, inp5, inp6, inp7, model_dropdown, inp_reverb, inp_delay], [out, cache_flag], api_name=False)
    api_model_name = gr.Textbox(visible=False)
    api_svc_f0_method = gr.Dropdown(choices=["fcpe", "rmvpe", "crepe", "harvest", "parselmouth", "dio"], value="fcpe", visible=False)
    api_uvr5_agg = gr.Slider(minimum=0, maximum=20, step=1, value=10, visible=False)
    api_uvr5_tta = gr.Checkbox(value=False, visible=False)
    api_uvr5_postprocess = gr.Checkbox(value=False, visible=False)
    api_uvr5_window_size = gr.Dropdown(choices=[256, 512, 1024], value=512, visible=False)
    api_uvr5_high_end_process = gr.Dropdown(choices=["mirroring", "none"], value="mirroring", visible=False)
    api_msst_batch_size = gr.Number(value=2, visible=False)
    api_msst_num_overlap = gr.Number(value=4, visible=False)
    api_msst_normalize = gr.Checkbox(value=True, visible=False)
    api_vocal_postprocess = gr.Checkbox(value=False, visible=False)
    api_shift_accompaniment = gr.Checkbox(value=True, visible=False)
    api_output = gr.Audio(visible=False)
    api_cache_flag = gr.Textbox(visible=False)
    gr.Button("API Convert", visible=False).click(
        convert,
        inputs=[inp1, inp5, inp6, inp7, api_model_name, inp_reverb, inp_delay, api_svc_f0_method, api_uvr5_agg, api_uvr5_tta, api_uvr5_postprocess, api_uvr5_window_size, api_uvr5_high_end_process, api_msst_batch_size, api_msst_num_overlap, api_msst_normalize, api_vocal_postprocess, api_shift_accompaniment],
        outputs=[api_output, api_cache_flag],
        api_name="convert"
    )
    gr.Button("API Show Model", visible=False).click(
        fn=get_models_list_api,
        inputs=[],
        outputs=[gr.JSON(visible=False)],
        api_name="show_model"
    )
    cache_scope = gr.Textbox(value="all", visible=False)
    cache_json = gr.JSON(visible=False)
    app.load(cache_info, inputs=[], outputs=cache_json, api_name="cache_info")
    gr.Button("API Clear Cache", visible=False).click(
        clear_cache, inputs=[cache_scope], outputs=[cache_json], api_name="clear_cache"
    )
    gr.Markdown("### <center>注意❗：请不要生成会对个人以及组织造成侵害的内容，此程序仅供科研、学习及个人娱乐使用。</center>")
    gr.HTML('''<div class="footer"><p>🌊🏞️🎶 - 江水东流急，滔滔无尽声。 明·顾璘</p></div>''')

print("Initializing and loading model list from SVC-Fusion...")
initial_models = refresh_models_svc()
if initial_models:
    display_list = [_model_display_name(m) for m in initial_models]
    model_dropdown.choices = display_list
    model_dropdown.value = display_list[0]
else:
    print("Warning: failed to load model list, ensure SVC-Fusion is running")

app.queue(max_size=40, api_open=True, default_concurrency_limit=1)
app.launch(
    server_name=os.environ.get("RVCSVC_HOST", "127.0.0.1"),
    server_port=9999,
    share=_env_flag("RVCSVC_SHARE", False),
    show_error=_env_flag("RVCSVC_SHOW_ERROR", False),
)
