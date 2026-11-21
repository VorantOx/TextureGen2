# -*- coding: utf-8 -*-
"""
Local web editor for the staged OTWNDATA texture pipeline.

Run:
    python otwn_editor.py

The app uses only the Python standard library for the web server.  Generated
stages are stored in editor_data/jobs/<job>/ and can be resumed after a
restart.  The editor opens at http://127.0.0.1:8765/.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import threading
import time
import traceback
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import otwn_pipeline as pipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EDITOR_DIR = os.path.join(BASE_DIR, "editor_data")
JOBS_DIR = os.path.join(EDITOR_DIR, "jobs")
HTML_PATH = os.path.join(BASE_DIR, "otwn_editor.html")


def now() -> float:
    return time.time()


def _atomic_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json(path: str, fallback=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return fallback


class JobManager:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.threads = {}
        self.load_all()

    # ------------------------------------------------------------------
    # persistence / lookup
    # ------------------------------------------------------------------
    def job_dir(self, jid: str) -> str:
        if not re.fullmatch(r"[a-f0-9]{10}", jid or ""):
            raise ValueError("Некорректный id задачи")
        return os.path.join(self.root, jid)

    def job_file(self, jid: str) -> str:
        return os.path.join(self.job_dir(jid), "job.json")

    def load_all(self) -> None:
        for name in os.listdir(self.root):
            p = os.path.join(self.root, name, "job.json")
            if not os.path.isfile(p):
                continue
            job = _read_json(p)
            if not job or not job.get("id"):
                continue
            job.setdefault("stage_params", {})
            self.jobs[job["id"]] = job
            self.sync_job(job)
            self.save(job)

    def get(self, jid: str):
        with self.lock:
            job = self.jobs.get(jid)
            if not job:
                raise KeyError("Задача не найдена")
            job.setdefault("stage_params", {})
            self.sync_job(job)
            return job

    def save(self, job) -> None:
        _atomic_json(self.job_file(job["id"]), job)

    # ------------------------------------------------------------------
    # creation / params
    # ------------------------------------------------------------------
    def _pipeline(self, job) -> pipeline.Pipeline:
        return pipeline.Pipeline(self.job_dir(job["id"]), job["params"],
                                 job.get("stage_params") or {})

    def create(self, params, name: str = "", source: str = "",
               stage_params=None) -> dict:
        params = pipeline.normalize_params(params)
        jid = uuid.uuid4().hex[:10]
        d = self.job_dir(jid)
        os.makedirs(os.path.join(d, "cache"), exist_ok=True)
        os.makedirs(os.path.join(d, "previews"), exist_ok=True)
        os.makedirs(os.path.join(d, "exports"), exist_ok=True)
        job = {
            "id": jid,
            "name": name.strip() or ("%s %d px" % (params["variant"],
                                                   params["size"])),
            "created": now(),
            "updated": now(),
            "params": params,
            "stage_params": pipeline.normalize_stage_params(stage_params),
            "hash": pipeline.params_hash(params),
            "running_stage": None,
            "stages": {},
            "log": [],
            "source": source,
        }
        for stage in pipeline.STAGE_ORDER:
            job["stages"][stage] = {
                "status": "pending",
                "approved": False,
                "message": "",
                "preview": False,
                "exports": [],
                "seconds": None,
            }
        self._log(job, "Создана задача %s" % jid)
        with self.lock:
            self.jobs[jid] = job
            self.save(job)
        return job

    def update_params(self, jid: str, params) -> dict:
        job = self.get(jid)
        with self.lock:
            if job.get("running_stage"):
                raise RuntimeError("Нельзя менять параметры, пока этап выполняется")
            new_params = pipeline.normalize_params(params)
            new_hash = pipeline.params_hash(new_params)
            old_hash = job.get("hash")
            job["params"] = new_params
            if new_hash != old_hash:
                job["hash"] = new_hash
                for st in pipeline.STAGE_ORDER:
                    job["stages"][st] = {
                        "status": "pending",
                        "approved": False,
                        "message": "параметры изменены — кэш не соответствует",
                        "preview": False,
                        "exports": [],
                        "seconds": None,
                    }
                self._log(job, "Параметры обновлены; старые подтверждения сброшены")
            job["updated"] = now()
            self.save(job)
            self.sync_job(job)
            self.save(job)
            return job

    def publish(self, jid: str) -> dict:
        """Copy approved final maps into textures_otwn/<variant>/."""
        job = self.get(jid)
        required = ["albedo", "height", "roughness", "ao",
                    "normal_gl", "normal_dx"]
        missing = [s for s in required
                   if not job["stages"][s].get("approved")]
        if missing:
            raise RuntimeError(
                "Сначала подтвердите финальные карты: %s"
                % ", ".join(pipeline.STAGE_LABELS.get(s, s) for s in missing))
        p = self._pipeline(job)
        dest = os.path.join(BASE_DIR, "textures_otwn", job["params"]["variant"])
        os.makedirs(dest, exist_ok=True)
        files = []
        for name in required:
            src = p.export_path(name)
            if not os.path.isfile(src):
                raise RuntimeError("Нет файла %s; перезапустите этап" % name)
            dst = os.path.join(dest, os.path.basename(src))
            shutil.copy2(src, dst)
            files.append(dst)
        with self.lock:
            self._log(job, "Карты опубликованы в %s" % dest)
            job["updated"] = now()
            self.save(job)
        return {"files": files, "directory": dest}

    def delete(self, jid: str) -> None:
        job = self.get(jid)
        with self.lock:
            if job.get("running_stage"):
                raise RuntimeError("Нельзя удалить задачу, пока этап выполняется")
            d = self.job_dir(jid)
            self.jobs.pop(jid, None)
            try:
                shutil.rmtree(d)
            except OSError as exc:
                raise RuntimeError("Не удалось удалить каталог: %s" % exc)

    def clone(self, jid: str, variant: str = None, size: int = None,
              name: str = "") -> dict:
        src = self.get(jid)
        params = dict(src["params"])
        if variant:
            params["variant"] = str(variant)
        if size:
            params["size"] = int(size)
        if not name:
            name = "%s / %s %d px" % (
                src.get("name", src["id"]), params["variant"], params["size"])
        return self.create(params, name=name, source=jid,
                           stage_params=src.get("stage_params") or {})

    # ------------------------------------------------------------------
    # stage status
    # ------------------------------------------------------------------
    def _log(self, job, text: str) -> None:
        job.setdefault("log", []).append({
            "t": time.strftime("%H:%M:%S"),
            "text": text,
        })
        job["log"] = job["log"][-120:]

    def _auto_discover(self, job, stage: str) -> None:
        st = job["stages"][stage]
        try:
            p = self._pipeline(job)
        except Exception:
            return
        cache = p.stage_path(stage)
        if st.get("status") == "running":
            t = self.threads.get(stage)
            if t is not None and t.is_alive():
                return
            if os.path.exists(cache):
                st["status"] = "ready"
                st["preview"] = os.path.exists(p.preview_path(stage))
                st["message"] = st.get("message") or "кэш найден после перезапуска"
                st["approved"] = False
            else:
                st["status"] = "interrupted"
                st["message"] = "выполнение было прервано"
            return
        if st.get("status") in (None, "pending", "interrupted"):
            if os.path.exists(cache):
                st["status"] = "ready"
                st["preview"] = os.path.exists(p.preview_path(stage))
                st["message"] = st.get("message") or "кэш найден"
                st["approved"] = False
        if st.get("status") in ("ready", "approved") and not os.path.exists(cache):
            st["status"] = "pending"
            st["approved"] = False
            st["preview"] = False
            st["message"] = "кэш этапа отсутствует — запустите заново"

    def sync_job(self, job) -> None:
        with self.lock:
            if job.get("running_stage"):
                # A crash can leave this flag behind; a live worker clears it.
                # Check the thread object itself, not only the dictionary key:
                # a dead thread may still be present after an interrupted run.
                rs = job["running_stage"]
                t = self.threads.get(rs)
                if t is None or not t.is_alive():
                    job["running_stage"] = None
            for stage in pipeline.STAGE_ORDER:
                self._auto_discover(job, stage)
            # Exports are discovered from the directory too, so a restart
            # does not lose links.
            p = None
            try:
                p = self._pipeline(job)
            except Exception:
                pass
            if p:
                for stage in pipeline.STAGE_ORDER:
                    st = job["stages"][stage]
                    names = []
                    for name in ("albedo", "height", "roughness", "ao",
                                 "normal_gl", "normal_dx"):
                        if os.path.exists(p.export_path(name)):
                            names.append(name)
                    # Only show the maps that belong to this stage, plus
                    # inherited maps are still useful on later confirmations.
                    mine = {
                        "albedo": ["albedo"],
                        "height": ["height"],
                        "roughness": ["roughness"],
                        "ao": ["ao"],
                        "normal_gl": ["normal_gl"],
                        "normal_dx": ["normal_dx"],
                    }.get(stage, [])
                    if st.get("status") in ("ready", "approved"):
                        st["exports"] = [n for n in mine if n in names]
                    elif stage not in mine:
                        st["exports"] = []

    def dependency_status(self, job, stage: str) -> list:
        out = []
        for dep in pipeline.STAGE_DEPS.get(stage, []):
            st = job["stages"].get(dep, {})
            if not st.get("approved"):
                out.append(dep)
        return out

    def _remove_stage_files(self, job, stage: str) -> None:
        """Remove a stage and every cached map belonging to it.

        This is used when an upstream stage is re-run or explicitly reset:
        downstream caches would otherwise look valid even though their input
        changed.  Parameter changes use a different hash and therefore never
        touch an unrelated old cache.
        """
        try:
            p = self._pipeline(job)
        except Exception:
            return
        for path in (p.stage_path(stage), p.preview_path(stage),
                     p.export_path(stage)):
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass

    def run(self, jid: str, stage: str, overrides=None) -> dict:
        if stage not in pipeline.STAGE_ORDER:
            raise ValueError("Неизвестный этап")
        job = self.get(jid)
        with self.lock:
            if job.get("running_stage"):
                raise RuntimeError("Уже выполняется этап %s" % job["running_stage"])

            # Stage-local controls.  They do not alter the base job hash,
            # so upstream caches remain valid.  Only this stage and the
            # stages below it are invalidated.
            if isinstance(overrides, dict) and overrides:
                current = dict(job.get("stage_params") or {})
                merged_stage = dict(current.get(stage) or {})
                merged_stage.update(overrides)
                merged = dict(current)
                merged[stage] = merged_stage
                norm = pipeline.normalize_stage_params(merged)
                if norm.get(stage, {}) != (current.get(stage) or {}):
                    job["stage_params"] = norm
                    idx = pipeline.STAGE_ORDER.index(stage)
                    for s in pipeline.STAGE_ORDER[idx:]:
                        self._remove_stage_files(job, s)
                        st0 = job["stages"][s]
                        st0["status"] = "pending"
                        st0["approved"] = False
                        st0["preview"] = False
                        st0["exports"] = []
                        st0["message"] = "параметры этапа изменены — нужен повтор"
                    self._log(job, "Параметры этапа %s обновлены"
                              % pipeline.STAGE_LABELS.get(stage, stage))
                    self.save(job)

            deps = self.dependency_status(job, stage)
            if deps:
                raise RuntimeError(
                    "Сначала подтвердите предыдущие этапы: %s"
                    % ", ".join(pipeline.STAGE_LABELS.get(d, d) for d in deps))
            st = job["stages"][stage]
            st["status"] = "running"
            st["approved"] = False
            st["message"] = "выполняется..."
            st["seconds"] = None
            job["running_stage"] = stage
            job["updated"] = now()
            self._log(job, "Запуск этапа %s" % pipeline.STAGE_LABELS.get(stage, stage))
            self.save(job)
            t = threading.Thread(target=self._worker, args=(jid, stage),
                                 daemon=True)
            self.threads[stage] = t
            t.start()
        return job

    def _worker(self, jid: str, stage: str) -> None:
        err = None
        info = None
        try:
            job = self.get(jid)
            p = self._pipeline(job)
            info = p.run(stage)
        except Exception as exc:
            err = "%s: %s" % (type(exc).__name__, exc)
            traceback.print_exc()
        with self.lock:
            job = self.jobs.get(jid)
            if not job:
                return
            job["running_stage"] = None
            self.threads.pop(stage, None)
            st = job["stages"][stage]
            if err:
                st["status"] = "failed"
                st["approved"] = False
                st["message"] = err
                self._log(job, "Ошибка этапа %s: %s" % (stage, err))
            else:
                st["status"] = "ready"
                st["approved"] = False
                st["message"] = info.get("message", "готово")
                st["preview"] = bool(info.get("preview"))
                st["exports"] = list(info.get("exports") or [])
                st["seconds"] = info.get("seconds")
                # A rerun invalidates downstream confirmations and their
                # caches.  Otherwise a restart could mistake an old downstream
                # array for a result of the newly generated upstream stage.
                idx = pipeline.STAGE_ORDER.index(stage)
                for later in pipeline.STAGE_ORDER[idx + 1:]:
                    self._remove_stage_files(job, later)
                    ls = job["stages"][later]
                    ls["status"] = "pending"
                    ls["approved"] = False
                    ls["message"] = "зависимость изменена — нужен повтор"
                    ls["preview"] = False
                    ls["exports"] = []
                self._log(job, "Этап %s готов за %s с" %
                          (stage, info.get("seconds")))
            job["updated"] = now()
            self.save(job)

    def approve(self, jid: str, stage: str) -> dict:
        job = self.get(jid)
        with self.lock:
            st = job["stages"][stage]
            if st.get("status") == "running":
                raise RuntimeError("Этап ещё выполняется")
            if st.get("status") != "ready":
                raise RuntimeError("Этап ещё не готов")
            deps = self.dependency_status(job, stage)
            if deps:
                raise RuntimeError(
                    "Сначала подтвердите предыдущие этапы: %s"
                    % ", ".join(pipeline.STAGE_LABELS.get(d, d) for d in deps))
            st["approved"] = True
            st["message"] = st.get("message") or "подтверждено"
            job["updated"] = now()
            self._log(job, "Подтверждён этап %s" %
                      pipeline.STAGE_LABELS.get(stage, stage))
            self.save(job)
        return job

    def reset(self, jid: str, stage: str) -> dict:
        job = self.get(jid)
        with self.lock:
            if job.get("running_stage"):
                raise RuntimeError("Нельзя сбрасывать, пока этап выполняется")
            if stage not in pipeline.STAGE_ORDER:
                raise ValueError("Неизвестный этап")
            idx = pipeline.STAGE_ORDER.index(stage)
            for s in pipeline.STAGE_ORDER[idx:]:
                self._remove_stage_files(job, s)
                st = job["stages"][s]
                st["status"] = "pending"
                st["approved"] = False
                st["message"] = "сброшено — можно запустить заново"
                st["preview"] = False
                st["exports"] = []
            job["updated"] = now()
            self._log(job, "Сброс с этапа %s" % stage)
            self.save(job)
        return job

    # ------------------------------------------------------------------
    # public JSON
    # ------------------------------------------------------------------
    def public(self, job) -> dict:
        self.sync_job(job)
        d = {
            "id": job["id"],
            "name": job.get("name"),
            "created": job.get("created"),
            "updated": job.get("updated"),
            "params": job["params"],
            "stage_params": job.get("stage_params") or {},
            "stage_param_specs": pipeline.resolved_stage_specs(
                job["params"], job.get("stage_params") or {}),
            "hash": job.get("hash"),
            "running_stage": job.get("running_stage"),
            "stages": {},
            "log": list(reversed(job.get("log", [])))[:40],
            "stage_order": pipeline.STAGE_ORDER,
            "stage_labels": pipeline.STAGE_LABELS,
            "stage_deps": pipeline.STAGE_DEPS,
        }
        for stage in pipeline.STAGE_ORDER:
            st = dict(job["stages"][stage])
            prev = st.get("preview", False)
            st["preview_url"] = ("/preview/%s/%s?v=%s" %
                                 (job["id"], stage, int(job.get("updated", 0))
                                  if prev else 0)) if prev else None
            st["export_urls"] = {
                n: "/map/%s/%s?v=%s" % (job["id"], n,
                                        int(job.get("updated", 0)))
                for n in st.get("exports", [])
            }
            d["stages"][stage] = st
        return d

    def list_public(self) -> list:
        with self.lock:
            items = list(self.jobs.values())
            items.sort(key=lambda j: j.get("updated", 0), reverse=True)
            return [{
                "id": j["id"],
                "name": j.get("name"),
                "params": j.get("params"),
                "created": j.get("created"),
                "updated": j.get("updated"),
                "running_stage": j.get("running_stage"),
            } for j in items]


MANAGER = JobManager(JOBS_DIR)


class Handler(BaseHTTPRequestHandler):
    server_version = "OTWNEditor/1.0"

    def log_message(self, fmt, *args):
        # Keep the console readable; errors are still printed by tracebacks.
        return

    def _send_json(self, obj, status=HTTPStatus.OK):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        try:
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def _send_file(self, path, content_type=None):
        if not os.path.isfile(path):
            self._send_json({"error": "Файл не найден"}, HTTPStatus.NOT_FOUND)
            return
        if content_type is None:
            content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        size = os.path.getsize(path)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(path, "rb") as fh:
            shutil.copyfileobj(fh, self.wfile)

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        return json.loads(raw.decode("utf-8"))

    def _job(self, jid):
        return MANAGER.get(jid)

    # ------------------------------------------------------------------
    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path in ("/", "/index.html"):
                self._send_file(HTML_PATH, "text/html; charset=utf-8")
                return
            if path == "/api/jobs":
                self._send_json({"jobs": MANAGER.list_public()})
                return
            m = re.fullmatch(r"/api/job/([a-f0-9]{10})", path)
            if m:
                self._send_json(MANAGER.public(self._job(m.group(1))))
                return
            m = re.fullmatch(r"/preview/([a-f0-9]{10})/([a-z_]+)", path)
            if m:
                job = self._job(m.group(1))
                st = job["stages"].get(m.group(2))
                if not st:
                    raise KeyError("Нет такого этапа")
                p = MANAGER._pipeline(job)
                self._send_file(p.preview_path(m.group(2)), "image/png")
                return
            m = re.fullmatch(r"/map/([a-f0-9]{10})/([a-z_]+)", path)
            if m:
                job = self._job(m.group(1))
                p = MANAGER._pipeline(job)
                self._send_file(p.export_path(m.group(2)), "image/png")
                return
            self._send_json({"error": "Не найдено"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_DELETE(self):
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            m = re.fullmatch(r"/api/job/([a-f0-9]{10})", path)
            if m:
                MANAGER.delete(m.group(1))
                self._send_json({"deleted": m.group(1)})
                return
            self._send_json({"error": "Не найдено"}, HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            body = self._read_body()
            if path == "/api/jobs":
                job = MANAGER.create(body.get("params") or {},
                                     name=body.get("name") or "")
                self._send_json(MANAGER.public(job), HTTPStatus.CREATED)
                return
            m = re.fullmatch(r"/api/job/([a-f0-9]{10})/params", path)
            if m:
                job = MANAGER.update_params(m.group(1), body.get("params") or {})
                self._send_json(MANAGER.public(job))
                return
            m = re.fullmatch(r"/api/job/([a-f0-9]{10})/run", path)
            if m:
                stage = body.get("stage") or ""
                overrides = body.get("overrides") or {}
                job = MANAGER.run(m.group(1), stage, overrides)
                self._send_json(MANAGER.public(job), HTTPStatus.ACCEPTED)
                return
            m = re.fullmatch(r"/api/job/([a-f0-9]{10})/approve", path)
            if m:
                job = MANAGER.approve(m.group(1), body.get("stage") or "")
                self._send_json(MANAGER.public(job))
                return
            m = re.fullmatch(r"/api/job/([a-f0-9]{10})/reset", path)
            if m:
                job = MANAGER.reset(m.group(1), body.get("stage") or "")
                self._send_json(MANAGER.public(job))
                return
            m = re.fullmatch(r"/api/job/([a-f0-9]{10})/clone", path)
            if m:
                job = MANAGER.clone(m.group(1), body.get("variant"),
                                    body.get("size"), body.get("name") or "")
                self._send_json(MANAGER.public(job), HTTPStatus.CREATED)
                return
            m = re.fullmatch(r"/api/job/([a-f0-9]{10})/publish", path)
            if m:
                self._send_json(MANAGER.publish(m.group(1)))
                return
            self._send_json({"error": "Не найдено"}, HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    if not os.path.isfile(HTML_PATH):
        raise SystemExit("Не найден otwn_editor.html рядом с otwn_editor.py")
    url = "http://%s:%d/" % (args.host, args.port)
    print("OTWN editor: %s" % url)
    print("Данные: %s" % EDITOR_DIR)
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")


if __name__ == "__main__":
    main()
