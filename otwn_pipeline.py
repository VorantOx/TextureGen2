# -*- coding: utf-8 -*-
"""
Staged, resumable OTWNDATA texture pipeline.

This module deliberately re-uses the low-level helpers from
make_otwn_textures.py, but keeps every visual stage in its own cache file.
The web editor (otwn_editor.py) drives this class.

Stage outputs are stored as small deltas in ``cache/<stage>_<hash>.npz``.
Loading a stage means loading all earlier deltas in order, so generation can
be stopped after any stage and resumed later without re-running the whole
texture from scratch.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

import make_otwn_textures as base


PIPELINE_VERSION = 4

STAGE_ORDER: List[str] = [
    "topology",
    "color",
    "seams",
    "moss",
    "chips",
    "cracks",
    "speckles",
    "albedo",
    "height",
    "roughness",
    "ao",
    "normal_gl",
    "normal_dx",
]

STAGE_LABELS: Dict[str, str] = {
    "topology": "1. Топология",
    "color": "2. Цвет камней",
    "seams": "3. Швы",
    "moss": "4. Мох",
    "chips": "5. Сколы",
    "cracks": "6. Царапины",
    "speckles": "7. Минеральные вкрапления",
    "albedo": "8. Albedo — итог",
    "height": "9. Height",
    "roughness": "10. Roughness",
    "ao": "11. AO",
    "normal_gl": "12. Normal GL",
    "normal_dx": "13. Normal DX",
}

STAGE_DEPS: Dict[str, List[str]] = {
    "topology": [],
    "color": ["topology"],
    "seams": ["color"],
    "moss": ["seams"],
    "chips": ["moss"],
    "cracks": ["chips"],
    "speckles": ["cracks"],
    "albedo": ["speckles"],
    "height": ["albedo"],
    "roughness": ["height"],
    "ao": ["height"],
    "normal_gl": ["ao"],
    "normal_dx": ["normal_gl"],
}

# Stage-local controls.  These are deliberately independent of the base
# job parameters: adjusting one of them re-renders only that stage (and
# invalidates the stages below it).  `default_from` resolves to a value in
# the job parameters; `default` is used for a new control.
STAGE_PARAM_SPECS: Dict[str, List[Dict[str, Any]]] = {
    "topology": [
        {"key": "roundness", "label": "Скругление камней", "min": 0.0,
         "max": 2.0, "step": 0.01, "default": 1.0,
         "hint": "0 — жёсткие многоугольники, 2 — почти овалы"},
    ],
    "color": [
        {"key": "color_variation", "label": "Разброс цвета камней",
         "min": 0.0, "max": 2.0, "step": 0.01, "default": 1.0},
        {"key": "color_shade", "label": "Сила объёмной тени",
         "min": 0.0, "max": 2.0, "step": 0.01, "default": 1.0},
    ],
    "seams": [
        {"key": "seam_width_px", "label": "Ширина шва / сглаживание, px",
         "min": 0.0, "max": 80.0, "step": 0.5, "default": 0.0,
         "hint": "0 — без сглаживания; растёт от границы к центру камня"},
        {"key": "seam_contrast", "label": "Контрастность шва",
         "min": 0.0, "max": 2.0, "step": 0.01, "default": 0.0,
         "hint": "0 — этап 2 без изменений; больше — проявляет тёмные границы"},
        {"key": "seam_darkness", "label": "Темнота шва",
         "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.5,
         "hint": "0 — глубокий тёмный шов, 1 — шов на уровне камня"},
    ],
    "moss": [
        {"key": "moss_amount", "label": "Количество мха",
         "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0},
    ],
    "chips": [
        {"key": "chip_count", "label": "Число сколов", "min": 0, "max": 200,
         "step": 1, "default_from": "chip_count"},
        {"key": "chip_strength", "label": "Заметность в albedo",
         "min": 0.0, "max": 2.0, "step": 0.01, "default": 1.0},
        {"key": "chip_depth", "label": "Глубина рельефа",
         "min": 0.0, "max": 2.0, "step": 0.01, "default": 1.0},
    ],
    "cracks": [
        {"key": "crack_count", "label": "Число трещин", "min": 0, "max": 100,
         "step": 1, "default_from": "crack_count"},
        {"key": "crack_strength", "label": "Сила трещин", "min": 0.0,
         "max": 2.0, "step": 0.01, "default": 1.0},
    ],
    "speckles": [
        {"key": "speckle_amount", "label": "Количество вкраплений",
         "min": 0.0, "max": 2.0, "step": 0.01, "default": 1.0},
    ],
    "albedo": [
        {"key": "baked_ao", "label": "Запечённое AO в albedo",
         "min": 0.0, "max": 0.40, "step": 0.005, "default": 0.04},
    ],
    "height": [
        {"key": "dome", "label": "Купол камня", "min": 0.0, "max": 2.0,
         "step": 0.01, "default_from": "dome"},
        {"key": "height_relief", "label": "Контраст высоты", "min": 0.3,
         "max": 1.5, "step": 0.01, "default": 1.0},
    ],
    "roughness": [
        {"key": "roughness_bias", "label": "Смещение roughness",
         "min": -0.20, "max": 0.20, "step": 0.005, "default": 0.0},
        {"key": "roughness_contrast", "label": "Контраст roughness",
         "min": 0.4, "max": 1.6, "step": 0.01, "default": 1.0},
    ],
    "ao": [
        {"key": "ao_strength", "label": "Сила AO", "min": 0.0, "max": 1.5,
         "step": 0.01, "default": 1.0},
    ],
    "normal_gl": [
        {"key": "normal_detail", "label": "Мелкая детализация",
         "min": 0.0, "max": 4.0, "step": 0.01,
         "default_from": "normal_detail"},
        {"key": "normal_strength", "label": "Сила normal map",
         "min": 0.0, "max": 2.0, "step": 0.01, "default": 1.0},
    ],
    "normal_dx": [
        {"key": "normal_detail", "label": "Мелкая детализация",
         "min": 0.0, "max": 4.0, "step": 0.01,
         "default_from": "normal_detail"},
        {"key": "normal_strength", "label": "Сила normal map",
         "min": 0.0, "max": 2.0, "step": 0.01, "default": 1.0},
    ],
}

DEFAULT_PARAMS: Dict[str, Any] = {
    "size": 512,
    "seed": 7,
    "variant": "gold",
    "joint_px": 12.0,
    "dome": 0.9,
    "chip_count": 15,
    "crack_count": 0,
    "spread": 3.4,
    "medium_count": 16,
    "medium_area": 0.09,
    "medium_min_px": 40.0,
    "seam_halo": 0.018,
    "seam_halo_amt": 0.44,
    "normal_detail": 1.0,
}

VARIANTS = ["gold", "aged", "cyclopean"]


def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(DEFAULT_PARAMS)
    p.update({k: v for k, v in (params or {}).items() if k in DEFAULT_PARAMS})
    p["size"] = int(np.clip(int(p["size"]), 128, 4096))
    p["seed"] = int(p["seed"])
    p["variant"] = str(p["variant"])
    if p["variant"] not in VARIANTS:
        p["variant"] = "gold"
    p["joint_px"] = float(p["joint_px"])
    p["dome"] = float(np.clip(float(p["dome"]), 0.0, 2.0))
    p["chip_count"] = int(np.clip(int(p["chip_count"]), 0, 200))
    p["crack_count"] = int(np.clip(int(p["crack_count"]), 0, 100))
    p["spread"] = float(np.clip(float(p["spread"]), 1.0, 10.0))
    p["medium_count"] = int(np.clip(int(p["medium_count"]), 0, 128))
    p["medium_area"] = float(np.clip(float(p["medium_area"]), 0.0, 0.40))
    p["medium_min_px"] = float(np.clip(float(p["medium_min_px"]), 0.0, 256.0))
    p["seam_halo"] = float(np.clip(float(p["seam_halo"]), 0.0, 0.20))
    p["seam_halo_amt"] = float(np.clip(float(p["seam_halo_amt"]), 0.0, 1.0))
    p["normal_detail"] = float(np.clip(float(p["normal_detail"]), 0.0, 4.0))
    return p


def normalize_stage_params(overrides: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for stage, spec in STAGE_PARAM_SPECS.items():
        raw = (overrides or {}).get(stage, {}) if isinstance(overrides, dict) else {}
        clean: Dict[str, float] = {}
        if not isinstance(raw, dict):
            continue
        for item in spec:
            key = item["key"]
            if key not in raw or raw[key] is None:
                continue
            try:
                val = float(raw[key])
            except (TypeError, ValueError):
                continue
            val = float(np.clip(val, float(item["min"]), float(item["max"])))
            if item.get("step", 0) and float(item["step"]).is_integer():
                val = float(round(val))
            clean[key] = val
        if clean:
            out[stage] = clean
    return out


def resolved_stage_specs(params: Dict[str, Any],
                         overrides: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Specs with concrete defaults/current values for the browser."""
    p = normalize_params(params)
    ov = normalize_stage_params(overrides)
    moss_default = variant_config(p["variant"])["moss_amt"]
    out: Dict[str, List[Dict[str, Any]]] = {}
    for stage, spec in STAGE_PARAM_SPECS.items():
        rows = []
        for item in spec:
            row = dict(item)
            if "default_from" in item:
                key = item["default_from"]
                if key == "joint_px":
                    default = p["joint_px"]
                else:
                    default = p.get(key, item.get("default", 0.0))
            elif item["key"] == "moss_amount":
                default = moss_default
            else:
                default = item.get("default", 0.0)
            row["default"] = default
            row["value"] = ov.get(stage, {}).get(item["key"], default)
            row["overridden"] = item["key"] in ov.get(stage, {})
            rows.append(row)
        out[stage] = rows
    return out


def params_hash(params: Dict[str, Any]) -> str:
    blob = json.dumps(normalize_params(params), sort_keys=True,
                      ensure_ascii=False).encode("utf-8")
    blob += str(PIPELINE_VERSION).encode("ascii")
    return hashlib.sha1(blob).hexdigest()[:12]


def _save_png_fast(path: str, arr) -> None:
    """Interactive previews: no zlib optimization, much faster at 2K."""
    a = (np.clip(np.asarray(arr), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(a).save(path, optimize=False, compress_level=1)


def suffix_for_size(size: int) -> str:
    return {512: "512", 1024: "1k", 2048: "2k", 4096: "4k"}.get(
        int(size), str(int(size)))


def variant_config(variant: str) -> Dict[str, Any]:
    if variant == "aged":
        return dict(
            big_amp=0.74, big_base=0.25, joint_px_default=10.0,
            big_rough=0.60, small_rough=0.68, moss_amt=0.38,
            pal=base.AGED_PAL, pal_w=base.AGED_W,
            mortar=np.array([0.105, 0.080, 0.050], np.float32),
            dirt=np.array([0.160, 0.145, 0.086], np.float32),
            small_pal=base.AGED_PAL * np.array([0.94, 0.96, 0.92],
                                                np.float32),
            fill_grid=88,
        )
    if variant == "cyclopean":
        return dict(
            big_amp=0.95, big_base=0.24, joint_px_default=9.0,
            big_rough=0.50, small_rough=0.60, moss_amt=0.0,
            pal=base.GOLD_PAL, pal_w=base.GOLD_W,
            mortar=np.array([0.160, 0.110, 0.058], np.float32),
            dirt=np.array([0.240, 0.170, 0.090], np.float32),
            small_pal=base.GOLD_PAL[:5] * np.array([0.90, 0.88, 0.84],
                                                    np.float32),
            fill_grid=80,
        )
    return dict(
        big_amp=0.82, big_base=0.25, joint_px_default=10.0,
        big_rough=0.54, small_rough=0.64, moss_amt=0.0,
        pal=base.GOLD_PAL, pal_w=base.GOLD_W,
        mortar=np.array([0.145, 0.100, 0.052], np.float32),
        dirt=np.array([0.235, 0.165, 0.088], np.float32),
        small_pal=base.GOLD_PAL[:5] * np.array([0.92, 0.90, 0.86],
                                                np.float32),
        fill_grid=86,
    )


class Pipeline:
    def __init__(self, job_dir: str, params: Dict[str, Any],
                 overrides: Optional[Dict[str, Any]] = None):
        self.job_dir = os.path.abspath(job_dir)
        self.params = normalize_params(params)
        self.overrides = normalize_stage_params(overrides)
        self.hash = params_hash(self.params)
        self.size = int(self.params["size"])
        self.seed = int(self.params["seed"])
        self.variant = self.params["variant"]
        self.cfg = variant_config(self.variant)
        self._noise_cache: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        self.cache_dir = os.path.join(self.job_dir, "cache")
        self.preview_dir = os.path.join(self.job_dir, "previews")
        self.export_dir = os.path.join(self.job_dir, "exports")
        for d in (self.cache_dir, self.preview_dir, self.export_dir):
            os.makedirs(d, exist_ok=True)

    def p(self, stage: str, key: str, default=None):
        """Value of a stage-local parameter, falling back to job params."""
        ov = self.overrides.get(stage, {}) or {}
        if key in ov:
            return ov[key]
        if default is None:
            default = self.params.get(key)
        return default

    # ------------------------------------------------------------------
    # cache helpers
    # ------------------------------------------------------------------
    def stage_path(self, stage: str) -> str:
        return os.path.join(self.cache_dir, "%s_%s.npz" % (stage, self.hash))

    def preview_path(self, stage: str) -> str:
        return os.path.join(self.preview_dir, "%s_%s.png" % (stage, self.hash))

    def export_path(self, name: str) -> str:
        return os.path.join(
            self.export_dir, "%s_%s.png" % (name, suffix_for_size(self.size)))

    def load_until(self, stage: str) -> Dict[str, np.ndarray]:
        """Load all deltas before *stage* (not the stage itself)."""
        state: Dict[str, np.ndarray] = {}
        for name in STAGE_ORDER:
            if name == stage:
                break
            p = self.stage_path(name)
            if not os.path.exists(p):
                raise RuntimeError(
                    "Нет кэша предыдущего этапа %s. Сначала выполните его."
                    % name)
            try:
                with np.load(p, allow_pickle=False) as z:
                    for k in z.files:
                        if k != "__done__":
                            state[k] = z[k]
            except Exception as exc:
                raise RuntimeError("Повреждён кэш этапа %s: %s" % (name, exc))
        return state

    def stage_cached(self, stage: str) -> bool:
        return os.path.exists(self.stage_path(stage))

    def _save_stage(self, stage: str, delta: Dict[str, Any],
                    preview: Optional[np.ndarray] = None,
                    exports: Optional[Dict[str, np.ndarray]] = None,
                    message: str = "") -> Dict[str, Any]:
        arrays = {}
        for k, v in (delta or {}).items():
            if v is None:
                continue
            arrays[k] = np.asarray(v)
        if not arrays:
            arrays["__done__"] = np.array([1], np.uint8)
        np.savez(self.stage_path(stage), **arrays)
        if preview is not None:
            _save_png_fast(self.preview_path(stage), preview)
        export_names = []
        if exports:
            for name, arr in exports.items():
                _save_png_fast(self.export_path(name), arr)
                export_names.append(name)
        return {
            "stage": stage,
            "cache": self.stage_path(stage),
            "preview": self.preview_path(stage),
            "exports": export_names,
            "message": message,
        }

    # ------------------------------------------------------------------
    # deterministic noise, regenerated when a new process resumes
    # ------------------------------------------------------------------
    def noise(self) -> Dict[str, np.ndarray]:
        key = (self.size, self.seed)
        if key in self._noise_cache:
            return self._noise_cache[key]
        h = w = self.size
        s = self.seed
        n = {
            "n_low": base.fft_noise((h, w), 3.2, s + 1),
            "n_med": base.fft_noise((h, w), 2.0, s + 2),
            "n_fine": base.fft_noise((h, w), 0.9, s + 3),
            "n_fine2": base.fft_noise((h, w), 0.7, s + 4),
            "n_tint": base.fft_noise((h, w), 2.4, s + 5),
            "n_shape": base.fft_noise((h, w), 3.0, s + 6),
            "n_pit": base.fft_noise((h, w), 1.5, s + 7),
            "n_stri": base.fft_noise((h, w), 1.2, s + 8),
            "n_dark": base.fft_noise((h, w), 0.8, s + 9),
            "n_spot": base.fft_noise((h, w), 1.3, s + 10),
            "n_micro": base.fft_noise((h, w), 0.55, s + 11),
            "n_edge": base.fft_noise((h, w), 9.0, s + 11),
        }
        n["n_low_s"] = base.gauss_blur(n["n_low"], 1.0)
        n["n_med_s"] = base.gauss_blur(n["n_med"], 1.0)
        n["n_fine_s"] = base.gauss_blur(n["n_fine"], 0.8)
        n["n_shape_s"] = base.gauss_blur(n["n_shape"], 1.6)
        n["n_shape_geom"] = base.gauss_blur(
            n["n_shape"], max(1.0, 0.0018 * self.size))
        n["n_edge_geom"] = base.gauss_blur(
            n["n_edge"], max(0.8, 0.0012 * self.size))
        self._noise_cache[key] = n
        return n

    NOISE_SPEC = {
        "n_low": (3.2, 1),
        "n_med": (2.0, 2),
        "n_fine": (0.9, 3),
        "n_fine2": (0.7, 4),
        "n_tint": (2.4, 5),
        "n_shape": (3.0, 6),
        "n_pit": (1.5, 7),
        "n_stri": (1.2, 8),
        "n_dark": (0.8, 9),
        "n_spot": (1.3, 10),
        "n_micro": (0.55, 11),
        "n_edge": (9.0, 11),
    }
    NOISE_DERIVED = {
        "n_low_s": ("n_low", 1.0),
        "n_med_s": ("n_med", 1.0),
        "n_fine_s": ("n_fine", 0.8),
        "n_shape_s": ("n_shape", 1.6),
    }

    def nfield(self, name: str) -> np.ndarray:
        """Compute just one noise field for a lightweight stage re-render."""
        key = ("field", name, self.size, self.seed)
        if key in self._noise_cache:
            return self._noise_cache[key]
        if name in self.NOISE_DERIVED:
            base_name, sigma = self.NOISE_DERIVED[name]
            raw = self.nfield(base_name)
            out = base.gauss_blur(raw, sigma)
        else:
            spec = self.NOISE_SPEC.get(name)
            if spec is None:
                raise KeyError("Неизвестное поле шума: %s" % name)
            sigma, off = spec
            out = base.fft_noise((self.size, self.size), sigma,
                                 self.seed + off)
        self._noise_cache[key] = out
        return out

    def geometry_noise(self) -> Dict[str, np.ndarray]:
        """Only the two fields needed by the topology geometry pass."""
        key = ("geometry", self.size, self.seed)
        if key in self._noise_cache:
            return self._noise_cache[key]
        h = w = self.size
        n_shape = base.fft_noise((h, w), 3.0, self.seed + 6)
        n_edge = base.fft_noise((h, w), 9.0, self.seed + 11)
        out = {
            "n_shape_geom": base.gauss_blur(
                n_shape, max(1.0, 0.0018 * self.size)),
            "n_edge_geom": base.gauss_blur(
                n_edge, max(0.8, 0.0012 * self.size)),
        }
        self._noise_cache[key] = out
        return out

    def _grid(self) -> Tuple[np.ndarray, np.ndarray]:
        h = w = self.size
        Y0, X0 = np.mgrid[0:h, 0:w].astype(np.float32)
        return (X0 + 0.5) / w, (Y0 + 0.5) / h

    def _layout_key(self) -> str:
        """Key for the expensive part: centres, weights, medium repair."""
        d = {
            "version": PIPELINE_VERSION,
            "size": self.size,
            "seed": self.seed,
            "spread": self.params["spread"],
            "medium_count": self.params["medium_count"],
            "medium_area": self.params["medium_area"],
            "medium_min_px": self.params["medium_min_px"],
            "joint_on": bool(float(self.params["joint_px"]) > 0.5),
        }
        blob = json.dumps(d, sort_keys=True).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:12]

    def _layout_path(self) -> str:
        return os.path.join(self.cache_dir, "layout_%s.npz" % self._layout_key())

    def _get_layout(self) -> Dict[str, Any]:
        """Load/build the power diagram, independent of visual roundness.

        The costly centre relaxation, weight optimisation and full-resolution
        nearest-neighbour fields are cached separately.  A roundness or seam
        slider can therefore re-render topology without repeating them.
        """
        path = self._layout_path()
        if os.path.exists(path):
            try:
                with np.load(path, allow_pickle=False) as z:
                    return {k: z[k] for k in z.files}
            except Exception:
                pass

        joint_on, joint_half, _, _ = self._joint()
        (ids, bedge_raw, bedge_s, prof, rim, inradius, cell_area,
         centers, weights, n_big, n_med) = base.build_multi_layout(
            self.size, self.seed, spread=self.params["spread"],
            medium_count=self.params["medium_count"] if joint_on else 0,
            medium_area=self.params["medium_area"], joint_half=joint_half,
            round_sigma=0.009, verbose=False)

        medium_min_px = float(self.params.get("medium_min_px", 0.0))
        if n_med and medium_min_px > 0.5:
            target_inr = medium_min_px / 2048.0
            adjusted = weights.copy()
            for _ in range(5):
                ids, bedge_raw, bedge_s, prof, rim, inradius, cell_area = \
                    self._retessellate(centers, adjusted, joint_half)
                med_inr = inradius[n_big:n_big + n_med]
                deficit = np.maximum(target_inr - med_inr, 0.0)
                if float(deficit.max()) < 1e-5:
                    break
                adjusted[n_big:n_big + n_med] += 0.22 * deficit
            weights = adjusted

        out = {
            "ids": ids.astype(np.int16),
            "bedge_raw": bedge_raw.astype(np.float32),
            "centers": centers.astype(np.float32),
            "weights": weights.astype(np.float32),
            "n_big": np.int32(n_big),
            "n_med": np.int32(n_med),
        }
        np.savez(path, **out)
        return out

    def _retessellate(self, centers: np.ndarray, weights: np.ndarray,
                      joint_half: float):
        """Recreate the full-res power diagram after a weight adjustment."""
        h = w = self.size
        seed = self.seed
        warp_x = base.fft_noise((h, w), 2.2, seed + 20)
        warp_y = base.fft_noise((h, w), 2.2, seed + 21)
        warp_x = base.gauss_blur(warp_x, max(0.8, 0.0012 * self.size))
        warp_y = base.gauss_blur(warp_y, max(0.8, 0.0012 * self.size))
        warp = (0.0055 * warp_x, 0.0055 * warp_y)
        ids, bedge_raw = base.power_fields(centers, weights, self.size, warp=warp)
        bedge = base.gauss_blur(bedge_raw, 0.009 * self.size)
        n_shape = base.fft_noise((h, w), 3.0, seed + 6)
        n_shape = base.gauss_blur(n_shape, max(1.0, 0.0018 * self.size))
        n_edge = base.fft_noise((h, w), 9.0, seed + 11)
        n_edge = base.gauss_blur(n_edge, max(0.8, 0.0012 * self.size))
        bedge += 0.0022 * n_shape + 0.0030 * n_edge
        inradius = np.zeros(len(centers), np.float32)
        np.maximum.at(inradius, ids.ravel(), bedge.ravel())
        inradius = np.maximum(inradius, 1e-3)
        cell_area = np.bincount(ids.ravel(),
                                minlength=len(centers)).astype(np.float32)
        cell_area /= float(h * w)
        t = np.clip((bedge - joint_half) /
                    np.maximum(inradius[ids] - joint_half, 1e-4), 0.0, 1.0)
        prof = (1.0 - (1.0 - t) ** 2.3) ** 0.55
        prof *= base.smoothstep(0.0, 0.085, t)
        rim = np.exp(-((bedge - joint_half) / 0.006) ** 2)
        return (ids, bedge_raw, bedge, prof, rim, inradius, cell_area)

    def _joint(self) -> Tuple[bool, float, float, float]:
        req = float(self.params["joint_px"])
        cfg = self.cfg
        # A None value is not possible in the editor, but keep compatibility.
        if req <= 0.5:
            joint_px_eff = 8.0
            return False, 0.0, joint_px_eff / 4096.0, joint_px_eff
        return True, req / 4096.0, 0.0, req

    # ------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------
    def run(self, stage: str) -> Dict[str, Any]:
        if stage not in STAGE_ORDER:
            raise ValueError("Неизвестный этап: %s" % stage)
        t0 = time.time()
        state = self.load_until(stage)
        fn = getattr(self, "_stage_" + stage)
        result = fn(state)
        if len(result) == 4:
            delta, preview, exports, message = result
        elif len(result) == 3:
            delta, preview, message = result
            exports = {}
        else:
            raise RuntimeError("Некорректный результат этапа %s" % stage)
        info = self._save_stage(stage, delta, preview, exports, message)
        info["seconds"] = round(time.time() - t0, 2)
        info["params_hash"] = self.hash
        return info

    def _topology_preview(self, ids: np.ndarray, bedge: np.ndarray,
                          centers: np.ndarray, inradius: np.ndarray,
                          cell_area: np.ndarray, n_big: int, n_med: int,
                          stone_mask: np.ndarray, roundness: float):
        """Schematic topology view.

        Large stones are marked with a small gold pin.  As roundness rises,
        large stones are drawn as superellipses that become ellipses/circles;
        the lost corner area is filled by a procedural small-pebble layer.
        This is a preview only; the exact stone masks are generated below.
        """
        h = w = self.size
        n_total = n_big + n_med
        rng = np.random.default_rng(self.seed + 902)
        cols = rng.uniform(0.25, 0.95, (n_total, 3)).astype(np.float32)
        cols[:n_big] = np.clip(
            cols[:n_big] * np.array([1.0, 0.93, 0.68], np.float32) + 0.04,
            0.0, 1.0)
        cols[n_big:] = np.clip(
            cols[n_big:] * np.array([0.85, 0.90, 0.82], np.float32) + 0.03,
            0.0, 1.0)
        out = np.zeros((h, w, 3), np.float32)
        out[:] = np.array([0.075, 0.055, 0.035], np.float32)

        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        X = (X + 0.5) / w
        Y = (Y + 0.5) / h
        # Medium stones keep their real cell silhouette.
        med_vis = (ids >= n_big) & (stone_mask > 0.5)
        out[med_vis] = cols[ids[med_vis]] * (
            0.65 + 0.35 * stone_mask[med_vis, None])

        roundness = float(np.clip(roundness, 0.0, 2.0))
        big_edge = np.zeros((h, w), bool)
        if roundness <= 1.0:
            # The normal topology view: real cell silhouettes, merely rounded
            # by the existing opening.  This keeps the default preview honest.
            big_inside = (ids < n_big) & (stone_mask > 0.5)
            out[big_inside] = cols[ids[big_inside]] * (
                0.72 + 0.28 * stone_mask[big_inside, None])
            occupied = med_vis | big_inside
        else:
            # Strong rounding: large stones become superellipses, then
            # ellipses/circles at the maximum.  The ellipse radius is capped
            # by the nearest large-stone centre so neighbouring stones do not
            # merge into one blob.
            p_exp = 2.0 + 7.0 * (1.0 - roundness / 2.0)
            best_q = np.full((h, w), 1.0e9, np.float32)
            best_id = np.full((h, w), -1, np.int32)
            for i in range(n_big):
                cx, cy = centers[i]
                dx = base.wrap(X - cx)
                dy = base.wrap(Y - cy)
                dd = base.wrap(centers - centers[i])
                centre_dist = np.sqrt((dd * dd).sum(axis=1))
                centre_dist[i] = 1.0e9
                r_eq = min(max(0.012, math.sqrt(float(cell_area[i]) / math.pi)),
                           0.42 * float(centre_dist.min()))
                aspect = float(np.clip(
                    1.0 + 2.2 * (float(inradius[i]) / max(r_eq, 1e-6) - 1.0),
                    0.78, 1.28))
                ax = r_eq * math.sqrt(aspect)
                ay = r_eq / math.sqrt(aspect)
                q = (np.abs(dx / max(ax, 1e-6)) ** p_exp +
                     np.abs(dy / max(ay, 1e-6)) ** p_exp) ** (1.0 / p_exp)
                m = q < best_q
                best_q[m] = q[m]
                best_id[m] = i
                big_edge |= (q < 1.04) & (q > 0.965)

            big_inside = (best_q < 1.0) & (best_id >= 0)
            out[big_inside] = cols[best_id[big_inside]]
            occupied = med_vis | big_inside
            rim = np.exp(-((best_q - 1.0) / 0.055) ** 2) * big_inside
            out[rim > 0.45] *= 0.82

        # Fill the voids that appear when large stones become round.
        if roundness > 1.05:
            sid, qmin, prof = base.build_small_layer(
                w, self.seed, grid=86, verbose=False)
            fill = (qmin < 1.0) & (~occupied)
            # Only fill the interstices, never the normal thin joints at low
            # roundness.  The fill strength ramps in with the slider.
            strength = float(base.smoothstep(0.85, 1.55, roundness))
            if strength > 0:
                fcols = rng.uniform(0.25, 0.85, (86 * 86, 3)).astype(np.float32)
                fcols = np.clip(fcols * np.array([0.95, 0.91, 0.78], np.float32)
                                + 0.02, 0, 1)
                # deterministic soft edge from qmin; avoids a black gap
                # between the pebble and the rounded large stone
                out[fill] = fcols[sid[fill]] * (
                    0.70 + 0.30 * prof[fill, None])

        # Separate overlapping ellipses with a joint ring; without this the
        # strong-rounding preview reads as a field of merged blobs.
        out[big_edge] = np.array([0.095, 0.072, 0.046], np.float32)

        # Small pin on every large stone.  The marker is deliberately tiny so
        # it does not hide the shape.
        img = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        r = max(4, int(round(0.0060 * self.size)))
        line = max(1, self.size // 1024)
        for i in range(n_big):
            cx, cy = centers[i]
            # draw wrapped copies near the tile border
            for ox in (-w, 0, w):
                for oy in (-h, 0, h):
                    px = int(round(cx * w + ox))
                    py = int(round(cy * h + oy))
                    if -r <= px < w + r and -r <= py < h + r:
                        # small diamond pin: compact, but clearly not a stone
                        draw.polygon([(px, py - r), (px + r, py),
                                      (px, py + r), (px - r, py)],
                                     fill=(255, 240, 170),
                                     outline=(25, 18, 10))
                        draw.line([px, py - r, px, py + r],
                                  fill=(110, 70, 24), width=line)
        return np.asarray(img).astype(np.float32) / 255.0

    # -- 1. topology ----------------------------------------------------
    def _stage_topology(self, _state: Dict[str, np.ndarray]):
        h = w = self.size
        seed = self.seed
        n = self.geometry_noise()
        X0, Y0 = self._grid()
        cfg = self.cfg
        joint_on, joint_half, seam_half, joint_px_eff = self._joint()

        layout = self._get_layout()
        ids = layout["ids"].astype(np.int32)
        bedge_raw = layout["bedge_raw"].astype(np.float32)
        centers = layout["centers"].astype(np.float32)
        weights = layout["weights"].astype(np.float32)
        n_big = int(layout["n_big"])
        n_med = int(layout["n_med"])
        n_total = n_big + n_med

        # Roundness is a stage-local control: it only changes the geometry
        # pass, never the expensive layout cache.  The two terms control
        # how much the cell edge is smoothed and how wide the raster opening
        # is, which reads as sharper polygons at 0 and pebbles/ovals at 2.
        roundness = float(np.clip(float(self.p("topology", "roundness", 1.0)),
                                  0.0, 2.0))
        round_sigma_px = max(0.35, 0.009 * self.size * roundness)
        bedge = base.gauss_blur(bedge_raw, round_sigma_px)
        bedge += 0.0022 * n["n_shape_geom"] + 0.0030 * n["n_edge_geom"]

        cell_area = np.bincount(
            ids.ravel(), minlength=n_total).astype(np.float32)
        cell_area /= float(self.size * self.size)
        inradius = np.zeros(n_total, np.float32)
        np.maximum.at(inradius, ids.ravel(), bedge.ravel())
        inradius = np.maximum(inradius, 1e-3)

        soft_px = max(0.8, 0.0015 * self.size)
        pad = max(16, int(0.020 * self.size))
        if joint_on:
            round_px = max(0.5, 0.0140 * self.size * roundness)
            core = bedge_raw > (round_px / self.size)
            dist_core = base.periodic_distance(core, pad)
            stone_mask = 1.0 - base.smoothstep(
                round_px - soft_px, round_px + soft_px, dist_core)
            rim_w = max(1.0, 0.003 * self.size)
            rim = np.exp(-((dist_core - round_px) / rim_w) ** 2)
        else:
            dist_core = np.zeros_like(bedge_raw)
            stone_mask = base.smoothstep(
                -soft_px / self.size, soft_px / self.size, bedge_raw)
            rim_w = max(1.0, 0.003 * self.size)
            rim = np.exp(-(bedge_raw / rim_w) ** 2)
        big_mask = stone_mask * (ids < n_big)
        med_mask = stone_mask * (ids >= n_big)
        gap = 1.0 - stone_mask

        inr_norm = np.maximum(inradius[ids], 1.0 / self.size)
        invp = 1.0 / inr_norm ** 6
        local_inr_norm = base.gauss_blur(
            invp, max(1.0, 0.0100 * self.size)) ** (-1.0 / 6.0)
        local_inr = np.maximum(local_inr_norm * self.size, 1.0)
        large_gate = base.smoothstep(0.035, 0.055, local_inr_norm)
        soft_col = max(2.0, 0.010 * self.size)
        soft_col_px = soft_col * (0.40 + 0.60 * large_gate)
        if joint_on:
            stone_col_mask = 1.0 - base.smoothstep(
                round_px - soft_col_px, round_px + soft_col_px, dist_core)
        else:
            stone_col_mask = base.smoothstep(
                -soft_col_px / self.size, soft_col_px / self.size, bedge_raw)
        gap_col_mask = 1.0 - stone_col_mask

        soft_h = max(1.5, 0.0035 * self.size)
        if joint_on:
            stone_h = 1.0 - base.smoothstep(
                round_px - soft_h, round_px + soft_h, dist_core)
        else:
            stone_h = base.smoothstep(
                -soft_h / self.size, soft_h / self.size, bedge_raw)
        stone_h = base.gauss_blur(stone_h, max(0.8, 0.0012 * self.size))
        gap_h = 1.0 - stone_h

        inr_s = np.zeros(n_total, np.float32)
        np.maximum.at(inr_s, ids.ravel(), bedge.ravel())
        t = np.clip(
            (bedge - joint_half) / np.maximum(inr_s[ids] - joint_half, 1e-4),
            0.0, 1.0)
        prof = (1.0 - (1.0 - t) ** 2.3) ** 0.55
        prof *= base.smoothstep(0.0, 0.085, t)

        edge_start = joint_half if joint_on else 0.0
        t_edge = np.clip(
            (bedge - edge_start) /
            np.maximum(inr_s[ids] - edge_start, 1e-4), 0.0, 1.0)
        edge_frac = float(np.clip(self.params["seam_halo"] * 5.5, 0.05, 0.15))
        height_small = 0.55 + 0.45 * large_gate
        edge_px = np.maximum(
            edge_frac * height_small * local_inr,
            np.minimum(soft_col * height_small, 0.25 * local_inr))
        edge_ramp = base.smoothstep(
            0.0, np.maximum(edge_px, 1e-4),
            (bedge - edge_start) * self.size)

        # Schematic colour view: marked large stones, real medium cells and
        # procedural pebbles in the voids produced by strong rounding.
        prev = self._topology_preview(
            ids, bedge, centers, inradius, cell_area, n_big, n_med,
            stone_mask, roundness)

        delta = dict(
            ids=ids.astype(np.int16), bedge_raw=bedge_raw.astype(np.float32),
            bedge=bedge.astype(np.float32), prof=prof.astype(np.float32),
            rim=rim.astype(np.float32), inradius=inradius.astype(np.float32),
            cell_area=cell_area.astype(np.float32),
            centers=centers.astype(np.float32), weights=weights.astype(np.float32),
            n_big=np.int32(n_big), n_med=np.int32(n_med),
            dist_core=dist_core.astype(np.float32),
            stone_mask=stone_mask.astype(np.float32),
            big_mask=big_mask.astype(np.float32),
            med_mask=med_mask.astype(np.float32),
            gap=gap.astype(np.float32),
            stone_col_mask=stone_col_mask.astype(np.float32),
            gap_col_mask=gap_col_mask.astype(np.float32),
            stone_h=stone_h.astype(np.float32), gap_h=gap_h.astype(np.float32),
            inr_s=inr_s.astype(np.float32), t_edge=t_edge.astype(np.float32),
            edge_ramp=edge_ramp.astype(np.float32),
            edge_start=np.float32(edge_start),
            local_inr_norm=local_inr_norm.astype(np.float32),
            local_inr=local_inr.astype(np.float32),
            large_gate=large_gate.astype(np.float32),
            soft_col_px=soft_col_px.astype(np.float32),
            soft_col=np.float32(soft_col),
            joint_half=np.float32(joint_half),
            seam_half=np.float32(seam_half),
            joint_on=np.int8(1 if joint_on else 0),
            joint_px_eff=np.float32(joint_px_eff),
            round_px=np.float32(round_px if joint_on else 0.0),
            rim_w=np.float32(rim_w),
            edge_frac=np.float32(edge_frac),
            seam=np.zeros((h, w), np.float32),
        )
        msg = ("%d крупных + %d средних камней; шов %.1f px"
               % (n_big, n_med, joint_px_eff))
        if n_med and float(self.params.get("medium_min_px", 0.0)) > 0.5:
            msg += "; min среднего %.0f px@2048" % self.params["medium_min_px"]
        msg += "; скругление %.2f" % roundness
        return delta, prev, msg

    # -- 2. stone colours ------------------------------------------------
    def _stage_color(self, state: Dict[str, np.ndarray]):
        h = w = self.size
        cfg = self.cfg
        n_low_s = self.nfield("n_low_s")
        n_med_s = self.nfield("n_med_s")
        n_fine_s = self.nfield("n_fine_s")
        n_stri = self.nfield("n_stri")
        n_tint = self.nfield("n_tint")
        X0, Y0 = self._grid()
        ids = state["ids"].astype(np.int32)
        n_big = int(state["n_big"])
        n_med = int(state["n_med"])
        n_total = n_big + n_med
        joint_on = bool(int(state["joint_on"]))
        joint_half = float(state["joint_half"])
        seam_half = float(state["seam_half"])

        rng = np.random.default_rng(self.seed + 101)
        pal, pal_w = cfg["pal"], cfg["pal_w"]
        cell_col = np.zeros((n_total, 3), np.float32)
        pal_p = pal_w / pal_w.sum()
        cell_col[:n_big] = pal[rng.choice(len(pal), n_big, p=pal_p)]
        cell_col[:n_big] *= rng.uniform(0.97, 1.08, (n_big, 1)).astype(np.float32)
        if n_med:
            cell_col[n_big:] = pal[rng.choice(len(pal), n_med, p=pal_p)]
            cell_col[n_big:] *= rng.uniform(0.95, 1.06, (n_med, 1)).astype(np.float32)

        pal_base = (pal * pal_w[:, None]).sum(0) / max(pal_w.sum(), 1e-6)
        cell_col = 0.90 * cell_col + 0.10 * pal_base[None, :]
        lum = cell_col @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        lum_floor = {"aged": 0.50, "cyclopean": 0.58}.get(self.variant, 0.60)
        cell_col *= np.maximum(1.0, lum_floor / np.maximum(lum, 1e-6))[:, None]
        color_variation = float(np.clip(
            float(self.p("color", "color_variation", 1.0)), 0.0, 2.0))
        cell_col = pal_base[None, :] + (cell_col - pal_base[None, :]) * color_variation
        cell_col = np.clip(cell_col, 0.04, 1.0)
        color_shade = float(np.clip(
            float(self.p("color", "color_shade", 1.0)), 0.0, 2.0))

        cell_hj = rng.normal(0.0, 1.0, n_total).astype(np.float32)
        cell_rot = rng.uniform(0.0, math.pi, n_total).astype(np.float32)
        cell_asp = rng.uniform(0.82, 1.18, n_total).astype(np.float32)
        cell_freq = rng.uniform(7.0, 13.0, n_total).astype(np.float32)
        cell_phase = rng.uniform(0.0, 2.0 * math.pi, n_total).astype(np.float32)
        cell_rj = rng.normal(0.0, 1.0, n_total).astype(np.float32)

        inradius = state["inradius"].astype(np.float32)
        centers = state["centers"].astype(np.float32)
        prof = state["prof"].astype(np.float32)
        dxc = base.wrap(X0 - centers[ids, 0])
        dyc = base.wrap(Y0 - centers[ids, 1])
        ca = np.cos(cell_rot[ids])
        sa = np.sin(cell_rot[ids])
        u = dxc * ca + dyc * sa
        v = -dxc * sa + dyc * ca
        a_cap = inradius[ids] * 0.62 * cell_asp[ids]
        b_cap = inradius[ids] * 0.62 / cell_asp[ids]
        rr = np.sqrt((u / np.maximum(a_cap, 1e-4)) ** 2 +
                     (v / np.maximum(b_cap, 1e-4)) ** 2)
        cap = (1.0 - np.clip(rr, 0.0, 1.0) ** 2) ** 0.55
        prof_shape = prof * (0.90 + 0.10 * cap)
        stri = np.sin(2.0 * math.pi * (u * cell_freq[ids] + cell_phase[ids]))
        stri *= 0.5 + 0.5 * n_stri

        if joint_on:
            seam = np.zeros((h, w), np.float32)
        else:
            seam = 1.0 - base.smoothstep(
                seam_half * 0.45, seam_half * 1.35, state["bedge_raw"])

        cell_col_px = cell_col[ids]
        cell_col_px *= (0.98 + 0.06 * n_low_s + 0.030 * n_med_s
                        + 0.020 * n_fine_s)[..., None]
        cell_col_px *= (1.0 + 0.028 * stri)[..., None]
        shade_amt = np.where(ids >= n_big, 0.09, 0.05) * color_shade
        cell_col_px *= (1.0 - shade_amt[..., None] *
                        self.params["dome"] * (1.0 - prof_shape[..., None]))
        cell_col_px *= (1.0 + 0.030 * n_tint)[..., None]

        mortar_col = cfg["mortar"] if joint_on else np.array(
            [0.095, 0.070, 0.042], np.float32)
        gap_col = np.clip(mortar_col * 2.20 + 0.04, 0.0, 1.0)
        gap = state["gap"].astype(np.float32)
        soft = max(1.0, 0.010 * self.size)
        gap_wide = base.gauss_blur(gap, soft)
        wide_fill = base.smoothstep(0.30, 0.60, gap_wide)
        wide_line = base.smoothstep(0.68, 0.90, gap_wide)
        fill_col = gap_col[None, None, :]
        cs = max(1.5, 0.004 * self.size)
        cell_col_px = np.stack([
            base.gauss_blur(cell_col_px[..., c], cs)
            for c in range(3)], axis=-1).astype(np.float32)

        prev = (cell_col_px * state["stone_col_mask"][..., None] +
                fill_col * state["gap_col_mask"][..., None])
        prev = np.clip(prev, 0, 1)
        delta = dict(
            col=prev.astype(np.float32),
            cell_col_px=cell_col_px.astype(np.float32),
            prof_shape=prof_shape.astype(np.float32),
            stri=stri.astype(np.float32),
            cell_hj=cell_hj.astype(np.float32),
            cell_rj=cell_rj.astype(np.float32),
            seam=seam.astype(np.float32),
            gap_col=gap_col.astype(np.float32),
            gap_wide=gap_wide.astype(np.float32),
            wide_fill=wide_fill.astype(np.float32),
            wide_line=wide_line.astype(np.float32),
            fill_col=fill_col.astype(np.float32),
            pal_base=pal_base.astype(np.float32),
        )
        return delta, prev, "камни окрашены"

    # -- 3. seams -------------------------------------------------------
    def _stage_seams(self, state: Dict[str, np.ndarray]):
        """Seam contrast + directional smoothing.

        The old mortar/grout model is intentionally gone.  Stage 3 starts
        with the exact stage-2 colour.  ``seam_contrast`` only darkens a
        narrow boundary line; ``seam_width_px`` fades that correction from
        the boundary into the stone.  Therefore 0/0 is a byte-identical
        no-op relative to stage 2.
        """
        h = w = self.size
        stone_col_mask = state["stone_col_mask"].astype(np.float32)
        if "col" in state:
            base_col = state["col"].astype(np.float32)
        else:
            # Compatibility with colour caches made before stage 2 began
            # storing its final schematic colour.
            cell_col_px = state["cell_col_px"].astype(np.float32)
            fill_col = state["fill_col"].astype(np.float32)
            gap_col_mask = state["gap_col_mask"].astype(np.float32)
            base_col = (cell_col_px * stone_col_mask[..., None] +
                        fill_col * gap_col_mask[..., None])
        base_col = np.clip(base_col, 0.0, 1.0).astype(np.float32)

        width_px = max(0.0, float(self.p("seams", "seam_width_px", 0.0)))
        contrast = float(np.clip(
            float(self.p("seams", "seam_contrast", 0.0)), 0.0, 2.0))
        # Zero contrast means no seam correction at all.  Width likewise
        # cannot invent a colour that is not there.
        if contrast <= 1e-6:
            col = base_col
            ramp = stone_col_mask
            msg = "контраст 0 — точная копия этапа 2"
        else:
            # bedge is zero on the cell boundary; work in current pixels so
            # the slider's number means what it says.  At width 0 the edge is
            # a sharp (but still antialiased) line; positive width is a
            # smooth falloff from the boundary toward the stone centre.
            s_px = np.maximum(state["bedge"].astype(np.float32) * self.size,
                              0.0)
            width_eff = max(1.5, width_px)
            u = np.clip(s_px / width_eff, 0.0, 1.0)
            smooth = u * u * (3.0 - 2.0 * u)
            alpha = contrast * (1.0 - smooth)
            alpha = np.clip(alpha, 0.0, 1.0)
            dark = np.array([0.075, 0.052, 0.032], np.float32)
            col = (base_col * (1.0 - alpha[..., None]) +
                   dark[None, None, :] * alpha[..., None])
            col = np.clip(col, 0.0, 1.0).astype(np.float32)

            # Post-process the already smoothed seam correction.  The
            # correction is measured as (current - stage2 colour), so this
            # changes only depth/darkness, never hue.
            #   0.0  -> deeper, darker groove
            #   0.5  -> neutral (unchanged)
            #   1.0  -> correction removed; seam is level with the stones
            darkness = float(np.clip(
                float(self.p("seams", "seam_darkness", 0.5)), 0.0, 1.0))
            if abs(darkness - 0.5) > 1e-6:
                correction = col - base_col
                if darkness < 0.5:
                    factor = 1.0 + ((0.5 - darkness) / 0.5) * 0.8
                else:
                    factor = (1.0 - darkness) / 0.5
                col = np.clip(base_col + correction * factor, 0.0, 1.0)
                col = col.astype(np.float32)

            ramp = stone_col_mask
            msg = ("шов: контраст %.2f, сглаживание %.1f px, темнота %.2f")
            msg = msg % (contrast, width_px, darkness)

        delta = dict(
            col=col.astype(np.float32), ramp=ramp.astype(np.float32),
            vis=stone_col_mask.astype(np.float32),
            joint_line=np.zeros((h, w), np.float32),
        )
        return delta, col, msg

    # -- 4. moss --------------------------------------------------------
    def _stage_moss(self, state: Dict[str, np.ndarray]):
        n_pit = self.nfield("n_pit")
        n_low = self.nfield("n_low")
        col = state["col"].astype(np.float32)
        joint_on = bool(int(state["joint_on"]))
        moss_amt = float(np.clip(float(self.p(
            "moss", "moss_amount", self.cfg["moss_amt"])), 0.0, 1.0))
        if moss_amt > 0:
            base_vis = state["ramp"] if joint_on else state["stone_col_mask"]
            moss_mask = base.smoothstep(0.55, 0.95, n_pit) * (1.0 - base_vis)
            moss_mask *= base.smoothstep(-0.2, 0.6, n_low)
            moss = np.array([0.205, 0.250, 0.110], np.float32)
            col = (col * (1.0 - moss_amt * moss_mask[..., None]) +
                   moss * (moss_amt * moss_mask)[..., None])
            msg = "мох: %.2f" % moss_amt
        else:
            moss_mask = np.zeros_like(col[..., 0])
            msg = "для варианта %s мох отключён" % self.variant
        col = np.clip(col, 0, 1)
        return dict(col=col.astype(np.float32),
                    moss_mask=moss_mask.astype(np.float32)), col, msg

    # -- 5. chips -------------------------------------------------------
    def _stage_chips(self, state: Dict[str, np.ndarray]):
        h = w = self.size
        n_pit = self.nfield("n_pit")
        ids = state["ids"].astype(np.int32)
        n_big = int(state["n_big"])
        stone_mask = state["stone_mask"].astype(np.float32)
        centers = state["centers"].astype(np.float32)
        inradius = state["inradius"].astype(np.float32)
        col = state["col"].astype(np.float32)
        rng = np.random.default_rng(self.seed + 120)
        chip_count = int(float(self.p(
            "chips", "chip_count", self.params["chip_count"])))
        chip_strength = float(np.clip(
            float(self.p("chips", "chip_strength", 1.0)), 0.0, 2.0))
        chip_depth_scale = float(np.clip(
            float(self.p("chips", "chip_depth", 1.0)), 0.0, 2.0))
        # More than one chip per large stone is allowed.  The first pass
        # visits every large stone once, then extra chips are distributed
        # randomly, so the slider really goes up to 200 instead of being
        # capped at n_big // 3.
        n_chips = int(np.clip(chip_count, 0, 200))
        if n_chips <= 0:
            chip_ids = np.zeros(0, np.int32)
        elif n_chips <= n_big:
            chip_ids = rng.choice(n_big, size=n_chips, replace=False)
        else:
            first = rng.permutation(n_big)
            extra = rng.choice(n_big, size=n_chips - n_big, replace=True)
            chip_ids = np.concatenate([first, extra]).astype(np.int32)
        n_chip_large = max(1, int(round(n_chips * 0.15))) if n_chips >= 4 else 0
        n_chip_med = max(1, int(round(n_chips * 0.30))) if n_chips >= 2 else 0
        n_chip_small = max(0, n_chips - n_chip_large - n_chip_med)
        if n_chip_large + n_chip_med + n_chip_small < n_chips:
            n_chip_small += n_chips - n_chip_large - n_chip_med
        chip_scales = np.concatenate([
            10.0 ** rng.uniform(math.log10(0.030), math.log10(0.080), n_chip_small),
            10.0 ** rng.uniform(math.log10(0.090), math.log10(0.200), n_chip_med),
            10.0 ** rng.uniform(math.log10(0.220), math.log10(0.360), n_chip_large),
        ]).astype(np.float32) if n_chips else np.zeros(0, np.float32)
        rng.shuffle(chip_scales)
        chip_mask = np.zeros((h, w), np.float32)
        chip_depth = np.zeros((h, w), np.float32)
        chip_core = np.zeros((h, w), np.float32)
        for k, cid in enumerate(chip_ids):
            cx, cy = centers[cid]
            inr = float(inradius[cid])
            r_chip = inr * float(chip_scales[k])
            max_off = 0.55 * max(0.0, 1.0 - r_chip / max(inr, 1e-9)) * inr
            off = rng.uniform(0.0, max_off)
            aoff = rng.uniform(0.0, 2.0 * math.pi)
            ccx = (cx + off * math.cos(aoff)) % 1.0
            ccy = (cy + off * math.sin(aoff)) % 1.0
            px = int(ccx * w) % w
            py = int(ccy * h) % h
            if int(ids[py, px]) != int(cid) or stone_mask[py, px] < 0.55:
                ccx, ccy = cx, cy
            aspect = rng.uniform(1.0, 2.0)
            rx = r_chip * math.sqrt(aspect)
            ry = r_chip / math.sqrt(aspect)
            ang = rng.uniform(0.0, math.pi)
            lobes = int(rng.integers(3, 7))
            lobe_a = rng.uniform(0.03, 0.17)
            lobe_b = rng.uniform(0.02, 0.08)
            ph1 = rng.uniform(0.0, 2.0 * math.pi)
            ph2 = rng.uniform(0.0, 2.0 * math.pi)
            depth = rng.uniform(0.12, 0.26) + 0.20 * (r_chip / max(inr, 1e-9))
            # Work only in the chip's little bounding box.  This is what
            # makes 200 chips practical even at 2K.
            r_px = max(rx, ry) * 1.8 * self.size + 4.0
            x0 = int(math.floor(ccx * w - r_px))
            x1 = int(math.ceil(ccx * w + r_px)) + 1
            y0 = int(math.floor(ccy * h - r_px))
            y1 = int(math.ceil(ccy * h + r_px)) + 1
            xidx = np.arange(x0, x1) % w
            yidx = np.arange(y0, y1) % h
            xs = (np.arange(x0, x1) + 0.5) / w
            ys = (np.arange(y0, y1) + 0.5) / h
            dx = base.wrap(xs[None, :] - ccx)
            dy = base.wrap(ys[:, None] - ccy)
            ca2, sa2 = math.cos(ang), math.sin(ang)
            uu = dx * ca2 + dy * sa2
            vv = -dx * sa2 + dy * ca2
            qc = np.sqrt((uu / max(rx, 1e-9)) ** 2 +
                         (vv / max(ry, 1e-9)) ** 2)
            th = np.arctan2(vv, uu)
            qc *= (1.0 + lobe_a * np.sin(lobes * th + ph1) +
                   lobe_b * np.sin((lobes + 2) * th + ph2))
            local_ids = ids[np.ix_(yidx, xidx)]
            local_stone = stone_mask[np.ix_(yidx, xidx)]
            local_pit = n_pit[np.ix_(yidx, xidx)]
            qc *= (1.0 + 0.06 * local_pit)
            hole = 1.0 - base.smoothstep(0.72, 1.15, qc)
            hole *= local_stone * (local_ids == cid)
            core = 1.0 - base.smoothstep(0.10, 0.60, qc)
            chip_mask[np.ix_(yidx, xidx)] = np.maximum(
                chip_mask[np.ix_(yidx, xidx)], hole)
            chip_depth[np.ix_(yidx, xidx)] = np.maximum(
                chip_depth[np.ix_(yidx, xidx)], hole * depth)
            chip_core[np.ix_(yidx, xidx)] = np.maximum(
                chip_core[np.ix_(yidx, xidx)],
                core * local_stone * (local_ids == cid))

        chip_depth = chip_depth * chip_depth_scale
        chip_col = np.array([0.290, 0.220, 0.125], np.float32)
        cs = 0.22 * chip_strength
        col = (col * (1.0 - cs * chip_mask[..., None]) +
               chip_col * (cs * chip_mask)[..., None])
        core_mix = 0.07 * chip_strength
        col = (col * (1.0 - core_mix * chip_core[..., None]) +
               chip_col * (core_mix * chip_core)[..., None])
        lip = chip_mask * (1.0 - chip_core)
        col = np.clip(col * (1.0 + 0.055 * chip_strength * lip[..., None]),
                      0.0, 1.0)

        # A small fake-bevel light makes the *depth* slider visible in this
        # stage's albedo preview as well as in the later height/normal maps.
        # It is derived from chip_depth, so depth=0 is a strict no-op here.
        if float(chip_depth.max()) > 1e-6:
            depth_shade = base.gauss_blur(
                chip_depth, max(1.0, 0.0015 * self.size))
            gx = (np.roll(depth_shade, -1, axis=1) -
                  np.roll(depth_shade, 1, axis=1)) * 0.5
            gy = (np.roll(depth_shade, -1, axis=0) -
                  np.roll(depth_shade, 1, axis=0)) * 0.5
            bevel = np.clip(-1.8 * gx - 1.8 * gy, -0.55, 0.55)
            col = col * (1.0 + bevel[..., None])
            # deeper centres read darker, like a real hollow
            col = col * (1.0 - 0.22 * np.clip(chip_depth, 0.0, 1.0)[..., None])
            col = np.clip(col, 0.0, 1.0).astype(np.float32)

        return (dict(col=col.astype(np.float32),
                     chip_mask=chip_mask.astype(np.float32),
                     chip_depth=chip_depth.astype(np.float32),
                     chip_core=chip_core.astype(np.float32)),
                col, "сколы: %d, глубина x%.2f" % (n_chips, chip_depth_scale))

    # -- 6. cracks ------------------------------------------------------
    def _stage_cracks(self, state: Dict[str, np.ndarray]):
        size = self.size
        stone_mask = state["stone_mask"].astype(np.float32)
        col = state["col"].astype(np.float32)
        count = int(float(self.p(
            "cracks", "crack_count", self.params["crack_count"])))
        crack_strength = float(np.clip(
            float(self.p("cracks", "crack_strength", 1.0)), 0.0, 2.0))
        if count > 0:
            crack = base.make_cracks(size, self.seed, count) * stone_mask
            crack = crack * crack_strength
            col = col * (1.0 - 0.035 * crack[..., None])
            msg = "трещины: %d, сила x%.2f" % (count, crack_strength)
        else:
            crack = np.zeros((size, size), np.float32)
            msg = "трещины отключены (0)"
        col = np.clip(col, 0, 1)
        return (dict(col=col.astype(np.float32),
                     crack=crack.astype(np.float32)), col, msg)

    # -- 7. speckles ----------------------------------------------------
    def _stage_speckles(self, state: Dict[str, np.ndarray]):
        n_fine2 = self.nfield("n_fine2")
        n_tint = self.nfield("n_tint")
        n_spot = self.nfield("n_spot")
        n_dark = self.nfield("n_dark")
        col = state["col"].astype(np.float32)
        joint_on = bool(int(state["joint_on"]))
        stone_vis = (state["ramp"] if joint_on
                     else np.clip(state["big_mask"] + state["med_mask"], 0, 1))
        speckle_amount = float(np.clip(
            float(self.p("speckles", "speckle_amount", 1.0)), 0.0, 2.0))
        fleck = base.smoothstep(1.75, 2.25, n_fine2)
        salmon = np.array([0.96, 0.74, 0.52], np.float32)
        yellow = np.array([0.88, 0.74, 0.12], np.float32)
        pick = base.smoothstep(0.0, 1.0, 0.5 + 0.5 * n_tint)[..., None]
        fleck_col = salmon[None, None, :] * pick + yellow[None, None, :] * (1.0 - pick)
        f_amt = np.clip(0.42 * speckle_amount, 0.0, 1.0)
        col = (col * (1.0 - f_amt * fleck[..., None] * stone_vis[..., None]) +
               fleck_col * (f_amt * fleck * stone_vis)[..., None])
        spot = base.smoothstep(2.05, 2.45, n_spot) * stone_vis
        spot_col = np.array([0.86, 0.58, 0.38], np.float32)
        s_amt = np.clip(0.28 * speckle_amount, 0.0, 1.0)
        col = (col * (1.0 - s_amt * spot[..., None]) +
               spot_col * (s_amt * spot)[..., None])
        grain = base.smoothstep(2.15, 2.60, n_dark) * stone_vis
        grain_col = np.array([0.36, 0.25, 0.12], np.float32)
        g_amt = np.clip(0.30 * speckle_amount, 0.0, 1.0)
        col = (col * (1.0 - g_amt * grain[..., None]) +
               grain_col * (g_amt * grain)[..., None])
        col = np.clip(col, 0, 1)
        return dict(col=col.astype(np.float32)), col, "вкрапления добавлены"

    def _compute_height(self, state: Dict[str, np.ndarray], stage: str):
        n_med_s = self.nfield("n_med_s")
        n_fine_s = self.nfield("n_fine_s")
        joint_on = bool(int(state["joint_on"]))
        cfg = self.cfg
        dome = float(np.clip(
            float(self.p(stage, "dome", self.params["dome"])), 0.0, 2.0))
        relief = float(np.clip(
            float(self.p(stage, "height_relief", 1.0)), 0.1, 3.0))
        prof = state["prof"].astype(np.float32)
        prof_shape = state["prof_shape"].astype(np.float32)
        stri = state["stri"].astype(np.float32)
        ids = state["ids"].astype(np.int32)
        chip_depth = state["chip_depth"].astype(np.float32)
        edge_ramp = state["edge_ramp"].astype(np.float32)
        seam = state["seam"].astype(np.float32)
        mortar_h = cfg["big_base"] + (0.035 if joint_on else -0.002)
        dome_shape = dome * prof_shape + 0.05 * (1.0 - dome) * prof
        h_cell = (cfg["big_base"] + cfg["big_amp"] * dome_shape +
                  0.010 * n_med_s + 0.0022 * n_fine_s +
                  0.006 * stri + 0.010 * state["cell_hj"][ids])
        height = mortar_h + (h_cell - mortar_h) * edge_ramp
        if not joint_on:
            height -= 0.012 * seam
        height -= 0.65 * chip_depth
        # Cracks are an albedo/roughness feature at this stage.  They are
        # deliberately not carved into height (and therefore not into
        # normals/AO either).
        height = base.gauss_blur(height, max(1.2, 0.0018 * self.size))
        edge_w = base.smoothstep(0.0, 0.35, 1.0 - edge_ramp)
        h_soft = base.gauss_blur(height, max(2.5, 0.0040 * self.size))
        height = height * (1.0 - 0.45 * edge_w) + h_soft * (0.45 * edge_w)
        lo, hi = np.percentile(height, [0.2, 99.8])
        height = np.clip((height - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        height = 0.32 + 0.68 * height
        height = 0.5 + (height - 0.5) * relief
        return np.clip(height, 0.0, 1.0).astype(np.float32)

    def _compute_ao(self, height: np.ndarray, state: Dict[str, np.ndarray],
                    stage: str):
        hs = base.gauss_blur(height, 1.4)
        cav = np.maximum(0.0, base.gauss_blur(hs, 4.0) - hs)
        cav2 = np.maximum(0.0, base.gauss_blur(hs, 18.0) - hs)
        gap_col_mask = state["gap_col_mask"].astype(np.float32)
        rim = state["rim"].astype(np.float32)
        stone_col_mask = state["stone_col_mask"].astype(np.float32)
        seam = state["seam"].astype(np.float32)
        ao = 1.0 - np.clip(cav * 4.2, 0.0, 0.55) - np.clip(cav2 * 1.7, 0.0, 0.35)
        ao *= (1.0 - 0.08 * gap_col_mask)
        ao *= (1.0 - 0.08 * rim * stone_col_mask)
        ao *= (1.0 - 0.30 * seam)
        ao = np.clip(ao, 0.20, 1.0).astype(np.float32)
        ao = base.gauss_blur(ao, 0.8)
        strength = float(np.clip(
            float(self.p(stage, "ao_strength", 1.0)), 0.0, 2.0))
        ao = 1.0 - (1.0 - ao) * strength
        return np.clip(ao, 0.0, 1.0).astype(np.float32)

    # -- 8. final albedo (also computes height + AO for baking) ---------
    def _stage_albedo(self, state: Dict[str, np.ndarray]):
        h = w = self.size
        height = self._compute_height(state, "albedo")
        ao = self._compute_ao(height, state, "albedo")

        baked_ao = float(np.clip(
            float(self.p("albedo", "baked_ao", 0.04)), 0.0, 1.0))
        col = state["col"].astype(np.float32) * (
            1.0 - baked_ao * (1.0 - ao[..., None]))
        col = np.clip(col, 0, 1)
        return (dict(col=col.astype(np.float32), height=height.astype(np.float32),
                     ao=ao.astype(np.float32)),
                col, {"albedo": col},
                "итоговый albedo (%dx%d)" % (w, h))

    # -- 9. height ------------------------------------------------------
    def _stage_height(self, state: Dict[str, np.ndarray]):
        height = self._compute_height(state, "height")
        prev = np.repeat(height[..., None], 3, axis=2)
        return (dict(height=height), prev, {"height": height},
                "height готов (купол %.2f)" %
                float(self.p("height", "dome", self.params["dome"])))

    # -- 10. roughness --------------------------------------------------
    def _stage_roughness(self, state: Dict[str, np.ndarray]):
        n_med_s = self.nfield("n_med_s")
        ids = state["ids"].astype(np.int32)
        n_big = int(state["n_big"])
        stone_col_mask = state["stone_col_mask"].astype(np.float32)
        gap_col_mask = state["gap_col_mask"].astype(np.float32)
        chip_mask = state["chip_mask"].astype(np.float32)
        crack = state["crack"].astype(np.float32)
        seam = state["seam"].astype(np.float32)
        stri = state["stri"].astype(np.float32)
        cell_rj = state["cell_rj"].astype(np.float32)
        rough_cell = (self.cfg["big_rough"] + 0.035 * n_med_s +
                      0.020 * cell_rj[ids] + 0.030 * stri +
                      0.030 * (ids >= n_big))
        rough = rough_cell * stone_col_mask + 0.80 * gap_col_mask
        rough += 0.08 * chip_mask
        rough += 0.06 * crack
        rough = rough * (1.0 - seam) + 0.82 * seam
        rough_bias = float(self.p("roughness", "roughness_bias", 0.0))
        rough_contrast = float(np.clip(
            float(self.p("roughness", "roughness_contrast", 1.0)), 0.1, 3.0))
        rough = 0.60 + (rough - 0.60) * rough_contrast + rough_bias
        rough = np.clip(rough, 0.0, 1.0).astype(np.float32)
        prev = np.repeat(rough[..., None], 3, axis=2)
        return (dict(rough=rough.astype(np.float32)), prev,
                {"roughness": rough},
                "roughness готов (смещение %+.2f)" % rough_bias)

    # -- 11. AO ---------------------------------------------------------
    def _stage_ao(self, state: Dict[str, np.ndarray]):
        height = state["height"].astype(np.float32)
        ao = self._compute_ao(height, state, "ao")
        prev = np.repeat(ao[..., None], 3, axis=2)
        return (dict(ao=ao), prev, {"ao": ao},
                "AO готов (сила %.2f)" %
                float(self.p("ao", "ao_strength", 1.0)))

    # -- 12/13. normals -------------------------------------------------
    def _normal(self, state: Dict[str, np.ndarray], stage: str,
                force: bool = False):
        if not force and "nx" in state:
            return (state["nx"].astype(np.float32),
                    state["ny"].astype(np.float32),
                    state["nz"].astype(np.float32))
        n_fine_s = self.nfield("n_fine_s")
        n_fine2 = self.nfield("n_fine2")
        n_micro = self.nfield("n_micro")
        n_med_s = self.nfield("n_med_s")
        height = state["height"].astype(np.float32)
        ids = state["ids"].astype(np.int32)
        n_big = int(state["n_big"])
        edge_ramp = state["edge_ramp"].astype(np.float32)
        wide_fill = state["wide_fill"].astype(np.float32)
        stone_col_mask = state["stone_col_mask"].astype(np.float32)
        micro = (0.0100 * n_fine_s + 0.0060 * n_fine2 +
                 0.0040 * n_micro + 0.0030 * n_med_s)
        detail_mask = edge_ramp * (1.0 - 0.80 * wide_fill)
        normal_detail = float(np.clip(
            float(self.p(stage, "normal_detail", self.params["normal_detail"])),
            0.0, 4.0))
        normal_strength = float(np.clip(
            float(self.p(stage, "normal_strength", 1.0)), 0.0, 2.0))
        hn = base.gauss_blur(height, 1.1) + normal_detail * micro * detail_mask
        gx = (np.roll(hn, -1, axis=1) - np.roll(hn, 1, axis=1)) * 0.5
        gy = (np.roll(hn, -1, axis=0) - np.roll(hn, 1, axis=0)) * 0.5
        s_big, s_med, s_gap = 68.0, 58.0, 12.0
        s_stone = np.where(ids < n_big, s_big, s_med)
        s = s_gap + (s_stone - s_gap) * stone_col_mask
        s *= (self.size / 2048.0) * normal_strength
        nx = -gx * s
        ny = -gy * s
        nz = np.ones_like(nx)
        ln = np.sqrt(nx * nx + ny * ny + nz * nz)
        return (nx / ln).astype(np.float32), (ny / ln).astype(np.float32), \
            (nz / ln).astype(np.float32)

    def _stage_normal_gl(self, state: Dict[str, np.ndarray]):
        nx, ny, nz = self._normal(state, "normal_gl")
        gl = np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5,
                       nz * 0.5 + 0.5], -1)
        return (dict(nx=nx, ny=ny, nz=nz), gl, {"normal_gl": gl},
                "Normal GL готов")

    def _stage_normal_dx(self, state: Dict[str, np.ndarray]):
        nx, ny, nz = self._normal(state, "normal_dx", force=True)
        dx = np.stack([nx * 0.5 + 0.5, 0.5 - ny * 0.5,
                       nz * 0.5 + 0.5], -1)
        return dict(), dx, {"normal_dx": dx}, "Normal DX готов"
