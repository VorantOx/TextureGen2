# -*- coding: utf-8 -*-
"""
Staged, resumable pipeline for floor and ceiling textures.

This is a second direction next to the irregular wall-cladding pipeline in
``otwn_pipeline.py``.  Instead of a power diagram of uneven stones, it builds
regular (almost symmetric) tile patterns from a small set of templates:

    grid          square tiles
    checker       square tiles with alternating colour groups
    running_bond  staggered brick bond
    basketweave   alternating horizontal/vertical 2:1 bricks
    hex           honeycomb
    octagon       octagons with small square inserts
    concentric    framed tiles with an inner panel

All maps are seamless.  Colours come from the same per-variant palettes and
the same mortar/dirt colours that the wall pipeline uses, so a floor or a
ceiling in ``gold``/``aged``/``cyclopean`` matches a wall in the same variant.

Every stage is cached as ``cache/<stage>_<hash>.npz``; the web editor
(``otwn_editor.py``) drives this class exactly like the wall pipeline.
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
import otwn_pipeline as wallpipe


SURFACE_PIPELINE_VERSION = 2

STAGE_ORDER: List[str] = [
    "layout",
    "color",
    "grout",
    "wear",
    "details",
    "edges",
    "albedo",
    "height",
    "roughness",
    "ao",
    "normal_gl",
    "normal_dx",
]

STAGE_LABELS: Dict[str, str] = {
    "layout": "1. Раскладка плитки",
    "color": "2. Цвет плитки",
    "grout": "3. Швы и затирка",
    "wear": "4. Износ и сколы",
    "details": "5. Прожилки и вкрапления",
    "edges": "6. Сбитые края и растительность",
    "albedo": "7. Albedo — итог",
    "height": "8. Height",
    "roughness": "9. Roughness",
    "ao": "10. AO",
    "normal_gl": "11. Normal GL",
    "normal_dx": "12. Normal DX",
}

STAGE_DEPS: Dict[str, List[str]] = {
    "layout": [],
    "color": ["layout"],
    "grout": ["color"],
    "wear": ["grout"],
    "details": ["wear"],
    "edges": ["details"],
    "albedo": ["edges"],
    "height": ["albedo"],
    "roughness": ["height"],
    "ao": ["height"],
    "normal_gl": ["ao"],
    "normal_dx": ["normal_gl"],
}

STAGE_DESCS: Dict[str, str] = {
    "layout": "Регулярная раскладка по выбранному шаблону. Ширина шва, фаска "
              "и вставка внутренней рамки задаются здесь. Шаблон и число "
              "плиток — в левой панели.",
    "color": "Цвета плиток из палитры выбранного варианта стен. «Созвучие» "
             "подтягивает средний тон к палитре стены, «разброс» оставляет "
             "плитки различимыми.",
    "grout": "Тёмная затирка, её контраст и загрязнение. Ширина шва берётся "
             "из этапа 1.",
    "wear": "Износ: приглушённые кромки, сколы, потёртости и (для потолка) "
            "пятна. Полировка делает пол глаже и светлее; на потолке она "
            "ослаблена.",
    "details": "Тонкие прожилки и минеральные вкрапления в духе OTWNDATA. "
               "Они не ломают регулярный рисунок, а добавляют породу.",
    "edges": "Сбитые кромки плиток и растительность в швах: сколы на краях, "
             "трава и мох, проросшие через затирку. Сколы углубляют рельеф, "
             "растительность добавляет высоту и шероховатость.",
    "albedo": "Итоговый albedo с запечённым AO.",
    "height": "Карта высот: плоские плитки, фаска, шов и неглубокий износ. "
              "Трещины в height не вырезаются.",
    "roughness": "Шероховатость плитки и затирки; полировка и износ её меняют.",
    "ao": "Ambient occlusion по height.",
    "normal_gl": "Normal map для OpenGL (Y+).",
    "normal_dx": "Normal map для DirectX (Y−).",
}

FINAL_MAPS: List[str] = ["albedo", "height", "roughness", "ao",
                         "normal_gl", "normal_dx"]

STAGE_EXPORTS: Dict[str, List[str]] = {
    "albedo": ["albedo"],
    "height": ["height"],
    "roughness": ["roughness"],
    "ao": ["ao"],
    "normal_gl": ["normal_gl"],
    "normal_dx": ["normal_dx"],
}

SURFACE_LABELS: Dict[str, str] = {
    "floor": "Пол",
    "ceiling": "Потолок",
}

TEMPLATE_ORDER: List[str] = [
    "grid", "checker", "running_bond", "basketweave",
    "hex", "octagon", "concentric",
]

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "grid": {
        "label": "Квадратная сетка",
        "hint": "ровные квадраты; самый нейтральный пол",
        "default_tiles": 4,
    },
    "checker": {
        "label": "Шахматка",
        "hint": "квадраты двух цветовых групп",
        "default_tiles": 4,
    },
    "running_bond": {
        "label": "Кирпичная перевязка",
        "hint": "ряды со сдвигом на полплитки",
        "default_tiles": 5,
    },
    "basketweave": {
        "label": "Плетёнка",
        "hint": "горизонтальные и вертикальные плитки 2:1",
        "default_tiles": 3,
    },
    "hex": {
        "label": "Соты",
        "hint": "шестиугольная плитка",
        "default_tiles": 6,
    },
    "octagon": {
        "label": "Восьмиугольники",
        "hint": "восьмиугольники с квадратными вставками",
        "default_tiles": 4,
    },
    "concentric": {
        "label": "Рамка и панно",
        "hint": "плитка с внутренней рамкой; подходит для потолка",
        "default_tiles": 3,
    },
}

# Stage-local controls.  Values not present here are base job parameters and
# are edited in the left panel.
STAGE_PARAM_SPECS: Dict[str, List[Dict[str, Any]]] = {
    "layout": [
        {"key": "grout_px", "label": "Ширина шва, px", "min": 0.0, "max": 60.0,
         "step": 0.5, "default_from": "grout_px",
         "hint": "0 — плитки стыкуются без затирки"},
        {"key": "bevel", "label": "Фаска плитки", "min": 0.0, "max": 2.0,
         "step": 0.01, "default_from": "bevel",
         "hint": "ширина мягкого спуска к шву"},
        {"key": "inset", "label": "Внутренняя рамка", "min": 0.04, "max": 0.45,
         "step": 0.01, "default_from": "inset",
         "hint": "только для шаблона «Рамка и панно»"},
    ],
    "color": [
        {"key": "tile_variation", "label": "Разброс цвета плиток",
         "min": 0.0, "max": 2.0, "step": 0.01, "default_from": "tile_variation",
         "hint": "0 — все плитки одного тона"},
        {"key": "accent_contrast", "label": "Контраст акцентов",
         "min": 0.0, "max": 1.0, "step": 0.01, "default_from": "accent_contrast",
         "hint": "шахматка, вставки, внутренние панно"},
        {"key": "harmony", "label": "Созвучие со стенами",
         "min": 0.0, "max": 1.0, "step": 0.01, "default_from": "harmony",
         "hint": "подтягивает средний тон к палитре выбранного варианта стен"},
        {"key": "color_shade", "label": "Объёмная тень плитки",
         "min": 0.0, "max": 2.0, "step": 0.01, "default": 1.0},
    ],
    "grout": [
        {"key": "grout_contrast", "label": "Контраст шва", "min": 0.0,
         "max": 2.0, "step": 0.01, "default": 0.35,
         "hint": "0 — этап 2 без изменений"},
        {"key": "grout_darkness", "label": "Темнота шва", "min": 0.0,
         "max": 1.0, "step": 0.01, "default": 0.5,
         "hint": "0 — глубокая канавка, 1 — шов на уровне плитки"},
        {"key": "grout_dirt", "label": "Загрязнение затирки", "min": 0.0,
         "max": 2.0, "step": 0.01, "default": 1.0},
    ],
    "wear": [
        {"key": "wear", "label": "Износ", "min": 0.0, "max": 1.0,
         "step": 0.01, "default_from": "wear",
         "hint": "кромки, сколы и потёртости"},
        {"key": "polish", "label": "Полировка", "min": 0.0, "max": 1.0,
         "step": 0.01, "default_from": "polish",
         "hint": "гладкие светлые участки; на потолке ослаблена"},
        {"key": "stains", "label": "Пятна и высолы", "min": 0.0, "max": 2.0,
         "step": 0.01, "default": 0.6,
         "hint": "сильнее читаются на потолке"},
        {"key": "crack_count", "label": "Трещины", "min": 0, "max": 60,
         "step": 1, "default_from": "crack_count"},
    ],
    "details": [
        {"key": "veins", "label": "Прожилки", "min": 0.0, "max": 2.0,
         "step": 0.01, "default_from": "veins"},
        {"key": "flecks", "label": "Минеральные вкрапления", "min": 0.0,
         "max": 2.0, "step": 0.01, "default_from": "flecks"},
    ],
    "edges": [
        {"key": "edge_chips", "label": "Сбитые края плиток", "min": 0.0,
         "max": 1.0, "step": 0.01, "default_from": "edge_chips",
         "hint": "сколы и выщербины на кромках"},
        {"key": "edge_chip_size", "label": "Крупность сколов", "min": 0.0,
         "max": 2.0, "step": 0.01, "default_from": "edge_chip_size"},
        {"key": "grass", "label": "Трава в швах", "min": 0.0, "max": 2.0,
         "step": 0.01, "default_from": "grass",
         "hint": "на потолке автоматически ослаблена"},
        {"key": "moss", "label": "Мох в швах", "min": 0.0, "max": 2.0,
         "step": 0.01, "default_from": "moss"},
        {"key": "growth_width", "label": "Ширина прорастания", "min": 0.0,
         "max": 2.0, "step": 0.01, "default_from": "growth_width",
         "hint": "насколько далеко трава и мох заходят на плитку"},
    ],
    "albedo": [
        {"key": "baked_ao", "label": "Запечённое AO в albedo",
         "min": 0.0, "max": 0.30, "step": 0.005, "default": 0.035},
    ],
    "height": [
        {"key": "tile_dome", "label": "Купол плитки", "min": 0.0, "max": 2.0,
         "step": 0.01, "default_from": "tile_dome"},
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
    "surface": "floor",
    "template": "grid",
    "tiles": 4,
    "grout_px": 16.0,
    "bevel": 1.0,
    "inset": 0.25,
    "tile_variation": 1.0,
    "accent_contrast": 0.35,
    "harmony": 0.70,
    "wear": 0.35,
    "polish": 0.45,
    "veins": 1.0,
    "flecks": 1.0,
    "edge_chips": 0.35,
    "edge_chip_size": 1.0,
    "grass": 0.60,
    "moss": 0.50,
    "growth_width": 1.0,
    "crack_count": 0,
    "tile_dome": 0.55,
    "normal_detail": 1.0,
}

SURFACES = ["floor", "ceiling"]


def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(DEFAULT_PARAMS)
    p.update({k: v for k, v in (params or {}).items() if k in DEFAULT_PARAMS})
    p["size"] = int(np.clip(int(p["size"]), 128, 4096))
    p["seed"] = int(p["seed"])
    p["variant"] = str(p["variant"])
    if p["variant"] not in wallpipe.VARIANTS:
        p["variant"] = "gold"
    p["surface"] = str(p["surface"])
    if p["surface"] not in SURFACES:
        p["surface"] = "floor"
    p["template"] = str(p["template"])
    if p["template"] not in TEMPLATES:
        p["template"] = "grid"
    p["tiles"] = int(np.clip(int(p["tiles"]), 2, 16))
    p["grout_px"] = float(np.clip(float(p["grout_px"]), 0.0, 60.0))
    p["bevel"] = float(np.clip(float(p["bevel"]), 0.0, 2.0))
    p["inset"] = float(np.clip(float(p["inset"]), 0.04, 0.45))
    p["tile_variation"] = float(np.clip(float(p["tile_variation"]), 0.0, 2.0))
    p["accent_contrast"] = float(np.clip(float(p["accent_contrast"]), 0.0, 1.0))
    p["harmony"] = float(np.clip(float(p["harmony"]), 0.0, 1.0))
    p["wear"] = float(np.clip(float(p["wear"]), 0.0, 1.0))
    p["polish"] = float(np.clip(float(p["polish"]), 0.0, 1.0))
    p["veins"] = float(np.clip(float(p["veins"]), 0.0, 2.0))
    p["flecks"] = float(np.clip(float(p["flecks"]), 0.0, 2.0))
    p["edge_chips"] = float(np.clip(float(p["edge_chips"]), 0.0, 1.0))
    p["edge_chip_size"] = float(np.clip(float(p["edge_chip_size"]), 0.0, 2.0))
    p["grass"] = float(np.clip(float(p["grass"]), 0.0, 2.0))
    p["moss"] = float(np.clip(float(p["moss"]), 0.0, 2.0))
    p["growth_width"] = float(np.clip(float(p["growth_width"]), 0.0, 2.0))
    p["crack_count"] = int(np.clip(int(p["crack_count"]), 0, 60))
    p["tile_dome"] = float(np.clip(float(p["tile_dome"]), 0.0, 2.0))
    p["normal_detail"] = float(np.clip(float(p["normal_detail"]), 0.0, 4.0))
    return p


def normalize_stage_params(
        overrides: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
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


def _surface_default(params: Dict[str, Any], stage: str, key: str) -> Optional[float]:
    """Surface-specific defaults for the few controls that differ."""
    surface = params.get("surface", "floor")
    if stage == "wear":
        if key == "wear":
            return 0.42 if surface == "floor" else 0.20
        if key == "polish":
            return 0.50 if surface == "floor" else 0.12
        if key == "stains":
            return 0.35 if surface == "floor" else 1.00
    return None


def resolved_stage_specs(
        params: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
    p = normalize_params(params)
    ov = normalize_stage_params(overrides)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for stage, spec in STAGE_PARAM_SPECS.items():
        rows = []
        for item in spec:
            row = dict(item)
            special = _surface_default(p, stage, item["key"])
            if "default_from" in item:
                key = item["default_from"]
                default = p.get(key, item.get("default", 0.0))
            elif special is not None:
                default = special
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
    blob += str(SURFACE_PIPELINE_VERSION).encode("ascii")
    return hashlib.sha1(blob).hexdigest()[:12]


def _save_png_fast(path: str, arr) -> None:
    a = (np.clip(np.asarray(arr), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(a).save(path, optimize=False, compress_level=1)


def _suffix_for_size(size: int) -> str:
    return {512: "512", 1024: "1k", 2048: "2k", 4096: "4k"}.get(
        int(size), str(int(size)))


def surface_config(surface: str) -> Dict[str, Any]:
    if surface == "ceiling":
        return dict(
            dome=0.45, rough_add=0.22, wear_scale=0.40, polish_scale=0.20,
            stain_scale=1.00, dust=np.array([0.30, 0.27, 0.22], np.float32),
            edge_shade=1.10, micro=1.10, chip_scale=0.60,
            grass_scale=0.30, moss_scale=1.20,
        )
    return dict(
        dome=1.00, rough_add=0.02, wear_scale=1.00, polish_scale=1.00,
        stain_scale=1.00, dust=np.array([0.32, 0.27, 0.20], np.float32),
        edge_shade=1.00, micro=1.00, chip_scale=1.00,
        grass_scale=1.00, moss_scale=1.00,
    )


def template_list() -> List[Dict[str, Any]]:
    return [dict(id=tid, **TEMPLATES[tid]) for tid in TEMPLATE_ORDER]


def meta() -> Dict[str, Any]:
    return {
        "stage_order": list(STAGE_ORDER),
        "stage_labels": dict(STAGE_LABELS),
        "stage_descs": dict(STAGE_DESCS),
        "stage_deps": dict(STAGE_DEPS),
        "stage_param_specs": STAGE_PARAM_SPECS,
        "default_params": dict(DEFAULT_PARAMS),
        "final_maps": list(FINAL_MAPS),
        "templates": template_list(),
        "surfaces": dict(SURFACE_LABELS),
    }


class Pipeline:
    def __init__(self, job_dir: str, params: Dict[str, Any],
                 overrides: Optional[Dict[str, Any]] = None,
                 topology_edit: Optional[Dict[str, Any]] = None):
        self.job_dir = os.path.abspath(job_dir)
        self.params = normalize_params(params)
        self.overrides = normalize_stage_params(overrides)
        # Kept for interface compatibility with the wall pipeline.
        self.topology_edit = topology_edit or {}
        self.hash = params_hash(self.params)
        self.size = int(self.params["size"])
        self.seed = int(self.params["seed"])
        self.variant = self.params["variant"]
        self.surface = self.params["surface"]
        self.template = self.params["template"]
        self.cfg = wallpipe.variant_config(self.variant)
        self.scfg = surface_config(self.surface)
        self._noise_cache: Dict[Any, np.ndarray] = {}
        self.cache_dir = os.path.join(self.job_dir, "cache")
        self.preview_dir = os.path.join(self.job_dir, "previews")
        self.export_dir = os.path.join(self.job_dir, "exports")
        for d in (self.cache_dir, self.preview_dir, self.export_dir):
            os.makedirs(d, exist_ok=True)

    def p(self, stage: str, key: str, default=None):
        ov = self.overrides.get(stage, {}) or {}
        if key in ov:
            return ov[key]
        if default is None:
            default = self.params.get(key)
        return default

    # ------------------------------------------------------------------
    # cache helpers (same layout as the wall pipeline)
    # ------------------------------------------------------------------
    def stage_path(self, stage: str) -> str:
        return os.path.join(self.cache_dir, "%s_%s.npz" % (stage, self.hash))

    def preview_path(self, stage: str) -> str:
        return os.path.join(self.preview_dir, "%s_%s.png" % (stage, self.hash))

    def idmap_path(self, stage: str = "layout") -> str:
        return os.path.join(self.preview_dir, "%s_ids_%s.png" %
                            (stage, self.hash))

    def export_path(self, name: str) -> str:
        return os.path.join(
            self.export_dir, "%s_%s.png" % (name, _suffix_for_size(self.size)))

    def load_until(self, stage: str) -> Dict[str, np.ndarray]:
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
    # noise / grid
    # ------------------------------------------------------------------
    NOISE_SPEC = {
        "n_low": (3.2, 1),
        "n_med": (2.0, 2),
        "n_fine": (0.9, 3),
        "n_fine2": (0.7, 4),
        "n_tint": (2.4, 5),
        "n_pit": (1.5, 7),
        "n_stri": (1.2, 8),
        "n_dark": (0.8, 9),
        "n_spot": (1.3, 10),
        "n_micro": (0.55, 11),
        "n_stain": (2.8, 12),
        "n_vein": (1.4, 13),
        "n_vein2": (2.2, 14),
    }

    def nfield(self, name: str) -> np.ndarray:
        key = (name, self.size, self.seed)
        if key in self._noise_cache:
            return self._noise_cache[key]
        sigma, off = self.NOISE_SPEC[name]
        out = base.fft_noise((self.size, self.size), sigma, self.seed + off)
        self._noise_cache[key] = out
        return out

    def _grid(self) -> Tuple[np.ndarray, np.ndarray]:
        h = w = self.size
        Y0, X0 = np.mgrid[0:h, 0:w].astype(np.float32)
        return (X0 + 0.5) / w, (Y0 + 0.5) / h

    # ------------------------------------------------------------------
    # regular tile patterns
    # ------------------------------------------------------------------
    def _hex_ids(self, X: np.ndarray, Y: np.ndarray,
                 cols: int, rows: int) -> np.ndarray:
        """Nearest-centre ids on a staggered (triangular) point lattice.

        The Voronoi cells of this lattice are hexagons.  Only a 3x3 block of
        candidate centres is checked, so this stays cheap at 2048.
        """
        row0 = np.floor(Y * rows).astype(np.int64)
        best = np.full(X.shape, 1.0e9, np.float32)
        ids = np.zeros(X.shape, np.int64)
        # The row count is rounded to an even number so the pattern is
        # periodic across Y.  Compensate the resulting non-ideal spacing in
        # the distance metric, otherwise hexagons look stretched.
        y_scale = float(rows) * math.sqrt(3.0) / (2.0 * max(cols, 1))
        for dr in (-1, 0, 1):
            rr = (row0 + dr) % rows
            shift = (rr & 1).astype(np.float32) * 0.5
            col0 = np.floor(X * cols - shift).astype(np.int64)
            for dc in (-1, 0, 1):
                cc = (col0 + dc) % cols
                cx = (cc.astype(np.float32) + 0.5 + shift) / cols
                cy = (rr.astype(np.float32) + 0.5) / rows
                dx = (X - cx + 0.5) % 1.0 - 0.5
                dy = ((Y - cy + 0.5) % 1.0 - 0.5) * y_scale
                d = dx * dx + dy * dy
                m = d < best
                best = np.where(m, d, best)
                ids = np.where(m, rr * cols + cc, ids)
        return ids

    def _build_pattern(self) -> Dict[str, Any]:
        X, Y = self._grid()
        template = self.template
        tiles = int(self.params["tiles"])
        accent_px = None

        if template in ("grid", "checker", "concentric"):
            n = tiles
            fx = X * n
            fy = Y * n
            ix = np.floor(fx).astype(np.int64) % n
            iy = np.floor(fy).astype(np.int64) % n
            lx = fx - np.floor(fx)
            ly = fy - np.floor(fy)
            ids = iy * n + ix
            if template == "checker":
                accent_px = ((ix + iy) & 1).astype(np.float32)
            elif template == "concentric":
                inset = float(self.p("layout", "inset", self.params["inset"]))
                half_inner = max(0.05, 0.5 - 0.5 * inset)
                inside = ((np.abs(lx - 0.5) < half_inner) &
                          (np.abs(ly - 0.5) < half_inner))
                ids = ids * 2 + inside.astype(np.int64)
                accent_px = inside.astype(np.float32)

        elif template == "running_bond":
            nx = ny = tiles
            fy = Y * ny
            iy = np.floor(fy).astype(np.int64) % ny
            off = ((iy & 1).astype(np.float32)) * 0.5
            fx = X * nx + off
            ix = np.floor(fx).astype(np.int64) % nx
            ids = iy * nx + ix

        elif template == "basketweave":
            m = max(2, tiles)
            fx = X * m
            fy = Y * m
            mx = np.floor(fx).astype(np.int64) % m
            my = np.floor(fy).astype(np.int64) % m
            lx = fx - np.floor(fx)
            ly = fy - np.floor(fy)
            horizontal = ((mx + my) & 1) == 0
            brick = np.where(horizontal, (ly >= 0.5), (lx >= 0.5)).astype(np.int64)
            ids = (my * m + mx) * 2 + brick
            accent_px = ((mx + my) & 1).astype(np.float32)

        elif template == "hex":
            cols = max(2, tiles)
            rows = max(2, 2 * int(round(cols / (2.0 * 0.8660254))))
            ids = self._hex_ids(X, Y, cols, rows)

        elif template == "octagon":
            n = tiles
            fx = X * n
            fy = Y * n
            ix = np.floor(fx).astype(np.int64) % n
            iy = np.floor(fy).astype(np.int64) % n
            u = fx - np.floor(fx) - 0.5
            v = fy - np.floor(fy) - 0.5
            oct_inside = ((np.abs(u) <= 0.5) & (np.abs(v) <= 0.5) &
                          ((np.abs(u) + np.abs(v)) <= 0.70710678))
            cell = iy * n + ix
            gx = np.rint(X * n).astype(np.int64) % n
            gy = np.rint(Y * n).astype(np.int64) % n
            insert = n * n + gy * n + gx
            ids = np.where(oct_inside, cell, insert)
            accent_px = (~oct_inside).astype(np.float32)

        else:  # defensive fallback
            n = tiles
            ix = np.floor(X * n).astype(np.int64) % n
            iy = np.floor(Y * n).astype(np.int64) % n
            ids = iy * n + ix

        # Compact labels into 0..n_tiles-1.
        uniq, inv = np.unique(ids.ravel(), return_inverse=True)
        ids = inv.reshape(self.size, self.size).astype(np.int32)
        n_tiles = int(uniq.size)

        cnt = np.bincount(ids.ravel(), minlength=n_tiles).astype(np.float64)
        cnt = np.maximum(cnt, 1.0)
        if accent_px is None:
            accent = np.zeros(n_tiles, np.float32)
        else:
            acc = np.bincount(ids.ravel(), weights=accent_px.ravel(),
                              minlength=n_tiles).astype(np.float64)
            accent = (acc / cnt > 0.5).astype(np.float32)

        sum_x = np.bincount(ids.ravel(), weights=X.ravel(), minlength=n_tiles)
        sum_y = np.bincount(ids.ravel(), weights=Y.ravel(), minlength=n_tiles)
        centers = np.stack([(sum_x / cnt) % 1.0,
                            (sum_y / cnt) % 1.0], axis=1).astype(np.float32)
        area = (cnt / float(self.size * self.size)).astype(np.float32)

        edge = ((ids != np.roll(ids, 1, 0)) | (ids != np.roll(ids, -1, 0)) |
                (ids != np.roll(ids, 1, 1)) | (ids != np.roll(ids, -1, 1)))
        pad = max(16, int(0.025 * self.size))
        dist = base.periodic_distance(edge, pad).astype(np.float32)

        inr = np.zeros(n_tiles, np.float32)
        np.maximum.at(inr, ids.ravel(), dist.ravel())
        inr = np.maximum(inr, 1.0)

        return {
            "ids": ids,
            "n_tiles": n_tiles,
            "accent": accent,
            "centers": centers,
            "area": area,
            "inr": inr,
            "dist": dist,
        }

    # ------------------------------------------------------------------
    # stage runner
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

    def _palette(self):
        pal = np.asarray(self.cfg["pal"], np.float32)
        w = np.asarray(self.cfg["pal_w"], np.float32)
        w = w / max(float(w.sum()), 1e-9)
        mean = (pal * w[:, None]).sum(0).astype(np.float32)
        return pal, w, mean

    def _tile_colors(self, n_tiles: int, accent: np.ndarray):
        pal, w, mean = self._palette()
        rng = np.random.default_rng(self.seed + 601)
        idx = rng.choice(len(pal), n_tiles, p=w)
        tile = pal[idx].copy()

        variation = float(np.clip(
            float(self.p("color", "tile_variation",
                        self.params["tile_variation"])), 0.0, 2.0))
        jitter = rng.uniform(-1.0, 1.0, (n_tiles, 1)).astype(np.float32)
        tile *= 1.0 + 0.055 * variation * jitter
        tile = mean[None, :] + (tile - mean[None, :]) * min(1.6, variation)

        accent_amt = float(np.clip(
            float(self.p("color", "accent_contrast",
                        self.params["accent_contrast"])), 0.0, 1.0))
        if accent_amt > 0 and np.any(accent > 0.5):
            lum = pal @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            accent_base = pal[int(np.argmin(lum))] * 0.78
            m = accent > 0.5
            tile[m] = (tile[m] * (1.0 - 0.55 * accent_amt) +
                       accent_base[None, :] * (0.55 * accent_amt))

        harmony = float(np.clip(
            float(self.p("color", "harmony", self.params["harmony"])),
            0.0, 1.0))
        tile = tile * (1.0 - 0.20 * harmony) + mean[None, :] * (0.20 * harmony)

        lum = np.maximum(tile @ np.array([0.2126, 0.7152, 0.0722],
                                         np.float32), 1e-6)
        floor = {"aged": 0.50, "cyclopean": 0.56}.get(self.variant, 0.58)
        lift = np.maximum(1.0, floor / lum)
        tile *= lift[:, None]

        lum = np.maximum(tile @ np.array([0.2126, 0.7152, 0.0722],
                                         np.float32), 1e-6)
        ceil = 0.72
        knee = ceil - 0.08
        over = np.maximum(lum - knee, 0.0)
        target = knee + 0.08 * (1.0 - np.exp(-over / 0.08))
        tile *= (target / lum)[:, None]
        return np.clip(tile, 0.03, 1.0).astype(np.float32)

    # -- 1. layout ------------------------------------------------------
    def _stage_layout(self, _state: Dict[str, np.ndarray]):
        h = w = self.size
        pattern = self._build_pattern()
        ids = pattern["ids"]
        dist = pattern["dist"]
        inr = pattern["inr"]
        n_tiles = int(pattern["n_tiles"])
        accent = pattern["accent"]
        centers = pattern["centers"]
        area = pattern["area"]

        grout_px = float(np.clip(
            float(self.p("layout", "grout_px", self.params["grout_px"])),
            0.0, 60.0))
        bevel = float(np.clip(
            float(self.p("layout", "bevel", self.params["bevel"])), 0.0, 2.0))
        # Like the wall joint width, the slider is expressed in pixels at
        # 2048, so the 512 preview and the 2048 final have the same relative
        # joint width.
        grout_eff = grout_px * (self.size / 2048.0)
        half = 0.5 * grout_eff
        soft = max(0.8, 0.0018 * self.size)
        grout_mask = 1.0 - base.smoothstep(half - soft, half + soft, dist)
        tile_mask = 1.0 - grout_mask
        bevel_px = max(1.0, bevel * 0.0125 * self.size)
        bevel_ramp = base.smoothstep(half, half + bevel_px, dist)
        tile_profile = np.clip(dist / np.maximum(inr[ids], 1.0),
                               0.0, 1.0).astype(np.float32)

        # Schematic preview: distinct tile colours, dark grout and a simple
        # bevel light.  Final colour shading is stage 2.
        pal, pal_w, _mean = self._palette()
        rng = np.random.default_rng(self.seed + 400)
        cell_col = pal[rng.choice(len(pal), n_tiles, p=pal_w)]
        cell_col *= rng.uniform(0.92, 1.06, (n_tiles, 1)).astype(np.float32)
        cell_col[accent > 0.5] *= 0.80
        grout_col = np.clip(self.cfg["mortar"] * 1.9 + 0.025, 0.0, 1.0)
        preview = (cell_col[ids] * (0.52 + 0.48 * bevel_ramp[..., None]) *
                   tile_mask[..., None] +
                   grout_col[None, None, :] * grout_mask[..., None])
        preview = np.clip(preview, 0.0, 1.0).astype(np.float32)

        delta = dict(
            ids=ids.astype(np.int32),
            n_tiles=np.int32(n_tiles),
            accent=accent.astype(np.float32),
            centers=centers.astype(np.float32),
            area=area.astype(np.float32),
            inr=inr.astype(np.float32),
            dist=dist.astype(np.float32),
            tile_mask=tile_mask.astype(np.float32),
            grout_mask=grout_mask.astype(np.float32),
            bevel_ramp=bevel_ramp.astype(np.float32),
            tile_profile=tile_profile.astype(np.float32),
            grout_px=np.float32(grout_px),
            grout_px_eff=np.float32(grout_eff),
            bevel_px=np.float32(bevel_px),
            template=np.array([self.template]),
            surface=np.array([self.surface]),
        )
        msg = ("%s: %s, %d плиток, шов %.1f px@2048, фаска %.2f"
               % (SURFACE_LABELS.get(self.surface, self.surface),
                  TEMPLATES[self.template]["label"], n_tiles, grout_px, bevel))
        return delta, preview, msg

    # -- 2. tile colour -------------------------------------------------
    def _stage_color(self, state: Dict[str, np.ndarray]):
        h = w = self.size
        X, Y = self._grid()
        ids = state["ids"].astype(np.int32)
        n_tiles = int(state["n_tiles"])
        accent = state["accent"].astype(np.float32)
        centers = state["centers"].astype(np.float32)
        inr = state["inr"].astype(np.float32)
        tile_profile = state["tile_profile"].astype(np.float32)
        tile_mask = state["tile_mask"].astype(np.float32)
        grout_mask = state["grout_mask"].astype(np.float32)
        bevel_ramp = state["bevel_ramp"].astype(np.float32)

        tile_col = self._tile_colors(n_tiles, accent)
        rng = np.random.default_rng(self.seed + 602)
        rot = rng.uniform(0.0, math.pi, n_tiles).astype(np.float32)
        asp = rng.uniform(0.88, 1.12, n_tiles).astype(np.float32)
        freq = rng.uniform(5.0, 10.0, n_tiles).astype(np.float32)
        phase = rng.uniform(0.0, 2.0 * math.pi, n_tiles).astype(np.float32)
        hgt = rng.normal(0.0, 1.0, n_tiles).astype(np.float32)

        dx = base.wrap(X - centers[ids, 0])
        dy = base.wrap(Y - centers[ids, 1])
        ca = np.cos(rot[ids])
        sa = np.sin(rot[ids])
        u = dx * ca + dy * sa
        v = -dx * sa + dy * ca
        r_norm = np.maximum(inr[ids] / float(self.size), 1.0 / self.size)
        a_cap = r_norm * 0.62 * asp[ids]
        b_cap = r_norm * 0.62 / asp[ids]
        rr = np.sqrt((u / np.maximum(a_cap, 1e-4)) ** 2 +
                     (v / np.maximum(b_cap, 1e-4)) ** 2)
        cap = (1.0 - np.clip(rr, 0.0, 1.0) ** 2) ** 0.55
        prof_shape = tile_profile * (0.90 + 0.10 * cap)

        n_low = self.nfield("n_low")
        n_med = self.nfield("n_med")
        n_fine = self.nfield("n_fine")
        n_tint = self.nfield("n_tint")
        n_pit = self.nfield("n_pit")
        n_stri = self.nfield("n_stri")

        col = tile_col[ids]
        col *= (0.985 + 0.040 * n_low + 0.030 * n_med +
                0.020 * n_fine + 0.028 * n_tint)[..., None]
        stri = np.sin(2.0 * math.pi * (u * freq[ids] + phase[ids]))
        stri *= 0.5 + 0.5 * n_stri
        col *= (1.0 + 0.020 * stri)[..., None]

        dome = float(np.clip(float(self.params["tile_dome"]), 0.0, 2.0))
        dome *= float(self.scfg["dome"])
        color_shade = float(np.clip(
            float(self.p("color", "color_shade", 1.0)), 0.0, 2.0))
        shade_amt = 0.10 * color_shade * dome
        col *= (1.0 - shade_amt * (1.0 - prof_shape))[..., None]

        # Grout: same mortar/dirt family as the walls.
        mortar = np.asarray(self.cfg["mortar"], np.float32)
        dirt = np.asarray(self.cfg["dirt"], np.float32)
        grout_base = np.clip(mortar * 1.95 + 0.030, 0.0, 1.0)
        dirt_n = base.smoothstep(0.40, 1.55, n_pit)
        grout_px = (grout_base[None, None, :] * (1.0 - 0.35 * dirt_n)[..., None] +
                    dirt[None, None, :] * (0.35 * dirt_n)[..., None])
        grout_px = np.clip(grout_px, 0.0, 1.0)

        col = (col * tile_mask[..., None] +
               grout_px * grout_mask[..., None])
        cs = max(1.0, 0.0028 * self.size)
        col = np.stack([base.gauss_blur(col[..., c], cs)
                        for c in range(3)], axis=-1).astype(np.float32)

        # Soft edge shading on the tile side, driven by the real distance to
        # the joint.  Keeps light tiles from looking flat.
        edge = np.clip(1.0 - tile_profile, 0.0, 1.0) * tile_mask
        col *= (1.0 - (0.055 * float(self.scfg["edge_shade"]) *
                       np.clip(bevel_ramp, 0.0, 1.0) * edge))[..., None]
        col = np.clip(col, 0.0, 1.0).astype(np.float32)

        delta = dict(
            col=col,
            tile_col=tile_col.astype(np.float32),
            grout_col=grout_base.astype(np.float32),
            cell_hj=hgt.astype(np.float32),
        )
        return delta, col, "плитки окрашены (%d плиток)" % n_tiles

    # -- 3. grout -------------------------------------------------------
    def _stage_grout(self, state: Dict[str, np.ndarray]):
        h = w = self.size
        base_col = np.clip(state["col"].astype(np.float32), 0.0, 1.0)
        grout_mask = state["grout_mask"].astype(np.float32)
        tile_mask = state["tile_mask"].astype(np.float32)
        dist = state["dist"].astype(np.float32)
        n_pit = self.nfield("n_pit")
        n_dark = self.nfield("n_dark")

        contrast = float(np.clip(
            float(self.p("grout", "grout_contrast", 0.35)), 0.0, 2.0))
        darkness = float(np.clip(
            float(self.p("grout", "grout_darkness", 0.5)), 0.0, 1.0))
        dirt_amt = float(np.clip(
            float(self.p("grout", "grout_dirt", 1.0)), 0.0, 2.0))

        if contrast <= 1e-6:
            col = base_col
            msg = "контраст 0 — точная копия этапа 2"
        else:
            fade_px = max(2.0, 0.006 * self.size)
            fade = (1.0 - base.smoothstep(0.0, fade_px, dist)) * tile_mask
            alpha = np.clip(contrast * np.maximum(grout_mask, 0.55 * fade),
                            0.0, 1.0)
            dark = np.clip(np.asarray(self.cfg["mortar"], np.float32) * 0.62,
                           0.0, 1.0)
            col = (base_col * (1.0 - alpha[..., None]) +
                   dark[None, None, :] * alpha[..., None])

            speck = (base.smoothstep(1.55, 2.35, n_pit) *
                     base.smoothstep(0.0, 1.0, n_dark))
            speck *= grout_mask * min(1.0, 0.65 * dirt_amt)
            dirt = np.asarray(self.cfg["dirt"], np.float32)
            col = (col * (1.0 - 0.55 * speck[..., None]) +
                   dirt[None, None, :] * (0.55 * speck)[..., None])

            if abs(darkness - 0.5) > 1e-6:
                correction = col - base_col
                if darkness < 0.5:
                    factor = 1.0 + ((0.5 - darkness) / 0.5) * 0.8
                else:
                    factor = (1.0 - darkness) / 0.5
                col = base_col + correction * factor
            col = np.clip(col, 0.0, 1.0).astype(np.float32)
            msg = ("шов: контраст %.2f, темнота %.2f, грязь %.2f"
                   % (contrast, darkness, dirt_amt))

        return dict(col=col.astype(np.float32)), col, msg

    # -- 4. wear --------------------------------------------------------
    def _stage_wear(self, state: Dict[str, np.ndarray]):
        h = w = self.size
        col = np.clip(state["col"].astype(np.float32), 0.0, 1.0)
        ids = state["ids"].astype(np.int32)
        dist = state["dist"].astype(np.float32)
        tile_mask = state["tile_mask"].astype(np.float32)
        grout_mask = state["grout_mask"].astype(np.float32)
        tile_profile = state["tile_profile"].astype(np.float32)
        half = float(state["grout_px_eff"]) * 0.5

        n_pit = self.nfield("n_pit")
        n_stri = self.nfield("n_stri")
        n_low = self.nfield("n_low")
        n_stain = self.nfield("n_stain")
        n_dark = self.nfield("n_dark")

        wear_amt = float(np.clip(
            float(self.p("wear", "wear", self.params["wear"])), 0.0, 1.0))
        polish_amt = float(np.clip(
            float(self.p("wear", "polish", self.params["polish"])), 0.0, 1.0))
        crack_count = int(float(self.p(
            "wear", "crack_count", self.params["crack_count"])))
        wear_amt *= float(self.scfg["wear_scale"])
        polish_amt *= float(self.scfg["polish_scale"])

        edge_band = 1.0 - base.smoothstep(
            half, half + max(3.0, 0.050 * self.size), dist)
        patch = base.smoothstep(-0.55, 0.95, n_low)
        pits = base.smoothstep(1.25, 2.15, n_pit)
        scuffs = base.smoothstep(1.55, 2.40, n_stri)

        wear_mask = (wear_amt * (0.40 * edge_band * pits +
                                 0.35 * patch * pits +
                                 0.25 * scuffs * edge_band))
        wear_mask = np.clip(wear_mask, 0.0, 1.0).astype(np.float32)
        polish_mask = np.clip(
            polish_amt * base.smoothstep(0.05, 1.05, n_low) *
            (0.35 + 0.65 * tile_profile), 0.0, 1.0).astype(np.float32)

        dust = np.asarray(self.scfg["dust"], np.float32)
        col = (col * (1.0 - 0.30 * wear_mask[..., None]) +
               dust[None, None, :] * (0.30 * wear_mask)[..., None])
        col *= (1.0 + 0.045 * polish_mask[..., None])

        # Stains / efflorescence.  Ceiling defaults are stronger; the
        # surface config makes the same slider behave differently.
        stain_amt = float(np.clip(
            float(self.p("wear", "stains", 0.6)), 0.0, 2.0))
        stain_amt *= float(self.scfg["stain_scale"])
        stain_mask = np.clip(
            stain_amt * base.smoothstep(0.65, 1.85, n_stain) *
            (0.35 + 0.65 * base.smoothstep(-0.4, 0.9, n_low)),
            0.0, 1.0).astype(np.float32)
        stain_mask *= tile_mask
        stain_col = np.clip(np.asarray(self.cfg["dirt"], np.float32) * 0.85,
                            0.0, 1.0)
        col = (col * (1.0 - 0.22 * stain_mask[..., None]) +
               stain_col[None, None, :] * (0.22 * stain_mask)[..., None])

        if crack_count > 0:
            crack = base.make_cracks(self.size, self.seed, crack_count)
            crack *= tile_mask * (0.45 + 0.55 * wear_amt)
            col *= (1.0 - 0.09 * crack[..., None])
        else:
            crack = np.zeros((h, w), np.float32)

        wear_depth = np.clip(wear_mask * 0.75 + scuffs * 0.25, 0.0, 1.0)
        wear_depth = (wear_depth * tile_mask).astype(np.float32)
        col = np.clip(col, 0.0, 1.0).astype(np.float32)
        msg = ("износ %.2f, полировка %.2f, трещины %d"
               % (wear_amt, polish_amt, crack_count))
        return (dict(col=col, wear_mask=wear_mask, wear_depth=wear_depth,
                     polish_mask=polish_mask, stain=stain_mask,
                     crack=crack.astype(np.float32)),
                col, msg)

    # -- 5. details -----------------------------------------------------
    def _stage_details(self, state: Dict[str, np.ndarray]):
        h = w = self.size
        col = np.clip(state["col"].astype(np.float32), 0.0, 1.0)
        tile_mask = state["tile_mask"].astype(np.float32)
        tile_profile = state["tile_profile"].astype(np.float32)
        n_fine2 = self.nfield("n_fine2")
        n_spot = self.nfield("n_spot")
        n_dark = self.nfield("n_dark")
        n_vein = self.nfield("n_vein")
        n_vein2 = self.nfield("n_vein2")

        veins_amt = float(np.clip(
            float(self.p("details", "veins", self.params["veins"])), 0.0, 2.0))
        flecks_amt = float(np.clip(
            float(self.p("details", "flecks", self.params["flecks"])), 0.0, 2.0))
        stain_amt = float(np.clip(
            float(self.p("wear", "stains", 0.6)), 0.0, 2.0))

        stone_vis = tile_mask * (0.35 + 0.65 * tile_profile)

        # Thin mineral veins: ridged low-frequency noise.
        vein = 1.0 - np.abs(n_vein)
        vein2 = 1.0 - np.abs(n_vein2)
        vein = np.maximum(base.smoothstep(0.930, 0.997, vein),
                          base.smoothstep(0.966, 0.999, vein2))
        vein *= stone_vis * min(1.0, veins_amt)
        bright = np.array([0.86, 0.78, 0.55], np.float32)
        dark = np.array([0.33, 0.25, 0.15], np.float32)
        col = (col * (1.0 - 0.13 * vein[..., None]) +
               bright[None, None, :] * (0.13 * vein)[..., None])
        col = (col * (1.0 - 0.05 * vein[..., None]) +
               dark[None, None, :] * (0.05 * vein)[..., None])

        # Mineral flecks, same family as the wall speckles.
        fleck = base.smoothstep(1.70, 2.30, n_fine2) * stone_vis
        salmon = np.array([0.94, 0.72, 0.50], np.float32)
        yellow = np.array([0.86, 0.72, 0.14], np.float32)
        pick = base.smoothstep(0.0, 1.0, 0.5 + 0.5 * n_dark)[..., None]
        fleck_col = salmon[None, None, :] * pick + yellow[None, None, :] * (1.0 - pick)
        f_amt = np.clip(0.24 * flecks_amt, 0.0, 1.0)
        col = (col * (1.0 - f_amt * fleck[..., None]) +
               fleck_col * (f_amt * fleck)[..., None])

        # Ceiling efflorescence / pale mineral bloom.
        bloom = (base.smoothstep(1.75, 2.45, n_spot) * stone_vis *
                 min(1.0, stain_amt) * (0.35 if self.surface == "floor" else 1.0))
        bloom_col = np.array([0.80, 0.78, 0.68], np.float32)
        b_amt = np.clip(0.16 * stain_amt, 0.0, 1.0)
        col = (col * (1.0 - b_amt * bloom[..., None]) +
               bloom_col[None, None, :] * (b_amt * bloom)[..., None])

        col = np.clip(col, 0.0, 1.0).astype(np.float32)
        return (dict(col=col, vein=vein.astype(np.float32),
                     fleck=fleck.astype(np.float32),
                     bloom=bloom.astype(np.float32)),
                col, "прожилки %.2f, вкрапления %.2f" % (veins_amt, flecks_amt))

    # -- 6. broken edges + vegetation in the joints ---------------------
    def _grass_blades_mask(self, grout_mask: np.ndarray, rim: np.ndarray,
                           patch: np.ndarray, amount: float) -> np.ndarray:
        """Draw small tapered blades emerging from the joints.

        A pure noise threshold reads as a green stain.  Drawing a few short
        triangular blades per tuft makes grass readable at 512 and 2048 while
        the 3x3 wrapped copies keep the texture seamless.
        """
        h = w = self.size
        zone = (grout_mask > 0.18) | (rim > 0.62)
        ys, xs = np.nonzero(zone)
        if len(xs) == 0 or amount <= 1e-4:
            return np.zeros((h, w), np.float32)

        rng = np.random.default_rng(self.seed + 703)
        n_tufts = int(round(min(
            900.0,
            amount * (30.0 + 10.0 * int(self.params["tiles"])) *
            (self.size / 512.0))))
        n_tufts = max(1, n_tufts)

        weights = 0.08 + patch[xs, ys].astype(np.float64)
        total = float(weights.sum())
        if total <= 1e-9:
            weights = np.ones(len(xs), np.float64)
            total = float(len(xs))
        pick = rng.choice(len(xs), size=n_tufts, p=weights / total)

        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        blade_w = max(1, int(round(0.0014 * self.size)))
        for idx in pick:
            bx = float(xs[idx]) + rng.normal(0.0, 0.0035 * self.size)
            by = float(ys[idx]) + rng.normal(0.0, 0.0035 * self.size)
            base_ang = rng.uniform(0.0, 2.0 * math.pi)
            n_blades = int(rng.integers(4, 9))
            for _ in range(n_blades):
                ang = base_ang + rng.normal(0.0, 0.85)
                length = rng.uniform(0.010, 0.030) * self.size
                length *= 0.70 + 0.60 * min(1.0, amount)
                tip_x = bx + math.cos(ang) * length
                tip_y = by + math.sin(ang) * length
                perp_x = -math.sin(ang) * blade_w
                perp_y = math.cos(ang) * blade_w
                pts = [(bx + perp_x, by + perp_y),
                       (bx - perp_x, by - perp_y),
                       (tip_x, tip_y)]
                for ox in (-w, 0, w):
                    for oy in (-h, 0, h):
                        draw.polygon([(x + ox, y + oy) for x, y in pts],
                                     fill=255)

        blades = np.asarray(img, np.float32) / 255.0
        blades = base.gauss_blur(blades, max(0.35, 0.0007 * self.size))
        return np.clip(blades, 0.0, 1.0).astype(np.float32)

    def _stage_edges(self, state: Dict[str, np.ndarray]):
        h = w = self.size
        col = np.clip(state["col"].astype(np.float32), 0.0, 1.0)
        ids = state["ids"].astype(np.int32)
        dist = state["dist"].astype(np.float32)
        tile_mask = state["tile_mask"].astype(np.float32)
        grout_mask = state["grout_mask"].astype(np.float32)
        cell_hj = state["cell_hj"].astype(np.float32)
        half = float(state["grout_px_eff"]) * 0.5

        n_low = self.nfield("n_low")
        n_pit = self.nfield("n_pit")
        n_fine = self.nfield("n_fine")
        n_fine2 = self.nfield("n_fine2")
        n_dark = self.nfield("n_dark")
        n_spot = self.nfield("n_spot")

        chips_amt = float(np.clip(
            float(self.p("edges", "edge_chips",
                         self.params["edge_chips"])), 0.0, 1.0))
        chip_size = float(np.clip(
            float(self.p("edges", "edge_chip_size",
                         self.params["edge_chip_size"])), 0.0, 2.0))
        grass_amt = float(np.clip(
            float(self.p("edges", "grass", self.params["grass"])), 0.0, 2.0))
        moss_amt = float(np.clip(
            float(self.p("edges", "moss", self.params["moss"])), 0.0, 2.0))
        growth_width = float(np.clip(
            float(self.p("edges", "growth_width",
                         self.params["growth_width"])), 0.0, 2.0))

        chips_amt *= float(self.scfg["chip_scale"])
        grass_amt *= float(self.scfg["grass_scale"])
        moss_amt *= float(self.scfg["moss_scale"])

        # Irregular broken rim: noise thresholds are applied only in a band
        # just inside the tile boundary, so the joints stay continuous.
        band = max(2.0, 0.018 * self.size * max(0.35, chip_size))
        inside_edge = tile_mask * (1.0 - base.smoothstep(
            half, half + band, dist))
        tile_gate = 0.45 + 0.55 * base.smoothstep(
            -0.8, 0.8, cell_hj[ids])
        chip_noise = base.smoothstep(
            0.15, 1.25, 0.45 * n_pit + 0.35 * n_low + 0.20 * n_fine2)
        chip = np.clip(chips_amt * inside_edge * tile_gate * chip_noise,
                       0.0, 1.0).astype(np.float32)
        core = np.clip(chip * (0.50 + 0.50 * base.smoothstep(
            0.20, 1.15, n_fine2)), 0.0, 1.0).astype(np.float32)
        chip_depth_pre = np.clip(chip * (0.45 + 0.55 * core),
                                 0.0, 1.0).astype(np.float32)

        raw = np.clip(np.asarray(self.cfg["pal"][0], np.float32) * 1.02 + 0.06,
                      0.0, 1.0)
        shadow = np.clip(np.asarray(self.cfg["mortar"], np.float32) * 0.45,
                         0.0, 1.0)
        lip = chip * (1.0 - core)
        col = (col * (1.0 - 0.62 * chip[..., None]) +
               raw[None, None, :] * (0.62 * chip)[..., None])
        col = (col * (1.0 - 0.52 * core[..., None]) +
               shadow[None, None, :] * (0.52 * core)[..., None])
        col *= (1.0 + 0.070 * lip[..., None])

        # Fake-bevel light makes the chips readable already in this stage's
        # albedo preview; the real recess is applied in height/normal/AO.
        depth_shade = base.gauss_blur(
            chip_depth_pre, max(0.8, 0.0015 * self.size))
        gx = (np.roll(depth_shade, -1, axis=1) -
              np.roll(depth_shade, 1, axis=1)) * 0.5
        gy = (np.roll(depth_shade, -1, axis=0) -
              np.roll(depth_shade, 1, axis=0)) * 0.5
        bevel = np.clip(-2.2 * gx - 2.2 * gy, -0.55, 0.55)
        col *= (1.0 + 0.55 * bevel[..., None])
        col *= (1.0 - 0.18 * chip_depth_pre[..., None])

        # Vegetation creeps out of the grout into the tile edge.  Broad
        # patches decide which joints are overgrown; a ridged high-frequency
        # field breaks the patch into blades and tufts.
        grow_span = max(2.0, 0.035 * self.size * growth_width)
        rim = 1.0 - base.smoothstep(half, half + grow_span, dist)
        zone_grass = np.clip(0.55 * grout_mask + 0.45 * rim, 0.0, 1.0)
        zone_moss = np.clip(0.80 * grout_mask + 0.20 * rim, 0.0, 1.0)

        patch = base.smoothstep(-0.30, 1.20, 0.55 * n_low + 0.45 * n_spot)
        tuft = base.smoothstep(-0.40, 1.10, 0.55 * n_pit + 0.45 * n_low)
        blades = (base.smoothstep(0.68, 0.997, 1.0 - np.abs(n_fine2)) +
                  0.50 * base.smoothstep(0.75, 0.998, 1.0 - np.abs(n_fine)))
        ground = zone_grass * patch * tuft * (
            0.45 + 0.55 * np.clip(blades, 0.0, 1.0))
        blade_mask = self._grass_blades_mask(grout_mask, rim, patch, grass_amt)
        grass_raw = 0.80 * blade_mask + 0.30 * ground
        grass = np.clip(grass_amt * grass_raw, 0.0, 1.0).astype(np.float32)

        moss_patch = base.smoothstep(-0.40, 1.10, 0.65 * n_low + 0.35 * n_spot)
        moss_raw = zone_moss * moss_patch * (
            0.60 + 0.40 * base.smoothstep(-0.20, 1.10, n_pit))
        moss = np.clip(moss_amt * moss_raw, 0.0, 1.0).astype(np.float32)

        grass_col = np.array([0.34, 0.45, 0.16], np.float32)
        grass_hi = np.clip(0.50 + 0.50 * n_fine, 0.0, 1.0)
        grass_px = grass_col[None, None, :] * (
            0.68 + 0.56 * grass_hi[..., None])
        col = (col * (1.0 - 0.92 * grass[..., None]) +
               grass_px * (0.92 * grass)[..., None])

        # Use the same moss family as the aged wall so the vegetation reads
        # as one world.
        moss_col = np.array([0.185, 0.245, 0.100], np.float32)
        moss_px = moss_col[None, None, :] * (
            0.72 + 0.38 * np.clip(0.5 + 0.5 * n_dark, 0.0, 1.0)[..., None])
        col = (col * (1.0 - 0.82 * moss[..., None]) +
               moss_px * (0.82 * moss)[..., None])

        # Dark contact shadow around tufts makes the blades read against the
        # tile instead of looking like a painted green fringe.
        grass_soft = base.gauss_blur(grass, max(1.0, 0.0018 * self.size))
        grass_shadow = np.clip(grass_soft - grass, 0.0, 1.0)
        col *= (1.0 - 0.22 * grass_shadow)[..., None]
        col *= (1.0 - 0.08 * np.clip(grass + moss, 0.0, 1.0))[..., None]
        col = np.clip(col, 0.0, 1.0).astype(np.float32)

        edge_chip_depth = chip_depth_pre
        growth_height = (0.065 * grass + 0.035 * moss).astype(np.float32)
        msg = ("сколы %.2f (крупность %.2f), трава %.2f, мох %.2f"
               % (chips_amt, chip_size, grass_amt, moss_amt))
        return (dict(col=col, edge_chip=chip,
                     edge_chip_depth=edge_chip_depth,
                     grass_mask=grass, moss_mask=moss,
                     growth_height=growth_height),
                col, msg)

    # ------------------------------------------------------------------
    # shared maps
    # ------------------------------------------------------------------
    def _compute_height(self, state: Dict[str, np.ndarray], stage: str):
        ids = state["ids"].astype(np.int32)
        dist = state["dist"].astype(np.float32)
        tile_profile = state["tile_profile"].astype(np.float32)
        bevel_ramp = state["bevel_ramp"].astype(np.float32)
        tile_mask = state["tile_mask"].astype(np.float32)
        wear_depth = state["wear_depth"].astype(np.float32)
        stain = state["stain"].astype(np.float32)
        vein = state["vein"].astype(np.float32)
        fleck = state["fleck"].astype(np.float32)
        cell_hj = state["cell_hj"].astype(np.float32)
        edge_chip_depth = state.get("edge_chip_depth")
        growth_height = state.get("growth_height")

        dome = float(np.clip(
            float(self.p(stage, "tile_dome", self.params["tile_dome"])),
            0.0, 2.0)) * float(self.scfg["dome"])
        relief = float(np.clip(
            float(self.p(stage, "height_relief", 1.0)), 0.1, 3.0))

        grout_h = 0.30
        tile_h = 0.62
        height = grout_h + (tile_h + 0.20 * dome * tile_profile - grout_h) * bevel_ramp
        height += 0.010 * cell_hj[ids] * tile_mask
        height -= 0.62 * wear_depth
        height -= 0.030 * stain * tile_mask
        height += 0.014 * vein
        height += 0.008 * fleck

        height = base.gauss_blur(height, max(1.2, 0.0018 * self.size))
        edge_w = base.smoothstep(0.0, 0.35, 1.0 - bevel_ramp)
        h_soft = base.gauss_blur(height, max(2.5, 0.0040 * self.size))
        height = height * (1.0 - 0.45 * edge_w) + h_soft * (0.45 * edge_w)

        # Chips and vegetation are applied after the broad edge softening:
        # otherwise the same blur that calms the joint also erases thin
        # blades and small broken notches from height/normal/AO.
        if edge_chip_depth is not None:
            height -= 0.62 * edge_chip_depth.astype(np.float32)
        if growth_height is not None:
            height += 1.45 * growth_height.astype(np.float32)
        height = base.gauss_blur(height, max(0.6, 0.0008 * self.size))

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
        grout_mask = state["grout_mask"].astype(np.float32)
        wear_depth = state["wear_depth"].astype(np.float32)
        ao = 1.0 - np.clip(cav * 4.2, 0.0, 0.55) - np.clip(cav2 * 1.7, 0.0, 0.35)
        ao *= (1.0 - 0.10 * grout_mask)
        ao *= (1.0 - 0.06 * wear_depth)
        grass = state.get("grass_mask")
        moss = state.get("moss_mask")
        if grass is not None:
            ao *= (1.0 - 0.05 * grass.astype(np.float32))
        if moss is not None:
            ao *= (1.0 - 0.03 * moss.astype(np.float32))
        ao = np.clip(ao, 0.20, 1.0).astype(np.float32)
        ao = base.gauss_blur(ao, 0.8)
        strength = float(np.clip(
            float(self.p(stage, "ao_strength", 1.0)), 0.0, 2.0))
        ao = 1.0 - (1.0 - ao) * strength
        return np.clip(ao, 0.0, 1.0).astype(np.float32)

    def _normal(self, state: Dict[str, np.ndarray], stage: str,
                force: bool = False):
        if not force and "nx" in state:
            return (state["nx"].astype(np.float32),
                    state["ny"].astype(np.float32),
                    state["nz"].astype(np.float32))
        n_fine = self.nfield("n_fine")
        n_fine2 = self.nfield("n_fine2")
        n_micro = self.nfield("n_micro")
        n_med = self.nfield("n_med")
        height = state["height"].astype(np.float32)
        ids = state["ids"].astype(np.int32)
        grout_mask = state["grout_mask"].astype(np.float32)
        tile_mask = state["tile_mask"].astype(np.float32)

        micro = (0.0075 * n_fine + 0.0050 * n_fine2 +
                 0.0040 * n_micro + 0.0030 * n_med)
        detail_mask = (0.35 + 0.65 * tile_mask) * float(self.scfg["micro"])
        normal_detail = float(np.clip(
            float(self.p(stage, "normal_detail",
                         self.params["normal_detail"])), 0.0, 4.0))
        normal_strength = float(np.clip(
            float(self.p(stage, "normal_strength", 1.0)), 0.0, 2.0))
        hn = base.gauss_blur(height, 1.1) + normal_detail * micro * detail_mask
        gx = (np.roll(hn, -1, axis=1) - np.roll(hn, 1, axis=1)) * 0.5
        gy = (np.roll(hn, -1, axis=0) - np.roll(hn, 1, axis=0)) * 0.5
        s_tile, s_gap = 54.0, 12.0
        s = s_gap + (s_tile - s_gap) * tile_mask
        s *= (self.size / 2048.0) * normal_strength
        nx = -gx * s
        ny = -gy * s
        nz = np.ones_like(nx)
        ln = np.sqrt(nx * nx + ny * ny + nz * nz)
        return ((nx / ln).astype(np.float32), (ny / ln).astype(np.float32),
                (nz / ln).astype(np.float32))

    # -- 7. final albedo ------------------------------------------------
    def _stage_albedo(self, state: Dict[str, np.ndarray]):
        h = w = self.size
        height = self._compute_height(state, "albedo")
        ao = self._compute_ao(height, state, "albedo")
        baked_ao = float(np.clip(
            float(self.p("albedo", "baked_ao", 0.035)), 0.0, 1.0))
        col = state["col"].astype(np.float32) * (
            1.0 - baked_ao * (1.0 - ao[..., None]))
        col = np.clip(col, 0.0, 1.0).astype(np.float32)
        return (dict(col=col, height=height, ao=ao), col,
                {"albedo": col}, "итоговый albedo (%dx%d)" % (w, h))

    # -- 8. height ------------------------------------------------------
    def _stage_height(self, state: Dict[str, np.ndarray]):
        height = self._compute_height(state, "height")
        prev = np.repeat(height[..., None], 3, axis=2)
        return (dict(height=height), prev, {"height": height},
                "height готов (купол %.2f)" %
                float(self.p("height", "tile_dome",
                             self.params["tile_dome"])))

    # -- 9. roughness ---------------------------------------------------
    def _stage_roughness(self, state: Dict[str, np.ndarray]):
        n_med = self.nfield("n_med")
        n_fine = self.nfield("n_fine")
        ids = state["ids"].astype(np.int32)
        tile_mask = state["tile_mask"].astype(np.float32)
        grout_mask = state["grout_mask"].astype(np.float32)
        wear_depth = state["wear_depth"].astype(np.float32)
        polish_mask = state["polish_mask"].astype(np.float32)
        stain = state["stain"].astype(np.float32)
        vein = state["vein"].astype(np.float32)
        fleck = state["fleck"].astype(np.float32)
        cell_hj = state["cell_hj"].astype(np.float32)
        edge_chip = state.get("edge_chip")
        grass = state.get("grass_mask")
        moss = state.get("moss_mask")

        rough_tile = (float(self.cfg.get("big_rough", 0.55)) +
                      float(self.scfg["rough_add"]) +
                      0.035 * n_med + 0.020 * n_fine +
                      0.025 * cell_hj[ids])
        rough = rough_tile * tile_mask + 0.82 * grout_mask
        rough += 0.10 * wear_depth
        rough -= 0.14 * polish_mask
        rough += 0.035 * stain
        rough += 0.025 * vein
        rough -= 0.015 * fleck
        if edge_chip is not None:
            rough += 0.12 * edge_chip.astype(np.float32)
        if grass is not None:
            rough += 0.20 * grass.astype(np.float32)
        if moss is not None:
            rough += 0.10 * moss.astype(np.float32)
        rough = np.clip(rough, 0.0, 1.0)
        rough_bias = float(self.p("roughness", "roughness_bias", 0.0))
        rough_contrast = float(np.clip(
            float(self.p("roughness", "roughness_contrast", 1.0)), 0.1, 3.0))
        rough = 0.62 + (rough - 0.62) * rough_contrast + rough_bias
        rough = np.clip(rough, 0.0, 1.0).astype(np.float32)
        prev = np.repeat(rough[..., None], 3, axis=2)
        return (dict(rough=rough), prev, {"roughness": rough},
                "roughness готов (смещение %+.2f)" % rough_bias)

    # -- 10. AO ---------------------------------------------------------
    def _stage_ao(self, state: Dict[str, np.ndarray]):
        height = state["height"].astype(np.float32)
        ao = self._compute_ao(height, state, "ao")
        prev = np.repeat(ao[..., None], 3, axis=2)
        return (dict(ao=ao), prev, {"ao": ao},
                "AO готов (сила %.2f)" %
                float(self.p("ao", "ao_strength", 1.0)))

    # -- 11/12. normals -------------------------------------------------
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
