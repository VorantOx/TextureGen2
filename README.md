# OTWNDATA Texture Generator

Процедурный генератор бесшовных PBR-текстур каменной кладки в стиле
`OTWNDATA.VGA` из **Wizardry VII: Crusaders of the Dark Savant (1992)**.

Генератор создаёт три варианта кладки:

- `gold` — золотистый песчаник, ближе всего к оригинальному тайлу;
- `aged` — выветренный камень с мхом;
- `cyclopean` — более высокие купола и контрастная кладка.

Каждый вариант содержит `albedo`, `height`, `roughness`, `ao`,
`normal_gl` и `normal_dx`. Все карты бесшовные, 2048×2048.

Редактор дополнительно умеет генерировать **пол и потолок** с регулярным
(почти симметричным) рисунком по выбранному шаблону плитки. Палитра и
цвета затирки берутся из того же варианта, что и у стен, поэтому
поверхности сочетаются по цвету с кладкой `gold`, `aged` или `cyclopean`.

## Что в репозитории

```text
make_otwn_textures.py     CLI-генератор полного пака
otwn_pipeline.py          пошаговый кэшируемый пайплайн стен
otwn_surface_pipeline.py  пошаговый пайплайн пола и потолка
otwn_editor.py            локальный веб-редактор
otwn_editor.html          интерфейс редактора
START_EDITOR.bat          запуск редактора в Windows
requirements.txt          зависимости Python
README_EDITOR.md          руководство по редактору
README_PACK.md            описание готового пака текстур
```

## Требования

- Python 3.9 или новее;
- NumPy;
- Pillow.

Установка зависимостей:

```powershell
python -m pip install -r requirements.txt
```

## Быстрый старт: CLI-генератор

Полная генерация всех трёх вариантов в 2048 px:

```powershell
python make_otwn_textures.py
```

Быстрый тест только `gold` в 512 px:

```powershell
python make_otwn_textures.py --size 512 --variants gold
```

Результат появится в `textures_otwn/`:

```text
textures_otwn/
    gold/
    aged/
    cyclopean/
    previews/
    README.md
    stats.csv
```

Основные параметры CLI:

```text
--size N             размер текстуры (2048 по умолчанию)
--variants LIST      gold,aged,cyclopean
--joint PX           ширина шва в px при 2048
--dome K             сила купола камней
--chips N            число сколов
--cracks N           число трещин (0 = выключено)
--spread K           разброс размеров крупных камней
--medium N           число средних камней
--medium-area F      доля площади под средние камни
--seam-halo F        ширина мягкой зоны цвета у шва
--seam-halo-amt F    сила градиента у шва
--normal-detail K    детализация normal map
```

Референсный тайл `png_vga/tiles/OTWNDATA/t000.png` не обязателен для
генерации. Если он лежит в репозитории, в контактный лист и превью
добавляется оригинальный тайл для сравнения.

## Быстрый старт: веб-редактор

```powershell
python otwn_editor.py
```

Откроется `http://127.0.0.1:8765/`.

Редактор позволяет проходить генерацию по этапам, подтверждать каждый
этап, менять локальные параметры этапа и продолжать работу из кэша.
На этапе топологии стен можно наводить курсор на камни, выбирать несколько
камней, объединять их и задавать скругление выбранным камням.

В режиме **«Пол и потолок»** доступны семь регулярных шаблонов плитки
(`grid`, `checker`, `running_bond`, `basketweave`, `hex`, `octagon`,
`concentric`) и отдельный набор из 12 этапов, включая сбитые края плиток
и прорастание травы/мха через швы. Цвета плитки, затирки и
грязи берутся из палитры выбранного варианта стен, поэтому пол/потолок
сочетается с кладкой. Подробности — в [README_EDITOR.md](README_EDITOR.md).

Запуск в Windows без терминала:

```text
START_EDITOR.bat
```

## Кэши и результаты

Редактор сохраняет промежуточные массивы в `editor_data/jobs/<id>/cache`
в формате `.npz`. Это позволяет не пересчитывать топологию и цвет камней
при изменении отдельных этапов. Каталог `editor_data/` исключён из git.

Готовые карты редактор публикует в `textures_otwn/<variant>/` по кнопке
«Опубликовать в textures_otwn».

## Документация

- [README_EDITOR.md](README_EDITOR.md) — этапы, локальные параметры,
  кэш, подтверждения и публикация.
- [README_PACK.md](README_PACK.md) — описание пака, палитра, формат карт,
  пример подключения в Three.js.

## Visual Studio Code

Откройте файл `TextureGen.code-workspace` или папку проекта в VS Code.
Установите рекомендованное расширение Python (`ms-python.python` и
`ms-python.debugpy`).

В панели **Run and Debug** доступны конфигурации:

- **Python: OTWN Editor** — запускает редактор с отладчиком Python на
  `http://127.0.0.1:8765/` без автоматического открытия браузера;
- **Yandex: OTWN Editor** — запускает Yandex Browser с remote debugging;
- **OTWN: Editor + Yandex** — составная конфигурация: сервер и браузер
  вместе;
- **Python: Generator CLI (512 gold)** — отладка быстрой генерации
  CLI-пака в 512 px.

Для отладки используется отдельный профиль Yandex в
`.vscode/yandex-debug-profile` (он исключён из git).

Если Yandex Browser установлен в другом месте, поправьте
`runtimeExecutable` в `.vscode/launch.json`. В текущем окружении путь:

```text
C:\Program Files\Yandex\YandexBrowser\Application\browser.exe
```

Дополнительные задачи доступны через **Terminal → Run Task**:

- `OTWN: compile Python` — проверить синтаксис;
- `OTWN: stop editor on 8765` — освободить порт;
- `OTWN: open in Yandex` — просто открыть редактор в Yandex без отладчика.

## Примечание по стилю

Палитра и манеры взяты с тайла `OTWNDATA` t000. Это не апскейл исходных
8-битных изображений, а новая процедурная кладка в духе оригинальной
игры. Для генерации не требуется запускать саму игру.
