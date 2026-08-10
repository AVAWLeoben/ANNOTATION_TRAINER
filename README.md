# YOLO Annotation Trainer

A local desktop training application for teaching annotation workers to recognize and classify objects using existing **Ultralytics/YOLO detection annotations** as ground truth.

The trainee does not draw boxes. Existing YOLO bounding boxes are used as hidden hit targets: the worker selects a class, clicks an object, and receives immediate visual and audible feedback. Worker performance is stored locally in SQLite and can be used to prioritize classes they struggle with.

> Documented build: `4.3-shortcuts-2026-08-10`

## Features

- PySide6/Qt desktop UI.
- Automatically scans `TRAINING_DATA/` at startup.
- Every immediate subfolder becomes a selectable training category.
- No native Windows directory selector in normal use.
- Standard YOLO detection-box parsing.
- Separate `images/` + `labels/` or mixed image/label layouts.
- Automatic class-name loading from text files or dataset YAML.
- In-app class-name editor.
- Randomized sessions and optional random image subsets.
- Worker profiles stored in SQLite.
- Per-worker and per-class accuracy statistics.
- Optional weak-class prioritization based on previous mistakes.
- Score, streaks, visual rewards, confetti, and generated sound feedback.
- Wrong answers lock the object until it is corrected.
- Ground-truth answer overlay with box + class names.
- Skip an image and revisit it at the end of the same session.
- Global keyboard shortcuts for fast training.

## How it works

For every image, the YOLO boxes are loaded but hidden by default.

1. Select the class you believe an object belongs to.
2. Click the object in the image.
3. The trainer finds the unsolved ground-truth box under the click.
4. The selected class is compared with the box's YOLO class ID.
5. Correct answers are rewarded immediately.
6. Incorrect answers highlight that object in red and require another attempt on the same object.
7. Once all objects are solved, the trainer advances automatically.

If multiple boxes overlap at the click position, the smallest matching unsolved box is used.

This application trains **class/object recognition**, not bounding-box drawing accuracy.

## Requirements

- Python
- PySide6
- PyYAML

SQLite uses Python's built-in `sqlite3` module; no database server or extra SQLite package is required.

Recommended conda environment on Windows:

```powershell
conda create -n ANNOTATION_TRAINER python=3.12
conda activate ANNOTATION_TRAINER
pip install --upgrade pip
pip install PySide6 PyYAML
```

Install dependencies:

```bash
python -m pip install PySide6 PyYAML
```

Recommended virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install PySide6 PyYAML
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install PySide6 PyYAML
```

## Installation

Place the trainer script in the repository root. The application always looks for `TRAINING_DATA` beside the Python file.

```text
YOLO-Annotation-Trainer/
├── yolo_annotation_trainer.py
└── TRAINING_DATA/
```

Run:

```bash
python yolo_annotation_trainer.py
```

If you are using the current versioned development file directly:

```bash
python yolo_annotation_trainer_v4_3_shortcuts.py
```

## Training data structure

Every immediate subdirectory of `TRAINING_DATA/` becomes a category card in the main menu.

```text
YOLO-Annotation-Trainer/
├── yolo_annotation_trainer.py
└── TRAINING_DATA/
    ├── FOOD_PACKAGING/
    │   ├── classes.txt
    │   ├── images/
    │   │   ├── image_001.jpg
    │   │   └── image_002.jpg
    │   └── labels/
    │       ├── image_001.txt
    │       └── image_002.txt
    │
    ├── SHREDDER_SCRAP/
    │   ├── classes.txt
    │   ├── images/
    │   └── labels/
    │
    └── METAL_SORTING/
        ├── data.yaml
        ├── images/
        └── labels/
```

The launcher will create cards for `FOOD_PACKAGING`, `SHREDDER_SCRAP`, and `METAL_SORTING` automatically.

If categories are added while the program is running, click **Rescan TRAINING_DATA**.

## Supported dataset layouts

Standard split layout:

```text
CATEGORY/
├── classes.txt
├── images/
│   ├── image_001.jpg
│   └── image_002.jpg
└── labels/
    ├── image_001.txt
    └── image_002.txt
```

Nested YOLO layouts also work, for example:

```text
CATEGORY/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

or:

```text
CATEGORY/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

Mixed folders are supported too:

```text
CATEGORY/
├── classes.txt
├── image_001.jpg
├── image_001.txt
├── image_002.jpg
└── image_002.txt
```

Supported image extensions:

```text
.jpg  .jpeg  .png  .bmp  .webp
```

Images without matching labels are skipped. Empty label files are not included as training samples.

## YOLO annotation format

The current trainer supports standard YOLO **object detection** rows with exactly five values:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Example:

```text
0 0.513000 0.421000 0.183000 0.296000
2 0.728000 0.655000 0.221000 0.192000
3 0.184000 0.544000 0.103000 0.164000
```

Coordinates are expected to be normalized to image width and height.

## `classes.txt`

Place `classes.txt` directly in the category root:

```text
TRAINING_DATA/
└── FOOD_PACKAGING/
    ├── classes.txt
    ├── images/
    └── labels/
```

Use one class name per line:

```text
person
forklift
pallet
cardboard_box
plastic_wrap
damaged_carton
```

The line order defines the YOLO IDs:

```text
0 = person
1 = forklift
2 = pallet
3 = cardboard_box
4 = plastic_wrap
5 = damaged_carton
```

Do not include numeric IDs inside `classes.txt`.

The trainer searches for class definitions in this order:

1. `classes.txt`
2. `class_names.txt`
3. `names.txt`
4. `obj.names`
5. `data.yaml`
6. `dataset.yaml`
7. `data.yml`
8. `dataset.yml`

For YAML files, the `names` field may be a list or ID-to-name mapping.

If an annotation references a class ID without a supplied name, the trainer creates a placeholder such as `class_7`.

Class names can also be edited inside the application. Saving from the editor writes `classes.txt` to the selected category root.

## Main menu and workers

At startup the application scans `TRAINING_DATA/` and displays category cards containing the category name, image count, class count, and class-name source.

Before starting, select:

1. a category
2. a worker profile
3. images per session
4. whether weak-class prioritization is enabled

Create additional workers with the `+` button. An empty database automatically receives a `Default Trainee` profile.

## Randomized sessions

Every session generates a new image order.

- **All images**: every labeled image is included in randomized order.
- **Limited session size**: a random subset/order is created.
- **Restart / reshuffle**: starts a fresh session with another ordering.

## Weak-class prioritization

When enabled, previous per-class performance changes future sampling weights.

Classes with more mistakes receive a higher weight, causing images containing those classes to appear earlier or more often in a limited-size session. Sampling remains randomized rather than becoming a fixed replay list.

Example:

```text
Worker: Anna
Task: FOOD_PACKAGING

cardboard_box      96%
plastic_wrap       88%
damaged_carton     62%
```

Anna's future sessions can therefore emphasize images containing `damaged_carton`.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `1` | First class / YOLO ID `0` |
| `2` | Second class / YOLO ID `1` |
| `3` | Third class / YOLO ID `2` |
| `4` | Fourth class / YOLO ID `3` |
| `5` | Fifth class / YOLO ID `4` |
| `6` | Sixth class / YOLO ID `5` |
| `7` | Seventh class / YOLO ID `6` |
| `8` | Eighth class / YOLO ID `7` |
| `9` | Ninth class / YOLO ID `8` |
| `0` | Tenth class / YOLO ID `9` |
| `H` | Show/hide correct boxes |
| `S` | Skip image and revisit later |

The shortcuts are bound to the entire training page, so they continue to work when buttons or combo boxes have focus.

Classes after the tenth currently need to be selected from the dropdown.

## Correct answers

Correct classifications trigger:

- green box feedback
- success sound
- confetti animation
- score increase
- streak increase
- streak bonus

The base reward is `+10` points. Consecutive correct answers increase the bonus up to the current built-in cap.

Solved boxes stay solved for the current session.

## Incorrect answers

Incorrect classifications:

- play an error sound
- reset the streak
- increment mistakes
- highlight the target object in red
- lock the user to that object

The worker must select another class and click the same red object again before continuing to other objects.

Each classification attempt is recorded in SQLite, including wrong attempts and the later correct retry.

## Show correct bounding boxes

Click **Show correct boxes** or press `H` to reveal every ground-truth box and its true class.

Labels are displayed as:

```text
3: cardboard_box
```

This is intended as a learning/review aid. New sessions always begin with the answers hidden.

## Skipping images

Click **Skip image — revisit later** or press `S`.

The current image is moved to the end of the remaining session queue.

Skipping:

- does not count as a mistake
- does not affect accuracy
- does not reset the streak
- does not create an SQLite classification attempt
- preserves objects already solved on that image

If it is the only remaining image, it stays current because there is nowhere else to defer it.

## SQLite statistics

The preferred statistics database is:

```text
TRAINING_DATA/.yolo_trainer_stats.sqlite3
```

If `TRAINING_DATA/` is not writable, the trainer falls back to a database below the current user's home directory. The actual active path is displayed on the main menu.

The schema contains:

### `trainees`

```text
id
name
created_at
```

### `sessions`

```text
id
trainee_id
task_name
started_at
ended_at
score
correct_answers
mistakes
completed
```

### `attempts`

```text
session_id
trainee_id
task_name
image_path
target_class_id
target_class_name
selected_class_id
selected_class_name
correct
attempted_at
```

Per-class statistics include attempts, correct answers, mistakes, and accuracy.

Example:

| Class | Attempts | Correct | Mistakes | Accuracy |
|---|---:|---:|---:|---:|
| cardboard_box | 87 | 84 | 3 | 96.6% |
| plastic_wrap | 54 | 47 | 7 | 87.0% |
| damaged_carton | 63 | 39 | 24 | 61.9% |
| food_tray | 71 | 69 | 2 | 97.2% |

## Data integrity

Normal training does not rewrite source images or YOLO annotation files.

The application writes only:

- the local SQLite statistics database
- `classes.txt` when class names are explicitly edited and saved

## Scope

The project is intended for:

- onboarding new annotation workers
- teaching a class taxonomy
- practicing distinctions between visually similar classes
- targeted retraining of weak classes
- measuring recognition accuracy per worker and class
- generating short randomized practice sessions from an existing dataset
- reviewing difficult images using known ground-truth boxes

It works best when the existing YOLO dataset has already been reviewed and is trusted as training ground truth.

## Current limitations

### Detection only

Supported:

```text
class_id x_center y_center width height
```

Not currently supported:

- segmentation polygons
- pose/keypoint labels
- oriented bounding boxes / OBB
- classification-only datasets
- drawing new annotations
- automatic label correction

### Ground truth is assumed correct

The trainer grades the worker against the existing annotations. Incorrect source labels therefore produce incorrect training feedback.

### Not a production annotation platform

This application is a trainer, not a replacement for a full annotation tool. Worker clicks are stored as statistics; they do not create a new YOLO dataset.

### Local statistics only

The current build uses local SQLite. It does not yet provide central accounts, multi-PC synchronization, a server API, or a web dashboard.

### Ten numeric shortcuts

Only the first ten classes currently have direct number shortcuts. Additional classes remain available through the dropdown.

## Recommended worker-training workflow

1. Review and clean the YOLO training dataset.
2. Create one category folder per training topic.
3. Add the correct class definitions.
4. Start the trainer.
5. Create/select the worker profile.
6. Begin with a short randomized session.
7. Use **Show correct boxes** during guided learning.
8. Keep answers hidden during independent practice.
9. Review per-class statistics.
10. Enable weak-class prioritization for follow-up sessions.
11. Continue until the worker reaches your desired accuracy level.

For large datasets, sessions of roughly 25–100 randomly selected images may be more practical than presenting every image at once.

## Suggested repository files

```text
YOLO-Annotation-Trainer/
├── README.md
├── yolo_annotation_trainer.py
├── requirements.txt
├── .gitignore
└── TRAINING_DATA/
```

`requirements.txt`:

```text
PySide6
PyYAML
```

Example `.gitignore`:

```gitignore
.venv/
__pycache__/
*.pyc
TRAINING_DATA/.yolo_trainer_stats.sqlite3
```

Whether `TRAINING_DATA/` itself belongs in Git depends on dataset size, licensing, and data sensitivity.

## Privacy

The current trainer is local-first:

- images are read locally
- labels are read locally
- statistics are stored locally
- no remote database is required
- no account service is required
- no network upload is part of the current workflow

Normal filesystem permissions and organizational data-handling requirements still apply to sensitive datasets.

## Troubleshooting

### No categories appear

The expected path is:

```text
<folder containing trainer>/TRAINING_DATA/<CATEGORY>/...
```

A category needs at least one supported image to appear. After adding one, click **Rescan TRAINING_DATA**.

### Images are found but training will not start

Check that images have matching non-empty YOLO detection `.txt` files.

Example:

```text
images/example.jpg
labels/example.txt
```

### Wrong class names are shown

Check the line order in `classes.txt`:

```text
line 1 -> YOLO ID 0
line 2 -> YOLO ID 1
line 3 -> YOLO ID 2
```

### Number keys select the wrong class

The shortcut number refers to the class position, not the literal YOLO ID shown to the user:

```text
1 -> first class -> ID 0
2 -> second class -> ID 1
0 -> tenth class -> ID 9
```

### No sound

Check operating-system audio output and application audio permissions. Feedback WAV files are generated temporarily at runtime and played through Qt Multimedia.

### SQLite database is not in `TRAINING_DATA`

The directory may be read-only. Check the SQLite path displayed in the main menu; the trainer automatically uses its home-directory fallback when necessary.

## Possible future improvements

- supervisor/admin mode
- password/PIN-protected answer reveal
- separate training and assessment modes
- pass/fail accuracy thresholds
- completion/certification tracking
- confusion matrices for commonly confused classes
- answer-time measurements
- skip-frequency statistics
- spaced repetition
- automatic mistake-review queues
- CSV/Excel statistics export
- centralized PostgreSQL/API storage
- multi-PC synchronization
- web dashboard
- segmentation/OBB/pose training
- packaged standalone executable

## Disclaimer

The trainer evaluates workers against the annotations supplied in the training dataset. Results are only as reliable as the quality and consistency of that ground truth.

For formal qualification or quality-control decisions, combine trainer results with reviewed production examples and an appropriate human QA process.

## License

No license is assumed here. Add the license appropriate for your organization and distribution model, whether private/proprietary or open source.
