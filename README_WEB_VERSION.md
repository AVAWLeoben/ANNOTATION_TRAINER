# YOLO Annotation Trainer — Browser build 2026-10-09a-web

Static browser conversion of `yolo_annotation_trainer.py` for GitHub Pages.

## Removed intentionally

- Worker profiles
- SQLite statistics database
- Per-worker statistics
- Previous-performance summaries
- Weak-class prioritization

## Retained

- TRAINING_DATA-style category menu
- Immediate subfolder = training category
- Recursive image discovery
- YOLO detection labels (`class cx cy width height`)
- `images/...` + `labels/...` layouts and image+txt same-folder layouts
- class names from `classes.txt`, `class_names.txt`, `names.txt`, `obj.names`, or common `data.yaml`/`dataset.yaml` names layouts
- Randomized sessions and optional image count limit
- Class selection + click-the-object training loop
- Smallest unsolved box wins when boxes overlap
- Red lock after wrong answer until that object is corrected
- Green solved boxes
- Answer reveal (`H`)
- Skip/requeue (`S`)
- Class hotkeys `1`–`9`, `0` for class 10
- Score, streak, correct, accuracy, object progress
- Reward/error tones and correct-answer celebration
- Edit class names; writes `classes.txt` when direct folder access is available
- Lazy image decoding with a small cache

## Run

Host `index.html` on GitHub Pages and open it in a Chromium-based browser for the best folder-access experience.

Click **Open TRAINING_DATA Folder** and select the local `TRAINING_DATA` directory. Files are read locally by the browser; the app has no upload backend.

A read-only `webkitdirectory` fallback is also provided for browsers without the File System Access API.
