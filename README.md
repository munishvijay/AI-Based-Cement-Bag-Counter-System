# AI-Based Cement Bag Counter System

An independent Python computer-vision project that detects and tracks cement bags on conveyor footage and counts tracked objects when they cross a configurable virtual line.

## Features

- Video-file and webcam/camera input.
- YOLO-based cement-bag detection.
- Multi-object tracking with Ultralytics and ByteTrack.
- Virtual line-crossing count with one count per tracker ID.
- Annotated preview and optional MP4 output.
- Optional CSV event log.
- YOLO training helper and unit tests for counting logic.
- GitHub Actions workflow for automated unit tests.

## Model and dataset requirement

**This repository contains the application and training helper, not trained cement-bag weights or a labelled dataset.** A generic COCO-pretrained model does not provide a dedicated cement-bag class. Train a detector on a dataset you own or are licensed to use, then supply its resulting `.pt` file. No model accuracy, speed, or production-readiness result is claimed.

## Project structure

```text
app.py
train.py
requirements.txt
tests/test_counter.py
.github/workflows/tests.yml
.gitignore
LICENSE
README.md
```

## Requirements and installation

- Python 3.10+
- A trained Ultralytics YOLO detection model (`.pt`)
- A video file or webcam; GPU is optional but may improve inference speed

```bash
git clone https://github.com/munishvijay/AI-Based-Cement-Bag-Counter-System.git
cd AI-Based-Cement-Bag-Counter-System
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

Install packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset format

Prepare a YOLO object-detection dataset with images and matching label files:

```text
cement-bag-dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each label row must use `class_id x_center y_center width height` with normalized coordinates. Create a dataset YAML, for example `data/cement_bags.yaml`:

```yaml
path: /absolute/path/to/cement-bag-dataset
train: images/train
val: images/val
names:
  0: cement_bag
```

Only use footage and images you own or have permission to use. Avoid committing confidential plant video or large datasets.

## Train the detector

```bash
python train.py --data data/cement_bags.yaml --model yolo11n.pt --epochs 100 --image-size 640 --batch 8
```

The training framework normally saves the best weights at `runs/cement_bag/detector/weights/best.pt`. Use the actual path printed by the training run. Model weights are excluded from Git by `.gitignore`.

## Run the counter

Video file:

```bash
python app.py --source videos/conveyor.mp4 --model runs/cement_bag/detector/weights/best.pt --class-id 0 --display
```

Webcam:

```bash
python app.py --source 0 --model runs/cement_bag/detector/weights/best.pt --class-id 0 --display
```

Save an annotated MP4 and CSV count-event log:

```bash
python app.py --source videos/conveyor.mp4 --model runs/cement_bag/detector/weights/best.pt --class-id 0 --output output/annotated.mp4 --csv output/count_events.csv
```

Press `q` in the preview window to stop.

### Command options

| Option | Default | Purpose |
|---|---:|---|
| `--source` | Required | Video path or camera index such as `0` |
| `--model` | Required | Trained YOLO `.pt` weights |
| `--class-id` | `0` | Cement-bag class index |
| `--line-y` | `0.5` | Horizontal line position, normalized 0.0-1.0 |
| `--confidence` | `0.35` | Detection confidence threshold |
| `--image-size` | `640` | YOLO inference size |
| `--tracker` | `bytetrack.yaml` | Ultralytics tracker configuration |
| `--output` | Optional | Annotated MP4 destination |
| `--csv` | Optional | CSV count-event destination |
| `--display` | Off | Show preview window |

The counter counts each tracker ID once per run. If tracking loses an object and assigns it a new ID, the same physical bag may be counted again. Tune and validate the system using representative camera footage.

## Unit tests

The counting logic can be tested without a model file or video:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions is configured to run these tests on pushes and pull requests to `main`. A passing workflow result must be confirmed in the Actions tab; no successful run is claimed here.

## Limitations and intended use

Detection quality depends on the training data, camera position, lighting, bag overlap, conveyor speed, and tracker behaviour. Validate results against manually checked footage before operational use. This is an experimental learning project, not a certified industrial safety, dispatch, billing, or quality-control system.

## License

Licensed under the BSD 3-Clause License. See [LICENSE](LICENSE).
