
# BEV IPM Prototype (Phase-1) — Step 1

## Goal (Step 1)
Select a nuScenes sample and resolve the 6 camera images + metadata.

## Setup (local)
1) Create venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
````

2. Install deps:

```bash
pip install -r requirements.txt
```

3. Run Step 1:

### Using scene + index

```bash
python runProjection.py --dataroot /path/to/nuscenes --version v1.0-mini --scene scene-0061 --index 10
```

### Using sample token

```bash
python runProjection.py --dataroot /path/to/nuscenes --version v1.0-mini --sample_token <token>
```

If it prints 6 camera paths and sizes, Step 1 is done.

## Move to EC2 tomorrow

1. Copy folder:

```bash
scp -i BEV_prototype.pem -r bev_ipm ubuntu@<EC2_HOST>:~/bev_ipm
```

2. On EC2:

```bash
cd ~/bev_ipm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python runProjection.py --dataroot /data/nuscenes --version v1.0-trainval --scene scene-0061 --index 10