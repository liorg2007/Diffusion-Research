#!/bin/sh
# usage: launch.sh <orig|fov512> [--twin-disjoint]
cd /home/zera/project/Diffusion-Research
D=experiments/005-nuisance-projection/official
ARM=$1; EXTRA=$2
if [ "$ARM" = orig ]; then SUB=feats; SPL=experiments/003-baselines-eval/splits.csv; else SUB=fov512; SPL=experiments/004-phase4-features/splits_fov512.csv; fi
OUT=$D/$ARM; [ -n "$EXTRA" ] && OUT=$D/${ARM}_twin_disjoint
env/.venv/bin/python -I experiments/005-nuisance-projection/run.py --subdir $SUB --splits $SPL --out-dir $OUT --label OFFICIAL \
  --generators dalle2 dalle3 firefly midjourney-v5 stable-diffusion-2 stable-diffusion-xl sd3_large --heldout-generators flux \
  --grouped-cv --balanced --group-bootstrap --per-pool --headroom-rule --primary-pools vision_native $EXTRA >> $OUT/run.log 2>&1
