FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace
COPY requirements-app.txt /tmp/requirements-app.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-app.txt \
    && python -m pip check \
    && python -c "import torch, triton, kornia, wandb; assert torch.__version__.startswith('2.5.1'); assert triton.__version__ == '3.1.0'; print('Environment ready:', torch.__version__, triton.__version__)"
