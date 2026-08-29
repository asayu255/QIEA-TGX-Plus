FROM python:3.10-slim

# Repository root inside the container
WORKDIR /QTGX

# Copy the repository before installing the local package
COPY . /QTGX

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        tk-dev && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir .
RUN pip install --no-cache-dir \
    jupyter \
    "ipywidgets<8.0.0" \
    jupyterlab \
    torch_geometric