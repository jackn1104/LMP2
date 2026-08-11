FROM ubuntu:24.04 AS build

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake ninja-build \
    libeigen3-dev libint2-dev libhdf5-dev \
    libopenmpi-dev openmpi-bin libopenblas-dev liblapack-dev \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/LMP2
COPY . .
RUN cmake --preset release && cmake --build --preset release

ENTRYPOINT ["/opt/LMP2/LMP2"]
CMD ["--help"]

