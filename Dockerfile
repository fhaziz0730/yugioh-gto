FROM python:3.12-bookworm
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git curl make libgomp1 cmake ninja-build liblua5.3-dev libsqlitecpp-dev libgoogle-glog-dev libconcurrentqueue-dev \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir 'numpy<2' tensorboard 'pybind11==2.13.6' dm-env 'gym==0.26.2' 'gymnasium==0.29.1' optree packaging
COPY engine/lock.json engine/fetch.py ./engine/
RUN python engine/fetch.py /app
COPY engine ./engine
RUN python engine/prepare.py /app \
    && cmake -S engine -B build/engine -G Ninja -DCMAKE_BUILD_TYPE=Release -DCORE_DIR=/app/vendor/ygopro-core-current -DENV_DIR=/app/vendor/ygo-agent \
    && cmake --build build/engine -j 4 \
    && pip install --no-cache-dir -e vendor/ygo-agent/ygoenv
COPY yugioh_gto ./yugioh_gto
RUN pip install --no-deps --no-cache-dir -e .
COPY decks ./decks
COPY data ./data
COPY tests ./tests
COPY scripts ./scripts
ENTRYPOINT ["python", "-m", "yugioh_gto.cli"]
CMD ["preflight"]
