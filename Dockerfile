# ==========================================
# STAGE 1: BUILDER
# ==========================================
FROM python:3.9-slim AS builder

# Install tools C++ (Compiler, CMake, dan OpenCV Headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependencies dan install pybind11
COPY requirements.txt .
RUN pip install --no-cache-dir pybind11

# Copy folder C++ dan compile jadi format wheel (.whl) yang siap pakai
COPY cpp_core/ cpp_core/
RUN cd cpp_core && pip wheel . -w /wheels


# ==========================================
# STAGE 2: PRODUCTION RUNNER
# ==========================================
FROM python:3.9-slim

WORKDIR /app

# Tarik semua keluarga OpenCV (LAPACK & OpenBLAS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libopencv-core-dev \
    libopencv-imgproc-dev \
    libopencv-photo-dev \
    liblapack3 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

# Ambil hasil kompilasi C++ dari Stage 1 (tanpa bawa compiler-nya)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]