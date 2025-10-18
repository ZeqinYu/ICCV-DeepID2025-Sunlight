FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

WORKDIR /app

# 安装系统依赖项，包括 gcc-7, g++-7 和 openblas
RUN apt-get update && \
    apt-get install -y gcc-7 g++-7 git build-essential cmake libopenblas-dev unzip && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 70



# 安装 pip 包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install ninja numpy

# 克隆并安装 MinkowskiEngine
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git && \
    cd MinkowskiEngine && \
    python setup.py install --blas_include_dirs=/usr/include --blas=openblas && \
    cd .. && rm -rf MinkowskiEngine

# 拷贝应用代码
COPY src src
# COPY src/Cue_Net/convnextv2_base_22k_224_ema.pt src/Cue_Net/
# COPY src/Cue_Net/convnextv2_large_22k_224_ema.pt src/Cue_Net/
ENV PYTHONPATH=/app/src

# 暴露端口
EXPOSE 8000

# 默认启动命令（FastAPI）
CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "8000", "src/main.py"]
