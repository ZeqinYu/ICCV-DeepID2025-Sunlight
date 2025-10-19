FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y gcc-7 g++-7 git build-essential cmake libopenblas-dev unzip && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 70

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install ninja numpy

RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git && \
    cd MinkowskiEngine && \
    python setup.py install --blas_include_dirs=/usr/include --blas=openblas && \
    cd .. && rm -rf MinkowskiEngine

COPY src src
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "8000", "src/main.py"]
