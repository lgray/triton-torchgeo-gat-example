ARG SERVERBASE=21.06-py3

FROM nvcr.io/nvidia/pytorch:${SERVERBASE} AS builder

ENV FORCE_CUDA=1
ARG LIB_WITH_CUDA=ON
ARG NPROC=4

RUN git clone https://github.com/rusty1s/pytorch_cluster.git -b 1.5.9
RUN pushd pytorch_cluster &&\
    mkdir build && pushd build &&\
    cmake -DCMAKE_PREFIX_PATH=/opt/conda/lib/python3.8/site-packages/torch -DWITH_CUDA=${LIB_WITH_CUDA} .. &&\
    make -j ${NPROC} && mv *.so /workspace/ && popd &&\
    popd

RUN git clone https://github.com/rusty1s/pytorch_scatter.git -b 2.0.7
RUN pushd pytorch_scatter &&\
    mkdir build && pushd build &&\
    cmake -DCMAKE_PREFIX_PATH=/opt/conda/lib/python3.8/site-packages/torch -DWITH_CUDA=${LIB_WITH_CUDA} .. &&\
    make -j ${NPROC} && mv *.so /workspace/ && popd &&\
    popd

RUN git clone https://github.com/rusty1s/pytorch_spline_conv.git -b 1.2.1
RUN pushd pytorch_spline_conv &&\
    mkdir build && pushd build &&\
    cmake -DCMAKE_PREFIX_PATH=/opt/conda/lib/python3.8/site-packages/torch -DWITH_CUDA=${LIB_WITH_CUDA} .. &&\
    make -j ${NPROC} && mv *.so /workspace/ && popd &&\
    popd

RUN git clone https://github.com/rusty1s/pytorch_sparse.git -b 0.6.10
RUN pushd pytorch_sparse &&\
    mkdir build && pushd build &&\
    cmake -DCMAKE_PREFIX_PATH=/opt/conda/lib/python3.8/site-packages/torch -DWITH_CUDA=${LIB_WITH_CUDA} .. &&\
    make -j ${NPROC} && mv *.so /workspace/ && popd &&\
    popd

RUN git clone https://github.com/rusty1s/pytorch_geometric.git
RUN pushd pytorch_geometric && pip install -e . && popd

FROM nvcr.io/nvidia/tritonserver:${SERVERBASE}

ENV LD_LIBRARY_PATH="/opt/tritonserver/backends/pytorch:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
ENV LD_PRELOAD="/torch_geometric/lib/libtorchscatter.so /torch_geometric/lib/libtorchsparse.so /torch_geometric/lib/libtorchcluster.so /torch_geometric/lib/libtorchsplineconv.so"

RUN mkdir -p /run/shm
RUN mkdir -p /models

RUN mkdir -p /torch_geometric/lib
RUN mkdir -p /torch_geometric/examples/

COPY --from=builder /workspace/libtorchscatter.so /torch_geometric/lib/
COPY --from=builder /workspace/libtorchsparse.so /torch_geometric/lib/
COPY --from=builder /workspace/libtorchcluster.so /torch_geometric/lib/
COPY --from=builder /workspace/libtorchsplineconv.so /torch_geometric/lib/
