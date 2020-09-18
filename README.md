# triton-torchgeo-gat-example
An example of running a pytorch-geometric graph attention model in [nvidia triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/).

NB: This works on Windows Subsystem for Linux 2 and Linux. Docker for Windows and MacOSX do not function well or at all.

NVIDIA Triton is an engine for running inference as a service, 
it handles load balancing, model replication, model versioning, and batching.
Client code need only send data to the service endpoint (i.e. model) and it will
receive the model predictions back from the server, without any need for a heavy
ML framework at the client. 

A user can interact with triton as depicted in the following diagram:
![nvidia triton interaction pattern](https://developer.nvidia.com/sites/default/files/akamai/datacenter.png)

There can be multiple clients of multiple types making requests for inference of some specified model(s).
Those requests are deferred to a computing cluster dedicated to running AI inference that is connected to a
library of models that can be loaded or unloaded as needed.

Graph neural networks using the pytorch geometric library are now able to be deployed on triton and this repository
aims to be an example of how to setup the various containers needed to do that. It uses a fast-training, self-contained
model as demonstration, but all the pieces here can be used to deploy much more complex and purpose specific models.

Requirements:
=============
- a recent docker installation (https://docs.docker.com/engine/install/)
- if you want to run the model on a gpu: 
  - install CUDA 11 and its bundled driver (https://developer.nvidia.com/cuda-downloads)
  - install NVIDIA Container Toolkit (https://github.com/NVIDIA/nvidia-docker) 
    - NB: don't install nvidia-docker2!!!!

Use:
====
```bash
./build-images.sh # to build the various triton images
./start.sh # (or ./start_gpu.sh) to run dockerized tests
./cleanup.sh # to clean up the docker containers and networks
```

Generating and running using a singularity container:
=====================================================
```bash
./build-images.sh # to build the various triton images
./make-singularity.sh # to build the singularity images from the triton images
./start-singularity.sh # to run the freshly made singularity containers in a test
./cleanup-singularity.sh # to clean up the singularity containers
```


# How to substitute in your own model:
(assuming your model training is separate from container generation)
1) Train your model and then save it using TorchScript's jit. See [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/jit.html) for details on how to tweak your Pytorch-Geometric model for torch scripting (it is very easy and backwards compatible with saved weights from previous trainings).
2) In Dockerfile.build remove the lines:
```Dockerfile
RUN git clone https://github.com/rusty1s/pytorch_geometric.git

RUN pushd pytorch_geometric && pip install -e . && popd

RUN pushd pytorch_geometric/examples/jit &&\
    echo "torch.jit.save(model, 'gat_test.pt')" >> gat.py &&\
    python gat.py &&\
    mv gat_test.pt /workspace/model.pt
```
and also remove the line:
```Dockerfile
COPY --from=builder /workspace/model.pt /opt/tritonserver/model.pt
```
3) You'll need to create a "model config" to describe the input and output tensor shapes. You can see an example of this for a pytorch model in this repository's `config.pbtxt`. You can find more documentation on how to write the model config [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_configuration.html). Warning - right now GNNs can only use a batch size of zero! This will be fixed in due time.
4) in build-images.sh you can remove the lines:
```bash
docker run -d --name triton_server_temp \
       --shm-size=1g --ulimit memlock=-1 \
       --ulimit stack=67108864 \
       -p8000:8000 -p8001:8001 -p8002:8002 \
       local/gat_test_server:latest
```
and
```bash
docker stop triton_server_temp
docker rm triton_server_temp
```
and then replace the lines:
```bash
mkdir -p ./models/gat_test/1

cp config.pbtxt models/gat_test/
docker cp triton_server_temp:/opt/tritonserver/model.pt models/gat_test/1/
```
with
```bash
mkdir -p ./models/<your model's name>/1

cp yourmodel_config.pbtxt models/<your model's name>/config.pbtxt
cp yourmodel.pt models/<your model's name>/1/model.pt # a .pt file is a saved TorchScript jit file
```
5) Modify `client.py` to supply the correct data for your model, and modify `start[_gpu].sh` to mount any directories containing data you want to read into the model. You should also adjust the arguments to client.py to call the correct model!
