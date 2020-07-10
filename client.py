#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import sys
import random

import tritongrpcclient

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-m',
                        '--model_name',
                        type=str,
                        required=True,
                        help='Model name')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')

    FLAGS = parser.parse_args()
    try:
        triton_client = tritongrpcclient.InferenceServerClient(url=FLAGS.url,
                                                               verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = FLAGS.model_name 

    for i in range(50):
        # Infer
        inputs = []
        outputs = []
        
        nnodes = random.randint(100, 4000)
        nedges = random.randint(8000, 15000)

        inputs.append(tritongrpcclient.InferInput('x__0', [1, nnodes, 1433], 'FP32'))
        inputs.append(tritongrpcclient.InferInput('edgeindex__1', [1, 2, nedges], "INT64"))

        x = np.random.normal(-10, 4, (nnodes, 1433)).astype(np.float32)
        x[x < 0] = 0.
        x[x > 1] = 1.
        edge_index = np.random.randint(0, nnodes, (2, nedges), dtype=np.int64)
       
        print(x.shape)
        print(edge_index.shape)

        # prepare inputs
        inputs[0].set_data_from_numpy(x[None])
        inputs[1].set_data_from_numpy(edge_index[None])

        # prepare outputs
        outputs.append(tritongrpcclient.InferRequestedOutput('logits__0'))

        # get the output
        results = triton_client.infer(model_name=model_name,
                                      inputs=inputs,
                                      outputs=outputs)
        output0_data = results.as_numpy('logits__0')
        print(output0_data)

    statistics = triton_client.get_inference_statistics(model_name=model_name)
    print(statistics)
    if len(statistics.model_stats) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)

    print('PASS: infer')
