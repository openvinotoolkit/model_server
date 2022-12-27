//*****************************************************************************
// Copyright 2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

package clients;


import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import inference.GRPCInferenceServiceGrpc;
import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub;
import inference.GrpcPredictV2.InferTensorContents;
import inference.GrpcPredictV2.ModelInferRequest;
import inference.GrpcPredictV2.ModelInferResponse;
import inference.GrpcPredictV2.ServerLiveRequest;
import inference.GrpcPredictV2.ServerLiveResponse;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

public class grpc_infer_dummy {

	public static void main(String[] args) {

		String host = args.length > 0 ? args[0] : "localhost";
		int port = args.length > 1 ? Integer.parseInt(args[1]) : 9000;

		String model_name = "dummy";
		String model_version = "";

		ManagedChannel channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
		GRPCInferenceServiceBlockingStub grpc_stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);

		// Generate the request
		ModelInferRequest.Builder request = ModelInferRequest.newBuilder();
		request.setModelName(model_name);
		request.setModelVersion(model_version);

		// Input data
		List<Float> lst = Arrays.asList(0f,1f,2f,3f,4f,5f,6f,7f,8f,9f);
		InferTensorContents.Builder input_data = InferTensorContents.newBuilder();
		input_data.addAllFp32Contents(lst);

		// Populate the inputs in inference request
		ModelInferRequest.InferInputTensor.Builder input = ModelInferRequest.InferInputTensor
				.newBuilder();
		input.setName("b");
		input.setDatatype("FP32");
		input.addShape(1);
		input.addShape(10);
		input.setContents(input_data);


		request.addInputs(0, input);

		// Populate the outputs in the inference request
		ModelInferRequest.InferRequestedOutputTensor.Builder output = ModelInferRequest.InferRequestedOutputTensor
				.newBuilder();
		output.setName("a");


		request.addOutputs(0, output);

		ModelInferResponse response = grpc_stub.modelInfer(request.build());

		// Get the response outputs
		float[] op = toArray(response.getRawOutputContentsList().get(0).asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer());

		for (int i = 0; i < op.length; i++) {
			System.out.println(
				Float.toString(lst.get(i)) + " => " +
				Float.toString(op[i]));
		}
		
		channel.shutdownNow();
		
	}

	public static float[] toArray(FloatBuffer b) {
		if (b.hasArray()) {
			if (b.arrayOffset() == 0)
				return b.array();

			return Arrays.copyOfRange(b.array(), b.arrayOffset(), b.array().length);
		}

		b.rewind();
		float[] tmp = new float[b.remaining()];
		b.get(tmp);

		return tmp;
	}

}
