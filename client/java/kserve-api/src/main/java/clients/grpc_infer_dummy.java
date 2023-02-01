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

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import inference.GRPCInferenceServiceGrpc;
import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub;
import inference.GrpcPredictV2.InferTensorContents;
import inference.GrpcPredictV2.ModelInferRequest;
import inference.GrpcPredictV2.ModelInferResponse;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

public class grpc_infer_dummy {

	public static void main(String[] args) {
		Options opt = new Options();
		opt.addOption("h", "help", false, "Show this help message and exit");
		opt.addOption(Option.builder("a").longOpt("grpc_address").hasArg().desc("Specify url to grpc service. ")
				.argName("GRPC_ADDRESS").build());
		opt.addOption(Option.builder("p").longOpt("grpc_port").hasArg().desc("Specify port to grpc service. ")
				.argName("GRPC_PORT").build());
		opt.addOption(Option.builder("i").longOpt("input_name").hasArg().desc("Specify input tensor name. ")
				.argName("INPUT_NAME").build());
		opt.addOption(Option.builder("o").longOpt("output_name").hasArg().desc("Specify output tensor name. ")
				.argName("OUTPUT_NAME").build());
		opt.addOption(Option.builder("n").longOpt("model_name").hasArg()
				.desc("Define model name, must be same as is in service").argName("MODEL_NAME").build());
		opt.addOption(Option.builder("v").longOpt("model_version").hasArg().desc("Define model version. ")
				.argName("MODEL_VERSION").build());
		CommandLineParser parser = new DefaultParser();
		CommandLine cmd = null;
		try {
			cmd = parser.parse(opt, args);
		} catch (ParseException exp) {
			System.err.println("Parsing failed.  Reason: " + exp.getMessage());
			System.exit(1);
		}

		if (cmd.hasOption("h")) {
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp("grpc_infer_dummy [OPTION]...", opt);
			System.exit(0);
		}

		ManagedChannel channel = ManagedChannelBuilder
				.forAddress(cmd.getOptionValue("a", "localhost"), Integer.parseInt(cmd.getOptionValue("p", "9000")))
				.usePlaintext().build();
		GRPCInferenceServiceBlockingStub grpc_stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);

		// Generate the request
		ModelInferRequest.Builder request = ModelInferRequest.newBuilder();
		request.setModelName(cmd.getOptionValue("n", "dummy"));
		request.setModelVersion(cmd.getOptionValue("v", ""));

		// Input data
		List<Float> lst = Arrays.asList(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f);
		InferTensorContents.Builder input_data = InferTensorContents.newBuilder();
		input_data.addAllFp32Contents(lst);

		// Populate the inputs in inference request
		ModelInferRequest.InferInputTensor.Builder input = ModelInferRequest.InferInputTensor
				.newBuilder();
		String defaultInputName = "b";
		input.setName(cmd.getOptionValue("i", defaultInputName));
		input.setDatatype("FP32");
		input.addShape(1);
		input.addShape(10);
		input.setContents(input_data);

		request.addInputs(0, input);

		ModelInferResponse response = grpc_stub.modelInfer(request.build());

		// Get the response outputs
		FloatBuffer fb = response.getRawOutputContentsList().get(0).asReadOnlyByteBuffer()
				.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
		float[] op = new float[fb.remaining()];
		fb.get(op);

		for (int i = 0; i < op.length; i++) {
			System.out.println(
					Float.toString(lst.get(i)) + " => " +
							Float.toString(op[i]));
		}

		channel.shutdownNow();

	}
}