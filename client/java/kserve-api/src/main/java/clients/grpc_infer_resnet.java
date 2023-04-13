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

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import com.google.protobuf.ByteString;

import inference.GrpcPredictV2.InferTensorContents;
import inference.GRPCInferenceServiceGrpc;
import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub;
import inference.GrpcPredictV2.ModelInferRequest;
import inference.GrpcPredictV2.ModelInferResponse;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

public class grpc_infer_resnet {

	public static void main(String[] args) {
		Options opt = new Options();
		opt.addOption("h", "help", false, "Show this help message and exit");
		opt.addOption(Option.builder("a").longOpt("grpc_address").hasArg().desc("Specify url to grpc service. ")
				.argName("GRPC_ADDRESS").build());
		opt.addOption(Option.builder("p").longOpt("grpc_port").hasArg().desc("Specify port to grpc service. ")
				.argName("GRPC_PORT").build());
		opt.addOption(Option.builder("i").longOpt("input_name").hasArg().desc("Specify input tensor name. ")
				.argName("INPUT_NAME").build());
		opt.addOption(Option.builder("n").longOpt("model_name").hasArg()
				.desc("Define model name, must be same as is in service").argName("MODEL_NAME").build());
		opt.addOption(Option.builder("v").longOpt("model_version").hasArg().desc("Define model version. ")
				.argName("MODEL_VERSION").build());
		opt.addOption(Option.builder("imgs").longOpt("image_list").hasArg()
				.desc("Path to a file with a list of labeled images. ")
				.argName("IMAGES").build());
		opt.addOption(Option.builder("lbs").longOpt("labels").hasArg().desc("Path to a file with a list of labels. ")
				.argName("LABELS").build());
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
			formatter.printHelp("grpc_infer_resnet [OPTION]...", opt);
			System.exit(0);
		}
		if (!cmd.hasOption("imgs")) {
			System.err.println("Missing required option: imgs");
			System.exit(1);
		}
		if (!cmd.hasOption("lbs")) {
			System.err.println("Missing required option: lbs");
			System.exit(1);
		}

		ManagedChannel channel = ManagedChannelBuilder
				.forAddress(cmd.getOptionValue("a", "localhost"), Integer.parseInt(cmd.getOptionValue("p", "9000")))
				.usePlaintext().build();
		GRPCInferenceServiceBlockingStub grpc_stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);

		// Generate the request
		ModelInferRequest.Builder request = ModelInferRequest.newBuilder();
		request.setModelName(cmd.getOptionValue("n", "resnet"));
		request.setModelVersion(cmd.getOptionValue("v", ""));

		// Populate the inputs in inference request
		ModelInferRequest.InferInputTensor.Builder input = ModelInferRequest.InferInputTensor
				.newBuilder();
		String defaultInputName = "0";
		input.setName(cmd.getOptionValue("i", defaultInputName));
		input.setDatatype("BYTES");
		input.addShape(1);

		List<String> labels = new ArrayList<>();
		try (FileInputStream fis = new FileInputStream(cmd.getOptionValue("lbs"))) {
			try (Scanner sc = new Scanner(fis)) {
				while (sc.hasNextLine()) {
					labels.add(sc.nextLine());
				}
			}
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		try (FileInputStream fis = new FileInputStream(cmd.getOptionValue("imgs"))) {
			try (Scanner sc = new Scanner(fis)) {
				while (sc.hasNext()) {
					request.clearInputs();
					String[] line = sc.nextLine().split(" ");
					String fileName = line[0];
					int label = Integer.parseInt(line[1]);
					FileInputStream imageStream = new FileInputStream(fileName);
					InferTensorContents.Builder input_data = InferTensorContents.newBuilder();
					input_data.addBytesContents(ByteString.readFrom(imageStream));
					input.setContents(input_data);
					request.addInputs(0, input);

					ModelInferResponse response = grpc_stub.modelInfer(request.build());

					// Get the response outputs
					FloatBuffer fb = response.getRawOutputContentsList().get(0).asReadOnlyByteBuffer()
							.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
					float[] op = new float[fb.remaining()];
					fb.get(op);
					int lb = arrayMax(op);

					if (lb == label)
						System.out.format("%s classified as %d %s\n", fileName, lb, labels.get(lb));
					else
						System.out.format("%s classified as %d %s should be %d %s\n", fileName, lb, labels.get(lb),
								label, labels.get(label));

				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		channel.shutdownNow();

	}

	private static int arrayMax(float[] arr) {
		float max = Float.NEGATIVE_INFINITY;
		int in = -1;

		for (int i = 0; i < arr.length; i++) {
			if (arr[i] > max) {
				in = i;
				max = arr[i];
			}
		}

		return in;
	}
}