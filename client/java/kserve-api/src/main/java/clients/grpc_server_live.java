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

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import inference.GRPCInferenceServiceGrpc;
import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub;
import inference.GrpcPredictV2.ServerLiveRequest;
import inference.GrpcPredictV2.ServerLiveResponse;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

public class grpc_server_live {

	public static void main(String[] args) {
		Options opt = new Options();
		opt.addOption("h", "help", false, "Show this help message and exit");
		opt.addOption(Option.builder("a").longOpt("grpc_address").hasArg().desc("Specify url to grpc service. ")
				.argName("GRPC_ADDRESS").build());
		opt.addOption(Option.builder("p").longOpt("grpc_port").hasArg().desc("Specify port to grpc service. ")
				.argName("GRPC_PORT").build());
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
			formatter.printHelp("grpc_server_live [OPTION]...", opt);
			System.exit(0);
		}

		ManagedChannel channel = ManagedChannelBuilder
				.forAddress(cmd.getOptionValue("a", "localhost"), Integer.parseInt(cmd.getOptionValue("p", "9000")))
				.usePlaintext().build();
		GRPCInferenceServiceBlockingStub grpc_stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);

		// Generate the request
		ServerLiveRequest.Builder request = ServerLiveRequest.newBuilder();

		ServerLiveResponse response = grpc_stub.serverLive(request.build());

		// Get the response outputs
		System.out.format("Server Live: %b\n", response.getLive());
		channel.shutdownNow();

	}
}