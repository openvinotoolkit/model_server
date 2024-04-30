run:
	docker run --rm --env https_proxy=http://proxy-iind.intel.com:912 \
		--env http_proxy=http://proxy-iind.intel.com:911 \
		--env no_proxy=10.190.0.0/16 \
		-ti --network=host -v ${PWD}/servable_stream:/workspace -v ${PWD}/${SELECTED_MODEL}:/llm_model \
		-v ${PWD}/all-mpnet-base-v2:/embed_model -v ${PWD}/documents:/documents -e SELECTED_MODEL=${SELECTED_MODEL} \
		--entrypoint /bin/bash \
		openvino/model_server1:py