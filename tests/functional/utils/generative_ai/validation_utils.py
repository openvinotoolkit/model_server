#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# pylint: disable=too-many-nested-blocks
# pylint: disable=unused-argument

import base64
import inspect
import os
import shutil
import sys
from io import BytesIO

import numpy as np
import soundfile as sf
from jiwer import wer, wer_standardize, Compose, RemovePunctuation
from PIL import Image

from tests.functional.utils.logger import get_logger, step
from tests.functional.utils.inference.serving.openai import OpenAIWrapper, OpenAIFinishReason
from tests.functional.config import save_image_to_artifacts
from tests.functional.config import artifacts_dir, pipeline_type
from tests.functional.models.models_datasets import FeatureExtractionModelDataset

logger = get_logger(__name__)


class GenerativeAIValidationUtils:

    @staticmethod
    def validate_llm_outputs(
            model_name,
            outputs,
            stream=False,
            validate_func=None,
            allow_empty_response=False,
            tools_enabled=False,
            validate_tools=False,
            **kwargs,
    ):
        logger.info(outputs)
        model_instance = kwargs.get("model_instance", None)
        outputs_content = []
        assert outputs is not None and len(outputs) > 0, f"No output collected for node with model: {model_name}"
        stream_content = []
        for index, output in enumerate(outputs):
            assert output.model == model_name, f"Invalid model name: {output.model}; Expected: {model_name}"
            for choice in output.choices:
                logger.debug(f"Choice: {choice}")
                if validate_func is not None:
                    validate_func(
                        stream,
                        choice,
                        outputs_content,
                        stream_content,
                        allow_empty_response,
                        tools_enabled,
                        validate_tools,
                        index=index,
                        **kwargs,
                    )
                if tools_enabled:
                    model_type = None
                    for _, model_type in inspect.getmembers(sys.modules['ovms.constants.models'], inspect.isclass):
                        if (model_instance is not None and hasattr(model_type, "name") and
                                model_instance.name == model_type.name):
                            if model_instance.allows_reasoning:
                                if hasattr(choice, "message") and hasattr(choice.message, "model_extra"):
                                    assert "reasoning_content" in choice.message.model_extra, \
                                        f"Empty reasoning content: {choice}"
                                    logger.info(f"Reasoning content: {choice.message.model_extra['reasoning_content']}")
                            else:
                                assert "reasoning_content" not in str(choice), \
                                    f"Reasoning content is not empty: {choice}"
        if stream:
            logger.info(stream_content)
            if tools_enabled:
                assert len(stream_content) > 0, f"Empty tool calls: {stream_content}"
                if validate_tools:
                    return stream_content
                outputs_content.append("".join(stream_content))
            else:
                assert len(stream_content) > 0, f"Empty stream_content: {stream_content}"
                outputs_content.append("".join(stream_content))
        return outputs_content

    @staticmethod
    def validate_finish_reason(endpoint, raw_outputs, request_params, finish_reason):
        # validate finish reason only for ignore_eos=True (default) - for ignore_eos=False value may vary
        if request_params.ignore_eos or request_params.ignore_eos is None:
            assert len(raw_outputs) > 0, "No outputs to check!"
            stream_finish_reason = None
            error_message = "Unexpected finish_reason: {}; expected: {}"
            for raw_output in raw_outputs:
                if endpoint == OpenAIWrapper.RESPONSES:
                    if request_params.stream:
                        if hasattr(raw_output, "response"):
                            if finish_reason == OpenAIFinishReason.STOP:
                                stream_finish_reason = OpenAIFinishReason.STOP \
                                    if raw_output.response.completed_at is not None else \
                                    raw_output.response.completed_at
                            else:
                                stream_finish_reason = raw_output.response.incomplete_details.reason \
                                    if hasattr(raw_output.response.incomplete_details, "reason") else \
                                    raw_output.response.incomplete_details
                            assert stream_finish_reason in [None, finish_reason], \
                                error_message.format(stream_finish_reason, finish_reason)
                    else:
                        if finish_reason == OpenAIFinishReason.STOP:
                            assert raw_output.completed_at is not None, \
                                error_message.format(raw_output.completed_at, finish_reason)
                        else:
                            assert raw_output.incomplete_details is not None and \
                                   raw_output.incomplete_details.reason == finish_reason, \
                                error_message.format(raw_output.incomplete_details, finish_reason)
                else:
                    for choice in raw_output.choices:
                        if request_params.stream:
                            stream_finish_reason = choice.finish_reason
                            assert stream_finish_reason in [None, finish_reason], \
                                error_message.format(stream_finish_reason, finish_reason)
                        else:
                            assert choice.finish_reason == finish_reason, \
                                error_message.format(choice.finish_reason, finish_reason)
            if request_params.stream:
                assert stream_finish_reason == finish_reason, \
                    error_message.format(stream_finish_reason, finish_reason)

    @staticmethod
    def validate_stop(outputs, stop, stream, include_stop_str_in_output):
        stop = [stop] if isinstance(stop, str) else stop
        if stream:
            assert any(stop_value in outputs[-1] for stop_value in stop), \
                f"None of the stop values: {stop} were found in the output: {outputs[-1]}"
        else:
            for output in outputs:
                if include_stop_str_in_output:
                    assert any(stop_value in output for stop_value in stop), \
                        f"None of the stop values: {stop} were found in the output: {output}"
                else:
                    assert all(stop_value not in output for stop_value in stop), \
                        f"Stop values: {stop} were found in the output: {output}"

    @staticmethod
    def validate_usage(endpoint, stream, raw_outputs, max_tokens=None):
        assert len(raw_outputs) > 0, "No outputs to check!"
        for raw_output in raw_outputs[:-1]:
            if endpoint == OpenAIWrapper.RESPONSES:
                if not hasattr(raw_output, "response"):
                    continue
                usage = raw_output.response.usage if stream else raw_output.usage
            else:
                usage = raw_output.usage
            assert usage is None, f"Unexpected usage value: {usage}; expected: None"

        last_usage = raw_outputs[-1].response.usage if endpoint == OpenAIWrapper.RESPONSES and stream else \
            raw_outputs[-1].usage
        generated_tokens = last_usage.output_tokens if endpoint == OpenAIWrapper.RESPONSES else \
            last_usage.completion_tokens
        prompt_tokens = last_usage.input_tokens if endpoint == OpenAIWrapper.RESPONSES else last_usage.prompt_tokens
        total_tokens = last_usage.total_tokens

        if max_tokens is not None:
            assert max_tokens == generated_tokens, \
                f"Unexpected token count: {generated_tokens}; expected: {max_tokens}"
        else:
            assert generated_tokens > 0, f"No generated tokens reported: {generated_tokens}"
        assert generated_tokens + prompt_tokens == total_tokens, \
            f"Unexpected total_tokens value: {total_tokens}; expected: {generated_tokens + prompt_tokens}"

        return generated_tokens, prompt_tokens, total_tokens

    @classmethod
    def validate_chat_completions_outputs(
            cls, model_name, outputs, stream=False, allow_empty_response=False, tools_enabled=False,
            validate_tools=False, **kwargs
    ):
        model_instance = kwargs.get("model_instance", None)
        model_pipeline_type = getattr(model_instance, "pipeline_type", None)
        effective_pipeline_type = model_pipeline_type if model_pipeline_type is not None else pipeline_type

        def validate_choice(
                stream,
                choice,
                outputs_content,
                stream_content,
                allow_empty_response,
                tools_enabled=False,
                validate_tools=False,
                **kwargs,
        ):
            if stream:
                if kwargs.get("index", None) == 0 and effective_pipeline_type in (None, "CB"):
                    if choice.delta.content is None:
                        # check role for empty streaming messages
                        assert choice.delta.role == "assistant", \
                            f"Unexpected role in the first response: {choice.delta.role}"
                if tools_enabled and validate_tools:
                    if choice.delta.tool_calls is not None:
                        stream_content.append(choice.delta.tool_calls)
                else:
                    if choice.delta.content is not None:
                        stream_content.append(choice.delta.content)
            else:
                if not allow_empty_response:
                    assert len(choice.message.content) > 0, f"Empty response content: {choice}"
                if tools_enabled and validate_tools:
                    # When tools are enabled content might not be empty
                    assert choice.message.role == "assistant", f"Unexpected role: {choice.message.role}"
                    assert len(choice.message.tool_calls) > 0, f"Empty tool calls: {choice.message}"
                    logger.info(choice.message.tool_calls)
                    outputs_content.append(choice.message.tool_calls)
                else:
                    logger.info(choice.message.content)
                    outputs_content.append(choice.message.content)

        return cls.validate_llm_outputs(
            model_name, outputs, stream, validate_choice, allow_empty_response, tools_enabled, validate_tools, **kwargs
        )

    @classmethod
    def validate_completions_outputs(
            cls, model_name, outputs, stream=False, allow_empty_response=False, **kwargs
    ):
        def validate_choice(
                stream,
                choice,
                outputs_content,
                stream_content,
                allow_empty_response,
                tools_enabled=False,
                validate_tools=False,
                **kwargs,
        ):
            if stream:
                if choice.text is not None:
                    stream_content.append(choice.text)
            else:
                if not allow_empty_response:
                    assert len(choice.text) > 0, f"Empty response content: {choice}"
                logger.info(choice.text)
                outputs_content.append(choice.text)

        return cls.validate_llm_outputs(model_name, outputs, stream, validate_choice, allow_empty_response, **kwargs)

    @classmethod
    def validate_responses_outputs(cls, model_name, outputs, stream=False, allow_empty_response=False, **kwargs):
        logger.info(outputs)
        outputs_content = []
        assert outputs is not None and len(outputs) > 0, f"No output collected for node with model: {model_name}"
        stream_content = []
        for output in outputs:
            if stream:
                if output.type == "response.created":
                    assert output.response.model == model_name,\
                        f"Invalid model name: {output.response.model}; Expected: {model_name}"
                elif output.type == "response.output_text.delta" and output.delta is not None:
                    stream_content.append(output.delta)
                elif output.type in ("response.completed", "response.incomplete"):
                    if not allow_empty_response:
                        assert len(stream_content) > 0, f"Empty stream_content: {stream_content}"
                    assert "".join(stream_content) == output.response.output_text, \
                        f"stream_content: {stream_content} does not match output_text: {output.response.output_text}"
                    logger.info(output.response.output_text)
                    outputs_content.append(output.response.output_text)
            else:
                assert output.model == model_name, f"Invalid model name: {output.model}; Expected: {model_name}"
                for output_item in output.output:
                    if output_item.type == "message":
                        for content_item in output_item.content:
                            if content_item.type == "output_text":
                                if not allow_empty_response:
                                    assert content_item.text, f"Empty response content: {content_item}"
                                logger.info(content_item.text)
                                outputs_content.append(content_item.text)
        return outputs_content

    @classmethod
    def validate_embeddings_outputs(cls, model_name, outputs, allow_empty_response=False):
        outputs_content = []
        assert outputs is not None and len(outputs.data) > 0, f"No output collected for node with model: {model_name}"
        for output in outputs.data:
            if not allow_empty_response:
                output_embedding = output.embedding
                assert len(output_embedding) > 0, f"Empty response content: {output_embedding}"
                logger.info(output_embedding)
                outputs_content.append(output_embedding)
        return outputs_content

    @classmethod
    def validate_rerank_outputs(cls, model_name, outputs, allow_empty_response=False):
        outputs_content = []
        assert outputs is not None and len(outputs.results) > 0, \
            f"No output collected for node with model: {model_name}"

        for i, output in enumerate(outputs.results):
            if not allow_empty_response:
                relevance_score = output.relevance_score
                assert relevance_score > 0, f"Empty response content: {relevance_score}"
                logger.info(f"{[i]}: relevance_score={relevance_score}")
                outputs_content.append(relevance_score)
        return outputs_content

    @staticmethod
    def is_valid_image_pillow(image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()
                return True
        except (IOError, SyntaxError) as e:
            logger.error(e)
            return False

    @classmethod
    def validate_image_outputs(cls, model_name, outputs, image_path=None, **kwargs):
        outputs_content = []
        request_parameters = kwargs.get("request_parameters", None)

        assert outputs is not None and len(outputs.data) == request_parameters.n, \
            f"No output collected for node with model: {model_name}"

        image_base64 = outputs.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        if image_path is not None:
            image = Image.open(BytesIO(image_bytes))
            image.save(image_path)
            logger.info(f"Image saved: {image_path}")
            if save_image_to_artifacts:
                image_dst = os.path.join(artifacts_dir, os.path.basename(image_path))
                shutil.copy(image_path, image_dst)
                logger.info(f"Image saved: {image_dst}")
            assert cls.is_valid_image_pillow(image_path), f"Image is invalid: {image_bytes}"
            width, height = image.size
            x, y = request_parameters.size.split("x")
            assert width == int(x) and height == int(y), f"Unexpected image size: {image.size}"

        outputs_content.append(image_base64)
        return outputs_content

    @staticmethod
    def validate_jinja_outputs(jinja_template, outputs):
        try:
            logger.info("Check if output contains jinja template keyword")
            assert jinja_template.lower() in str(outputs).lower()
        except AssertionError:
            logger.info("Check if output does not contain OpenVINO keyword (default prompt)")
            assert "OpenVINO".lower() not in str(outputs).lower(), f"Jinja template was probably not used correctly. " \
                                                                   f"Outputs: {outputs}"

    @classmethod
    def validate_models_list_outputs(cls, models, outputs):
        models_names = [model.name for model in models]
        models_list = []
        for model_object in outputs.data:
            cls.validate_models_retrieve_outputs(model_object.id, model_object)
            models_list.append(model_object.id)
        assert set(models_names) == set(models_list), \
            f"v3/models output: {models_list} does not match models loaded to OVMS: {models_names}"
        return models_list

    @classmethod
    def validate_models_retrieve_outputs(cls, model_name, outputs):
        assert outputs.id == model_name, f"Unexpected model id retrieved. Expected: {model_name}. Actual: {outputs.id}"
        assert outputs.object == "model", \
            f"Unexpected model object type retrieved. Expected: model. Actual: {outputs.object}"
        assert isinstance(outputs.created, int), \
            f"Wrong format for created parameter. Expected type int: Actual output: {outputs.created}"
        assert outputs.owned_by == "OVMS", f"Wrong model name retrieved. Expected: OVMS. Actual: {outputs.owned_by}"
        return outputs.id

    @staticmethod
    def _analyze_audio(file_path):
        """Read audio file and compute quality metrics using soundfile + numpy.

        Supports WAV, FLAC, OGG natively. MP3 support requires libmpg123 on the system.

        Returns:
            dict with keys: duration_sec, sample_rate, rms, spectral_flatness
        """
        data, sample_rate = sf.read(file_path, dtype="float32")
        # Convert stereo to mono if needed
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        n_samples = len(data)
        duration_sec = n_samples / sample_rate

        # RMS energy — silence detector
        rms = float(np.sqrt(np.mean(data ** 2)))

        # Spectral flatness — noise vs speech detector
        # Wiener entropy: geometric_mean(|FFT|) / arithmetic_mean(|FFT|)
        # White noise → ~1.0, speech → ~0.05-0.4
        magnitude = np.abs(np.fft.rfft(data))
        magnitude = magnitude[magnitude > 0]  # avoid log(0)
        if len(magnitude) > 0:
            log_mean = np.mean(np.log(magnitude))
            spectral_flatness = float(np.exp(log_mean) / np.mean(magnitude))
        else:
            spectral_flatness = 0.0

        metrics = {
            "duration_sec": round(duration_sec, 3),
            "sample_rate": sample_rate,
            "rms": round(rms, 6),
            "spectral_flatness": round(spectral_flatness, 4),
        }
        logger.info(f"Audio metrics for {os.path.basename(file_path)}: {metrics}")
        return metrics

    @classmethod
    def validate_audio_speech_outputs(
            cls, speech_file_path, allow_empty_response=False,
            min_duration_sec=5, max_spectral_flatness=0.85,
    ):
        """Validate speech audio output file.

        Checks:
        - File exists and is not empty
        - Audio duration >= min_duration_sec
        - Audio is not silence (RMS > 0)
        - Audio is not pure noise (spectral_flatness < max_spectral_flatness;
          speech typically has flatness 0.05-0.4, white noise ~1.0)

        Args:
            speech_file_path: Path to the generated audio file (WAV, MP3, etc.)
            allow_empty_response: If True, skip content checks.
            min_duration_sec: Minimum expected audio duration in seconds.
            max_spectral_flatness: Maximum spectral flatness (above = likely noise, not speech).
        """
        assert os.path.exists(speech_file_path), f"Speech output file not found: {speech_file_path}"
        file_size = os.path.getsize(speech_file_path)
        if not allow_empty_response:
            assert file_size > 0, f"Speech output file is empty: {speech_file_path}"
        logger.info(f"Audio speech output saved to {speech_file_path} ({file_size} bytes)")

        if not allow_empty_response:
            metrics = cls._analyze_audio(speech_file_path)

            assert metrics["duration_sec"] >= min_duration_sec, (
                f"Audio too short: {metrics['duration_sec']}s < {min_duration_sec}s minimum. "
                f"File: {speech_file_path}"
            )
            assert metrics["rms"] > 0, (
                f"Audio is silent (RMS=0). File: {speech_file_path}"
            )
            assert metrics["spectral_flatness"] < max_spectral_flatness, (
                f"Audio appears to be noise, not speech "
                f"(spectral_flatness={metrics['spectral_flatness']} >= {max_spectral_flatness}). "
                f"File: {speech_file_path}"
            )

        return speech_file_path

    @staticmethod
    def validate_audio_asr_outputs(outputs, allow_empty_response=False):
        assert outputs is not None, "Audio ASR output is None"
        if not allow_empty_response:
            assert len(outputs.strip()) > 0, f"Audio ASR output is empty: '{outputs}'"
            word_count = len(outputs.strip().split())
            assert word_count >= 1, f"Audio ASR output has no words: '{outputs}'"
        logger.info(f"Audio ASR output ({len(outputs.split())} words): {outputs}")
        return outputs

    @staticmethod
    def validate_wer(reference, hypothesis, threshold=0.4):
        """Calculate Word Error Rate (WER) and assert it is below the threshold."""
        normalize_text = Compose([
            RemovePunctuation(),
            wer_standardize
        ])
        error_rate = wer(reference, hypothesis, reference_transform=normalize_text, hypothesis_transform=normalize_text)
        assert error_rate < threshold, (
            f"Output WER is too high. "
            f"threshold={threshold}, error_rate={error_rate:.4f}. "
            f"Reference: '{str(reference)[:100]}'. "
            f"Hypothesis: '{str(hypothesis)[:100]}'."
        )

    @staticmethod
    def compute_cosine_similarity(embedding_a, embedding_b):
        a = np.array(embedding_a)
        b = np.array(embedding_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    @classmethod
    def create_embeddings_getter(  # pylint: disable=import-outside-toplevel
            cls, embeddings_model, api_type, port, request_parameters=None, inference_fn=None):
        """Create a callable that returns an embedding vector for a given text string.

        Uses an OVMS-hosted embeddings model to compute embeddings. The returned callable
        can be passed to validate_text_similarity as the embeddings_getter parameter.

        Args:
            embeddings_model: Embeddings model instance (e.g. AlibabaNLPGteLargeEnv15).
            api_type: OpenAI REST API type.
            port: OVMS port where the embeddings model is served.
            request_parameters: Optional pre-built request parameters for embeddings endpoint.
                If None, will be built automatically via LLMUtils.prepare_request_params.
            inference_fn: Callable to run LLM inference (e.g. run_llm_inference).
                Injected to avoid circular import between this module and inference_helpers.

        Returns:
            Callable[[str], list[float]] that takes text and returns its embedding vector.
        """
        assert inference_fn is not None, (
            "inference_fn is required"
        )

        if request_parameters is None:
            from llm.utils import LLMUtils
            request_parameters = LLMUtils.prepare_request_params(OpenAIWrapper.EMBEDDINGS)

        def getter(text):
            class TextDataset(FeatureExtractionModelDataset):
                input_data = [text]

            _, raw_outputs, _, _ = inference_fn(
                embeddings_model, api_type, port,
                OpenAIWrapper.EMBEDDINGS,
                dataset=TextDataset,
                input_data_type="list",
                request_parameters=request_parameters,
            )
            return raw_outputs.data[0].embedding

        return getter

    @classmethod
    def validate_text_similarity(
            cls,
            reference_text,
            hypothesis_texts,
            embeddings_getter,
            cos_sim_threshold=0.7,
    ):
        if isinstance(hypothesis_texts, str):
            hypothesis_texts = [hypothesis_texts]

        step(f"Validate text similarity (threshold={cos_sim_threshold})")
        reference_embedding = embeddings_getter(reference_text)

        cos_sim_errors = []
        for text in hypothesis_texts:
            hypothesis_embedding = embeddings_getter(text)
            cos_sim = cls.compute_cosine_similarity(reference_embedding, hypothesis_embedding)
            logger.info(f"Text similarity cos_sim={cos_sim:.4f} for: '{text[:80]}...'")
            if cos_sim < cos_sim_threshold:
                cos_sim_errors.append({"cos_sim": round(cos_sim, 4), "text": text})

        assert not cos_sim_errors, (
            f"Text similarity below threshold ({cos_sim_threshold}). "
            f"Reference: '{reference_text[:100]}'. "
            f"Failed comparisons: {cos_sim_errors}"
        )
