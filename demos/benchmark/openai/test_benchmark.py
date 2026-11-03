"""Tests for synthetic VLM requests, turn history, and usage statistics."""
"""python -m unittest -v test_benchmark"""

import asyncio
import base64
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aiohttp import web

import benchmark


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, tokens, **kwargs):
        return ' '.join(tokens)


class VLMInputTests(unittest.TestCase):
    def test_requested_and_corrected_aliases(self):
        for output_flag, turn_flag in (('--output_lenght', '--turn_lenght'),
                                       ('--output_length', '--turn_length')):
            options = benchmark.parser.parse_args(['--backend', 'ovms-vlm', output_flag, '7',
                                                    turn_flag, '3', '--max_concurrency', '2'])
            self.assertEqual(options.output_length, 7)
            self.assertEqual(options.turn_length, 3)
            self.assertEqual(options.max_concurrency, 2)

    def test_positive_limits(self):
        for flag in ('--input_length', '--output_lenght', '--max_concurrency', '--turns', '--turn_lenght'):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark.parser.parse_args([flag, '0'])

    def test_exact_synthetic_text_lengths(self):
        tokenizer = WordTokenizer()
        for length in (1, 16, 128):
            self.assertEqual(len(tokenizer.encode(benchmark.synthetic_token_text(tokenizer, length))), length)

    def test_image_url_and_local_file(self):
        url = 'https://example.org/image.jpeg'
        self.assertEqual(benchmark.image_context_url(url), url)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            payload = b'png image bytes'
            path.write_bytes(payload)
            encoded = benchmark.image_context_url(str(path))
            self.assertTrue(encoded.startswith('data:image/png;base64,'))
            self.assertEqual(base64.b64decode(encoded.split(',', 1)[1]), payload)
            self.assertEqual(benchmark.image_context_url(encoded), encoded)

    def test_prepare_vlm_does_not_load_hf_dataset(self):
        benchmark.args = vars(benchmark.parser.parse_args(['--backend', 'ovms-vlm', '--limit', '3',
                                                           '--input_length', '10', '--turn_lenght', '2',
                                                           '--tokenizer', 'local-tokenizer']))
        with patch.object(benchmark.AutoTokenizer, 'from_pretrained', return_value=WordTokenizer()), \
                patch.object(benchmark, 'load_dataset') as load_dataset:
            docs = benchmark.prepare_documents()
        load_dataset.assert_not_called()
        self.assertEqual(len(docs), 3)
        self.assertEqual(len(docs[0]['text'].split()), 10)
        self.assertEqual(len(set(docs['text'])), 3)
        self.assertEqual(len(benchmark.args['turn_text'].split()), 2)

    def test_default_vlm_uses_served_tokenizer(self):
        benchmark.args = vars(benchmark.parser.parse_args(['--backend', 'ovms-vlm', '--model', 'served-alias',
                                                           '--input_length', '10', '--turn_lenght', '2', '--limit', '3']))

        def tokenize(text, model):
            self.assertEqual(model, 'served-alias')
            return [list(range(len(text.split())))]

        with patch.object(benchmark, 'tokenize_texts', side_effect=tokenize), \
                patch.object(benchmark.AutoTokenizer, 'from_pretrained') as local:
            docs = benchmark.prepare_documents()
        local.assert_not_called()
        self.assertEqual(len(docs[0]['text'].split()), 10)
        self.assertEqual(len(set(docs['text'])), 3)
        self.assertEqual(len(benchmark.args['turn_text'].split()), 2)

    def test_embedding_counts_use_batched_server_tokenization(self):
        benchmark.args = {'backend': 'ovms-embeddings', 'tokenizer': None}
        docs = benchmark.Dataset.from_dict({'text': ['one', 'two words', 'three words here']})
        with patch.object(benchmark, 'tokenize_texts', return_value=[[1], [1, 2], [1, 2, 3]]) as tokenize, \
                patch.object(benchmark.AutoTokenizer, 'from_pretrained') as local:
            count = benchmark.count_tokens(docs, 'embedding-alias')
        self.assertEqual(count, 6)
        tokenize.assert_called_once_with(['one', 'two words', 'three words here'], 'embedding-alias')
        local.assert_not_called()

    def test_input_text_prefix_is_within_token_limit(self):
        source = 'one\n  two three four'
        prefix = benchmark.truncate_input_text(source, 2, lambda value: len(value.split()))
        self.assertTrue(source.startswith(prefix))
        self.assertEqual(prefix.split(), ['one', 'two'])
        self.assertIn('\n  ', prefix)

    def test_short_and_empty_input_text_are_not_padded(self):
        self.assertEqual(benchmark.truncate_input_text('one two', 10, lambda value: len(value.split())), 'one two')
        self.assertEqual(benchmark.truncate_input_text('', 10, lambda value: len(value.split())), '')

    def test_text_file_uses_server_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prompt.txt'
            source = 'one two\nthree four five'
            path.write_text(source, encoding='utf-8')
            benchmark.args = vars(benchmark.parser.parse_args(['--backend', 'ovms-vlm', '--model', 'served-alias',
                                                               '--text_input_file', str(path), '--input_length', '3',
                                                               '--turn_lenght', '2', '--limit', '2']))
            with patch.object(benchmark, 'tokenize_texts', side_effect=lambda text, model:
                              [list(range(len(text.split())))]), \
                    patch.object(benchmark.AutoTokenizer, 'from_pretrained') as local:
                docs = benchmark.prepare_documents()
        local.assert_not_called()
        text = docs[0]['text']
        self.assertTrue(source.startswith(text.partition(' ')[2]))
        self.assertEqual(len(text.split()), 3)
        self.assertNotEqual(docs[1]['text'], text)
        self.assertEqual(len(benchmark.args['turn_text'].split()), 2)

    def test_text_file_uses_local_tokenizer_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prompt.txt'
            path.write_text('one two three four', encoding='utf-8')
            benchmark.args = vars(benchmark.parser.parse_args(['--backend', 'ovms-vlm', '--tokenizer', 'local',
                                                               '--text_input_file', str(path), '--input_length', '2',
                                                               '--turn_lenght', '2', '--limit', '1']))
            with patch.object(benchmark.AutoTokenizer, 'from_pretrained', return_value=WordTokenizer()), \
                    patch.object(benchmark, 'tokenize_texts') as remote:
                docs = benchmark.prepare_documents()
        remote.assert_not_called()
        self.assertEqual(docs[0]['text'].split()[1:], ['one'])

    def test_unique_initial_text_respects_token_budget(self):
        used = set()
        texts = [benchmark.unique_initial_text('one two three', 3, lambda value: len(value.split()), used)
                 for index in range(10)]
        self.assertEqual(len(set(texts)), 10)
        self.assertTrue(all(len(text.split()) == 3 for text in texts))

    def test_duplicate_prompt_generation_fails_instead_of_reusing(self):
        with patch.object(benchmark.secrets, 'token_hex', return_value='fixed'):
            with self.assertRaisesRegex(ValueError, 'increase --input_length'):
                benchmark.unique_initial_text('text', 2, lambda value: len(value.split()), {'fixed text'})

    def test_missing_text_file_reports_setup_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as status:
                benchmark.main(['--backend', 'ovms-vlm', '--model', 'test',
                                '--text_input_file', str(Path(directory) / 'missing.txt')])
        self.assertEqual(status.exception.code, 2)
        self.assertIn('missing.txt', errors.getvalue())

    def test_text_file_rejected_for_other_backends(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as status:
            benchmark.main(['--text_input_file', 'prompt.txt'])
        self.assertEqual(status.exception.code, 2)

    def test_tokenize_contract_and_url(self):
        benchmark.args = {'api_url': 'http://localhost:9000/v3/embeddings', 'tokenize_url': None}
        response = io.StringIO('{"tokens":[[1,2],[3]]}')
        with patch.object(benchmark, 'urlopen', return_value=response) as post:
            self.assertEqual(benchmark.tokenize_texts(['one two', 'three'], 'served'), [[1, 2], [3]])
        request = post.call_args.args[0]
        self.assertEqual(request.full_url, 'http://localhost:9000/v1/tokenize')
        self.assertEqual(benchmark.json.loads(request.data),
                         {'model': 'served', 'text': ['one two', 'three'], 'add_special_tokens': False})

    def test_single_model_discovery(self):
        with patch.object(benchmark, 'urlopen', return_value=io.StringIO('{"data":[{"id":"served-alias"}]}')) as get:
            model = benchmark.discover_model('http://localhost:9000/v3/embeddings')
        self.assertEqual(model, 'served-alias')
        request = get.call_args.args[0]
        self.assertEqual(request.full_url, 'http://localhost:9000/v1/models')
        self.assertEqual(request.get_method(), 'GET')

    def test_multiple_models_require_explicit_model(self):
        response = io.StringIO('{"data":[{"id":"first"},{"id":"second"}]}')
        with patch.object(benchmark, 'urlopen', return_value=response):
            with self.assertRaisesRegex(ValueError, '--model is required'):
                benchmark.discover_model('http://localhost:9000/v1/chat/completions')

    def test_empty_or_invalid_models_response(self):
        for data in ('{"data":[]}', '{}', '[]', '{"data":[{}]}', '{"data":[{"id":42}]}'):
            with self.subTest(data=data), patch.object(benchmark, 'urlopen', return_value=io.StringIO(data)), \
                    self.assertRaisesRegex(ValueError, '--model'):
                benchmark.discover_model('http://localhost:9000/v1/chat/completions')

    def test_explicit_model_skips_discovery(self):
        docs = benchmark.Dataset.from_dict({'text': ['hello']})
        results = {'duration': 1.0, 'successes': [True], 'latencies': [0.1],
                   'input_tokens': [1], 'output_tokens': [1], 'cached_tokens': [0], 'errors': ['']}
        with patch.object(benchmark, 'discover_model') as discover, \
                patch.object(benchmark, 'prepare_documents', return_value=docs), \
                patch.object(benchmark, 'benchmark', new=lambda **kwargs: kwargs), \
                patch.object(benchmark.asyncio, 'run', return_value=results), redirect_stdout(io.StringIO()):
            status = benchmark.main(['--backend', 'ovms-vlm', '--model', 'explicit-model'])
        discover.assert_not_called()
        self.assertEqual(status, 0)
        self.assertEqual(benchmark.args['model'], 'explicit-model')

    def test_tokenize_override_and_invalid_response(self):
        benchmark.args = {'api_url': 'http://localhost:9000/v1/chat/completions',
                          'tokenize_url': 'http://localhost:9000/proxy/v1/tokenize'}
        with patch.object(benchmark, 'urlopen', return_value=io.StringIO('{"tokens":[1,2]}')) as post:
            self.assertEqual(benchmark.tokenize_texts('text', 'served'), [[1, 2]])
        self.assertEqual(post.call_args.args[0].full_url, benchmark.args['tokenize_url'])
        for data in ('{"tokens":[true]}', '{}', '{"tokens":[[1],[2]]}', '[]'):
            with self.subTest(data=data), patch.object(benchmark, 'urlopen', return_value=io.StringIO(data)), \
                    self.assertRaises(ValueError):
                benchmark.tokenize_texts('text', 'served')

    def test_usage_validation(self):
        data = {'usage': {'prompt_tokens': 10, 'completion_tokens': 4}}
        self.assertEqual(benchmark.read_vlm_usage(data), (10, 4, 0))
        for usage in ({'prompt_tokens': 10, 'completion_tokens': 4, 'prompt_tokens_details': {'cached_tokens': 11}},
                      {'prompt_tokens': -1, 'completion_tokens': 4},
                      {'prompt_tokens': True, 'completion_tokens': 4}, {}):
            with self.subTest(usage=usage), self.assertRaises((ValueError, KeyError)):
                benchmark.read_vlm_usage({'usage': usage})

    def test_usage_statistics_exclude_failed_turns(self):
        benchmark.args = {'limit': 2, 'turns': 2}
        results = {'duration': 2.0, 'successes': [True, False, True], 'latencies': [1.0, 0.0, 3.0],
                   'input_tokens': [10, 1000, 20], 'output_tokens': [4, 1000, 4],
                   'cached_tokens': [2, 1000, 5], 'errors': ['', 'failed', '']}
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            benchmark.print_vlm_statistics(results)
        text = output.getvalue()
        self.assertIn('Input tokens (including cached): 30', text)
        self.assertIn('Output tokens: 8', text)
        self.assertIn('Cached input tokens: 7', text)
        self.assertIn('Uncached input tokens: 23', text)
        self.assertIn('Mean latency: 2000.000 ms', text)
        self.assertIn('Output tokens per second: 4.000', text)

    def test_stream_timing_statistics_exclude_failed_requests(self):
        benchmark.args = {'limit': 3, 'turns': 1}
        results = {'duration': 1.0, 'successes': [True, False, True], 'latencies': [0.2, 0.0, 0.4],
                   'input_tokens': [10, 0, 20], 'output_tokens': [4, 0, 4],
                   'cached_tokens': [0, 0, 5], 'errors': ['', 'failed', ''],
                   'ttfts': [0.1, 100, 0.3], 'tpots': [0.01, 100, 0.03],
                   'inter_chunk_latencies': [[0.01, 0.03], [100], [0.02]]}
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            benchmark.print_vlm_statistics(results)
        self.assertIn('Mean TTFT: 200.000 ms', output.getvalue())
        self.assertIn('Median TTFT: 200.000 ms', output.getvalue())
        self.assertIn('Mean TPOT: 20.000 ms', output.getvalue())
        self.assertIn('Median TPOT: 20.000 ms', output.getvalue())
        self.assertIn('Median inter-chunk latency: 20.000 ms', output.getvalue())

    def test_vlm_main_defaults_and_failure_exit_code(self):
        results = {'duration': 1.0, 'successes': [False], 'latencies': [0.1],
                   'input_tokens': [0], 'output_tokens': [0], 'cached_tokens': [0], 'errors': ['failed']}
        docs = benchmark.Dataset.from_dict({'text': ['hello']})
        with patch.object(benchmark, 'discover_model', return_value='discovered-model') as discover, \
            patch.object(benchmark, 'prepare_documents', return_value=docs), \
                patch.object(benchmark, 'benchmark', new=lambda **kwargs: kwargs) as unused, \
                patch.object(benchmark.asyncio, 'run', return_value=results) as run, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = benchmark.main(['--backend', 'ovms-vlm', '--limit', '1'])
        self.assertEqual(status, 1)
        self.assertEqual(benchmark.batch_size, 1)
        self.assertEqual(benchmark.args['api_url'], 'http://localhost:8000/v1/chat/completions')
        self.assertEqual(benchmark.args['model'], 'discovered-model')
        discover.assert_called_once_with(benchmark.args['api_url'])
        self.assertIs(run.call_args.args[0]['backend_function'], benchmark.async_request_vlm)

    def test_existing_embedding_default_batch_size(self):
        docs = benchmark.Dataset.from_dict({'text': ['hello']})
        results = {'duration': 1.0, 'successes': [True], 'latencies': [0.1]}
        with patch.object(benchmark, 'discover_model', return_value='embedding-model'), \
            patch.object(benchmark, 'prepare_documents', return_value=docs), \
                patch.object(benchmark, 'benchmark', new=lambda **kwargs: kwargs), \
                patch.object(benchmark.asyncio, 'run', return_value=results), \
                patch.object(benchmark, 'count_tokens', return_value=1), redirect_stdout(io.StringIO()):
            status = benchmark.main([])
        self.assertEqual(status, 0)
        self.assertEqual(benchmark.batch_size, 16)


class VLMRequestTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        benchmark.args = vars(benchmark.parser.parse_args(['--backend', 'ovms-vlm', '--limit', '4',
                                                           '--turns', '3', '--max_concurrency', '2',
                                                           '--output_lenght', '4']))
        benchmark.args.update(image_url='https://example.org/image.jpeg', turn_text='next next', batch_size=1)
        benchmark.batch_size = 1
        self.payloads = []
        self.active = 0
        self.maximum = 0
        self.overlap = asyncio.Event()
        self.require_overlap = False
        self.fail_turn = None
        self.missing_usage = False
        self.missing_done = False
        app = web.Application()
        app.router.add_post('/v1/chat/completions', self.chat)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        port = self.runner.addresses[0][1]
        self.url = f'http://127.0.0.1:{port}/v1/chat/completions'

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def chat(self, request):
        payload = await request.json()
        self.payloads.append(payload)
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            if self.require_overlap:
                if self.active == 2:
                    self.overlap.set()
                await asyncio.wait_for(self.overlap.wait(), timeout=3)
            turn = (len(payload['messages']) + 1) // 2
            if self.fail_turn == turn:
                return web.json_response({'error': 'failed turn'}, status=500)
            chunks = [
                {'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}}]},
                {'choices': [{'index': 0, 'delta': {'content': 'reply '}}]},
                {'choices': [{'index': 0, 'delta': {'content': str(turn)}}]},
                {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}]},
            ]
            if not self.missing_usage:
                chunks.append({'choices': [], 'usage': {'prompt_tokens': 10 * turn, 'completion_tokens': 4,
                                                       'prompt_tokens_details': {'cached_tokens': 5 * (turn - 1)}}})
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            for chunk in chunks:
                event = f'data: {benchmark.json.dumps(chunk)}\r\n\r\n'.encode()
                await response.write(event[:7])
                await response.write(event[7:])
            if not self.missing_done:
                await response.write(b'data: [DONE]\n\n')
            await response.write_eof()
            return response
        finally:
            self.active -= 1

    async def test_multiturn_context_and_generation_parameters(self):
        request = benchmark.RequestFuncInput(self.url, ['initial prompt'], 'test-model')
        outputs = await benchmark.async_request_vlm(request)
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(output.success for output in outputs))
        self.assertEqual([len(payload['messages']) for payload in self.payloads], [1, 3, 5])
        for payload in self.payloads:
            self.assertEqual(payload['model'], 'test-model')
            self.assertEqual(payload['max_completion_tokens'], 4)
            self.assertIs(payload['ignore_eos'], True)
            self.assertIs(payload['stream'], True)
            self.assertEqual(payload['stream_options'], {'include_usage': True})
            self.assertEqual(payload['messages'][0]['content'][1]['image_url']['url'], benchmark.args['image_url'])
        self.assertEqual(self.payloads[2]['messages'][1], {'role': 'assistant', 'content': 'reply 1'})
        self.assertEqual(self.payloads[2]['messages'][2], {'role': 'user', 'content': 'next next'})
        self.assertEqual(self.payloads[2]['messages'][3], {'role': 'assistant', 'content': 'reply 2'})
        self.assertEqual([output.cached_tokens for output in outputs], [0, 5, 10])
        self.assertTrue(all(output.ttft is not None for output in outputs))
        self.assertTrue(all(output.tpot is not None for output in outputs))

    async def test_concurrency_and_all_turns_counted(self):
        self.require_overlap = True
        docs = benchmark.Dataset.from_dict({'text': ['initial'] * 4})
        results = await benchmark.benchmark(docs, 'test-model', self.url, float('inf'), benchmark.async_request_vlm)
        self.assertEqual(self.maximum, 2)
        self.assertEqual(len(results['successes']), 12)
        self.assertTrue(all(results['successes']))
        self.assertEqual(sum(results['input_tokens']), 240)
        self.assertEqual(sum(results['output_tokens']), 48)
        self.assertEqual(sum(results['cached_tokens']), 60)

    async def test_failed_turn_stops_conversation(self):
        self.fail_turn = 2
        outputs = await benchmark.async_request_vlm(benchmark.RequestFuncInput(self.url, ['initial'], 'test'))
        self.assertEqual([output.success for output in outputs], [True, False])
        self.assertIn('HTTP 500', outputs[1].error)
        self.assertEqual(len(self.payloads), 2)

    async def test_missing_usage_is_not_silently_counted_as_zero(self):
        self.missing_usage = True
        outputs = await benchmark.async_request_vlm(benchmark.RequestFuncInput(self.url, ['initial'], 'test'))
        self.assertEqual(len(outputs), 1)
        self.assertFalse(outputs[0].success)
        self.assertIn('usage', outputs[0].error)

    async def test_incomplete_stream_is_a_failure(self):
        self.missing_done = True
        outputs = await benchmark.async_request_vlm(benchmark.RequestFuncInput(self.url, ['initial'], 'test'))
        self.assertFalse(outputs[0].success)
        self.assertIn('missing [DONE]', outputs[0].error)

    async def test_first_and_subsequent_token_timing(self):
        events = [
            {'choices': [{'delta': {'role': 'assistant'}}]},
            {'choices': [{'delta': {'content': 'first'}}]},
            {'choices': [{'delta': {'content': ' second'}}]},
            {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 3}},
        ]

        async def lines():
            yield b': keepalive\n'
            for event in events:
                yield f'data: {benchmark.json.dumps(event)}\n'.encode()
                yield b'\n'
            yield b'data: [DONE]\n'
            yield b'\n'

        output = benchmark.RequestFuncOutput()
        with patch.object(benchmark.time, 'perf_counter', side_effect=[10.1, 10.4, 10.8, 11.0, 11.2]):
            await benchmark.read_vlm_stream(SimpleNamespace(content=lines()), output, 10.0)
        self.assertAlmostEqual(output.ttft, 0.4)
        self.assertAlmostEqual(output.tpot, 0.4 / 3)
        self.assertAlmostEqual(output.inter_chunk_latencies[0], 0.4)
        self.assertEqual(output.text, 'first second')

    async def test_single_chunk_has_zero_first_to_last_duration(self):
        async def lines():
            data = {'choices': [{'delta': {'content': 'one'}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 1}}
            yield f'data: {benchmark.json.dumps(data)}\n'.encode()
            yield b'\n'
            yield b'data: [DONE]\n\n'

        output = benchmark.RequestFuncOutput()
        await benchmark.read_vlm_stream(SimpleNamespace(content=lines()), output, 0.0)
        self.assertIsNotNone(output.ttft)
        self.assertEqual(output.tpot, 0.0)

    async def test_zero_output_tokens_has_no_tpot(self):
        async def lines():
            data = {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 0}}
            yield f'data: {benchmark.json.dumps(data)}\n'.encode()
            yield b'\n'
            yield b'data: [DONE]\n\n'

        output = benchmark.RequestFuncOutput()
        await benchmark.read_vlm_stream(SimpleNamespace(content=lines()), output, 0.0)
        self.assertIsNone(output.ttft)
        self.assertIsNone(output.tpot)

    async def test_existing_embedding_backend_result_shape(self):
        benchmark.args.update(backend='ovms-embeddings')
        benchmark.batch_size = 2
        docs = benchmark.Dataset.from_dict({'text': ['one', 'two', 'three']})

        async def embeddings(request_func_input, pbar):
            pbar.update(1)
            return benchmark.RequestFuncOutput(success=True, latency=0.01)

        results = await benchmark.benchmark(docs, 'test', self.url, float('inf'), embeddings)
        self.assertEqual(results['successes'], [True, True])
        self.assertEqual(results['latencies'], [0.01, 0.01])


if __name__ == '__main__':
    unittest.main()