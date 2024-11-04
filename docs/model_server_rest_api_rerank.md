# Cohere API rerank endpoint {#ovms_docs_rest_api_rerank}

## API Reference
OpenVINO Model Server includes now the `rerank` endpoint based on Cohere API.
Please see the [Cohere rerank API Reference](https://docs.cohere.com/reference/rerank) for more information on the API.
The endpoint can be accessed via a path:

<b>http://server_name:port/v3/rerank</b>

In the cohere client library specify the `base_url=http://server_name:port/v3`.

### Example request

```
curl http://localhost:8000/v3/rerank \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-reranker-large", \
  "query": "Hello", "documents":["Welcome","Farewell"]}'
```

### Example response

```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.3886180520057678
    },
    {
      "index": 1,
      "relevance_score": 0.0055549247190356255
    }
  ]
}
```


### Request

| Param | OpenVINO Model Server | Cohere | Type | Description |
|-----|----------|----------|---------|-----|
| model | ✅ | ✅ | string (required) | Name of the model to use. Name assigned to a servable configured to run a reranking model.  |
| query | ✅ | ✅ | string (required) | Text to compare the similarity with the documents |
| documents | ⚠️ | ✅ | list of strings  | Documents to rerank |
| top_n | ✅ | ✅ | integer  | Limit response to n most similar |
| return_documents | ✅ | ✅ | boolean  | Return the documents in the response |

#### Unsupported params from Cohere service:
- max_chunks_per_doc - configurable on the server side in the graph parameters.
- rank_fields
- documents in a form of a dictionary. A list of strings is allowed.

## Response

| Param | OpenVINO Model Server | Cohere | Type | Description |
|-----|----------|----------|---------|-----|
| results.index | ✅ | ✅ | integer | index of the document |
| results.relevance_score | ✅ | ✅ | float in a range 0-1 | relevance score with the query  |
| results.document | ✅ | ✅ | string | Assessed document |
| id | ❌ | ✅ | integer | Response index |
| metadata | ❌ | ✅ | string |  Model name

#### Unsupported params from OpenAI service:

- id
- metadata

## Error handling
Endpoint can raise an error related to incorrect request in the following conditions:
- incorrect format of any of the fields based on the schema
- Amount of documents exceeds allowed configured value - default 100
- Number of chunks needed to split any of the input documents exceed the configured value - default 10


## References

[End to end demo with rerank endpoint](../demos/rerank/README.md)

[Code snippets](./clients_genai.md)

