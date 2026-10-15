# OVMS Task System: Comprehensive Guide

## 1. Task Enum Definitions

### GraphExportType Enum
**File:** [src/capi_frontend/server_settings.hpp](src/capi_frontend/server_settings.hpp#L26-L32)

```cpp
enum GraphExportType : unsigned int {
    TEXT_GENERATION_GRAPH,        // LLM inference (text_generation)
    RERANK_GRAPH,                 // Ranking/reranking models (rerank)
    EMBEDDINGS_GRAPH,             // Embeddings generation (embeddings)
    IMAGE_GENERATION_GRAPH,       // Image generation (image_generation)
    TEXT_TO_SPEECH_GRAPH,         // TTS models (text2speech)
    SPEECH_TO_TEXT_GRAPH,         // STT/ASR models (speech2text)
    UNKNOWN_GRAPH                 // Unknown/error state
};
```

### String Mappings
**File:** [src/capi_frontend/server_settings.hpp](src/capi_frontend/server_settings.hpp#L42-L50)

```cpp
const std::map<GraphExportType, std::string> typeToString = {
    {TEXT_GENERATION_GRAPH, "text_generation"},
    {RERANK_GRAPH, "rerank"},
    {EMBEDDINGS_GRAPH, "embeddings"},
    {IMAGE_GENERATION_GRAPH, "image_generation"},
    {TEXT_TO_SPEECH_GRAPH, "text2speech"},
    {SPEECH_TO_TEXT_GRAPH, "speech2text"},
    {UNKNOWN_GRAPH, "unknown_graph"}
};

const std::map<std::string, GraphExportType> stringToType = {
    {"text_generation", TEXT_GENERATION_GRAPH},
    {"rerank", RERANK_GRAPH},
    {"embeddings", EMBEDDINGS_GRAPH},
    {"image_generation", IMAGE_GENERATION_GRAPH},
    {"text2speech", TEXT_TO_SPEECH_GRAPH},
    {"speech2text", SPEECH_TO_TEXT_GRAPH},
    {"unknown_graph", UNKNOWN_GRAPH}
};
```

### String-to-Enum Conversion
**File:** [src/capi_frontend/server_settings.cpp](src/capi_frontend/server_settings.cpp#L38-L40)

```cpp
GraphExportType stringToEnum(const std::string& inString) {
    auto it = stringToType.find(inString);
    return (it != stringToType.end()) ? it->second : UNKNOWN_GRAPH;
}
```

---

## 2. Task Detection/Inference Logic

### 2.1 Detection Flow

When `--task` is **not provided**, OVMS attempts automatic detection:

**File:** [src/cli_parser.cpp](src/cli_parser.cpp#L410-L440)

```cpp
// Task inference decision logic:
bool shouldInferTask = false;
if (sourceModel.has_value() && !sourceModel->empty()) {
    // Always infer task when pulling from HuggingFace
    shouldInferTask = true;
} else if (isConfigureMode(this->result) && modelPath.has_value() && !modelPath->empty()) {
    // Configure mode always infers task from model when --task not provided
    shouldInferTask = true;
} else if (modelPath.has_value() && !modelPath->empty()) {
    // For local model_path, infer task if:
    // 1. Unmatched options (task-specific parameters) are present, OR
    // 2. graph.pbtxt doesn't exist (need to create in-memory graph)
    bool hasUnmatchedOptions = ::ovms::hasTaskSpecificParameters(result->unmatched());
    bool graphExists = ::ovms::graphPbtxtExists(*modelPath);
    shouldInferTask = hasUnmatchedOptions || !graphExists;
}

if (shouldInferTask) {
    inferredTaskParameter = determineDefaultTaskParameter(modelPath, sourceModel, modelRepositoryPath);
}
```

### 2.2 Detector Classes & Architecture

**File:** [src/default_task_detector.hpp](src/default_task_detector.hpp#L55-L100)

Abstract detector interface:
```cpp
class BaseTaskDetector {
public:
    virtual bool scan(const ModelCatalogContext& ctx) const = 0;
    virtual std::string getName() const = 0;
    virtual ~BaseTaskDetector() = default;
};
```

Concrete detectors:
- `Speech2TextDetector` — detects speech-to-text models
- `Text2SpeechDetector` — detects text-to-speech models
- `RerankDetector` — detects reranking models
- `ImageGenerationDetector` — detects image generation models
- `EmbeddingsDetector` — detects embedding models
- `TextGenerationDetector` — detects LLM/text generation models

### 2.3 Detection Priority Order

**File:** [src/default_task_detector.cpp](src/default_task_detector.cpp#L323-L330)

```cpp
DefaultTaskDetector::DefaultTaskDetector() {
    // Priority order is critical for correctness:
    // - Speech2Text / Text2Speech before TextGen (share ForConditionalGeneration suffix)
    // - ImageGen before Embeddings (CLIPTextModel / UNet2DConditionModel end with "Model")
    // - Embeddings before TextGen (general "Model" suffix check)
    // - TextGen last (catch-all for ForCausalLM / ForConditionalGeneration)
    detectors.push_back(std::make_unique<Speech2TextDetector>());
    detectors.push_back(std::make_unique<Text2SpeechDetector>());
    detectors.push_back(std::make_unique<RerankDetector>());
    detectors.push_back(std::make_unique<ImageGenerationDetector>());
    detectors.push_back(std::make_unique<EmbeddingsDetector>());
    detectors.push_back(std::make_unique<TextGenerationDetector>());
}
```

### 2.4 Detection Methods by Task

**Speech2Text Detection**
**File:** [src/default_task_detector.cpp](src/default_task_detector.cpp#L136-L143)

```cpp
// Looks for architectures in config.json:
// - WhisperForConditionalGeneration
// - Qwen3ASRForConditionalGeneration
// - SeamlessM4T* (prefix match)
// Example: Kokoro → text2speech, Whisper → speech2text
```

**Text2Speech Detection**
**File:** [src/default_task_detector.cpp](src/default_task_detector.cpp#L153-L170)

```cpp
// Special case: null architectures with n_mels field (e.g. Kokoro TTS)
// Looks for architectures:
// - ParlerTTSForConditionalGeneration
// - SpeechT5ForTextToSpeech
// - KModel (Kokoro TTS)
// - *ForTextToSpeech (suffix match)
```

**Embeddings Detection**
**File:** [src/default_task_detector.cpp](src/default_task_detector.cpp#L250-L290)

```cpp
// Layer 1: modules.json (authoritative for sentence-transformers)
//   - Looks for "Pooling" module type
// Layer 2: Known embedding architectures in config.json
//   - BertModel, JinaBertModel, MPNetModel, Qwen2Model, RobertaModel, T5EncoderModel, XLMRobertaModel
// Layer 3: Keyword-based fallback (when modules.json absent)
//   - Model name contains "embed" + Qwen3ForCausalLM architecture
```

**Rerank Detection**
**File:** [src/default_task_detector.cpp](src/default_task_detector.cpp#L173-L207)

```cpp
// Layer 1: modules.json (authoritative for sentence-transformers cross-encoders)
//   - Looks for "LogitScore" or "CrossEncoder" module types
// Layer 2: Architecture suffix in config.json
//   - *ForSequenceClassification
// Layer 3: Keyword-based fallback (when modules.json absent)
//   - Model name contains "rerank" + Qwen3ForCausalLM architecture
```

**Image Generation Detection**
**File:** [src/default_task_detector.cpp](src/default_task_detector.cpp#L222-L241)

```cpp
// From config.json architectures:
// - CLIPTextModel, UNet2DConditionModel, AutoencoderKL
// - *Transformer2DModel (suffix match)
// From model_index.json (Diffusers pipelines):
// - _class_name contains "StableDiffusion" or "Flux"
```

**Text Generation Detection**
**File:** [src/default_task_detector.cpp](src/default_task_detector.cpp#L304-L312)

```cpp
// Catch-all detector (runs last after higher-priority detectors fail)
// Looks for architectures in config.json:
// - InternVLChatModel (exact match)
// - *ForCausalLM (suffix match)
// - *ForConditionalGeneration (suffix match)
```

**ModelCatalogContext**
**File:** [src/default_task_detector.hpp](src/default_task_detector.hpp#L32-L55)

Handles lazy-loaded JSON file access:
- Reads from local filesystem or in-memory preloaded content
- Caches parsed JSON documents
- Looks for: `config.json`, `modules.json`, `model_index.json`

---

## 3. Task-Specific CLI Parameter Parsing

When task is identified (explicit or inferred), task-specific CLI parsers are instantiated:

**File:** [src/cli_parser.cpp](src/cli_parser.cpp#L443-L485)

```cpp
task = stringToEnum(taskValue);
switch (hfSettings.task) {
    case TEXT_GENERATION_GRAPH: {
        GraphCLIParser cliParser;
        unmatchedOptions = cliParser.parse(result->unmatched());
        this->graphOptionsParser = std::move(cliParser);
        break;
    }
    case EMBEDDINGS_GRAPH: {
        EmbeddingsGraphCLIParser cliParser;
        unmatchedOptions = cliParser.parse(result->unmatched());
        this->graphOptionsParser = std::move(cliParser);
        break;
    }
    case RERANK_GRAPH: {
        RerankGraphCLIParser cliParser;
        unmatchedOptions = cliParser.parse(result->unmatched());
        this->graphOptionsParser = std::move(cliParser);
        break;
    }
    case IMAGE_GENERATION_GRAPH: {
        ImageGenerationGraphCLIParser cliParser;
        unmatchedOptions = cliParser.parse(result->unmatched());
        this->graphOptionsParser = std::move(cliParser);
        break;
    }
    case TEXT_TO_SPEECH_GRAPH: {
        TextToSpeechGraphCLIParser cliParser;
        unmatchedOptions = cliParser.parse(result->unmatched());
        this->graphOptionsParser = std::move(cliParser);
        break;
    }
    case SPEECH_TO_TEXT_GRAPH: {
        SpeechToTextGraphCLIParser cliParser;
        unmatchedOptions = cliParser.parse(result->unmatched());
        this->graphOptionsParser = std::move(cliParser);
        break;
    }
    // ...
}
```

### Task-Specific Parser Classes

**File:** [src/graph_export/graph_cli_parser.hpp](src/graph_export/graph_cli_parser.hpp#L31-L44)

```cpp
class GraphCLIParser {  // Text generation (base class)
public:
    std::vector<std::string> parse(const std::vector<std::string>& unmatchedOptions);
    void prepare(OvmsServerMode serverMode, HFSettingsImpl& hfSettings, const std::string& modelName);
    void printHelp();
    void createOptions();
};
```

**File:** [src/graph_export/embeddings_graph_cli_parser.hpp](src/graph_export/embeddings_graph_cli_parser.hpp#L33-L43)

```cpp
class EmbeddingsGraphCLIParser : public GraphCLIParser {  // Embeddings-specific
public:
    std::vector<std::string> parse(const std::vector<std::string>& unmatchedOptions);
    void prepare(OvmsServerMode serverMode, HFSettingsImpl& hfSettings, const std::string& modelName);
};
```

Other task parsers exist similarly:
- `RerankGraphCLIParser`
- `ImageGenerationGraphCLIParser`
- `TextToSpeechGraphCLIParser`
- `SpeechToTextGraphCLIParser`

---

## 4. Task-Specific Graph Generation

### Graph Export Flow

**File:** [src/cli_parser.cpp](src/cli_parser.cpp#L838-L930)

The `prepareGraph()` function routes based on task:

```cpp
void CLIParser::prepareGraph(ServerSettingsImpl& serverSettings, HFSettingsImpl& hfSettings, const std::string& modelName) {
    // ... (setup code)
    
    if (!taskValue.empty()) {
        hfSettings.task = stringToEnum(taskValue);
        switch (hfSettings.task) {
            case TEXT_GENERATION_GRAPH: {
                std::get<GraphCLIParser>(this->graphOptionsParser).prepare(serverSettings.serverMode, hfSettings, modelName);
                break;
            }
            case EMBEDDINGS_GRAPH: {
                std::get<EmbeddingsGraphCLIParser>(this->graphOptionsParser).prepare(serverSettings.serverMode, hfSettings, modelName);
                break;
            }
            // ... other cases
        }
    }
}
```

### Graph Template Creation

**File:** [src/graph_export/graph_export.cpp](src/graph_export/graph_export.cpp#L613-L625)

Each task generates a specific graph template:

```cpp
if (hfSettings.task == TEXT_GENERATION_GRAPH) {
    status = createTextGenerationGraphTemplate(directoryPath, hfSettings, pbtxtContent);
} else if (hfSettings.task == EMBEDDINGS_GRAPH) {
    status = createEmbeddingsGraphTemplate(directoryPath, hfSettings, pbtxtContent);
} else if (hfSettings.task == RERANK_GRAPH) {
    status = createRerankGraphTemplate(directoryPath, hfSettings, pbtxtContent);
} else if (hfSettings.task == IMAGE_GENERATION_GRAPH) {
    status = createImageGenerationGraphTemplate(directoryPath, hfSettings, pbtxtContent);
} else if (hfSettings.task == TEXT_TO_SPEECH_GRAPH) {
    status = createTextToSpeechGraphTemplate(directoryPath, hfSettings, pbtxtContent);
} else if (hfSettings.task == SPEECH_TO_TEXT_GRAPH) {
    status = createSpeechToTextGraphTemplate(directoryPath, hfSettings, pbtxtContent);
}
```

Each template:
- Defines task-specific MediaPipe calculator nodes
- Configures input/output tensor shapes and names
- Sets model loading parameters
- Registers task-specific plugin configurations

---

## 5. Endpoint Registration Based on Task

### HTTP Request Routing

**File:** [src/http_rest_api_handler.cpp](src/http_rest_api_handler.cpp#L581)

```cpp
bool isTextGenerationEndpoint = (uri.find("completions") != std::string_view::npos) || 
                                (uri.find("responses") != std::string::npos);
```

### Task Validation in Config

**File:** [src/config.cpp](src/config.cpp#L177-L196)

Task is validated based on source model:
```cpp
if (!this->serverSettings.hfSettings.sourceModel.empty() && 
    this->serverSettings.hfSettings.task == UNKNOWN_GRAPH) {
    throw std::logic_error("Could not infer model task - specify --task value explicitly");
}

if (this->serverSettings.hfSettings.task == TEXT_GENERATION_GRAPH) {
    // Text generation specific validation
}
```

---

## 6. CLI Parameter Exposure

**File:** [src/cli_parser.cpp](src/cli_parser.cpp#L363-L373)

```cpp
options->add_options("generative task (applies to: pull hf model, configure, single model)")
    ("task",
        "Specifies the generative task for the local model. It should be followed by task specific parameters. "
        "Supported tasks: text_generation, embeddings, rerank, image_generation, text2speech, speech2text. "
        "It creates the pipeline graph in memory based on the provided task-specific options. "
        "If not provided, default task value is inferred from model config.",
        cxxopts::value<std::string>(),
        "TASK")
    ("configure",
        "Create or update graph.pbtxt for the model specified by --model_path and --task. "
        "Does not start the server.",
        cxxopts::value<bool>()->default_value("false"),
        "CONFIGURE");
```

---

## 7. Downloader Task Routing

**File:** [src/pull_module/optimum_export.cpp](src/pull_module/optimum_export.cpp#L138-L169)

Different optimization/export commands per task:

```cpp
switch (this->task) {
    case TEXT_GENERATION_GRAPH:
        cmd = getConvertCmdWithDetokenizer();  // Includes detokenizer conversion
        break;
    case EMBEDDINGS_GRAPH:
        cmd = getConvertCmdOnlyTokenizer();
        break;
    case RERANK_GRAPH:
        cmd = getConvertCmdOnlyTokenizer();
        break;
    case TEXT_TO_SPEECH_GRAPH:
        cmd = getConvertCmdOnlyTokenizer();
        break;
    case SPEECH_TO_TEXT_GRAPH:
        cmd = getConvertCmdOnlyTokenizer();
        break;
    case IMAGE_GENERATION_GRAPH:
        cmd = "";  // No conversion for image generation
        break;
}
```

---

## 8. Task Detection Examples

| Model | Detection Method | Task |
|-------|-----------------|------|
| `Kokoro` (TTS) | `n_mels` field + null architectures | `text2speech` |
| `Whisper` | `WhisperForConditionalGeneration` | `speech2text` |
| `Qwen3-8B` | `Qwen3ForCausalLM` (no modules.json, no keyword) | `text_generation` |
| `Qwen3-Embedding-0.6B` | Name keyword "Embedding" + `Qwen3ForCausalLM` | `embeddings` |
| `Qwen3-Reranker-0.6B` | Name keyword "Rerank" + `Qwen3ForCausalLM` | `rerank` |
| `SDXL` / `Flux` | `model_index.json` _class_name | `image_generation` |
| `BGE / Sentence-BERT` | `modules.json` with `Pooling` | `embeddings` |

---

## 9. Server Modes & Task

**File:** [src/cli_parser.cpp](src/cli_parser.cpp#L847-L856)

Tasks interact with server modes:

```cpp
if (isConfigureMode(this->result)) {
    serverSettings.serverMode = CONFIGURE_MODE;        // Generate graph.pbtxt only
} else if (isInMemoryGraphMode(this->result)) {
    serverSettings.serverMode = IN_MEMORY_GRAPH_MODE;  // Task + model_path, no download
} else if (result->count("pull")) {
    serverSettings.serverMode = HF_PULL_MODE;          // Pull model, generate graph
} else {
    serverSettings.serverMode = HF_PULL_AND_START_MODE; // Pull + start server
}
```

---

## 10. Testing Task Detection

**File:** [src/test/task_determine_test.cpp](src/test/task_determine_test.cpp#L38-L58)

Test cases verify task detection for known models:

```cpp
std::vector<std::pair<std::string, std::string>> modelTaskMapping = {
    {"llama", "text_generation"},
    {"qwen3", "text_generation"},
    {"phi3", "text_generation"},
    {"bge", "embeddings"},
    {"t5_encoder", "embeddings"},
    {"qwen2_embedding", "embeddings"},
    {"cross_encoder", "rerank"},
    {"xlm_roberta", "rerank"},
    {"qwen2_rerank", "rerank"},
    {"stable_diffusion", "image_generation"},
    {"flux", "image_generation"},
    {"speecht5", "text2speech"},
    {"parlertts", "text2speech"},
    {"whisper", "speech2text"},
    {"seamlessm4t", "speech2text"},
};
```

---

## Summary: Task System Flow

```
1. User provides --task or allows inference
                     ↓
2. If inference needed: determineDefaultTaskParameter()
                     ↓
3. DefaultTaskDetector iterates detectors in priority order
   (Speech2Text → Text2Speech → Rerank → ImageGen → Embeddings → TextGen)
                     ↓
4. Task detected → stringToEnum() → GraphExportType enum
                     ↓
5. Task-specific CLI parser instantiated (GraphCLIParser, EmbeddingsGraphCLIParser, etc.)
                     ↓
6. prepareGraph() routes based on task
                     ↓
7. Task-specific graph template generated (graph.pbtxt)
                     ↓
8. Graph loaded into MediaPipe framework
                     ↓
9. HTTP/gRPC endpoints serve requests
```
