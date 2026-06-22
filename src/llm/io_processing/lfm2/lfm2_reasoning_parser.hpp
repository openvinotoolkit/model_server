#include "../qwen3/reasoning_parser.hpp"

namespace ovms {
class Lfm2ReasoningParser : public Qwen3ReasoningParser {
public:

    Lfm2ReasoningParser(ov::genai::Tokenizer& tokenizer) : Qwen3ReasoningParser(tokenizer) {}

    bool requiresStreamingWithSpecialTokens() const override {
        return true;
    }

};
}