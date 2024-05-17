#include <atomic>
#include <thread>

#include "continouos_batching_pipeline.hpp"

class LLMLoop {

std::atomic<bool> end = false;
std::shared_ptr<ContinuousBatchingPipieline> pipe;
std::thread llmLoopThread;

void run() {
    while (!end) {
        pipe->step();
    }
}

public:

LLMLoop(std::shared_ptr<ContinuousBatchingPipieline> pipe) : pipe(pipe) {
    llmLoopThread = std::thread(run);
}

~LLMLoop() {
    end = true;
    llmLoopThread.join();
}

}