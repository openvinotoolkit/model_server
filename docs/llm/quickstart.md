# QuickStart - LLM models {#ovms_docs_llm_quickstart}

Let's deploy [OpenVINO/Qwen3-4B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-4B-int4-ov) model on Intel iGPU or ARC GPU.
It is [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) quantized to INT4 precision and converted to IR format.
Note that this model has reasoning capability. Look out for the content between `<think>...</think>` tags in the response.

## Requirements
- Linux or Windows 11
- Docker Engine or `ovms` binary package [installed](../deploying_server_baremetal.md)
- Intel iGPU or ARC GPU

## Deployment Steps
 
### 1. Deploy the Model
::::{tab-set}

:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
mkdir models
docker run --user $(id -u):$(id -g) -d --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render*) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --source_model OpenVINO/Qwen3-4B-int4-ov --model_repository_path models --rest_port 8000 --target_device GPU --cache_size 2
```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package with Python support - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bat
ovms.exe --source_model OpenVINO/Qwen3-4B-int4-ov --model_repository_path models --rest_port 8000 --target_device GPU --cache_size 2
```
:::
::::

First run of the command will download the https://huggingface.co/OpenVINO/Qwen3-4B-int4-ov to models/OpenVINO/Qwen3-4B-int4-ov directory and start serving it with ovms.
The consecutive run of the command will check that the model exists and start serving it. 

**Note**: Above deployment commands require internet access, so if you deploy behind the proxy, set appropriate proxy-related environment variables. 

### 2. Check Model Readiness

Wait for the model to load. You can check the status with a simple command:

```console
curl http://localhost:8000/v1/config
```

:::{dropdown} Expected Response
```json
{
  "OpenVINO/Qwen3-4B-int4-ov": {
    "model_version_status": [
      {
        "version": "1",
        "state": "AVAILABLE",
        "status": {
          "error_code": "OK",
          "error_message": "OK"
        }
      }
    ]
  }
}
```
:::

### 3. Run Generation

::::{tab-set}

:::{tab-item} Linux
```bash
curl -s http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenVINO/Qwen3-4B-int4-ov",
    "stream": false,
    "messages": [
      { "role": "system", "content": "You are a student on a math class. Provide solutions with straightforward explanations." },
      { "role": "user", "content": "You have the following equation: 3x + 4 = 16. Solve for x." }
    ]
  }' | jq .
```
:::

:::{tab-item} Windows

Windows Powershell
```powershell
(Invoke-WebRequest -Uri "http://localhost:8000/v3/chat/completions" `
 -Method POST `
 -Headers @{ "Content-Type" = "application/json" } `
 -Body '{"model": "OpenVINO/Qwen3-4B-int4-ov", "stream": false, "messages": [{"role": "system", "content": "You are a student on a math class. Provide solutions with straightforward explanations."}, {"role": "user", "content": "You have the following equation: 3x + 4 = 16. Solve for x."}]}').Content
```

Windows Command Prompt
```bat
curl -s http://localhost:8000/v3/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"OpenVINO/Qwen3-4B-int4-ov\", \"stream\": false, \"messages\": [{\"role\": \"system\", \"content\": \"You are a student on a math class. Provide solutions with straightforward explanations.\"}, {\"role\": \"user\", \"content\": \"You have the following equation: 3x + 4 = 16. Solve for x.\"}]}"
```
:::

::::

:::{dropdown} Expected Response
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "<think>\nOkay, so I need to solve the equation 3x + 4 = 16 for x. Let me think about how to approach this. I remember from class that solving an equation means finding the value of the variable that makes the equation true. In this case, the variable is x.\n\nFirst, I need to isolate x on one side of the equation. The equation has 3x plus 4 equals 16. To isolate x, I should get rid of the 4 that's being added to 3x. How do I do that? Well, the opposite of addition is subtraction, right? So if I subtract 4 from both sides of the equation, that should keep things balanced.\n\nLet me write that down. Starting with 3x + 4 = 16. Subtract 4 from both sides: 3x + 4 - 4 = 16 - 4. Simplifying both sides, the +4 and -4 on the left cancel out, leaving 3x = 12. Okay, that makes sense.\n\nNow, I need to get x by itself. The equation is 3x = 12. The 3 is multiplied by x, so to undo that, I should divide both sides by 3. That way, I can solve for x.\n\nSo, dividing both sides by 3: (3x)/3 = 12/3. Simplifying that, the 3s on the left cancel out, leaving x = 4. Let me check if that's correct.\n\nPlugging x = 4 back into the original equation: 3*(4) + 4 = 12 + 4 = 16. Which matches the right side of the equation, 16. So that works. Therefore, x = 4 is the solution.\n\nWait, did I do everything right? Let me go through the steps again. Original equation: 3x + 4 = 16. Subtract 4: 3x = 12. Divide by 3: x = 4. Yep, that seems correct. I don't think I made any mistakes here. The steps are straightforward: inverse operations to isolate x. Maybe I can think of another way to check, like using algebra tiles or something, but since it's a simple equation, the algebraic method is sufficient.\n\nAlternatively, maybe I can think of it as a balance scale. If I have 3x + 4 on one side and 16 on the other, I need to keep the balance. So subtracting 4 from both sides is like removing 4 from both sides, which keeps the balance. Then, dividing both sides by 3 is like splitting the remaining 12 into three equal parts, each part being 4. That makes sense.\n\nI think that's all. The key steps are inverse operations to isolate the variable. I don't see any errors in my reasoning. The answer checks out when substituting back into the original equation. So I'm confident that x = 4 is the correct solution.\n</think>\n\nTo solve the equation $ 3x + 4 = 16 $ for $ x $, follow these steps:\n\n1. **Isolate the term with the variable**:  \n   Subtract 4 from both sides to eliminate the constant term on the left:  \n   $$\n   3x + 4 - 4 = 16 - 4\n   $$  \n   Simplifying both sides:  \n   $$\n   3x = 12\n   $$\n\n2. **Solve for $ x $**:  \n   Divide both sides by 3 to isolate $ x $:  \n   $$\n   \\frac{3x}{3} = \\frac{12}{3}\n   $$  \n   Simplifying:  \n   $$\n   x = 4\n   $$\n\n**Check**:  \nSubstitute $ x = 4 $ back into the original equation:  \n$$\n3(4) + 4 = 12 + 4 = 16\n$$  \nThis matches the right-hand side, confirming the solution is correct.\n\n**Answer**:  \n$$\n\\boxed{4}\n$$",
        "role": "assistant",
        "tool_calls": []
      }
    }
  ],
  "created": 1749567097,
  "model": "OpenVINO/Qwen3-4B-int4-ov",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 49,
    "completion_tokens": 877,
    "total_tokens": 926
  }
}
```
:::

#### Using OpenAI Python Client:

First, install the openai client library:
```console
pip3 install openai
```
Then run the following Python code:

```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.chat.completions.create(
    model="OpenVINO/Qwen3-4B-int4-ov",
    messages=[
      { "role": "system", "content": "You are a student on a math class. Provide solutions with straightforward explanations." },
      { "role": "user", "content": "You have the following equation: 3x + 4 = 16. Solve for x." }
    ],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Expected output:
```
<think>
Okay, so I need to solve the equation 3x + 4 = 16 for x. Let me think about how to approach this. I remember from class that solving an equation means finding the value of the variable that makes the equation true. In this case, the variable is x.

First, I need to isolate x on one side of the equation. The equation has 3x plus 4 equals 16. To isolate x, I should get rid of the 4 that's being added to 3x. How do I do that? Well, the opposite of addition is subtraction, right? So if I subtract 4 from both sides of the equation, that should keep things balanced.

Let me write that down. Subtracting 4 from both sides:

3x + 4 - 4 = 16 - 4

Simplifying both sides, the +4 and -4 on the left side cancel each other out, leaving just 3x. On the right side, 16 minus 4 is 12. So now the equation is 3x = 12.

Now, I need to get x by itself. Since 3 is multiplied by x, I should divide both sides by 3 to undo that multiplication. Let me do that:

3x / 3 = 12 / 3

Simplifying again, the 3s on the left side cancel out, leaving x. On the right side, 12 divided by 3 is 4. So x equals 4.

Wait, let me check if that makes sense. If I plug x = 4 back into the original equation, does it hold true?

Original equation: 3x + 4 = 16

Substitute x with 4: 3*(4) + 4 = 12 + 4 = 16. Yes, that's correct. 12 + 4 is indeed 16. So that checks out.

Hmm, but wait, did I do all the steps right? Let me go through it again. Starting with 3x + 4 = 16. Subtract 4 from both sides: 3x = 12. Divide by 3: x = 4. Yep, that seems right. I don't think I made any mistakes here. The operations are straightforward: inverse operations to isolate x.

Another way to think about it: maybe using inverse operations step by step. The equation is 3x + 4 = 16. So first, subtract 4 from both sides to get 3x = 12. Then divide by 3 to get x = 4. That's the same as before.

Alternatively, could I have done something else? Like, maybe dividing first? Let's see. If I try to divide both sides by 3 first, that would give me x + 4/3 = 16/3. Then subtract 4/3 from both sides, which would be x = 16/3 -
4/3 = 12/3 = 4. So that also works. But that seems more complicated. The first method is simpler because it's easier to subtract 4 first, then divide.

But in any case, both methods lead to the same answer. So x = 4 is definitely the solution.

I think that's all. The key steps are to perform inverse operations to isolate the variable. First, subtract the constant term (4) from both sides, then divide by the coefficient of x (3) to solve for x. Checking the solution by substituting back into the original equation confirms that it's correct. So I'm confident that x = 4 is the right answer.
</think>

To solve the equation $ 3x + 4 = 16 $ for $ x $, follow these steps:

1. **Subtract 4 from both sides** to isolate the term with $ x $:
   $$
   3x + 4 - 4 = 16 - 4
   $$
   Simplifying:
   $$
   3x = 12
   $$

2. **Divide both sides by 3** to solve for $ x $:
   $$
   \frac{3x}{3} = \frac{12}{3}
   $$
   Simplifying:
   $$
   x = 4
   $$

**Check the solution**:
Substitute $ x = 4 $ back into the original equation:
$$
3(4) + 4 = 12 + 4 = 16
$$
The left side equals the right side, confirming the solution is correct.

**Answer:** $ x = 4 
```

## References
- [Efficient LLM Serving - reference](reference.md)
- [Exporting GEN AI Models](../../demos/common/export_models/README.md)
- [Chat Completions API](../model_server_rest_api_chat.md)
- [Completions API](../model_server_rest_api_completions.md)
- [Demo with Llama3 serving](../../demos/continuous_batching/README.md)
