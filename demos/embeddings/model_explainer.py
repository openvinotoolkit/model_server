from openvino import compile_model, convert_model
import openvino as ov
import argparse

parser = argparse.ArgumentParser(description='Description of your program')

parser.add_argument('input_file', help='Path to the input file')
args = parser.parse_args()

print(args.input_file)

core = ov.Core()
core.add_extension("./../../.venv_dk/lib/python3.10/site-packages/openvino_tokenizers/lib/libopenvino_tokenizers.so")
# embedding model
tokenizer_model = core.read_model(model=args.input_file)

tk = compile_model(tokenizer_model)

# List inputs and outputs for the tokenizer model
tokenizer_inputs = tk.inputs
tokenizer_outputs = tk.outputs

print("Model Inputs:")
for input in tokenizer_inputs:
    print(f"Name: {input.get_any_name()}, Shape: {input.get_partial_shape()}, Precision: {input.get_element_type()}")

print("\nModel Outputs:")
for output in tokenizer_outputs:
    print(f"Name: {output.get_any_name()}, Shape: {output.get_partial_shape()}, Precision: {input.get_element_type()}")

