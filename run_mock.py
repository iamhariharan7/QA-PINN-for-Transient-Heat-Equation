import sys
import builtins
import run_experiment

inputs = ["1", "3"]
input_idx = 0
def mock_input(prompt):
    global input_idx
    print(prompt, end="")
    val = inputs[input_idx]
    print(val)
    input_idx += 1
    return val

builtins.input = mock_input

if __name__ == "__main__":
    run_experiment.main()
