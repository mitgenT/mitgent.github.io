---
# the default layout is 'page'
icon: fas fa-info-circle
order: 1
---

# MitGen

## Description

MitGen is an approach that can effectively find failure-inducing test cases with the help of the compliable code synthesized by the provided intention. This command line tool allows users to interact with LLMs to automatically generate reference versions, code, and test input pools.

## Installation

Clone or download our code to your local machine:

   ```bash
   git clone https://github.com/mitgenT/MitGen.git
   ```

## Usage

Before using MitGen, you should set your own LLM API Key or local LLM.

For API Key, please create a file api.py on `/mutation` and write:

`[API Key Name] = [your_key]`

API Key Name:
- Qwen: qwen_api_key
- ChatGPT: chatgpt_api_key
- Llama: llama_api_key

Details can be obtained from functions `prompt_*` in `/mutation/generate_code_with_llm.py`

For local LLMs, please go to `GenerateCodeWithLLM.initialise_hf_model(), prompt_hf_model()` and add your LLM.

Next, put the following files(mostly in plain text) into corresponding locations:

- PUT: `/mutation/subjects/target_code`
- argument: properties of PUT, in json format in `/mutation/subjects/args`. e.g.
```json
{
  "mask_location": null,
  "func_name": "make_palindrome",
  "subject_type": "function"
}
```
  - mask_location: For debug, just keep null.
  - subject_type: Please refer to the following "input_mode" part.
  - func_name: Function name of PUT. This can be omitted if the target code is not a function.
- correct code: Bug-free version of PUT in `/mutation/subjects/correct_code`. This is for evaluation usage.
- Docstring: In `/mutation/subjects/prompt`.
- Test Case: In `/mutation/subjects/playground_template`. This is for evaluation usage. This can be omitted if the target
code is not a class.

To use the command line tool, run the following command first:

```bash
python mutation/run.py [target] [model] [mode] [input_mode]
```

The available options and commands are:

- target: file name of PUT, located in `/mutation/subjects/target_code`. (e.g. cf835.txt)
- model: name of LLM, check `GenerateCodeWithLLM.prompt_llm()` (for API) or `initialise_hf_model() and prompt_hf_model()`
(for local LLMs) for details.
- mode: just input "ast"
- input_mode:
  - If your target code accepts input from command line, input "console".
  - If your target code accepts input from function parameters, input "function".
  - If your target code is a class, input "class".

The above command performs the following operations in our paper:

- Stage 1 of CamPri: Parse a PUT into an AST
- Stage 3 of CamPri: Generate test inputs(seed inputs only)
- Stage 1 of IRVGen: Generate reference versions

Parsed AST code will be saved in `/mutation/subjects/input`, reference versions will be saved in
`/mutation/subjects/output`, seed inputs will be saved in `/mutation/subjects/test_input`.

Next, run the following command:

```bash
python mutation/prioritization.py [model]
```

The above command performs the following operations in our paper:

- Stage 2 of CamPri: Infill each masked location.
- Stage 4 of CamPri: Compute context similarity between generated snippets and original snippet.
- Stage 5 of CamPri: Prioritize masked location.

Finally, run the following command:

```bash
python mutation/end2end.py
```

The above command performs the following operations in our paper:

- Stage 2 of IRVGen: Infer an expected output.
- Stage 3 of IRVGen: Compare actual output and expected output.

Once the command is executed, the results will be saved in the `output` directory.

Overall, this optimized instruction should be easier for users to understand and follow.
