import os
import pickle

try:
    import openai
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_DPG"))
except ModuleNotFoundError as err:
    print(str(err))
    print("openai not installed.")

import time
import tqdm
import warnings
import utils


class OpenAIInterface:

    def __init__(self, cfg, run_timestamp):
        self.time_sleep = 3
        self.max_time_sleep = 3**3
        self.zero_cot_first_outputs = []
        self.cfg = cfg
        self.run_timestamp = run_timestamp

    def _build_structured_prompts_conversation(self, conversation_prompts):
        structured_prompts = []
        for conversation in conversation_prompts:
            structured_prompt = []
            for line_idx, line in enumerate(conversation):
                if line_idx % 2 == 0:
                    structured_prompt.append({"role": "user", "content": line})
                else:
                    structured_prompt.append({"role": "assistant", "content": line})
            structured_prompts.append(structured_prompt)
        return structured_prompts

    def _build_structured_prompts_simple(self, prompts, role_description=None):
        if role_description is not None:
            return [
                [{"role": "system", "content": role_description}]
                + [{"role": "user", "content": prompt}]
                for prompt in prompts
            ]
        else:
            return [[{"role": "user", "content": prompt}] for prompt in prompts]

    def _build_structured_prompts(self, prompts, role_description=None):
        assert isinstance(prompts, list)

        if isinstance(prompts[0], list):
            return self._build_structured_prompts_conversation(prompts)
        elif isinstance(prompts[0], str):
            return self._build_structured_prompts_simple(prompts, role_description)
        else:
            assert False, f"Wrong prompt type {type(prompts[0])}"

    def _get_time_sleep(self):
        if self.time_sleep == self.max_time_sleep:
            self.time_sleep = 3
        else:
            self.time_sleep *= 3
        return self.time_sleep

    @staticmethod
    def _convert_model_name(model_name):
        if model_name == "gpt4":
            return "gpt-4"
        elif model_name == "gpt35":
            return "gpt-3.5-turbo"
        elif model_name in ["o1-mini", "o1-preview"]:
            return model_name
        else:
            assert False, f"Wrong model name: {model_name}."

    def _single_query_model(self, model_name, structured_prompt):
        res = ""
        while res == "":
            try:
                res = client.chat.completions.create(
                    model=model_name, messages=structured_prompt
                )
                res = res.model_dump()  # make a dict
            except (
                openai.RateLimitError,
                # openai.Timeout,
                openai.BadRequestError,
                openai.APIError,
                openai.APIConnectionError,
            ) as e:
                warnings.warn(str(e))
                time.sleep(self._get_time_sleep())

                if "context length" in str(e):
                    res = {"choices": [{"message": {"content": ""}}]}
        output = res["choices"][0]["message"]["content"].strip()
        return output

    def query_o1(self, prompts):
        outputs = []
        model_name = self._convert_model_name(self.cfg.model_name)
        print(f"Querying {model_name} with {len(prompts)} prompts...")
        structured_prompts = self._build_structured_prompts(prompts)
        for prompt in tqdm.tqdm(structured_prompts):
            outputs.append(self._single_query_model(model_name, prompt))
        return outputs

    def query_model_zero_shot_cot(self, prompts):
        model_name = self._convert_model_name(self.cfg.model_name)
        print(f"Querying {model_name} with {2*len(prompts)} prompts...")
        self.zero_cot_first_outputs = []

        if not utils.exists_run_tmp(self.cfg):
            structured_prompts = self._build_structured_prompts(prompts)
            for prompt in tqdm.tqdm(structured_prompts):
                self.zero_cot_first_outputs.append(
                    self._single_query_model(model_name, prompt)
                )
            with open(
                f"output/gpt4/{utils.get_run_filename(self.cfg)}.tmp",
                "wb",
            ) as tmp_f:
                pickle.dump(self.zero_cot_first_outputs, tmp_f)
        else:
            ans = ""
            while ans not in ["y", "n"]:
                ans = input(">>> Found tmp file, do you want to use it? [y/n] ")
            if ans == "n":
                structured_prompts = self._build_structured_prompts(prompts)
                for prompt in tqdm.tqdm(structured_prompts):
                    self.zero_cot_first_outputs.append(
                        self._single_query_model(model_name, prompt)
                    )
            else:
                with open(
                    f"output/gpt4/{utils.get_run_filename(self.cfg)}.tmp",
                    "wb",
                ) as tmp_f:
                    self.zero_cot_first_outputs = pickle.load(tmp_f)
                structured_prompts = self._build_structured_prompts(prompts)
                for prompt_idx, prompt in enumerate(tqdm.tqdm(structured_prompts)):
                    if prompt_idx < len(self.zero_cot_first_outputs):
                        continue
                    self.zero_cot_first_outputs.append(
                        self._single_query_model(model_name, prompt)
                    )

        prompts_with_answer = [
            prompt + "\n\n" + output + "\n\n" + "Therefore, the final answer is:"
            for prompt, output in zip(prompts, self.zero_cot_first_outputs)
        ]
        structured_prompts_with_answer = self._build_structured_prompts(
            prompts_with_answer
        )
        final_outputs = []

        if os.path.exists(
            f"output/gpt4/{utils.get_run_filename(self.cfg)}_final_outputs.tmp"
        ):
            with open(
                f"output/gpt4/{utils.get_run_filename(self.cfg)}_final_outputs.tmp",
                "rb",
            ) as out_f:
                final_outputs = pickle.load(out_f)

        for prompt_idx, prompt in enumerate(tqdm.tqdm(structured_prompts_with_answer)):
            if prompt_idx < len(final_outputs):
                continue
            final_outputs.append(self._single_query_model(model_name, prompt))
            with open(
                f"output/gpt4/{utils.get_run_filename(self.cfg)}_final_outputs.tmp",
                "wb",
            ) as tmp_f:
                pickle.dump(final_outputs, tmp_f)
        return final_outputs
