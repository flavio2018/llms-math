import re
import warnings


class GPT4OutputParser:

    def __init__(self):
        self.formatted_output_re = None
        self.few_shot_re = None
        self.output_type = None
        self.error_counter = 0

    def _parse_formatted_outputs(self, outputs):
        match = self.formatted_output_re.findall(outputs)
        if len(match) > 0:
            match = match[-1][1]
            if match == "":
                breakpoint()
                warnings.warn(f"Match is empty for GPT-4 output: {outputs}")
                match = "-100"
        else:
            breakpoint()
            warnings.warn(
                f"The following GPT-4 output did not produce any match: {outputs}."
            )
            match = -100
        if self.output_type == int:
            return int(match)
        elif self.output_type == str:
            return match

    def _parse_few_shot_outputs(self, outputs):
        try:
            match = self.few_shot_re.findall(outputs)[-1]
        except IndexError:
            breakpoint()
            warnings.warn(f"Match is empty for GPT-4 output: {outputs}")
            match = "-100"
        if isinstance(match, tuple):  # for algebra
            match = match[1]
            if match == "":
                breakpoint()
                warnings.warn(f"Match is empty for GPT-4 output: {outputs}")
                match = "-100"
        if self.output_type == int:
            return int(match)
        elif self.output_type == str:
            return match

    def _filter_matches(self, matches):
        if isinstance(matches[0], tuple):
            matches = [(m[0].strip(), m[1].strip(), m[2].strip()) for m in matches]
            matches = [match for match in matches if match != ("", "", "")]
            # matches = [match for match in matches if match[0] not in ['a', 'b', 'by', 'x', 'y', ')']]
        return matches

    def _simple_parse_outputs(self, outputs):
        # print(outputs.split('\n')[-1])
        outputs = self._preprocessing_step(outputs)
        try:
            matches = self.simple_output_re.findall(outputs)
            matches = self._filter_matches(matches)
            # matches = [match[0] for match in matches]
            # print(matches)
            match = matches[-1]
        except IndexError:
            # breakpoint()
            warnings.warn(f"Match is empty for GPT-4 output: {outputs}")
            match = "-100"
            self.error_counter += 1

        if isinstance(match, tuple):
            match = match[0]
            if match == "":
                # breakpoint()
                warnings.warn(f"Match is empty for GPT-4 output: {outputs}")
                match = "-100"
                self.error_counter += 1

        if self.output_type == int:
            return int(match)
        elif self.output_type == str:
            return match

    def parse_outputs(self, outputs):
        return self._simple_parse_outputs(outputs)


class O1ListopsOutputParser(GPT4OutputParser):
    def __init__(self):
        super().__init__()
        self.formatted_output_re = re.compile(
            r"(So|Thus)+[,]* the final result is[:]*[\n]*[\n]*[ ]*[max(7069,)= ]*(\d)"
        )
        # self.few_shot_re = re.compile(r'=\n[ ]*[\[]*(\d)[\]]*')
        self.few_shot_re = re.compile(
            r"(final result is|is |= *)+[,]*[:]*[\n]*[\n]*[ ]*(\d)"
        )
        self.simple_output_re = re.compile(r"\d")
        self.output_type = int

    def _preprocessing_step(self, output):
        return output


class GPT4ListopsOutputParser(GPT4OutputParser):

    def __init__(self):
        super().__init__()
        self.formatted_output_re = re.compile(
            r"(So|Thus)+[,]* the final result is[:]*[\n]*[\n]*[ ]*[max(7069,)= ]*(\d)"
        )
        # self.few_shot_re = re.compile(r'=\n[ ]*[\[]*(\d)[\]]*')
        self.few_shot_re = re.compile(
            r"(final result is|is |= *)+[,]*[:]*[\n]*[\n]*[ ]*(\d)"
        )
        self.simple_output_re = re.compile(r"\d")
        self.output_type = int

    def _preprocessing_step(self, output):
        output = output.replace("modulo 10", "")
        output = output.replace("mod 10", "")
        output = output.replace("Modulo 10", "")
        output = output.replace("Mod 10", "")
        output = output.replace("modulus 10", "")
        output = output.replace("Modulus 10", "")
        return output


class O1ArithmeticOutputParser(GPT4OutputParser):

    def __init__(self):
        super().__init__()
        self.formatted_output_re = re.compile(
            r"(So|Thus)+[,]* the final result is[:]*[\n]*[\n]*[ ]*([-]*\d+)"
        )
        # self.few_shot_re = re.compile(r'=\n[ ]*([-]*\d+)')
        self.few_shot_re = re.compile(
            r"(final result is|final result is therefore|the result is|final result of the arithmetic expression is|the arithmetic expression results in|= *)+[:]*[\n]*[ ]*([-]*\d+)"
        )
        self.simple_output_re = re.compile(r"\-{0,1}\d\d*")
        self.output_type = int

    def _preprocessing_step(self, output):
        return output


class GPT4ArithmeticOutputParser(GPT4OutputParser):

    def __init__(self):
        super().__init__()
        self.formatted_output_re = re.compile(
            r"(So|Thus)+[,]* the final result is[:]*[\n]*[\n]*[ ]*([-]*\d+)"
        )
        # self.few_shot_re = re.compile(r'=\n[ ]*([-]*\d+)')
        self.few_shot_re = re.compile(
            r"(final result is|final result is therefore|the result is|final result of the arithmetic expression is|the arithmetic expression results in|= *)+[:]*[\n]*[ ]*([-]*\d+)"
        )
        self.simple_output_re = re.compile(r"\-{0,1}\d\d*")
        self.output_type = int

    def _preprocessing_step(self, output):
        output = output.replace("modulo -100", "")
        output = output.replace("Modulo -100", "")
        output = output.replace("(mod -100)", "")
        output = output.replace("mod -100", "")
        output = output.replace("Mod -100", "")
        output = output.replace("modulo 100", "")
        output = output.replace("Modulo 100", "")
        output = output.replace("(mod 100)", "")
        output = output.replace("mod 100", "")
        output = output.replace("Mod 100", "")
        output = output.replace("modulus 100", "")
        output = output.replace("Modulus 100", "")
        return output


class O1AlgebraOutputParser(GPT4OutputParser):

    def __init__(self):
        super().__init__()
        # self.formatted_output_re = re.compile(r'(So|Thus)+[,]* the final result is[:]*[ ]*[\n]*[ ]*([+-]*[0-9]*[0-9* ]*[abxy* ]*([(]([-+0-9]|[-abxy])+)*[abxy* ]*[+-]*[0-9 ]*[abxy* ]*[/0-9]*[)]*)')
        self.formatted_output_re = re.compile(
            r"(final results* is|final result after simplification and considering modulo 100 for coefficients is|final result in the form of a · x · y is|final result after factoring by grouping will be| or )+[ :]*[\n]*[\n]*[=]*[ ]*[`]*([+-]*[0-9]*[0-9* ]*[abxXy* ]*([(][ ]*([-0-9]|[-abxy])+)*[abxy* ]*[0-9 ]*[ +-]*[0-9 ]*[abxy* ]*[/0-9]*[abxy*]*[)]*)[`]*"
        )
        self.simple_output_re = re.compile(
            r"([+-]*[0-9]*[0-9]*[* ]*[abxy*]*[ ]*([(]([-+0-9]|[-abxy])+)*[abxy*]*[ ]*[+-]*[ ]*[0-9]*[* ]*[abxy* ]*[/0-9]*[)]*[abxy*]*)"
        )
        self.output_type = str

    def _preprocessing_step(self, output):
        return output


class GPT4AlgebraOutputParser(GPT4OutputParser):

    def __init__(self):
        super().__init__()
        # self.formatted_output_re = re.compile(r'(So|Thus)+[,]* the final result is[:]*[ ]*[\n]*[ ]*([+-]*[0-9]*[0-9* ]*[abxy* ]*([(]([-+0-9]|[-abxy])+)*[abxy* ]*[+-]*[0-9 ]*[abxy* ]*[/0-9]*[)]*)')
        self.formatted_output_re = re.compile(
            r"(final results* is|final result after simplification and considering modulo 100 for coefficients is|final result in the form of a · x · y is|final result after factoring by grouping will be| or )+[ :]*[\n]*[\n]*[=]*[ ]*[`]*([+-]*[0-9]*[0-9* ]*[abxXy* ]*([(][ ]*([-0-9]|[-abxy])+)*[abxy* ]*[0-9 ]*[ +-]*[0-9 ]*[abxy* ]*[/0-9]*[abxy*]*[)]*)[`]*"
        )
        # self.few_shot_re = re.compile(r'(=\n*|= \n|\n=|\n= |^|factoring by grouping is |[A-Za-z]+:[ ]*\n*|the final result is |simplifies to |simplified form of the expression is |simplified version of your expression is |simplified algebraic expression is |final result is |or |step \d+:|Step \d+:\n*)([(]*[+-]*[0-9]*[0-9* ]*[abxy* ]*[)]?[ ]?[+-]?[ ]?([(]([-+0-9]|[-abxy])+)*[abxy* ]*[+-]*[0-9 ]*[abxy* ]*[/0-9]*[)]*)($| *\.$|\.\n| \n|\n\n)')
        # self.few_shot_re = re.compile(r'([(]*[+-]*[0-9]*[0-9* ]*[abxy* ]*[)]?[ ]?[+-]?[ ]?([(]([-+0-9]|[-abxy])+)*[abxy* ]*[+-]*[0-9 ]*[abxy* ]*[/0-9]*[)]*)+')
        # self.few_shot_re = re.compile(r'([+-]*[0-9]*[0-9* ]*[abxy* ]|[+-]*[0-9]*[0-9* ]*[abxy* ]([(]([-+0-9]|[-abxy])+)*[abxy* ]*[+-]*[0-9 ]*[abxy* ]*[/0-9]*[)]|[(]*[+-]*[0-9]*[0-9* ]*[abxy* ]*[)]?[ ]?[+-]?[ ]?[(]*[+-]*[0-9]*[0-9* ]*[abxy* ]*[)]?)')
        self.few_shot_re = re.compile(
            (
                "(our answer is|"
                "the final simplified form is|"
                "the simplified form of the given expression is|"
                "the simplified answer is|"
                "Final expression becomes|"
                "or simply|"
                "equivalent in modulus, |"
                "the expression becomes|"
                "the whole expression simplifies to|"
                "final simplified result is|"
                "simplified result is|"
                "final answer is|"
                "final simplified form of the given algebraic expression is|"
                "final result is|"
                "the final result is same as|"
                "the simplified expression is|"
                "the final result of simplification is|"
                "the final result would be|"
                "the final expression is|"
                "final result of the expression is|"
                "final result of the simplified expression is|"
                "then factor this result|"
                "final simplified expression modulo 100 is|"
                "final simplified algebraic expression is|"
                "the result is|"
                " or |"
                "inal result:|"
                "final simplified expression is|"
                "final result as|"
                "modulo 100 as |"
                "the final result of the simplification is|"
                "we get|"
                "= )+[ ]*[=]*[:]*[ ]*[\n]*[\n]*[ ]*[`]*[$]*([+-]*[0-9]*[0-9* ]*[abxy* ]*([(]([-+0-9]|[-abxy])+)*[abxy* ]*[ +-]*[0-9 ]*[abxy* ]*[/0-9]*[)]*)[`]*"
            )
        )
        self.simple_output_re = re.compile(
            r"([+-]*[0-9]*[0-9]*[* ]*[abxy*]*[ ]*([(]([-+0-9]|[-abxy])+)*[abxy*]*[ ]*[+-]*[ ]*[0-9]*[* ]*[abxy* ]*[/0-9]*[)]*[abxy*]*)"
        )
        self.output_type = str

    def _preprocessing_step(self, output):
        output = output.replace("modulo -100", "")
        output = output.replace("Modulo -100", "")
        output = output.replace("(mod -100)", "")
        output = output.replace("mod -100", "")
        output = output.replace("Mod -100", "")
        output = output.replace("modulo 100", "")
        output = output.replace("Modulo 100", "")
        output = output.replace("(mod 100)", "")
        output = output.replace("mod 100", "")
        output = output.replace("Mod 100", "")
        output = output.replace("modulus 100", "")
        output = output.replace("Modulus 100", "")
        return output


class O1LogicOutputParser(GPT4OutputParser):

    def __init__(self):
        super().__init__()
        self.simple_output_re = re.compile(r"[A-Za-z]")
        self.output_type = str

    def _preprocessing_step(self, output):
        return output


class GPT4LogicOutputParser(GPT4OutputParser):

    def __init__(self):
        super().__init__()
        self.simple_output_re = re.compile(r"[A-Za-z]")
        self.output_type = str

    def _preprocessing_step(self, output):
        output = output.replace("True", "T")
        output = output.replace("true", "T")
        output = output.replace("False", "F")
        output = output.replace("false", "F")
        if "&" in output or "|" in output or "(" in output or ")" in output:
            output = "Z"
        return output


def build_parser(task_name, model_name="GPT4"):
    if model_name == "GPT4":
        if task_name == "algebra":
            return GPT4AlgebraOutputParser()
        elif task_name == "arithmetic":
            return GPT4ArithmeticOutputParser()
        elif task_name == "listops":
            return GPT4ListopsOutputParser()
        elif task_name == "logic":
            return GPT4LogicOutputParser()
        else:
            assert False, f"Wrong task name {task_name}"
    elif model_name == "o1-preview":
        if task_name == "algebra":
            return O1AlgebraOutputParser()
        elif task_name == "arithmetic":
            return O1ArithmeticOutputParser()
        elif task_name == "listops":
            return O1ListopsOutputParser()
        elif task_name == "logic":
            return O1LogicOutputParser()
        else:
            assert False, f"Wrong task name {task_name}"
    else:
        assert False, f"Wrong model name {model_name}"
