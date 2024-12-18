import glob

import pandas as pd

from eval_self_consistency import add_parsed_output_to_df, eval_df
from parse import build_parser


def main():
    model_name = "o1-preview"
    task_names = ["listops", "arithmetic", "algebra", "logic"]

    nesting_ranges = {
        "arithmetic": range(1, 7),
        "algebra": range(1, 7),
        "listops": range(1, 7),
        "logic": range(1, 13),
    }

    num_operands_ranges = {
        "arithmetic": range(2, 3),
        "algebra": range(2, 3),
        "listops": range(2, 5),
        "logic": range(2, 3),
    }

    for task_name in task_names:
        if task_name == "listops":
            accuracy_table = pd.DataFrame(index=[1, 2, 3, 4, 5, 6], columns=[2, 3, 4])
        elif task_name == "logic":
            accuracy_table = pd.DataFrame(
                index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], columns=[2]
            )
        else:
            accuracy_table = pd.DataFrame(index=[1, 2, 3, 4, 5, 6], columns=[2])

        for nesting in nesting_ranges[task_name]:
            for num_operands in num_operands_ranges[task_name]:
                output_files = glob.glob(
                    f"output/{model_name}/{task_name}__nes{nesting}__nop{num_operands}*.csv"
                )
                if output_files:
                    print(
                        f"\n#### Evaluating {task_name}, ({nesting}, {num_operands}). ####"
                    )

                    if len(output_files) > 1:
                        df = pd.concat(
                            [
                                pd.read_csv(output_file, index_col=0)
                                for output_file in output_files
                            ]
                        )
                        df.reset_index(drop=True, inplace=True)
                    else:
                        df = pd.read_csv(output_files[0], index_col=0)
                    df["gpt_output_original"] = df["gpt_output"]
                    cut_last = df["gpt_output_original"].str[-32:]
                    df["gpt_output"] = cut_last.apply(
                        lambda x: (
                            x.replace("\n\n", "\n")
                            .replace("\n", " ")
                            .replace("\\", "")
                            .replace("**", "")
                            .replace("boxed", "")
                        )
                    )
                    if task_name in ["algebra"]:
                        df["gpt_output"] = df["gpt_output"].apply(
                            lambda x: x.replace("\xa0", " ")
                            .replace("\u202f", " ")
                            .replace("\u2003", " ")
                            .replace("\u2009", " ")
                            .replace("×", "*")
                            .replace("·", "*")
                            .replace("cdot", "*")
                            .replace(",", "*")
                            .replace("times", "*")
                            .replace("displaystyle", "")
                            .replace("[", "")
                            .replace("]", "")
                            .replace("(", "")
                            .replace(")", "")
                            .replace(" a", "a")
                            .replace(" b", "b")
                            .replace(" x", "x")
                            .replace(" y", "y")
                            .replace(" * ", "*")
                            .replace("*   ", "")
                        )
                    df["gpt_output"] = (
                        df["gpt_output"]
                        .apply(lambda x: x.split("Answer:")[1] if "Answer:" in x else x)
                        .apply(lambda x: x.split(":")[1] if ":" in x else x)
                    )
                    if task_name == "algebra":
                        df["gpt_output"] = (
                            df["gpt_output"]
                            .apply(
                                lambda x: x.split("simply")[1] if "simply" in x else x
                            )
                            .apply(lambda x: "-" + x.split(" -")[1] if " -" in x else x)
                        )
                    if task_name == "logic":
                        df["gpt_output"] = df["gpt_output"].apply(
                            lambda x: x.replace("True", "T").replace("False", "F")
                        )
                    print(df["gpt_output"])
                    # if task_name == "algebra":
                    #     breakpoint()
                    parser = build_parser(df["task_name"][0], model_name="o1-preview")
                    add_parsed_output_to_df(df, parser)
                    print(f"{parser.error_counter} errors while parsing file.")
                    print(df["parsed_output"])
                    run_acc = eval_df(df)
                    accuracy_table.loc[nesting, num_operands] = run_acc
                else:
                    accuracy_table.loc[nesting, num_operands] = -1

        if task_name == "listops":
            accuracy_table = accuracy_table.mean(axis=1)

        accuracy_table.to_csv(f"output/accuracy_tables/{model_name}_{task_name}.csv")


if __name__ == "__main__":
    main()
