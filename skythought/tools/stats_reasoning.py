import json
import pandas as pd

numina_result_files = [
    "../../data/R1_NUMINA_train_amc_aime_0_-1.json",
    "../../data/R1_NUMINA_train_math_0_-1.json",
    "../../data/R1_NUMINA_train_olympiads_0_-1.json",
]


token_usage_data = []
for file in numina_result_files:
    print(file)
    with open(file, "r") as f:
        data = json.load(f)
        print(len(data))

    for key, value in data.items():
        token_usage_data.append({
            "source": value["source"],
            "problem": value["problem"],
            "difficulty": value["difficulty"],
            "prompt_tokens": value["token_usages"]["0"]["prompt_tokens"],
            "completion_tokens": value["token_usages"]["0"]["completion_tokens"],
        })

df = pd.DataFrame.from_records(token_usage_data)
print(df)

df.to_csv("../../data/numina_token_usage_data.csv", index=False, index_label=False)


source_list = df["source"].unique()
for source in source_list:
    print(f"======= Source: {source} =======")
    df_source = df[df["source"] == source]
    # print(df_source)
    print(df_source.groupby("difficulty")["completion_tokens"].mean())
