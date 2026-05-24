from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.llm_client import LocalQwenClient


def main():
    llm = LocalQwenClient(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="auto"
    )

    prompt = "请用一句话说明轴承外圈故障通常会导致什么振动特征。"

    answer = llm.generate(
        prompt,
        temperature=0.1,
        max_new_tokens=256
    )

    print(answer)


if __name__ == "__main__":
    main()
