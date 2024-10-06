'''
python  main.py \
        --experiment ./results/gpt4/imdb/test \
        --filepath ./data/imdb/imdb_test.csv \
        --task cls \
        --dataset IMDB \
        --split test \
        --text_column text \
        --label_column label 

ArgumentParser：用于解析命令行参数。
Guided 和 General：这两个类包含用于生成指导性复制和通用复制任务的提示模板。
ICLEvaluation：用于生成用于评估复制文本准确性的提示模板。
TongyiQianwenClient：一个客户端类，用于调用通义千问API来生成文本。
ExperimentResultSaver：用于保存实验结果。
ReplicationPhase：用于执行复制任务并保存生成的文本。
ICL：用于执行评估复制文本准确性的任务。
PatternCounter：用于统计评估结果中的模式。
Alg2EvalPhase：用于执行评估阶段并保存结果
'''

# 导入所需的库
import re  # 正则表达式库
import pandas as pd  # 数据处理库
from dashscope import Generation  # 用于调用通义千问API的库
import argparse  # 命令行参数解析库
import logging  # 日志记录库
from pathlib import Path  # 路径操作库
from tqdm import tqdm  # 进度条库
import time  # 时间库

# 定义一个用于解析命令行参数的类
class ArgumentParser:
    def __init__(self):
        # 初始化argparse.ArgumentParser对象，设置帮助信息的格式
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        # 调用initialize_parser方法来添加参数
        self.initialize_parser()

    # 定义一个方法来添加命令行参数
    def initialize_parser(self):
        self.parser.add_argument(
            "--filepath",
            required=True,
            type=str,
            help="The filepath to the CSV file containing the original dataset instances.",
        )
        # ... 其他参数的添加 ...

        self.parser.add_argument(
            "--task",
            required=True,
            type=str,
            choices=["cls", "nli", "sum", "xsum"],
            help="The task corresponding to the dataset. "
            "For NLI and classification tasks, label column should be specified. "
            "(Choices: %(choices)s)",
        )
        self.parser.add_argument(
            "--dataset",
            required=True,
            type=str,
            help="Dataset name.",
        )
        self.parser.add_argument(
            "--split",
            required=True,
            type=str,
            choices=["train", "test", "validation"],
            help="Dataset partition. (Choices: %(choices)s)",
        )
        self.parser.add_argument(
            "--model",
            # required=True,
            required=False,
            type=str,
            help="Model name to be evaluated for contamination. "
            "Select an OpenAI model snapshot, such as a version "
            "of GPT-4 or GPT-3.5.",
        )
        self.parser.add_argument(
            "--text_column",
            required=True,
            nargs="+",
            type=str,
            help="Column name for where the replication should be "
            "performed on. For NLI task, provide column name for "
            "sentence1/premise and sentence2/hypothesis, respetively, "
            "separated by a single space.",
        )
        self.parser.add_argument(
            "--label_column",
            type=str,
            default=None,
            help="Column name for labels corresponding to the dataset instances "
            "if the task comes with label.",
        )
        self.parser.add_argument(
            "--should_split_text",
            action="store_true",
            help="Use it to split text randomly. For pre-split text, "
            "ensure it is in two columns named 'first_piece' "
            "and 'second_piece' for single-instance datasets.",
        )
        self.parser.add_argument(
            "--min_p",
            type=float,
            default=40.0,
            help="Specify the minimum percentage for the range "
            "[min_p, max_p], that will randomly determine the length of the "
            "first piece of text.",
        )
        self.parser.add_argument(
            "--max_p",
            type=float,
            default=70.0,
            help="Specify the maximum percentage for the range "
            "[min_p, max_p], that will randomly determine the length of the "
            "first piece of text.",
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Set the seed value upon which random text splits will be based.",
        )
        self.parser.add_argument(
            "--bleurt_eval",
            action="store_true",
            help="If the evaluation should be performed based on the BLEURT score.",
        )
        self.parser.add_argument(
            "--rouge_eval",
            action="store_true",
            help="If the evaluation should be performed based on the ROUGE-L score.",
        )
        self.parser.add_argument(
            "--icl_eval",
            action="store_true",
            help="If the evaluation should be performed based on the GPT-4 ICL prompt.",
        )
        self.parser.add_argument(
            "--process_guided_replication",
            required=True,
            action="store_true",
            help="Whether to perform replication using guided instructions. "
            "If false, guided replication is disabled. When provided without "
            "--process_general_replication, it only performs GPT-4 ICL "
            "evaluation.",
        )
        '''
        是否使用指导性指令进行复制。
        如果为假（false），则禁用指导性复制。当没有提供
        --process_general_replication时，它仅执行GPT-4 ICL评估。
        '''
        self.parser.add_argument(
            "--process_general_replication",
            action="store_true",
            help="Whether to perform replication using general instructions. "
            "If false, general replication is disabled, and therefore, "
            "evaluations based on BLEURT and ROUGE-L cannot be performed unless"
            "'generated_general_completions' and 'generated_guided_completions'"
            "columns are already provided in the csv file.",
        )
        
        '''
        是否使用通用指令进行复制。
        如果为假（false），则禁用通用复制，因此，除非csv文件中已经提供了“generated_general_completions”和“generated_guided_completions”列，否则无法进行基于BLEURT和ROUGE-L的评估。

        '''
        self.parser.add_argument(
            "--experiment",
            type=str,
            required=True,
            help="The name of the experiment. All final results will be saved in this directory.",
        )
    # 定义一个方法来检查text_column参数是否符合要求
    def check_text_column(self, args):
        # 如果任务是NLI并且text_column参数少于2个，则报错
        if args.task == "nli" and len(args.text_column) < 2:
            self.parser.error(
                "For an NLI-based dataset, two columns should be provided, "
                "corresponding to sentence1/premise and sentence2/hypothesis, "
                "respectively, separated by a space."
            )
        # 如果text_column参数超过2个，也报错
        if len(args.text_column) > 2:
            self.parser.error(
                "Exceeded maximum allowed arguments for text_column. "
                "You should provide 1 string for single-instance datasets, "
                "or 2 strings for double-instance datasets, "
                "separated by a space."
            )

    # 定义一个方法来检查label_column参数是否符合要求
    def check_label_column(self, args):
        # 如果任务是NLI或分类任务并且没有提供label_column参数，则报错
        if args.task in ["nli", "cls"] and not args.label_column:
            self.parser.error(
                "The '--label_column' argument is required when the task "
                "is 'nli' or 'cls'."
            )

    # 定义一个方法来检查文本分割参数是否有效
    def check_text_split_params(self, args):
        # 如果最小百分比不在0到100之间，则报错
        if not 0 <= args.min_p <= 100:
            raise self.parser.error("Minimum percentage should be between 0 and 100.")
        # 如果最大百分比不在0到100之间，则报错
        if not 0 <= args.max_p <= 100:
            raise self.parser.error("Maximum percentage should be between 0 and 100.")
        # 如果最小百分比大于最大百分比，则报错
        if args.min_p > args.max_p:
            raise self.parser.error(
                "Minimum percentage should be smaller or equal to "
                "maximum percentage."
            )

    # 定义一个方法来解析命令行参数
    def parse_args(self):
        # 解析命令行参数
        args = self.parser.parse_args()
        # 检查text_column参数
        self.check_text_column(args)
        # 检查label_column参数
        self.check_label_column(args)
        # 检查文本分割参数
        self.check_text_split_params(args)
        # 返回解析后的参数对象
        return args

# 定义一个Guided类，用于生成指导性复制任务的提示模板
class Guided:
    def __init__(self) -> None:
        # 初始化一个字典来存储不同任务类型的提示模板
        self.prompts = {
            "cls": """INSTRUCTION:
You are provided with the FIRST PIECE of an instance from the {split_name} split of the {dataset_name} dataset.
Finish the SECOND PIECE of the instance as EXACTLY appeared in the dataset.
ONLY rely on the original form of the instance in the dataset to finish the SECOND PIECE.

LABEL: {label}

FIRST PIECE:
{first_piece}

SECOND PIECE:
""",
            "nli": """INSTRUCTION:
You are provided with SENTENCE 1 from the {split_name} split of the {dataset_name} dataset.
Finish SENTENCE 2 as appeared in the dataset.
SENTENCE 2 MUST EXACTLY match the instance in the dataset.

SENTENCE 1:
{first_piece}

LABEL: {label}

SENTENCE 2:
""",
            "sum": """INSTRUCTION:
You are provided with the FIRST PIECE of a summary from the {split_name} split of the {dataset_name} dataset.
Finish the SECOND PIECE of the summary as EXACTLY appeared in the dataset.
ONLY rely on the original form of the summary in the dataset to finish the SECOND PIECE.

FIRST PIECE:
{first_piece}

SECOND PIECE:
""",
            "xsum": """INSTRUCTION:
You are provided with the FIRST PIECE of a one-sentence summary from the {split_name} split of the {dataset_name} dataset.
Finish the SECOND PIECE of the summary as EXACTLY appeared in the dataset.
ONLY rely on the original form of the summary in the dataset to finish the SECOND PIECE.

FIRST PIECE:
{first_piece}

SECOND PIECE:
""",
        }

    # 定义一个方法来获取指定任务类型的提示模板
    def get_prompt(self, prompt_type):
        # 返回指定任务类型的提示模板，如果类型无效，则返回错误信息
        return self.prompts.get(prompt_type, "Invalid prompt type")

# 定义一个General类，用于生成通用复制任务的提示模板
class General:
    def __init__(self) -> None:
        # 初始化一个字典来存储不同任务类型的提示模板
        self.prompts = {
            "cls": """INSTRUCTION:
Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single instance with the following LABEL.

LABEL: {label}

FIRST PIECE:
{first_piece}

SECOND PIECE:
""",
            "nli": """INSTRUCTION:
Finish SENTENCE 2 based on SENTENCE 1, such that the following LABEL shows the logical relationship between SENTENCE 1 and SENTENCE 2.

SENTENCE 1:
{first_piece}

LABEL: {label}

SENTENCE 2:
""",
            "sum": """INSTRUCTION:
Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single summary.

FIRST PIECE:
{first_piece}

SECOND PIECE:
""",
            "xsum": """INSTRUCTION:
Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single one-sentence summary.

FIRST PIECE:
{first_piece}

SECOND PIECE:
""",
        }

    # 定义一个方法来获取指定任务类型的提示模板
    def get_prompt(self, prompt_type):
        # 返回指定任务类型的提示模板，如果类型无效，则返回错误信息
        return self.prompts.get(prompt_type, "Invalid prompt type")

# 定义一个ICLEvaluation类，用于生成用于评估复制文本准确性的提示模板
class ICLEvaluation:
    def __init__(self):
        # 初始化一个字典来存储不同评估类型的提示模板
        self.prompts = {
            "qwen_icl_prompt": """INSTRUCTION:
You are provided with a reference text and a candidate text.
The candidate text is a generated replica of the reference text.
Your task is to determine if the candidate text is an exact or near-exact match of the reference text.
Near-exact match candidates must PRECISELY match the reference candidates in terms of sentence structure, overlap, and contextual similarity.
Respond only with 'Yes' or 'No'.
---
Example 1:
...
Example 5:
REFERENCE TEXT:
{reference_text}

CANDIDATE TEXT:
{candidate_text}

ANSWER:
"""
        }

    # 定义一个方法来获取指定评估类型的提示模板
    def get_prompt(self, prompt_type):
        # 返回指定评估类型的提示模板，如果类型无效，则返回错误信息
        return self.prompts.get(prompt_type, "Invalid prompt type")

# 配置并返回一个日志记录器对象
def configure_logger(name):
    logger = logging.getLogger(name)  # 获取名为name的日志记录器
    logger.setLevel(logging.INFO)  # 设置日志记录器的级别为INFO

    ch = logging.StreamHandler()  # 创建一个流处理器
    ch.setLevel(logging.INFO)  # 设置流处理器的级别为INFO
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")  # 定义日志格式
    ch.setFormatter(formatter)  # 为流处理器设置格式

    logger.addHandler(ch)  # 为日志记录器添加流处理器
    logger.propagate = False  # 设置日志不向上传播

    return logger  # 返回配置好的日志记录器

# 创建并配置日志记录器
logger = configure_logger(__name__)

# 定义通义千问客户端类
class TongyiQianwenClient:
    def __init__(self):
        # 初始化通义千问客户端（这里没有具体实现）
        pass

    def get_text(self, text, model="qwen-plus"):  # 定义获取文本的方法，使用默认模型"qwen-plus"
        # 构建API调用的消息列表
        messages = [{'role': 'user', 'content': text}]
        
        print("messages:", messages)  # 打印消息列表
        try:
            # 调用通义千问API生成文本
            response = Generation.call(
                model="qwen-plus",
                messages=messages,
                result_format='message',
                temperature=0.0,
                top_p=1.00
            )
            print("response:", response)  # 打印API响应
        except Exception as e:
            # 如果发生异常，抛出异常
            raise Exception(f"Failed to create completion with Tongyi Qianwen API: {str(e)}")
        
        # 检查响应是否有有效数据
        if response.output and response.output.choices and len(response.output.choices) > 0:
            first_choice = response.output.choices[0]  # 获取第一个选择

            if first_choice.message and first_choice.message.content:  # 如果有消息内容
                return str(first_choice.message.content)  # 返回内容
            else:
                # 如果没有消息内容，抛出异常
                raise Exception(
                    "Response from Tongyi Qianwen API does not "
                    "contain 'message' or 'content'."
                )
        else:
            # 如果响应中没有选择，抛出异常
            raise Exception(
                "Response from Tongyi Qianwen API does not contain "
                "'choices' or choices list is empty."
            )

# 定义实验结果保存类
class ExperimentResultSaver:
    def __init__(self, df, filepath, experiment, save_intermediate_results):
        self.df = df  # 要保存的数据框
        self.filepath = Path(filepath)  # 文件路径
        self.experiment = Path(experiment)  # 实验目录
        self.save_intermediate_results = save_intermediate_results  # 是否保存中间结果

    def check_or_create_experiment_result_folder(self):
        # 如果实验目录不存在，则创建它
        if not self.experiment.exists():
            self.experiment.mkdir(parents=True, exist_ok=True)

    def save_to_csv(self):
        # 如果需要保存中间结果
        if self.save_intermediate_results:
            self.check_or_create_experiment_result_folder()  # 检查或创建实验目录
            csv_filepath = self.experiment / self.filepath.name  # 构建CSV文件路径
            self.df.to_csv(  # 保存数据框到CSV文件
                csv_filepath,
                encoding="utf-8",
                index=False,
            )
            logger.info(f"File saved to: {csv_filepath}")  # 记录保存的文件路径

# 定义复制阶段类，继承自实验结果保存类
class ReplicationPhase(ExperimentResultSaver):
    def __init__(self, df, args, instruction, save_intermediate_results):
        # 调用父类的初始化方法
        super().__init__(df, args.filepath, args.experiment, save_intermediate_results)
        self.df = df  # 要处理的数据框
        self.args = args  # 命令行参数
        self.instruction = instruction  # 指令对象
        self.instruction_type = str(instruction.__class__.__name__).lower()  # 指令类型
        self.generated_text_column = f"generated_{self.instruction_type}_completion"  # 生成的文本列名
        self.openai_client = TongyiQianwenClient()  # 通义千问客户端对象

    def split_text(self):
        # 如果任务是NLI或者数据框中已经包含分割的文本列，则返回
        if self.args.task == "nli" or all(
            item in self.df.columns for item in ["first_piece", "second_piece"]
        ):
            return
        # 如果任务不是NLI，数据框中不包含分割的文本列，且没有指定随机分割文本，则抛出异常
        elif (
            self.args.task != "nli"
            and not all(
                item in self.df.columns for item in ["first_piece", "second_piece"]
            )
            and not self.args.should_split_text
        ):
            raise ValueError(
                "For generating completions for single-instance datasets, "
                "the text must be splitted randomly. If you have pre-split "
                "text, ensure they are listed as 'first_piece' and "
                "'second_piece' columns in the csv file. Otherwise, you can "
                "get the text splitted by running --should_split_text."
            )

        # 如果需要随机分割文本，则分割文本并保存到数据框中
        self.df[["first_piece", "second_piece"]] = (
            self.df[self.args.text_column[0]]
            .apply(
                split_text_randomly,
                min_p=self.args.min_p,
                max_p=self.args.max_p,
                seed=self.args.seed,
            )
            .apply(pd.Series)
        )

    def process(self):
        # 记录开始复制过程的日志
        logger.info(f"Starting {self.instruction_type} replication process ...")

        self.split_text()  # 分割文本

        # 使用进度条遍历数据框中的每一行
        with tqdm(total=len(self.df), desc="Generating completions") as pbar:
            for index, row in self.df.iterrows():
                self._perform_task(index, row)  # 执行任务
                pbar.update(1)  # 更新进度条
                time.sleep(3)  # 等待3秒

            pbar.close()  # 关闭进度条
            self.save_to_csv()  # 保存到CSV文件

        return self.df  # 返回处理后的数据框

    def _perform_task(self, index, row):
        # 获取指令模板
        prompt = self.instruction.get_prompt(self.args.task)
        first_piece = (
            row[self.args.text_column[0]]
            if self.args.task == "nli"
            else row["first_piece"]
        )

        # 准备指令
        formatted_prompt = self._prepare_prompt(prompt, row, first_piece)

        if index == 0:
            logger.info(f"Input prompt:\n\n{formatted_prompt}")  # 记录输入的指令

        # 使用通义千问客户端获取文本
        self.df.at[index, self.generated_text_column] = self.openai_client.get_text(
            text=formatted_prompt, model=self.args.model
        )

    def _prepare_prompt(self, prompt, row, first_piece):
        # 如果有标签列，则格式化指令模板，包含标签
        if self.args.label_column:
            formatted_prompt = prompt.format(
                split_name=self.args.split,
                dataset_name=self.args.dataset,
                label=str(row[self.args.label_column]),
                first_piece=str(first_piece),
            )
        else:
            # 如果没有标签列，则格式化指令模板，不包含标签
            formatted_prompt = prompt.format(
                split_name=self.args.split,
                dataset_name=self.args.dataset,
                first_piece=str(first_piece),
            )
        return formatted_prompt  # 返回格式化后的指令模板

# 定义ICL类，用于评估文本的相似度
class ICL:
    def __init__(self):
        # 初始化ICL评估对象和通义千问客户端对象
        self.icl_eval = ICLEvaluation()
        self.qianwen_client = TongyiQianwenClient()

    # 定义评分方法，用于评估参考文本和候选文本的相似度
    def score(self, reference, candidate, model="qwen-plus", prompt_type="qwen_icl_prompt"):
        # 获取ICL提示模板
        icl_prompt = self.icl_eval.get_prompt(prompt_type=prompt_type)
        # 格式化提示模板，填充参考文本和候选文本
        formatted_icl_prompt = icl_prompt.format(reference_text=reference, candidate_text=candidate)
        # 调用通义千问客户端，生成评估结果
        evaluation = self.qianwen_client.get_text(text=formatted_icl_prompt, model=model)
        # 打印模型、格式化的提示和评估结果，用于调试
        print("model:", model)
        print("formatted_icl_prompt:", formatted_icl_prompt)
        print("evaluation:", evaluation)
        # 返回评估结果
        return evaluation

# 定义PatternCounter类，用于统计评估结果中的模式
class PatternCounter:
    def __init__(self, evaluations, pattern_severity):
        # 初始化评估结果列表和模式严重性字典
        self.evaluations = evaluations
        self.pattern_severity = pattern_severity

    # 定义统计模式的方法
    def count_patterns(self):
        # 初始化模式计数字典
        counts = {pattern: 0 for pattern in self.pattern_severity.keys()}
        # 遍历评估结果，统计模式出现次数
        for evaluation in self.evaluations:
            for pattern in self.pattern_severity.keys():
                if re.search(pattern, evaluation):
                    counts[pattern] += 1
        # 返回模式计数字典
        return counts

    # 定义保存结果的方法
    def save_results(self, result_filepath, counts):
        # 打开文件，准备写入结果
        with open(result_filepath, "w") as f:
            # 写入标题行
            f.write(f"{'Metric':<15} {'Match Type':<30}{'Count':<10} {'Contaminated'}\n")
            f.write(f"{'-' * 75}\n")
            # 写入模式计数和是否污染的判断结果
            for pattern, count in counts.items():
                f.write(f"{'GPT-4 ICL:':<15} {pattern:<30}{count:<10} {'Yes' if count >= self.pattern_severity[pattern] else 'No'}\n")

    # 定义评估并保存结果的方法
    def evaluate_and_save_results(self, result_filepath):
        # 统计模式
        counts = self.count_patterns()
        # 保存结果
        self.save_results(result_filepath, counts)

# 定义Alg2EvalPhase类，用于执行评估阶段并保存结果
class Alg2EvalPhase(ExperimentResultSaver):
    def __init__(self, df, args, scorer, pattern_severity, save_intermediate_results):
        # 调用父类构造函数
        super().__init__(df, args.filepath, args.experiment, save_intermediate_results)
        # 初始化数据框、参数、评分器、模式严重性和文件路径
        self.df = df
        self.args = args
        self.scorer = scorer
        self.pattern_severity = pattern_severity
        self.filepath = Path(self.args.filepath)

    # 定义评估方法
    def evaluate(self):
        # 记录开始评估的日志
        logger.info("Starting evaluation using Tongyi Qianwen ICL ...")
        # 检查数据框中是否包含必要的列
        if "generated_guided_completion" not in self.df.columns:
            raise ValueError("For evaluation using BLEURT, completions from guided instructions must be provided...")
        # 使用进度条遍历数据框中的每一行
        with tqdm(total=len(self.df), desc="Generating evaluations") as pbar:
            for index, row in self.df.iterrows():
                # 获取参考文本和候选文本
                reference = str(row[self.args.text_column[1]]) if self.args.task == "nli" else str(row["second_piece"])
                candidate = str(row["generated_guided_completion"])
                # 如果是第一行，记录参考文本和候选文本，用于检查
                if index == 0:
                    logger.info(f"Example of reference text: {reference}")
                    logger.info(f"Example of guided completion: {candidate}")
                # 使用评分器评分
                icl_evaluation = self.scorer.score(reference=reference, candidate=candidate)
                # 将评分结果保存到数据框中
                self.df.at[index, "qwen_icl_evaluation"] = icl_evaluation
                # 更新进度条
                pbar.update(1)
                time.sleep(3)
        # 关闭进度条
        pbar.close()
        # 保存数据框到CSV文件
        self.save_to_csv()
        # 创建PatternCounter对象，用于统计和保存评估结果
        pattern_counter = PatternCounter(
            evaluations=list(self.df["qwen_icl_evaluation"]),
            pattern_severity=self.pattern_severity,
        )
        # 构建结果文件路径
        result_filepath = self.experiment / f"qwen_icl_evaluations_for_{self.filepath.stem}.txt"
        # 统计和保存评估结果
        pattern_counter.evaluate_and_save_results(result_filepath=result_filepath)
        # 返回处理后的数据框
        return self.df

# 程序的主入口
if __name__ == "__main__":
    # 解析命令行参数
    args = ArgumentParser().parse_args()
    # 读取CSV文件到数据框
    df = pd.read_csv(args.filepath, encoding="utf-8")
    # 执行复制任务，使用指导性指令
    df = ReplicationPhase(df=df, args=args, instruction=Guided(), save_intermediate_results=True).process()
    # 执行复制任务，使用通用指令
    df = ReplicationPhase(df=df, args=args, instruction=General(), save_intermediate_results=True).process()
    # 执行评估任务
    df = Alg2EvalPhase(df=df, args=args, scorer=ICL(), pattern_severity={"Yes (exact match)": 1, "Yes (near-exact match)": 2}, save_intermediate_results=True).evaluate()
