import logging
import math
import os
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from utils import convert_to_wav_tensor, does_end_with_speech_token
from utils import does_start_with_speech_token, find_prompt_last_speech_start_position, get_forbidden_tokens
from speech_tokenizer import spiritlm_base, spiritlm_expressive
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, set_seed


# 从环境变量中获取基础检查点目录，如果未设置，则使用默认路径
base_checkpoints_dir = Path(os.getenv("SPIRITLM_CHECKPOINTS_DIR", Path(__file__).parent.parent.parent / "checkpoints"))

# 在基础路径后追加 'spiritlm_model'，得到最终的检查点目录
CHECKPOINT_DIR = base_checkpoints_dir / "spiritlm_model"


class ContentType(Enum):
    """
    内容类型枚举类。

    该类定义了生成任务中内容可能具有的类型。
    """
    TEXT = "TEXT"
    SPEECH = "SPEECH"


class OutputModality(Enum):
    """
    输出模态枚举类。

    该类定义了生成任务中输出可能具有的模态。
    """
    TEXT = auto()
    SPEECH = auto()
    ARBITRARY = auto()


@dataclass
class GenerationInput:
    """
    生成任务输入数据类。

    该类用于表示生成任务的输入数据，包括内容及其类型。
    """
    # 输入内容，可以是字符串、文件路径、张量或 numpy 数组
    content: Union[str, os.PathLike, torch.Tensor, np.ndarray]
    # 输入内容类型，必须是 ContentType 枚举中的一个
    content_type: ContentType

    @classmethod
    def from_tuple(cls, tup):
        """
        从元组创建 GenerationInput 实例的类方法。

        参数:
            tup (tuple): 包含内容类型和内容的元组。

        返回:
            GenerationInput: 创建的 GenerationInput 实例。

        异常:
            AssertionError: 如果内容类型不在预期范围内，则抛出断言错误。
        """
        content_type, content = tup
        # 将内容类型转换为大写
        content_type = content_type.upper()
        assert content_type in [
            "SPEECH",
            "TEXT",
        ], f"expects content_type to be one of ['SPEECH', 'TEXT'], found '{content_type}'"
        if content_type == "TEXT":
            content_type = ContentType.TEXT # 设置为文本类型
        elif content_type == "SPEECH":
            content_type = ContentType.SPEECH # 设置为语音类型
        return cls(content=content, content_type=content_type)


@dataclass
class GenerationOuput:
    """
    生成任务输出数据类。

    该类用于表示生成任务的输出数据，包括内容及其类型。
    """
    content: Union[str, np.ndarray]  # 输出内容，可以是字符串或 numpy 数组
    content_type: ContentType  # 输出内容类型，必须是 ContentType 枚举中的一个


# 输入内容列表，类型为 GenerationInput 的列表
InterleavedInputs = List[GenerationInput]
# 输出内容列表，类型为 GenerationOuput 的列表
InterleavedOutputs = List[GenerationOuput]


_logger = logging.getLogger(__name__)


# 定义 SpiritLM 模型的不同变体，使用 Enum 枚举类
class SpiritlmVariants(Enum):
    """
    SpiritLM 模型的不同变体。

    该枚举类定义了 SpiritLM 模型的不同变体，包括基础版本和表现力版本。
    """
    BASE_7B = "spirit-lm-base-7b"
    EXPRESSIVIE_7B = "spirit-lm-expressive-7b"

    @classmethod
    def values_as_list(cls):
        """
        获取所有变体名称的列表。

        返回:
            List[str]: 包含所有变体名称的列表。
        """
        return [e.value for e in cls]


def _ensure_model_name(name: str):
    """
    确保模型名称有效。

    该函数检查给定的模型名称是否有效。如果提供的名称是一个存在的文件路径，则提取文件名部分进行比较。

    参数:
        name (str): 模型名称或文件路径。

    异常:
        AssertionError: 如果模型名称不在预期的变体列表中，则抛出断言错误。
    """
    if Path(name).exists():
        name = Path(name).stem
    expected_names = SpiritlmVariants.values_as_list()
    assert (
        name in SpiritlmVariants.values_as_list()
    ), f"Unknown model name, expected one of {expected_names}"


def _set_device_and_return():
    """
    设置设备并返回设备对象。

    该函数检查 CUDA 是否可用，并设置设备为 GPU 或 CPU。如果 CUDA 可用，则根据环境变量 LOCAL_RANK 设置设备。

    返回:
        torch.device: 设置的设备对象。
    """
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device(local_rank)


def _convert_str_output_modality(output_modality):
    """Convert from string to an instance of OutputModality"""
    """
    将字符串转换为 OutputModality 枚举实例。

    该函数将表示输出模态的字符串转换为 OutputModality 枚举的实例。

    参数:
        output_modality (str): 输出模态的字符串表示。

    返回:
        OutputModality: 对应的 OutputModality 枚举实例。

    异常:
        AssertionError: 如果字符串不是有效的输出模态，则抛出断言错误。
    """
    output_modality_str_map = {
        "TEXT": OutputModality.TEXT,
        "SPEECH": OutputModality.SPEECH,
        "ARBITRARY": OutputModality.ARBITRARY,
    }
    if isinstance(output_modality, str):
        output_modality = output_modality.upper()
        assert (
            output_modality in output_modality_str_map
        ), f"invalid string output_modality (found '{output_modality}', but expects one of {list(output_modality_str_map)})"
        # 转换为对应的枚举实例
        output_modality = output_modality_str_map[output_modality]
    assert isinstance(output_modality, OutputModality)
    return output_modality


def _get_generation_inputs(interleaved_inputs):
    """Convert from a list of tuple (content_type, content) to a list of GenrationInput"""
    """
    将输入列表转换为生成任务的输入格式。

    该函数将一个包含元组或 GenerationInput 实例的列表转换为仅包含 GenerationInput 实例的列表。
    如果列表中的元素是元组，则将其转换为 GenerationInput 实例。

    参数:
        interleaved_inputs (List[Union[Tuple, GenerationInput]]): 输入列表，包含元组或 GenerationInput 实例。

    返回:
        List[GenerationInput]: 转换后的仅包含 GenerationInput 实例的列表。

    异常:
        AssertionError: 如果列表中的元素既不是元组也不是 GenerationInput 实例，则抛出断言错误。
    """
    for i, item in enumerate(interleaved_inputs):
        assert isinstance(item, tuple) or isinstance(item, GenerationInput), (
            "Each element of interleaved_inputs is expected to be either an instance of GenerationInput "
            "or a tuple of (content_modality, content)"
        )
        if isinstance(item, tuple):
            # 将元组转换为 GenerationInput 实例
            interleaved_inputs[i] = GenerationInput.from_tuple(interleaved_inputs[i])
    return interleaved_inputs


def _overwrite_generation_config(generation_config, kwargs):
    """Overwrite generation_config from the kwargs"""
    """
    根据传入的参数覆盖生成配置。

    该函数根据传入的关键字参数覆盖生成配置。如果未提供生成配置，则创建一个默认的生成配置实例。

    参数:
        generation_config (Optional[GenerationConfig]): 原始生成配置，可以为 None。
        **kwargs: 关键字参数，用于覆盖生成配置中的属性。

    返回:
        GenerationConfig: 覆盖后的生成配置实例。

    异常:
        AssertionError: 如果提供的关键字参数不在生成配置的属性中，则抛出断言错误。
    """
    if generation_config is None:
        # 如果未提供生成配置，则创建一个默认实例
        generation_config = GenerationConfig()
    assert isinstance(generation_config, GenerationConfig)
    # 获取生成配置的不同字典表示
    gen_diff_dict = generation_config.to_diff_dict()
    for attr_name, attr_value in kwargs.items():
        assert hasattr(
            generation_config, attr_name
        ), f"attribute '{attr_name}' not found in transformers.GenerationConfig"
        if attr_name in gen_diff_dict and attr_value != gen_diff_dict[attr_name]:
            _logger.warning(
                f"Overwrite generation_config's {attr_name} to {attr_value}"
            )
        setattr(generation_config, attr_name, attr_value)
    return generation_config


class Spiritlm:
    """
    SpiritLM 模型类。

    该类实现了 SpiritLM 模型，包括初始化、构建提示和生成禁止标记的功能。
    """
    # 文本提示前缀
    TEXT_PROMPT_PREFIX = "[Text]"
    # 语音提示前缀
    SPEECH_PROMPT_PREFIX = "[Speech]"

    def __init__(self, name: str, **speech_tokenizer_kwargs):
        """
        初始化 SpiritLM 模型。

        参数:
            name (str): 模型名称或模型检查点路径。
            **speech_tokenizer_kwargs: 语音分词器的关键字参数。
        """
        if Path(name).exists():
            # 如果提供的名称是一个存在的路径，则直接使用
            path = name
        else:
            # 否则，拼接基础检查点目录
            path = CHECKPOINT_DIR / name
        _ensure_model_name(name)
        self.device = _set_device_and_return()
        _logger.info(f"Loading SPIRIT-LM model from the path {path}...")
        self.model = LlamaForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16 # 从指定路径加载模型，并设置数据类型为 bfloat16
        ).to(self.device)
        _logger.info(f"SPIRIT-LM model is loaded.")
        self.tokenizer = LlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path=path, # 从指定路径加载分词器
            add_bos_token=True, # 添加起始标记
            add_eos_token=False, # 不添加结束标记
        )
        _logger.info("Loading SPIRIT-LM speech tokenizers ...")
        if name == SpiritlmVariants.BASE_7B.value:
            # 加载基础版本的语音分词器
            self.speech_tokenizer = spiritlm_base(**speech_tokenizer_kwargs)
            self.is_expressive_model = False
        elif name == SpiritlmVariants.EXPRESSIVIE_7B.value:
            # 加载表现力版本的语音分词器
            self.speech_tokenizer = spiritlm_expressive(**speech_tokenizer_kwargs)
            self.is_expressive_model = True
        _logger.info("SPIRIT-LM speech tokenizers are loaded.")

    def _build_prompt(
        self,
        generation_inputs: List[GenerationInput],
        output_modality: OutputModality,
    ) -> str:
        """
        Build the prompt according the input content and the output modality.
        """
        """
        根据输入内容和输出模态构建提示。

        参数:
            generation_inputs (List[GenerationInput]): 输入内容列表。
            output_modality (OutputModality): 输出模态。

        返回:
            str: 构建的提示字符串。

        异常:
            ValueError: 如果输出模态或内容类型未知，则抛出值错误。
        """
        if not isinstance(output_modality, OutputModality):
            # 如果输出模态不是 OutputModality 的实例，则抛出错误
            raise ValueError(f"Unknown output_modality: {output_modality}")
        
        # 初始化提示列表
        prompts = []
        # 初始化上一个模态
        prev_modality = None
        for gen_input in generation_inputs:
            if gen_input.content_type.value == ContentType.SPEECH.value:
                # 将内容转换为 WAV 张量
                gen_input.content = convert_to_wav_tensor(gen_input.content)
                if prev_modality != "s":
                    # 如果上一个模态不是语音，则添加语音提示前缀
                    prompts.append(Spiritlm.SPEECH_PROMPT_PREFIX)
                # 添加语音分词结果
                prompts.append(self.speech_tokenizer(gen_input.content))
                # 更新上一个模态为语音
                prev_modality = "s"  # speech

            elif gen_input.content_type.value == ContentType.TEXT.value:
                if prev_modality != "t":
                    # 如果上一个模态不是文本，则添加文本提示前缀
                    prompts.append(Spiritlm.TEXT_PROMPT_PREFIX)
                # 添加文本内容
                prompts.append(gen_input.content)
                # 更新上一个模态为文本
                prev_modality = "t"  # text
            else:
                raise ValueError(
                    f"Unknown content type: {gen_input.content_type.value}"
                )
            
        if output_modality == OutputModality.TEXT:
            if prev_modality != "t":
                # 如果输出模态是文本，并且上一个模态不是文本，则添加文本提示前缀
                prompts.append(Spiritlm.TEXT_PROMPT_PREFIX)
        elif output_modality == OutputModality.SPEECH:
            if prev_modality != "s":
                # 如果输出模态是语音，并且上一个模态不是语音，则添加语音提示前缀
                prompts.append(Spiritlm.SPEECH_PROMPT_PREFIX)

        # 将提示列表连接为字符串并返回
        return "".join(prompts)

    @cache
    def _build_forbidden_tokens(
        self,
        output_modality: OutputModality,
    ) -> List[int]:
        """
        Build a set of token ids that we don't want to generate according the modality direction.

        For instance, when the modality direction is speech to text (S2T), i.e., we continue
        generating text given a speech prompt, we want that the output contains only the text tokens.
        """
        """
        根据输出模态构建禁止生成的标记列表。

        例如，当输出模态是语音到文本（S2T）时，即给定语音提示继续生成文本，我们希望输出只包含文本标记。

        参数:
            output_modality (OutputModality): 输出模态。

        返回:
            List[int]: 禁止生成的标记列表。

        异常:
            ValueError: 如果输出模态未知，则抛出值错误。
        """
        if output_modality == OutputModality.TEXT:
            forbidden_tokens = get_forbidden_tokens(
                ban_special_tokens=True, # 禁止特殊标记
                generate_only_text=True, # 仅生成文本
                ban_expressivity_tokens=True if self.is_expressive_model else False, # 如果是表现力模型，则禁止表现力标记
            )
        elif output_modality == OutputModality.SPEECH:
            forbidden_tokens = get_forbidden_tokens(
                ban_special_tokens=True, # 禁止特殊标记
                generate_only_speech=True, # 仅生成语音
            )
        elif output_modality == OutputModality.ARBITRARY:
            forbidden_tokens = [] # 如果是任意模态，则不禁止任何标记
        else:
            # 如果输出模态未知，则抛出错误
            raise ValueError(f"Unknown output_modality: {output_modality}")
        # 返回禁止标记列表
        return forbidden_tokens

    def _parse_speech_and_text(
        self,
        generated_content: str,
    ):
        """
        解析生成的文本内容，将语音和文本部分分离出来。

        该函数遍历生成的文本内容，识别出语音标记（如 [Hu1]、[Pi1]、[St1]）和文本标记（如 [Text]）。
        根据标记将内容分割为语音和文本两部分，并返回一个包含分割结果的列表。

        参数:
            generated_content (str): 生成的文本内容。

        返回:
            List[Tuple[str, str]]: 分割后的语音和文本内容列表，每个元素是一个元组 (内容, 类型)。
                                   类型可以是 's'（语音）或 't'（文本）。
        """
        
        splits = []  # 初始化分割结果列表
        i = 0  # 初始化当前索引
        last_pos = len(generated_content)  # 获取内容总长度
        char_and_types = []  # 初始化字符和类型列表，用于临时存储当前部分的内容和类型
        is_speech_token = False  # 标记当前是否在语音标记内
        is_text_token = False  # 标记当前是否在文本标记内
        text_prefix_length = len(Spiritlm.TEXT_PROMPT_PREFIX)  # 文本提示前缀的长度
        speech_prefix_length = len(Spiritlm.SPEECH_PROMPT_PREFIX)  # 语音提示前缀的长度

        while i < last_pos:
            ch = generated_content[i]
            j = i
            # 如果当前字符是 '['，则可能是标记的开始
            if ch == "[":
                if (
                    j + text_prefix_length - 1 < last_pos
                    and generated_content[j : j + text_prefix_length]
                    == Spiritlm.TEXT_PROMPT_PREFIX
                ):  # text prefix token
                    j += text_prefix_length  # skip "[Text]
                elif (
                    j + speech_prefix_length - 1 < last_pos
                    and generated_content[j : j + speech_prefix_length]
                    == Spiritlm.SPEECH_PROMPT_PREFIX
                ):  # speech prefix token
                    j += speech_prefix_length  # skip "[Speech]"
                elif j + 2 < last_pos and generated_content[j + 1 : j + 3] in (
                    "Hu",
                    "Pi",
                    "St",
                ):
                    j += 3  # skip "["" and Hu/Pi/St
                    while j < last_pos and generated_content[j] != "]":
                        j += 1
                    j += 1  # skip "]"
                    is_speech_token = True
                else:  # other texts starting with "[" e.g., "[abc"
                    is_text_token = True
                    j += 1
            else:
                is_text_token = True
                while j < last_pos and generated_content[j] != "[":
                    j += 1

            # 获取当前部分的内容
            cur_content = generated_content[i:j]
            if is_speech_token:
                if len(char_and_types) and char_and_types[-1][1] == "t":
                    # 如果之前的部分类型是文本，则将之前的部分添加到分割结果中
                    splits.append(
                        (
                            "".join(
                                (
                                    content_and_type[0]
                                    for content_and_type in char_and_types
                                )
                            ),
                            "t",
                        )
                    )
                    char_and_types = []
                char_and_types.append((cur_content, "s"))  # speech
            elif is_text_token:
                if len(char_and_types) and char_and_types[-1][1] == "s":
                    # 如果之前的部分类型是语音，则将之前的部分添加到分割结果中
                    splits.append(
                        (
                            "".join(
                                (
                                    content_and_type[0]
                                    for content_and_type in char_and_types
                                )
                            ),
                            "s",
                        )
                    )
                    char_and_types = []
                char_and_types.append((cur_content, "t"))  # text
            is_speech_token, is_text_token = False, False
            i = j

        if len(char_and_types):
            if char_and_types[-1][1] == "t":
                # 处理最后一部分内容
                splits.append(
                    (
                        "".join(
                            (content_and_type[0] for content_and_type in char_and_types)
                        ),
                        "t",
                    )
                )
            else:
                splits.append(
                    (
                        "".join(
                            (content_and_type[0] for content_and_type in char_and_types)
                        ),
                        "s",
                    )
                )
        # 返回分割结果
        return splits

    def _decode_from_generated_output(
        self,
        output_modality: OutputModality,
        generated_content: str,
        prompt: str,
        speaker_id: int = 2,
    ) -> InterleavedOutputs:
        """
        根据输出模态解码生成的标记内容。

        如果输出是文本，则直接返回。
        如果输出是语音，则使用语音分词器解码语音标记。
        如果输出是任意模态，则根据其模态解码生成的内容。

        参数:
            output_modality (OutputModality): 输出模态。
            generated_content (str): 生成的内容。
            prompt (str): 提示文本。
            speaker_id (int, 可选): 说话人 ID，默认为 2。

        返回:
            List[GenerationOuput]: 解码后的生成输出列表。
        """

        def _decode(
            modality: OutputModality,
            gen: str,
        ) -> InterleavedOutputs:
            """
            根据模态解码生成的内容。

            参数:
                modality (OutputModality): 输出模态。
                gen (str): 生成的内容。

            返回:
                List[GenerationOuput]: 解码后的生成输出列表。
            """
            if modality == OutputModality.TEXT:
                return [
                    GenerationOuput(
                        content=gen,
                        content_type=ContentType.TEXT,
                    )
                ]
            elif modality == OutputModality.SPEECH:
                return [
                    GenerationOuput(
                        content=self.speech_tokenizer.decode(
                            gen, speaker_id=speaker_id
                        ),
                        content_type=ContentType.SPEECH,
                    )
                ]
            elif modality == OutputModality.ARBITRARY:
                # 初始化解码块列表
                decoded_chunks = []
                for i, (chunk_content, chunk_modality) in enumerate(
                    self._parse_speech_and_text(gen)
                ):
                    if chunk_modality == "s":
                        # 计算内容中的 Hubert 标记数量
                        nb_content_hubert_tokens = len(chunk_content.split("[Hu"))
                        # 解码语音内容
                        decoded = _decode(
                            modality=OutputModality.SPEECH,
                            gen=chunk_content,
                        )[0]
                        if i == 0 and is_last_content_speech:
                            # 当提示以语音结束，并且生成内容以语音开始时的边缘情况
                            nb_prompt_hubert_tokens = (
                                len(prompt[last_speech_start_pos:].split("[Hu")) - 1
                            )  # 计算提示中最后一个语音标记后的 Hubert 标记数量，减去前缀中的一个
                            if nb_content_hubert_tokens - nb_prompt_hubert_tokens < 25:
                                # 如果继续的语音内容太短，则跳过
                                continue
                            # 从生成内容中删除提示部分
                            prompt_ratio = (
                                nb_prompt_hubert_tokens / nb_content_hubert_tokens
                            )
                            decoded.content = decoded.content[
                                math.ceil(decoded.content.size * prompt_ratio) :
                            ]
                        elif i > 0 and nb_content_hubert_tokens < 25:
                            # 如果新的语音内容太短，则跳过
                            continue
                    else:
                        # 解码文本内容
                        decoded = _decode(
                            modality=OutputModality.TEXT,
                            gen=chunk_content,
                        )[0]
                    # 添加解码后的内容到列表中
                    decoded_chunks.append(decoded)
                # 返回解码后的内容列表
                return decoded_chunks
            else:
                # 如果输出模态未知，则抛出错误
                raise ValueError(f"Unknown output_modality: {output_modality}")

        # 去除提示部分，获取新生成的内容
        generated_new_content = generated_content[len(prompt) :].strip()
        is_last_content_speech, last_speech_start_pos = False, 0
        if (
            output_modality == OutputModality.ARBITRARY
            and does_end_with_speech_token(prompt)
            and does_start_with_speech_token(generated_new_content)
        ):
            # 如果提示以语音结束，并且生成内容以语音开始，则标记为最后的内容是语音
            is_last_content_speech = True
            # 查找提示中最后一个语音标记的位置
            last_speech_start_pos = find_prompt_last_speech_start_position(prompt)
            # 如果提示以语音结束，则解码提示和生成内容，因为我们可能在生成内容中没有音高和风格标记
            # 更新生成内容为从最后一个语音标记开始
            generated_new_content = generated_content[last_speech_start_pos:]
        # 返回解码后的生成内容
        return _decode(output_modality, generated_new_content)

    def generate(
        self,
        interleaved_inputs: Optional[List[Union[GenerationInput, tuple]]] = None,
        prompt: Optional[str] = None,
        output_modality: Union[OutputModality, str] = OutputModality.ARBITRARY,
        generation_config: Optional[GenerationConfig] = None,
        force_tokens_to_output_modality: bool = True,
        speaker_id: int = 2,
        return_prompt: bool = False,
        seed: Optional[int] = None,
        **kwargs,  # GenerationConfig args can be passing here
    ) -> Union[InterleavedOutputs, Tuple[InterleavedOutputs, str]]:
        """
        Speech/text generation given speech/text prompt.

        Parameters:
            interleaved_inputs (List of `GenerationInput` or list of tuples):
                List of speech/text inputs.
                Each element can be an instance of `GenerationInput` or a tuple of (content_type, content)
                Text content is string; Speech content is either audio path, audio tensor, or nummpy array.
                The prompt will be built by interleaving them in order.
            prompt (str):
                The prompt in encoded tokens string,
                e.g., "[Speech][Hu99][Hu38]...", "[Text]whatever text" or mix of speech & text.
            output_modality (str or `OutputModality`):
                'TEXT' or OutputModality.TEXT: generate text
                'SPEECH' or OutputModality.SPEECH: generate speech
                'ARBITRARY' or OutputModality.ARBITRARY: generate arbitrary modality output (default)
            generation_config (`GenerationConfig`):
                Generation configuration used by Huggingface `generate` function.
            force_tokens_to_output_modality (bool):
                Whether to force generating tokens to the output modality that you specify in `output_modality`.
                For instance, if the `output_modality` is TEXT and force_tokens_to_output_modality is True,
                we force the model to generate only the text tokens.
            speaker_id (int):
                Speaker id, 0, 1, 2 or 3.
            return_prompt (bool):
                Whether to return the constructed prompt (could be used for debug).
            **kwargs:
                Directly passing arguments from transformers.GenerationConfig (e.g. temperature, max_new_tokens, do_sample).
                See: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
        """
        """
        根据语音或文本提示生成文本或语音内容。

        参数:
            interleaved_inputs (List of `GenerationInput` 或 list of tuples):
                输入内容列表。每个元素可以是 `GenerationInput` 实例或 (内容类型, 内容) 的元组。
                文本内容是字符串；语音内容可以是音频路径、音频张量或 numpy 数组。
                提示将通过按顺序交错这些输入来构建。
            prompt (str):
                编码标记字符串的提示，例如 "[Speech][Hu99][Hu38]..."、"[Text]whatever text" 或语音和文本的混合。
            output_modality (str 或 `OutputModality`):
                'TEXT' 或 OutputModality.TEXT: 生成文本
                'SPEECH' 或 OutputModality.SPEECH: 生成语音
                'ARBITRARY' 或 OutputModality.ARBITRARY: 生成任意模态的输出（默认）
            generation_config (`GenerationConfig`):
                Huggingface `generate` 函数使用的生成配置。
            force_tokens_to_output_modality (bool):
                是否强制生成指定在 `output_modality` 中指定的输出模态的标记。
                例如，如果 `output_modality` 是 TEXT 并且 force_tokens_to_output_modality 是 True，
                我们强制模型只生成文本标记。
            speaker_id (int):
                说话人 ID，0、1、2 或 3。
            return_prompt (bool):
                是否返回构建的提示（可用于调试）。
        """

        if seed is not None:
            _logger.info(f"Set seed to {seed}")
            set_seed(seed)

        # Set the output modality
        # 设置输出模态
        output_modality = _convert_str_output_modality(output_modality)

        # Get the input prompt
        # 获取输入提示
        assert not (
            interleaved_inputs is None and prompt is None
        ), "interleaved_inputs and prompt can not both be None"
        if (
            prompt is not None
            and interleaved_inputs is not None
            and len(interleaved_inputs) > 0
        ):
            _logger.warning(
                "When prompt is specified, interleaved_inputs will not be used."
            )
        if prompt is None:
            if not isinstance(interleaved_inputs, list):
                interleaved_inputs = [interleaved_inputs]
            interleaved_inputs = _get_generation_inputs(interleaved_inputs)
            prompt = self._build_prompt(
                interleaved_inputs,
                output_modality,
            )

        # Get input tensor
        # 获取输入张量
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Get generation config from kwargs
        generation_config = _overwrite_generation_config(generation_config, kwargs)

        # Get forbidden token ids
        if (
            force_tokens_to_output_modality
            and output_modality != OutputModality.ARBITRARY
        ):
            forbidden_token_ids = [
                [tok_id] for tok_id in self._build_forbidden_tokens(output_modality)
            ]
        else:
            forbidden_token_ids = None

        # Perform the generation
        # 执行生成
        generate_ids = self.model.generate(
            **inputs,
            generation_config=generation_config,
            bad_words_ids=forbidden_token_ids,
            pad_token_id=-1,
        )

        # Decode the output
        # 解码输出
        gen = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        try:
            decoded_output = self._decode_from_generated_output(
                output_modality=output_modality,
                generated_content=gen,
                prompt=prompt,
                speaker_id=speaker_id,
            )
        except Exception as e:
            _logger.error(f"Fail to decode the content: {gen[len(prompt) :].strip()}")
            raise e

        if return_prompt:
            return decoded_output, prompt
        else:
            return decoded_output
