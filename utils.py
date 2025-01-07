import os
import re
from io import BytesIO
from typing import List, Optional, Union

import numpy as np
import torch
import torchaudio


EXPECTED_SAMPLING_RATE = 16_000


def find_prompt_last_speech_start_position(prompt: str) -> Optional[int]:
    """
    在提示文本中查找最后一个语音标记的开始位置。

    该函数通过从右到左搜索提示文本，查找最后一个语音标记（如 ']d+tS['、']d+iP[' 或 ']d+uH['）的位置。
    找到后，返回该标记在原始提示文本中的开始索引。如果未找到，则返回 None。

    参数:
        prompt (str): 提示文本字符串。

    返回:
        Optional[int]: 最后一个语音标记的开始索引，如果未找到则返回 None。
    """
    # 初始化上一个匹配结束位置
    prev_end = None
    # revert the prompt so we can search from right to left, the speech token patterns are also reverted.
    # 反转提示文本，以便从右到左搜索，语音标记的模式也会被反转
    for match in re.finditer("(\]\d+uH\[)|(\]\d+iP\[)|(\]\d+tS\[)", prompt[::-1]):
        # 获取匹配的开始和结束位置（反转后的索引）
        start, end = match.start(), match.end()
        if prev_end is not None and start != prev_end:
            # 如果找到新的匹配，并且上一个匹配结束位置不等于当前匹配开始位置
            # 返回上一个匹配结束位置在原始提示文本中的索引
            return len(prompt) - prev_end
        # 更新上一个匹配结束位置
        prev_end = end
    if prev_end is None:
        # speech token is not found in the prompt
        # 如果未找到任何语音标记
        return None
    # 返回最后一个语音标记在原始提示文本中的开始索引
    return len(prompt) - prev_end


def convert_to_wav_tensor(
    content: Union[str, os.PathLike, torch.Tensor, np.ndarray]
) -> torch.Tensor:
    """
    将不同类型的音频内容转换为 WAV 格式的张量。

    该函数支持从文件路径、字符串、numpy 数组、字节对象或张量中加载音频数据。
    如果输入是文件路径或字符串，则使用 torchaudio 加载音频文件。
    如果输入是 numpy 数组或字节对象，则将其转换为张量。
    如果输入已经是张量，则直接返回。
    所有输入都会被转换为单声道音频，并重采样到预期的采样率。

    参数:
        content (Union[str, os.PathLike, torch.Tensor, np.ndarray, bytes]): 要转换的音频内容。

    返回:
        torch.Tensor: 单声道 WAV 格式的音频张量。
    """
    if isinstance(content, os.PathLike) or isinstance(content, str):
        # 如果输入是文件路径或字符串，则加载音频文件
        audio_path = str(content)
        wav, sr = torchaudio.load(audio_path)
        if sr != EXPECTED_SAMPLING_RATE:
            # 如果采样率不是预期的，则进行重采样
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=EXPECTED_SAMPLING_RATE
            )
    elif isinstance(content, np.ndarray):
        # 如果输入是 numpy 数组，则转换为张量
        wav = torch.from_numpy(content)
    elif isinstance(content, bytes):
        # 如果输入是字节对象，则使用 torchaudio 从字节中加载音频
        wav, sr = torchaudio.load(BytesIO(content))
        if sr != EXPECTED_SAMPLING_RATE:
            # 如果采样率不是预期的，则进行重采样
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=EXPECTED_SAMPLING_RATE
            )
    else:
        # 如果输入已经是张量，则直接使用
        wav = content

    # 返回单声道音频张量
    return wav.squeeze()


def does_start_with_speech_token(encoded_string) -> bool:
    """
    检查编码字符串是否以语音标记开头。

    该函数检查给定的编码字符串是否以语音标记（如 '[Hu1]'）开头。
    语音标记的格式为 '[Hu1]'、'[Pi1]' 或 '[St1]'，其中数字部分可以变化。

    参数:
        encoded_string (Optional[str]): 要检查的编码字符串。

    返回:
        bool: 如果字符串以语音标记开头，则返回 True，否则返回 False。
    """
    if encoded_string is None or len(encoded_string) <= 4:
        # 如果字符串为空或长度小于等于 4，则不可能是语音标记
        # 最短的语音标记是 '[Hu1]'，长度为 5
        return False
    if encoded_string[0] != "[":
        # 如果第一个字符不是 '[', 则不可能是语音标记
        return False
    end_pos = 1
    while end_pos < len(encoded_string):
        if encoded_string[end_pos] == "]" and end_pos >= 4:
            # 如果当前字符是 ']' 且位置大于等于 4，则可能是语音标记
            # 检查前两个字符是否为 'Hu', 'Pi' 或 'St'
            if any(encoded_string[1:3].startswith(tok) for tok in ["Hu", "Pi", "St"]):
                return True
            return False
        # longest speech token is "[Huxxxxx]"
        # 最长的语音标记是 '[Huxxxxx]'，长度为 11
        if end_pos >= 10:
            return False
        end_pos += 1
    return False


def does_end_with_speech_token(encoded_string: str) -> bool:
    """
    检查编码字符串是否以语音标记结尾。

    该函数检查给定的编码字符串是否以语音标记（如 '[Hu1]'）结尾。
    语音标记的格式为 '[Hu1]'、'[Pi1]' 或 '[St1]'，其中数字部分可以变化。

    参数:
        encoded_string (str): 要检查的编码字符串。

    返回:
        bool: 如果字符串以语音标记结尾，则返回 True，否则返回 False。
    """
    if encoded_string is None or len(encoded_string) <= 4:
        # 如果字符串为空或长度小于等于 4，则不可能是语音标记
        # 最短的语音标记是 '[Hu1]'，长度为 5
        return False
    if encoded_string[-1] != "]":
        # 如果最后一个字符不是 ']', 则不可能是语音标记
        return False
    start_pos = len(encoded_string) - 2
    while start_pos >= 0:
        if encoded_string[start_pos] == "[" and start_pos + 3 < len(encoded_string):
            # 如果当前字符是 '[' 且位置加上 3 小于字符串长度，则可能是语音标记
            # 检查接下来的两个字符是否为 'Hu', 'Pi' 或 'St'
            if any(
                encoded_string[start_pos + 1 : start_pos + 3].startswith(tok)
                for tok in ["Hu", "Pi", "St"]
            ):
                return True
            return False
        # longest speech token is "[Huxxxxx]"
        # 最长的语音标记是 '[Huxxxxx]'，长度为 11
        if start_pos < len(encoded_string) - 10:
            return False
        start_pos -= 1
    return False


def get_forbidden_tokens(
    ban_special_tokens: bool = True,
    generate_only_speech: bool = False,
    generate_only_text: bool = False,
    ban_expressivity_tokens: bool = False,
) -> List[int]:
    """
    生成禁止的标记列表。

    该函数根据参数生成禁止的标记列表，用于控制生成内容。
    可以禁止特殊标记、仅生成语音或仅生成文本，以及禁止表现力标记。

    参数:
        ban_special_tokens (bool, 可选): 是否禁止特殊标记，默认为 True。
        generate_only_speech (bool, 可选): 是否仅生成语音，默认为 False。
        generate_only_text (bool, 可选): 是否仅生成文本，默认为 False。
        ban_expressivity_tokens (bool, 可选): 是否禁止表现力标记，默认为 False。

    返回:
        List[int]: 禁止的标记列表。
    """
    assert not (
        generate_only_speech and generate_only_text
    ), "Nothing will be generated when generate_only_speech and generate_only_text is all True."
    forbidden_tokens = []
    if ban_special_tokens:
        # 禁止 [Text] 和 [Speech] 标记
        forbidden_tokens += [
            32000,
            32001,
        ]  # [Text], [Speech]
    if generate_only_speech:
        # 禁止所有小于 32000 的标记
        forbidden_tokens += list(range(32000))
    elif generate_only_text:
        # 禁止 hubert 标记
        forbidden_tokens += list(range(32002, 32002 + 501))  # hubert tokens
        if ban_expressivity_tokens:
            # 禁止音高标记
            forbidden_tokens += list(range(32503, 32503 + 64))  # pitch tokens
            # 禁止风格标记
            forbidden_tokens += list(
                range(32567, 32567 + 100)
            )  # forbidden style tokens
    return forbidden_tokens
