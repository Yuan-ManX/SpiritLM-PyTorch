import torch
import torchaudio
from torch import nn

from hubert_model import load_hubert_model
from quantizer_model import load_quantizer_model


class HubertTokenizer(nn.Module):
    """
    HubertTokenizer 类。

    该类实现了基于 HuBERT（Hidden-Unit BERT）模型的标记器，用于将音频信号转换为离散标记序列。
    """
    def __init__(
        self,
        hubert_ckpt,
        hubert_layer,
        quantizer_ckpt,
        is_linear_quantizer=True,
        min_chunk=400,
        max_chunk=100 * 16_000,
    ):
        """
        初始化 HubertTokenizer。

        参数:
            hubert_ckpt (str): HuBERT 模型检查点路径。
            hubert_layer (int): 要使用的 HuBERT 层数。
            quantizer_ckpt (str): 量化器模型检查点路径。
            is_linear_quantizer (bool, 可选): 是否使用线性量化器，默认为 True。
            min_chunk (int, 可选): 最小块大小，默认为 400。
            max_chunk (int, 可选): 最大块大小，默认为 100 * 16,000。
        """
        super().__init__()

        # hubert model
        # HuBERT 模型相关参数
        self.hubert_ckpt = str(hubert_ckpt)  # HuBERT 模型检查点路径
        self.hubert_layer = hubert_layer  # 要使用的 HuBERT 层数
        self.hubert_model = None  # HuBERT 模型实例
        self.should_normalize = False  # 是否需要标准化音频信号
        self.min_chunk = min_chunk  # 最小块大小
        self.max_chunk = max_chunk  # 最大块大小

        # quantizer model
        # 量化器模型相关参数
        self.quantizer_ckpt = str(quantizer_ckpt)  # 量化器模型检查点路径
        self.is_linear_quantizer = is_linear_quantizer  # 是否使用线性量化器
        self.quantizer_model = None  # 量化器模型实例

        # this is useful for determining the device
        # 注册一个浮点张量缓冲区，用于确定设备
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))
        self.load_models()

    @torch.no_grad()  # otherwise some non-leaf nodes appear which breaks serialization
    def load_models(self):
        """
        加载 HuBERT 和量化器模型。

        该方法加载预训练的 HuBERT 模型和量化器模型，并将其移动到适当的设备上。
        """
        # 加载 HuBERT 模型
        hubert_model, model_cfg, task_cfg = load_hubert_model(self.hubert_ckpt)  # 加载 HuBERT 模型
        self.hubert_task_cfg = task_cfg  # 保存任务配置
        self.hubert_model_cfg = model_cfg  # 保存模型配置
        self.hubert_model = hubert_model  # 保存 HuBERT 模型实例
        self.hubert_model.to(self.device)  # 将模型移动到设备
        self.hubert_model.eval()  # 设置模型为评估模式
        for parameter in self.hubert_model.parameters():
            parameter.requires_grad_(False) # 冻结模型参数
        self.should_normalize = task_cfg.normalize # 设置是否需要标准化音频信号

        # Load quantizer model
        # 加载量化器模型
        self.quantizer_model = load_quantizer_model(
            self.quantizer_ckpt, is_linear_quantizer=self.is_linear_quantizer
        ) # 加载量化器模型
        self.quantizer_model.to(self.device)
        self.quantizer_model.eval()

    @property
    def device(self):
        """
        获取当前设备。

        返回:
            torch.device: 当前设备。
        """
        return self._float_tensor.device

    @property
    def code_hop_size(self) -> int:
        """
        计算码跳大小。

        码跳大小是音频信号中每个离散标记对应的样本数。

        返回:
            int: 码跳大小。对于 50Hz 模型为 320，对于 25Hz 模型为 640。
        """
        # 初始化跳步大小
        hop_size = 1
        for dim, kernel, stride in eval(self.hubert_model_cfg.conv_feature_layers):
            hop_size *= stride # 计算跳步大小
        return hop_size  # 320 for 50hz model and 640 for 25hz model

    @property
    def frame_rate(self) -> int:
        """
        计算帧率。

        帧率是每秒的离散标记数。

        返回:
            int: 帧率。对于 50Hz 模型为 50，对于 25Hz 模型为 25。
        """
        return self.expected_sample_rate / self.code_hop_size  # 50 or 25

    @property
    def n_units(self) -> int:
        """
        获取量化器的单位数。

        返回:
            int: 量化器的单位数。
        """
        return self.kmeans_model.K

    @property
    def expected_sample_rate(self) -> int:
        """
        获取期望的采样率。

        返回:
            int: 期望的采样率，默认为 16,000。
        """
        return self.hubert_task_cfg.sample_rate  # 16_000

    def load_audio(self, path):
        """
        加载音频文件并预处理。

        参数:
            path (str): 音频文件路径。

        返回:
            torch.Tensor: 预处理后的音频张量。
        """
        wav, sr = torchaudio.load(path)
        if sr != self.expected_sample_rate:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=self.expected_sample_rate
            )
        # 返回预处理后的音频张量
        return wav

    @torch.inference_mode()
    def forward(self, x, separate_channels=False, dense=False):
        """
        前向传播方法，将输入音频转换为离散标记或密集特征。

        参数:
            x: 输入音频，可以是文件路径、音频张量或 numpy 数组。
            separate_channels (bool, 可选): 是否分别处理每个音频通道，默认为 False。
            dense (bool, 可选): 是否返回密集特征而不是离散标记，默认为 False。

        返回:
            torch.Tensor: 离散标记或密集特征。
        """
        if isinstance(x, str):
            # 如果输入是字符串，则加载音频文件
            x = self.load_audio(x)
        # 获取输入的维度
        i_ndim = x.dim()
        if i_ndim == 2:
            # 如果输入是二维张量，则增加一个批次维度
            x = x.unsqueeze(0)
        elif i_ndim == 1:
            # 如果输入是一维张量，则增加批次和通道维度
            x = x.view(1, 1, -1)

        # x should expect a shape [B, C, T], where C is number of channels
        # 输入张量应该具有形状 [B, C, T]，其中 C 是通道数
        assert len(x.shape) == 3
        # 获取密集特征，形状为 [B, T_enc]
        feats = self.get_dense_features(x)  # [B, T_enc]

        if dense:
            return feats

        # 使用量化器模型将密集特征转换为离散标记，形状为 [B, T_enc]
        tokens = self.quantizer_model(feats)  # [B, T_enc]

        if i_ndim == 3:
            # 如果输入是三维张量，则调整标记的形状
            tokens = tokens.view(x.shape[0], 1, -1)
        else:
            # 否则，去除批次维度
            tokens = tokens.squeeze(0)

        if not separate_channels:
            # 如果不需要分别处理每个通道，则返回标记
            return tokens

    @torch.inference_mode()
    def get_dense_features(self, x, separate_channels=False):
        """
        获取密集特征。

        参数:
            x: 输入音频张量，形状为 [B, C, T]。
            separate_channels (bool, 可选): 是否分别处理每个音频通道，默认为 False。

        返回:
            torch.Tensor: 密集特征，形状为 [B, T_enc]。
        """
        x = x.to(self.device)

        assert separate_channels == False, "Not supported yet"  # TODO: Fix this

        if not separate_channels:
            # 如果不需要分别处理每个通道，则对通道维度求平均，得到形状 [B, T]
            x = x.mean(1)  # [B, T]

        if self.should_normalize:
            # 如果需要标准化，则对每个样本进行层归一化
            x = torch.cat([nn.functional.layer_norm(item, item.shape) for item in x])

        # 初始化特征列表
        feat = []
        for start in range(0, x.size(1), self.max_chunk):
            # 分割输入张量为块
            x_chunk = x[:, start : start + self.max_chunk]
            if x_chunk.size(1) < self.min_chunk:
                # 如果块大小小于最小块大小，则跳过
                continue
            feat_chunk, _ = self.hubert_model.extract_features(
                source=x_chunk,  # 输入块
                padding_mask=None,  # 填充掩码
                mask=False,  # 是否使用掩码
                output_layer=self.hubert_layer,  # 输出层
            )
            # 将特征块添加到列表中
            feat.append(feat_chunk)

        # 连接所有特征块，得到最终的密集特征
        return torch.cat(feat, 1)

