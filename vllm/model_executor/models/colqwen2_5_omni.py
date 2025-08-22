from typing import Iterable, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor import SamplingMetadata

from .utils import AutoWeightsLoader, WeightsMapper

from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.pooler import DispatchPooler, Pooler, PoolingType, PoolingMethod, PoolingTask, PoolingMetadata, PoolingParamsUpdate, build_output
from vllm.sequence import IntermediateTensors
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig

class ColQwen2_5OmniPooler(Pooler):
    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__()

        pooling_type = PoolingType.ALL
        self.pooling = PoolingMethod.from_pooling_type(pooling_type)

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"embed"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return self.pooling.get_pooling_updates(task)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        pooled_output = self.pooling(hidden_states, pooling_metadata)

        return build_output(pooled_output) 


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5OmniThinkerMultiModalProcessor,
    info=Qwen2_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder
)
class ColQwen2_5Omni(Qwen2_5OmniThinkerForConditionalGeneration):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "audio_tower.": "audio_tower.",
            "visual.": "visual.",
            "model.": "language_model.model.",
            "custom_text_proj.wright": "custom_text_proj.wright",
            "custom_text_proj.bias": "custom_text_proj.bias",
        })
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", mask_non_image_embeddings: bool = False):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.config: Qwen2_5OmniThinkerConfig = vllm_config.model_config.hf_config.thinker_config
            
        self.dim = 128
        self.custom_text_proj = ColumnParallelLinear(
            self.config.text_config.hidden_size,
            self.dim,
            bias=True,
            gather_output=True,
            prefix=f"{prefix}.custom_text_proj"
        )
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.language_model.lm_head = nn.Identity()  # Disable the original lm_head
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

        self.pooler = ColQwen2_5OmniPooler(self.config)
        pooler_config = vllm_config.model_config.pooler_config
        if pooler_config is not None:
            self.pooler = DispatchPooler({
                "embed":
                Pooler.for_encode(pooler_config),
                "embed":
                ColQwen2_5OmniPooler(vllm_config.model_config.hf_config.thinker_config),
            })

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object
    ) -> torch.Tensor:
        # Call the language model to get hidden states
        last_hidden_states = super().forward(input_ids=input_ids,positions=positions, inputs_embeds=inputs_embeds, intermediate_tensors=intermediate_tensors, **kwargs)

        # Project to lower dimension
        proj, _ = self.custom_text_proj(last_hidden_states)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True) # (sequence_length, dim)

        return proj

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata
    ) -> Optional[torch.Tensor]:
        # For embedding models, we don't compute logits
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # For embedding models, we don't need to sample
        return None

    def load_weights(self, weights: Iterable[Tuple[str,
        torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["language_model.lm_head.weight"])
        loaded_weights = loader.load_weights(weights,
                                             mapper=self.hf_to_vllm_mapper)
        return loaded_weights

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.spatial_merge_size