# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    compute_causal_conv1d_metadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        if self.speculative_config:
            self.num_spec = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.decode_cudagraph_max_bs = min(
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1),
            self.compilation_config.max_cudagraph_capture_size,
        )

        self.spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def build(  # type: ignore[override]
    self,
    common_prefix_len: int,
    common_attn_metadata: CommonAttentionMetadata,
    num_accepted_tokens: torch.Tensor | None = None,
    num_decode_draft_tokens_cpu: torch.Tensor | None = None,
    fast_build: bool = False,
) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        device = query_start_loc.device

        # OPT-1: 尽量使用 metadata 自带的 CPU 副本做“计数/分支”，避免 GPU .item() 同步。
        #       如果没有 CPU 副本，fallback 会产生 D2H 拷贝（会慢）；建议确保上游提供 query_start_loc_cpu。
        query_start_loc_cpu = getattr(m, "query_start_loc_cpu", None)
        if query_start_loc_cpu is None:
            query_start_loc_cpu = query_start_loc.detach().cpu()

        context_lens_cpu = m.num_computed_tokens_cpu  # CPU tensor
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        # ---------------------------
        # 1) 判定是否存在 spec decode（严格保持原语义）
        # ---------------------------
        spec_sequence_masks_cpu = None          # CPU bool mask (len = num_reqs)
        spec_sequence_masks = None              # GPU bool mask
        num_spec_decodes = 0

        # OPT-2: “是否存在 spec”的判定保持原逻辑：
        #        只有 masked 的 draft token 之和 > 0 才认为本 step 有 spec，
        #        而不是只要 mask.sum()>0 就进入 spec。
        if self.use_spec_decode and num_decode_draft_tokens_cpu is not None:
            mask_cpu = (num_decode_draft_tokens_cpu >= 0)
            if mask_cpu.any().item():
                total_draft = num_decode_draft_tokens_cpu[mask_cpu].sum().item()
            else:
                total_draft = 0

            if total_draft > 0:
                spec_sequence_masks_cpu = mask_cpu
                num_spec_decodes = mask_cpu.sum().item()
                if num_spec_decodes > 0:
                    # OPT-3: 只有在确实需要走 spec 路径时才把 mask H2D（减少无效 H2D）
                    spec_sequence_masks = spec_sequence_masks_cpu.to(device, non_blocking=True)

        # ---------------------------
        # 2) 走非 spec 或 spec 路径，构造 metadata
        # ---------------------------
        if spec_sequence_masks is None:
            # --- 非 spec 路径（保持原行为） ---
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0

            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = m.block_table_tensor[:, 0]

            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            num_accepted_tokens = None

        else:
            # --- spec 路径 ---
            assert spec_sequence_masks_cpu is not None

            # OPT-4: 计数全部用 CPU query_start_loc_cpu / query_lens_cpu 来做，
            #        避免原版 (tensor.sum().item()) 在 GPU 上触发同步。
            query_lens_cpu = (query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])
            non_spec_mask_cpu = ~spec_sequence_masks_cpu
            non_spec_query_lens_cpu = query_lens_cpu[non_spec_mask_cpu]

            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            num_prefills = non_spec_query_lens_cpu.numel() - num_decodes
            num_decode_tokens = num_decodes
            num_prefill_tokens = non_spec_query_lens_cpu.sum().item() - num_decode_tokens

            total_tokens_cpu = query_start_loc_cpu[-1].item()
            num_spec_decode_tokens = total_tokens_cpu - num_prefill_tokens - num_decode_tokens

            # GPU query_lens 仅在需要构造 token mask / cumsum 时用
            query_lens = query_start_loc[1:] - query_start_loc[:-1]

            if num_prefills == 0 and num_decodes == 0:
                # --- 纯 spec 场景 ---
                # OPT-5: 避免 query_start_loc[-1].item()（GPU 同步），用 CPU total_tokens_cpu。
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    total_tokens_cpu,
                )
                spec_token_indx = torch.arange(
                    spec_token_size, dtype=torch.int32, device=device
                )
                non_spec_token_indx = torch.empty(0, dtype=torch.int32, device=device)

                spec_state_indices_tensor = m.block_table_tensor[:, : self.num_spec + 1]
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc
                non_spec_query_start_loc = None

            else:
                # --- 混合场景（同时存在 spec / non-spec token） ---
                # OPT-6: 用 stable partition 代替 torch.argsort(bool_mask)：
                #        argsort 是 O(N log N)；我们只需要把 False/True 分组且保持原顺序（stable）。
                #        nonzero 返回的索引天然递增 = stable。
                spec_token_masks = torch.repeat_interleave(spec_sequence_masks, query_lens)

                non_spec_token_indx = torch.nonzero(
                    ~spec_token_masks, as_tuple=False
                ).flatten()

                spec_token_indx = torch.nonzero(
                    spec_token_masks, as_tuple=False
                ).flatten()

                # （可选健壮性检查，线上可关）
                # assert non_spec_token_indx.numel() == (num_prefill_tokens + num_decode_tokens)

                spec_state_indices_tensor = m.block_table_tensor[
                    spec_sequence_masks, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = m.block_table_tensor[
                    ~spec_sequence_masks, 0
                ]

                # OPT-7: 减少 memset：原版用 torch.zeros 再 cumsum，
                #        这里用 empty + 手动置 0 + out=cumsum（小优化，但更干净）。
                spec_query_start_loc = torch.empty(
                    (num_spec_decodes + 1,),
                    dtype=torch.int32,
                    device=device,
                )
                spec_query_start_loc[0].zero_()
                torch.cumsum(
                    query_lens[spec_sequence_masks],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )

                non_spec_query_start_loc = torch.empty(
                    (query_lens.size(0) - num_spec_decodes + 1,),
                    dtype=torch.int32,
                    device=device,
                )
                non_spec_query_start_loc[0].zero_()
                torch.cumsum(
                    query_lens[~spec_sequence_masks],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )

            assert num_accepted_tokens is not None
            # 保持原逻辑：accepted_tokens 仅保留 spec sequences
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]

        # ---------------------------
        # 3) Conv1D metadata（prefill 才需要）
        # ---------------------------
        if num_prefills > 0:
            # OPT-8: 避免把 context_lens(int) 整体搬到 GPU。
            #        先在 CPU 上算 bool，再把 bool H2D（更小、更省带宽）。
            has_initial_state_cpu = (context_lens_cpu > 0)
            if spec_sequence_masks_cpu is not None:
                has_initial_state_cpu = has_initial_state_cpu[~spec_sequence_masks_cpu]
            has_initial_state = has_initial_state_cpu.to(device, non_blocking=True)

            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(non_spec_query_start_loc)
            )
        else:
            has_initial_state = None

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks, non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()
        m._num_computed_tokens_cpu = m.seq_lens_cpu - num_accepted_tokens.cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
