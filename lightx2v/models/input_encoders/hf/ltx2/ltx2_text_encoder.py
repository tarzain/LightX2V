import os
from typing import List, Optional, Union

import torch
from loguru import logger

from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class LTX2TextEncoder:
    """
    Text encoder for LTX-2 video generation model.

    Uses T5EncoderModel from HuggingFace transformers library.
    The default model is google/t5-v1_1-xxl for high-quality text understanding.

    For LTX-2, the text encoder processes prompts into embeddings that
    condition the video generation process.
    """

    def __init__(
        self,
        config,
        checkpoint_path: str,
        device=None,
        cpu_offload: bool = False,
        dtype=None,
        max_sequence_length: int = 256,
    ):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = device if device is not None else torch.device(AI_DEVICE)
        self.cpu_offload = cpu_offload
        self.dtype = dtype if dtype is not None else GET_DTYPE()
        self.max_sequence_length = max_sequence_length

        self._load_model()

    def _load_model(self):
        """Load T5 text encoder and tokenizer."""
        try:
            from transformers import T5EncoderModel, T5TokenizerFast
        except ImportError:
            raise ImportError(
                "transformers library is required for LTX-2 text encoder. "
                "Install it with: pip install transformers>=4.40.0"
            )

        logger.info(f"Loading LTX-2 text encoder from {self.checkpoint_path}")

        # Determine if path is local or HuggingFace model ID
        if os.path.isdir(self.checkpoint_path):
            # Local path - check for text_encoder subfolder
            encoder_path = self.checkpoint_path
            if os.path.isdir(os.path.join(self.checkpoint_path, "text_encoder")):
                encoder_path = os.path.join(self.checkpoint_path, "text_encoder")

            self.tokenizer = T5TokenizerFast.from_pretrained(
                encoder_path,
                local_files_only=True,
            )
            self.model = T5EncoderModel.from_pretrained(
                encoder_path,
                torch_dtype=self.dtype,
                local_files_only=True,
            )
        else:
            # HuggingFace model ID - try with subfolder first
            try:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    self.checkpoint_path,
                    subfolder="text_encoder",
                )
                self.model = T5EncoderModel.from_pretrained(
                    self.checkpoint_path,
                    subfolder="text_encoder",
                    torch_dtype=self.dtype,
                )
            except Exception:
                # Fall back to direct loading (for standalone T5 models)
                self.tokenizer = T5TokenizerFast.from_pretrained(self.checkpoint_path)
                self.model = T5EncoderModel.from_pretrained(
                    self.checkpoint_path,
                    torch_dtype=self.dtype,
                )

        # Move to device if not using CPU offload
        if not self.cpu_offload:
            self.model = self.model.to(self.device)

        self.model.eval()

        logger.info(f"LTX-2 text encoder loaded successfully")

    @torch.no_grad()
    def encode(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: int = 1,
        return_attention_mask: bool = False,
    ):
        """
        Encode text prompt(s) to embeddings.

        Args:
            prompt: Text prompt or list of prompts
            negative_prompt: Optional negative prompt(s) for CFG
            num_videos_per_prompt: Number of videos to generate per prompt
            return_attention_mask: Whether to return attention mask

        Returns:
            Dictionary containing:
                - context: Text embeddings [B, seq_len, hidden_size]
                - context_null: Negative prompt embeddings (if negative_prompt provided)
                - attention_mask: Attention mask (if return_attention_mask=True)
        """
        if self.cpu_offload:
            self.model = self.model.to(AI_DEVICE)

        # Ensure prompt is a list
        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)

        # Get prompt embeddings
        prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
        )

        result = {
            "context": prompt_embeds,
        }

        # Get negative prompt embeddings if provided
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            negative_prompt_embeds, _ = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
            )
            result["context_null"] = negative_prompt_embeds
        else:
            # Create zero embeddings for unconditional
            result["context_null"] = torch.zeros_like(prompt_embeds)

        if return_attention_mask:
            result["attention_mask"] = prompt_attention_mask

        if self.cpu_offload:
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()

        return result

    def _get_t5_prompt_embeds(
        self,
        prompt: List[str],
        num_videos_per_prompt: int = 1,
    ):
        """
        Get T5 prompt embeddings.

        Args:
            prompt: List of text prompts
            num_videos_per_prompt: Number of videos to generate per prompt

        Returns:
            Tuple of (prompt_embeds, attention_mask)
        """
        batch_size = len(prompt)

        # Tokenize
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(self.model.device)
        attention_mask = text_inputs.attention_mask.to(self.model.device)

        # Check for truncation
        untruncated_ids = self.tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt",
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1]:
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.max_sequence_length - 1 : -1]
            )
            for i, text in enumerate(removed_text):
                if text.strip():
                    logger.warning(
                        f"Prompt was truncated. Removed text: {text}"
                    )

        # Get embeddings from T5
        prompt_embeds = self.model(
            input_ids=text_input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        # Duplicate embeddings for multiple videos per prompt
        if num_videos_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_videos_per_prompt, dim=0)

        return prompt_embeds, attention_mask

    def infer(self, prompt: Union[str, List[str]]):
        """
        Convenience method for encoding a prompt.

        Args:
            prompt: Text prompt or list of prompts

        Returns:
            Text embeddings tensor [B, seq_len, hidden_size]
        """
        result = self.encode(prompt)
        return result["context"]

    def to(self, device):
        """Move text encoder to specified device."""
        self.device = device
        if self.model is not None and not self.cpu_offload:
            self.model = self.model.to(device)
        return self

    def to_cuda(self):
        """Move text encoder to CUDA."""
        return self.to(AI_DEVICE)

    def to_cpu(self):
        """Move text encoder to CPU."""
        return self.to("cpu")
