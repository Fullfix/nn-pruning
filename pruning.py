from abc import ABC, abstractmethod
import torch
import math
import random
import time
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from typing import Callable
from transformers import OPTForCausalLM, PreTrainedTokenizerBase, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from datasets import load_dataset, Dataset


def parse_structure(structure: str) -> tuple[int, int]:
    try:
        n_str, m_str = structure.split(":")
        n, m = int(n_str), int(m_str)
    except Exception as e:
        raise ValueError(f"Invalid structure='{structure}'. Expected like '2:4' or '4:8'.") from e
    if n <= 0 or m <= 0 or n >= m:
        raise ValueError(f"Invalid structure='{structure}'. Must satisfy 0 < n < m.")
    return n, m


def verify_unstructured_sparsity(W: torch.Tensor, prune_k: int) -> bool:
    actual_zeros = W.numel() - torch.count_nonzero(W)
    return actual_zeros >= prune_k


def verify_nm_sparsity(W: torch.Tensor, n: int, m: int) -> bool:
    out_features, in_features = W.shape

    if in_features % m > 0:
        raise ValueError(f"In_features {in_features} must be divisible by m: {m}")

    W_groups = W.reshape(-1, m)
    non_zeros_per_group = torch.count_nonzero(W_groups, dim=1)
    is_correct = torch.all(non_zeros_per_group <= n).item()

    return is_correct


class BaseUnstructuredPruner(ABC):
    @abstractmethod
    def prune_model(
            self,
            model: PreTrainedModel,
            sparsity: str | float,
            calib_dataloader: DataLoader | None = None,
            pbar: bool = True
    ) -> None:
        """
        Base method for unstructured pruning
        :param model: HF model to prune in-place
        :param sparsity: target sparsity. Float in range [0, 1] means fraction of neurons to prune within linear layers. 'M:N' string means semi-structured pruning.
        :param calib_dataloader: dataloader for calibration data. Required only for activation aware methods.
        :param pbar: enable progress bar
        """
        pass


class BaseAgnosticUnstructuredPruner(BaseUnstructuredPruner):
    @abstractmethod
    def prune_linear_unstruct(self, W: torch.Tensor, k: int) -> torch.Tensor:
        """
        Base method for unstructured pruning
        :param W: linear weights to prune, shape (D_out, D_in)
        :param k: pruning number.
        :return: pruned linear weights. Must have more or equal than k zero elements.
        """
        pass

    @abstractmethod
    def prune_linear_nm(self, W: torch.Tensor, n: int, m: int) -> torch.Tensor:
        """
        Base method for N:M pruning
        :param W: linear weights to prune, shape (D_out, D_in)
        :param n: number of neurons to prune within each block
        :param m: block size
        :return: pruned linear weights. Must satisfy N:M block sparsity.
        """
        pass

    def prune_opt_agnostic(self, model: OPTForCausalLM, sparsity: str | float, pbar: bool) -> None:
        layers = model.model.decoder.layers
        layer_idx_iterable = range(len(layers))
        for i in (tqdm(layer_idx_iterable, desc="Agnostic pruning") if pbar else layer_idx_iterable):
            layer: OPTDecoderLayer = layers[i]
            pruned_modules: list[torch.nn.Linear] = [
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.q_proj,
                layer.self_attn.out_proj,
                layer.fc1,
                layer.fc2
            ]
            for module in tqdm(pruned_modules, desc=f"Layer{i}: Pruning per module", leave=False) if pbar else pruned_modules:
                W = module.weight.detach()
                if isinstance(sparsity, float):
                    k = int(math.ceil(W.numel() * sparsity))
                    if k >= W.numel() or k < 0:
                        raise ValueError(f"Invalid k: {k} for layer {i}, module {module}")
                    elif k == 0:
                        W_pruned = W
                    else:
                        W_pruned = self.prune_linear_unstruct(W, k=k)
                        if not verify_unstructured_sparsity(W_pruned, prune_k=k):
                            raise RuntimeError(f"Unstructured sparsity invalid for layer {i}")
                elif isinstance(sparsity, str):
                    n, m = parse_structure(sparsity)
                    W_pruned = self.prune_linear_nm(W, n=n, m=m)
                    if not verify_nm_sparsity(W_pruned, n=n, m=m):
                        raise RuntimeError(f"NM sparsity invalid for layer {i}")
                else:
                    raise ValueError(f"Invalid sparsity: {sparsity}")
                module.weight.copy_(W_pruned.to(dtype=module.weight.dtype, device=module.weight.device))

    @torch.no_grad()
    def prune_model(self, model: PreTrainedModel, sparsity: str | float, calib_dataloader: DataLoader | None = None, pbar: bool = True) -> None:
        if isinstance(model, OPTForCausalLM):
            self.prune_opt_agnostic(model, sparsity, pbar=pbar)
        else:
            raise NotImplementedError(f"Pruning not implemented for {type(model)}")


class OptModelForwarder:
    def __init__(self, model: OPTForCausalLM, pbar: bool = True):
        self.model = model
        self.pbar = pbar
        self.device = torch.device('cuda')
        self.old_use_cache = model.config.use_cache

        self.inps = None
        self.outs = None
        self.attn_masks = None
        self.n_samples = None
        self.calib_batch_size = None

    def embed(self, dataloader: DataLoader) -> None:
        model = self.model
        device = self.device
        pbar = self.pbar

        model.eval()

        hidden_size = model.config.hidden_size
        model_dtype = next(iter(model.parameters())).dtype
        decoder = model.model.decoder
        layers = decoder.layers

        model.config.use_cache = False

        # Move embeds and first layer to device
        decoder.embed_tokens = decoder.embed_tokens.to(device)
        decoder.embed_positions = decoder.embed_positions.to(device)
        if decoder.project_out is not None:
            decoder.project_out = decoder.project_out.to(device)
        if decoder.project_in is not None:
            decoder.project_in = decoder.project_in.to(device)
        layers[0] = layers[0].to(device)

        # Infer seqlen and n_samples
        first_batch = next(iter(dataloader))
        if not isinstance(first_batch, dict) or "input_ids" not in first_batch:
            raise KeyError("Expected collated batch to be a dict with key 'input_ids'.")
        seqlen = int(first_batch["input_ids"].shape[1])
        calib_batch_size = int(first_batch['input_ids'].shape[0])
        n_samples = len(dataloader.dataset)

        inps = torch.empty(
            (n_samples, seqlen, hidden_size),
            dtype=model_dtype,
            device='cpu',
            pin_memory=True
        )
        outs = torch.empty(
            (n_samples, seqlen, hidden_size),
            dtype=model_dtype,
            device="cpu",
            pin_memory=True,
        )
        cache: dict[str, int | torch.Tensor | None] = {"i": 0, "attn_mask": None}

        class StopForward(Exception):
            pass

        class Catcher(torch.nn.Module):
            def __init__(self, module: torch.nn.Module):
                super().__init__()
                self.module = module

            def forward(self, inp: torch.Tensor, **kwargs):
                i = int(cache["i"])
                bsz = int(inp.shape[0])
                take = min(bsz, inps.shape[0] - i)

                if take > 0:
                    # Copy hidden states to CPU
                    inps[i:i+take].copy_(inp[:take].detach(), non_blocking=True)

                    # Capture the attention_mask seen by the layer (may be 2D or expanded 4D).
                    # We store per-sample slices to fix the "last batch only" bug.
                    if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                        am = kwargs["attention_mask"].detach()
                        # Ensure batch is first dimension
                        if am.shape[0] != bsz:
                            # If something unexpected happens, keep behavior explicit
                            raise RuntimeError(f"Unexpected attention_mask batch dim: got {am.shape}, bsz={bsz}")

                        if cache["attn_mask"] is None:
                            cache["attn_mask"] = torch.empty(
                                (n_samples, *am.shape[1:]),
                                dtype=am.dtype,
                                device="cpu",
                                pin_memory=True,
                            )
                        cache["attn_mask"][i:i+take].copy_(am[:take], non_blocking=True)

                    cache["i"] = i + take

                # Stop model forward once we captured layer0 inputs
                raise StopForward

        layers[0] = Catcher(layers[0])

        for batch in tqdm(dataloader, desc="Catching", leave=False) if pbar else dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            try:
                _ = model(**batch)
            except StopForward:
                pass

            if int(cache["i"]) >= n_samples:
                break

        layers[0] = layers[0].module

        if int(cache["i"]) < n_samples:
            raise RuntimeError(f"Captured only {cache['i']} sequences, but n_samples={n_samples}.")

        # Offload embeddings and first layer back to CPU
        decoder.embed_tokens = decoder.embed_tokens.cpu()
        decoder.embed_positions = decoder.embed_positions.cpu()
        if decoder.project_out is not None:
            decoder.project_out = decoder.project_out.cpu()
        if decoder.project_in is not None:
            decoder.project_in = decoder.project_in.cpu()
        layers[0] = layers[0].cpu()

        torch.cuda.empty_cache()

        self.inps = inps
        self.outs = outs
        self.n_samples = n_samples
        self.calib_batch_size = calib_batch_size
        self.attn_masks = cache["attn_mask"]

    def forward_layer(self, layer_idx: int) -> None:
        n_samples = self.n_samples
        calib_batch_size = self.calib_batch_size
        decoder = self.model.model.decoder
        device = self.device
        layer: OPTDecoderLayer = decoder.layers[layer_idx]
        attn_masks = self.attn_masks
        pbar = self.pbar

        layer = layer.to(device)

        calib_iterable = range(0, n_samples, calib_batch_size)
        for j0 in tqdm(calib_iterable, desc=f"Layer {layer_idx}: Collecting activations", leave=False) if pbar else calib_iterable:
            j1 = min(j0 + calib_batch_size, n_samples)

            inp = self.inps[j0:j1].to(device, non_blocking=True)
            if attn_masks is None:
                out = layer(inp)[0]
            else:
                attention_mask = attn_masks[j0:j1].to(device, non_blocking=True)
                out = layer(inp, attention_mask=attention_mask)[0]

            self.outs[j0:j1].copy_(out.detach(), non_blocking=True)

        decoder.layers[layer_idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        self.inps, self.outs = self.outs, self.inps

    def output_logits(self):
        model = self.model
        decoder = model.model.decoder
        device = self.device
        n_samples = self.n_samples
        calib_batch_size = self.calib_batch_size
        pbar = self.pbar

        if decoder.final_layer_norm is not None:
            decoder.final_layer_norm = decoder.final_layer_norm.to(device)

        if hasattr(decoder, "project_out") and decoder.project_out is not None:
            decoder.project_out = decoder.project_out.to(device)

        model.lm_head = model.lm_head.to(device)

        logits = None
        calib_iterable = range(0, n_samples, calib_batch_size)

        for j0 in tqdm(calib_iterable, desc=f"Output logits: Iterating", leave=False) if pbar else calib_iterable:
            j1 = min(j0 + calib_batch_size, n_samples)

            inp = self.inps[j0:j1].to(device, non_blocking=True)

            if decoder.final_layer_norm is not None:
                inp = decoder.final_layer_norm(inp)

            if hasattr(decoder, "project_out") and decoder.project_out is not None:
                inp = decoder.project_out(inp)

            batch_logits = model.lm_head(inp)

            if logits is None:
                logits = torch.empty(
                    (n_samples, inp.shape[1], batch_logits.shape[-1]),
                    dtype=batch_logits.dtype,
                    device="cpu",
                    pin_memory=True,
                )

            logits[j0:j1].copy_(batch_logits.detach(), non_blocking=True)

        if decoder.final_layer_norm is not None:
            decoder.final_layer_norm = decoder.final_layer_norm.cpu()
        if hasattr(decoder, "project_out") and decoder.project_out is not None:
            decoder.project_out = decoder.project_out.cpu()
        model.lm_head = model.lm_head.cpu()

        model.config.use_cache = self.old_use_cache

        torch.cuda.empty_cache()

        return logits


class BaseActAwareUnstructuredPruner(BaseUnstructuredPruner):
    @abstractmethod
    def add_linear_activations(self, module_name: str, X: torch.Tensor) -> None:
        """
        Method for accumulating activation statistics, for example: covariance matrix.
        :param module_name: unique module name to attach the activation statistics to.
        :param X: input activations tensor. Shape (n_samples, seqlen, D_in).
        """
        pass

    @abstractmethod
    def clear_linear_activations(self, module_name: str) -> None:
        """
        Method for clearing activation statistics.
        :param module_name: module name which was passed when accumulating activation statistics.
        """
        pass

    @abstractmethod
    def prune_linear_unstruct(self, W: torch.Tensor, module_name: str, k: int) -> torch.Tensor:
        """
        Base method for unstructured pruning
        :param W: linear weights to prune, shape (D_out, D_in)
        :param module_name: unique module name of the pruned model. Use this to retrieve activation statistics.
        :param k: pruning number.
        :return: pruned linear weights. Must have more or equal than k zero elements.
        """
        pass

    @abstractmethod
    def prune_linear_nm(self, W: torch.Tensor, module_name: str, n: int, m: int) -> torch.Tensor:
        """
        Base method for N:M pruning
        :param W: linear weights to prune, shape (D_out, D_in)
        :param module_name: unique module name of the pruned model. Use this to retrieve activation statistics.
        :param n: number of neurons to prune within each block
        :param m: block size
        :return: pruned linear weights. Must satisfy N:M block sparsity.
        """
        pass

    def prune_opt(self, model: OPTForCausalLM, sparsity: str | float, calib_dataloader: DataLoader, pbar: bool) -> None:
        decoder = model.model.decoder
        device = torch.device('cuda')
        forwarder = OptModelForwarder(model, pbar=pbar)

        forwarder.embed(calib_dataloader)

        layer_idx_iterable = range(len(decoder.layers))
        for i in tqdm(layer_idx_iterable, desc="Activation aware pruning") if pbar else layer_idx_iterable:
            layer = decoder.layers[i]
            pruned_modules: dict[str, torch.nn.Linear] = {
                f'layer{i}.self_attn.k_proj': layer.self_attn.k_proj,
                f'layer{i}.self_attn.v_proj': layer.self_attn.v_proj,
                f'layer{i}.self_attn.q_proj': layer.self_attn.q_proj,
                f'layer{i}.self_attn.out_proj': layer.self_attn.out_proj,
                f'layer{i}.fc1': layer.fc1,
                f'layer{i}.fc2': layer.fc2
            }

            def make_hook(module_name: str):
                def hook(_mod: torch.nn.Module, inp: tuple[torch.Tensor, ...], _out: torch.Tensor):
                    self.add_linear_activations(module_name, inp[0].detach())

                return hook

            handles = []
            try:
                for module_name, module in pruned_modules.items():
                    handles.append(module.register_forward_hook(make_hook(module_name)))

                forwarder.forward_layer(i)
            finally:
                for h in handles:
                    h.remove()

            pruned_modules_iterable = pruned_modules.items()
            for module_name, module in tqdm(
                    pruned_modules_iterable,
                    total=len(pruned_modules),
                    desc=f"Layer {i}: Pruning",
                    leave=False
            ) if pbar else pruned_modules_iterable:
                W = module.weight.detach().to(device)
                if isinstance(sparsity, float):
                    k = int(math.ceil(W.numel() * sparsity))
                    if k >= W.numel() or k < 0:
                        raise ValueError(f"Invalid k: {k} for layer {i}, module {module}")
                    elif k == 0:
                        W_pruned = W
                    else:
                        W_pruned = self.prune_linear_unstruct(W, module_name=module_name, k=k)
                        if not verify_unstructured_sparsity(W_pruned, prune_k=k):
                            raise RuntimeError(f"Unstructured sparsity invalid for layer {i}")
                elif isinstance(sparsity, str):
                    n, m = parse_structure(sparsity)
                    W_pruned = self.prune_linear_nm(W, module_name=module_name, n=n, m=m)
                    if not verify_nm_sparsity(W_pruned, n=n, m=m):
                        raise RuntimeError(f"NM sparsity invalid for layer {i}")
                else:
                    raise ValueError(f"Invalid sparsity: {sparsity}")
                module.weight.copy_(W_pruned.to(dtype=module.weight.dtype, device=module.weight.device))
                self.clear_linear_activations(module_name)

    @torch.no_grad()
    def prune_model(self, model: PreTrainedModel, sparsity: str | float, calib_dataloader: DataLoader | None = None, pbar: bool = True) -> None:
        if not isinstance(calib_dataloader, DataLoader):
            raise ValueError("calib_dataloader must be an instance of DataLoader")

        if isinstance(model, OPTForCausalLM):
            self.prune_opt(model, sparsity, calib_dataloader, pbar=pbar)
        else:
            raise NotImplementedError(f"Pruning not implemented for {type(model)}")


class MagnitudePruner(BaseAgnosticUnstructuredPruner):
    def prune_linear_unstruct(self, W: torch.Tensor, k: int) -> torch.Tensor:
        flat_abs = W.abs().flatten()
        prune_idx = torch.topk(flat_abs, k=k, largest=False, sorted=False).indices
        mask = torch.ones(W.numel(), device=W.device, dtype=torch.bool)
        mask[prune_idx] = False
        return (W.flatten() * mask.to(W.dtype)).view_as(W)

    def prune_linear_nm(self, W: torch.Tensor, n: int, m: int) -> torch.Tensor:
        orig_shape = W.shape
        columns = orig_shape[-1]
        if columns % m > 0:
            raise ValueError(f"Column size {columns} must be divisible by m: {m}")

        w_blocks = W.reshape(*orig_shape[:-1], columns // m, m)
        a_blocks = w_blocks.abs()

        topk_idx = torch.topk(a_blocks, k=n, dim=-1, largest=True, sorted=False).indices

        mask = torch.zeros_like(w_blocks, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)

        pruned_main = w_blocks * mask.to(W.dtype)
        pruned_main = pruned_main.reshape(*orig_shape[:-1], columns)
        return pruned_main


class ActCovCreator:
    def __init__(self, dtype=torch.float32):
        self.H: torch.Tensor | None = None
        self.count: int = 0
        self.dtype = dtype

    @torch.no_grad()
    def add(self, X: torch.Tensor) -> None:
        X_flat = X.reshape(-1, X.shape[-1]).to(dtype=self.dtype)
        XTX = (X_flat.transpose(0, 1) @ X_flat).cpu()
        if self.H is None:
            self.H = XTX
            self.count = X_flat.shape[0]
        else:
            self.H.add_(XTX)
            self.count += X_flat.shape[0]

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        if self.count == 0:
            raise ValueError("No activations were added")
        return self.H / float(self.count)

    @torch.no_grad()
    def clear(self) -> None:
        del self.H


class SparseGPTPruner(BaseActAwareUnstructuredPruner):
    def __init__(self, block_size: int = 128, mask_block: int = 64, eps=1e-10):
        self.block_size = block_size
        self.mask_block = mask_block
        self.eps = eps

        if self.block_size % self.mask_block > 0:
            raise ValueError(f"Block size {self.block_size} must be divisible by mask block {self.mask_block}")

        self.act_cov_creators: dict[str, ActCovCreator] = {}

    def add_linear_activations(self, module_name: str, X: torch.Tensor) -> None:
        if module_name not in self.act_cov_creators:
            self.act_cov_creators[module_name] = ActCovCreator(dtype=torch.float32)
        self.act_cov_creators[module_name].add(X)

    def clear_linear_activations(self, module_name: str) -> None:
        act_cov_creator = self.act_cov_creators.get(module_name, None)
        if act_cov_creator is not None:
            act_cov_creator.clear()
            del act_cov_creator
        del self.act_cov_creators[module_name]

    @staticmethod
    def _safe_cholesky_lower(H: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        diag_mean = torch.diagonal(H, dim1=-2, dim2=-1).mean().abs().clamp(min=eps)
        eye = torch.eye(H.shape[-1], device=H.device, dtype=H.dtype)

        for multiplier in [1e-2, 1e-1, 1.0]:
            A_damped = H + (multiplier * diag_mean * eye)
            L, info = torch.linalg.cholesky_ex(A_damped, upper=False)
            if not torch.any(info) and not torch.isnan(L).any().item():
                return L

        raise RuntimeError("Failed to perform Cholesky")

    def _prune(self, W: torch.Tensor, module_name: str, compute_mask: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        out_features, in_features = W.shape

        H = self.act_cov_creators[module_name].compute().to(device=W.device, dtype=torch.float32)
        dead = torch.diag(H) <= 1e-12
        H[dead, dead] = 1
        W[:, dead] = 0

        L = self._safe_cholesky_lower(H)
        H_inv = torch.cholesky_inverse(L)
        Hinv = self._safe_cholesky_lower(H_inv).transpose(0, 1)
        Hinvdiag = torch.diag(Hinv)

        W_out = W.clone().float()
        M = torch.zeros_like(W_out)
        E = torch.zeros((out_features, self.block_size), device=W_out.device, dtype=torch.float32)

        for i1 in range(0, in_features, self.block_size):
            i2 = i1 + self.block_size

            for delta in range(self.block_size):
                j = i1 + delta

                if delta % self.mask_block == 0:
                    seg_start = j
                    seg_end = j + self.mask_block
                    denom = (Hinvdiag[seg_start:seg_end] ** 2).clamp_min(self.eps)
                    scores = (W_out[:, seg_start:seg_end] ** 2) / denom.unsqueeze(0)  # [out, seg_len]
                    M_seg = compute_mask(scores)
                    M[:, seg_start:seg_end] = M_seg

                E[:, delta] = (1 - M[:, j]) * W_out[:, j] / (Hinv[j, j].clamp_min(self.eps))
                W_out[:, j:i2] -= torch.outer(E[:, delta], Hinv[j, j:i2])

            if i2 < in_features:
                W_out[:, i2:] -= E[:, :self.block_size] @ Hinv[i1:i2, i2:]

        return (W_out * M).to(dtype=W.dtype, device=W.device)

    def prune_linear_unstruct(self, W: torch.Tensor, module_name: str, k: int) -> torch.Tensor:
        out_features, in_features = W.shape
        if in_features % self.block_size > 0:
            raise Exception(f"Column size {in_features} must be divisible by block size: {self.block_size}")

        num_masks = in_features // self.mask_block
        prune_k_per_mask = int(math.ceil(k / num_masks))
        keep_k_per_mask_row = self.mask_block - int(math.ceil(prune_k_per_mask / out_features))

        def compute_mask(scores: torch.Tensor) -> torch.Tensor:
            topk_idx = scores.topk(keep_k_per_mask_row, dim=1, largest=True).indices
            M_seg = torch.zeros_like(scores)
            M_seg.scatter_(1, topk_idx, 1.0)
            return M_seg

        return self._prune(W, module_name, compute_mask=compute_mask)

    def prune_linear_nm(self, W: torch.Tensor, module_name: str, n: int, m: int) -> torch.Tensor:
        out_features, in_features = W.shape
        if in_features % self.block_size > 0:
            raise Exception(f"Column size {in_features} must be divisible by block size: {self.block_size}")

        if self.mask_block % m > 0:
            raise Exception(f"Mask block {self.mask_block} must be divisible by m: {m}")

        def compute_mask(scores: torch.Tensor):
            M_seg = torch.zeros_like(scores)

            for g_start in range(0, self.mask_block, m):
                g_end = g_start + m
                g_scores = scores[:, g_start:g_end]  # [out, m]
                topk_idx = g_scores.topk(n, dim=1, largest=True).indices  # [out, n]
                M_seg_group = torch.zeros_like(g_scores)
                M_seg_group.scatter_(1, topk_idx, 1.0)
                M_seg[:, g_start:g_end] = M_seg_group
            return M_seg

        return self._prune(W, module_name, compute_mask=compute_mask)


class WandaPruner(BaseActAwareUnstructuredPruner):
    def __init__(self):
        self.acts_diag: dict[str, torch.Tensor] = {}

    def add_linear_activations(self, module_name: str, X: torch.Tensor) -> None:
        X_flat = X.reshape(-1, X.shape[-1]).float()
        norm2 = (X_flat * X_flat).sum(dim=0).cpu()

        if module_name not in self.acts_diag:
            self.acts_diag[module_name] = norm2
        else:
            self.acts_diag[module_name].add_(norm2)

    def clear_linear_activations(self, module_name: str) -> None:
        act_diag = self.acts_diag.get(module_name, None)
        if act_diag is not None:
            del self.acts_diag[module_name]

    def _get_scores(self, W: torch.Tensor, module_name: str) -> torch.Tensor:
        H_diag = self.acts_diag[module_name].to(device=W.device, dtype=torch.float32)
        scores = W.abs().to(dtype=torch.float32) * H_diag.sqrt_().unsqueeze(0)
        return scores

    def prune_linear_unstruct(self, W: torch.Tensor, module_name: str, k: int) -> torch.Tensor:
        scores = self._get_scores(W, module_name)

        k_per_row = int(math.ceil(k / W.shape[0]))
        topk_idx = scores.topk(k_per_row, dim=1, largest=False).indices
        M = torch.ones_like(W, dtype=torch.bool)
        M.scatter_(1, topk_idx, False)

        return W * M.to(dtype=W.dtype)

    def prune_linear_nm(self, W: torch.Tensor, module_name: str, n: int, m: int) -> torch.Tensor:
        scores = self._get_scores(W, module_name)

        in_features = W.shape[-1]

        if in_features % m > 0:
            raise ValueError(f"In_features {in_features} must be divisible by m: {m}")

        score_blocks = scores.reshape(-1, in_features // m, m)
        weight_blocks = W.reshape(-1, in_features // m, m)

        topk_idx = torch.topk(score_blocks, k=n, dim=-1, largest=True, sorted=False).indices
        M = torch.zeros_like(score_blocks, dtype=torch.bool)
        M.scatter_(-1, topk_idx, True)
        pruned_W = weight_blocks * M.to(dtype=W.dtype)
        return pruned_W.reshape(W.shape)


def get_c4_calib_dataloader(
        tokenizer: PreTrainedTokenizerBase,
        n_samples: int,
        seqlen: int,
        batch_size: int,
        random_seed: int = 69,
        pbar: bool = True
) -> DataLoader:
    if n_samples % batch_size > 0:
        raise ValueError(f"n_samples {n_samples} must be divisible by batch_size: {batch_size}")

    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    rng = random.Random(random_seed)

    segments_input_ids: list[list[int]] = []
    segments_attention: list[list[int]] = []

    sample_iterable = range(n_samples)
    for _ in tqdm(sample_iterable, desc="Sampling C4 segments") if pbar else sample_iterable:
        while True:
            idx = rng.randrange(len(traindata))
            enc = tokenizer(
                traindata[idx]["text"],
                add_special_tokens=False,
                return_attention_mask=False,
                return_tensors=None,
            )
            ids = enc["input_ids"]
            if len(ids) >= seqlen:
                break

        L = len(ids)
        start = rng.randint(0, L - seqlen)
        seg = ids[start:start+seqlen]

        segments_input_ids.append(seg)
        segments_attention.append([1] * seqlen)

    calib_dataset = Dataset.from_dict(
        {"input_ids": segments_input_ids, "attention_mask": segments_attention}
    )

    def collate_fn(features: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    calib_dataloader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return calib_dataloader


@torch.no_grad()
def validate_perplexity_wikitext(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        seqlen: int,
        batch_size: int,
        pbar: bool = True
):
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    input_ids = testenc.input_ids
    n_samples = input_ids.numel() // seqlen
    input_ids = input_ids[:, :n_samples * seqlen].view(n_samples, seqlen)

    dataset = TensorDataset(input_ids)

    def collate_fn(batch):
        return {"input_ids": torch.stack([item[0] for item in batch])}

    val_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    if isinstance(model, OPTForCausalLM):
        forwarder = OptModelForwarder(model, pbar=pbar)
    else:
        raise NotImplementedError(f"Validation not implemented for {type(model)}")

    forwarder.embed(val_dataloader)
    num_layers = len(model.model.decoder.layers)
    for layer_idx in range(num_layers):
        forwarder.forward_layer(layer_idx)

    logits_cpu = forwarder.output_logits()

    # 4. Batched GPU Loss Calculation
    device = torch.device("cuda")

    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_tokens = 0

    # Iterate in batches to keep GPU memory stable
    for i in tqdm(range(0, n_samples, batch_size), desc="Calculating PPL", disable=not pbar):
        j = min(i + batch_size, n_samples)

        # Move batch to GPU and convert to float32 for stability
        batch_logits = logits_cpu[i:j].to(device).float()
        batch_targets = input_ids[i:j].to(device)

        # Shift: Predict next token
        shift_logits = batch_logits[:, :-1, :].contiguous()
        shift_labels = batch_targets[:, 1:].contiguous()

        # Compute sum of losses for this batch
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        total_nll += loss.item()
        total_tokens += (j - i) * (seqlen - 1)

        # Explicit cleanup to prevent VRAM accumulation
        del batch_logits, batch_targets, shift_logits, shift_labels
        # torch.cuda.empty_cache() # Uncomment if you are extremely tight on VRAM

    # Final Perplexity: exp(average log-likelihood)
    ppl = math.exp(total_nll / total_tokens)

    return ppl


@torch.no_grad()
def get_model_sparsity(model: torch.nn.Module) -> float:
    total_params = 0
    nonzero_params = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        numel = param.numel()
        total_params += numel
        nonzero = torch.count_nonzero(param)
        nonzero_params += nonzero.item()

    return 1 - nonzero_params / total_params


def get_linear_sparsity(model: torch.nn.Module) -> float:
    total_params = 0
    nonzero_params = 0

    if isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
        for layer in layers:
            pruned_modules: list[torch.nn.Linear] = [
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.q_proj,
                layer.self_attn.out_proj,
                layer.fc1,
                layer.fc2
            ]
            for module in pruned_modules:
                numel = module.weight.numel()
                total_params += numel
                nonzero = torch.count_nonzero(module.weight)
                nonzero_params += nonzero.item()
    else:
        raise NotImplementedError(f"Linear sparsity calculation not implemented for {type(model)}")
    return 1 - nonzero_params / total_params


def prune_opt_unstructured(
    pruner: BaseUnstructuredPruner,
    model_name: str = "facebook/opt-350m",
    n_samples: int = 128,
    seqlen: int = 2048,
    batch_size: int = 16,
    sparsity: str | float = 0.5,
    random_seed: int = 69,
    dtype: torch.dtype = torch.bfloat16,
):
    print("-" * 30)
    print("HYPERPARAMETERS")
    print(f"Pruner          : {type(pruner).__name__}")
    print(f"Model           : {model_name}")
    print(f"Sparsity        : {sparsity}")
    print(f"N Samples       : {n_samples}")
    print(f"Seqlen          : {seqlen}")
    print(f"Batch Size      : {batch_size}")
    print(f"Dtype           : {dtype}")
    print(f"Seed            : {random_seed}")
    print("-" * 30)

    # --- 2. Setup ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    ppl_initial = validate_perplexity_wikitext(
        model,
        tokenizer,
        seqlen=seqlen,
        batch_size=batch_size,
        pbar=True
    )
    print(f"\nInitial Dense PPL: {ppl_initial:.2f}")

    calib_dataloader = get_c4_calib_dataloader(
        tokenizer,
        n_samples=n_samples,
        seqlen=seqlen,
        batch_size=batch_size,
        random_seed=random_seed
    )


    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    pruner.prune_model(
        model,
        sparsity=sparsity,
        calib_dataloader=calib_dataloader,
        pbar=True
    )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    duration = end_time - start_time
    linear_sparsity = get_linear_sparsity(model)
    total_sparsity = get_model_sparsity(model)

    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("\n" + "=" * 40)
    print("PRUNING SUMMARY")
    print("-" * 40)
    print(f"Pruning Duration : {duration:.2f} seconds")
    print(f"Peak VRAM Usage  : {peak_mem:.2f} GB")
    print(f"Linear sparsity  : {linear_sparsity * 100:.2f}%")
    print(f"Actual Sparsity  : {total_sparsity * 100:.2f}%")

    ppl_final = validate_perplexity_wikitext(
        model,
        tokenizer,
        seqlen=seqlen,
        batch_size=batch_size,
        pbar=True
    )
    print(f"Final Pruned PPL : {ppl_final:.2f}")
    print("=" * 40)


if __name__ == "__main__":
    for model_name in ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b", "facebook/opt-13b"]:
        for pruner_cls in [MagnitudePruner, WandaPruner, SparseGPTPruner]:
            for sparsity in [0.5, '2:4', '4:8']:
                pruner = pruner_cls()
                prune_opt_unstructured(pruner=pruner, model_name=model_name, sparsity=sparsity)
