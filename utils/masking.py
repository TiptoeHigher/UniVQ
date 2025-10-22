import torch


# class TriangularCausalMask():
#     def __init__(self, B, L, device="cpu"):
#         mask_shape = [B, 1, L, L]
#         with torch.no_grad():
#             self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
#
#     @property
#     def mask(self):
#         return self._mask
#
#
# class ProbMask():
#     def __init__(self, B, H, L, index, scores, device="cpu"):
#         _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
#         _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
#         indicator = _mask_ex[torch.arange(B)[:, None, None],
#                     torch.arange(H)[None, :, None],
#                     index, :].to(device)
#         self._mask = indicator.view(scores.shape).to(device)
#
#     @property
#     def mask(self):
#         return self._mask


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        # 使用 torch.ones_like() 来创建 mask，可以直接在 GPU 上创建
        _mask = torch.ones_like(scores, dtype=torch.bool, device=device).triu(1)
        # 使用 torch.gather() 来选择元素,而不是 _mask_ex.expand()
        indicator = _mask[torch.arange(B)[:, None, None],
                         torch.arange(H)[None, :, None],
                         index]
        self._mask = indicator

    @property
    def mask(self):
        return self._mask
