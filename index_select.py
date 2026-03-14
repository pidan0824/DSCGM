from torch import Tensor

def index_select(inputs: Tensor, indices: Tensor, dim: int) -> Tensor:
    """Advanced index_select that supports multi-dimensional indices.

    Unlike torch.index_select, `indices` can be multi-dimensional. The indexed dimension
    is expanded to match the shape of `indices`.
    """
    outputs = inputs.index_select(dim, indices.view(-1))

    if indices.dim() > 1:
        if dim < 0:
            dim += inputs.dim()
        output_shape = inputs.shape[:dim] + indices.shape + inputs.shape[dim + 1 :]
        outputs = outputs.view(*output_shape)

    return outputs
