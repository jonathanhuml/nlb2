#!/usr/bin/env python3

# This file taken from gpytorch with MIT license https://github.com/cornellius-gp/gpytorch/blob/011679a806bc2fe825e3fee8865f82a9e8152c8a/gpytorch/utils/toeplitz.py
"""MIT License

Copyright (c) 2017 Jake Gardner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
import torch
from torch.fft import fft, ifft


def softplus(x):
    return torch.log(1 + torch.exp(x))


def inv_softplus(x):
    return torch.log(torch.exp(x) - 1)


def _mul_broadcast_shape(*shapes, error_msg=None):
    """Compute dimension suggested by multiple tensor indices (supports broadcasting)"""

    # Pad each shape so they have the same number of dimensions
    num_dims = max(len(shape) for shape in shapes)
    shapes = tuple(
        [1] * (num_dims - len(shape)) + list(shape) for shape in shapes)

    # Make sure that each dimension agrees in size
    final_size = []
    for size_by_dim in zip(*shapes):
        non_singleton_sizes = tuple(size for size in size_by_dim if size != 1)
        if len(non_singleton_sizes):
            if any(size != non_singleton_sizes[0]
                   for size in non_singleton_sizes):
                if error_msg is None:
                    raise RuntimeError(
                        "Shapes are not broadcastable for mul operation")
                else:
                    raise RuntimeError(error_msg)
            final_size.append(non_singleton_sizes[0])
        # In this case - all dimensions are singleton sizes
        else:
            final_size.append(1)

    return torch.Size(final_size)


def _matmul_broadcast_shape(shape_a, shape_b, error_msg=None):
    """Compute dimension of matmul operation on shapes (supports broadcasting)"""
    m, n, p = shape_a[-2], shape_a[-1], shape_b[-1]

    if len(shape_b) == 1:
        if n != p:
            if error_msg is None:
                raise RuntimeError(
                    f"Incompatible dimensions for matmul: {shape_a} and {shape_b}"
                )
            else:
                raise RuntimeError(error_msg)
        return shape_a[:-1]

    if n != shape_b[-2]:
        if error_msg is None:
            raise RuntimeError(
                f"Incompatible dimensions for matmul: {shape_a} and {shape_b}")
        else:
            raise RuntimeError(error_msg)

    tail_shape = torch.Size([m, p])

    # Figure out batch shape
    batch_shape_a = shape_a[:-2]
    batch_shape_b = shape_b[:-2]
    if batch_shape_a == batch_shape_b:
        bc_shape = batch_shape_a
    else:
        bc_shape = _mul_broadcast_shape(batch_shape_a, batch_shape_b)
    return bc_shape + tail_shape


def toeplitz_matmul(toeplitz_column, toeplitz_row, tensor):
    """
    Performs multiplication T * M where the matrix T is Toeplitz.
    Args:
        - toeplitz_column (vector n or b x n) - First column of the Toeplitz matrix T.
        - toeplitz_row (vector n or b x n) - First row of the Toeplitz matrix T.
        - tensor (matrix n x p or b x n x p) - Matrix or vector to multiply the Toeplitz matrix with.
    Returns:
        - tensor (n x p or b x n x p) - The result of the matrix multiply T * M.
    """
    if toeplitz_column.size() != toeplitz_row.size():
        raise RuntimeError(
            "c and r should have the same length (Toeplitz matrices are necessarily square)."
        )

    toeplitz_shape = torch.Size((*toeplitz_column.shape, toeplitz_row.size(-1)))
    output_shape = _matmul_broadcast_shape(toeplitz_shape,
                                                        tensor.shape)
    broadcasted_t_shape = output_shape[:-1] if tensor.dim(
    ) > 1 else output_shape

    if tensor.ndimension() == 1:
        tensor = tensor.unsqueeze(-1)
    toeplitz_column = toeplitz_column.expand(*broadcasted_t_shape)
    toeplitz_row = toeplitz_row.expand(*broadcasted_t_shape)
    tensor = tensor.expand(*output_shape)

    if not torch.equal(toeplitz_column[..., 0], toeplitz_row[..., 0]):
        raise RuntimeError(
            "The first column and first row of the Toeplitz matrix should have "
            "the same first element, otherwise the value of T[0,0] is ambiguous. "
            "Got: c[0]={} and r[0]={}".format(toeplitz_column[0],
                                              toeplitz_row[0]))

    if type(toeplitz_column) != type(toeplitz_row) or type(
            toeplitz_column) != type(tensor):
        raise RuntimeError("The types of all inputs to ToeplitzMV must match.")

    *batch_shape, orig_size, num_rhs = tensor.size()
    r_reverse = toeplitz_row[..., 1:].flip(dims=(-1,))

    c_r_rev = torch.zeros(*batch_shape,
                          orig_size + r_reverse.size(-1),
                          dtype=tensor.dtype,
                          device=tensor.device)
    c_r_rev[..., :orig_size] = toeplitz_column
    c_r_rev[..., orig_size:] = r_reverse

    temp_tensor = torch.zeros(*batch_shape,
                              2 * orig_size - 1,
                              num_rhs,
                              dtype=toeplitz_column.dtype,
                              device=toeplitz_column.device)
    temp_tensor[..., :orig_size, :] = tensor

    fft_M = fft(temp_tensor.transpose(-1, -2).contiguous())
    fft_c = fft(c_r_rev).unsqueeze(-2).expand_as(fft_M)
    fft_product = fft_M.mul_(fft_c)

    output = ifft(fft_product).real.transpose(-1, -2)
    output = output[..., :orig_size, :]
    return output


def sym_toeplitz_matmul(toeplitz_column, tensor):
    """
    Performs a matrix-matrix multiplication TM where the matrix T is symmetric Toeplitz.
    Args:
        - toeplitz_column (vector n) - First column of the symmetric Toeplitz matrix T.
        - matrix (matrix n x p) - Matrix or vector to multiply the Toeplitz matrix with.
    Returns:
        - tensor
    """
    return toeplitz_matmul(toeplitz_column, toeplitz_column, tensor)
