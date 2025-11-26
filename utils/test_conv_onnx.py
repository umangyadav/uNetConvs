#!/usr/bin/env python3
"""
Tests for conv_onnx.py - ONNX Convolution Model Generator
"""

import pytest
import os
import tempfile
import onnx
import numpy as np
from types import SimpleNamespace

from conv_onnx import (
    create_conv_onnx_model,
    validate_output_dims,
    DTYPE_MAP,
)


def get_conv_node_attr(conv_node, attr_name):
    """Helper to get a Conv node attribute by name."""
    for attr in conv_node.attribute:
        if attr.name == attr_name:
            if attr_name == 'group':
                return attr.i
            else:
                return list(attr.ints)
    return None


def verify_onnx_model(
    model,
    expected_input_shape,
    expected_output_shape,
    expected_weight_shape,
    expected_kernel_shape,
    expected_pads,
    expected_strides,
    expected_dilations,
    expected_group,
    expected_input_dtype,
    expected_output_dtype=None,
    expected_weight_dtype=None,
):
    """
    Comprehensive verification of ONNX model attributes.
    
    Args:
        model: Loaded ONNX model
        expected_input_shape: Expected input tensor shape [N, C, H, W]
        expected_output_shape: Expected output tensor shape [N, C, H, W]
        expected_weight_shape: Expected weight tensor shape [OC, IC/G, KH, KW]
        expected_kernel_shape: Expected kernel shape [KH, KW]
        expected_pads: Expected padding [top, left, bottom, right]
        expected_strides: Expected strides [H, W]
        expected_dilations: Expected dilations [H, W]
        expected_group: Expected group count
        expected_input_dtype: Expected input data type (onnx.TensorProto enum)
        expected_output_dtype: Expected output data type (defaults to input dtype)
        expected_weight_dtype: Expected weight data type (defaults to input dtype)
    """
    if expected_output_dtype is None:
        expected_output_dtype = expected_input_dtype
    if expected_weight_dtype is None:
        expected_weight_dtype = expected_input_dtype

    # Validate model
    onnx.checker.check_model(model)

    # Check graph structure
    assert len(model.graph.node) == 1, "Model should have exactly one node"
    conv_node = model.graph.node[0]
    assert conv_node.op_type == 'Conv', f"Expected Conv node, got {conv_node.op_type}"

    # Check input/output names
    assert conv_node.input[0] == 'input', f"Expected input name 'input', got {conv_node.input[0]}"
    assert conv_node.input[1] == 'weight', f"Expected weight name 'weight', got {conv_node.input[1]}"
    assert conv_node.output[0] == 'output', f"Expected output name 'output', got {conv_node.output[0]}"

    # Check input shape and dtype
    input_info = model.graph.input[0]
    input_shape = [d.dim_value for d in input_info.type.tensor_type.shape.dim]
    input_dtype = input_info.type.tensor_type.elem_type
    assert input_shape == expected_input_shape, f"Input shape mismatch: {input_shape} != {expected_input_shape}"
    assert input_dtype == expected_input_dtype, f"Input dtype mismatch: {input_dtype} != {expected_input_dtype}"

    # Check output shape and dtype
    output_info = model.graph.output[0]
    output_shape = [d.dim_value for d in output_info.type.tensor_type.shape.dim]
    output_dtype = output_info.type.tensor_type.elem_type
    assert output_shape == expected_output_shape, f"Output shape mismatch: {output_shape} != {expected_output_shape}"
    assert output_dtype == expected_output_dtype, f"Output dtype mismatch: {output_dtype} != {expected_output_dtype}"

    # Check weight shape and dtype
    weight_init = model.graph.initializer[0]
    weight_shape = list(weight_init.dims)
    weight_dtype = weight_init.data_type
    assert weight_shape == expected_weight_shape, f"Weight shape mismatch: {weight_shape} != {expected_weight_shape}"
    assert weight_dtype == expected_weight_dtype, f"Weight dtype mismatch: {weight_dtype} != {expected_weight_dtype}"

    # Check Conv node attributes
    kernel_shape = get_conv_node_attr(conv_node, 'kernel_shape')
    pads = get_conv_node_attr(conv_node, 'pads')
    strides = get_conv_node_attr(conv_node, 'strides')
    dilations = get_conv_node_attr(conv_node, 'dilations')
    group = get_conv_node_attr(conv_node, 'group')

    assert kernel_shape == expected_kernel_shape, f"Kernel shape mismatch: {kernel_shape} != {expected_kernel_shape}"
    assert pads == expected_pads, f"Pads mismatch: {pads} != {expected_pads}"
    assert strides == expected_strides, f"Strides mismatch: {strides} != {expected_strides}"
    assert dilations == expected_dilations, f"Dilations mismatch: {dilations} != {expected_dilations}"
    assert group == expected_group, f"Group mismatch: {group} != {expected_group}"

    # Check opset version
    assert model.opset_import[0].version == 13, f"Opset version mismatch: {model.opset_import[0].version} != 13"


class TestValidateOutputDims:
    """Test cases for validate_output_dims function."""

    def test_valid_default_params(self):
        """Test with default parameters - should produce valid output."""
        args = SimpleNamespace(
            input_height=128,
            input_width=128,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[1],
            dilations=[1],
        )
        valid, message = validate_output_dims(args)
        assert valid is True
        assert "128x128" in message

    def test_valid_strided_conv(self):
        """Test with stride=2 - should halve output dimensions."""
        args = SimpleNamespace(
            input_height=128,
            input_width=128,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[2],
            dilations=[1],
        )
        valid, message = validate_output_dims(args)
        assert valid is True
        assert "64x64" in message

    def test_valid_asymmetric_padding(self):
        """Test with 2-value padding (height, width)."""
        args = SimpleNamespace(
            input_height=128,
            input_width=256,
            filter_height=3,
            filter_width=5,
            padding=[1, 2],  # H padding=1, W padding=2
            strides=[1],
            dilations=[1],
        )
        valid, message = validate_output_dims(args)
        assert valid is True
        assert "128x256" in message

    def test_valid_four_value_padding(self):
        """Test with 4-value padding (top, left, bottom, right)."""
        args = SimpleNamespace(
            input_height=64,
            input_width=64,
            filter_height=3,
            filter_width=3,
            padding=[1, 1, 1, 1],  # top, left, bottom, right
            strides=[1],
            dilations=[1],
        )
        valid, message = validate_output_dims(args)
        assert valid is True
        assert "64x64" in message

    def test_valid_dilated_conv(self):
        """Test dilated convolution."""
        args = SimpleNamespace(
            input_height=128,
            input_width=128,
            filter_height=3,
            filter_width=3,
            padding=[2],  # Need more padding for dilation
            strides=[1],
            dilations=[2],
        )
        valid, message = validate_output_dims(args)
        assert valid is True
        assert "128x128" in message

    def test_invalid_negative_output_height(self):
        """Test that negative output height is detected."""
        args = SimpleNamespace(
            input_height=5,
            input_width=128,
            filter_height=7,
            filter_width=3,
            padding=[0],  # No padding
            strides=[1],
            dilations=[1],
        )
        valid, message = validate_output_dims(args)
        assert valid is False
        assert "Invalid convolution parameters" in message

    def test_invalid_negative_output_width(self):
        """Test that negative output width is detected."""
        args = SimpleNamespace(
            input_height=128,
            input_width=5,
            filter_height=3,
            filter_width=7,
            padding=[0],
            strides=[1],
            dilations=[1],
        )
        valid, message = validate_output_dims(args)
        assert valid is False
        assert "Invalid convolution parameters" in message

    def test_invalid_padding_count(self):
        """Test invalid padding count (3 values)."""
        args = SimpleNamespace(
            input_height=128,
            input_width=128,
            filter_height=3,
            filter_width=3,
            padding=[1, 1, 1],  # Invalid: 3 values
            strides=[1],
            dilations=[1],
        )
        valid, message = validate_output_dims(args)
        assert valid is False
        assert "Padding must have 1, 2, or 4 values" in message

    def test_invalid_strides_count(self):
        """Test invalid strides count (3 values)."""
        args = SimpleNamespace(
            input_height=128,
            input_width=128,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[1, 1, 1],  # Invalid: 3 values
            dilations=[1],
        )
        valid, message = validate_output_dims(args)
        assert valid is False
        assert "Strides must have 1 or 2 values" in message

    def test_invalid_dilations_count(self):
        """Test invalid dilations count (3 values)."""
        args = SimpleNamespace(
            input_height=128,
            input_width=128,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[1],
            dilations=[1, 1, 1],  # Invalid: 3 values
        )
        valid, message = validate_output_dims(args)
        assert valid is False
        assert "Dilations must have 1 or 2 values" in message

    def test_asymmetric_strides(self):
        """Test with different strides for height and width."""
        args = SimpleNamespace(
            input_height=128,
            input_width=256,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[2, 4],  # H stride=2, W stride=4
            dilations=[1],
        )
        valid, message = validate_output_dims(args)
        assert valid is True
        # 128/2=64, 256/4=64
        assert "64x64" in message


class TestCreateConvOnnxModel:
    """Test cases for create_conv_onnx_model function."""

    @pytest.fixture
    def temp_output_file(self):
        """Create a temporary file for ONNX output."""
        fd, path = tempfile.mkstemp(suffix='.onnx')
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

    def test_basic_model_creation_float32(self, temp_output_file):
        """Test basic model creation with float32."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=64,
            output_channels=128,
            input_height=128,
            input_width=128,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[1],
            dilations=[1],
            group=1,
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        assert os.path.exists(temp_output_file)
        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[1, 64, 128, 128],
            expected_output_shape=[1, 128, 128, 128],
            expected_weight_shape=[128, 64, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[1, 1, 1, 1],
            expected_strides=[1, 1],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )

    def test_model_creation_float16(self, temp_output_file):
        """Test model creation with float16."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=32,
            output_channels=64,
            input_height=64,
            input_width=64,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[1],
            dilations=[1],
            group=1,
            data_type='float16',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[1, 32, 64, 64],
            expected_output_shape=[1, 64, 64, 64],
            expected_weight_shape=[64, 32, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[1, 1, 1, 1],
            expected_strides=[1, 1],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT16,
        )

    def test_model_creation_int8(self, temp_output_file):
        """Test model creation with int8."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=32,
            output_channels=64,
            input_height=64,
            input_width=64,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[1],
            dilations=[1],
            group=1,
            data_type='int8',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[1, 32, 64, 64],
            expected_output_shape=[1, 64, 64, 64],
            expected_weight_shape=[64, 32, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[1, 1, 1, 1],
            expected_strides=[1, 1],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.INT8,
        )

    def test_pointwise_conv_1x1(self, temp_output_file):
        """Test 1x1 pointwise convolution."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=256,
            output_channels=64,
            input_height=56,
            input_width=56,
            filter_height=1,
            filter_width=1,
            padding=[0],
            strides=[1],
            dilations=[1],
            group=1,
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[1, 256, 56, 56],
            expected_output_shape=[1, 64, 56, 56],
            expected_weight_shape=[64, 256, 1, 1],
            expected_kernel_shape=[1, 1],
            expected_pads=[0, 0, 0, 0],
            expected_strides=[1, 1],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )

    def test_strided_conv_downsampling(self, temp_output_file):
        """Test strided convolution for downsampling."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=3,
            output_channels=64,
            input_height=224,
            input_width=224,
            filter_height=7,
            filter_width=7,
            padding=[3],
            strides=[2],
            dilations=[1],
            group=1,
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[1, 3, 224, 224],
            expected_output_shape=[1, 64, 112, 112],
            expected_weight_shape=[64, 3, 7, 7],
            expected_kernel_shape=[7, 7],
            expected_pads=[3, 3, 3, 3],
            expected_strides=[2, 2],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )

    def test_depthwise_conv(self, temp_output_file):
        """Test depthwise convolution (groups == input_channels)."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=64,
            output_channels=64,
            input_height=56,
            input_width=56,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[1],
            dilations=[1],
            group=64,  # Depthwise: groups == channels
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        # Weight shape for depthwise: [output_channels, input_channels/groups, KH, KW]
        # = [64, 64/64, 3, 3] = [64, 1, 3, 3]
        verify_onnx_model(
            model,
            expected_input_shape=[1, 64, 56, 56],
            expected_output_shape=[1, 64, 56, 56],
            expected_weight_shape=[64, 1, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[1, 1, 1, 1],
            expected_strides=[1, 1],
            expected_dilations=[1, 1],
            expected_group=64,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )

    def test_grouped_conv(self, temp_output_file):
        """Test grouped convolution."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=128,
            output_channels=256,
            input_height=28,
            input_width=28,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[1],
            dilations=[1],
            group=4,  # 4 groups
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        # Weight shape: [output_channels, input_channels/groups, KH, KW]
        # = [256, 128/4, 3, 3] = [256, 32, 3, 3]
        verify_onnx_model(
            model,
            expected_input_shape=[1, 128, 28, 28],
            expected_output_shape=[1, 256, 28, 28],
            expected_weight_shape=[256, 32, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[1, 1, 1, 1],
            expected_strides=[1, 1],
            expected_dilations=[1, 1],
            expected_group=4,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )

    def test_dilated_conv(self, temp_output_file):
        """Test dilated convolution."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=128,
            output_channels=256,
            input_height=64,
            input_width=64,
            filter_height=3,
            filter_width=3,
            padding=[2],  # Padding for dilation=2
            strides=[1],
            dilations=[2],
            group=1,
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[1, 128, 64, 64],
            expected_output_shape=[1, 256, 64, 64],
            expected_weight_shape=[256, 128, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[2, 2, 2, 2],
            expected_strides=[1, 1],
            expected_dilations=[2, 2],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )

    def test_asymmetric_kernel(self, temp_output_file):
        """Test asymmetric kernel size."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=64,
            output_channels=64,
            input_height=128,
            input_width=128,
            filter_height=1,
            filter_width=7,
            padding=[0, 3],  # Different H and W padding
            strides=[1],
            dilations=[1],
            group=1,
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[1, 64, 128, 128],
            expected_output_shape=[1, 64, 128, 128],
            expected_weight_shape=[64, 64, 1, 7],
            expected_kernel_shape=[1, 7],
            expected_pads=[0, 3, 0, 3],
            expected_strides=[1, 1],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )

    def test_batch_size_greater_than_one(self, temp_output_file):
        """Test with batch size > 1."""
        args = SimpleNamespace(
            batch_size=8,
            input_channels=64,
            output_channels=128,
            input_height=32,
            input_width=32,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[1],
            dilations=[1],
            group=1,
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[8, 64, 32, 32],
            expected_output_shape=[8, 128, 32, 32],
            expected_weight_shape=[128, 64, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[1, 1, 1, 1],
            expected_strides=[1, 1],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )

    def test_four_value_padding(self, temp_output_file):
        """Test with 4-value asymmetric padding."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=64,
            output_channels=64,
            input_height=32,
            input_width=32,
            filter_height=3,
            filter_width=3,
            padding=[1, 2, 1, 2],  # top=1, left=2, bottom=1, right=2
            strides=[1],
            dilations=[1],
            group=1,
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        # Output: H = (32 + 1 + 1 - 3) / 1 + 1 = 32
        #         W = (32 + 2 + 2 - 3) / 1 + 1 = 34
        verify_onnx_model(
            model,
            expected_input_shape=[1, 64, 32, 32],
            expected_output_shape=[1, 64, 32, 34],
            expected_weight_shape=[64, 64, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[1, 2, 1, 2],
            expected_strides=[1, 1],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )


class TestDtypeMap:
    """Test the DTYPE_MAP constant."""

    def test_float32_mapping(self):
        """Test float32 mapping."""
        onnx_dtype, np_dtype = DTYPE_MAP['float32']
        assert onnx_dtype == onnx.TensorProto.FLOAT
        assert np_dtype == np.float32

    def test_float16_mapping(self):
        """Test float16 mapping."""
        onnx_dtype, np_dtype = DTYPE_MAP['float16']
        assert onnx_dtype == onnx.TensorProto.FLOAT16
        assert np_dtype == np.float16

    def test_int8_mapping(self):
        """Test int8 mapping."""
        onnx_dtype, np_dtype = DTYPE_MAP['int8']
        assert onnx_dtype == onnx.TensorProto.INT8
        assert np_dtype == np.int8


class TestConvNodeAttributes:
    """Test that Conv node has correct attributes."""

    @pytest.fixture
    def temp_output_file(self):
        """Create a temporary file for ONNX output."""
        fd, path = tempfile.mkstemp(suffix='.onnx')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_conv_node_structure(self, temp_output_file):
        """Test the overall structure of the Conv node."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=64,
            output_channels=128,
            input_height=32,
            input_width=32,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[2],
            dilations=[1],
            group=1,
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[1, 64, 32, 32],
            expected_output_shape=[1, 128, 16, 16],
            expected_weight_shape=[128, 64, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[1, 1, 1, 1],
            expected_strides=[2, 2],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )

    def test_opset_version(self, temp_output_file):
        """Test that opset version is set correctly."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=64,
            output_channels=64,
            input_height=32,
            input_width=32,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[1],
            dilations=[1],
            group=1,
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        # Opset version check is included in verify_onnx_model
        verify_onnx_model(
            model,
            expected_input_shape=[1, 64, 32, 32],
            expected_output_shape=[1, 64, 32, 32],
            expected_weight_shape=[64, 64, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[1, 1, 1, 1],
            expected_strides=[1, 1],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def temp_output_file(self):
        """Create a temporary file for ONNX output."""
        fd, path = tempfile.mkstemp(suffix='.onnx')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_minimum_input_size(self, temp_output_file):
        """Test with minimum viable input size."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=1,
            output_channels=1,
            input_height=3,
            input_width=3,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[1],
            dilations=[1],
            group=1,
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[1, 1, 3, 3],
            expected_output_shape=[1, 1, 3, 3],
            expected_weight_shape=[1, 1, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[1, 1, 1, 1],
            expected_strides=[1, 1],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )

    def test_large_kernel(self, temp_output_file):
        """Test with a large kernel (AlexNet-style first layer)."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=3,
            output_channels=64,
            input_height=224,
            input_width=224,
            filter_height=11,
            filter_width=11,
            padding=[5],
            strides=[4],
            dilations=[1],
            group=1,
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[1, 3, 224, 224],
            expected_output_shape=[1, 64, 56, 56],
            expected_weight_shape=[64, 3, 11, 11],
            expected_kernel_shape=[11, 11],
            expected_pads=[5, 5, 5, 5],
            expected_strides=[4, 4],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )

    def test_no_padding(self, temp_output_file):
        """Test convolution with no padding (valid convolution)."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=64,
            output_channels=64,
            input_height=32,
            input_width=32,
            filter_height=3,
            filter_width=3,
            padding=[0],
            strides=[1],
            dilations=[1],
            group=1,
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[1, 64, 32, 32],
            expected_output_shape=[1, 64, 30, 30],
            expected_weight_shape=[64, 64, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[0, 0, 0, 0],
            expected_strides=[1, 1],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )

    def test_asymmetric_strides(self, temp_output_file):
        """Test with different strides for height and width."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=64,
            output_channels=64,
            input_height=64,
            input_width=128,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[1, 2],  # H stride=1, W stride=2
            dilations=[1],
            group=1,
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[1, 64, 64, 128],
            expected_output_shape=[1, 64, 64, 64],
            expected_weight_shape=[64, 64, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[1, 1, 1, 1],
            expected_strides=[1, 2],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )

    def test_asymmetric_dilations(self, temp_output_file):
        """Test with different dilations for height and width."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=64,
            output_channels=64,
            input_height=64,
            input_width=64,
            filter_height=3,
            filter_width=3,
            padding=[2, 4],  # Different padding for different dilations
            strides=[1],
            dilations=[2, 4],  # H dilation=2, W dilation=4
            group=1,
            data_type='float32',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        # Output H: (64 + 2 + 2 - 2*(3-1) - 1) / 1 + 1 = (64 + 4 - 4 - 1) / 1 + 1 = 64
        # Output W: (64 + 4 + 4 - 4*(3-1) - 1) / 1 + 1 = (64 + 8 - 8 - 1) / 1 + 1 = 64
        verify_onnx_model(
            model,
            expected_input_shape=[1, 64, 64, 64],
            expected_output_shape=[1, 64, 64, 64],
            expected_weight_shape=[64, 64, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[2, 4, 2, 4],
            expected_strides=[1, 1],
            expected_dilations=[2, 4],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT,
        )

    def test_large_batch_with_float16(self, temp_output_file):
        """Test large batch size with float16."""
        args = SimpleNamespace(
            batch_size=32,
            input_channels=256,
            output_channels=512,
            input_height=14,
            input_width=14,
            filter_height=3,
            filter_width=3,
            padding=[1],
            strides=[1],
            dilations=[1],
            group=1,
            data_type='float16',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[32, 256, 14, 14],
            expected_output_shape=[32, 512, 14, 14],
            expected_weight_shape=[512, 256, 3, 3],
            expected_kernel_shape=[3, 3],
            expected_pads=[1, 1, 1, 1],
            expected_strides=[1, 1],
            expected_dilations=[1, 1],
            expected_group=1,
            expected_input_dtype=onnx.TensorProto.FLOAT16,
        )

    def test_depthwise_with_int8(self, temp_output_file):
        """Test depthwise convolution with int8."""
        args = SimpleNamespace(
            batch_size=1,
            input_channels=32,
            output_channels=32,
            input_height=28,
            input_width=28,
            filter_height=5,
            filter_width=5,
            padding=[2],
            strides=[2],
            dilations=[1],
            group=32,
            data_type='int8',
            output_file=temp_output_file,
        )
        create_conv_onnx_model(args)

        model = onnx.load(temp_output_file)

        verify_onnx_model(
            model,
            expected_input_shape=[1, 32, 28, 28],
            expected_output_shape=[1, 32, 14, 14],
            expected_weight_shape=[32, 1, 5, 5],
            expected_kernel_shape=[5, 5],
            expected_pads=[2, 2, 2, 2],
            expected_strides=[2, 2],
            expected_dilations=[1, 1],
            expected_group=32,
            expected_input_dtype=onnx.TensorProto.INT8,
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
