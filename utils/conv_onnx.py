#!/usr/bin/env python3
"""
ONNX Convolution Model Generator

Generate ONNX models with a single convolution operation with customizable parameters.
"""

import onnx
from onnx import helper, TensorProto
import numpy as np
import argparse

# Data type mappings
DTYPE_MAP = {
    'float32': (TensorProto.FLOAT, np.float32),
    'float16': (TensorProto.FLOAT16, np.float16),
    'int8': (TensorProto.INT8, np.int8),
}

def create_conv_onnx_model(args):
    """Create an ONNX model with a single convolution operation."""
    
    # Get data type information
    onnx_dtype, np_dtype = DTYPE_MAP[args.data_type]
    
    # Define input and output names
    input_name = "input"
    weight_name = "weight"
    output_name = "output"
    
    # Parse padding, strides, and dilations
    if len(args.padding) == 1:
        pads = [args.padding[0]] * 4  # [top, left, bottom, right]
    elif len(args.padding) == 2:
        pads = [args.padding[0], args.padding[1], args.padding[0], args.padding[1]]  # [h, w, h, w]
    elif len(args.padding) == 4:
        pads = args.padding  # [top, left, bottom, right]
    else:
        raise ValueError("Padding must have 1, 2, or 4 values")
    
    # Handle strides - only allow 1 or 2 values
    if len(args.strides) == 1:
        strides = [args.strides[0]] * 2
    elif len(args.strides) == 2:
        strides = args.strides
    else:
        raise ValueError(f"Strides must have 1 or 2 values, got {len(args.strides)}")
    
    # Handle dilations - only allow 1 or 2 values
    if len(args.dilations) == 1:
        dilations = [args.dilations[0]] * 2
    elif len(args.dilations) == 2:
        dilations = args.dilations
    else:
        raise ValueError(f"Dilations must have 1 or 2 values, got {len(args.dilations)}")
    
    # Calculate output dimensions (already validated in parse_args)
    out_height = ((args.input_height + pads[0] + pads[2] - 
                   dilations[0] * (args.filter_height - 1) - 1) // strides[0]) + 1
    out_width = ((args.input_width + pads[1] + pads[3] - 
                  dilations[1] * (args.filter_width - 1) - 1) // strides[1]) + 1
    
    # Create input tensor
    input_tensor = helper.make_tensor_value_info(
        input_name,
        onnx_dtype,
        [args.batch_size, args.input_channels, args.input_height, args.input_width]
    )
    
    # Create weight shape and initializer
    weight_shape = [args.output_channels, args.input_channels // args.group, 
                    args.filter_height, args.filter_width]
    
    # Generate weight data based on data type
    if args.data_type == 'int8':
        # For int8, use random integers in the valid range
        weight_data = np.random.randint(-127, 127, size=weight_shape, dtype=np_dtype)
    else:
        # For float types, use random normal distribution
        weight_data = np.random.randn(*weight_shape).astype(np_dtype)
    
    # Convert weight data for ONNX tensor
    if args.data_type == 'float16':
        # For float16, we need to convert to float32 for the list
        weight_list = weight_data.astype(np.float32).flatten().tolist()
    else:
        weight_list = weight_data.flatten().tolist()
    
    weight_initializer = helper.make_tensor(
        weight_name,
        onnx_dtype,
        weight_shape,
        weight_list
    )
    
    # Create output tensor
    output_tensor = helper.make_tensor_value_info(
        output_name,
        onnx_dtype,
        [args.batch_size, args.output_channels, out_height, out_width]
    )
    
    # Create Conv node
    conv_node = helper.make_node(
        "Conv",
        inputs=[input_name, weight_name],
        outputs=[output_name],
        kernel_shape=[args.filter_height, args.filter_width],
        pads=pads,
        strides=strides,
        dilations=dilations,
        group=args.group
    )
    
    # Create the graph
    graph = helper.make_graph(
        [conv_node],           # nodes
        "convolution_graph",   # graph name
        [input_tensor],        # inputs
        [output_tensor],       # outputs
        [weight_initializer]   # initializers
    )
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name="onnx-convolution-generator"
    )
    
    # Set opset version
    model.opset_import[0].version = 13
    
    # Verify the model
    onnx.checker.check_model(model)
    
    # Save the model
    onnx.save(model, args.output_file)
    
    # Print configuration
    print(f"✓ ONNX model saved as '{args.output_file}'")
    print(f"\nModel Configuration:")
    print(f"  Data type:    {args.data_type}")
    print(f"  Input shape:  [{args.batch_size}, {args.input_channels}, {args.input_height}, {args.input_width}]")
    print(f"  Weight shape: {weight_shape}")
    print(f"  Output shape: [{args.batch_size}, {args.output_channels}, {out_height}, {out_width}]")
    print(f"\nConvolution Parameters:")
    print(f"  Kernel:    {args.filter_height}x{args.filter_width}")
    print(f"  Padding:   {pads}")
    print(f"  Strides:   {strides}")
    print(f"  Dilations: {dilations}")
    print(f"  Groups:    {args.group}")

def validate_output_dims(args):
    """Validate that the convolution parameters produce valid output dimensions."""
    # Parse padding
    if len(args.padding) == 1:
        pads = [args.padding[0]] * 4
    elif len(args.padding) == 2:
        pads = [args.padding[0], args.padding[1], args.padding[0], args.padding[1]]
    elif len(args.padding) == 4:
        pads = args.padding
    else:
        return False, "Padding must have 1, 2, or 4 values"
    
    # Parse strides
    if len(args.strides) == 1:
        strides = [args.strides[0]] * 2
    elif len(args.strides) == 2:
        strides = args.strides
    else:
        return False, f"Strides must have 1 or 2 values, got {len(args.strides)}"
    
    # Parse dilations
    if len(args.dilations) == 1:
        dilations = [args.dilations[0]] * 2
    elif len(args.dilations) == 2:
        dilations = args.dilations
    else:
        return False, f"Dilations must have 1 or 2 values, got {len(args.dilations)}"
    
    # Calculate output dimensions
    out_height = ((args.input_height + pads[0] + pads[2] - 
                   dilations[0] * (args.filter_height - 1) - 1) // strides[0]) + 1
    out_width = ((args.input_width + pads[1] + pads[3] - 
                  dilations[1] * (args.filter_width - 1) - 1) // strides[1]) + 1
    
    if out_height <= 0 or out_width <= 0:
        return False, (f"Invalid convolution parameters: output dimensions would be "
                      f"height={out_height}, width={out_width}. "
                      f"Please check your padding, stride, kernel size, and dilation values.\n"
                      f"  Formula: out = floor((in + pad_top + pad_bottom - dilation*(kernel-1) - 1) / stride) + 1\n"
                      f"  Height: floor(({args.input_height} + {pads[0]} + {pads[2]} - {dilations[0]}*({args.filter_height}-1) - 1) / {strides[0]}) + 1 = {out_height}\n"
                      f"  Width:  floor(({args.input_width} + {pads[1]} + {pads[3]} - {dilations[1]}*({args.filter_width}-1) - 1) / {strides[1]}) + 1 = {out_width}")
    
    return True, f"Output dimensions: {out_height}x{out_width}"

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate ONNX model with a single convolution operation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 3x3 convolution
  python conv_onnx.py --input-channels 64 --output-channels 128

  # 1x1 pointwise convolution with float16
  python conv_onnx.py -ic 256 -oc 64 -fh 1 -fw 1 -p 0 -dt float16

  # Strided convolution for downsampling
  python conv_onnx.py -ic 3 -oc 64 -fh 7 -fw 7 -p 3 -s 2

  # Depthwise convolution with int8 quantization
  python conv_onnx.py -ic 64 -oc 64 -g 64 -dt int8

  # Dilated convolution
  python conv_onnx.py -ic 128 -oc 256 -d 2
        """
    )
    
    # Input dimensions
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--input-channels', '-ic', type=int, default=64,
                        help='Number of input channels (default: 64)')
    parser.add_argument('--output-channels', '-oc', type=int, default=128,
                        help='Number of output channels (default: 128)')
    parser.add_argument('--input-height', '-ih', type=int, default=128,
                        help='Input height (default: 128)')
    parser.add_argument('--input-width', '-iw', type=int, default=128,
                        help='Input width (default: 128)')
    
    # Filter dimensions
    parser.add_argument('--filter-height', '-fh', type=int, default=3,
                        help='Filter/kernel height (default: 3)')
    parser.add_argument('--filter-width', '-fw', type=int, default=3,
                        help='Filter/kernel width (default: 3)')
    
    # Convolution parameters
    parser.add_argument('--padding', '-p', type=int, nargs='+', default=[1],
                        help='Padding values. Can be: single value (same for all), '
                             '2 values (height, width), or 4 values (top, left, bottom, right) '
                             '(default: 1)')
    parser.add_argument('--strides', '-s', type=int, nargs='+', default=[1],
                        help='Stride values. Must be: single value (same for H and W) or '
                             '2 values (height, width) (default: 1)')
    parser.add_argument('--dilations', '-d', type=int, nargs='+', default=[1],
                        help='Dilation values. Must be: single value (same for H and W) or '
                             '2 values (height, width) (default: 1)')
    parser.add_argument('--group', '-g', type=int, default=1,
                        help='Number of groups for grouped convolution (default: 1)')
    
    # Data type
    parser.add_argument('--data-type', '-dt', type=str, default='float32',
                        choices=['float32', 'float16', 'int8'],
                        help='Data type for tensors: float32, float16, or int8 (default: float32)')
    
    # Output file
    parser.add_argument('--output-file', '-o', type=str, default='convolution_model.onnx',
                        help='Output ONNX file name (default: convolution_model.onnx)')
    
    args = parser.parse_args()
    
    # Basic validation
    if args.input_channels % args.group != 0:
        parser.error(f"Input channels ({args.input_channels}) must be divisible by groups ({args.group})")
    if args.output_channels % args.group != 0:
        parser.error(f"Output channels ({args.output_channels}) must be divisible by groups ({args.group})")
    if len(args.padding) not in [1, 2, 4]:
        parser.error("Padding must have 1, 2, or 4 values")
    if len(args.strides) not in [1, 2]:
        parser.error(f"Strides must have 1 or 2 values, got {len(args.strides)}")
    if len(args.dilations) not in [1, 2]:
        parser.error(f"Dilations must have 1 or 2 values, got {len(args.dilations)}")
    if args.batch_size <= 0:
        parser.error("Batch size must be positive")
    if any(x <= 0 for x in [args.input_channels, args.output_channels, 
                             args.input_height, args.input_width,
                             args.filter_height, args.filter_width]):
        parser.error("All dimension parameters must be positive")
    if any(x <= 0 for x in args.strides):
        parser.error("All stride values must be positive")
    if any(x <= 0 for x in args.dilations):
        parser.error("All dilation values must be positive")
    if any(x < 0 for x in args.padding):
        parser.error("All padding values must be non-negative")
    
    # Validate output dimensions using the convolution formula
    valid, message = validate_output_dims(args)
    if not valid:
        parser.error(message)
    
    return args

if __name__ == "__main__":
    args = parse_args()
    try:
        create_conv_onnx_model(args)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
