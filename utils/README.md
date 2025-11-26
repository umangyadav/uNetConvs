# ONNX Convolution Model Generator

A Python script to generate ONNX models containing a single convolution operation with customizable parameters. This tool is useful for testing, benchmarking, and prototyping convolution operations in various deep learning frameworks.

## Features

- Generate ONNX models with a single Conv operation
- Fully customizable convolution parameters
- Support for multiple data types: float32, float16, and int8
- Automatic output dimension calculation with validation
- Input parameter validation using convolution formula
- Model validation using ONNX checker
- Support for grouped, strided, dilated, and padded convolutions
- Flexible stride and dilation specifications

## Requirements

Install the required dependencies from the project root:

```bash
pip install -r ../requirements.txt
```

Or from the project root directory:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Generate a convolution model with default parameters:

```bash
python conv_onnx.py
```

This creates a model with:
- Input shape: [1, 64, 128, 128]
- Output channels: 128
- Kernel size: 3x3
- Padding: 1
- Stride: 1
- Dilation: 1

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--batch-size` | | 1 | Batch size |
| `--input-channels` | `-ic` | 64 | Number of input channels |
| `--output-channels` | `-oc` | 128 | Number of output channels |
| `--input-height` | `-ih` | 128 | Input height |
| `--input-width` | `-iw` | 128 | Input width |
| `--filter-height` | `-fh` | 3 | Filter/kernel height |
| `--filter-width` | `-fw` | 3 | Filter/kernel width |
| `--padding` | `-p` | 1 | Padding (see padding options below) |
| `--strides` | `-s` | 1 | Stride values |
| `--dilations` | `-d` | 1 | Dilation values |
| `--group` | `-g` | 1 | Number of groups for grouped convolution |
| `--data-type` | `-dt` | float32 | Data type: float32, float16, or int8 |
| `--output-file` | `-o` | convolution_model.onnx | Output ONNX file name |

#### Padding Options

Padding can be specified in three ways:
- **Single value**: Same padding for all sides (e.g., `--padding 1`)
- **Two values**: Different padding for height and width (e.g., `--padding 1 2`)
- **Four values**: Specific padding for top, left, bottom, right (e.g., `--padding 1 2 3 4`)

#### Stride and Dilation Options

Both strides and dilations must be specified as:
- **Single value**: Same for both height and width (e.g., `--strides 2`)
- **Two values**: Different for height and width (e.g., `--strides 2 1`)

All stride and dilation values must be positive integers. The script will error if more than 2 values are provided.

#### Data Type Options

The `--data-type` parameter supports three tensor data types:
- **float32**: Standard 32-bit floating point (default)
- **float16**: Half-precision floating point for reduced memory usage
- **int8**: 8-bit integer for quantized models and faster inference

## Examples

### Example 1: Standard 3x3 Convolution

```bash
python conv_onnx.py \
    --input-channels 64 \
    --output-channels 128 \
    --input-height 224 \
    --input-width 224 \
    --filter-height 3 \
    --filter-width 3 \
    --padding 1 \
    --strides 1
```

Creates a standard 3x3 convolution that maintains spatial dimensions.

### Example 2: 1x1 Pointwise Convolution

```bash
python conv_onnx.py \
    --input-channels 256 \
    --output-channels 64 \
    --input-height 56 \
    --input-width 56 \
    --filter-height 1 \
    --filter-width 1 \
    --padding 0
```

Creates a 1x1 convolution for channel reduction/expansion.

### Example 3: Strided Convolution for Downsampling

```bash
python conv_onnx.py \
    --input-channels 3 \
    --output-channels 64 \
    --input-height 224 \
    --input-width 224 \
    --filter-height 7 \
    --filter-width 7 \
    --padding 3 \
    --strides 2 \
    --output-file downsample_conv.onnx
```

Creates a 7x7 convolution with stride 2 (commonly used as first layer in ResNet).

### Example 4: Depthwise Separable Convolution

```bash
# Depthwise convolution (groups = input_channels)
python conv_onnx.py \
    --input-channels 64 \
    --output-channels 64 \
    --filter-height 3 \
    --filter-width 3 \
    --padding 1 \
    --group 64 \
    --output-file depthwise_conv.onnx
```

Creates a depthwise convolution where each channel is convolved separately.

### Example 5: Dilated/Atrous Convolution

```bash
python conv_onnx.py \
    --input-channels 128 \
    --output-channels 256 \
    --input-height 64 \
    --input-width 64 \
    --filter-height 3 \
    --filter-width 3 \
    --padding 2 \
    --dilations 2 \
    --output-file dilated_conv.onnx
```

Creates a dilated convolution with dilation rate 2 for increased receptive field.

### Example 6: Asymmetric Padding

```bash
python conv_onnx.py \
    --input-channels 32 \
    --output-channels 64 \
    --input-height 100 \
    --input-width 100 \
    --filter-height 5 \
    --filter-width 5 \
    --padding 2 3 2 3 \
    --output-file asymmetric_pad_conv.onnx
```

Creates a convolution with different padding on each side (top=2, left=3, bottom=2, right=3).

### Example 7: Different Strides for Height and Width

```bash
python conv_onnx.py \
    --input-channels 64 \
    --output-channels 128 \
    --input-height 256 \
    --input-width 128 \
    --filter-height 3 \
    --filter-width 3 \
    --padding 1 \
    --strides 2 1 \
    --output-file anisotropic_stride_conv.onnx
```

Creates a convolution with different strides (2 for height, 1 for width).

### Example 8: Large Kernel Convolution

```bash
python conv_onnx.py \
    --input-channels 3 \
    --output-channels 96 \
    --input-height 224 \
    --input-width 224 \
    --filter-height 11 \
    --filter-width 11 \
    --padding 5 \
    --strides 4 \
    --output-file large_kernel_conv.onnx
```

Creates an 11x11 convolution with stride 4 (similar to AlexNet's first layer).

### Example 9: Float16 Convolution

```bash
python conv_onnx.py \
    --input-channels 128 \
    --output-channels 256 \
    --input-height 56 \
    --input-width 56 \
    --filter-height 3 \
    --filter-width 3 \
    --padding 1 \
    --data-type float16 \
    --output-file fp16_conv.onnx
```

Creates a convolution with half-precision (float16) tensors for reduced memory usage.

### Example 10: INT8 Quantized Convolution

```bash
python conv_onnx.py \
    --input-channels 64 \
    --output-channels 128 \
    --input-height 128 \
    --input-width 128 \
    --filter-height 3 \
    --filter-width 3 \
    --padding 1 \
    --data-type int8 \
    --output-file int8_conv.onnx
```

Creates a convolution with INT8 quantization for inference acceleration.

### Example 11: Large Stride and Dilation Values

```bash
python conv_onnx.py \
    --input-channels 3 \
    --output-channels 64 \
    --input-height 256 \
    --input-width 256 \
    --filter-height 3 \
    --filter-width 3 \
    --padding 4 \
    --strides 4 \
    --dilations 3 \
    --output-file large_stride_dilation.onnx
```

Creates a convolution with stride=4 and dilation=3, useful for aggressive downsampling with expanded receptive field.

## Output Format

The script generates an ONNX model file containing:
- A single Conv node with the specified parameters
- Random weight initialization (using numpy's randn)
- Proper input and output tensor specifications
- ONNX opset version 13

The generated model can be:
- Loaded and executed with ONNX Runtime
- Converted to other formats (TensorFlow, PyTorch, etc.)
- Used for benchmarking and testing
- Imported into MIGraphX or other AMD tools

## Validation

The script automatically validates the generated model using ONNX's built-in checker to ensure:
- Correct tensor shapes
- Valid convolution parameters
- Proper ONNX format compliance

## Common Use Cases

1. **Performance Benchmarking**: Generate specific convolution configurations for performance testing
2. **Model Prototyping**: Quickly create convolution layers for experimentation
3. **Framework Testing**: Test framework support for various convolution configurations
4. **Educational Purpose**: Understand how different parameters affect output dimensions
5. **Compiler Testing**: Generate test cases for deep learning compilers like MIGraphX or rocMLIR

## Output Dimension Calculation

The output dimensions are automatically calculated using the formula:

```
out_height = floor((input_height + pad_top + pad_bottom - dilation_h * (filter_height - 1) - 1) / stride_h) + 1
out_width = floor((input_width + pad_left + pad_right - dilation_w * (filter_width - 1) - 1) / stride_w) + 1
```

## Troubleshooting

### Common Issues

1. **Group convolution errors**: Ensure input_channels and output_channels are divisible by the group parameter
2. **Negative output dimensions**: The script will validate parameters and show a detailed error with the formula:
   ```
   Invalid convolution parameters: output dimensions would be height=-1, width=-1.
   Formula: out = floor((in + pad_top + pad_bottom - dilation*(kernel-1) - 1) / stride) + 1
   Height: floor((5 + 0 + 0 - 1*(7-1) - 1) / 1) + 1 = -1
   ```
   This happens when the kernel is too large for the input with given padding/stride/dilation
3. **Memory issues with large tensors**: Be mindful of tensor sizes when using large input dimensions

## Running Tests

The test suite validates all functionality of the convolution model generator.

### Run All Tests

```bash
# From the utils directory
python -m pytest test_conv_onnx.py -v

# Or from the project root
python -m pytest utils/test_conv_onnx.py -v
```

### Run Specific Test Classes

```bash
# Test output dimension validation
python -m pytest test_conv_onnx.py::TestValidateOutputDims -v

# Test model creation
python -m pytest test_conv_onnx.py::TestCreateConvOnnxModel -v

# Test edge cases
python -m pytest test_conv_onnx.py::TestEdgeCases -v
```

### Run a Single Test

```bash
python -m pytest test_conv_onnx.py::TestCreateConvOnnxModel::test_depthwise_conv -v
```

### Test Coverage

The test suite covers:
- **Validation tests**: Output dimension calculations, parameter validation, error handling
- **Model creation tests**: float32/float16/int8 data types, various kernel sizes, grouped convolutions
- **Edge cases**: Minimum input sizes, large kernels, asymmetric strides/dilations

## License

This tool is provided as-is for educational and testing purposes.

## Contributing

Feel free to submit issues or pull requests for improvements and bug fixes.