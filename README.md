This benchmarks SDXL UNet Convs with both MIOpen and MLIR with NCHW and NHWC layouts. 

First make following changes : 
1. `run_conv.py` script generates individual convolution onnx files in FP16 data type. If you can change that if you wish.
2. Change path to `migraphx-driver` inside the script.

To run, follow these steps. 
1. `pip3 install -r requirements.txt`
2.  Build MIGraphX from source with required rocMLIR SHA and then do `export PYTHONPATH=/path/to/migraphx/build/lib`
3. `python3 run_conv.py uniq_convs.txt`. This would create a `unet_conv_summary.xlsx` file with 4 different sheets. `MIOpen_NCHW`, `MLIR_NCHW`, `MIOpen_NHWC`, `MLIR_NHWC`

To get Unique convolutions for a onnx file, you can instrument the `src/onnx/parse_convolution.cpp` inside the MIGraphX with following: 
```cpp
        op.from_value(values);
        std::cout << "--activation ";
        for(const auto& i : l0->get_shape().lens())
        {
            std::cout << i << " ";
        }
        std::cout << "--filter ";
        for(const auto& i : weights->get_shape().lens())
        {
            std::cout << i << " ";
        }
        value v                       = op.to_value();
        std::vector<size_t> v_padding = v["padding"].to_vector<size_t>();
        std::cout << "--padding ";
        for(const auto& i : v_padding)
        {
            std::cout << i << " ";
        }
        std::vector<size_t> v_strides = v["stride"].to_vector<size_t>();
        std::cout << "--stride ";
        for(const auto& i : v_strides)
        {
            std::cout << i << " ";
        }
        std::vector<size_t> v_dilation = v["dilation"].to_vector<size_t>();
        std::cout << "--dilation ";
        for(const auto& i : v_dilation)
        {
            std::cout << i << " ";
        }
        int v_padding_mode = v["padding_mode"].to<int>();
        std::cout << "--padding_mode " << v_padding_mode << " ";
        int v_group = v["group"].to<int>();
        std::cout << "--group " << v_group << "\n";
```

Then you can run `migraphx-driver read model.onnx |& tee convs.txt` and then do `cat convs.txt | sort -q |& tee uniq_convs.txt`


