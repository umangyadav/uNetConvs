This benchmarks SDXL UNet Convs with both MIOpen and MLIR with NCHW and NHWC layouts. 

To run follow these steps. 
1. `pip3 install -r requirements.txt`
2.  Build MIGraphX from source with required rocMLIR SHA and then do `export PYTHONPATH=/path/to/migraphx/build/lib`
3. `python3 run_conv.py uniq_convs.txt`. This would create a `unet_conv_summary.xlsx` file with 4 different sheets. `MIOpen_NCHW`, `MLIR_NCHW`, `MIOpen_NHWC`, `MLIR_NHWC`

