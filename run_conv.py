import numpy as np
import onnx
import argparse
import openpyxl
import sys
from onnx import helper
from onnx import TensorProto
from onnx.numpy_helper import from_array
from onnx.checker import check_model
import os
import pandas as pd
from pandas import DataFrame
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', type=int, nargs="+")
    parser.add_argument('--filter', type=int, nargs="+")
    parser.add_argument('--padding', type=int, nargs="+")
    parser.add_argument('--stride', type=int, nargs="+")
    parser.add_argument('--dilation', type=int, nargs="+")
    parser.add_argument('--group', type=int)
    parser.add_argument('--padding_mode', type=int)
    return parser


def gen_graph(op_info, name):
    graph_def = helper.make_graph(op_info[0], name,
                                  op_info[1], op_info[2])
    model_def = helper.make_model(graph_def,
                                  producer_name=name)
    onnx.shape_inference.infer_shapes(model_def)
    check_model(model_def)
    onnx.save_model(model_def,
                    '{}.onnx'.format(name),
                    save_as_external_data=False,
                    location='{}.weight'.format(name),
                    size_threshold=0,
                    convert_attribute=True)


def gen_conv(activation_lens: list[int], filter_lens: list[int], padding: list[int], stride: list[int],  dilation: list[int], name: str):
    x = helper.make_tensor_value_info(
        '0', TensorProto.FLOAT16, activation_lens)
    y = helper.make_tensor_value_info('1', TensorProto.FLOAT16, filter_lens)
    out = helper.make_tensor_value_info(
        '2', TensorProto.FLOAT16, [None, None, None, None])

    node = onnx.helper.make_node('Conv',
                                 inputs=['0', '1'],
                                 outputs=['2'],
                                 dilations=dilation,
                                 strides=stride,
                                 pads=padding)

    gen_graph(([node], [x, y], [out]), name)


def set_env_flags(mlir_flag, layout_flag):
    os.environ["MIGRAPHX_ENABLE_NHWC"] = str(layout_flag)
    if mlir_flag:
        os.environ["MIGRAPHX_MLIR_USE_SPECIFIC_OPS"] = "dot,fused,convolution"
    else:
        os.environ["MIGRAPHX_DISABLE_MLIR"] = "1"
        os.environ["MIOPEN_FIND_ENFORCE"] = "3"


def unset_env_flags():
    os.environ.pop("MIGRAPHX_DISABLE_MLIR", None)
    os.environ.pop("MIGRAPHX_MLIR_USE_SPECIFIC_OPS", None)
    os.environ.pop("MIGRAPHX_ENABLE_NHWC", None)
    os.environ.pop("MIOPEN_FIND_ENFORCE", None)


def get_sheet_name(mlir_flag, layout_flag):
    sheet_name = "miopen"
    if mlir_flag:
        sheet_name = "mlir"
    if layout_flag:
        sheet_name += "_nhwc"
    else:
        sheet_name += "_nchw"
    return sheet_name


if __name__ == "__main__":
    conv_config_file = open(sys.argv[1], 'r')
    conv_configs = conv_config_file.readlines()
    parser = parse_args()
    layout_switch = [0, 1]
    mlir_switch = [0, 1]
    # create empty file first
    wb = openpyxl.Workbook()
    wb.save("unet_conv_summary.xlsx")
    for layout_flag in layout_switch:
        for mlir_flag in mlir_switch:
            set_env_flags(mlir_flag, layout_flag)
            onnx_counter = 0
            activation_list = []
            filter_list = []
            hip_fill_list = []
            padding_list = []
            stride_list = []
            dilation_list = []
            conv_perf_list = []
            layout_list = []
            total_time_list = []
            rate_list = []
            for config in conv_configs:
                args = parser.parse_args(config.split())
                print(args)
                activation_list.append(args.activation)
                filter_list.append(args.filter)
                padding_list.append(args.padding)
                stride_list.append(args.stride)
                dilation_list.append(args.dilation)
                name = "conv_unet_"+str(onnx_counter)
                onnx_counter = onnx_counter+1
                if not os.path.exists("name"):
                    gen_conv(args.activation, args.filter, args.padding,
                         args.stride, args.dilation, name)
                command_list = ['/home/umayadav/repo/AMDMIGraphX/build/bin/driver',
                                'perf', '--exhaustive-tune', '--iterations', '10000', name+".onnx"]
                env_copy = os.environ.copy()
                result = subprocess.run(
                    args=command_list, capture_output=True, env=env_copy).stdout.decode('utf-8')
                print(result[result.find("Summary:"):])
                summary = result[result.find("Summary:"):].splitlines()
                is_splitk = False
                for i in summary:
                    if i.startswith("gpu::convolution:"):
                        conv_perf_list.append(i.split(' ')[1])
                    elif i.startswith("gpu::code_object::mlir_convolution:"):
                        conv_perf_list.append(i.split(' ')[1])
                    elif i.startswith("hip::fill:"):
                        hip_fill_list.append(i.split(' ')[1])
                        is_splitk = True
                    elif i.startswith("Rate:"):
                        rate_list.append(i.split(' ')[1])
                    elif i.startswith("Total time:"):
                        total_time_list.append(i.split(' ')[2])
                    elif i.startswith("gpu::code_object::layout_kernel:"):
                        assert (layout_flag)
                        layout_list.append(i.split(' ')[1])
                if not layout_flag:
                    layout_list.append(0)
                if not is_splitk:
                    hip_fill_list.append(0)
                else:
                    is_splitk = False

            unset_env_flags()
            d = {'activation_size': activation_list, 'filter_size': filter_list, 'padding': padding_list,
                 'stride': stride_list, 'dilation': dilation_list, 'conv_time': conv_perf_list, 'hip::fill': hip_fill_list, 'layout_kernel': layout_list, 'rate': rate_list, 'total_time': total_time_list}
            df = DataFrame(data=d)
            with pd.ExcelWriter('unet_conv_summary.xlsx', engine='openpyxl', mode='a') as writer:
                sheet_name = get_sheet_name(
                    mlir_flag=mlir_flag, layout_flag=layout_flag)
                df.to_excel(writer, sheet_name=sheet_name)
