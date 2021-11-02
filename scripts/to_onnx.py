'''
Convert saved model to onnx
'''

import os
import pathlib
import datetime
import argparse
import numpy as np
import torch
import torch.onnx
import onnx
import nnio
from shutil import copyfile
from onnxmltools.utils.float16_converter import convert_float_to_float16
import onnxmltools

import pf2


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main(config):
    # Default name
    config.out_name = config.out_name or f'pf2'

    # Create lit module
    if config.load_from is not None:
        lit_module = pf2.pl_module.LitPet.load_from_checkpoint(checkpoint_path=config.load_from, strict=False)
    else:
        print('\n\nWARNING!!!!\nArgument --load_from is not set!!!!!!\n\n')
        lit_module = pf2.pl_module.LitPet(**config.__dict__)
    lit_module.eval()

    # Create path
    now = datetime.datetime.now()
    out_path = os.path.join(args.out_path, f'{now.strftime("%Y.%m.%d")}_{config.out_name}')
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file
    copyfile(config.config, os.path.join(out_path, 'config.yaml'))

    # Convert to openvino script
    openvino_command = f'''
docker run --rm -it \\
    -v /etc/timezone:/etc/timezone:ro \\
    -v /etc/localtime:/etc/localtime:ro \\
    -v {pathlib.Path(out_path).absolute()}:/input \\
    openvino/ubuntu18_dev \\
    python3 deployment_tools/model_optimizer/mo.py \\
    --input_model /input/{config.out_name}_op{config.opset}.onnx \\
    --model_name {config.out_name}_fp16 \\
    --data_type FP16 \\
    --output_dir /input/ \\
    --input_shape "[1,6,{config.image_h},{config.image_w}]"
'''

    # Create readme file
    readme_path = os.path.join(out_path, 'readme.md')
    with open(readme_path, 'w') as readme:
        readme.write('# Optical flow model\n\n')
        readme.write(f'Exported at {now.strftime("%H:%M:%S %d.%m.%Y")}\n\n')
        readme.write(f'`train_datasets = {config.train_datasets}`\n')
        readme.write(f'`dev_datasets = {config.dev_datasets}`\n')
        readme.write(f'`test_datasets = {config.test_datasets}`\n\n')
        nnio_preproc = f'''preprocessing = nnio.Preprocessing(
    resize=({config.image_w}, {config.image_h}),
    channels_first=True,
    divide_by_255=True,
    dtype='float32',
    batch_dimension=True
)'''
        readme.write(f'## For nnio:\n```\n{nnio_preproc}\n```\n')

    # Save model using pickle
    if config.save_pkl:
        pkl_path = os.path.join(out_path, f'{config.out_name}.pkl')
        torch.save(lit_module, pkl_path)
        print('Model saved as', pkl_path)

    # Convert model to torchscript
    if config.save_torchscript:
        # Remove all None attributes of the model object
        remove_attributes = []
        for key, value in vars(lit_module).items():
            if value is None:
                remove_attributes.append(key)
        for key in remove_attributes:
            delattr(lit_module, key)

        with torch.jit.optimized_execution(True):
            # Save in full precision
            # model = torch.jit.script(lit_module)
            sample_image = torch.randn(1, 6, config.image_h, config.image_w, requires_grad=True)
            model = torch.jit.trace(lit_module, sample_image)
            ts_model_path = os.path.join(out_path, f'{config.out_name}.pt')
            model.save(ts_model_path)
            print('Saved as', ts_model_path)

    # Convert model to onnx
    if config.save_onnx:
        sample_image = torch.randn(
            1, 6, config.image_h, config.image_w, requires_grad=True)
        torch_out = lit_module(sample_image)
        onnx_model_path = os.path.join(out_path, f'{config.out_name}_op{config.opset}.onnx')
        print('Exporting model as:', onnx_model_path)
        lit_module.to_onnx(
            onnx_model_path,
            sample_image,
            export_params=True,
            opset_version=config.opset,
            do_constant_folding=True,
            input_names = ['image'],
            output_names = ['flow'],
            dynamic_axes={ # variable length axes
                'image': {0: 'batch_size'},
                'flow': {0 : 'batch_size'}
            }
        )

        # Check model
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)

        # Check runtime
        onnx_model = nnio.ONNXModel(onnx_model_path)
        for i in range(10):
            ort_outs, info = onnx_model(to_numpy(sample_image), return_info=True)
            print('ONNX run info:', info)
        print('Torch out:', ort_outs.shape)
        print('ONNX out: ',to_numpy(torch_out).shape)
        if np.allclose(ort_outs, to_numpy(torch_out), rtol=1e-3, atol=1e-3):
            print('Model is working correctly')
        else:
            print('Torch out:', ort_outs)
            print('ONNX out: ',to_numpy(torch_out))
            raise BaseException('ONNX model gave different results from torch model''s')

        print('\nTo convert model to openVINO, please run:\n', openvino_command)

    print('Success!')
    print('\nAlso check out', readme_path)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Convert saved model to onnx.')

    parser.add_argument('--config', type=str,
                        help='configuration file in yaml format (ex.: configs/train/kitti_oakd_d455.yaml)')
    parser.add_argument('--load_from', type=str,
                        help='trained model saved using pytorch-lightning')
    parser.add_argument('--out_path', type=str,
                        default='./onnx_output',
                        help='path to the output folder')
    parser.add_argument('--out_name', type=str,
                        default=None,
                        help='name of the output .onnx file')
    parser.add_argument('--opset', type=int,
                        default=12,
                        help='ONNX opset version')
    parser.add_argument('--save_onnx', type=bool,
                        default=False,
                        help='Save onnx model')
    parser.add_argument('--save_torchscript', type=bool,
                        default=False,
                        help='Save scripted torch model')
    parser.add_argument('--save_pkl', type=bool,
                        default=False,
                        help='Save pickled torch model')

    parser = pf2.pl_module.LitPet.add_argparse_args(parser)
    parser = pf2.datasets.PetDataModule.add_argparse_args(parser)
    parser = pf2.utils.initialization.set_argparse_defaults_from_yaml(parser)

    args = parser.parse_args()

    # Run program
    main(args)
