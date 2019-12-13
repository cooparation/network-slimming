import torch
import torch.nn as nn
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.ssd_lite_prune import create_mobilenetv2_ssd_lite_prune
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import logging
import sys
import os

parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument('--model_path', type=str, help="The network weights file.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--save", default=".", type=str, help="The directory to store evaluation results.")
parser.add_argument('--prune_percent', default=0.5, type=float, help='get the threshold by the prune ratio')
parser.add_argument('--mb2_width_mult', default=1.0, type=float, help='Width Multiplifier for MobilenetV2')

args = parser.parse_args()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

def update_cfg(orgCfg, inChannelCfg):
    newCfg = []
    if len(orgCfg) == len(inChannelCfg) - 1:
        for i in range(len(inChannelCfg) - 1): # remove last channel
            layer_cfg = [inChannelCfg[i], orgCfg[i][1], orgCfg[i][2], orgCfg[i][3]]
            newCfg.append(layer_cfg)
    else:
        print('Error: dim error:', len(orgCfg), len(inChannelCfg))
        exit(-1)
    return newCfg

if __name__ == '__main__':

    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]

    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    elif args.net == 'mb2-ssd-lite-prune':
        input_channel = 32
        last_channel = 1280
        invert_cfg = [# hidden_c, out_c, s, expand_ratio
                [input_channel, 16, 1, 1],
                [96,  24,  2, -1],
                [144, 24,  2, -1],
                [144, 32,  2, -1],
                [192, 32,  2, -1],
                [192, 32,  2, -1],
                [192, 64,  2, -1],
                [384, 64,  2, -1],
                [384, 64,  2, -1],
                [384, 64,  2, -1],
                [384, 96,  1, -1],
                [576, 96,  1, -1],
                [576, 96,  1, -1],
                [576, 160, 2, -1], # conv_13/expand
                [960, 160, 2, -1],
                [960, 160, 2, -1],
                [960, 320, 1, -1]]
        net = create_mobilenetv2_ssd_lite_prune(len(class_names),
                width_mult=args.mb2_width_mult, cfg=invert_cfg,
                in_channel=input_channel, last_channel=last_channel)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    #print('model ===================== \n')
    #print(net)
    #print('=========================== \n')

    percent = args.prune_percent
    x = torch.zeros(1, 3, 300, 300, dtype=torch.float, requires_grad=False)
    if args.model_path:
        model_path = args.model_path
    else:
        print('Error: the model should be offered')
        exit(-1)
    with torch.no_grad():
        net.to(DEVICE)
        net.eval()
        net.load_state_dict(torch.load(model_path))
        #print('====================== net\n', net)

        print('############## get the batchnorm data ##############')
        print(' ============== named parameters =================')
        for param in net.named_parameters():
            print(param[0])
        print(' ============== batchnorm info =================')
        total = 0
        for name, module in net.named_modules():
            #print('======== ', name, module, ' =========\n')
            if 'dw' not in name and 'linear' not in name \
               and 'base_net' in name and 'bn' in name and isinstance(module, nn.BatchNorm2d):
                #print('======== ', name, module, ' =========\n')
                total += module.weight.data.shape[0]
            #if 'dw' in name or 'linear' not in name:
            #    print('======== ', name, module, ' =========\n')
        #bn = np.zeros(total)
        bn = torch.zeros(total)
        index = 0
        for name, module in net.named_modules():
            #print('======== ', name, module, ' =========\n')
            if 'dw' not in name and 'linear' not in name \
               and 'base_net' in name and 'bn' in name and isinstance(module, nn.BatchNorm2d):
                #print('======== ', name, module, ' =========\n')
                size = module.weight.data.shape[0]
                #print('Size:', size)
                bn[index:(index+size)] = module.weight.data.abs().clone()
                index += size

        print('############## get the threshold ##############')
        sorted_bn, i = torch.sort(bn)
        threshold_index = int(total * percent)
        threshold = sorted_bn[threshold_index]
        print('------- threshold:', threshold)

        ########## get the mask ##############
        pruned = 0
        cfg = []
        cfg_mask = []
        for name, module in net.named_modules():
            if 'dw' not in name and 'linear' not in name \
               and 'base_net' in name and 'bn' in name and isinstance(module, nn.BatchNorm2d):
               # if 'Conv_1' in name or 'conv_13' in name: # keep Conv_1 and conv_13 the same
               #     weight_copy = module.weight.data.abs().clone()
               #     mask = weight_copy.gt(-1000.0).float().cuda()
               # else:
               #     weight_copy = module.weight.data.abs().clone()
               #     mask = weight_copy.gt(threshold.cuda()).float().cuda()
                    #mask = weight_copy.gt(threshold).float().cuda()

                weight_copy = module.weight.data.abs().clone()
                mask = weight_copy.gt(threshold.cuda()).float().cuda()

                pruned = pruned + mask.shape[0] - torch.sum(mask)
                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                print('layer name: {:30s} \t total channel: {:d} \t remaining channel: {:d}'.format(name, mask.shape[0], int(torch.sum(mask))))
            elif isinstance(module, nn.MaxPool2d):
                print('max polling')
                cfg.append('M')

        ### save model just for testing
        torch.save(net.state_dict(), os.path.join(args.save, 'pruned_mask.pth'))

        pruned_ratio = pruned/total
        print('pruned config:', cfg)
        print('------- pruned_ratio:', pruned_ratio)
        new_invert_cfg = update_cfg(invert_cfg, cfg)
        new_input_channel = cfg[0]
        new_last_channel = cfg[-1] # Conv_1

        print('org invert config:', invert_cfg)
        print('new invert config:', new_invert_cfg)

        new_net = create_mobilenetv2_ssd_lite_prune(len(class_names),
                width_mult=args.mb2_width_mult, cfg=new_invert_cfg,
                in_channel=new_input_channel, last_channel=new_last_channel)

        print('############## prune the weights ##############')
        layer_id_in_cfg = 0
        start_mask = torch.ones(3)
        end_mask = cfg_mask[layer_id_in_cfg]
        conv_count = 0

        for (name0, module0), (name1, module1) in zip(net.named_modules(), new_net.named_modules()):
            if 'base_net' in name0 and 'bn' in name0 and isinstance(module0, nn.BatchNorm2d):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1,(1,))

                if 'linear' in name0: # update the mask
                    module1.weight.data = module0.weight.data.clone()
                    module1.bias.data = module0.bias.data.clone()
                    module1.running_mean = module0.running_mean.clone()
                    module1.running_var = module0.running_var.clone()
                    layer_id_in_cfg += 1
                    start_mask = end_mask.clone()
                    if layer_id_in_cfg < len(cfg_mask):
                        end_mask = cfg_mask[layer_id_in_cfg]
                    input_shape = module0.weight.data.shape[0]
                    print('{:20s} In shape: {:d}, Out shape: {:d}.'.format(name0, idx1.size, input_shape, input_shape))
                    continue
                else:
                    module1.weight.data = module0.weight.data[idx1.tolist()].clone()
                    module1.bias.data = module0.bias.data[idx1.tolist()].clone()
                    module1.running_mean = module0.running_mean[idx1.tolist()].clone()
                    module1.running_var = module0.running_var[idx1.tolist()].clone()
                    print('{:20s} In shape: {:d}, Out shape {:d}.'.format(name0, idx1.size, idx1.size))
                    continue
            elif 'base_net' in name0 and isinstance(module0, nn.Conv2d):
                if conv_count == 0 or 'Conv_1' in name0: # first layer = conv or Conv_1
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('{:20s} In shape: {:d}, Out shape {:d}.'.format(name0, idx0.size, idx1.size))
                    # module1.weight.data = module0.weight.data.clone() # first layer conv no bias
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    if not (module0.bias is None):
                        print(name0, 'has bias')
                        module1.bias.data = module0.bias.data[idx1.tolist()].clone()
                    w1 = module0.weight.data[idx1.tolist(), :, :, :].clone() # output channels change
                    module1.weight.data = w1.clone()
                    conv_count += 1
                    continue
                elif 'pw' in name0 and isinstance(module0, nn.Conv2d):
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('{:20s} In shape keep, Out shape {:d}.'.format(name0, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    if not (module0.bias is None):
                        print(name0, 'has bias')
                        module1.bias.data = module0.bias.data[idx1.tolist()].clone()
                    w1 = module0.weight.data[idx1.tolist(), :, :, :].clone() # output channels change
                    module1.weight.data = w1.clone()
                    conv_count += 1
                    continue
                elif 'dw' in name0 and isinstance(module0, nn.Conv2d):
                    # depthwise convolutions input_channel=out_channel, maybe changed
                    conv_count += 1
                    idx0 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('{:20s} In shape: {:d}, Out shape {:d}.'.format(name0, idx0.size, idx1.size))
                    #print(module0, ' shape ', module0.weight.shape)
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    if not (module0.bias is None):
                        print(name0, 'has bias')
                        module1.bias.data = module0.bias.data[idx1.tolist()].clone()

                    module1.weight.data = module0.weight.data[idx1.tolist(), :, :, :].clone()
                    continue
                elif 'linear' in name0 and isinstance(module0, nn.Conv2d):
                    # linear convolutions out_channel,input_channel maybe changed
                    conv_count += 1
                    idx0 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('{:20s} In shape: {:d}, Out shape keep.'.format(name0, idx0.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if not (module0.bias is None):
                        print(name0, 'has bias')
                        module1.bias.data = module0.bias.data.clone()
                    module1.weight.data = module0.weight.data[:, idx0.tolist(), :, :].clone()
                    continue
            elif 'extras' in name0 and 'Conv_1/pw' in name0 and isinstance(module0, nn.Conv2d):
                # extras/Conv_1/pw input-shape changed and output-shape keep
                if 'pw' in name0 and isinstance(module0, nn.Conv2d):
                    #idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    #idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[17].cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask[17].cpu().numpy())))
                    print('{:20s} In shape {:d}, Out shape keep.'.format(name0, idx0.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    if not (module0.bias is None):
                        print(name0, 'has bias')
                        module1.bias.data = module0.bias.data.clone()
                    w1 = module0.weight.data[:, idx0.tolist(), :, :].clone() # output channels change
                    module1.weight.data = w1.clone()
                    conv_count += 1
                    continue
            elif ('regression_headers' in name0 or 'classification_headers' in name0) \
                    and ('Conv_1' in name0 or 'conv_13' in name0):
                # (regression_headers|classification_headers)/(Conv_1|conv_13) input-shape changed and output-shape keep
                cfg_index = 13 if 'conv_13' in name0 else 17
                cfg_tmp = cfg_mask[cfg_index]
                if 'dw' in name0 and isinstance(module0, nn.Conv2d):
                    # depthwise convolutions input_channel=out_channel, maybe changed
                    conv_count += 1
                    idx0 = np.squeeze(np.argwhere(np.asarray(cfg_tmp.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(cfg_tmp.cpu().numpy())))
                    print('{:20s} In shape: {:d}, Out shape {:d}.'.format(name0, idx0.size, idx1.size))
                    #print(module0, ' shape ', module0.weight.shape)
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    if not (module0.bias is None):
                        print(name0, 'has bias')
                        module1.bias.data = module0.bias.data[idx1.tolist()].clone()
                    module1.weight.data = module0.weight.data[idx1.tolist(), :, :, :].clone()
                    continue
                elif 'bn' in name0 and isinstance(module0, nn.BatchNorm2d):
                    idx0 = np.squeeze(np.argwhere(np.asarray(cfg_tmp.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(cfg_tmp.cpu().numpy())))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1,(1,))
                    module1.weight.data = module0.weight.data[idx1.tolist()].clone()
                    module1.bias.data = module0.bias.data[idx1.tolist()].clone()
                    module1.running_mean = module0.running_mean[idx1.tolist()].clone()
                    module1.running_var = module0.running_var[idx1.tolist()].clone()
                    print('{:20s} In shape: {:d}, Out shape {:d}.'.format(name0, idx1.size, idx1.size))
                    continue
                elif 'linear' in name0 and isinstance(module0, nn.Conv2d):
                    # linear convolutions out_channel,input_channel maybe changed
                    conv_count += 1
                    idx0 = np.squeeze(np.argwhere(np.asarray(cfg_tmp.cpu().numpy())))
                    print('{:20s} In shape: {:d}, Out shape keep.'.format(name0, idx0.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if not (module0.bias is None):
                        print(name0, 'has bias')
                        module1.bias.data = module0.bias.data.clone()
                    module1.weight.data = module0.weight.data[:, idx0.tolist(), :, :].clone()
                    continue
            elif isinstance(module0, nn.BatchNorm2d):
                module1.weight.data = module0.weight.data.clone()
                module1.bias.data = module0.bias.data.clone()
                module1.running_mean = module0.running_mean.clone()
                module1.running_var = module0.running_var.clone()
                print('{:20s} {:} In shape keep, Out shape keep.'.format(name0, module0))
                continue
            elif isinstance(module0, nn.Conv2d):
                module1.weight.data = module0.weight.data.clone()
                if not (module0.bias is None):
                    print(name0, 'has bias')
                    module1.bias.data = module0.bias.data.clone()
                print('{:20s} {:} In shape keep, Out shape keep.'.format(name0, module0))
                continue

        ## save the results
        cfg_str = str()
        for i in range(len(new_invert_cfg)):
            cfg_str += '[{:5s}{:5s}{:3s}{:2s}],\n'\
                .format(str(new_invert_cfg[i][0]) + ','
                ,str(new_invert_cfg[i][1]) + ','
                ,str(new_invert_cfg[i][2]) + ','
                ,str(new_invert_cfg[i][3]))
        cfg_str += 'input_channel: {:}, last_channel:{:}'\
                .format(new_input_channel, new_last_channel)
        with open('pruned.txt', 'w') as file:
            file.writelines(cfg_str)

        torch.save(new_net.state_dict(), os.path.join(args.save, 'pruned.pth'))

        print('=====================================\n')
