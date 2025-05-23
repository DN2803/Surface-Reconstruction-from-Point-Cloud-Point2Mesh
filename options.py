import argparse
import os
import numpy as np
import torch
import sys

MANIFOLD_DIR = r'~/code/Manifold/build'  # path to manifold software (https://github.com/hjwdzh/Manifold)


class Options:
    def __init__(self):
        self.args = None
        self.parser = self.create_parser()

    def create_parser(self):
        parser = argparse.ArgumentParser(description='Point2Mesh options')
        parser.add_argument('--save-path', type=str, default='./checkpoints/result', help='path to save results to')
        parser.add_argument('--input-pc', type=str, default='./data/pointcloud.ply', help='input point cloud')
        parser.add_argument('--initial-mesh', type=str, default='./data/pointcloud_hull.obj', help='initial mesh')
        # HYPER PARAMETERS - RECONSTRUCTION
        parser.add_argument('--torch-seed', type=int, metavar='N', default=5, help='torch random seed')
        parser.add_argument('--samples', type=int, metavar='N', default=25000,
                            help='number of points to sample reconstruction with')
        parser.add_argument('--begin-samples', type=int, metavar='N', default=15000, help='num pts to start with')
        parser.add_argument('--iterations', type=int, metavar='N', default=5000, help='number of iterations to do')
        parser.add_argument('--upsamp', type=int, metavar='N', default=1000, help='upsample each {upsamp} iteration')
        parser.add_argument('--max-faces', type=int, metavar='N', default=10000,
                            help='maximum number of faces to upsample to')
        parser.add_argument('--faces-to-part', nargs='+', default=[8000, 16000, 20000], type=int,
                            help='after how many faces to split')
        # HYPER PARAMETERS - NETWORK
        parser.add_argument('--lr', type=float, metavar='1eN', default=1.1e-4, help='learning rate')
        parser.add_argument('--ang-wt', type=float, metavar='1eN', default=1e-1,
                            help='weight of the cosine loss for normals')
        parser.add_argument('--res-blocks', type=int, metavar='N', default=3, help='')
        parser.add_argument('--leaky-relu', type=float, metavar='1eN', default=0.01, help='slope for leaky relu')
        parser.add_argument('--local-non-uniform', type=float, metavar='1eN', default=0.1,
                            help='weight of local non uniform loss')
        parser.add_argument('--gpu', type=int, metavar='N', default=0, help='gpu to use')
        parser.add_argument('--convs', nargs='+', default=[16, 32, 64, 64, 128], type=int, help='convs to do')
        parser.add_argument('--pools', nargs='+', default=[0.0, 0.0, 0.0, 0.0], type=float,
                            help='percent to pool from original resolution in each layer')
        parser.add_argument('--transfer-data', action='store_true', help='')
        parser.add_argument('--overlap', type=int, default=0, help='overlap for bfs')
        parser.add_argument('--global-step', action='store_true',
                            help='perform the optimization step after all the parts are forwarded (only matters if nparts > 2)')
        parser.add_argument('--manifold-res', default=100000, type=int, help='resolution for manifold upsampling')
        parser.add_argument('--unoriented', action='store_true',
                            help='take the normals loss term without any preferred orientation')
        parser.add_argument('--init-weights', type=float, default=0.002, help='initialize NN with this size')
        #
        parser.add_argument('--export-interval', type=int, metavar='N', default=100, help='export interval')
        parser.add_argument('--beamgap-iterations', type=int, default=0,
                            help='the # iters to which the beamgap loss will be calculated')
        parser.add_argument('--beamgap-modulo', type=int, default=1, help='skip iterations with beamgap loss'
                                                                          '; calc beamgap when:'
                                                                          ' iter % (--beamgap-modulo) == 0')
        parser.add_argument('--manifold-always', action='store_true',
                            help='always run manifold even when the maximum number of faces is reached')
        return parser

    def parse(self, args=None):
        """
        Parse arguments.

        args can be:
         - None: parse from sys.argv (excluding Jupyter args)
         - list of strings: like command line args
         - dict: keys are argument names (without --), values are argument values

        After parsing, will create save_path folder and write opt.txt
        """
        if args is None:
            # exclude -f=... and .json args that can come from Jupyter
            args_list = [arg for arg in sys.argv[1:] if not arg.startswith("-f=") and not arg.endswith(".json")]
            print("Parsing args from sys.argv:", args_list)
            self.args = self.parser.parse_args(args_list)
        elif isinstance(args, list):
            print("Parsing args from list:", args)
            self.args = self.parser.parse_args(args)
        elif isinstance(args, dict):
            # Convert dict to list of strings
            args_list = []
            for k, v in args.items():
                key = f"--{k.replace('_','-')}"
                if isinstance(v, bool):
                    # For flags
                    if v:
                        args_list.append(key)
                elif isinstance(v, list):
                    args_list.append(key)
                    args_list.extend(str(x) for x in v)
                else:
                    args_list.extend([key, str(v)])
            print("Parsing args from dict:", args_list)
            self.args = self.parser.parse_args(args_list)
        else:
            raise ValueError("args must be None, list, or dict")

        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

        with open(f'{self.args.save_path}/opt.txt', '+w') as f:
            for k, v in sorted(vars(self.args).items()):
                f.write('%s: %s\n' % (str(k), str(v)))

        return self.args

    def get_num_parts(self, num_faces):
        lookup_num_parts = [1, 2, 4, 8]
        num_parts = lookup_num_parts[np.digitize(num_faces, self.args.faces_to_part, right=True)]
        return num_parts

    def dtype(self):
        return torch.float32

    def get_num_samples(self, cur_iter):
        slope = (self.args.samples - self.args.begin_samples) / int(0.8 * self.args.upsamp)
        return int(slope * min(cur_iter, 0.8 * self.args.upsamp)) + self.args.begin_samples
