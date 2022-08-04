# FROM https://stackoverflow.com/a/57364423/10917436
# istarmap.py for Python 3.8+
# import torch.multiprocessing.pool as mpp
# def istarmap(self, func, iterable, chunksize=1):
#     """starmap-version of imap
#     """
#     self._check_running()
#     if chunksize < 1:
#         raise ValueError(
#             "Chunksize must be 1+, not {0:n}".format(
#                 chunksize))

#     task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
#     result = mpp.IMapIterator(self)
#     self._taskqueue.put(
#         (
#             self._guarded_task_generation(result._job,
#                                           mpp.starmapstar,
#                                           task_batches),
#             result._set_length
#         ))
#     return (item for chunk in result for item in chunk)


# mpp.Pool.istarmap = istarmap
# FROM https://stackoverflow.com/a/57364423/10917436

from itertools import cycle, islice
from torch.multiprocessing import current_process
import torch.multiprocessing as mp
import torch
from tqdm import tqdm
from pathlib import Path
import glob
import dnnlib
import click
from collections import namedtuple
from lpips.lpips import LPIPS, spatial_average, upsample
import lpips
from pprint import pprint
# import image_similarity_measures
# needed for pickling
VGG_outputs = namedtuple(
    "VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])


def repeatlist(it, count):
    return islice(cycle(it), count)


class HackedLPIPs(LPIPS):
    def forward(self, in0, in1, retPerLayer=False, normalize=False, precomputed=False):

        if not precomputed:
            # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            if normalize:
                in0 = 2 * in0 - 1
                in1 = 2 * in1 - 1

            # v0.0 - original release had a bug, where input was not scaled
            in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(
                in1)) if self.version == '0.1' else (in0, in1)
            outs0, outs1 = self.net.forward(
                in0_input), self.net.forward(in1_input)
        else:
            outs0, outs1 = in0, in1
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = lpips.normalize_tensor(
                outs0[kk]), lpips.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk].model(diffs[kk]),
                                out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk].model(
                    diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1, keepdim=True),
                                out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(
                    dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1, self.L):
            val += res[l]

        if(retPerLayer):
            return (val, res)
        else:
            return val

    def precompute(self, in0, normalize=False):
        # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
        if normalize:
            in0 = 2 * in0 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input = self.scaling_layer(in0) if self.version == '0.1' else in0
        outs0 = self.net.forward(in0_input)
        return outs0


def hacked_vgg16_forward(self, X):
    h = self.slice1(X)
    h_relu1_2 = h
    h = self.slice2(h)
    h_relu2_2 = h
    h = self.slice3(h)
    h_relu3_3 = h
    h = self.slice4(h)
    h_relu4_3 = h
    h = self.slice5(h)
    h_relu5_3 = h
    # vgg_outputs = namedtuple(
    #     "VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
    out = (h_relu1_2, h_relu2_2,
           h_relu3_3, h_relu4_3, h_relu5_3)

    return out


def chunker_list(seq, size):
    return list((seq[i::size] for i in range(size)))


def split_dict_equally(input_dict, chunks=2):
    "Splits dict by keys. Returns a list of dictionaries."
    # prep with empty dicts
    return_list = [dict() for idx in range(chunks)]
    idx = 0
    for k, v in input_dict.iteritems():
        return_list[idx][k] = v
        if idx < chunks-1:  # indexes start at 0
            idx += 1
        else:
            idx = 0
    return return_list


def l1_for_image(data_image_paths, image_dict, device):
    current_proc = current_process()._identity[0]-1

    result_dict = dict()
    for image_path in image_dict.keys():
        # image_dict[image_path].tensor =  image_dict[image_path].tensor
        result_dict[image_path] = dnnlib.EasyDict(
            lowest_distance=10000000, lowest_path="mock")
    for data_image_path in tqdm(data_image_paths, position=current_proc):
        neighbor_tensor = lpips.im2tensor(
            lpips.load_image(data_image_path), cent=0., factor=255.).to(device)
        for image_path in image_dict.keys():
            # print(neighbor_tensor.shape, image_dict[image_path].tensor.shape)
            with torch.no_grad():
                distance = torch.norm(
                    image_dict[image_path].tensor - neighbor_tensor, p=1)
                # distance = ssim(
                #     image_dict[image_path].tensor, neighbor_tensor, data_range=1,  size_average=True)
                if distance < result_dict[image_path].lowest_distance:
                    result_dict[image_path].lowest_distance = distance.detach()
                    result_dict[image_path].lowest_path = data_image_path
    return result_dict

def distance_for_image(data_image_paths, image_dict, distance_metric: HackedLPIPs, device):

    current_proc = current_process()._identity[0]-1

    result_dict = dict()
    for image_path in image_dict.keys():
        # image_dict[image_path].tensor =  image_dict[image_path].tensor
        result_dict[image_path] = dnnlib.EasyDict(
            lowest_distance=10000000, lowest_path="mock")

    for data_image_path in tqdm(data_image_paths, position=current_proc):
        neighbor_tensor = distance_metric.precompute(
            lpips.im2tensor(lpips.load_image(data_image_path)).to(device))
        for image_path in image_dict.keys():
            # print(neighbor_tensor.shape, image_dict[image_path].tensor.shape)
            with torch.no_grad():
                distance = distance_metric.forward(
                    image_dict[image_path].tensor, neighbor_tensor, precomputed=True)
                if distance < result_dict[image_path].lowest_distance:
                    result_dict[image_path].lowest_distance = distance.detach()
                    result_dict[image_path].lowest_path = data_image_path

    return result_dict


def distance_for_image_precomputed(data_image_dict, image_dict, distance_metric: HackedLPIPs, device):

    current_proc = current_process()._identity[0]-1

    result_dict = dict()
    for image_path in image_dict.keys():
        # image_dict[image_path].tensor =  image_dict[image_path].tensor
        result_dict[image_path] = dnnlib.EasyDict(
            lowest_distance=10000000, lowest_path="mock")

    for data_image_path, data_image_tensor in tqdm(data_image_dict.items(), position=current_proc):
        neighbor_tensor = data_image_tensor.to(device)
        for image_path in image_dict.keys():
            # print(neighbor_tensor.shape, image_dict[image_path].tensor.shape)
            with torch.no_grad():
                distance = distance_metric.forward(
                    image_dict[image_path].tensor, neighbor_tensor, precomputed=True)
                if distance < result_dict[image_path].lowest_distance:
                    result_dict[image_path].lowest_distance = distance.detach()
                    result_dict[image_path].lowest_path = data_image_path

    return result_dict


def precompute_dataset_ims(data_image_paths, distance_metric: HackedLPIPs, device):

    current_proc = current_process()._identity[0]-1

    result_dict = dict()

    for data_image_path in tqdm(data_image_paths, position=current_proc):
        data_tensor = distance_metric.precompute(
            lpips.im2tensor(lpips.load_image(data_image_path)).to(device))
        data_tensor = [tens.to("cpu") for tens in data_tensor]
        result_dict[data_image_path] = data_tensor

    return result_dict


@click.command()
@click.pass_context
@click.option('--dataset', help='Where to look for nearest neighbors',  metavar='DIR')
@click.option('-i', '--images', help='Which images do we want the nearest neighbors for?', multiple=True)
@click.option('--gpu', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('-t', '--threads', help='Number of threads to use [default: 1]', type=int, metavar='INT')
@click.option('--pickle-out', help='precompute featires for dataset', metavar='FILE')
@click.option('--pickle-in', help='Load precomputed pickle for dataset', metavar='FILE')
@click.option('--l1', is_flag=True)
def main(ctx, dataset, images, gpu, threads, pickle_out, pickle_in, l1):
    mp.set_start_method("spawn")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    devices = ["cpu"] if gpu == 0 else ["cuda:" + str(i) for i in range(gpu)]

    print(devices)
    lpips.pn.vgg16.forward = hacked_vgg16_forward

    image_dicts_per_device = []
    distance_metrics_per_device = []
    for device in devices:
        distance_metric = HackedLPIPs(net='vgg', verbose=False).to(device)
        image_dict = dict()
        for image_path in images:
            if not l1:
                image_dict[image_path] = dnnlib.EasyDict(tensor=distance_metric.precompute(
                lpips.im2tensor(lpips.load_image(image_path)).to(device)))
            else:
                image_dict[image_path] = dnnlib.EasyDict(
                    tensor=lpips.im2tensor(lpips.load_image(image_path), cent=0., factor=255.).to(device))

        image_dicts_per_device.append(image_dict)
        distance_metrics_per_device.append(distance_metric)
    if not pickle_in:
        print(dataset + "/**/*.png")
        dataset_paths = glob.glob(dataset + "/**/*.png", recursive=True)
        dataset_paths = chunker_list(dataset_paths, threads)
        d_len = len(dataset_paths)

    if pickle_out:
        with mp.Pool(threads) as pool:
            result_chunks = []
            result_chunks = pool.starmap(precompute_dataset_ims, zip(dataset_paths, repeatlist(
                distance_metrics_per_device, d_len), repeatlist(devices, d_len)))
            result_dict = {}
            for result in result_chunks:
                result_dict = {**result_dict, **result}
            print("Saving to ", pickle_out)
            torch.save(result_dict, pickle_out)
            return
    if pickle_in:
        precomputed_dataset = torch.load(pickle_in, map_location="cpu")
        precomputed_dataset = split_dict_equally(
            precomputed_dataset, chunks=threads)
        d_len = len(precomputed_dataset)
        with mp.Pool(threads) as pool:
            result_chunks = pool.starmap(distance_for_image_precomputed, zip(precomputed_dataset, repeatlist(
                image_dicts_per_device, d_len), repeatlist(distance_metrics_per_device, d_len), repeatlist(devices, d_len)))
    else:
    # print(image_dicts_per_device, distance_metrics_per_device)
    # multiprocessing.set_start_method("fork", force=True)
        with mp.Pool(threads) as pool:
            if not l1:
                result_chunks = pool.starmap(distance_for_image, zip(dataset_paths, repeatlist(
                    image_dicts_per_device, d_len), repeatlist(distance_metrics_per_device, d_len), repeatlist(devices, d_len)))
            else:
                result_chunks = pool.starmap(l1_for_image, zip(dataset_paths, repeatlist(
                    image_dicts_per_device, d_len),  repeatlist(devices, d_len)))

    # print(result_chunks)

    nearest_neighbors = result_chunks[0]
    for result in result_chunks:
        for image_path in result.keys():
            if result[image_path].lowest_distance.to("cpu") < nearest_neighbors[image_path].lowest_distance.to("cpu"):
                nearest_neighbors[image_path] = result[image_path]
    print("Nearest neighbors for images:", nearest_neighbors)
    pprint(nearest_neighbors)
    # concurrent.process_map(find_nearest_neighbor)
    # find_nearest_neighbor()
    # print(image_dict)


if __name__ == "__main__":
    main()
