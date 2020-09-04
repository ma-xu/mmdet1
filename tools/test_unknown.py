import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from tools.fuse_conv_bn import fuse_module

from mmdet.apis import multi_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.core import encode_mask_results, tensor2imgs
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--openeval', action='store_true')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # print("\n\n")
            # print(f" tuple: {isinstance(result, tuple)}")
            bbox_result, segm_result = result

            new_bbox_result = [[]]*(len(bbox_result)+1)
            new_segm_result = [[]]*(len(bbox_result)+1)
            unknown_bbox = []
            unknown_segm = []
            for i in range(0, len(bbox_result)):
                if len(bbox_result[i])==0:
                    new_bbox_result[i] = bbox_result[i]
                    new_segm_result[i] = segm_result[i]
                else:
                    temp_bbox = []
                    temp_segm = []
                    # print(f"\n{i} bbox_result len: {len(bbox_result[i])}")
                    for j in range(0,len(bbox_result[i])):
                        if bbox_result[i][j][-1] <= 0.1:
                            temp_bbox.append(bbox_result[i][j])
                            temp_segm.append(segm_result[i][j])
                        else:
                            unknown_bbox.append(bbox_result[i][j])
                            unknown_segm.append(segm_result[i][j])
                    new_bbox_result[i] = temp_bbox
                    new_segm_result[i] = temp_segm
                    # print(f"\n{i} new bbox_result len: {len(new_bbox_result[i])}")
                    # print(f"\n{i} new unknown_bbox len: {len(unknown_bbox)}")
            new_bbox_result[-1] = unknown_bbox
            new_segm_result[-1] = unknown_segm
            # print("___________________")
            result = new_bbox_result,new_segm_result

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result, tuple):
            bbox_results, mask_results = result
            encoded_mask_results = encode_mask_results(mask_results)
            result = bbox_results, encoded_mask_results
        results.append(result)

        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    # Added by Nokia Intern Xu Ma
    # Load initial centroids
    if cfg.centroids_from is not None:
        centroids = mmcv.load(cfg.centroids_from)
        if not isinstance(centroids, torch.Tensor):
            # centroids would be dict from openmax
            centroids = torch.tensor([centroids[i] for i in range(1, len(centroids) + 1)]).float()
        # Consider the background class, padding 0
        centroids = torch.cat([centroids, torch.zeros(1, centroids.shape[1]).to(centroids.device)], dim=0)
        centroids = torch.cat([centroids, torch.zeros(centroids.shape[0], 1).to(centroids.device)], dim=1)
        model.roi_head.bbox_head.centroids = centroids
        print(f'Initialized centroids from {cfg.centroids_from}')



    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    # if 'CLASSES' in checkpoint['meta']:
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    model.CLASSES = dataset.CLASSES
    print(model.CLASSES)


    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:

            if args.openeval:
                dataset.openevaluate(outputs, args.eval, **kwargs)
            else:
                dataset.evaluate(outputs, args.eval, **kwargs)


if __name__ == '__main__':
    main()
