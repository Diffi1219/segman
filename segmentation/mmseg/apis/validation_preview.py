from pathlib import Path

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import TensorboardLoggerHook
from PIL import Image, ImageDraw
from torch.utils.data import Subset

from mmseg.datasets import build_dataloader

from .test import single_gpu_test


class ValidationPreview:
    """Render a fixed, ordered set of binary segmentation validation images."""

    def __init__(self, dataset, names):
        self.dataset = dataset
        self.names = names
        self.indices = []
        stems = [Path(info['filename']).stem for info in dataset.img_infos]
        for name in names:
            matches = [index for index, stem in enumerate(stems) if stem == name]
            if len(matches) != 1:
                raise ValueError(f'Expected one validation image named {name!r}, '
                                 f'found {len(matches)}')
            self.indices.append(matches[0])
        self.loader = build_dataloader(
            Subset(dataset, self.indices),
            samples_per_gpu=1,
            workers_per_gpu=0,
            dist=False,
            shuffle=False,
            persistent_workers=False)

    def __call__(self, runner, by_epoch):
        if by_epoch:
            step = runner.epoch + 1
            folder = f'epoch_{step:04d}'
        else:
            step = runner.iter + 1
            folder = f'iter_{step:06d}'
        out_dir = Path(runner.work_dir) / 'val_vis' / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        model = runner.model
        if isinstance(model, MMDistributedDataParallel):
            # Only rank zero renders previews; bypass distributed collectives.
            model = MMDataParallel(model.module,
                                   device_ids=[torch.cuda.current_device()])
        predictions = single_gpu_test(model, self.loader)
        writers = [
            hook.writer for hook in runner.hooks
            if isinstance(hook, TensorboardLoggerHook)
        ]
        palette = np.asarray(self.dataset.PALETTE, dtype=np.uint8)
        for name, index, prediction in zip(self.names, self.indices, predictions):
            filename = self.dataset.img_infos[index]['filename']
            with Image.open(Path(self.dataset.img_dir) / filename) as image:
                original = image.convert('RGB')
            gt = self.dataset.get_gt_seg_map_by_idx(index)
            panel = _make_panel(original, gt, prediction, palette)
            panel.save(out_dir / f'{name}.png')
            for writer in writers:
                writer.add_image(f'val/preview/{name}', np.asarray(panel), step,
                                 dataformats='HWC')
        for writer in writers:
            writer.flush()
        runner.logger.info('Saved validation previews to %s', out_dir)


def _make_panel(original, gt, prediction, palette):
    gt_color = np.full((*gt.shape, 3), 128, dtype=np.uint8)
    valid = (gt >= 0) & (gt < len(palette))
    gt_color[valid] = palette[gt[valid]]
    gt_image = Image.fromarray(gt_color)
    pred_image = Image.fromarray(palette[prediction])
    overlay = Image.composite(
        Image.blend(original, pred_image, 0.5), original,
        Image.fromarray(prediction != 0))

    width, height = original.size
    header = 24
    panel = Image.new('RGB', (4 * width, height + header), (24, 24, 24))
    draw = ImageDraw.Draw(panel)
    titles = ('Image', 'Ground truth', 'Prediction', 'Prediction overlay')
    for i, (title, image) in enumerate(
            zip(titles, (original, gt_image, pred_image, overlay))):
        panel.paste(image, (i * width, header))
        draw.text((i * width + 8, 6), title, fill='white')
    return panel
