from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
try:
  from ..generic_dataset import GenericDataset
except:
  import sys
  sys.path.append('/u/jozhang/code/motion3d/external/CenterTrack/src/lib/dataset')
  from generic_dataset import GenericDataset

class TAODataset(GenericDataset):
  num_categories = 833
  default_resolution = [256, 256]
  class_name = ['']
  max_objs = 128
  cat_ids = {1: 1}
  def __init__(self, opt, split):
    img_dir = opt.data_dir / 'keyframes'
    ann_path = opt.data_dir / 'annotations' / f'{split}.json'
    self.class_name = ['' for _ in range(self.num_categories)]
    # self.default_resolution = [opt.input_h, opt.input_w]
    self.cat_ids = {i: i for i in range(1, self.num_categories + 1)}

    self.images = None
    # load image list and coco
    super().__init__(opt, split, ann_path, img_dir)

    self.num_samples = len(self.images)
    print('Loaded Custom dataset {} samples'.format(self.num_samples))
  
  def __len__(self):
    return self.num_samples

  def run_eval(self, results, save_dir):
    pass

if __name__ == '__main__':
  sys.path.append('/u/jozhang/code/motion3d/external/CenterTrack/src/lib')
  from opts import opts
  opt = opts().parse()
  opt.data_dir = Path('/scratch/cluster/jozhang/datasets/TAO')
  opt.tracking = True
  opt.not_max_crop = True
  opt.output_w = 256
  opt.output_h = 256
  opt.num_classes = 833
  dataset = TAODataset(opt, 'train')
  opt = opts().update_dataset_info_and_set_heads(opt, TAODataset)
  sample = dataset[6]
  print(sample)
