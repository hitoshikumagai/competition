from pathlib import Path
from typing import Callable, List

import cv2
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import os,sys
from .transforms import tensor_transform
from .utils import ON_KAGGLE
'''
参考:https://qiita.com/takurooo/items/e4c91c5d78059f92e76d
Datasetを実装するのに必要な要件
オリジナルDatasetを実装するときに守る必要がある要件は以下３つ。
torch.utils.data.Datasetを継承する。
__len__を実装する。
__getitem__を実装する。
__len__は、len(obj)で実行されたときにコールされる関数。
__getitem__は、obj[i]のようにインデックスで指定されたときにコールされる関数。

'''

N_CLASSES = 1103
DATA_ROOT = Path('../data/input/imet-2019-fgvc6')

class TrainDataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame,
        #image_transform:前処理、後で関数を呼ぶ。
                 image_transform: Callable, debug: bool = True):
        super().__init__()

        self._root = root
        self._df = df
        self._image_transform = image_transform
        self._debug = debug

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        #とってくる写真のidを選択して取得。
        item = self._df.iloc[idx]
        #別ファイルで定義されている関数
        image = load_transform_image(
            item, self._root, self._image_transform, debug=self._debug)
        #とりあえずクラス=0(あてはまらないとする)
        target = torch.zeros(N_CLASSES)
        #train.csvのattribute_ids列から1をつける。
        for cls in item.attribute_ids.split():
            target[int(cls)] = 1
        return image, target


class TTADataset:
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable, tta: int):
        self._root = root
        self._df = df
        self._image_transform = image_transform
        self._tta = tta

    def __len__(self):
        return len(self._df) * self._tta

    def __getitem__(self, idx):
        item = self._df.iloc[idx % len(self._df)]
        image = load_transform_image(item, self._root, self._image_transform)
        return image, item.id


def load_transform_image(
        item, root: Path, image_transform: Callable, debug: bool = False):
    image = load_image(item, root)
    image = image_transform(image)
    if debug:
        image.save('_debug.png')
    return tensor_transform(image)


def load_image(item, root: Path) -> Image.Image:
    image = cv2.imread(str(root / f'{item.id}.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def get_ids(root: Path) -> List[str]:
    #_で文字列を分けた後に、その先頭を取得し、ソートする。(id取得)
    return sorted({p.name.split('_')[0] for p in root.glob('*.png')})
