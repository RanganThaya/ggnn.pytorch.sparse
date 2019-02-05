import torch
import numpy as np
from torch.utils.data import DataLoader


class bAbIDataloader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(bAbIDataloader, self).__init__(
            *args, **kwargs, collate_fn=bAbIDataloader.collate_fn)

    @staticmethod
    def collate_fn(data):
        offset = 0
        b_am = None
        b_annotation = None
        b_target = []
        for i, d in enumerate(data):
            am, annotation, target = d
            if b_am is None:
                b_am = [[] for _ in range(len(am))]

            if b_annotation is None:
                b_annotation = np.zeros(annotation[0].shape)

            b_target.append(target)

            b_annotation = np.vstack(
                (b_annotation, annotation))

            for i, edge_list in enumerate(am):
                for src, dest in edge_list:
                    b_am[i].append([src+offset, dest+offset])

            offset += len(annotation)

        max_length = 0
        for edge_list in b_am:
            max_length = max(max_length, len(edge_list))

        for edge_list in b_am:
            for i in range(0, max_length - len(edge_list)):
                edge_list.append([0, 0])
        b_am = torch.Tensor(b_am).long()
        b_annotation = torch.from_numpy(b_annotation).float()
        b_target = torch.Tensor(b_target).long()
        return b_am, b_annotation, b_target
