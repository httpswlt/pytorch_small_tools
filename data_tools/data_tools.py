import torch


class DataPrefetch(object):
    def __init__(self, loader, gpu=None):
        self.loader = iter(loader)
        self.next_input = None
        self.next_target = None
        self.gpu = gpu
        if self.gpu is not None:
            assert isinstance(self.gpu, int), "GPU should is a int type."
        self.stream = torch.cuda.Stream(self.gpu)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(self.gpu, non_blocking=True).float()
            self.next_target = self.next_target.long().cuda(self.gpu, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        target = self.next_target
        self.preload()
        return inputs, target