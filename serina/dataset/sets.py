from serina.dataset.hir import EasySet


class TrainSet(EasySet):
    def __init__(self):
        super().__init__(0, 0.6)


class ValidationSet(EasySet):
    def __init__(self):
        super().__init__(0.6, 0.8)


class TestSet(EasySet):
    def __init__(self):
        super().__init__(0.8, 1)
