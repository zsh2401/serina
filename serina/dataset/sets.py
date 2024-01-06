from serina.dataset.hir import TransformationCachedWaveSet


class TrainSet(TransformationCachedWaveSet):
    def __init__(self):
        super().__init__(0, 0.6)


class ValidationSet(TransformationCachedWaveSet):
    def __init__(self):
        super().__init__(0.6, 0.8)


class TestSet(TransformationCachedWaveSet):
    def __init__(self):
        super().__init__(0.8, 1)
