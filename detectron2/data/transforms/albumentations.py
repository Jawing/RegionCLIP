import numpy as np
from detectron2.data.transforms import Augmentation
from fvcore.transforms.transform import Transform, NoOpTransform


class AlbumentationsTransform(Transform):
    def __init__(self, aug, params):
        self.aug = aug
        self.params = params

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_image(self, image):
        return self.aug.apply(image, **self.params)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        try:
            return np.array(self.aug.apply_to_bboxes(box.tolist(), **self.params))
        except AttributeError:
            return box

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        try:
            return self.aug.apply_to_mask(segmentation, **self.params)
        except AttributeError:
            return segmentation


class AlbumentationsWrapper(Augmentation):
    """
    Wrap an augmentor form the albumentations library: https://github.com/albu/albumentations.
    Image, Bounding Box and Segmentation are supported.
    Example:
    .. code-block:: python
        import albumentations as A
        from detectron2.data import transforms as T
        from detectron2.data.transforms.albumentations import AlbumentationsWrapper

        augs = T.AugmentationList([
            AlbumentationsWrapper(A.RandomCrop(width=256, height=256)),
            AlbumentationsWrapper(A.HorizontalFlip(p=1)),
            AlbumentationsWrapper(A.RandomBrightnessContrast(p=1)),
        ])  # type: T.Augmentation

        # Transform XYXY_ABS -> XYXY_REL
        h, w, _ = IMAGE.shape
        bbox = np.array(BBOX_XYXY) / [w, h, w, h]

        # Define the augmentation input ("image" required, others optional):
        input = T.AugInput(IMAGE, boxes=bbox, sem_seg=IMAGE_MASK)

        # Apply the augmentation:
        transform = augs(input)
        image_transformed = input.image  # new image
        sem_seg_transformed = input.sem_seg  # new semantic segmentation
        bbox_transformed = input.boxes   # new bounding boxes

        # Transform XYXY_REL -> XYXY_ABS
        h, w, _ = image_transformed.shape
        bbox_transformed = bbox_transformed * [w, h, w, h]
    """

    def __init__(self, augmentor):
        """
        Args:
            augmentor (albumentations.BasicTransform):
        """
        # super(Albumentations, self).__init__() - using python > 3.7 no need to call rng
        self._aug = augmentor

    def get_transform(self, image):
        do = self._rand_range() < self._aug.p
        if do:
            params = self.prepare_param(image)
            return AlbumentationsTransform(self._aug, params)
        else:
            return NoOpTransform()

    def prepare_param(self, image):
        params = self._aug.get_params()
        if self._aug.targets_as_params:
            targets_as_params = {"image": image}
            params_dependent_on_targets = self._aug.get_params_dependent_on_targets(targets_as_params)
            params.update(params_dependent_on_targets)
        params = self._aug.update_params(params, **{"image": image})
        return params