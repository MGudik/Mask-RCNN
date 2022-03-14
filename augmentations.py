import imgaug.augmenters as iaa


def get_augmentations():

    def sometimes(aug):
        return iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.LinearContrast((0.75, 1.5)),
        iaa.Multiply((0.8, 1.2), per_channel=False),
        iaa.Multiply((0.9, 1.1), per_channel=0.2),
        sometimes(iaa.GaussianBlur(sigma=(0, 10))),
        sometimes(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
        sometimes(iaa.Dropout((0.01, 0.1), per_channel=0.5)),
        sometimes(iaa.Grayscale(alpha=(0.0, 0.5)))
    ], random_order=True)

    return seq


def augment(images):
    seq = get_augmentations()

    augmented = seq(images=images)

    return augmented