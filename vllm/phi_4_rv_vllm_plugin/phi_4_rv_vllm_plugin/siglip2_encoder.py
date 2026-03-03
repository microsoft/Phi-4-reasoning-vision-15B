from transformers.processing_utils import ImagesKwargs
import transformers.models.siglip2.image_processing_siglip2 as ips

class Siglip2ImageProcessorKwargsNoUpscale(ImagesKwargs, total=False):
    patch_size: int
    max_num_patches: int
    min_num_patches: int

class Siglip2ImageProcessorNoUpscale(ips.Siglip2ImageProcessor):
    model_input_names = ["pixel_values", "pixel_attention_mask", "spatial_shapes"]
    valid_kwargs = Siglip2ImageProcessorKwargsNoUpscale

    def __init__(
        self,
        do_resize: bool = True,
        resample: "PILImageResampling" = ips.PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: ips.Optional[ips.Union[float, list[float]]] = None,
        image_std: ips.Optional[ips.Union[float, list[float]]] = None,
        do_convert_rgb: ips.Optional[bool] = None,
        patch_size: int = 16,
        max_num_patches: int = 256,
        min_num_patches: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]

        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.min_num_patches = min_num_patches

    @ips.filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ips.ImageInput,
        resample: ips.Optional["PILImageResampling"] = None,
        do_rescale: ips.Optional[bool] = None,
        rescale_factor: ips.Optional[float] = None,
        do_normalize: ips.Optional[bool] = None,
        image_mean: ips.Optional[ips.Union[float, list[float]]] = None,
        image_std: ips.Optional[ips.Union[float, list[float]]] = None,
        return_tensors: ips.Optional[ips.Union[str, ips.TensorType]] = None,
        input_data_format: ips.Optional[ips.Union[str, ips.ChannelDimension]] = None,
        do_convert_rgb: ips.Optional[bool] = None,
        patch_size: ips.Optional[int] = None,
        max_num_patches: ips.Optional[int] = None,
        min_num_patches: ips.Optional[int] = None,
    ) -> "Image.Image":
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        patch_size = patch_size if patch_size is not None else self.patch_size
        max_num_patches = max_num_patches if max_num_patches is not None else self.max_num_patches
        min_num_patches = min_num_patches if min_num_patches is not None else self.min_num_patches

        # Explicitly specify data format to be channels last for image preprocessing.
        # Image processor does not support different output formats, because it returns patches.
        data_format = ips.ChannelDimension.LAST

        try:
            images = self.fetch_images(images)
        except TypeError:
            pass
        images = ips.make_flat_list_of_images(images)

        if not ips.valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor")
        ips.validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )
        if do_convert_rgb:
            images = [ips.convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [ips.to_numpy_array(image) for image in images]

        if do_rescale and ips.is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = ips.infer_channel_dimension_format(images[0])

        pixel_masks = []
        pixel_values = []
        spatial_shapes = []

        for image in images:
            image = ips.to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

            num_patches = max((image.shape[1] // patch_size) * (image.shape[0] // patch_size), 1)
            # Resize only if image is too large to fit in max_num_patches, unless it's really small.
            if num_patches < min_num_patches:
                height, width = ips.get_image_size_for_max_num_patches(
                    image_height=image.shape[0],
                    image_width=image.shape[1],
                    patch_size=patch_size,
                    max_num_patches=min_num_patches,
                )
            elif num_patches > max_num_patches:
                height, width = ips.get_image_size_for_max_num_patches(
                    image_height=image.shape[0],
                    image_width=image.shape[1],
                    patch_size=patch_size,
                    max_num_patches=max_num_patches,
                )
            # Else resize only s.t. each side is divisible by patch size.
            else:
                height, width = ips.get_image_size_for_max_num_patches(
                    image_height=image.shape[0],
                    image_width=image.shape[1],
                    patch_size=patch_size,
                    max_num_patches=num_patches,
                )
            image = ips.resize(image=image, size=(height, width), resample=resample, input_data_format=data_format)

            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=data_format)

            if do_normalize:
                image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=data_format)

            patches = ips.convert_image_to_patches(image, patch_size)
            patches, mask = ips.pad_along_first_dim(patches, max_num_patches)
            num_patches_height = image.shape[0] // patch_size
            num_patches_width = image.shape[1] // patch_size

            spatial_shapes.append((num_patches_height, num_patches_width))
            pixel_values.append(patches)
            pixel_masks.append(mask)

        batch_feature = ips.BatchFeature(
            data={
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_masks,
                "spatial_shapes": spatial_shapes,
            },
            tensor_type=return_tensors,
        )

        return batch_feature
