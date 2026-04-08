from mmengine.registry import Registry

DATASETS = Registry(
    'dataset',
    locations=['mova.datasets'],  # triggered build_from_cfg() => Registry.get() => Registry.import_from_location()
)

TRANSFORMS = Registry(
    'transform',
    locations=['mova.datasets.transforms'],
)

DIFFUSION_PIPELINES = Registry(
    'diffusion_pipelines',
    locations=['mova.diffusion.pipelines'],
)

MODELS = Registry(
    'model',
    locations=['mova.diffusion.models'],
)

DIFFUSION_SCHEDULERS = Registry(
    'diffusion_scheduler',
    locations=['mova.diffusion.schedulers'],
)

OPTIMIZERS = Registry(
    'optimizer',
    locations=['mova.engine.optimizers'],
)
