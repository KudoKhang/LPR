import hydra


def cfg(config_name="config"):
    with hydra.initialize(version_base=None, config_path="../../conf"):
        cfg = hydra.compose(config_name=config_name)
    return cfg
