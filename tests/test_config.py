from transithunter import config


def test_data_dir_is_on_secondary_drive_or_overridden():
    assert config.DATA_DIR.is_dir(), f"create {config.DATA_DIR} or set TRANSITHUNTER_DATA_DIR"
