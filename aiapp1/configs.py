from pydantic_config import SettingsConfig, SettingsModel
from pydantic import BaseModel


class AppConfig(SettingsModel):
    infer_model_name: str = 'mobilenetv4_conv_small.e2400_r224_in1k'

    model_config = SettingsConfig(
        env_file='.env',
        case_sensitive=False,
    )


cfg = AppConfig()
