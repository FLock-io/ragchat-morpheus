import os
import json
import yaml
from utils.log import Logger

logger = Logger().getlog()


def from_config(config_path):
    if config_path:
        file_extension = os.path.splitext(config_path)[1]
        with open(config_path, "r", encoding="UTF-8") as file:
            if file_extension in [".yaml", ".yml"]:
                config_data = yaml.safe_load(file)
            elif file_extension == ".json":
                config_data = json.load(file)
            else:
                raise ValueError("config_path must be a path to a YAML or JSON file.")
    else:
        logger.error(
            "Please provide either a config file path (YAML or JSON) or a config dictionary. Falling back to defaults because no config is provided.",
            # noqa: E501
        )
        config_data = {}

    embedding_model_config_data = config_data.get("embedder", {})
    llm_config_data = config_data.get("llm", {})
    rerank_model_config_data = config_data.get("rerank", {})
    vectordb_config_data = config_data.get("vectordb", {})

    return {
            "embedding": embedding_model_config_data,
            "llm": llm_config_data,
            "rerank": rerank_model_config_data,
            "vectordb": vectordb_config_data
            }


