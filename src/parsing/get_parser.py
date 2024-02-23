from parsing.iemocap import ParserForIEMOCAP


def get_parser_for_dataset(dataset_name):
    if dataset_name == "iemocap":
        return ParserForIEMOCAP
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
