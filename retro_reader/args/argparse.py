import dataclasses
import re
import copy
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Iterable, List, NewType, Optional, Tuple, Union, Dict

from transformers.hf_argparser import DataClass, HfArgumentParser as OriginalHfArgumentParser

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)

def lambda_field(default, **kwargs):
    return field(default_factory=lambda: copy.copy(default))

class HfArgumentParser(OriginalHfArgumentParser):
    def parse_yaml_file(self, yaml_file: str) -> Tuple[DataClass, ...]:
        """
        Parse a YAML file and return a tuple of dataclass instances.

        Args:
            yaml_file (str): Path to the YAML file.

        Returns:
            Tuple[DataClass, ...]: A tuple of dataclass instances.
        """
        # Create a custom YAML loader that allows parsing of floats with exponents
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.')
        )

        # Load the YAML data from the file
        data = yaml.load(Path(yaml_file).read_text(), Loader=loader)

        # Create a list to store the dataclass instances
        outputs = []

        # Iterate over each dataclass type
        for dtype in self.dataclass_types:
            # Get the names of the fields that are initialized
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            # Get the name of the argument from the dataclass's mro
            arg_name = dtype.__mro__[-2].__name__
            # Create a dictionary of the inputs for the dataclass
            inputs = {k: v for k, v in data[arg_name].items() if k in keys}
            # Create an instance of the dataclass with the inputs
            obj = dtype(**inputs)
            # Add the instance to the list
            outputs.append(obj)

        # Return the list of dataclass instances as a tuple
        return tuple(outputs)
