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
    def parse_yaml_file(self, yaml_file: str ) -> Tuple[DataClass, ...]:
        
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
        
        data = yaml.load(Path(yaml_file).read_text(), Loader=loader)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            arg_name = dtype.__mro__[-2].__name__
            inputs = {k: v for k, v in data[arg_name].items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        
        return (*outputs,)