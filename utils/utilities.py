from typing import Callable, Union, Any
import inspect
import torch


def shape_debuger(func_name: Callable, *args: Union[torch.Tensor, Any]) -> torch.Tensor:
    '''
    Debug the shape of the input tensor
    '''
    stack = inspect.stack()

    for frame in stack:
        if 'self' in frame.frame.f_locals:
            print(
                f"🐞 {frame.frame.f_locals['self'].__class__.__name__} {func_name.__class__.__name__}: ", end="")
            str_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    str_args.append(str(tuple(arg.shape)))
                elif isinstance(arg, tuple):
                    for a in arg:
                        if isinstance(a, torch.Tensor):
                            str_args.append(str(tuple(a.shape)))
                        else:
                            str_args.append(str(a))
                else:
                    str_args.append(str(arg))
            print(", ".join(str_args), end=" -> ")
            out = func_name(*args)
            print(f"{tuple(out.shape)}")
            return out
