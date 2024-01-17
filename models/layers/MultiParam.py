from torch.nn import Module

class MultiParam(Module):
    def __init__(self, model, args_to_take, return_original_x = False):
        super(MultiParam, self).__init__()
        self.model = model
        self.args_to_take = args_to_take
        self.return_original_x = return_original_x
        
        
    def forward(self, args):
        original_x = args[0]
        fn_args = [args[k] for k in self.args_to_take]
        
        res = self.model(*fn_args)
        args[0] = res
        if self.return_original_x:
            return [*args, original_x]
        return args