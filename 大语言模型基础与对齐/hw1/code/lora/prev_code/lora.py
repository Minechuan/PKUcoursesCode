import torch
import transformers


from utils import recursive_getattr, recursive_setattr


class LoRALinear(torch.nn.Module):
    def __init__(self, weight, bias, lora_dim, lora_scaling):
        super(LoRALinear, self).__init__()
        # Save original weight and bias
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias) if bias is not None else None
        # TODO: Implement lora left and right weights
        in_features, out_features = weight.shape

        self.lora_left_weight = torch.nn.Parameter(torch.zeros(in_features, lora_dim))
        self.lora_right_weight = torch.nn.Parameter(torch.zeros(lora_dim, out_features))
        #############################################
        self.lora_scaling = lora_scaling / lora_dim
        self.init_parameters()
        # TODO: Freeze original weight and bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        #######################################

    def init_parameters(self):
        # TODO: Initialize LoRA parameters
        torch.nn.init.kaiming_uniform_(self.lora_left_weight, a=torch.sqrt(5))
        torch.nn.init.zeros_(self.lora_right_weight)
        ##################################

    def forward(self, input):
        # TODO: Implement the forward function
        original_output = torch.nn.functional.linear(input, self.weight, self.bias)

        lora_output = (input @ self.lora_left_weight) @ self.lora_right_weight
        lora_output *= self.lora_scaling

        return original_output + lora_output
        ######################################


def convert_linear_layer_to_lora(model, part_module_name, lora_dim=0, lora_scaling=1):
    replace_name = []
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear) or isinstance(module, transformers.pytorch_utils.Conv1D)) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        if isinstance(module, torch.nn.Linear):
            tmp = LoRALinear(module.weight, module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            tmp = LoRALinear(module.weight.t().detach(), module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        else:
            raise ValueError("Unsupported module type")
        recursive_setattr(model, name, tmp)
    return model


def only_optimize_lora_parameters(model):
    # TODO: Turn off the gradient of all the parameters except the LoRA parameters
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_left_weight.requires_grad = True
            module.lora_right_weight.requires_grad = True
    ##############################################################################

def get_lora_state_dict(model):
    # TODO: return lora left and right weights as state dict
    # The saved state dict will be used later for loading
    state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = f"{name}." if name else ""
            state_dict[f"{prefix}lora_left_weight"] = module.lora_left_weight
            state_dict[f"{prefix}lora_right_weight"] = module.lora_right_weight
    return state_dict

    ########################################################