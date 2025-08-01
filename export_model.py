import torch
from torchvision.models import resnet18

# An instance of your model.
model = resnet18(weights='DEFAULT')
model.fc = torch.nn.Identity()
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
# traced_script_module = torch.jit.trace(model, example)
#
# # Save the TorchScript model
# traced_script_module.save("resnet18_identity.pt")

# onnx_program = torch.onnx.export(model, (example,), dynamo=True)
# onnx_program.save("resnet18_identity.onnx")

# Export the model
torch.onnx.export(model,  # model being run
                  (example,),  # model input (or a tuple for multiple inputs)
                  "resnet18_identity.onnx",  # where to save the model
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['modelInput'],  # the model's input names
                  output_names=['modelOutput'],  # the model's output names
                  dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                'modelOutput': {0: 'batch_size'}})
print(" ")
print('Model has been converted to ONNX')