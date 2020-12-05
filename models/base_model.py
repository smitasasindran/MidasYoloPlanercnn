import torch

class BaseModel(torch.nn.Module):
    # ToDo Smita: Move this from here
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))

        print("Load Path is:", path, ", Parameters are:")
        if "optimizer" in parameters:
            parameters = parameters["model"]

        # print(parameters.keys())
        self.load_state_dict(parameters, strict=False)


    # def load_yolo(self):
    #     pass