import os
import numpy as np
import torch
import cv2

from model import get_instance_segmentation_model


def test():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = get_instance_segmentation_model(2)
    model.load_state_dict(torch.load("checkpoints/model.pt"))
    model.to(device)

    # set to evaluation mode
    model.eval()

    # run inference on the test dataset
    with torch.no_grad():
        # load image 
        image = cv2.imread("dataset/PNGImages/FudanPed00001.png")
        original = image.copy()

        # convert to tensor
        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        image = np.ascontiguousarray(image)

        # uint8 to fp32 
        image = image.astype(np.float32)
        image /= 255.0  # 0 - 255 to 0.0 - 1.0

        # to tensor
        image = torch.from_numpy(image).to(device)

        # add batch dimension if necessary
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        prediction = model(image)[0] # batch with only one element

        masks = prediction["masks"].cpu().detach().numpy()

        cv2.imshow("image", original)

        for mask in masks:
            mask = (255 * mask).astype("uint8")
            mask = mask.squeeze()

            cv2.imshow("mask", mask)
            cv2.waitKey(0)


if __name__ == "__main__":
    test()
