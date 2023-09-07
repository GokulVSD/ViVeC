from torchvision.transforms import transforms

def partition_to_grid(img, num_rows, num_cols, hor_pixels, ver_pixels):

    # Convert PIL to tensor.
    img_tensor = transforms.Compose([transforms.PILToTensor()])(img)

    grid = []

    for i in range(num_rows):
        grid.append([])
        for j in range(num_cols):
            grid[-1].append(
                img_tensor[:, ver_pixels*i : ver_pixels*(i+1), hor_pixels*j : hor_pixels*(j+1)]
            )

    return grid