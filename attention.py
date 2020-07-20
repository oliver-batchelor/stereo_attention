import torch
import torch.nn.functional as F

from structs.torch import shape

def correlation_volume(left, right, max_disparity):
    width = left.size(3)

    def disparity_slice(i):
        slice_l = F.pad(left[..., i:width], (i, 0, 0, 0))
        slice_r = F.pad(right[..., :width-i], (i, 0, 0, 0))

        print(i,    slice_l)

        return (slice_l * slice_r).sum(dim=1) 

    slices = [disparity_slice(max_disparity - i - 1) for i in range(max_disparity)]
    return torch.stack(slices, dim=1)


def expanded_volume(image, max_disparity):
    b, c, h, w = image.shape
    return image.unsqueeze(3).expand([b, c, h, max_disparity, w])


def correlation_volume2(left, right, max_disparity):
    b, c, h, w = right.shape
    padded = F.pad(right, (max_disparity - 1, 0, 0, 0))
    
    strides = padded.stride()
    right_vol = padded.as_strided((b, c, h, max_disparity, w), [*strides[:-1], 1, 1], storage_offset=0)
    left_vol = expanded_volume(left, max_disparity)
 
    volume = (left_vol * right_vol).sum(1)
    return volume.permute(0, 2, 1, 3)



def correlation_volume3(left, right, max_disparity):
    b, c, h, w = right.shape
    padded = F.pad(right.permute(0, 2, 3, 1), (0, 0, max_disparity - 1, 0))
    
    _, _, sh, sw = padded.stride()
    right_vol = padded.as_strided((b * h * w, c, max_disparity), [sh, sw, c]).contiguous()

    left_vol = left.permute(0, 2, 3, 1).view(b * h * w, c, 1).expand([b * h * w, c, max_disparity])
 
    volume = (left_vol * right_vol).sum(1).view(b, h, w, max_disparity).permute(0, 3, 1, 2)
    print(volume.shape)

    # left_vol = left.permute(0, 2, 3, 1).reshape(-1, 1, c).contiguous()

    print(right_vol.shape)
    print(left_vol.shape)


    # volume = torch.bmm(left_vol, right_vol).view(b, h, w, max_disparity).permute(0, 3, 1, 2)
    return volume



if __name__=='__main__':

    batch = 1
    channels = 1
    
    width = 3
    height = 2

    # left = torch.arange(6).view(batch, channels, height, width)
    # right = torch.arange(6).view(batch, channels, height, width) + 1


    left = torch.randn(batch, channels, height, width)
    right = torch.randn(batch, channels, height, width)


    v1 = correlation_volume(left, right, max_disparity=4)
    v2 = correlation_volume2(left, right, max_disparity=4)
    v3 = correlation_volume3(left, right, max_disparity=4)

    # rint(v1.shape, v2.shape, v3.shape)
    print(torch.allclose(v1, v2))
    print(torch.allclose(v1, v3))
    # print(torch.allclose(v2, v3))