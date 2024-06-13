import torch
import torch.nn.functional as F

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 3
        self.window = self.create_window(window_size, sigma)

    def gaussian_window(self, window_size, sigma):
        """
        Generates a 1-D Gaussian window.
        """
        gauss = torch.arange(window_size//2, -window_size//2, -1).to(torch.float32).to("cuda")
        gauss = torch.exp(-gauss ** 2 / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        return gauss.unsqueeze(1) * gauss.unsqueeze(0)

    def create_window(self, window_size, sigma):
        """
        Generates a 2-D Gaussian window.
        """
        _1D_window = self.gaussian_window(window_size, sigma).to(torch.float32).to("cuda")
        _2D_window = _1D_window.mm(_1D_window.t())
        window = _2D_window.unsqueeze(0).unsqueeze(0)
        return window

    def forward(self, img1, img2):
        """
        Calculate the SSIM loss between two images.
        """
        (_, channel, _, _) = img1.size()
        window = self.window.expand(channel, 1, self.window_size, self.window_size).contiguous()

        ssim_loss = 0
        for c in range(channel):
            mu1 = F.conv2d(img1[:, c:c+1], window, padding=self.window_size//2)
            mu2 = F.conv2d(img2[:, c:c+1], window, padding=self.window_size//2)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(img1[:, c:c+1] * img1[:, c:c+1], window, padding=self.window_size//2) - mu1_sq
            sigma2_sq = F.conv2d(img2[:, c:c+1] * img2[:, c:c+1], window, padding=self.window_size//2) - mu2_sq
            sigma12 = F.conv2d(img1[:, c:c+1] * img2[:, c:c+1], window, padding=self.window_size//2) - mu1_mu2

            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            ssim_loss += (1 - ssim_map.mean())

        return ssim_loss / channel

# Example usage
if __name__ == "__main__":
    img1 = torch.rand(1, 3, 256, 256)  # RGB image
    img2 = torch.rand(1, 3, 256, 256)  # RGB image
    ssim_loss = SSIMLoss()
    loss = ssim_loss(img1, img2)
    print(f"SSIM Loss: {loss.item()}")

