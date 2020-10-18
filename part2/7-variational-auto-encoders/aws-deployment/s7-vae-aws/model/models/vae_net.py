import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None, groups=1):
        super(ResidualBlock, self).__init__()
        p = kernel_size//2
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=p),
            #nn.BatchNorm2d(planes),
            #nn.LeakyReLU(0.2)
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size, padding=p),
            nn.ReLU()
        )
        self.proj = nn.Conv2d(inplanes, planes, 1) if inplanes != planes else None
    
    def forward(self, x):
        identity = x
        
        y = self.conv1(x)
        y = self.conv2(y)
        
        identity = identity if self.proj is None else self.proj(identity)
        y = y + identity
        return y

class Encoder(nn.Module):
    """
        Convolutional Encoder
    """
    def __init__(self, channels=3, latent_dims=10):
        super(Encoder, self).__init__()

        self.latent_dims = latent_dims

        self.E = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), # (N, 32, 96, 96)
            ResidualBlock(32, 64),                 
            nn.MaxPool2d(2,2),                     # (N, 64, 48, 48)
            ResidualBlock(64, 128),
            nn.MaxPool2d(2,2),                     # (N, 128, 24, 24)
            ResidualBlock(128, 256),
            nn.MaxPool2d(2,2),                     # (N, 256, 12, 12)
            ResidualBlock(256, 512),
            nn.MaxPool2d(2,2),                     # (N, 512, 6, 6)
            ResidualBlock(512, 1024),
            nn.MaxPool2d(2,2)                      # (N, 1024, 3, 3)
        )

        # this last layer bottlenecks through latent_dims connections

        # Output Block - mu
        self.mu_out = nn.Sequential(
            nn.AvgPool2d(kernel_size=3),    
            nn.Conv2d(in_channels=1024, out_channels=latent_dims, kernel_size=(1,1), padding=0, bias=False)
        ) 

        # Output Block - logvar
        self.logvar_out = nn.Sequential(
            nn.AvgPool2d(kernel_size=3),    
            nn.Conv2d(in_channels=1024, out_channels=latent_dims, kernel_size=(1,1), padding=0, bias=False)
        ) 
        
    def forward(self, x):
        B = x.size(0)
        h = self.E(x)
 
        mu = self.mu_out(h)
        mu = mu.view(B,-1)

        logvar = self.logvar_out(h)
        logvar = logvar.view(B,-1)

        return mu, logvar

class Decoder(nn.Module):
    """
        Convolutional Decoder
    """
    def __init__(self, channels=3, latent_dims=10):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_dims, 1024*3*3)
        
        self.D = nn.Sequential(
            ResidualBlock(1024, 512),
            nn.Upsample(scale_factor=2, mode='bilinear'), # (N, 512, 6, 6)
            ResidualBlock(512, 256),
            nn.Upsample(scale_factor=2, mode='bilinear'), # (N, 256, 12, 12)
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear'), # (N, 128, 24, 24)
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear'), # (N, 64, 48, 48)
            ResidualBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear'), # (N, 32, 96, 96)
            ResidualBlock(32, 32),
            nn.Conv2d(32, channels, 3, padding=1)         # (N, 3, 96, 96)
        )
        
        #self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, z):
        B = z.size(0)
        h = self.fc(z)
        h = h.view(B,1024,3,3)
        x = self.D(h)
        return self.tanh(x)

class VAE(nn.Module):
    def __init__(self, channels=3, latent_dims=10):
        super(VAE, self).__init__()

        self.encoder = Encoder(channels, latent_dims)
        self.decoder = Decoder(channels, latent_dims)

    def reparameterize(self, mu, logvar):
        """THE REPARAMETERIZATION IDEA:

        For each training sample (we get N batched at a time)

        - take the current learned mu, stddev for each of the latent_dims
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        Parameters
        ----------
        mu : [N, latent_dims] mean matrix
        logvar : [N, latent_dims] variance matrix

        Returns
        -------

        During training random sample from the learned latent_dims dimensional
        normal distribution; during inference its mean.

        """

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation
            std = logvar.mul(0.5).exp_()  # type: Variable
            # - std.data is the [N,latent_dims] tensor that is wrapped by std
            # - so eps is [N,latent_dims] with all elements drawn from a mean 0
            #   and stddev 1 normal distribution that is 128 samples
            #   of random latent_dims-float vectors
            eps = std.data.new(std.size()).normal_()
            # - sample from a normal distribution with standard
            #   deviation = std and mean = mu by multiplying mean 0
            #   stddev 1 sample with desired std and mu, see
            #   https://stats.stackexchange.com/a/16338
            # - so we have N sets (the batch) of random latent_dims-float
            #   vectors sampled from normal distribution with learned
            #   std and mu for the current input
            return eps.mul(std).add_(mu)

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
