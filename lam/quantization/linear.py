'''Not really a quantization method, simply reduces the action space to a lower dimensional space with a linear transformation.'''
from torch import nn

class LinearDimensionReduction(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 use_sigreg,
                 sigreg_num_points,
                 sigreg_num_slices,
                 ):
        super().__init__()
        self.in_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.out_projection = nn.Linear(hidden_dim, input_dim, bias=False)

        if use_sigreg:
            import lejepa
            univariate_test = lejepa.univariate.EppsPulley(n_points=sigreg_num_points)
            self.sig_reg = lejepa.multivariate.SlicingUnivariateTest(univariate_test=univariate_test, num_slices=sigreg_num_slices)


    def forward(self, x):
        bottleneck = self.in_projection(x)
        output = self.out_projection(bottleneck)

        if hasattr(self, "sig_reg"):
            return output, bottleneck, self.sig_reg(bottleneck)

        return output, bottleneck, 0.0