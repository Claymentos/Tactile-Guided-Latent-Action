'''Not really a quantization method, simply reduces the action space to a lower dimensional space with a linear transformation.'''
from torch import nn

class ContinuousDimensionReduction(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 sig_reg,
                 sig_reg_num_points,
                 sig_reg_num_slices,
                 ):
        super().__init__()
        self.in_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.out_projection = nn.Linear(hidden_dim, input_dim, bias=False)

        if sig_reg:
            import lejepa
            univariate_test = lejepa.univariate.EppsPulley(n_points=sig_reg_num_points)
            self.sig_reg = lejepa.multivariate.SlicingUnivariateTest(univariate_test=univariate_test, num_slices=sig_reg_num_slices)


    def forward(self, x):
        bottleneck = self.in_projection(x)
        output = self.out_projection(bottleneck)

        regularization_loss = self.sig_reg(bottleneck) if hasattr(self, "sig_reg") else 0.0

        return output, bottleneck, regularization_loss