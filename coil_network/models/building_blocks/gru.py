import torch.nn as nn
import torch
class GRUWaypointsPredictorTransFuser(nn.Module):
  """
  The waypoint GRU used in TransFuser.
  It enters the target point as input.
  The hidden state is initialized with the scene features.
  The input is autoregressive and starts either at 0 or learned.
  """

  def __init__(self, config, target_point_size):
    super().__init__()
    self.wp_decoder = nn.GRUCell(input_size=2 + target_point_size, hidden_size=config.gru_hidden_size)
    self.output = nn.Linear(config.gru_hidden_size, 2)
    self.config = config
    self.prediction_len = config.pred_len

  def forward(self, z, target_point):
    output_wp = []

    # initial input variable to GRU
    if self.config.learn_origin:
      x = z[:, self.config.gru_hidden_size:(self.config.gru_hidden_size + 2)]  # Origin of the waypoints
      z = z[:, :self.config.gru_hidden_size]
    else:
      x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)

    target_point = target_point.clone()

    # autoregressive generation of output waypoints
    for _ in range(self.prediction_len):
      if self.config.use_tp:
        x_in = torch.cat([x, target_point], dim=1)
      else:
        x_in = x

      z = self.wp_decoder(x_in, z)
      dx = self.output(z)

      x = dx + x

      output_wp.append(x)

    pred_wp = torch.stack(output_wp, dim=1)

    return pred_wp