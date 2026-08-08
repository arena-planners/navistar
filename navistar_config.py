"""Config for the NaviSTAR policy in `model/00500.pt`.

Values are upstream SAN-NaviSTAR `data/navigation/star/configs/config.py`, which
is the config that ships alongside this exact checkpoint
(`data/navigation/star/checkpoints/00500.pt`). The transformer sizes are
corroborated by the checkpoint's tensor shapes; `n_head`, `dropout` and
`activation` are not derivable from shapes and come from upstream. `ppo.num_steps`
only shapes training-time batches - `star_net.py` reads it solely on the
`infer=False` branch.
"""


class BaseConfig:
    pass


class Config:
    sim = BaseConfig()
    sim.human_num = 10

    trans = BaseConfig()
    trans.hidden_size = 128      # attention w_q/w_k/w_v.weight (128, 128)
    trans.forward_size = 1024    # ffn.linear1.weight (1024, 128)
    trans.n_layers = 3           # Temporal_Transformer.blocks.{0,1,2}
    trans.n_head = 8
    trans.dropout = 0.1
    trans.activation = "ReLU"

    ppo = BaseConfig()
    ppo.num_steps = 50
