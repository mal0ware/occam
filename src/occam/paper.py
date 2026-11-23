"""Published values from DeMiguel, Garlappi and Uppal (2009), typed from Tables 3, 4 and 5.

Only the four datasets built from the French library and the five rows this project
implements are kept. ``None`` marks a cell the paper leaves empty. P-values are the figures
in parentheses in the paper.
"""

from __future__ import annotations

DATASETS = ("industry", "mkt_smb_hml", "ff1", "ff4")

#: The paper's sample: July 1963 to November 2004, estimation window of 120 months.
SAMPLE_START = 196307
SAMPLE_END = 200411
WINDOW = 120

Row = tuple[float | None, float | None, float | None, float | None]

#: Table 3: monthly Sharpe ratios.
SHARPE: dict[str, Row] = {
    "ew": (0.1353, 0.2240, 0.1623, 0.1753),
    "mv-is": (0.2124, 0.2851, 0.5098, 0.5364),
    "mv": (0.0679, 0.2186, 0.0128, 0.1841),
    "min": (0.1554, 0.2493, 0.2778, -0.0183),
    "min-c": (0.1425, 0.2493, 0.1546, 0.3580),
    "vw": (0.1138, 0.1138, 0.1138, 0.1138),
}
SHARPE_P: dict[str, Row] = {
    "mv": (0.17, 0.46, 0.02, 0.45),
    "min": (0.30, 0.23, 0.01, 0.01),
    "min-c": (0.41, 0.23, 0.35, 0.00),
    "vw": (0.01, 0.00, 0.01, 0.00),
}

#: Table 4: monthly certainty-equivalent returns, risk aversion 1.
CEQ: dict[str, Row] = {
    "ew": (0.0050, 0.0039, 0.0073, 0.0072),
    "mv-is": (0.0106, 0.0047, 0.0300, 0.0304),
    "mv": (-0.7816, 0.0045, -2.7142, -0.0829),
    "min": (0.0052, 0.0039, 0.0100, -0.0002),
    "min-c": (0.0047, 0.0039, 0.0060, 0.0051),
    "vw": (0.0042, 0.0042, 0.0042, 0.0042),
}
CEQ_P: dict[str, Row] = {
    "mv": (0.00, 0.31, 0.00, 0.01),
    "min": (0.45, 0.45, 0.12, 0.00),
    "min-c": (0.40, 0.45, 0.12, 0.17),
    "vw": (0.04, 0.44, 0.00, 0.00),
}

#: Table 5: monthly turnover of 1/N in absolute terms; other rows relative to 1/N (panel A).
TURNOVER: dict[str, Row] = {
    "ew": (0.0216, 0.0237, 0.0162, 0.0198),
    "mv": (606594.36, 2.83, 10466.10, 3553.03),
    "min": (21.65, 1.11, 45.47, 6.83),
    "min-c": (2.58, 1.11, 3.93, 1.76),
    "vw": (0.0, 0.0, 0.0, 0.0),
}

#: Table 5, panel B: return-loss against 1/N per month, with 50 basis points of cost.
RETURN_LOSS: dict[str, Row] = {
    "mv": (231.8504, 0.0003, 7.4030, 1.5740),
    "min": (0.0015, -0.0004, -0.0008, 0.0024),
    "min-c": (-0.0001, -0.0004, 0.0006, -0.0025),
    "vw": (0.0037, 0.0157, 0.0021, 0.0028),
}
