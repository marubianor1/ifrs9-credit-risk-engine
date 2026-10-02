# LGD Model Decision For ECL

`lgd_v1_2` is the production baseline LGD for this portfolio project.

Rejected challengers:

- `lgd_v1_3`;
- `lgd_fl_v1`;
- `lgd_fl_v2`.

The challengers are retained as research and governance evidence, but they are
not used in the baseline ECL engine. They were rejected because they did not
improve temporal/OOT stability and the direct macro relationship was weak. The
collateral-driven cure-LGD challenger created a transparent HPI-to-LTV mechanism,
but OOT cure-LGD and combined ELGD deteriorated versus `lgd_v1_2`.

The default ECL engine therefore uses macro-neutral structural LGD from
`lgd_v1_2`. The existing validated downturn LGD factors from `lgd_v1_2` remain
available as a sensitivity mode. They are not presented as fitted
forward-looking scenario LGD.
