# Earlier DaYa checkpoints

An earlier set of the three DaYa arms, trained between April and August 2026.

**No number in the paper comes from these files.** The paper reports the
2026-09-06 series in the parent directory, in which the control and both proposed
variants were trained back to back at the same five seeds. That is what makes the
control an architecture-matched one.

This set is kept so the two can be compared. It is not a matched series: the runs
are months apart, the control has a single seed, and the two sets do not give the
same numbers.

These checkpoints predate the rename of the chromatic encoder class. The
reproduction scripts register the old name as an alias, so they load.
