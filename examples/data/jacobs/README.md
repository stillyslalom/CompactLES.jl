# Digitized measurements of the Jacobs air/SF6 experiments

Points read from the published figures of two papers, for
`examples/jacobs_air_sf6.jl`:

- B. D. Collins and J. W. Jacobs, PLIF flow visualization and measurements of the
  Richtmyer–Meshkov instability of an air/SF6 interface, J. Fluid Mech. 464,
  113–136 (2002), doi:10.1017/S0022112002008844.
- J. W. Jacobs and V. V. Krivets, Experiments on the late-time development of
  single-mode Richtmyer–Meshkov instability, Phys. Fluids 17, 034105 (2005),
  doi:10.1063/1.1852574.

| file | figure | content |
|:--- | :--- | :--- |
| `cj2002_fig03_xt.csv` | CJ Fig. 3 | wave diagram, Ms = 1.11: shock and characteristic lines, interface |
| `cj2002_fig04_xt.csv` | CJ Fig. 4 | wave diagram, Ms = 1.21 |
| `cj2002_fig11.csv` | CJ Fig. 11 | interface displacement against time, both Mach numbers |
| `cj2002_fig12.csv` | CJ Fig. 12 | amplitude against time, every Ms = 1.11 firing |
| `cj2002_fig13.csv` | CJ Fig. 13 | early amplitude, Ms = 1.11, means of five firings with 95% intervals |
| `cj2002_fig14.csv` | CJ Fig. 14 | ``k(a - a_0^+)`` against ``\dot a_0 k t``, both Mach numbers |
| `jk2005_fig07.csv` | JK Fig. 7 | the same variables, λ = 59 and 36 mm |

Each file's header gives the citation, the axes and units, the symbol of each
series, the ticks used for the calibration and its residual, the method, the
one-standard-deviation digitizing uncertainty of each axis and the meaning of
the flags. Every figure is a raster image; the axes were calibrated from the
tick marks on all four sides of the frame, and the markers located by template
correlation, with overlapping markers separated by fitting their outlines.

Checks against numbers the papers state:

- Fig. 11: straight-line fits give 32.95 and 60.73 m/s; the paper gives 33.0
  and 60.6.
- Fig. 13: the fitted slope is 3.917 m/s; the paper gives 3.92 ± 0.23.
- Fig. 12 against Fig. 13: at each of the seven early times the mean of the
  visible markers is within 0.045 mm of the five-firing mean.
- Fig. 14 against its re-plot in JK Fig. 4: 74 of 75 markers match, within
  0.006 in either variable.
- Fig. 12 against its re-plot by T. Kaman and R. Holley, arXiv:2207.11404
  (2022), Fig. 6: 46 nearest pairs differ by 0.023 mm rms. The flag
  `confirmed_KH2022fig6` marks a point found there.
- Fig. 14, converted back with the post-shock amplitude and the measured growth
  rate of CJ Table 1, reproduces Fig. 12 to 0.06 mm. The pre-shock amplitude
  would shift every point by 0.25 mm.

Limits: overlapping markers that are entirely hidden cannot be recovered, so
Fig. 12 has 53 of about 60 firings and Fig. 11 fewer than the 50 per series the
paper reports. In JK Fig. 7 below ``k\dot a_0 t \approx 4`` the markers lie on a
bundle of model curves, and an estimated 15 to 25 of them were not recovered.
