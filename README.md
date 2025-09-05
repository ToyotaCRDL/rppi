# RPPI

Reservoir Predictive Path Integral (RPPI) control implementation in Julia.

## Overview

This project implements control algorithms using Echo State Networks for system identification combined with MPPI for optimal control. The implementation focuses on two main plant systems:

- **Four-tank system** (fourtank): A nonlinear hydraulic system
- **Spring-mass-damper (Duffing) system** (smd): A mechanical system with nonlinear dynamics

## Project Structure

### Configuration Files

- `config/esn-*.yaml`: ESN network configuration files
- `config/mpc-*.yaml`: Model Predictive Control configuration
- `config/mppi-*.yaml`: MPPI configuration
- `config/plant-*.yaml`: Plant system definitions
- `config/simulation-*.yaml`: Simulation parameters

### Scripts

- `scripts/esn.jl`: ESN implementation
- `scripts/mppi-esn-fast.jl`: MPPI-ESN controller
- `scripts/plant-*.jl`: Plant system implementations
- `scripts/*_lesn_mpc.jl`: MPC simulation script with learned ESN models
- `scripts/*_lesn_mppi.jl`: MPPI simulation script with learned ESN models
- `scripts/utils.jl`: Utility functions

## Target Systems

### Four-tank System
- Nonlinear hydraulic system with coupled tanks
- Files: `*fourtank*`

### Spring-Mass-Damper System
- Mechanical system with nonlinear spring characteristics
- Files: `*smd*`



## Citation

If you use this code in your research, please cite our paper:

**arXiv preprint**: https://arxiv.org/abs/2509.03839

```bibtex
@misc{Inoue2025Reservoir,
  title = {Reservoir Predictive Path Integral Control for Unknown Nonlinear Dynamics},
  author = {Inoue, Daisuke and Matsumori, Tadayoshi and Tanaka, Gouhei and Ito, Yuji},
  year = {2025},
  month = sep,
  number = {arXiv:2509.03839},
  eprint = {2509.03839},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2509.03839},
}
```


## License

This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.html).