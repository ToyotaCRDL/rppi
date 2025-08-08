# RPPI

Reservoir Predictive Path Integral (RPPI) control implementation in Julia.

## Overview

This project implements control algorithms using Echo State Networks for system identification combined with MPPI for optimal control. The implementation focuses on two main plant systems:

- **Four-tank system** (fourtank): A nonlinear hydraulic system
- **Spring-mass-damper system** (smd): A mechanical system with nonlinear dynamics

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

## Systems

### Four-tank System
- Nonlinear hydraulic system with coupled tanks
- Files: `*fourtank*`

### Spring-Mass-Damper System
- Mechanical system with nonlinear spring characteristics
- Files: `*smd*` with kernel parameter variations


## Usage

Refer to individual script files for specific usage.